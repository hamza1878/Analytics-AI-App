"""
serving/inference.py

Client d'inférence Triton pour tous les modèles Moviroo.
Latence cible p99 < 10ms.
"""
import numpy as np
import time
from typing import Optional
from config import ML

try:
    import tritonclient.http as tritonhttp
    import tritonclient.grpc as tritongrpc
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class TritonInferenceClient:
    """
    Client unifié pour appeler les modèles via Triton Inference Server.
    Supporte HTTP et gRPC (préférer gRPC pour la latence).
    """

    def __init__(
        self,
        host: str = ML.triton_host,
        http_port: int = ML.triton_port,
        grpc_port: int = 8001,
        use_grpc: bool = True,
    ):
        self.host      = host
        self.use_grpc  = use_grpc
        self._client   = None
        self._latencies: list[float] = []

        if not TRITON_AVAILABLE:
            print("[triton] Client non disponible — mode simulation activé")
            return

        if use_grpc:
            self._client = tritongrpc.InferenceServerClient(
                url=f"{host}:{grpc_port}", verbose=False
            )
        else:
            self._client = tritonhttp.InferenceServerClient(
                url=f"{host}:{http_port}", verbose=False
            )

    def is_live(self) -> bool:
        if not self._client:
            return False
        try:
            return self._client.is_server_live()
        except Exception:
            return False

    def _infer_grpc(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        output_names: list[str],
    ) -> dict[str, np.ndarray]:
        infer_inputs = []
        for name, data in inputs.items():
            inp = tritongrpc.InferInput(name, data.shape, "FP32")
            inp.set_data_from_numpy(data.astype(np.float32))
            infer_inputs.append(inp)

        infer_outputs = [tritongrpc.InferRequestedOutput(n) for n in output_names]

        t0 = time.perf_counter()
        response = self._client.infer(
            model_name=model_name,
            inputs=infer_inputs,
            outputs=infer_outputs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(latency_ms)

        return {n: response.as_numpy(n) for n in output_names}

    def _simulate(self, model_name: str, inputs: dict) -> dict:
        """Mode simulation quand Triton n'est pas disponible."""
        self._latencies.append(np.random.uniform(5, 12))
        simulated = {
            "demand_lstm":    {"output_0": np.array([[0.42]])},
            "surge_xgboost":  {"output_0": np.array([[1.35]])},
            "churn_rf":       {"output_0": np.array([[0.73]])},
            "eta_lgbm":       {"output_0": np.array([[12.4]])},
            "fraud_iforest":  {"output_0": np.array([[-0.12]])},
        }
        return simulated.get(model_name, {"output_0": np.array([[0.5]])})

    def predict(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        output_names: list[str] = None,
    ) -> dict[str, np.ndarray]:
        output_names = output_names or ["output_0"]
        if not TRITON_AVAILABLE or not self._client:
            return self._simulate(model_name, inputs)
        return self._infer_grpc(model_name, inputs, output_names)

    @property
    def p99_latency_ms(self) -> Optional[float]:
        if not self._latencies:
            return None
        return float(np.percentile(self._latencies, 99))

    @property
    def mean_latency_ms(self) -> Optional[float]:
        if not self._latencies:
            return None
        return float(np.mean(self._latencies))

    def latency_report(self) -> dict:
        if not self._latencies:
            return {}
        arr = np.array(self._latencies)
        return {
            "n_calls":   len(arr),
            "mean_ms":   round(float(arr.mean()), 2),
            "p50_ms":    round(float(np.percentile(arr, 50)), 2),
            "p95_ms":    round(float(np.percentile(arr, 95)), 2),
            "p99_ms":    round(float(np.percentile(arr, 99)), 2),
            "max_ms":    round(float(arr.max()), 2),
        }


# ─────────────────────────────────────────────
# Helpers haut niveau
# ─────────────────────────────────────────────

_client = TritonInferenceClient()


def predict_demand(X_sequence: np.ndarray) -> float:
    """Prédit la demande pour une séquence temporelle (seq_len, n_features)."""
    inp = X_sequence[np.newaxis, :, :].astype(np.float32)   # (1, seq_len, n_features)
    result = _client.predict("demand_lstm", {"input_0": inp})
    return float(result["output_0"][0, 0])


def predict_surge(feature_vector: np.ndarray) -> float:
    """Prédit le surge multiplier. Retourné dans [1.0, 3.5]."""
    inp = feature_vector[np.newaxis, :].astype(np.float32)
    result = _client.predict("surge_xgboost", {"input_0": inp})
    return float(np.clip(result["output_0"][0, 0], 1.0, 3.5))


def predict_churn_proba(feature_vector: np.ndarray) -> float:
    """Retourne la probabilité de churn [0, 1]."""
    inp = feature_vector[np.newaxis, :].astype(np.float32)
    result = _client.predict("churn_rf", {"input_0": inp})
    return float(np.clip(result["output_0"][0, 0], 0.0, 1.0))


def predict_eta(feature_vector: np.ndarray) -> float:
    """Prédit l'ETA en minutes."""
    inp = feature_vector[np.newaxis, :].astype(np.float32)
    result = _client.predict("eta_lgbm", {"input_0": inp})
    return float(max(1.0, result["output_0"][0, 0]))


def predict_fraud_score(feature_vector: np.ndarray) -> float:
    """Retourne le score d'anomalie. Plus élevé = plus suspect."""
    inp = feature_vector[np.newaxis, :].astype(np.float32)
    result = _client.predict("fraud_iforest", {"input_0": inp})
    raw = float(result["output_0"][0, 0])
    # Normaliser vers [0, 1] : score Triton est négatif pour IsolationForest
    return float(np.clip(-raw / 0.5, 0.0, 1.0))


def get_server_status() -> dict:
    return {
        "live":         _client.is_live(),
        "latency":      _client.latency_report(),
        "triton_host":  ML.triton_host,
        "triton_port":  ML.triton_port,
    }


if __name__ == "__main__":
    import json

    print("[inference] Test des prédictions Triton (mode simulation si non dispo)…\n")

    # Demand
    seq = np.random.randn(24, 9).astype(np.float32)
    demand = predict_demand(seq)
    print(f"  demand forecast : {demand:.3f}")

    # Surge
    surge_feat = np.random.randn(14).astype(np.float32)
    surge = predict_surge(surge_feat)
    print(f"  surge multiplier: {surge:.2f}×")

    # Churn
    churn_feat = np.random.randn(18).astype(np.float32)
    churn = predict_churn_proba(churn_feat)
    print(f"  churn proba     : {churn:.1%}")

    # ETA
    eta_feat = np.random.randn(15).astype(np.float32)
    eta = predict_eta(eta_feat)
    print(f"  ETA             : {eta:.1f} min")

    # Fraud
    fraud_feat = np.random.randn(14).astype(np.float32)
    fraud = predict_fraud_score(fraud_feat)
    print(f"  fraud score     : {fraud:.3f}")

    print()
    print(json.dumps(_client.latency_report(), indent=2))
