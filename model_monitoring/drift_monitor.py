"""
model_monitoring/drift_monitor.py

Calcule le Population Stability Index (PSI) pour surveiller
la dérive des features entre le jeu d'entraînement et la prod.

Seuils PSI :
  < 0.05  → stable (vert)
  < 0.10  → attention (orange)
  >= 0.10 → drift significatif → déclencher retraining (rouge)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from config import ML


@dataclass
class DriftReport:
    feature_name: str
    psi_score: float
    status: str        # "stable" | "warning" | "critical"
    n_bins: int
    computed_at: datetime

    def to_dict(self) -> dict:
        return {
            "feature":      self.feature_name,
            "psi":          round(self.psi_score, 4),
            "status":       self.status,
            "n_bins":       self.n_bins,
            "computed_at":  self.computed_at.isoformat(),
        }


def psi_score(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """
    Calcule le PSI entre une distribution de référence (expected)
    et la distribution courante (actual).

    PSI = Σ (A_i - E_i) × ln(A_i / E_i)
    """
    # Bins sur expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    def bin_frequencies(arr: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=breakpoints)
        freq = counts / (counts.sum() + epsilon)
        return np.clip(freq, epsilon, None)

    e_freq = bin_frequencies(expected)
    a_freq = bin_frequencies(actual)

    psi = float(np.sum((a_freq - e_freq) * np.log(a_freq / e_freq)))
    return psi


def status_from_psi(score: float) -> str:
    if score < ML.psi_warning:
        return "stable"
    elif score < ML.psi_critical:
        return "warning"
    return "critical"


class DriftMonitor:
    """
    Monitore la dérive des features ML en prod.

    Usage :
        monitor = DriftMonitor()
        monitor.fit_reference(train_df, feature_cols)
        report = monitor.check(prod_df)
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_cols: Optional[list[str]] = None

    def fit_reference(self, df: pd.DataFrame, feature_cols: list[str]) -> "DriftMonitor":
        """Enregistre la distribution de référence (jeu d'entraînement)."""
        self.reference_data = df[feature_cols].copy()
        self.reference_cols = feature_cols
        print(f"[drift_monitor] Référence établie sur {len(df)} samples, "
              f"{len(feature_cols)} features")
        return self

    def check(self, current_df: pd.DataFrame) -> list[DriftReport]:
        """
        Compare la distribution courante à la référence.
        Retourne un DriftReport par feature.
        """
        if self.reference_data is None:
            raise RuntimeError("Appeler fit_reference() d'abord.")

        reports = []
        now = datetime.now(timezone.utc)

        for col in self.reference_cols:
            if col not in current_df.columns:
                continue

            ref = self.reference_data[col].dropna().values
            cur = current_df[col].dropna().values

            if len(cur) < 30:
                continue  # pas assez de données prod

            score = psi_score(ref, cur, n_bins=self.n_bins)
            status = status_from_psi(score)

            reports.append(DriftReport(
                feature_name=col,
                psi_score=score,
                status=status,
                n_bins=self.n_bins,
                computed_at=now,
            ))

        # Trier par PSI décroissant
        reports.sort(key=lambda r: r.psi_score, reverse=True)

        n_critical = sum(1 for r in reports if r.status == "critical")
        n_warning  = sum(1 for r in reports if r.status == "warning")
        print(f"[drift_monitor] {n_critical} critiques | {n_warning} warnings "
              f"sur {len(reports)} features")

        return reports

    def check_and_alert(
        self, current_df: pd.DataFrame, model_name: str
    ) -> tuple[list[DriftReport], bool]:
        """
        Vérifie la dérive et retourne (reports, should_retrain).
        should_retrain = True si au moins 30% des features sont critiques.
        """
        reports = self.check(current_df)
        critical_ratio = sum(1 for r in reports if r.status == "critical") / (len(reports) + 1e-8)
        should_retrain = critical_ratio >= 0.30

        if should_retrain:
            print(f"[drift_monitor] ⚠ RETRAINING RECOMMANDÉ pour {model_name} "
                  f"(critical ratio={critical_ratio:.0%})")
        return reports, should_retrain


# ─────────────────────────────────────────────
# Rapport global par modèle
# ─────────────────────────────────────────────

MODEL_FEATURE_GROUPS = {
    "demand_forecast": [
        "ride_count", "avg_distance_km", "avg_price",
        "demand_trend_1d", "demand_trend_7d",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ],
    "surge_predictor": [
        "zone_lat", "zone_lon", "hour_of_day", "concurrent_rides_in_hour",
        "rolling_surge_1h", "rolling_surge_3h", "price_ratio",
    ],
    "churn_classifier": [
        "rating_average", "total_trips", "days_since_last_trip",
        "accept_rate", "reject_rate", "avg_dispatch_score",
        "recent_avg_rating", "is_online",
    ],
    "eta_estimator": [
        "distance_km", "avg_speed_kmh", "detour_ratio",
        "waypoint_count", "effective_speed_kmh",
    ],
    "route_optimizer": [
        "distance_to_pickup_km", "trip_distance_km",
        "surge_multiplier", "driver_rating", "dispatch_score",
    ],
}


def compute_model_psi(
    reference_dfs: dict[str, pd.DataFrame],
    current_dfs: dict[str, pd.DataFrame],
) -> dict[str, list[dict]]:
    """
    Calcule le PSI pour chaque groupe de features de modèle.

    Args:
        reference_dfs : {model_name: train_df}
        current_dfs   : {model_name: prod_df}

    Returns:
        {model_name: [DriftReport.to_dict(), ...]}
    """
    results = {}
    for model_name, feature_cols in MODEL_FEATURE_GROUPS.items():
        if model_name not in reference_dfs or model_name not in current_dfs:
            continue

        monitor = DriftMonitor()
        available_cols = [c for c in feature_cols if c in reference_dfs[model_name].columns]
        monitor.fit_reference(reference_dfs[model_name], available_cols)
        reports = monitor.check(current_dfs[model_name])
        results[model_name] = [r.to_dict() for r in reports]

    return results


if __name__ == "__main__":
    # Démo avec données synthétiques
    rng = np.random.default_rng(42)
    ref_df  = pd.DataFrame(rng.normal(0, 1, (1000, 3)), columns=["a", "b", "c"])
    prod_df = pd.DataFrame(rng.normal(0.3, 1.2, (500, 3)), columns=["a", "b", "c"])

    monitor = DriftMonitor()
    monitor.fit_reference(ref_df, ["a", "b", "c"])
    reports, retrain = monitor.check_and_alert(prod_df, "demo_model")
    for r in reports:
        print(f"  {r.feature_name}: PSI={r.psi_score:.4f} [{r.status}]")
    print(f"  → Retraining: {retrain}")
