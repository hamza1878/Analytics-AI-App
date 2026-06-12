"""
models/demand_forecast.py

Ensemble LSTM + Prophet pour la prédiction de la demande.
Métriques cibles : MAPE < 6%, R² > 0.95
"""
import os
import warnings

# Supprime le warning protobuf/MessageFactory avant tout import lourd
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
warnings.filterwarnings("ignore", message=".*MessageFactory.*")

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from config import ML

# TensorFlow/Keras importés à la demande (optionnel si non dispo)
try:
    import tensorflow as tf
    import keras
    from keras import layers
    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# ─────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────

def build_lstm(seq_len: int, n_features: int) -> "keras.Model":
    """Architecture LSTM bi-directionnel pour la demande."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow non disponible. pip install tensorflow")

    inp = keras.Input(shape=(seq_len, n_features))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


def train_lstm(X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 64) -> dict:
    """Entraîne le LSTM et retourne le modèle + métriques."""

    # ── FIX : guard explicite avant train_test_split ─────────────────────────
    if X.ndim != 3 or X.shape[0] == 0:
        raise ValueError(
            f"X invalide : shape={X.shape}. "
            "Les séquences sont vides — vérifiez demand_features.py. "
            "Cause probable : toutes les zones ont moins de seq_len buckets."
        )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)

    model = build_lstm(X.shape[1], X.shape[2])

    callbacks = []
    if TF_AVAILABLE:
        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
        ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = model.predict(X_val, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, None)

    # ── Guard NaN : loss=nan depuis l'epoch 1 → poids divergés → prédictions NaN
    # Cause racine : NaN dans X_train (normalisation défectueuse dans demand_features.py)
    nan_ratio = np.isnan(y_pred).mean()
    if nan_ratio > 0.5:
        nan_in_X = np.isnan(X).sum()
        raise ValueError(
            f"Le modèle produit {nan_ratio:.0%} de prédictions NaN — "
            f"les poids ont divergé dès l'epoch 1. "
            f"NaN détectés dans X : {nan_in_X}. "
            "Vérifiez la normalisation dans build_lstm_sequences() : "
            "colonnes constantes ou valeurs NULL dans la DB (distance_km, price_final…)."
        )
    # Remplace les NaN résiduels isolés par la médiane
    if np.isnan(y_pred).any():
        y_pred = np.where(np.isnan(y_pred), np.nanmedian(y_pred), y_pred)

    mape = mean_absolute_percentage_error(y_val + 1e-8, y_pred + 1e-8)
    r2   = r2_score(y_val, y_pred)

    return {
        "model":           model,
        "history":         history.history,
        "mape":            mape,
        "r2":              r2,
        "val_predictions": y_pred,
        "val_actuals":     y_val,
    }


# ─────────────────────────────────────────────
# Prophet
# ─────────────────────────────────────────────

def train_prophet(zone_df: pd.DataFrame, target_col: str = "ride_count") -> dict:
    """Entraîne Prophet sur la série agrégée toutes zones (plus de points = meilleure tendance)."""
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet non disponible. pip install prophet")

    # Agrège toutes les zones par heure pour une série globale robuste
    ts = (
        zone_df.groupby("hour_bucket")[target_col]
        .mean()
        .reset_index()
        .rename(columns={"hour_bucket": "ds", target_col: "y"})
        .sort_values("ds")
    )

    # Prophet n'accepte pas les timestamps avec timezone
    if hasattr(ts["ds"].dtype, "tz") and ts["ds"].dt.tz is not None:
        ts["ds"] = ts["ds"].dt.tz_localize(None)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    model.fit(ts)

    future   = model.make_future_dataframe(periods=24, freq="h")
    forecast = model.predict(future)

    return {"model": model, "forecast": forecast}


# ─────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────

def ensemble_predict(
    lstm_pred: np.ndarray,
    prophet_pred: np.ndarray,
    lstm_weight: float = 0.65,
) -> np.ndarray:
    """Combine LSTM et Prophet par pondération linéaire."""
    prophet_weight = 1.0 - lstm_weight
    return lstm_weight * lstm_pred + prophet_weight * prophet_pred


# ─────────────────────────────────────────────
# MLflow training run
# ─────────────────────────────────────────────

def train_and_log(X: np.ndarray, y: np.ndarray, zone_df: pd.DataFrame) -> str:
    """
    Entraîne l'ensemble, loggue dans MLflow et retourne le run_id.
    """
    # ── FIX : guard précoce avec message clair ───────────────────────────────
    if X.ndim != 3 or X.shape[0] == 0:
        raise ValueError(
            f"X invalide : shape={X.shape}. "
            "Les séquences LSTM sont vides — vérifiez demand_features.py. "
            "Cause probable : seq_len > nombre de buckets par zone."
        )
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        raise ValueError(
            f"X contient {nan_count} valeurs NaN. "
            "Cause probable : NULL dans distance_km ou price_final en DB → "
            "avg_distance_km/avg_price NaN → normalisation NaN. "
            "Vérifiez build_lstm_sequences() dans demand_features.py."
        )

    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    mlflow.set_experiment(ML.mlflow_experiment)

    with mlflow.start_run(run_name="demand_forecast_v4") as run:

        # ── LSTM ──────────────────────────────────────────────────────────────
        lstm_result = train_lstm(X, y)
        n_val       = len(lstm_result["val_actuals"])

        # ── Prophet ───────────────────────────────────────────────────────────
        lstm_weight  = 0.65
        prophet_vals = np.zeros(n_val, dtype=np.float32)   # fallback si Prophet KO

        if PROPHET_AVAILABLE:
            try:
                prophet_result = train_prophet(zone_df)

                # FIX : aligne sur les n_val derniers points historiques (pas le futur)
                historical_yhat = prophet_result["forecast"]["yhat"].values[:-24]  # exclut les 24h futures
                if len(historical_yhat) >= n_val:
                    prophet_vals = historical_yhat[-n_val:].astype(np.float32)
                else:
                    # padding par la moyenne si Prophet a moins de points que n_val
                    pad = np.full(n_val - len(historical_yhat), historical_yhat.mean(), dtype=np.float32)
                    prophet_vals = np.concatenate([pad, historical_yhat.astype(np.float32)])

                prophet_vals = np.clip(prophet_vals, 0, None)

                # Normalise dans [0,1] pour correspondre à l'échelle LSTM
                p_max = prophet_vals.max()
                if p_max > 1e-8:
                    prophet_vals = prophet_vals / p_max

            except Exception as exc:
                print(f"  ⚠  Prophet a échoué ({exc}) — fallback 100% LSTM.")
                lstm_weight = 1.0

        # ── Ensemble ──────────────────────────────────────────────────────────
        ensemble_pred = ensemble_predict(lstm_result["val_predictions"], prophet_vals, lstm_weight)
        ensemble_mape = mean_absolute_percentage_error(
            lstm_result["val_actuals"] + 1e-8,
            ensemble_pred              + 1e-8,
        )
        ensemble_r2 = r2_score(lstm_result["val_actuals"], ensemble_pred)

        # ── MLflow logging ────────────────────────────────────────────────────
        mlflow.log_params({
            "lstm_layers":               "BiLSTM(128) → LSTM(64) → Dense(32)",
            "lstm_weight":               lstm_weight,
            "prophet_changepoint_prior": 0.05,
            "seq_len":                   X.shape[1],
            "n_features":                X.shape[2],
        })
        mlflow.log_metrics({
            "lstm_mape":     lstm_result["mape"],
            "lstm_r2":       lstm_result["r2"],
            "ensemble_mape": ensemble_mape,
            "ensemble_r2":   ensemble_r2,
        })

        # FIX : format .keras (le .h5 est déprécié depuis Keras 3)
        if TF_AVAILABLE:
            save_path = "/tmp/demand_lstm.keras"
            lstm_result["model"].save(save_path)
            mlflow.log_artifact(save_path)

        print(f"[demand_forecast] MAPE={ensemble_mape:.2%} | R²={ensemble_r2:.4f}")
        return run.info.run_id


if __name__ == "__main__":
    from feature_engineering.demand_features import run as fe_run
    data = fe_run()
    train_and_log(data["X"], data["y"], data["zone_hour_df"])