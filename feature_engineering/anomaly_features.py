"""
feature_engineering/anomaly_features.py

Features d'entrée pour le détecteur d'anomalies hybride
(IsolationForest + LSTM residuals).
Combine toutes les sources de signal du schéma Moviroo.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from config import DB


ANOMALY_QUERY = """
WITH passenger_hourly AS (
    SELECT
        passenger_id,
        DATE_TRUNC('hour', created_at)          AS hour_bucket,
        COUNT(*)                                AS ride_count_1h,
        SUM(price_final)                        AS spend_1h,
        AVG(price_final)                        AS avg_price_1h,
        SUM(CASE WHEN status = 'CANCELLED'
                  AND driver_id IS NOT NULL
                  AND trip_started_at IS NULL
             THEN 1 ELSE 0 END)                AS suspicious_cancel_1h
    FROM rides
    WHERE created_at >= NOW() - INTERVAL '7 days'
    GROUP BY passenger_id, DATE_TRUNC('hour', created_at)
),
driver_hourly AS (
    SELECT
        driver_id,
        DATE_TRUNC('hour', offered_at)          AS hour_bucket,
        COUNT(*)                                AS offer_count_1h,
        AVG(score)                              AS avg_score_1h,
        SUM(CASE WHEN status = 'EXPIRED'
             THEN 1 ELSE 0 END)                AS expired_1h
    FROM dispatch_offers
    WHERE offered_at >= NOW() - INTERVAL '7 days'
    GROUP BY driver_id, DATE_TRUNC('hour', offered_at)
),
ride_price_signals AS (
    SELECT
        r.id                                    AS ride_id,
        r.passenger_id,
        r.driver_id,
        r.surge_multiplier,
        r.price_estimate,
        r.price_final,
        COALESCE(r.price_final, 0)
            / NULLIF(r.price_estimate, 0)       AS price_ratio,
        r.distance_km,
        r.duration_min,
        EXTRACT(HOUR FROM r.created_at)         AS hour_of_day,
        EXTRACT(DOW  FROM r.created_at)         AS day_of_week,
        r.created_at
    FROM rides r
    WHERE r.created_at >= NOW() - INTERVAL '7 days'
      AND r.price_estimate > 0
)
SELECT
    rps.*,
    COALESCE(ph.ride_count_1h, 0)              AS passenger_ride_count_1h,
    COALESCE(ph.spend_1h, 0)                   AS passenger_spend_1h,
    COALESCE(ph.suspicious_cancel_1h, 0)       AS passenger_suspicious_cancel_1h,
    COALESCE(dh.offer_count_1h, 0)             AS driver_offer_count_1h,
    COALESCE(dh.avg_score_1h, 0)               AS driver_avg_score_1h,
    COALESCE(dh.expired_1h, 0)                 AS driver_expired_1h
FROM ride_price_signals rps
LEFT JOIN passenger_hourly ph
    ON ph.passenger_id = rps.passenger_id
   AND ph.hour_bucket  = DATE_TRUNC('hour', rps.created_at)
LEFT JOIN driver_hourly dh
    ON dh.driver_id    = rps.driver_id
   AND dh.hour_bucket  = DATE_TRUNC('hour', rps.created_at)
ORDER BY rps.created_at DESC
"""

ANOMALY_FEATURE_COLS = [
    "price_ratio",
    "surge_multiplier",
    "distance_km",
    "duration_min",
    "hour_of_day",
    "day_of_week",
    "passenger_ride_count_1h",
    "passenger_spend_1h",
    "passenger_suspicious_cancel_1h",
    "driver_offer_count_1h",
    "driver_avg_score_1h",
    "driver_expired_1h",
]


def load_raw() -> pd.DataFrame:
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        return pd.read_sql(text(ANOMALY_QUERY), conn, parse_dates=["created_at"])


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    feat["price_ratio"]         = feat["price_ratio"].fillna(1.0).clip(0, 15)
    feat["surge_multiplier"]    = feat["surge_multiplier"].fillna(1.0)
    feat["distance_km"]         = feat["distance_km"].fillna(feat["distance_km"].median())
    feat["duration_min"]        = feat["duration_min"].fillna(feat["duration_min"].median())
    feat                        = feat.fillna(0)

    # Z-scores internes (sans fitter de scaler externe)
    for col in ANOMALY_FEATURE_COLS:
        if col in feat.columns:
            mu  = feat[col].mean()
            sig = feat[col].std() + 1e-8
            feat[f"{col}_z"] = (feat[col] - mu) / sig

    z_cols    = [c for c in feat.columns if c.endswith("_z")]
    available = [c for c in ANOMALY_FEATURE_COLS if c in feat.columns]

    id_cols = ["ride_id", "passenger_id", "driver_id", "created_at"]
    id_cols = [c for c in id_cols if c in feat.columns]

    return feat[id_cols + available + z_cols]


def build_lstm_residual_inputs(
    zone_hour_df: pd.DataFrame,
    predictions: np.ndarray,
    target_col: str = "ride_count",
) -> pd.DataFrame:
    """
    Calcule les résidus entre la demande réelle et la prédiction LSTM.
    Utilisé pour alimenter le détecteur de résidus.

    Args:
        zone_hour_df : DataFrame avec target_col (demande réelle)
        predictions  : array de prédictions LSTM (même longueur)
        target_col   : nom de la colonne cible dans zone_hour_df

    Returns:
        DataFrame avec colonnes : actual, predicted, residual, residual_z
    """
    n = min(len(zone_hour_df), len(predictions))
    df = zone_hour_df.iloc[:n].copy().reset_index(drop=True)
    df["predicted"]  = predictions[:n]
    df["residual"]   = df[target_col] - df["predicted"]
    mu  = df["residual"].mean()
    sig = df["residual"].std() + 1e-8
    df["residual_z"] = (df["residual"] - mu) / sig
    return df[["hour_bucket", target_col, "predicted", "residual", "residual_z"]]


def run() -> dict:
    print("[anomaly_features] Chargement…")
    raw = load_raw()
    print(f"  → {len(raw):,} rides chargés")
    feat = build_anomaly_features(raw)
    print(f"  → {len(feat):,} vecteurs anomalie prêts ({feat.shape[1]} colonnes)")
    return {"df": feat}


if __name__ == "__main__":
    run()
