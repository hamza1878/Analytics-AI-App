"""
feature_engineering/fraud_features.py

Features pour la détection de fraude / anomalies financières.
Sources : rides, passengers, ride_ratings.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import DB, ML


FRAUD_FEATURE_QUERY = """
SELECT
    r.id                        AS ride_id,
    r.passenger_id,
    r.driver_id,
    r.price_estimate,
    r.price_final,
    r.surge_multiplier,
    r.distance_km,
    r.duration_min,
    COALESCE(r.price_final, 0) / NULLIF(r.price_estimate, 0)
                                AS price_ratio,
    EXTRACT(HOUR FROM r.created_at) AS hour_of_day,
    EXTRACT(DOW  FROM r.created_at) AS day_of_week,
    r.created_at,
    p.total_bookings,
    p.membership_points,
    p.membership_level,
    CASE WHEN p.default_payment_method IS NULL THEN 1 ELSE 0 END
                                AS no_payment_method,
    CASE WHEN r.cancelled_at IS NOT NULL
         AND r.driver_id IS NOT NULL
         AND r.trip_started_at IS NULL THEN 1 ELSE 0 END
                                AS suspicious_cancel,
    COALESCE(rr.passenger_rating, 5)  AS last_passenger_rating,
    COALESCE(rr.driver_rating, 5)     AS last_driver_rating
FROM rides r
JOIN passengers p ON p.user_id = r.passenger_id
LEFT JOIN ride_ratings rr ON rr.ride_id = r.id
WHERE r.created_at >= NOW() - INTERVAL ':lookback days'
  AND r.price_estimate > 0
ORDER BY r.created_at DESC
"""

FRAUD_FEATURE_COLS = [
    "price_ratio",
    "surge_multiplier",
    "distance_km",
    "duration_min",
    "hour_of_day",
    "day_of_week",
    "total_bookings",
    "membership_points",
    "no_payment_method",
    "suspicious_cancel",
    "last_passenger_rating",
    "last_driver_rating",
]


def load_raw(lookback_days: int = 30) -> pd.DataFrame:
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        return pd.read_sql(
            text(FRAUD_FEATURE_QUERY.replace(":lookback", str(lookback_days))),
            conn,
            parse_dates=["created_at"],
        )


def build_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    # Nettoyage et imputation
    feat["price_ratio"]            = feat["price_ratio"].fillna(1.0).clip(0, 15)
    feat["surge_multiplier"]       = feat["surge_multiplier"].fillna(1.0)
    feat["distance_km"]            = feat["distance_km"].fillna(feat["distance_km"].median())
    feat["duration_min"]           = feat["duration_min"].fillna(feat["duration_min"].median())
    feat["last_passenger_rating"]  = feat["last_passenger_rating"].fillna(5.0)
    feat["last_driver_rating"]     = feat["last_driver_rating"].fillna(5.0)
    feat                           = feat.fillna(0)

    # Features dérivées
    feat["price_per_km"]     = feat["price_final"] / (feat["distance_km"] + 0.1)
    feat["speed_kmh"]        = feat["distance_km"] / ((feat["duration_min"] / 60) + 0.01)
    feat["speed_kmh"]        = feat["speed_kmh"].clip(0, 200)
    feat["hour_sin"]         = np.sin(2 * np.pi * feat["hour_of_day"] / 24)
    feat["hour_cos"]         = np.cos(2 * np.pi * feat["hour_of_day"] / 24)

    all_cols = FRAUD_FEATURE_COLS + ["price_per_km", "speed_kmh", "hour_sin", "hour_cos"]
    available = [c for c in all_cols if c in feat.columns]

    return feat[["ride_id", "passenger_id"] + available].dropna()


def run(lookback_days: int = 30) -> dict:
    print(f"[fraud_features] Chargement ({lookback_days}j)…")
    raw = load_raw(lookback_days)
    print(f"  → {len(raw):,} rides chargés")
    feat = build_fraud_features(raw)
    print(f"  → {len(feat):,} vecteurs de features fraude prêts")
    return {"df": feat}


if __name__ == "__main__":
    run()
