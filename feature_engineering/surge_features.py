"""
feature_engineering/surge_features.py

Features pour le modèle de prédiction du surge_multiplier.
Source principale : table `rides` (price_final / price_estimate).
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import DB, ML


SURGE_QUERY = """
SELECT
    r.id,
    r.pickup_lat,
    r.pickup_lon,
    r.surge_multiplier,
    r.price_estimate,
    r.price_final,
    r.distance_km,
    r.created_at,
    EXTRACT(HOUR FROM r.created_at)    AS hour_of_day,
    EXTRACT(DOW  FROM r.created_at)    AS day_of_week,
    EXTRACT(MONTH FROM r.created_at)   AS month,
    DATE_TRUNC('hour', r.created_at)   AS hour_bucket,
    COUNT(*) OVER (
        PARTITION BY DATE_TRUNC('hour', r.created_at)
    ) AS concurrent_rides_in_hour,
    c.name AS class_name
FROM rides r
JOIN classes c ON c.id = r.class_id
WHERE r.created_at >= NOW() - INTERVAL ':lookback days'
  AND r.price_estimate > 0
  AND r.price_final IS NOT NULL
ORDER BY r.created_at
"""


def load_raw(lookback_days: int = ML.surge_lookback_days) -> pd.DataFrame:
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        return pd.read_sql(
            text(SURGE_QUERY.replace(":lookback", str(lookback_days))),
            conn,
            parse_dates=["created_at", "hour_bucket"],
        )


def build_surge_features(df: pd.DataFrame, resolution: float = 0.05) -> pd.DataFrame:
    """
    Construit les features XGBoost pour prédire surge_multiplier.

    Features retournées :
        zone_lat, zone_lon, hour_of_day, day_of_week, month,
        concurrent_rides_in_hour, rolling_surge_1h, rolling_surge_3h,
        price_ratio (price_final / price_estimate),
        hour_sin, hour_cos, dow_sin, dow_cos,
        class_economy, class_business, class_premium  (one-hot)
    """
    df = df.copy()

    # Zone grille
    df["zone_lat"] = (df["pickup_lat"] // resolution) * resolution
    df["zone_lon"] = (df["pickup_lon"] // resolution) * resolution

    # Ratio réel du surge (cible auxiliaire pour validation)
    df["price_ratio"] = df["price_final"] / df["price_estimate"].clip(lower=0.01)

    # Rolling surge par zone
    df = df.sort_values("created_at")
    df["rolling_surge_1h"] = (
        df.groupby(["zone_lat", "zone_lon"])["surge_multiplier"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean())  # 4 × 15-min slots
    )
    df["rolling_surge_3h"] = (
        df.groupby(["zone_lat", "zone_lon"])["surge_multiplier"]
        .transform(lambda x: x.rolling(12, min_periods=1).mean())
    )

    # Features cycliques
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Classe one-hot
    df = pd.get_dummies(df, columns=["class_name"], prefix="class", dtype=float)

    feature_cols = [
        "zone_lat", "zone_lon",
        "hour_of_day", "day_of_week", "month",
        "concurrent_rides_in_hour",
        "rolling_surge_1h", "rolling_surge_3h",
        "price_ratio", "distance_km",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ] + [c for c in df.columns if c.startswith("class_")]

    available = [c for c in feature_cols if c in df.columns]
    return df[available + ["surge_multiplier"]].dropna()


def run(lookback_days: int = ML.surge_lookback_days) -> dict:
    print(f"[surge_features] Chargement ({lookback_days}j)…")
    raw = load_raw(lookback_days)
    print(f"  → {len(raw):,} rides chargés")
    feat = build_surge_features(raw)
    print(f"  → {len(feat):,} lignes de features prêtes")
    return {"df": feat}


if __name__ == "__main__":
    run()
