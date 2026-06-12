"""
feature_engineering/eta_features.py

Features pour l'estimateur ETA (LightGBM).
Source principale : trip_waypoints (GPS séquentiel réel) + rides.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import DB, ML


ETA_QUERY = """
WITH waypoint_stats AS (
    SELECT
        ride_id,
        COUNT(*)                    AS waypoint_count,
        AVG(speed_kmh)              AS avg_speed_kmh,
        MAX(speed_kmh)              AS max_speed_kmh,
        STDDEV(speed_kmh)           AS speed_stddev,
        MIN(recorded_at)            AS first_waypoint_at,
        MAX(recorded_at)            AS last_waypoint_at,
        EXTRACT(EPOCH FROM (MAX(recorded_at) - MIN(recorded_at))) / 60
                                    AS actual_duration_min
    FROM trip_waypoints
    GROUP BY ride_id
)
SELECT
    r.id                            AS ride_id,
    r.distance_km,
    r.distance_km_real,
    r.duration_min                  AS estimated_duration_min,
    r.duration_min_real             AS actual_duration_min_rides,
    r.pickup_lat,
    r.pickup_lon,
    r.dropoff_lat,
    r.dropoff_lon,
    r.surge_multiplier,
    r.price_final,
    EXTRACT(HOUR FROM r.trip_started_at) AS hour_of_day,
    EXTRACT(DOW  FROM r.trip_started_at) AS day_of_week,
    c.name                          AS class_name,
    ws.waypoint_count,
    ws.avg_speed_kmh,
    ws.max_speed_kmh,
    ws.speed_stddev,
    ws.actual_duration_min          AS actual_duration_waypoints
FROM rides r
JOIN classes c ON c.id = r.class_id
JOIN waypoint_stats ws ON ws.ride_id = r.id
WHERE r.status = 'COMPLETED'
  AND r.completed_at >= NOW() - INTERVAL ':lookback days'
  AND r.distance_km IS NOT NULL
  AND r.duration_min_real IS NOT NULL
"""


def load_raw(lookback_days: int = ML.eta_lookback_days) -> pd.DataFrame:
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        return pd.read_sql(
            text(ETA_QUERY.replace(":lookback", str(lookback_days))),
            conn,
        )


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_eta_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    # Distance orthodromique vs distance GPS réelle
    feat["straight_line_km"] = haversine_km(
        feat["pickup_lat"], feat["pickup_lon"],
        feat["dropoff_lat"], feat["dropoff_lon"],
    )
    feat["detour_ratio"] = (
        feat["distance_km_real"].fillna(feat["distance_km"]) / feat["straight_line_km"].clip(lower=0.1)
    )

    # Vitesse moyenne réelle
    feat["effective_speed_kmh"] = (
        feat["distance_km_real"].fillna(feat["distance_km"]) /
        (feat["actual_duration_min_rides"].fillna(1) / 60)
    ).clip(upper=200)

    # Ratio ETA estimé vs waypoints
    feat["eta_estimate_error"] = (
        feat["estimated_duration_min"] - feat["actual_duration_waypoints"].fillna(feat["estimated_duration_min"])
    )

    # Features cycliques
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour_of_day"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour_of_day"] / 24)
    feat["dow_sin"]  = np.sin(2 * np.pi * feat["day_of_week"] / 7)
    feat["dow_cos"]  = np.cos(2 * np.pi * feat["day_of_week"] / 7)

    # One-hot class
    feat = pd.get_dummies(feat, columns=["class_name"], prefix="class", dtype=float)

    feature_cols = [
        "distance_km", "straight_line_km", "detour_ratio",
        "avg_speed_kmh", "max_speed_kmh", "speed_stddev", "effective_speed_kmh",
        "waypoint_count", "surge_multiplier",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "eta_estimate_error",
    ] + [c for c in feat.columns if c.startswith("class_")]

    available = [c for c in feature_cols if c in feat.columns]
    target = "actual_duration_min_rides"

    return feat[available + [target]].dropna()


def run(lookback_days: int = ML.eta_lookback_days) -> dict:
    print(f"[eta_features] Chargement ({lookback_days}j)…")
    raw = load_raw(lookback_days)
    print(f"  → {len(raw):,} trajets complétés chargés")
    feat = build_eta_features(raw)
    print(f"  → {len(feat):,} feature vecteurs ETA prêts")
    return {"df": feat}


if __name__ == "__main__":
    run()
