"""
feature_engineering/route_features.py

Features pour le route optimizer (RL agent DQN).
Sources : dispatch_offers, rides, drivers, driver_locations.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import DB, ML


ROUTE_QUERY = """
SELECT
    do.id                           AS offer_id,
    do.ride_id,
    do.driver_id,
    do.status,
    do.distance_to_pickup_km,
    do.score                        AS dispatch_score,
    EXTRACT(EPOCH FROM (do.expires_at - do.offered_at)) / 60
                                    AS offer_window_min,
    do.offered_at,
    r.pickup_lat,
    r.pickup_lon,
    r.dropoff_lat,
    r.dropoff_lon,
    r.distance_km                   AS trip_distance_km,
    r.duration_min                  AS trip_duration_min,
    r.surge_multiplier,
    r.class_id,
    c.name                          AS class_name,
    EXTRACT(HOUR FROM do.offered_at) AS hour_of_day,
    EXTRACT(DOW  FROM do.offered_at) AS day_of_week,
    d.rating_average                AS driver_rating,
    d.total_trips                   AS driver_total_trips,
    d.availability_status,
    dl.latitude                     AS driver_lat,
    dl.longitude                    AS driver_lon,
    dl.speed_kmh                    AS driver_speed,
    dl.is_on_trip
FROM dispatch_offers do
JOIN rides r   ON r.id   = do.ride_id
JOIN classes c ON c.id   = r.class_id
JOIN drivers d ON d.user_id = do.driver_id
LEFT JOIN driver_locations dl ON dl.driver_id = do.driver_id
WHERE do.offered_at >= NOW() - INTERVAL ':lookback days'
ORDER BY do.offered_at
"""

ROUTE_FEATURE_COLS = [
    "distance_to_pickup_km",
    "trip_distance_km",
    "trip_duration_min",
    "surge_multiplier",
    "dispatch_score",
    "offer_window_min",
    "driver_rating",
    "driver_total_trips",
    "driver_speed",
    "hour_of_day",
    "day_of_week",
    "is_on_trip",
]


def load_raw(lookback_days: int = 30) -> pd.DataFrame:
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        return pd.read_sql(
            text(ROUTE_QUERY.replace(":lookback", str(lookback_days))),
            conn,
            parse_dates=["offered_at"],
        )


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_route_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()

    # Distance réelle chauffeur → pickup (si coords dispo)
    mask = feat["driver_lat"].notna() & feat["driver_lon"].notna()
    feat.loc[mask, "driver_to_pickup_km"] = haversine_km(
        feat.loc[mask, "driver_lat"], feat.loc[mask, "driver_lon"],
        feat.loc[mask, "pickup_lat"],  feat.loc[mask, "pickup_lon"],
    )
    feat["driver_to_pickup_km"] = feat["driver_to_pickup_km"].fillna(
        feat["distance_to_pickup_km"]
    )

    # Estimation temps d'arrivée chauffeur (ETA pickup)
    feat["eta_pickup_min"] = (
        feat["driver_to_pickup_km"] / (feat["driver_speed"].clip(lower=10) / 60)
    ).clip(upper=60)

    # Score normalisé
    feat["score_normalized"] = (
        (feat["dispatch_score"] - feat["dispatch_score"].min()) /
        (feat["dispatch_score"].max() - feat["dispatch_score"].min() + 1e-8)
    )

    # Features cycliques
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour_of_day"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour_of_day"] / 24)
    feat["dow_sin"]  = np.sin(2 * np.pi * feat["day_of_week"] / 7)
    feat["dow_cos"]  = np.cos(2 * np.pi * feat["day_of_week"] / 7)

    # Label : accepted=1, sinon 0
    feat["accepted"] = (feat["status"] == "accepted").astype(int)

    # One-hot class
    feat = pd.get_dummies(feat, columns=["class_name"], prefix="class", dtype=float)

    all_cols = ROUTE_FEATURE_COLS + [
        "driver_to_pickup_km", "eta_pickup_min", "score_normalized",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ] + [c for c in feat.columns if c.startswith("class_")]

    available = [c for c in all_cols if c in feat.columns]
    feat[available] = feat[available].fillna(0)

    return feat[["offer_id", "ride_id", "driver_id"] + available + ["accepted", "status"]].dropna(
        subset=["distance_to_pickup_km", "trip_distance_km"]
    )


def run(lookback_days: int = 30) -> dict:
    print(f"[route_features] Chargement ({lookback_days}j)…")
    raw = load_raw(lookback_days)
    print(f"  → {len(raw):,} offres dispatch chargées")

    accept_rate = (raw["status"] == "accepted").mean()
    print(f"  → Taux d'acceptation historique : {accept_rate:.1%}")

    feat = build_route_features(raw)
    print(f"  → {len(feat):,} vecteurs route prêts")
    return {"df": feat, "accept_rate": accept_rate}


if __name__ == "__main__":
    run()
