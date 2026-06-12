"""
feature_engineering/churn_features.py

Features pour le classificateur de churn des chauffeurs.
Sources : drivers, driver_locations, dispatch_offers, ride_ratings, rides.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import DB, ML


CHURN_QUERY = """
WITH driver_base AS (
    SELECT
        d.user_id                       AS driver_id,
        d.rating_average,
        d.total_trips,
        d.total_ratings,
        d.availability_status,
        d.created_at                    AS driver_since,
        u.is_active,
        u.is_banned,
        u.last_login_at,
        dl.is_online,
        dl.is_on_trip,
        dl.last_seen_at
    FROM drivers d
    JOIN users u ON u.id = d.user_id
    LEFT JOIN driver_locations dl ON dl.driver_id = d.user_id
    WHERE d.deleted_at IS NULL
),
offer_stats AS (
    SELECT
        driver_id,
        COUNT(*)                                                   AS total_offers,
        SUM(CASE WHEN status = 'ACCEPTED' THEN 1 ELSE 0 END)      AS accepted_offers,
        SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END)      AS rejected_offers,
        SUM(CASE WHEN status = 'EXPIRED'  THEN 1 ELSE 0 END)      AS expired_offers,
        AVG(distance_to_pickup_km)                                 AS avg_pickup_distance,
        AVG(score)                                                 AS avg_dispatch_score,
        MAX(offered_at)                                            AS last_offer_at
    FROM dispatch_offers
    WHERE offered_at >= NOW() - INTERVAL ':lookback days'
    GROUP BY driver_id
),
rating_stats AS (
    SELECT
        r.driver_id,
        AVG(rr.driver_rating)       AS recent_avg_rating,
        STDDEV(rr.driver_rating)    AS rating_stddev,
        COUNT(rr.id)                AS recent_rating_count
    FROM ride_ratings rr
    JOIN rides r ON r.id = rr.ride_id
    WHERE r.completed_at >= NOW() - INTERVAL ':lookback days'
      AND rr.driver_rating IS NOT NULL
    GROUP BY r.driver_id
),
trip_stats AS (
    SELECT
        driver_id,
        COUNT(*)                                       AS recent_trips,
        AVG(distance_km_real)                          AS avg_trip_distance,
        AVG(duration_min_real)                         AS avg_trip_duration,
        AVG(price_final)                               AS avg_earnings_per_trip,
        MAX(completed_at)                              AS last_trip_at
    FROM rides
    WHERE completed_at >= NOW() - INTERVAL ':lookback days'
      AND status = 'COMPLETED'
    GROUP BY driver_id
)
SELECT
    db.*,
    COALESCE(os.total_offers, 0)        AS total_offers,
    COALESCE(os.accepted_offers, 0)     AS accepted_offers,
    COALESCE(os.rejected_offers, 0)     AS rejected_offers,
    COALESCE(os.expired_offers, 0)      AS expired_offers,
    os.avg_pickup_distance,
    os.avg_dispatch_score,
    os.last_offer_at,
    rs.recent_avg_rating,
    rs.rating_stddev,
    COALESCE(rs.recent_rating_count, 0) AS recent_rating_count,
    COALESCE(ts.recent_trips, 0)        AS recent_trips,
    ts.avg_trip_distance,
    ts.avg_trip_duration,
    ts.avg_earnings_per_trip,
    ts.last_trip_at
FROM driver_base db
LEFT JOIN offer_stats   os ON os.driver_id   = db.driver_id
LEFT JOIN rating_stats  rs ON rs.driver_id   = db.driver_id
LEFT JOIN trip_stats    ts ON ts.driver_id   = db.driver_id
"""


def load_raw(lookback_days: int = ML.churn_lookback_days) -> pd.DataFrame:
    engine = create_engine(DB.url)
    query = CHURN_QUERY.replace(":lookback", str(lookback_days))
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, parse_dates=["driver_since", "last_login_at",
                                                            "last_seen_at", "last_offer_at",
                                                            "last_trip_at"])


def label_churn(df: pd.DataFrame, inactive_days: int = 14) -> pd.Series:
    """
    Étiquette un chauffeur comme churné si :
      - Pas de trajet depuis `inactive_days` jours ET
      - Taux d'acceptation des offres < 30% OU pas d'offre du tout
    """
    now = pd.Timestamp.utcnow().tz_localize(None)
    days_since_trip = (now - df["last_trip_at"].dt.tz_localize(None)).dt.days.fillna(999)
    accept_rate = df["accepted_offers"] / (df["total_offers"] + 1)

    churned = (days_since_trip >= inactive_days) & (accept_rate < 0.30)
    return churned.astype(int)


def build_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le raw en feature matrix.

    Feature set final :
        rating_average, total_trips, total_ratings,
        days_since_last_trip, days_since_last_offer, days_since_last_login,
        accept_rate, reject_rate, expire_rate,
        avg_pickup_distance, avg_dispatch_score,
        recent_avg_rating, rating_stddev,
        recent_trips, avg_trip_distance, avg_earnings_per_trip,
        is_online (bool), availability_status_encoded
    """
    now = pd.Timestamp.utcnow().tz_localize(None)

    def days_since(col):
        return (now - df[col].dt.tz_localize(None)).dt.days.fillna(999)

    feat = pd.DataFrame()
    feat["driver_id"]               = df["driver_id"]
    feat["rating_average"]          = df["rating_average"].fillna(5.0)
    feat["total_trips"]             = df["total_trips"]
    feat["total_ratings"]           = df["total_ratings"]
    feat["days_since_last_trip"]    = days_since("last_trip_at")
    feat["days_since_last_offer"]   = days_since("last_offer_at")
    feat["days_since_last_login"]   = days_since("last_login_at")
    feat["accept_rate"]             = df["accepted_offers"] / (df["total_offers"] + 1)
    feat["reject_rate"]             = df["rejected_offers"] / (df["total_offers"] + 1)
    feat["expire_rate"]             = df["expired_offers"]  / (df["total_offers"] + 1)
    feat["avg_pickup_distance"]     = df["avg_pickup_distance"].fillna(df["avg_pickup_distance"].median())
    feat["avg_dispatch_score"]      = df["avg_dispatch_score"].fillna(0)
    feat["recent_avg_rating"]       = df["recent_avg_rating"].fillna(df["rating_average"])
    feat["rating_stddev"]           = df["rating_stddev"].fillna(0)
    feat["recent_trips"]            = df["recent_trips"]
    feat["avg_trip_distance"]       = df["avg_trip_distance"].fillna(0)
    feat["avg_earnings_per_trip"]   = df["avg_earnings_per_trip"].fillna(0)
    feat["is_online"] = np.where(df["is_online"].isna(), 0, df["is_online"].astype(bool).astype(int))

    status_map = {"active": 0, "pending": 1, "suspended": 2, "inactive": 3}
    feat["availability_encoded"]    = (
        df["availability_status"].map(status_map).fillna(1).astype(int)
    )

    feat["churn_label"] = label_churn(df)

    return feat


def run(lookback_days: int = ML.churn_lookback_days) -> dict:
    print(f"[churn_features] Chargement ({lookback_days}j)…")
    raw = load_raw(lookback_days)
    print(f"  → {len(raw):,} chauffeurs chargés")
    feat = build_churn_features(raw)
    churn_rate = feat["churn_label"].mean()
    print(f"  → {len(feat):,} feature vecteurs | taux de churn : {churn_rate:.1%}")
    return {"df": feat, "churn_rate": churn_rate}


if __name__ == "__main__":
    run()