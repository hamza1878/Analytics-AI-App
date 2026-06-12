"""
models/fraud_detector.py

IsolationForest pour la détection d'anomalies / fraudes.
Sources : rides (price), passengers (payment_addresses), dispatch_offers.
Cible : Precision > 0.95 sur les alertes générées
"""
import numpy as np
import pandas as pd
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sqlalchemy import create_engine, text
from config import DB, ML


FRAUD_QUERY = """
SELECT
    r.id                            AS ride_id,
    r.passenger_id,
    r.price_estimate,
    r.price_final,
    r.surge_multiplier,
    r.distance_km,
    r.duration_min,
    COALESCE(r.price_final, 0) / NULLIF(r.price_estimate, 0)
                                    AS price_ratio,
    r.created_at,
    EXTRACT(HOUR FROM r.created_at) AS hour_of_day,
    p.total_bookings,
    p.membership_points,
    p.membership_level,
    CASE WHEN p.default_payment_method IS NULL THEN 1 ELSE 0 END
                                    AS no_payment_method,
    CASE WHEN r.cancelled_at IS NOT NULL
         AND r.driver_id IS NOT NULL
         AND r.trip_started_at IS NULL THEN 1 ELSE 0 END
                                    AS suspicious_cancel,
    COALESCE(AVG(rr.passenger_rating) OVER (
        PARTITION BY r.passenger_id
    ), 5) AS avg_passenger_rating
FROM rides r
JOIN passengers p ON p.user_id = r.passenger_id
LEFT JOIN ride_ratings rr ON rr.ride_id = r.id
WHERE r.created_at >= NOW() - INTERVAL ':lookback days'
  AND r.price_estimate > 0
"""


FRAUD_FEATURE_COLS = [
    "price_ratio",
    "surge_multiplier",
    "distance_km",
    "duration_min",
    "hour_of_day",
    "total_bookings",
    "membership_points",
    "no_payment_method",
    "suspicious_cancel",
    "avg_passenger_rating",
]


def load_raw(lookback_days: int = 30) -> pd.DataFrame:
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        return pd.read_sql(
            text(FRAUD_QUERY.replace(":lookback", str(lookback_days))),
            conn,
            parse_dates=["created_at"],
        )


def build_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df[FRAUD_FEATURE_COLS].copy()

    # Imputation
    feat["price_ratio"]         = feat["price_ratio"].fillna(1.0).clip(0, 10)
    feat["surge_multiplier"]    = feat["surge_multiplier"].fillna(1.0)
    feat["distance_km"]         = feat["distance_km"].fillna(feat["distance_km"].median())
    feat["duration_min"]        = feat["duration_min"].fillna(feat["duration_min"].median())
    feat["avg_passenger_rating"] = feat["avg_passenger_rating"].fillna(5.0)
    feat = feat.fillna(0)

    return feat


class FraudDetector:
    """Wrapper autour d'IsolationForest avec scaler intégré."""

    def __init__(self, contamination: float = 0.008):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=contamination,
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X: np.ndarray) -> "FraudDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne 1 (normal) ou -1 (anomalie)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Retourne le score d'anomalie (plus négatif = plus suspect)."""
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

    def anomaly_probability(self, X: np.ndarray) -> np.ndarray:
        """Normalise les scores vers [0, 1] où 1 = très suspect."""
        scores = self.score_samples(X)
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return 1 - norm  # inverser : 1 = anomalie


def evaluate_with_labels(
    detector: FraudDetector,
    X: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Évaluation si des étiquettes sont disponibles (ex : fraudes confirmées)."""
    preds = detector.predict(X)
    y_pred = (preds == -1).astype(int)

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "anomaly_rate": y_pred.mean(),
    }


def train_and_log(df: pd.DataFrame) -> str:
    feat = build_fraud_features(df)
    X = feat.values

    detector = FraudDetector(contamination=0.008)
    detector.fit(X)

    preds = detector.predict(X)
    anomaly_count = (preds == -1).sum()
    anomaly_rate  = anomaly_count / len(preds)

    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    mlflow.set_experiment(ML.mlflow_experiment)

    with mlflow.start_run(run_name="fraud_detector_v2") as run:
        mlflow.log_params({
            "model": "IsolationForest",
            "n_estimators": 300,
            "contamination": 0.008,
            "n_features": len(FRAUD_FEATURE_COLS),
        })
        mlflow.log_metrics({
            "anomaly_count": float(anomaly_count),
            "anomaly_rate":  float(anomaly_rate),
        })

        print(f"[fraud_detector] {anomaly_count} anomalies sur {len(preds)} "
              f"({anomaly_rate:.2%}) — contamination={0.008}")

        return run.info.run_id


if __name__ == "__main__":
    raw = load_raw()
    train_and_log(raw)
