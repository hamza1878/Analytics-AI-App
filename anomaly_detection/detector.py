"""
anomaly_detection/detector.py

Détection d'anomalies hybride :
  - IsolationForest (features statiques : paiements, prix)
  - LSTM residuals (séries temporelles : demande, surge)

Génère des alertes structurées avec sévérité et payload.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlalchemy import create_engine, text
from config import DB


class Severity(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


@dataclass
class Anomaly:
    alert_type: str
    severity: Severity
    description: str
    affected_entity: str          # ride_id, driver_id, zone, etc.
    score: float                  # 0–1, 1 = plus anormal
    detected_at: datetime
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "alert_type":       self.alert_type,
            "severity":         self.severity.value,
            "description":      self.description,
            "affected_entity":  self.affected_entity,
            "score":            round(self.score, 4),
            "detected_at":      self.detected_at.isoformat(),
            "metadata":         self.metadata,
        }


# ─────────────────────────────────────────────
# 1. Payment Spike
# ─────────────────────────────────────────────

PAYMENT_QUERY = """
SELECT
    r.passenger_id,
    COUNT(r.id)                 AS rides_last_hour,
    SUM(r.price_final)          AS total_spend_last_hour,
    AVG(r.price_final)          AS avg_price_last_hour,
    p.total_bookings,
    p.membership_level
FROM rides r
JOIN passengers p ON p.user_id = r.passenger_id
WHERE r.created_at >= NOW() - INTERVAL '1 hour'
  AND r.price_final IS NOT NULL
GROUP BY r.passenger_id, p.total_bookings, p.membership_level
HAVING COUNT(r.id) >= 3 OR SUM(r.price_final) > 500
"""


def detect_payment_spikes(threshold_z: float = 3.5) -> list[Anomaly]:
    """Détecte des volumes inhabituels de paiement par passager."""
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        df = pd.read_sql(text(PAYMENT_QUERY), conn)

    if df.empty:
        return []

    z_rides = (df["rides_last_hour"] - df["rides_last_hour"].mean()) / (df["rides_last_hour"].std() + 1e-8)
    z_spend = (df["total_spend_last_hour"] - df["total_spend_last_hour"].mean()) / (df["total_spend_last_hour"].std() + 1e-8)

    anomalies = []
    for _, row in df[z_rides.abs() > threshold_z | (z_spend > threshold_z)].iterrows():
        score = float(max(abs(z_rides[row.name]), z_spend[row.name]) / 10)
        score = min(1.0, score)
        anomalies.append(Anomaly(
            alert_type="PAYMENT_SPIKE",
            severity=Severity.HIGH if score > 0.7 else Severity.MEDIUM,
            description=(
                f"Passager {str(row['passenger_id'])[:8]} — "
                f"{int(row['rides_last_hour'])} trajets en 1h, "
                f"spend total {row['total_spend_last_hour']:.0f}€"
            ),
            affected_entity=str(row["passenger_id"]),
            score=score,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "rides_last_hour":        int(row["rides_last_hour"]),
                "total_spend":            float(row["total_spend_last_hour"]),
                "membership_level":       row["membership_level"],
                "z_score_rides":          float(z_rides[row.name]),
            },
        ))
    return anomalies


# ─────────────────────────────────────────────
# 2. Surge Mismatch
# ─────────────────────────────────────────────

SURGE_MISMATCH_QUERY = """
SELECT
    id,
    pickup_lat,
    pickup_lon,
    price_estimate,
    price_final,
    surge_multiplier,
    COALESCE(price_final, 0) / NULLIF(price_estimate, 0) AS actual_ratio
FROM rides
WHERE created_at >= NOW() - INTERVAL '2 hours'
  AND price_estimate > 0
  AND price_final IS NOT NULL
  AND ABS(COALESCE(price_final, 0) / NULLIF(price_estimate, 0) - COALESCE(surge_multiplier, 1)) > 0.5
"""


def detect_surge_mismatch() -> list[Anomaly]:
    """Détecte les incohérences entre surge_multiplier déclaré et ratio prix réel."""
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        df = pd.read_sql(text(SURGE_MISMATCH_QUERY), conn)

    anomalies = []
    for _, row in df.iterrows():
        delta = abs(row["actual_ratio"] - (row["surge_multiplier"] or 1.0))
        score = min(1.0, delta / 2.0)
        anomalies.append(Anomaly(
            alert_type="SURGE_MISMATCH",
            severity=Severity.HIGH if delta > 1.0 else Severity.MEDIUM,
            description=(
                f"Ride {str(row['id'])[:8]} — surge déclaré {row['surge_multiplier']:.2f}× "
                f"vs ratio réel {row['actual_ratio']:.2f}× (Δ={delta:.2f})"
            ),
            affected_entity=str(row["id"]),
            score=score,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "price_estimate":  float(row["price_estimate"]),
                "price_final":     float(row["price_final"]),
                "surge_declared":  float(row["surge_multiplier"] or 1),
                "actual_ratio":    float(row["actual_ratio"]),
                "delta":           float(delta),
            },
        ))
    return anomalies


# ─────────────────────────────────────────────
# 3. Rating Drift
# ─────────────────────────────────────────────

RATING_DRIFT_QUERY = """
SELECT
    r.driver_id,
    AVG(rr.driver_rating)                                           AS recent_avg,
    STDDEV(rr.driver_rating)                                        AS recent_std,
    COUNT(rr.id)                                                    AS recent_count,
    d.rating_average                                                AS historical_avg
FROM ride_ratings rr
JOIN rides r ON r.id = rr.ride_id
JOIN drivers d ON d.user_id = r.driver_id
WHERE r.completed_at >= NOW() - INTERVAL '7 days'
  AND rr.driver_rating IS NOT NULL
GROUP BY r.driver_id, d.rating_average
HAVING ABS(AVG(rr.driver_rating) - d.rating_average) > 0.5
   AND COUNT(rr.id) >= 5
"""


def detect_rating_drift() -> list[Anomaly]:
    """Détecte les chauffeurs dont la note récente dérive significativement."""
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        df = pd.read_sql(text(RATING_DRIFT_QUERY), conn)

    anomalies = []
    for _, row in df.iterrows():
        drift = abs(row["recent_avg"] - row["historical_avg"])
        score = min(1.0, drift / 2.0)
        anomalies.append(Anomaly(
            alert_type="RATING_DRIFT",
            severity=Severity.MEDIUM if drift < 1.0 else Severity.HIGH,
            description=(
                f"Driver {str(row['driver_id'])[:8]} — "
                f"note 7j: {row['recent_avg']:.2f} vs historique: {row['historical_avg']:.2f} "
                f"(Δ={drift:.2f})"
            ),
            affected_entity=str(row["driver_id"]),
            score=score,
            detected_at=datetime.now(timezone.utc),
            metadata={
                "recent_avg":     float(row["recent_avg"]),
                "historical_avg": float(row["historical_avg"]),
                "drift":          float(drift),
                "sample_count":   int(row["recent_count"]),
            },
        ))
    return anomalies


# ─────────────────────────────────────────────
# 4. LSTM Residual Anomaly
# ─────────────────────────────────────────────

def detect_lstm_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    zone_id: str,
    threshold_sigma: float = 2.5,
) -> list[Anomaly]:
    """
    Détecte les points où l'erreur LSTM dépasse threshold_sigma écarts-types.
    À appeler après chaque inférence du modèle de demande.
    """
    residuals = actual - predicted
    mu  = residuals.mean()
    sig = residuals.std() + 1e-8
    z   = (residuals - mu) / sig

    anomalies = []
    for i, (zi, res) in enumerate(zip(z, residuals)):
        if abs(zi) > threshold_sigma:
            score = min(1.0, abs(zi) / 5.0)
            anomalies.append(Anomaly(
                alert_type="DEMAND_FORECAST_RESIDUAL",
                severity=Severity.HIGH if abs(zi) > 4 else Severity.MEDIUM,
                description=(
                    f"Zone {zone_id} h+{i} — demande réelle {actual[i]:.0f} "
                    f"vs prédite {predicted[i]:.0f} (z={zi:.2f})"
                ),
                affected_entity=f"zone:{zone_id}:h+{i}",
                score=score,
                detected_at=datetime.now(timezone.utc),
                metadata={
                    "actual":    float(actual[i]),
                    "predicted": float(predicted[i]),
                    "residual":  float(res),
                    "z_score":   float(zi),
                },
            ))
    return anomalies


# ─────────────────────────────────────────────
# Orchestrateur principal
# ─────────────────────────────────────────────

def run_all_detectors() -> list[dict]:
    """Exécute tous les détecteurs et retourne une liste unifiée d'alertes."""
    all_anomalies: list[Anomaly] = []

    print("[detector] Analyse des paiements…")
    try:
        all_anomalies += detect_payment_spikes()
    except Exception as e:
        print(f"  Erreur payment_spikes : {e}")

    print("[detector] Analyse surge mismatch…")
    try:
        all_anomalies += detect_surge_mismatch()
    except Exception as e:
        print(f"  Erreur surge_mismatch : {e}")

    print("[detector] Analyse rating drift…")
    try:
        all_anomalies += detect_rating_drift()
    except Exception as e:
        print(f"  Erreur rating_drift : {e}")

    # Trier par score décroissant
    all_anomalies.sort(key=lambda a: a.score, reverse=True)

    print(f"[detector] {len(all_anomalies)} anomalies détectées")
    return [a.to_dict() for a in all_anomalies]


if __name__ == "__main__":
    alerts = run_all_detectors()
    for a in alerts:
        print(f"  [{a['severity']}] {a['alert_type']} — {a['description'][:80]}")
