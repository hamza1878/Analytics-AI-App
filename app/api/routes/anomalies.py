from fastapi import APIRouter, Depends, Request, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime, timezone
import json, uuid

from app.db.database import get_db
from app.models.schemas import (
    AnomaliesResponse, AnomalyActionRequest, AnomalyActionResponse, AnomalySeverity
)
from app.services import ml_service, feature_service

router = APIRouter()


# GET /anomalies/anomalies          ← used by frontend via /api/anomalies/anomalies
# GET /anomalies/churn              ← used by frontend via /api/anomalies/churn
# GET /anomalies/surge              ← used by frontend via /api/anomalies/surge

@router.get("/anomalies", response_model=AnomaliesResponse)
async def get_anomalies(
    request: Request,
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
):
    redis = request.app.state.redis
    cache_key = f"moviroo:anomalies:{hours}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    payment_features = await feature_service.get_payment_features(db, hours=hours)

    # ── ml_service.detect_anomalies must call GET not POST ─────────────────
    raw_anomalies = await ml_service.detect_anomalies(payment_features)

    anomalies = []
    for a in raw_anomalies:
        anomalies.append({
            "id":          a.get("id", str(uuid.uuid4())),
            "type":        a["type"],
            "severity":    a["severity"],
            "confidence":  a["confidence"],
            "impact":      a["impact"],
            "action":      a["action"],
            "ride_id":     a.get("ride_id"),
            "driver_id":   a.get("driver_id"),
            "detected_at": a.get("detected_at", datetime.now(timezone.utc).isoformat()),
            "resolved":    False,
        })

    payload = AnomaliesResponse(
        anomalies=anomalies,
        total=len(anomalies),
        critical_count=sum(1 for a in anomalies if a["severity"] == AnomalySeverity.CRITICAL),
        generated_at=datetime.now(timezone.utc),
    )

    await redis.setex(cache_key, 60, json.dumps(payload.model_dump(), default=str))
    return payload


@router.get("/churn")
async def get_churn_anomalies(
    request: Request,
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
):
    """
    Churn-specific anomalies — subset of anomalies where type contains 'churn'.
    Frontend calls: GET /api/anomalies/churn
    """
    redis = request.app.state.redis
    cache_key = f"moviroo:anomalies:churn:{hours}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    payment_features = await feature_service.get_payment_features(db, hours=hours)
    raw_anomalies    = await ml_service.detect_anomalies(payment_features)

    churn_anomalies = [
        a for a in raw_anomalies
        if "churn" in str(a.get("type", "")).lower()
    ]

    result = {
        "anomalies":     churn_anomalies,
        "total":         len(churn_anomalies),
        "generated_at":  datetime.now(timezone.utc).isoformat(),
    }
    await redis.setex(cache_key, 120, json.dumps(result, default=str))
    return result


@router.get("/surge")
async def get_surge_anomalies(
    request: Request,
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
):
    """
    Surge-related anomalies.
    Frontend calls: GET /api/anomalies/surge  (previously /api/demand-forecast/surge)
    """
    redis = request.app.state.redis
    cache_key = f"moviroo:anomalies:surge:{hours}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    payment_features = await feature_service.get_payment_features(db, hours=hours)
    raw_anomalies    = await ml_service.detect_anomalies(payment_features)

    surge_anomalies = [
        a for a in raw_anomalies
        if "surge" in str(a.get("type", "")).lower()
           or float(a.get("confidence", 0)) > 0.8
    ]

    result = {
        "anomalies":    surge_anomalies,
        "total":        len(surge_anomalies),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    await redis.setex(cache_key, 120, json.dumps(result, default=str))
    return result


@router.post("/anomalies/{anomaly_id}/action", response_model=AnomalyActionResponse)
async def trigger_anomaly_action(
    request: Request,
    anomaly_id: str = Path(...),
    body: AnomalyActionRequest = ...,
    db: AsyncSession = Depends(get_db),
):
    await db.execute(
        text("""
            INSERT INTO anomaly_actions (anomaly_id, action_type, notes, executed_at)
            VALUES (:anomaly_id, :action_type, :notes, NOW())
            ON CONFLICT DO NOTHING
        """),
        {"anomaly_id": anomaly_id, "action_type": body.action_type, "notes": body.notes},
    )
    await db.commit()

    redis = request.app.state.redis
    await redis.delete("moviroo:anomalies:24")

    return AnomalyActionResponse(
        anomaly_id=anomaly_id,
        action_taken=body.action_type,
        status="executed",
        executed_at=datetime.now(timezone.utc),
    )