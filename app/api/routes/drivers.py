from fastapi import APIRouter, Depends, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import json

from app.db.database import get_db
from app.models.schemas import DriverRiskResponse, PaginationMeta, RiskLevel
from app.services import ml_service, feature_service

router = APIRouter()


def _classify_risk(prob: float) -> RiskLevel:
    if prob >= 0.75:
        return RiskLevel.CRITICAL
    elif prob >= 0.50:
        return RiskLevel.HIGH
    elif prob >= 0.25:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


@router.get("/driver-risk", response_model=DriverRiskResponse)
async def driver_risk(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    risk_level: str = Query(default=None, description="Filter: low|medium|high|critical"),
    db: AsyncSession = Depends(get_db),
):
    redis = request.app.state.redis
    cache_key = f"moviroo:driver_risk:{page}:{page_size}:{risk_level}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    driver_features = await feature_service.get_driver_churn_features(db)
    predictions = await ml_service.predict_churn(driver_features)

    # Merge predictions with features
    feature_map = {f["driver_id"]: f for f in driver_features}
    items = []
    for pred in predictions:
        feat = feature_map.get(pred["driver_id"], {})
        prob = pred["churn_probability"]
        level = _classify_risk(prob)

        if risk_level and level.value != risk_level:
            continue

        items.append({
            "driver_id": pred["driver_id"],
            "driver_name": feat.get("name", "Unknown"),
            "churn_probability": prob,
            "risk_level": level,
            "rating": feat.get("rating", 0.0),
            "total_rides": feat.get("total_rides", 0),
            "days_since_last_ride": int(feat.get("days_since_last_ride", 0) or 0),
        })

    # Sort by churn_probability desc
    items.sort(key=lambda x: x["churn_probability"], reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    paginated = items[start: start + page_size]

    payload = DriverRiskResponse(
        drivers=paginated,
        total_at_risk=sum(1 for i in items if i["churn_probability"] >= 0.5),
        meta=PaginationMeta(page=page, page_size=page_size, total=total),
        generated_at=datetime.now(timezone.utc),
    )

    await redis.setex(cache_key, 120, json.dumps(payload.model_dump(), default=str))
    return payload
