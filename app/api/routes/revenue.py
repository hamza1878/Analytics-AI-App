from fastapi import APIRouter, Depends, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import json

from app.db.database import get_db
from app.models.schemas import RevenueForecastResponse
from app.services import ml_service, feature_service

router = APIRouter()


@router.get("/revenue-forecast", response_model=RevenueForecastResponse)
async def revenue_forecast(
    request: Request,
    days: int = Query(default=7, ge=1, le=30),
    db: AsyncSession = Depends(get_db),
):
    redis = request.app.state.redis
    cache_key = f"moviroo:revenue:{days}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    features = await feature_service.get_revenue_features(db, days=days * 4)
    ml_result = await ml_service.predict_revenue({"features": features, "forecast_days": days})

    payload = RevenueForecastResponse(
        predictions=ml_result["predictions"],
        total_predicted=ml_result["total_predicted"],
        total_baseline=ml_result["total_baseline"],
        total_uplift_pct=ml_result["total_uplift_pct"],
        metrics=ml_result["metrics"],
        generated_at=datetime.now(timezone.utc),
    )

    await redis.setex(cache_key, 300, json.dumps(payload.model_dump(), default=str))
    return payload
