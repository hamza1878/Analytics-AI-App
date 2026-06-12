from fastapi import APIRouter, Depends, Request, Query
from pydantic.v1 import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import json

from app.db.database import get_db
from app.models.schemas import DemandForecastResponse
from app.services import ml_service, feature_service

router = APIRouter()

@router.get("/demand-forecast", response_model=DemandForecastResponse)
async def demand_forecast(
    request: Request,
    hours: int = Query(default=24, ge=1, le=168, description="Forecast horizon in hours"),
    db: AsyncSession = Depends(get_db),
):
    redis = request.app.state.redis
    cache_key = f"moviroo:demand:{hours}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # 1. Extract features from DB
    features = await feature_service.get_demand_features(db, hours=hours * 4)  # 4x lookback

    # 2. Call ML service
    ml_result = await ml_service.predict_demand(hours=hours)

    payload = DemandForecastResponse(
        predictions=ml_result["predictions"],
        metrics=ml_result["metrics"],
        model_version=ml_result.get("model_version", "1.0.0"),
        generated_at=datetime.now(timezone.utc),
    )

    await redis.setex(cache_key, 300, json.dumps(payload.model_dump(), default=str))
    return payload
