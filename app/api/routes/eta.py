from fastapi import APIRouter, Request, Query
from datetime import datetime, timezone
import json

from app.models.schemas import ETAResponse
from app.services import ml_service

router = APIRouter()


@router.get("/eta-prediction", response_model=ETAResponse)
async def eta_prediction(
    request: Request,
    distance: float = Query(..., gt=0, le=500, description="Distance in kilometers"),
):
    redis = request.app.state.redis
    cache_key = f"moviroo:eta:{distance}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    hour_of_day = datetime.now(timezone.utc).hour
    ml_result = await ml_service.predict_eta(distance_km=distance, hour_of_day=hour_of_day)

    payload = ETAResponse(
        distance_km=distance,
        predicted_minutes=ml_result["predicted_minutes"],
        confidence_interval=ml_result["confidence_interval"],
        traffic_factor=ml_result["traffic_factor"],
        model_version=ml_result.get("model_version", "1.0.0"),
        generated_at=datetime.now(timezone.utc),
    )

    await redis.setex(cache_key, 60, json.dumps(payload.model_dump(), default=str))
    return payload
