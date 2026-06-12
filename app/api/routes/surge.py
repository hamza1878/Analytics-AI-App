from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import json

from app.db.database import get_db
from app.models.schemas import SurgeResponse
from app.services import ml_service, feature_service

router = APIRouter()

# Static zone metadata (in production: PostGIS table)
ZONE_META = {
    "zone_tunis_centre":  {"name": "Tunis Centre",  "lat": 36.8065, "lng": 10.1815},
    "zone_lac":           {"name": "Les Berges du Lac", "lat": 36.8320, "lng": 10.2298},
    "zone_marsa":         {"name": "La Marsa",       "lat": 36.8790, "lng": 10.3238},
    "zone_ariana":        {"name": "Ariana",          "lat": 36.8625, "lng": 10.1956},
    "zone_default":       {"name": "General",         "lat": 36.8190, "lng": 10.1660},
}


@router.get("/surge-recommendation", response_model=SurgeResponse)
async def surge_recommendation(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    redis = request.app.state.redis
    cache_key = "moviroo:surge"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    zone_features = await feature_service.get_surge_features(db)
    ml_zones = await ml_service.predict_surge(zone_features)

    zones = []
    for z in ml_zones:
        meta = ZONE_META.get(z["zone_id"], ZONE_META["zone_default"])
        zones.append({
            "zone_id": z["zone_id"],
            "zone_name": meta["name"],
            "lat": meta["lat"],
            "lng": meta["lng"],
            "current_surge": z["current_surge"],
            "recommended_surge": z["recommended_surge"],
            "demand_score": z["demand_score"],
            "supply_score": z["supply_score"],
            "demand_supply_ratio": z["demand_supply_ratio"],
        })

    payload = SurgeResponse(zones=zones, generated_at=datetime.now(timezone.utc))
    await redis.setex(cache_key, 30, json.dumps(payload.model_dump(), default=str))
    return payload
