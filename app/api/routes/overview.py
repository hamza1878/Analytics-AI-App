from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import json
import logging  

from app.db.database import get_db
from app.models.schemas import OverviewResponse
from app.services.feature_service import get_overview_kpis

logger = logging.getLogger(__name__)  

router = APIRouter()
router = APIRouter()


def safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default
@router.get("/overview", response_model=OverviewResponse, summary="Aggregated KPI overview")
async def get_overview(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        redis = getattr(request.app.state, "redis", None)
        CACHE_KEY = "moviroo:overview"

        if redis:
            try:
                cached = await redis.get(CACHE_KEY)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        kpis = await get_overview_kpis(db) or {}
        logger.info(f"✅ KPIs raw: {kpis}")  # ← ajoute ça

        payload = OverviewResponse(
            total_revenue_7d=safe_float(kpis.get("total_revenue_7d")),
            total_rides_7d=safe_int(kpis.get("total_rides_7d")),
            active_drivers=safe_int(kpis.get("active_drivers")),
            avg_rating=safe_float(kpis.get("avg_rating")),
            revenue_growth_pct=safe_float(kpis.get("revenue_growth_pct")),
            rides_growth_pct=safe_float(kpis.get("rides_growth_pct")),
            generated_at=datetime.now(timezone.utc),
        )
        return payload

    except Exception as e:
        logger.exception(f"❌ /overview crashed: {e}")  # ← et ça
        raise