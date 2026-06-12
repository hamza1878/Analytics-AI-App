"""
dashboard.py — Single endpoint that aggregates all ML outputs
for a frontend dashboard (KPIs, charts, alerts, rankings, heatmap).
"""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
import json
import asyncio

from app.db.database import get_db
from app.models.schemas import DashboardResponse, KPICard, ChartSeries, AlertItem
from app.services import ml_service, feature_service

router = APIRouter()


@router.get("/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
async def get_dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    redis = request.app.state.redis
    CACHE_KEY = "moviroo:dashboard"

    cached = await redis.get(CACHE_KEY)
    if cached:
        return json.loads(cached)

    # Parallel fetch
    kpis_task      = feature_service.get_overview_kpis(db)
    demand_task    = ml_service.predict_demand(hours=24)
    driver_task    = feature_service.get_driver_churn_features(db)
    payment_task   = feature_service.get_payment_features(db, hours=6)
    surge_task     = feature_service.get_surge_features(db)

    kpis, demand_result, driver_feats, payment_feats, zone_feats = await asyncio.gather(
        kpis_task, demand_task, driver_task, payment_task, surge_task
    )

    churn_preds, anomaly_raw, surge_raw = await asyncio.gather(
        ml_service.predict_churn(driver_feats),
        ml_service.detect_anomalies(payment_feats),
        ml_service.predict_surge(zone_feats),
    )

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    kpi_cards = [
        KPICard(
            label="Revenue (7d)",
            value=kpis.get("total_revenue_7d", 0),
            unit="USD",
            change_pct=float(kpis.get("revenue_growth_pct", 0)),
            trend="up" if float(kpis.get("revenue_growth_pct", 0)) >= 0 else "down",
        ),
        KPICard(
            label="Total Rides (7d)",
            value=kpis.get("total_rides_7d", 0),
            unit="rides",
            change_pct=float(kpis.get("rides_growth_pct", 0)),
            trend="up" if float(kpis.get("rides_growth_pct", 0)) >= 0 else "down",
        ),
        KPICard(
            label="Active Drivers",
            value=kpis.get("active_drivers", 0),
            unit="drivers",
            change_pct=0.0,
            trend="flat",
        ),
        KPICard(
            label="Avg Rating",
            value=kpis.get("avg_rating", 0),
            unit="/ 5.0",
            change_pct=0.0,
            trend="flat",
        ),
    ]

    # ── Demand Chart ───────────────────────────────────────────────────────────
    demand_chart = ChartSeries(
        name="Demand Forecast",
        data=[
            {"x": p["timestamp"], "y": p["demand"], "lower": p["lower"], "upper": p["upper"]}
            for p in demand_result["predictions"]
        ],
    )

    # ── Anomaly Alerts ─────────────────────────────────────────────────────────
    alerts = [
        AlertItem(
            id=a["id"],
            message=f"{a['type'].replace('_', ' ').title()} — {a['impact']}",
            severity=a["severity"],
            timestamp=a["detected_at"],
        )
        for a in anomaly_raw
    ]

    # ── Driver Risk Ranking (top 10) ───────────────────────────────────────────
    feature_map = {f["driver_id"]: f for f in driver_feats}
    driver_ranking = []
    for pred in sorted(churn_preds, key=lambda x: x["churn_probability"], reverse=True)[:10]:
        feat = feature_map.get(pred["driver_id"], {})
        prob = pred["churn_probability"]
        from app.api.routes.drivers import _classify_risk
        driver_ranking.append({
            "driver_id": pred["driver_id"],
            "driver_name": feat.get("name", "Unknown"),
            "churn_probability": prob,
            "risk_level": _classify_risk(prob),
            "rating": feat.get("rating", 0.0),
            "total_rides": feat.get("total_rides", 0),
            "days_since_last_ride": int(feat.get("days_since_last_ride") or 0),
        })

    # ── Surge Heatmap ─────────────────────────────────────────────────────────
    ZONE_META = {
        "zone_default": {"name": "General", "lat": 36.819, "lng": 10.166},
    }
    surge_zones = []
    for z in surge_raw["zones"]:
        meta = ZONE_META.get(z["zone_id"], ZONE_META["zone_default"])
        surge_zones.append({
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

    # ── Revenue Chart (placeholder 7-day) ──────────────────────────────────────
    revenue_chart = ChartSeries(
        name="Revenue Trend",
        data=[
            {"x": f"Day {i+1}", "y": round(14000 + i * 500 + (i % 2) * 200, 2)}
            for i in range(7)
        ],
    )

    payload = DashboardResponse(
        kpis=kpi_cards,
        demand_chart=demand_chart,
        revenue_chart=revenue_chart,
        anomaly_alerts=alerts,
        driver_risk_ranking=driver_ranking,
        surge_heatmap=surge_zones,
        generated_at=datetime.now(timezone.utc),
    )

    await redis.setex(CACHE_KEY, 30, json.dumps(payload.model_dump(), default=str))
    return payload
