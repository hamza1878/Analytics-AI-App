"""
ml_service.py
Thin HTTP client wrapping all ML model endpoints.
The ML models run as a separate Python microservice (ml_server.py on :8005).
"""
import httpx
import logging
from typing import Any, Dict, List
from app.core.config import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient = None


def get_ml_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.ML_SERVICE_URL,   # http://localhost:8005
            timeout=settings.ML_SERVICE_TIMEOUT,
        )
    return _client


# ── Demand ────────────────────────────────────────────────────────────────────
# ml_server: GET /predict/demand?hours=N

async def predict_demand(hours: int = 24) -> Dict[str, Any]:
    client = get_ml_client()
    resp = await client.get("/predict/demand", params={"hours": hours})
    resp.raise_for_status()
    return resp.json()


# ── Revenue ───────────────────────────────────────────────────────────────────
# ml_server: GET or POST /predict/revenue?forecast_days=N

async def predict_revenue(features: Dict[str, Any]) -> Dict[str, Any]:
    client = get_ml_client()
    forecast_days = features.get("forecast_days", 7)
    resp = await client.get("/predict/revenue", params={"forecast_days": forecast_days})
    resp.raise_for_status()
    return resp.json()


# ── Driver Churn ──────────────────────────────────────────────────────────────
# ml_server: POST /predict/churn  {"driver_ids": [...] | null}

async def predict_churn(driver_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    client = get_ml_client()
    # Extract driver_ids if passed as feature dicts
    driver_ids = [
        f["driver_id"] for f in driver_features
        if "driver_id" in f
    ] or None
    resp = await client.post("/predict/churn", json={"driver_ids": driver_ids})
    resp.raise_for_status()
    return resp.json().get("predictions", [])


# ── Anomaly Detection ─────────────────────────────────────────────────────────
# ml_server: GET /predict/anomalies?hours=N   ← GET not POST!

async def detect_anomalies(
    payment_features: List[Dict[str, Any]],
    hours: int = 24,
) -> List[Dict[str, Any]]:
    client = get_ml_client()
    try:
        resp = await client.get("/predict/anomalies", params={"hours": hours})
        resp.raise_for_status()
        return resp.json().get("anomalies", [])
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[ml_service] detect_anomalies HTTP {e.response.status_code}: {e.response.text}"
        )
        return []
    except httpx.RequestError as e:
        logger.error(f"[ml_service] detect_anomalies connection error: {e}")
        return []


# ── Surge Pricing ─────────────────────────────────────────────────────────────
# ml_server: POST /predict/surge  {"zones": [...]}

async def predict_surge(zone_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    client = get_ml_client()
    zones = [
        {"zone_id": z.get("zone_id", "zone_default")}
        for z in zone_features
    ] if zone_features else [{"zone_id": "zone_default"}]
    resp = await client.post("/predict/surge", json={"zones": zones})
    resp.raise_for_status()
    return resp.json().get("zones", [])


# ── ETA ───────────────────────────────────────────────────────────────────────
# ml_server: POST /predict/eta  {"distance_km": N, "hour_of_day": N}

async def predict_eta(distance_km: float, hour_of_day: int = 12) -> Dict[str, Any]:
    client = get_ml_client()
    resp = await client.post(
        "/predict/eta",
        json={"distance_km": distance_km, "hour_of_day": hour_of_day},
    )
    resp.raise_for_status()
    return resp.json()