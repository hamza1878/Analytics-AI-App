"""
app/api/routes/models.py

Endpoint /models — expose les métriques MLflow de tous les modèles en temps réel.
"""
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config import ML

router = APIRouter()

# ── Seuils de promotion (même que mlflow_tracking.py) ────────────────────────
PROMOTION_THRESHOLDS = {
    "demand_forecast":  {"ensemble_r2": ("gt", 0.92)},
    "surge_predictor":  {"r2": ("gt", 0.90), "mae": ("lt", 0.30)},
    "churn_classifier": {"cv_auc_roc": ("gt", 0.88)},
    "eta_estimator":    {"mae_minutes": ("lt", 3.0)},
    "fraud_detector":   {"anomaly_rate": ("lt", 0.02)},
    "route_optimizer":  {"dispatch_accuracy": ("gt", 0.85)},
}

CACHE_TTL = 60  # secondes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_health(model_name: str, metrics: dict) -> str:
    """Retourne 'healthy' | 'warning' | 'critical' selon les seuils."""
    thresholds = PROMOTION_THRESHOLDS.get(model_name, {})
    if not thresholds:
        return "unknown"
    passed = 0
    for metric, (op, threshold) in thresholds.items():
        val = metrics.get(metric)
        if val is None:
            continue
        if op == "gt" and val > threshold:
            passed += 1
        elif op == "lt" and val < threshold:
            passed += 1
    ratio = passed / len(thresholds)
    if ratio == 1.0:
        return "healthy"
    elif ratio >= 0.5:
        return "warning"
    return "critical"


def _fetch_models_from_mlflow() -> list[dict]:
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow non disponible")

    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    model_names = list(PROMOTION_THRESHOLDS.keys())
    results = []

    for model_name in model_names:
        try:
            runs = mlflow.search_runs(
                experiment_names=[ML.mlflow_experiment],
                filter_string=f"tags.mlflow.runName LIKE '{model_name}%'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs.empty:
                results.append({
                    "name":    model_name,
                    "status":  "no_run",
                    "health":  "unknown",
                    "metrics": {},
                    "run_id":  None,
                    "started": None,
                })
                continue

            run      = runs.iloc[0]
            raw_metrics = {
                c.replace("metrics.", ""): round(float(run[c]), 6)
                for c in run.index
                if c.startswith("metrics.") and not str(run[c]) in ("nan", "None")
            }
            # Filtre les feat_imp_ pour alléger la réponse
            core_metrics = {
                k: v for k, v in raw_metrics.items()
                if not k.startswith("feat_imp_")
            }
            feat_importance = {
                k.replace("feat_imp_", ""): v
                for k, v in raw_metrics.items()
                if k.startswith("feat_imp_")
            }

            results.append({
                "name":             model_name,
                "run_id":           run["run_id"][:8],
                "status":           run["status"],
                "health":           _check_health(model_name, core_metrics),
                "started":          str(run.get("start_time", "")),
                "metrics":          core_metrics,
                "feat_importance":  feat_importance,
                "thresholds":       PROMOTION_THRESHOLDS.get(model_name, {}),
            })

        except Exception as e:
            results.append({
                "name":   model_name,
                "status": "error",
                "health": "critical",
                "error":  str(e),
            })

    return results


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/")
async def list_models(request: Request):
    """
    Retourne le statut et les métriques de tous les modèles ML.
    Résultat mis en cache Redis 60 secondes.
    """
    redis = getattr(request.app.state, "redis", None)
    cache_key = "ml:models:all"

    # Lecture cache
    if redis:
        cached = await redis.get(cache_key)
        if cached:
            return {"cached": True, "data": json.loads(cached)}

    try:
        models = _fetch_models_from_mlflow()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow indisponible : {e}")

    summary = {
        "healthy":  sum(1 for m in models if m.get("health") == "healthy"),
        "warning":  sum(1 for m in models if m.get("health") == "warning"),
        "critical": sum(1 for m in models if m.get("health") == "critical"),
        "unknown":  sum(1 for m in models if m.get("health") == "unknown"),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment":   ML.mlflow_experiment,
        "summary":      summary,
        "models":       models,
    }

    # Écriture cache
    if redis:
        await redis.setex(cache_key, CACHE_TTL, json.dumps(payload, default=str))

    return {"cached": False, "data": payload}


@router.get("/{model_name}")
async def get_model(model_name: str, request: Request):
    """Retourne les détails d'un modèle spécifique."""
    if model_name not in PROMOTION_THRESHOLDS:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model_name}' inconnu. "
                   f"Disponibles : {list(PROMOTION_THRESHOLDS.keys())}",
        )

    redis = getattr(request.app.state, "redis", None)
    cache_key = f"ml:models:{model_name}"

    if redis:
        cached = await redis.get(cache_key)
        if cached:
            return {"cached": True, "data": json.loads(cached)}

    try:
        models = _fetch_models_from_mlflow()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow indisponible : {e}")

    model = next((m for m in models if m["name"] == model_name), None)
    if not model:
        raise HTTPException(status_code=404, detail=f"Aucun run pour {model_name}")

    if redis:
        await redis.setex(cache_key, CACHE_TTL, json.dumps(model, default=str))

    return {"cached": False, "data": model}


@router.get("/{model_name}/promote")
async def check_promotion(model_name: str):
    """Vérifie si un modèle est éligible à la promotion en production."""
    if model_name not in PROMOTION_THRESHOLDS:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name}' inconnu")

    try:
        models = _fetch_models_from_mlflow()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    model = next((m for m in models if m["name"] == model_name), None)
    if not model or not model.get("metrics"):
        raise HTTPException(status_code=404, detail="Aucune métrique disponible")

    metrics    = model["metrics"]
    thresholds = PROMOTION_THRESHOLDS[model_name]
    details    = []
    all_passed = True

    for metric, (op, threshold) in thresholds.items():
        actual = metrics.get(metric)
        passed = False
        if actual is not None:
            passed = (actual < threshold) if op == "lt" else (actual > threshold)
        details.append({
            "metric":    metric,
            "op":        op,
            "threshold": threshold,
            "actual":    actual,
            "passed":    passed,
        })
        if not passed:
            all_passed = False

    return {
        "model":    model_name,
        "eligible": all_passed,
        "health":   model.get("health"),
        "details":  details,
    }