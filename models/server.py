"""
api/server.py
Moviroo — FastAPI Backend ML
Port 8001 — endpoints pour le frontend

Démarrage : python -m api.server
"""
import os, json, logging, threading
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("api.server")

app = FastAPI(title="Moviroo ML API", version="1.0.0")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATUS_FILE   = "data/pipeline_status.json"
_pipeline_lock = threading.Lock()
_is_running    = False

def _get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {"status": "never_run", "steps": []}

def _run_pipeline_background():
    global _is_running
    with _pipeline_lock:
        _is_running = True
    try:
        from daily_runner import run_all_pipelines
        run_all_pipelines()
    finally:
        with _pipeline_lock:
            _is_running = False

# ── GET /status ────────────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    """Retourne le statut de la dernière exécution du pipeline."""
    status = _get_status()
    status["is_running"] = _is_running
    return status

# ── POST /restart ──────────────────────────────────────────────────────────────
@app.post("/restart")
def restart_pipeline(background_tasks: BackgroundTasks):
    """Déclenche une exécution complète du pipeline (bouton Redémarrer)."""
    global _is_running
    if _is_running:
        raise HTTPException(status_code=409, detail="Pipeline déjà en cours d'exécution")
    log.info("🔄 Restart demandé via API")
    background_tasks.add_task(_run_pipeline_background)
    return {
        "message": "Pipeline démarré",
        "started_at": datetime.utcnow().isoformat(),
    }

# ── GET /results ───────────────────────────────────────────────────────────────
@app.get("/results")
def get_all_results():
    """Retourne les métriques de tous les modèles entraînés."""
    results = {}
    model_files = [
        ("surge_predictor",   "data/models/surge_predictor_results.json"),
        ("demand_forecast",   "data/models/demand_forecast_results.json"),
        ("churn_classifier",  "data/models/churn_classifier_results.json"),
        ("eta_estimator",     "data/models/eta_estimator_results.json"),
        ("fraud_detector",    "data/models/fraud_detector_results.json"),
    ]
    for name, path in model_files:
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            results[name] = {"status": "not_trained"}
    return results

# ── GET /health ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5011, reload=False)