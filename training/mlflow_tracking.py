"""
training/mlflow_tracking.py

Utilitaires MLflow : gestion des expériences, comparaison de runs,
promotion de modèles, et export de rapports.
"""
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
from config import ML

try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[warning] MLflow non installé : pip install mlflow")


# ─────────────────────────────────────────────
# Client helper
# ─────────────────────────────────────────────

def get_client():
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow non disponible")
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    return mlflow.tracking.MlflowClient()


def get_or_create_experiment(name: str = ML.mlflow_experiment) -> str:
    """Retourne l'experiment_id, le crée s'il n'existe pas."""
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        print(f"[mlflow] Expérience créée : {name} (id={exp_id})")
        return exp_id
    return exp.experiment_id


# ─────────────────────────────────────────────
# Comparaison de runs
# ─────────────────────────────────────────────

def compare_runs(
    run_ids: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """
    Compare plusieurs runs MLflow sur un ensemble de métriques.

    Returns:
        DataFrame avec run_id, run_name, start_time, status + métriques
    """
    client = get_client()
    rows = []
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            row = {
                "run_id":    run_id[:8],
                "run_name":  run.data.tags.get("mlflow.runName", ""),
                "status":    run.info.status,
                "started":   datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
            }
            for m in metrics:
                row[m] = run.data.metrics.get(m)
            rows.append(row)
        except Exception as e:
            print(f"  Run {run_id[:8]} introuvable : {e}")

    return pd.DataFrame(rows)


def get_best_run(
    experiment_name: str,
    metric: str,
    mode: str = "min",   # "min" ou "max"
) -> Optional[str]:
    """Retourne le run_id du meilleur run selon une métrique."""
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"metrics.{metric} > 0",
        order_by=[f"metrics.{metric} {'ASC' if mode == 'min' else 'DESC'}"],
        max_results=1,
    )
    if runs.empty:
        return None
    return runs.iloc[0]["run_id"]


# ─────────────────────────────────────────────
# Promotion de modèles
# ─────────────────────────────────────────────

PROMOTION_THRESHOLDS = {
    "demand_forecast": {"ensemble_mape": ("lt", 0.08), "ensemble_r2": ("gt", 0.92)},
    "surge_predictor": {"r2": ("gt", 0.90), "mae": ("lt", 0.30)},
    "churn_classifier": {"cv_accuracy": ("gt", 0.83), "cv_auc_roc": ("gt", 0.88)},
    "eta_estimator":   {"mae_minutes": ("lt", 3.0)},
    "fraud_detector":  {"anomaly_rate": ("lt", 0.02)},
    "route_optimizer": {"dispatch_accuracy": ("gt", 0.85)},
}


def check_promotion_criteria(run_id: str, model_name: str) -> dict:
    """
    Vérifie si un run respecte les seuils de promotion en production.

    Returns:
        {"eligible": bool, "details": [{metric, threshold, actual, passed}]}
    """
    client = get_client()
    run    = client.get_run(run_id)
    thresholds = PROMOTION_THRESHOLDS.get(model_name, {})

    details = []
    all_passed = True

    for metric, (op, threshold) in thresholds.items():
        actual = run.data.metrics.get(metric)
        if actual is None:
            passed = False
        elif op == "lt":
            passed = actual < threshold
        elif op == "gt":
            passed = actual > threshold
        else:
            passed = False

        details.append({
            "metric":    metric,
            "op":        op,
            "threshold": threshold,
            "actual":    actual,
            "passed":    passed,
        })
        if not passed:
            all_passed = False

    return {"eligible": all_passed, "run_id": run_id, "model": model_name, "details": details}


def promote_to_production(model_name: str, run_id: str) -> dict:
    """
    Enregistre un modèle dans le registry MLflow et le promeut en Production.
    Vérifie les critères avant promotion.
    """
    criteria = check_promotion_criteria(run_id, model_name)
    if not criteria["eligible"]:
        failed = [d for d in criteria["details"] if not d["passed"]]
        raise ValueError(
            f"Promotion refusée pour {model_name} : "
            f"{len(failed)} critère(s) non satisfait(s) : "
            f"{[d['metric'] for d in failed]}"
        )

    client = get_client()
    model_uri = f"runs:/{run_id}/{model_name}"

    try:
        result = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"[mlflow] ✓ {model_name} v{result.version} → Production")
        return {
            "model":   model_name,
            "version": result.version,
            "stage":   "Production",
            "run_id":  run_id,
        }
    except Exception as e:
        print(f"[mlflow] Erreur registry : {e}")
        return {"model": model_name, "run_id": run_id, "error": str(e)}


# ─────────────────────────────────────────────
# Rapport de retraining
# ─────────────────────────────────────────────

def generate_retraining_report(experiment_name: str = ML.mlflow_experiment) -> dict:
    """
    Génère un rapport consolidé du dernier cycle de retraining.
    """
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)

    model_names = list(PROMOTION_THRESHOLDS.keys())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment":   experiment_name,
        "models":       [],
    }

    for model_name in model_names:
        try:
            runs = mlflow.search_runs(
                experiment_names=[experiment_name],
                filter_string=f"tags.mlflow.runName LIKE '{model_name}%'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs.empty:
                report["models"].append({"name": model_name, "status": "no_run"})
                continue

            run = runs.iloc[0]
            metric_cols = [c for c in run.index if c.startswith("metrics.")]
            metrics = {c.replace("metrics.", ""): run[c] for c in metric_cols if pd.notna(run[c])}

            report["models"].append({
                "name":     model_name,
                "run_id":   run["run_id"][:8],
                "status":   run["status"],
                "started":  str(run.get("start_time", "")),
                "metrics":  metrics,
            })
        except Exception as e:
            report["models"].append({"name": model_name, "status": "error", "error": str(e)})

    return report


# ─────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if MLFLOW_AVAILABLE:
        exp_id = get_or_create_experiment()
        print(f"Experiment ID : {exp_id}")

        report = generate_retraining_report()
        print(json.dumps(report, indent=2, default=str))
    else:
        print("MLflow non disponible. Installer avec : pip install mlflow")
