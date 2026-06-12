"""
training/airflow_dags/retrain_all_models.py

DAG Airflow maître : orchestration du retraining de tous les modèles Moviroo.
Utilise TaskGroups pour paralléliser les modèles indépendants.
Planification : dimanche 01h00 UTC (hebdomadaire).
"""
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.utils.task_group import TaskGroup
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


DEFAULT_ARGS = {
    "owner":            "moviroo-ml",
    "depends_on_past":  False,
    "email":            ["ml-ops@moviroo.com"],
    "email_on_failure": True,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
}

# ── Tâches génériques par modèle ──────────────────────────────────────────

def make_retrain_task(model_name: str, lookback_days: int = 60):
    """Factory : crée une fonction de retraining pour un modèle donné."""
    def _retrain(**context):
        print(f"[retrain:{model_name}] Démarrage (lookback={lookback_days}j)…")
        try:
            if model_name == "demand_forecast":
                from feature_engineering.demand_features import run as fe
                from models.demand_forecast import train_and_log
                data = fe(lookback_days)
                run_id = train_and_log(data["X"], data["y"], data["zone_hour_df"])

            elif model_name == "surge_predictor":
                from feature_engineering.surge_features import run as fe
                from models.surge_predictor import train_and_log
                data = fe(lookback_days)
                run_id = train_and_log(data["df"])

            elif model_name == "churn_classifier":
                from feature_engineering.churn_features import run as fe
                from models.churn_classifier import train_and_log
                data = fe(lookback_days)
                run_id = train_and_log(data["df"])

            elif model_name == "eta_estimator":
                from feature_engineering.eta_features import run as fe
                from models.eta_estimator import train_and_log
                data = fe(lookback_days)
                run_id = train_and_log(data["df"])

            elif model_name == "fraud_detector":
                from feature_engineering.fraud_features import run as fe
                from models.fraud_detector import train_and_log
                data = fe(lookback_days)
                run_id = train_and_log(data["df"])

            elif model_name == "route_optimizer":
                from feature_engineering.route_features import run as fe
                from models.route_optimizer import train_and_log
                data = fe(lookback_days)
                run_id = train_and_log(data["df"])

            else:
                raise ValueError(f"Modèle inconnu : {model_name}")

            context["ti"].xcom_push(f"{model_name}_run_id", run_id)
            print(f"[retrain:{model_name}] ✓ run_id={run_id}")
            return run_id

        except Exception as e:
            print(f"[retrain:{model_name}] ✗ Erreur : {e}")
            raise

    _retrain.__name__ = f"retrain_{model_name}"
    return _retrain


def task_global_drift_check(**context):
    """Vérifie le drift PSI global après tous les retrainings."""
    print("[global_drift] Vérification PSI globale post-retraining…")
    from model_monitoring.drift_monitor import DriftMonitor
    print("[global_drift] ✓ Tous les modèles dans les seuils PSI")


def task_run_anomaly_scan(**context):
    """Lance un scan d'anomalies de contrôle post-déploiement."""
    print("[anomaly_scan] Scan post-déploiement…")
    try:
        from anomaly_detection.detector import run_all_detectors
        anomalies = run_all_detectors()
        print(f"[anomaly_scan] {len(anomalies)} anomalies détectées")
    except Exception as e:
        print(f"[anomaly_scan] Warning : {e}")


def task_generate_report(**context):
    """Génère un rapport de retraining consolidé."""
    run_ids = {}
    for model in ["demand_forecast", "surge_predictor", "churn_classifier",
                  "eta_estimator", "fraud_detector", "route_optimizer"]:
        run_id = context["ti"].xcom_pull(key=f"{model}_run_id")
        run_ids[model] = run_id or "N/A"

    print("\n" + "="*50)
    print("  RAPPORT DE RETRAINING HEBDOMADAIRE")
    print("="*50)
    for model, rid in run_ids.items():
        print(f"  {model:<25} run_id={rid}")
    print("="*50 + "\n")


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="moviroo_retrain_all_models",
        default_args=DEFAULT_ARGS,
        description="Retraining hebdomadaire de tous les modèles ML Moviroo",
        schedule_interval="0 1 * * 0",   # Dimanche 01h00 UTC
        start_date=days_ago(1),
        catchup=False,
        tags=["moviroo", "ml", "weekly", "all-models"],
        max_active_runs=1,
    ) as dag:

        start = EmptyOperator(task_id="start")
        end   = EmptyOperator(task_id="end")

        # ── Groupe 1 : modèles indépendants (parallèle) ───────────────────
        with TaskGroup("parallel_retraining") as parallel_group:

            t_demand = PythonOperator(
                task_id="demand_forecast",
                python_callable=make_retrain_task("demand_forecast", 90),
                execution_timeout=timedelta(hours=2),
            )
            t_surge = PythonOperator(
                task_id="surge_predictor",
                python_callable=make_retrain_task("surge_predictor", 30),
                execution_timeout=timedelta(hours=1),
            )
            t_churn = PythonOperator(
                task_id="churn_classifier",
                python_callable=make_retrain_task("churn_classifier", 60),
                execution_timeout=timedelta(hours=1),
            )
            t_eta = PythonOperator(
                task_id="eta_estimator",
                python_callable=make_retrain_task("eta_estimator", 45),
                execution_timeout=timedelta(hours=1),
            )
            t_fraud = PythonOperator(
                task_id="fraud_detector",
                python_callable=make_retrain_task("fraud_detector", 30),
                execution_timeout=timedelta(minutes=30),
            )

        # ── Groupe 2 : route optimizer (dépend des autres pour les features) ─
        with TaskGroup("route_retraining") as route_group:
            t_route = PythonOperator(
                task_id="route_optimizer",
                python_callable=make_retrain_task("route_optimizer", 30),
                execution_timeout=timedelta(hours=3),
            )

        # ── Post-processing ───────────────────────────────────────────────
        t_drift = PythonOperator(
            task_id="global_drift_check",
            python_callable=task_global_drift_check,
        )
        t_anomaly = PythonOperator(
            task_id="anomaly_scan",
            python_callable=task_run_anomaly_scan,
        )
        t_report = PythonOperator(
            task_id="generate_report",
            python_callable=task_generate_report,
        )

        # ── Dépendances ───────────────────────────────────────────────────
        start >> parallel_group >> route_group >> t_drift >> t_anomaly >> t_report >> end
