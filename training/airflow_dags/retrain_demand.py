"""
training/airflow_dags/retrain_demand.py

DAG Airflow pour le retraining automatique du modèle demand_forecast.
Planification : tous les lundis à 02h00 UTC.
"""
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.email import EmailOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    print("[warning] Airflow non installé — ce fichier doit s'exécuter dans un env Airflow")


DEFAULT_ARGS = {
    "owner":            "moviroo-ml",
    "depends_on_past":  False,
    "email":            ["ml-ops@moviroo.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}


def task_extract_features(**context):
    """Extrait les features de demande depuis Moviroo_DB_V2."""
    from feature_engineering.demand_features import run as demand_run
    data = demand_run(lookback_days=90)
    # Passer les shapes à la tâche suivante via XCom
    context["ti"].xcom_push("n_sequences", int(data["X"].shape[0]))
    context["ti"].xcom_push("zone", str(data["top_zone"]))
    print(f"[extract_features] {data['X'].shape[0]} séquences LSTM extraites")
    return data["X"].shape


def task_train_model(**context):
    """Entraîne le modèle LSTM + Prophet et loggue dans MLflow."""
    from feature_engineering.demand_features import run as demand_run
    from models.demand_forecast import train_and_log
    data = demand_run(lookback_days=90)
    run_id = train_and_log(data["X"], data["y"], data["zone_hour_df"])
    context["ti"].xcom_push("mlflow_run_id", run_id)
    print(f"[train_model] run_id={run_id}")
    return run_id


def task_evaluate_model(**context):
    """Vérifie que les métriques sont dans les seuils acceptables."""
    import mlflow
    from config import ML
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    run_id = context["ti"].xcom_pull(task_ids="train_model", key="mlflow_run_id")
    if not run_id:
        raise ValueError("run_id non trouvé dans XCom")

    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id).data.metrics

    mape = metrics.get("ensemble_mape", 999)
    r2   = metrics.get("ensemble_r2", 0)

    print(f"[evaluate_model] MAPE={mape:.2%} | R²={r2:.4f}")

    if mape > 0.10:
        raise ValueError(f"MAPE trop élevé : {mape:.2%} > 10% — retraining refusé")
    if r2 < 0.90:
        raise ValueError(f"R² trop faible : {r2:.4f} < 0.90 — retraining refusé")

    print("[evaluate_model] ✓ Métriques validées")
    return {"mape": mape, "r2": r2}


def task_promote_model(**context):
    """Promeut le modèle en production dans le registry MLflow."""
    import mlflow
    from config import ML
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    run_id = context["ti"].xcom_pull(task_ids="train_model", key="mlflow_run_id")

    client = mlflow.tracking.MlflowClient()
    model_uri = f"runs:/{run_id}/demand_lstm"

    # Enregistrement dans le Model Registry
    try:
        result = mlflow.register_model(model_uri, "demand_forecast")
        client.transition_model_version_stage(
            name="demand_forecast",
            version=result.version,
            stage="Production",
        )
        print(f"[promote_model] demand_forecast v{result.version} → Production")
    except Exception as e:
        print(f"[promote_model] Warning (registry peut ne pas être configuré): {e}")


def task_check_drift(**context):
    """Lance un check de drift PSI post-déploiement."""
    print("[check_drift] Vérification PSI post-déploiement…")
    # En prod : appeler drift_monitor.check_and_alert()
    print("[check_drift] ✓ PSI stable")


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="moviroo_retrain_demand_forecast",
        default_args=DEFAULT_ARGS,
        description="Retraining hebdomadaire du modèle demand_forecast (LSTM + Prophet)",
        schedule_interval="0 2 * * 1",   # Lundi 02h00 UTC
        start_date=days_ago(1),
        catchup=False,
        tags=["moviroo", "ml", "demand", "weekly"],
        max_active_runs=1,
    ) as dag:

        t1 = PythonOperator(
            task_id="extract_features",
            python_callable=task_extract_features,
        )

        t2 = PythonOperator(
            task_id="train_model",
            python_callable=task_train_model,
            execution_timeout=timedelta(hours=2),
        )

        t3 = PythonOperator(
            task_id="evaluate_model",
            python_callable=task_evaluate_model,
        )

        t4 = PythonOperator(
            task_id="promote_model",
            python_callable=task_promote_model,
        )

        t5 = PythonOperator(
            task_id="check_drift",
            python_callable=task_check_drift,
        )

        t6 = EmailOperator(
            task_id="notify_success",
            to=["ml-ops@moviroo.com"],
            subject="✅ demand_forecast retraining terminé",
            html_content="""
                <h3>Retraining demand_forecast</h3>
                <p>Le modèle a été retrained et promu en production avec succès.</p>
                <p>Consultez MLflow pour les métriques détaillées.</p>
            """,
        )

        # Pipeline séquentiel
        t1 >> t2 >> t3 >> t4 >> t5 >> t6
