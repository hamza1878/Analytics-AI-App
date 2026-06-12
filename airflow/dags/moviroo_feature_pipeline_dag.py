"""
moviroo_feature_pipeline_dag.py
Airflow DAG: PostgreSQL → Feature Engineering → Feature Store (Redis/Feast)
Schedule: every hour
"""
from airflow import DAG
from airflow.operators.python import PythonOperator # type: ignore
from airflow.providers.postgres.hooks.postgres import PostgresHook # type: ignore
from datetime import datetime, timedelta
import pandas as pd # type: ignore
import redis # type: ignore
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "moviroo-data",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["data-alerts@moviroo.com"],
}

dag = DAG(
    dag_id="moviroo_feature_pipeline",
    default_args=DEFAULT_ARGS,
    description="Batch feature engineering: PostgreSQL → Feature Store",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["moviroo", "features", "ml"],
)

REDIS_HOST = "redis"
REDIS_PORT = 6379
POSTGRES_CONN_ID = "moviroo_postgres"


# ── Task 1: Extract demand features ───────────────────────────────────────────

def extract_demand_features(**kwargs):
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    df = hook.get_pandas_df("""
        SELECT
            DATE_TRUNC('hour', created_at) AS hour_bucket,
            COUNT(*)                        AS ride_count
        FROM rides
        WHERE created_at >= NOW() - INTERVAL '7 days'
          AND status = 'completed'
        GROUP BY 1
        ORDER BY 1
    """)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=1)
    r.setex("features:demand:hourly", 7200, df.to_json(orient="records"))
    logger.info(f"Demand features stored: {len(df)} rows")


# ── Task 2: Extract driver churn features ─────────────────────────────────────

def extract_churn_features(**kwargs):
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    df = hook.get_pandas_df("""
        SELECT
            d.id,
            d.rating,
            d.total_rides,
            d.status,
            COUNT(r.id)  FILTER (WHERE r.created_at >= NOW() - '30 days'::interval) AS rides_last_30d,
            COALESCE(SUM(r.price) FILTER (WHERE r.created_at >= NOW() - '30 days'::interval), 0) AS revenue_last_30d,
            EXTRACT(EPOCH FROM (NOW() - MAX(r.created_at)))/86400 AS days_since_last_ride,
            COUNT(r.id) FILTER (WHERE r.status = 'cancelled') * 1.0
                / NULLIF(COUNT(r.id), 0) AS cancellation_rate
        FROM drivers d
        LEFT JOIN rides r ON r.driver_id = d.id
        GROUP BY d.id, d.rating, d.total_rides, d.status
    """)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=1)
    r.setex("features:churn:drivers", 3600, df.to_json(orient="records"))
    logger.info(f"Churn features stored: {len(df)} drivers")


# ── Task 3: Compute PSI drift detection ───────────────────────────────────────

def compute_psi_drift(**kwargs):
    """
    Population Stability Index — triggers retraining if PSI > 0.2.
    """
    import numpy as np

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=1)
    raw = r.get("features:demand:hourly")
    if not raw:
        logger.warning("No demand features for PSI calculation")
        return

    df = pd.read_json(raw)
    baseline_mean = df["ride_count"].mean()
    baseline_std = df["ride_count"].std()

    # Simulate current distribution check
    current_vals = df["ride_count"].tail(24)
    buckets = 10
    expected_pct = np.ones(buckets) / buckets
    observed_pct, _ = np.histogram(current_vals, bins=buckets, density=True)
    observed_pct = observed_pct / observed_pct.sum()

    psi = np.sum((observed_pct - expected_pct) * np.log(observed_pct / (expected_pct + 1e-8)))

    logger.info(f"PSI = {psi:.4f}")
    r.set("monitoring:psi:demand", str(psi))

    if psi > 0.2:
        logger.warning(f"PSI {psi:.4f} > 0.2 — triggering retraining signal")
        r.set("trigger:retrain:demand", "1")
        # In production: trigger Airflow retraining DAG or Kubeflow pipeline


# ── Task 4: Trigger retraining if needed ──────────────────────────────────────

def conditional_retrain(**kwargs):
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=1)
    if r.get("trigger:retrain:demand") == b"1":
        logger.info("Retraining triggered — submitting ML job")
        # In production: trigger Kubernetes Job or Airflow TriggerDagRunOperator
        r.delete("trigger:retrain:demand")
    else:
        logger.info("No retraining needed")


# ── Wire up DAG ────────────────────────────────────────────────────────────────

t1 = PythonOperator(task_id="extract_demand_features",  python_callable=extract_demand_features, dag=dag)
t2 = PythonOperator(task_id="extract_churn_features",   python_callable=extract_churn_features,  dag=dag)
t3 = PythonOperator(task_id="compute_psi_drift",        python_callable=compute_psi_drift,       dag=dag)
t4 = PythonOperator(task_id="conditional_retrain",      python_callable=conditional_retrain,     dag=dag)

[t1, t2] >> t3 >> t4
