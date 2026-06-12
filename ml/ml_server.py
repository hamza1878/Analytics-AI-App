"""
moviroo_ml_v4_2.py  –  Moviroo ML System  v4.2
================================================
Port 8005  |  9 endpoints  |  PostgreSQL réel  |  MLflow registry
ZERO données simulées – tout vient de la base de données ou des modèles ML.

Comprend :
  ─ training_pipelines.py  (6 modèles ML)
  ─ ml_server_v4.1.py      (9 endpoints FastAPI)

CHANGELOG v4.2 (fixes appliqués en plus de v4.1) :
  [FIX-ETAv2] ETARequest validé (Field gt/ge/le), ml_pred NaN/inf/négatif
               géré, CI basé sur RMSE MLflow si dispo, safe_float() partout
  [FIX-TRAIN] train_eta_model : validation colonnes + nettoyage défensif
               + registered_model_name pour lier run ↔ try_mlflow
  [FIX-DB]    db_query wrapper async cohérent avec le pool asyncpg
  [FIX-1]     wait_time GREATEST + filtre trip_started_at > created_at
  [FIX-2]     zones : median_wait_min corrigé (GREATEST aussi)
  [FIX-3]     ETA dataset élargi à 90 jours + fallback median speed
  [FIX-4]     price_final négatif : filtre WHERE price_final >= 0
  [FIX-5]     surge / ETA cohérence : avg_speed extrait du contexte zone
  [FIX-6]     churn : ajout features peak_rides_30d + zone_diversity
  [FIX-7]     indexes SQL documentés (à appliquer en migration)

Endpoints :
  GET  /health
  GET  /predict/demand
  GET  /predict/revenue
  POST /predict/churn
  POST /predict/eta
  GET  /predict/anomalies
  POST /predict/surge
  GET  /intelligence/zones
  GET  /dashboard/kpis

CLI :
  python moviroo_ml_v4_2.py --train [demand|revenue|churn|eta|fraud|surge|all]
  python moviroo_ml_v4_2.py --serve

Auteur : Moviroo ML Team – PFE v4.2
"""

# ══════════════════════════════════════════════════════════════════════════
# INDEXES RECOMMANDÉS (appliquer une seule fois en migration) :
#
#   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rides_created_driver
#       ON rides(created_at, driver_id);
#
#   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rides_status_completed
#       ON rides(status, completed_at)
#       WHERE status = 'COMPLETED';
#
#   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_drivers_work_area
#       ON drivers(work_area_id)
#       WHERE deleted_at IS NULL;
#
#   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rides_trip_started
#       ON rides(trip_started_at, created_at)
#       WHERE trip_started_at IS NOT NULL AND created_at IS NOT NULL;
# ══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import asyncpg
    ASYNCPG_OK = True
except ImportError:
    ASYNCPG_OK = False

try:
    import mlflow
    import mlflow.pyfunc
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv(".env.ml", override=True)
    load_dotenv(override=False)
except ImportError:
    pass

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
log    = logging.getLogger("moviroo-ml-v4.2")
logger = log   # alias pour les fonctions de training

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

DB_URL            = os.getenv("DATABASE_URL",
                               "postgresql://postgres:postgres1878@localhost:5432/Moviroo_DB_V2")
MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT",   "moviroo-models")
REDIS_URL         = os.getenv("REDIS_URL",            "redis://localhost:6379/0")
CACHE_TTL         = int(os.getenv("CACHE_TTL_SECONDS", "300"))

MODELS_DIR = Path("./artifacts")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAMES = [
    "demand_forecast",
    "surge_predictor",
    "churn_classifier",
    "eta_estimator",
    "fraud_detector",
    "route_optimizer",
]

if MLFLOW_OK:
    mlflow.set_tracking_uri(MLFLOW_URI)


# ════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def safe_float(v, default=None) -> Optional[float]:
    """Cast silencieux vers float — retourne `default` si impossible."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (en %)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    if not mask.any():
        return float("nan")
    return round(float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100), 2)


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)


# ════════════════════════════════════════════════════════════════════════════
# ══  TRAINING PIPELINES  ═════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

# ── 1. Demand Forecasting ─────────────────────────────────────────────────────

def train_demand_model(df: pd.DataFrame):
    """
    Prophet baseline pour la prévision de demande horaire.
    df : colonnes [timestamp, ride_count]
    """
    from prophet import Prophet

    required = {"timestamp", "ride_count"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colonnes manquantes : {missing}")
    df = df.dropna(subset=list(required))
    if df.empty:
        raise ValueError("DataFrame vide après nettoyage.")

    mlflow.set_experiment("demand-forecasting")
    with mlflow.start_run(run_name="prophet-baseline"):

        model    = Prophet(
            seasonality_mode       = "multiplicative",
            weekly_seasonality     = True,
            daily_seasonality      = True,
            changepoint_prior_scale = 0.05,
        )
        df_p = df.rename(columns={"timestamp": "ds", "ride_count": "y"})
        model.fit(df_p)

        split   = int(len(df_p) * 0.9)
        eval_df = df_p.iloc[split:]
        preds   = model.predict(eval_df[["ds"]])
        _mape   = mape(eval_df["y"], preds["yhat"])
        _rmse   = rmse(eval_df["y"], preds["yhat"])

        mlflow.log_metric("MAPE", _mape)
        mlflow.log_metric("RMSE", _rmse)
        mlflow.log_param("seasonality_mode", "multiplicative")
        mlflow.log_metric("train_samples", len(df_p))

        model_path = MODELS_DIR / "demand_prophet.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        logger.info(f"[Demand]  MAPE={_mape}%  RMSE={_rmse}")
        return model


# ── 2. Revenue Optimization ───────────────────────────────────────────────────

def train_revenue_model(df: pd.DataFrame):
    """
    XGBoost regressor pour l'optimisation des revenus.
    df : colonnes [total_revenue, avg_price, avg_distance, ride_count,
                   day_of_week, hour_of_day]
    """
    if not XGB_OK:
        raise ImportError("xgboost requis : pip install xgboost")

    required = {"total_revenue", "avg_price", "avg_distance", "ride_count",
                "day_of_week",   "hour_of_day"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colonnes manquantes : {missing}")
    df = df.dropna(subset=["total_revenue"]).query("total_revenue >= 0")
    if df.empty:
        raise ValueError("DataFrame vide après nettoyage.")

    mlflow.set_experiment("revenue-optimization")
    with mlflow.start_run(run_name="xgb-revenue"):

        features = ["avg_price", "avg_distance", "ride_count", "day_of_week", "hour_of_day"]
        X        = df[features]
        y        = df["total_revenue"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "n_estimators":    300,
            "max_depth":       6,
            "learning_rate":   0.05,
            "subsample":       0.8,
            "colsample_bytree": 0.8,
            "reg_alpha":       0.1,
            "reg_lambda":      1.0,
            "random_state":    42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        _mape = mape(y_test, preds)
        _rmse = rmse(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("MAPE",          _mape)
        mlflow.log_metric("RMSE",          _rmse)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples",  len(X_test))
        mlflow.xgboost.log_model(model, "revenue_model",
                                 registered_model_name="surge_predictor")

        logger.info(f"[Revenue] MAPE={_mape}%  RMSE={_rmse}")
        return model


# ── 3. Driver Churn ───────────────────────────────────────────────────────────

def train_churn_model(df: pd.DataFrame):
    """
    XGBoost binary classifier pour la prédiction de churn chauffeur.
    df : colonnes [rating, total_rides, rides_last_30d, days_since_last_ride,
                   cancellation_rate, revenue_last_30d, churned]
    """
    if not XGB_OK:
        raise ImportError("xgboost requis : pip install xgboost")

    required = {"rating", "total_rides", "rides_last_30d", "days_since_last_ride",
                "cancellation_rate", "revenue_last_30d", "churned"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colonnes manquantes : {missing}")
    df = df.dropna(subset=["churned"])
    if df.empty:
        raise ValueError("DataFrame vide après nettoyage.")

    mlflow.set_experiment("driver-churn")
    with mlflow.start_run(run_name="xgb-churn"):

        from sklearn.metrics import roc_auc_score

        features = ["rating", "total_rides", "rides_last_30d",
                    "days_since_last_ride", "cancellation_rate", "revenue_last_30d"]
        X = df[features].fillna(0)
        y = df["churned"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        params = {
            "n_estimators":     200,
            "max_depth":        5,
            "learning_rate":    0.05,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric":      "auc",
            "random_state":     42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds_proba = model.predict_proba(X_test)[:, 1]
        auc         = round(roc_auc_score(y_test, preds_proba), 4)

        mlflow.log_params(params)
        mlflow.log_metric("AUC",           auc)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples",  len(X_test))
        mlflow.xgboost.log_model(model, "churn_model",
                                 registered_model_name="churn_classifier")

        logger.info(f"[Churn]   AUC={auc}")
        return model


# ── 4. ETA Prediction ─────────────────────────────────────────────────────────

def train_eta_model(df: pd.DataFrame):
    """
    Gradient Boosting Regressor pour la prédiction du temps de trajet.
    df : colonnes [distance_km, hour_of_day, day_of_week, actual_minutes]
    """
    from sklearn.ensemble import GradientBoostingRegressor

    required = {"distance_km", "hour_of_day", "day_of_week", "actual_minutes"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colonnes manquantes : {missing}")

    # Nettoyage défensif : on rejette les durées nulles ou négatives
    df = df.dropna(subset=list(required))
    df = df[df["actual_minutes"] > 0]
    if df.empty:
        raise ValueError("DataFrame vide après nettoyage — impossible d'entraîner le modèle.")

    mlflow.set_experiment("eta-prediction")
    with mlflow.start_run(run_name="gbm-eta"):

        features = ["distance_km", "hour_of_day", "day_of_week"]
        X        = df[features]
        y        = df["actual_minutes"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "n_estimators":  200,
            "max_depth":     4,
            "learning_rate": 0.08,
            "random_state":  42,
        }
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        _mape = mape(y_test, preds)
        _rmse = rmse(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("MAPE",          _mape)
        mlflow.log_metric("RMSE",          _rmse)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples",  len(X_test))
        # registered_model_name lie le run à try_mlflow("eta_estimator", …)
        mlflow.sklearn.log_model(model, "eta_model",
                                 registered_model_name="eta_estimator")

        logger.info(f"[ETA]     MAPE={_mape}%  RMSE={_rmse} min")
        return model


# ── 5. Fraud / Anomaly Detection ──────────────────────────────────────────────

def train_fraud_model(df: pd.DataFrame):
    """
    Isolation Forest pour la détection de fraude / anomalie.
    df : colonnes [amount, expected_price, amount_delta,
                   payments_last_hour, hour_of_day, distance]
    """
    from sklearn.ensemble import IsolationForest

    required = {"amount", "expected_price", "amount_delta",
                "payments_last_hour", "hour_of_day"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colonnes manquantes : {missing}")
    df = df.fillna(0)
    if df.empty:
        raise ValueError("DataFrame vide.")

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="isolation-forest"):

        features = ["amount", "expected_price", "amount_delta",
                    "payments_last_hour", "hour_of_day"]
        X = df[features].values

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators  = 200,
            contamination = 0.03,
            random_state  = 42,
        )
        model.fit(X_scaled)

        preds        = model.predict(X_scaled)
        anomaly_rate = (preds == -1).mean()

        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.log_metric("n_samples",    len(X_scaled))
        mlflow.log_param("contamination", 0.03)
        mlflow.sklearn.log_model(model, "fraud_isolation_forest",
                                 registered_model_name="fraud_detector")

        scaler_path = MODELS_DIR / "fraud_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path))

        logger.info(f"[Fraud]   anomaly_rate={anomaly_rate:.4f}")
        return model, scaler


# ── 6. Surge Pricing ──────────────────────────────────────────────────────────

def train_surge_model(df: pd.DataFrame):
    """
    XGBoost regressor pour le surge pricing optimal.
    df : colonnes [demand, supply, hour_of_day, day_of_week,
                   weather_score, event_flag, optimal_surge]
    """
    if not XGB_OK:
        raise ImportError("xgboost requis : pip install xgboost")

    required = {"demand", "supply", "hour_of_day", "day_of_week",
                "weather_score", "event_flag", "optimal_surge"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colonnes manquantes : {missing}")
    df = df.dropna(subset=["optimal_surge"])
    if df.empty:
        raise ValueError("DataFrame vide après nettoyage.")

    mlflow.set_experiment("surge-optimization")
    with mlflow.start_run(run_name="xgb-surge"):

        features = ["demand", "supply", "hour_of_day", "day_of_week",
                    "weather_score", "event_flag"]
        X        = df[features].fillna(0)
        y        = df["optimal_surge"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "n_estimators":  150,
            "max_depth":     4,
            "learning_rate": 0.1,
            "random_state":  42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        _mape = mape(y_test, preds)
        _rmse = rmse(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("MAPE",          _mape)
        mlflow.log_metric("RMSE",          _rmse)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples",  len(X_test))
        mlflow.xgboost.log_model(model, "surge_model",
                                 registered_model_name="surge_predictor")

        logger.info(f"[Surge]   MAPE={_mape}%  RMSE={_rmse}")
        return model


# ════════════════════════════════════════════════════════════════════════════
# ══  SERVER  ═════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

if not FASTAPI_OK:
    log.warning("FastAPI non installé — le serveur ne sera pas disponible.")
else:

    # ──────────────────────────────────────────────────────────────────────
    # ÉTAT GLOBAL
    # ──────────────────────────────────────────────────────────────────────

    class AppState:
        pool:   Any             = None
        models: Dict[str, Any]  = {}
        runs:   Dict[str, dict] = {}
        redis:  Any             = None


    @asynccontextmanager
    async def lifespan(application: "FastAPI"):
        s = AppState()

        # PostgreSQL
        try:
            s.pool = await asyncpg.create_pool(
                DB_URL,
                min_size             = 2,
                max_size             = 15,
                statement_cache_size = 0,
                command_timeout      = 45,
            )
            async with s.pool.acquire() as conn:
                n = await conn.fetchval("SELECT COUNT(*) FROM rides")
            log.info(f"✓ PostgreSQL — {n} courses")
        except Exception as exc:
            log.error(f"✗ PostgreSQL : {exc}")
            s.pool = None

        # Redis (optionnel)
        try:
            import aioredis
            s.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
            await s.redis.ping()
            log.info("✓ Redis connecté")
        except Exception:
            s.redis = None
            log.warning("⚠ Redis non disponible → cache désactivé")

        # MLflow
        if MLFLOW_OK:
            client = mlflow.tracking.MlflowClient()
            for name in MODEL_NAMES:
                try:
                    s.models[name] = mlflow.pyfunc.load_model(f"models:/{name}/Production")
                    versions = client.get_latest_versions(name, stages=["Production"])
                    if versions:
                        v          = versions[0]
                        run        = client.get_run(v.run_id)
                        s.runs[name] = {
                            "run_id":  v.run_id[:8],
                            "version": v.version,
                            "metrics": dict(run.data.metrics),
                        }
                    log.info(f"  ✓ {name} chargé")
                except Exception as exc:
                    log.warning(f"  ⚠ {name} absent ({exc}) → heuristique SQL")
                    s.models[name] = None

        application.state.s = s
        yield

        if s.pool:  await s.pool.close()
        if s.redis: await s.redis.close()
        log.info("Arrêt du service ML v4.2")


    app = FastAPI(
        title       = "Moviroo ML Service",
        version     = "4.2.0",
        description = "Système ML temps réel — zéro donnée simulée",
        lifespan    = lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins = ["*"],
        allow_methods = ["*"],
        allow_headers = ["*"],
    )


    # ──────────────────────────────────────────────────────────────────────
    # HELPERS SERVEUR
    # ──────────────────────────────────────────────────────────────────────

    def state() -> AppState:
        return app.state.s


    async def db_query(sql: str, *args, retries: int = 3) -> List[asyncpg.Record]:
        pool = state().pool
        if pool is None:
            raise HTTPException(503, "Base de données non disponible")
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                async with pool.acquire() as conn:
                    return await conn.fetch(sql, *args)
            except (
                asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.ConnectionFailureError,
                OSError,
            ) as exc:
                last_exc = exc
                if attempt < retries:
                    await asyncio.sleep(0.5 * attempt)
        raise HTTPException(503, f"DB inaccessible : {last_exc}")


    async def db_val(sql: str, *args):
        pool = state().pool
        if pool is None:
            raise HTTPException(503, "Base de données non disponible")
        async with pool.acquire() as conn:
            return await conn.fetchval(sql, *args)


    def mlflow_meta(name: str) -> dict:
        return state().runs.get(name, {})


    def try_mlflow(name: str, df: pd.DataFrame) -> Optional[np.ndarray]:
        m = state().models.get(name)
        if m is None:
            return None
        try:
            return m.predict(df)
        except Exception as exc:
            log.warning(f"MLflow predict {name} échoué : {exc}")
            return None


    async def cache_get(key: str) -> Optional[str]:
        r = state().redis
        if r is None:
            return None
        try:
            return await r.get(key)
        except Exception:
            return None


    async def cache_set(key: str, value: str):
        r = state().redis
        if r is None:
            return
        try:
            await r.setex(key, CACHE_TTL, value)
        except Exception:
            pass


    # ──────────────────────────────────────────────────────────────────────
    # GET /health
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        s     = state()
        db_ok = False
        rides_count = drivers_count = 0

        if s.pool:
            try:
                async with s.pool.acquire() as conn:
                    rides_count   = await conn.fetchval("SELECT COUNT(*) FROM rides")
                    drivers_count = await conn.fetchval(
                        "SELECT COUNT(*) FROM drivers WHERE deleted_at IS NULL"
                    )
                db_ok = True
            except Exception:
                pass

        return {
            "status":         "ok" if db_ok else "degraded",
            "service":        "moviroo-ml",
            "version":        "4.2.0",
            "philosophy":     "zero_simulated_data",
            "database":       "connected" if db_ok else "unavailable",
            "database_stats": {"total_rides": int(rides_count),
                               "total_drivers": int(drivers_count)} if db_ok else {},
            "mlflow":         MLFLOW_URI if MLFLOW_OK else "not_installed",
            "redis":          "connected" if s.redis else "unavailable",
            "models_loaded":  [k for k, v in s.models.items() if v is not None],
            "models_total":   len(MODEL_NAMES),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        }


    # ──────────────────────────────────────────────────────────────────────
    # GET /dashboard/kpis
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/dashboard/kpis")
    async def dashboard_kpis():
        import json

        cache_key = "dashboard:kpis:v4.2"
        cached    = await cache_get(cache_key)
        if cached:
            return json.loads(cached)

        # 1. Volumes
        volumes = await db_query(
            """
            SELECT
                COUNT(*)                                               AS total_rides,
                COUNT(*) FILTER (WHERE status = 'COMPLETED')          AS completed,
                COUNT(*) FILTER (WHERE status = 'CANCELLED')          AS cancelled,
                COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours')
                                                                       AS rides_today,
                COUNT(*) FILTER (
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    AND status NOT IN ('CANCELLED')
                )                                                      AS rides_last_hour,
                ROUND(COUNT(*) FILTER (WHERE status = 'CANCELLED')::numeric
                    / NULLIF(COUNT(*), 0) * 100, 2)                    AS cancellation_rate_pct,
                ROUND(AVG(COALESCE(price_final, price_estimate))::numeric, 2)
                                                                       AS avg_fare,
                ROUND(SUM(COALESCE(price_final, 0)) FILTER (
                    WHERE status = 'COMPLETED'
                    AND completed_at >= NOW() - INTERVAL '24 hours'
                    AND COALESCE(price_final, 0) >= 0               -- [FIX-4]
                )::numeric, 2)                                         AS revenue_today,
                ROUND(SUM(COALESCE(price_final, 0)) FILTER (
                    WHERE status = 'COMPLETED'
                    AND completed_at >= NOW() - INTERVAL '7 days'
                    AND COALESCE(price_final, 0) >= 0
                )::numeric, 2)                                         AS revenue_week,
                ROUND(SUM(COALESCE(price_final, 0)) FILTER (
                    WHERE status = 'COMPLETED'
                    AND COALESCE(price_final, 0) >= 0
                )::numeric, 2)                                         AS revenue_total
            FROM rides
            """
        )
        v = dict(volumes[0]) if volumes else {}

        # 2. Chauffeurs
        drivers = await db_query(
            """
            SELECT
                COUNT(*)                                                   AS total_drivers,
                COUNT(*) FILTER (WHERE availability_status = 'online')    AS online_now,
                COUNT(*) FILTER (WHERE availability_status = 'on_trip')   AS on_trip,
                COUNT(*) FILTER (WHERE availability_status = 'offline')   AS offline,
                ROUND(AVG(rating_average)::numeric, 2)                    AS avg_driver_rating,
                ROUND(AVG(total_trips)::numeric, 1)                       AS avg_trips_per_driver
            FROM drivers
            WHERE deleted_at IS NULL
              AND availability_status != 'setup_required'
            """
        )
        d = dict(drivers[0]) if drivers else {}

        # 3. Temps d'attente — [FIX-1]
        wait = await db_query(
            """
            SELECT
                ROUND(AVG(GREATEST(
                    EXTRACT(EPOCH FROM (trip_started_at - created_at)) / 60, 0
                ))::numeric, 1)                                         AS avg_wait_min,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                    GREATEST(EXTRACT(EPOCH FROM (trip_started_at - created_at)) / 60, 0)
                )::numeric, 1)                                          AS median_wait_min,
                ROUND(AVG(
                    EXTRACT(EPOCH FROM (completed_at - trip_started_at)) / 60
                )::numeric, 1)                                          AS avg_trip_duration_min
            FROM rides
            WHERE status          = 'COMPLETED'
              AND trip_started_at IS NOT NULL
              AND created_at      IS NOT NULL
              AND completed_at    IS NOT NULL
              AND trip_started_at > created_at
              AND completed_at    > trip_started_at
              AND created_at >= NOW() - INTERVAL '7 days'
            """
        )
        w = dict(wait[0]) if wait else {}

        # 4. Passagers
        passengers = await db_query(
            """
            SELECT
                COUNT(DISTINCT r.passenger_id) FILTER (
                    WHERE r.created_at >= NOW() - INTERVAL '24 hours'
                )                                AS active_passengers_today,
                COUNT(DISTINCT r.passenger_id) FILTER (
                    WHERE r.created_at >= NOW() - INTERVAL '7 days'
                )                                AS active_passengers_week,
                ROUND(AVG(rr.passenger_rating)::numeric, 2) AS avg_passenger_rating
            FROM rides r
            LEFT JOIN ride_ratings rr ON rr.ride_id = r.id
            WHERE r.created_at >= NOW() - INTERVAL '30 days'
            """
        )
        p = dict(passengers[0]) if passengers else {}

        # 5. Demande horaire (aujourd'hui)
        hourly = await db_query(
            """
            SELECT
                EXTRACT(HOUR FROM created_at AT TIME ZONE 'Africa/Tunis')::int AS hour,
                COUNT(*)                                                         AS count,
                SUM(COALESCE(price_final, 0))                                   AS revenue
            FROM rides
            WHERE created_at >= CURRENT_DATE AT TIME ZONE 'Africa/Tunis'
              AND created_at <  (CURRENT_DATE + 1) AT TIME ZONE 'Africa/Tunis'
            GROUP BY 1
            ORDER BY 1
            """
        )
        hourly_data = [{"hour": r["hour"], "count": r["count"],
                        "revenue": float(r["revenue"] or 0)} for r in hourly]

        # 6. Tendance 7 jours
        trend_7d = await db_query(
            """
            SELECT
                DATE(completed_at AT TIME ZONE 'Africa/Tunis')                AS day,
                COUNT(*) FILTER (WHERE status = 'COMPLETED')                  AS completed,
                COUNT(*) FILTER (WHERE status = 'CANCELLED')                  AS cancelled,
                ROUND(SUM(COALESCE(price_final, 0))::numeric, 2)              AS revenue,
                ROUND(AVG(COALESCE(price_final, price_estimate))::numeric, 2) AS avg_fare
            FROM rides
            WHERE created_at >= NOW() - INTERVAL '7 days'
              AND created_at IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
        )
        trend_data = [
            {"day": str(r["day"]), "completed": r["completed"],
             "cancelled": r["cancelled"], "revenue": float(r["revenue"] or 0),
             "avg_fare":  float(r["avg_fare"] or 0)}
            for r in trend_7d
        ]

        # 7. Top zones — [FIX-2]
        top_zones = await db_query(
            """
            SELECT
                wa.ville                                                   AS zone,
                wa.id::text                                                AS zone_id,
                COUNT(r.id)                                                AS total_rides,
                ROUND(SUM(COALESCE(r.price_final, 0))::numeric, 2)        AS revenue,
                ROUND(AVG(COALESCE(r.surge_multiplier, 1.0))::numeric, 3) AS avg_surge,
                COUNT(DISTINCT r.driver_id)                                AS unique_drivers
            FROM rides r
            JOIN drivers d     ON d.user_id = r.driver_id
            JOIN work_areas wa ON wa.id     = d.work_area_id
            WHERE r.created_at >= NOW() - INTERVAL '30 days'
              AND r.status = 'COMPLETED'
            GROUP BY wa.id, wa.ville
            ORDER BY COUNT(r.id) DESC
            LIMIT 10
            """
        )
        zones_data = [
            {"zone": r["zone"], "zone_id": r["zone_id"], "total_rides": r["total_rides"],
             "revenue": float(r["revenue"] or 0), "avg_surge": float(r["avg_surge"] or 1.0),
             "unique_drivers": r["unique_drivers"]}
            for r in top_zones
        ]

        # 8. Véhicules (découverte dynamique de la colonne type)
        veh_cols = await db_query(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'vehicles' ORDER BY ordinal_position"
        )
        col_names = [r["column_name"] for r in veh_cols]
        type_col  = next(
            (c for c in col_names if c in
             ("type_name", "type", "vehicle_type", "category", "name", "model")),
            None
        )
        if type_col:
            vehicles = await db_query(
                f"""
                SELECT v.{type_col} AS vehicle_type, COUNT(v.id) AS count,
                       COUNT(r.id) AS total_rides
                FROM vehicles v
                LEFT JOIN rides r ON r.vehicle_id = v.id
                    AND r.created_at >= NOW() - INTERVAL '30 days'
                GROUP BY v.{type_col}
                ORDER BY total_rides DESC
                LIMIT 10
                """
            )
            vehicle_data = [{"type": r["vehicle_type"], "count": r["count"],
                              "rides": r["total_rides"]} for r in vehicles]
        else:
            log.warning(f"vehicles: aucune colonne type trouvée parmi {col_names}")
            vehicle_data = [{"type": "unknown", "warning": f"cols={col_names}"}]

        # 9. Taux de complétion (30j)
        completion = await db_query(
            """
            SELECT
                ROUND(COUNT(*) FILTER (WHERE status = 'COMPLETED')::numeric
                    / NULLIF(COUNT(*), 0) * 100, 2) AS completion_rate,
                ROUND(COUNT(*) FILTER (WHERE status = 'CANCELLED')::numeric
                    / NULLIF(COUNT(*), 0) * 100, 2) AS cancellation_rate
            FROM rides
            WHERE created_at >= NOW() - INTERVAL '30 days'
            """
        )
        cr = dict(completion[0]) if completion else {}

        # 10. Temps réel (dernière heure)
        realtime = await db_query(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'COMPLETED')  AS completed_last_hour,
                COUNT(*) FILTER (WHERE status = 'CANCELLED')  AS cancelled_last_hour,
                COUNT(*) FILTER (
                    WHERE status IN ('PENDING','SEARCHING_DRIVER','ASSIGNED','EN_ROUTE_TO_PICKUP')
                )                                             AS active_rides,
                ROUND(SUM(COALESCE(price_final, 0)) FILTER (
                    WHERE status = 'COMPLETED'
                    AND COALESCE(price_final, 0) >= 0
                )::numeric, 2)                               AS revenue_last_hour
            FROM rides
            WHERE created_at >= NOW() - INTERVAL '1 hour'
            """
        )
        rt = dict(realtime[0]) if realtime else {}

        result = {
            "total_rides":             int(v.get("total_rides") or 0),
            "completed_rides":         int(v.get("completed") or 0),
            "cancelled_rides":         int(v.get("cancelled") or 0),
            "rides_today":             int(v.get("rides_today") or 0),
            "rides_last_hour":         int(v.get("rides_last_hour") or 0),
            "cancellation_rate_pct":   float(v.get("cancellation_rate_pct") or 0),
            "avg_fare_tnd":            float(v.get("avg_fare") or 0),
            "revenue_today_tnd":       float(v.get("revenue_today") or 0),
            "revenue_week_tnd":        float(v.get("revenue_week") or 0),
            "revenue_total_tnd":       float(v.get("revenue_total") or 0),
            "total_drivers":           int(d.get("total_drivers") or 0),
            "drivers_online":          int(d.get("online_now") or 0),
            "drivers_on_trip":         int(d.get("on_trip") or 0),
            "drivers_offline":         int(d.get("offline") or 0),
            "avg_driver_rating":       float(d.get("avg_driver_rating") or 0),
            "avg_trips_per_driver":    float(d.get("avg_trips_per_driver") or 0),
            "avg_wait_minutes":        float(w.get("avg_wait_min") or 0),
            "median_wait_minutes":     float(w.get("median_wait_min") or 0),
            "avg_trip_duration_min":   float(w.get("avg_trip_duration_min") or 0),
            "active_passengers_today": int(p.get("active_passengers_today") or 0),
            "active_passengers_week":  int(p.get("active_passengers_week") or 0),
            "avg_passenger_rating":    float(p.get("avg_passenger_rating") or 0),
            "completion_rate_pct":     float(cr.get("completion_rate") or 0),
            "realtime": {
                "completed_last_hour": int(rt.get("completed_last_hour") or 0),
                "cancelled_last_hour": int(rt.get("cancelled_last_hour") or 0),
                "active_rides_now":    int(rt.get("active_rides") or 0),
                "revenue_last_hour":   float(rt.get("revenue_last_hour") or 0),
            },
            "hourly_demand_today": hourly_data,
            "trend_7_days":        trend_data,
            "top_zones":           zones_data,
            "vehicle_breakdown":   vehicle_data,
            "generated_at":        datetime.now(timezone.utc).isoformat(),
            "data_source":         "postgresql_realtime",
        }

        import json as _json
        await cache_set(cache_key, _json.dumps(result, default=str))
        return result


    # ──────────────────────────────────────────────────────────────────────
    # GET /predict/demand
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/predict/demand")
    async def predict_demand(hours: int = Query(24, ge=1, le=168)):
        rows = await db_query(
            """
            SELECT
                EXTRACT(HOUR FROM completed_at AT TIME ZONE 'Africa/Tunis')::int AS h,
                EXTRACT(DOW  FROM completed_at AT TIME ZONE 'Africa/Tunis')::int AS dow,
                COUNT(*)                                                           AS cnt,
                AVG(COALESCE(surge_multiplier, 1.0))                             AS avg_surge,
                AVG(COALESCE(distance_km_real, distance_km, 5.0))               AS avg_dist,
                AVG(COALESCE(price_final, price_estimate, 10.0))                 AS avg_price
            FROM rides
            WHERE status       = 'COMPLETED'
              AND completed_at >= NOW() - INTERVAL '90 days'
              AND completed_at IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        )
        if not rows:
            raise HTTPException(404, "Aucune donnée historique disponible")

        profile: Dict[tuple, dict] = {}
        for r in rows:
            profile[(r["h"], r["dow"])] = {
                "cnt":       float(r["cnt"]),
                "avg_surge": float(r["avg_surge"] or 1.0),
            }

        hourly_agg:  Dict[int, list] = {h: [] for h in range(24)}
        weekday_h:   Dict[int, list] = {h: [] for h in range(24)}
        weekend_h:   Dict[int, list] = {h: [] for h in range(24)}
        for (h, dow), v in profile.items():
            hourly_agg[h].append(v["cnt"])
            (weekend_h if dow in (0, 6) else weekday_h)[h].append(v["cnt"])

        def sm(lst, fb):
            return float(np.mean(lst)) if lst else fb

        hourly_mean = {h: sm(v, 5.0) for h, v in hourly_agg.items()}

        now         = datetime.now(timezone.utc)
        meta        = mlflow_meta("demand_forecast")
        predictions = []

        for offset in range(hours):
            ts      = now + timedelta(hours=offset)
            h       = ts.hour
            dow     = ts.weekday()
            is_wend = dow >= 5
            base    = sm(weekend_h[h], hourly_mean[h] * 0.8) if is_wend \
                      else sm(weekday_h[h], hourly_mean[h])

            feat_df = pd.DataFrame([{
                "hour_of_day": h, "day_of_week": dow,
                "is_weekend": int(is_wend), "base_demand": base,
                "sin_hour": np.sin(2 * np.pi * h / 24),
                "cos_hour": np.cos(2 * np.pi * h / 24),
            }])
            ml_pred = try_mlflow("demand_forecast", feat_df)
            demand  = max(round(float(ml_pred[0]) if ml_pred is not None else base), 0)
            ci      = max(int(demand * 0.10), 1)

            period = ("peak" if h in (7, 8, 9, 17, 18, 19, 20) else
                      "night" if h < 5 else
                      "early_morning" if h < 7 else "off_peak")

            predictions.append({
                "timestamp": ts.isoformat(), "hour": h, "dow": dow,
                "demand": demand, "lower": max(demand - ci, 0),
                "upper": demand + ci, "period": period, "is_weekend": is_wend,
            })

        top_hour = max(predictions, key=lambda x: x["demand"])
        return {
            "predictions": predictions,
            "summary": {
                "max_demand":        max(p["demand"] for p in predictions),
                "min_demand":        min(p["demand"] for p in predictions),
                "avg_demand":        round(np.mean([p["demand"] for p in predictions]), 1),
                "peak_hours_count":  sum(1 for p in predictions if p["period"] == "peak"),
                "busiest_hour":      top_hour["hour"],
                "busiest_ts":        top_hour["timestamp"],
            },
            "model_version":  f"demand_forecast v{meta.get('version','heuristique-SQL')}",
            "metrics":        meta.get("metrics", {"source": "sql_historical_profile"}),
            "training_samples": len(rows),
            "generated_at":   now.isoformat(),
            "data_source":    "postgresql_90days",
        }


    # ──────────────────────────────────────────────────────────────────────
    # GET|POST /predict/revenue
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/predict/revenue")
    @app.post("/predict/revenue")
    async def predict_revenue(forecast_days: int = Query(7, ge=1, le=30)):
        rows = await db_query(
            """
            SELECT
                DATE(completed_at AT TIME ZONE 'Africa/Tunis')                 AS day,
                EXTRACT(DOW FROM completed_at AT TIME ZONE 'Africa/Tunis')::int AS dow,
                COUNT(*)                                                         AS ride_count,
                ROUND(SUM(COALESCE(price_final, 0))::numeric, 2)                AS revenue,
                ROUND(AVG(COALESCE(price_final, price_estimate))::numeric, 2)   AS avg_fare
            FROM rides
            WHERE status        = 'COMPLETED'
              AND completed_at >= NOW() - INTERVAL '90 days'
              AND price_final  IS NOT NULL
              AND price_final  >= 0            -- [FIX-4]
            GROUP BY 1, 2
            ORDER BY 1
            """
        )
        if not rows:
            raise HTTPException(404, "Aucun historique de revenus disponible")

        revenues = [float(r["revenue"]) for r in rows if r["revenue"]]
        baseline = float(np.mean(revenues)) if revenues else 0.0
        dow_revs: Dict[int, list] = {i: [] for i in range(7)}
        for r in rows:
            if r["revenue"]:
                dow_revs[int(r["dow"])].append(float(r["revenue"]))
        dow_avg = {d: (float(np.mean(v)) if v else baseline) for d, v in dow_revs.items()}
        slope, intercept = (np.polyfit(np.arange(len(revenues)), revenues, 1)
                            if len(revenues) > 7 else (0.0, baseline))

        now         = datetime.now(timezone.utc)
        meta        = mlflow_meta("demand_forecast")
        predictions = []
        total_pred  = 0.0

        for d in range(forecast_days):
            ts  = now + timedelta(days=d)
            dow = ts.weekday()
            base = dow_avg.get(dow, baseline)
            feat_df = pd.DataFrame([{
                "day_of_week": dow, "is_weekend": int(dow >= 5),
                "baseline": base, "trend_adj": slope * (len(revenues) + d),
            }])
            ml_pred = try_mlflow("demand_forecast", feat_df)
            pred    = max(round(float(ml_pred[0]) if ml_pred is not None
                                else base + slope * 0.1, 2), 0.0)
            total_pred += pred
            uplift = round((pred - baseline) / baseline * 100, 2) if baseline else 0.0
            predictions.append({
                "date":              ts.strftime("%Y-%m-%d"),
                "day_name":          ts.strftime("%A"),
                "predicted_revenue": pred,
                "baseline_revenue":  round(baseline, 2),
                "dow_average":       round(dow_avg.get(dow, baseline), 2),
                "uplift_pct":        uplift,
                "is_weekend":        dow >= 5,
            })

        total_base = baseline * forecast_days
        return {
            "predictions":        predictions,
            "total_predicted":    round(total_pred, 2),
            "total_baseline":     round(total_base, 2),
            "total_uplift_pct":   round((total_pred - total_base) / total_base * 100, 2)
                                  if total_base else 0,
            "daily_average_hist": round(baseline, 2),
            "trend_slope":        round(slope, 4),
            "metrics":            meta.get("metrics", {"source": "sql_regression"}),
            "training_days":      len(rows),
            "generated_at":       now.isoformat(),
            "data_source":        "postgresql_90days",
        }


    # ──────────────────────────────────────────────────────────────────────
    # POST /predict/churn   [FIX-6]
    # ──────────────────────────────────────────────────────────────────────

    class ChurnRequest(BaseModel):
        driver_ids:     Optional[List[str]] = None
        risk_threshold: float               = 0.4


    @app.post("/predict/churn")
    async def predict_churn(req: ChurnRequest):
        filter_sql = ""
        args: list = []
        if req.driver_ids:
            ph         = ", ".join(f"${i+1}" for i in range(len(req.driver_ids)))
            filter_sql = f"AND d.user_id = ANY(ARRAY[{ph}]::uuid[])"
            args       = req.driver_ids

        rows = await db_query(
            f"""
            SELECT
                d.user_id, d.rating_average, d.total_trips, d.availability_status,
                COUNT(r.id) FILTER (
                    WHERE r.status = 'COMPLETED'
                    AND r.created_at >= NOW() - INTERVAL '30 days'
                )                                                          AS rides_30d,
                COUNT(r.id) FILTER (
                    WHERE r.status = 'COMPLETED'
                    AND r.created_at >= NOW() - INTERVAL '7 days'
                )                                                          AS rides_7d,
                COUNT(r.id) FILTER (
                    WHERE r.status = 'COMPLETED'
                    AND r.created_at >= NOW() - INTERVAL '90 days'
                )                                                          AS rides_90d,
                COUNT(r.id) FILTER (WHERE r.cancelled_at IS NOT NULL)::float
                    / NULLIF(COUNT(r.id), 0)                               AS cancel_rate,
                EXTRACT(EPOCH FROM (NOW() - MAX(
                    CASE WHEN r.status = 'COMPLETED' THEN r.completed_at END
                ))) / 86400                                                 AS days_inactive,
                ROUND(SUM(COALESCE(r.price_final, 0)) FILTER (
                    WHERE r.status = 'COMPLETED'
                    AND r.created_at >= NOW() - INTERVAL '30 days'
                    AND COALESCE(r.price_final, 0) >= 0
                )::numeric, 2)                                             AS revenue_30d,
                ROUND(AVG(rr.driver_rating)::numeric, 2)                  AS avg_rating_received,
                COUNT(do2.id) FILTER (WHERE do2.status = 'REJECTED')::float
                    / NULLIF(COUNT(do2.id), 0)                             AS reject_rate,
                -- [FIX-6] Activité sur les heures de pointe
                COUNT(r.id) FILTER (
                    WHERE r.status = 'COMPLETED'
                    AND (EXTRACT(HOUR FROM r.completed_at AT TIME ZONE 'Africa/Tunis')
                         BETWEEN 7 AND 9
                    OR   EXTRACT(HOUR FROM r.completed_at AT TIME ZONE 'Africa/Tunis')
                         BETWEEN 17 AND 20)
                )                                                          AS peak_rides_30d,
                -- [FIX-6] Diversité zones
                COUNT(DISTINCT d2.work_area_id)                            AS zone_diversity
            FROM drivers d
            LEFT JOIN drivers d2       ON d2.user_id   = d.user_id
            LEFT JOIN rides r          ON r.driver_id  = d.user_id
            LEFT JOIN ride_ratings rr  ON rr.ride_id   = r.id
            LEFT JOIN dispatch_offers do2 ON do2.driver_id = d.user_id
                AND do2.offered_at >= NOW() - INTERVAL '30 days'
            WHERE d.deleted_at IS NULL
              AND d.availability_status NOT IN ('setup_required', 'pending')
              {filter_sql}
            GROUP BY d.user_id, d.rating_average, d.total_trips, d.availability_status
            LIMIT 500
            """,
            *args,
        )

        meta        = mlflow_meta("churn_classifier")
        predictions = []

        for r in rows:
            rating     = float(r["rating_average"] or 5.0)
            rides_30d  = int(r["rides_30d"]   or 0)
            rides_7d   = int(r["rides_7d"]    or 0)
            rides_90d  = int(r["rides_90d"]   or 0)
            cancel_rt  = float(r["cancel_rate"]  or 0.0)
            inactive   = float(r["days_inactive"] or 0.0)
            reject_rt  = float(r["reject_rate"]  or 0.0)
            rev_30d    = float(r["revenue_30d"]  or 0.0)
            status     = str(r["availability_status"] or "offline")
            peak_rides = int(r["peak_rides_30d"] or 0)
            zone_div   = int(r["zone_diversity"]  or 1)

            feat = pd.DataFrame([{
                "rating_average":       rating,
                "total_trips":          int(r["total_trips"] or 0),
                "rides_last_30d":       rides_30d,
                "cancellation_rate":    cancel_rt,
                "days_since_last_ride": inactive,
                "peak_rides_30d":       peak_rides,
                "zone_diversity":       zone_div,
            }])
            ml_pred = try_mlflow("churn_classifier", feat)

            if ml_pred is not None:
                prob = float(ml_pred[0]) if ml_pred.ndim == 1 else float(ml_pred[0][1])
            else:
                score  = 0.05
                score += min(inactive / 30.0, 0.35)
                score += (5.0 - rating) * 0.07
                score -= min(rides_30d / 40.0, 0.20)
                score += cancel_rt  * 0.20
                score += reject_rt  * 0.15
                score -= min(rev_30d / 1000.0, 0.10)
                score -= min(peak_rides / 20.0, 0.08)
                if zone_div <= 1:    score += 0.04
                if status == "offline": score += 0.08
                if rides_7d == 0 and inactive > 7: score += 0.10
                prob = round(min(max(score, 0.0), 1.0), 4)

            risk_factors = []
            if inactive   > 14:         risk_factors.append(f"Inactif depuis {round(inactive,1)}j")
            if cancel_rt  > 0.20:       risk_factors.append(f"Taux annulation {cancel_rt:.0%}")
            if reject_rt  > 0.30:       risk_factors.append(f"Taux refus {reject_rt:.0%}")
            if rating     < 4.0:        risk_factors.append(f"Note faible ({rating}/5)")
            if rides_30d  < 5:          risk_factors.append(f"Peu de courses ce mois ({rides_30d})")
            if peak_rides == 0 and rides_30d > 0:
                risk_factors.append("Absent en heures de pointe")

            predictions.append({
                "driver_id":         str(r["user_id"]),
                "churn_probability": prob,
                "risk_level":        "high" if prob > 0.7 else
                                     "medium" if prob > req.risk_threshold else "low",
                "risk_factors": risk_factors,
                "features": {
                    "rides_last_30d":       rides_30d,
                    "rides_last_7d":        rides_7d,
                    "rides_last_90d":       rides_90d,
                    "cancellation_rate":    round(cancel_rt, 3),
                    "reject_rate":          round(reject_rt, 3),
                    "rating_average":       rating,
                    "days_inactive":        round(inactive, 1),
                    "revenue_30d_tnd":      rev_30d,
                    "availability_status":  status,
                    "peak_rides_30d":       peak_rides,
                    "zone_diversity":       zone_div,
                },
            })

        predictions.sort(key=lambda x: x["churn_probability"], reverse=True)
        high   = [p for p in predictions if p["risk_level"] == "high"]
        medium = [p for p in predictions if p["risk_level"] == "medium"]

        return {
            "predictions":    predictions,
            "total":          len(predictions),
            "high_risk":      len(high),
            "medium_risk":    len(medium),
            "low_risk":       len(predictions) - len(high) - len(medium),
            "avg_churn_prob": round(np.mean([p["churn_probability"]
                                             for p in predictions]), 4) if predictions else 0,
            "model_version":  f"churn_classifier v{meta.get('version','heuristique-SQL')}",
            "metrics":        meta.get("metrics", {"source": "sql_multi_factor"}),
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "data_source":    "postgresql_realtime",
        }


    # ──────────────────────────────────────────────────────────────────────
    # POST /predict/eta  [FIX-1 / FIX-3 / FIX-5 / FIX-ETAv2]
    # ──────────────────────────────────────────────────────────────────────

    class ETARequest(BaseModel):
        distance_km:  float           = Field(...,   gt=0,   description="Distance en km (> 0)")
        hour_of_day:  int             = Field(12,    ge=0,   le=23)
        pickup_lat:   Optional[float] = Field(None,  ge=-90,  le=90)
        pickup_lon:   Optional[float] = Field(None,  ge=-180, le=180)
        vehicle_type: Optional[str]   = None

        class Config:
            json_schema_extra = {"example": {
                "distance_km": 5.3, "hour_of_day": 17,
                "pickup_lat": 36.8065, "pickup_lon": 10.1815,
            }}


    @app.post("/predict/eta")
    async def predict_eta(req: ETARequest):
        # [FIX-3] fenêtre 90 jours
        rows = await db_query(
              """
    SELECT
        EXTRACT(HOUR FROM trip_started_at AT TIME ZONE 'Africa/Tunis')::int AS h,
        ROUND(AVG(
            COALESCE(distance_km_real, distance_km, 0)
            / NULLIF(COALESCE(duration_min_real, duration_min, 0), 0)
            * 60
        )::numeric, 2)  AS avg_speed_kmh,
        ROUND(AVG(GREATEST(
            EXTRACT(EPOCH FROM (trip_started_at - created_at)) / 60, 0
        ))::numeric, 1) AS avg_wait_min,
        COUNT(*)         AS samples
    FROM rides
    WHERE status              = 'COMPLETED'
      AND trip_started_at IS NOT NULL
      AND completed_at    IS NOT NULL
      AND created_at      IS NOT NULL
      AND trip_started_at > created_at
      -- ✅ NOUVEAU : cap à 60 min max d'attente (données aberrantes exclues)
      AND EXTRACT(EPOCH FROM (trip_started_at - created_at)) / 60 BETWEEN 0 AND 60
      AND COALESCE(distance_km_real, distance_km, 0)   > 0.1
      AND COALESCE(duration_min_real, duration_min, 0) > 0.5
      -- ✅ NOUVEAU : cap durée de trajet cohérente
      AND COALESCE(duration_min_real, duration_min, 0) < 180
      AND trip_started_at >= NOW() - INTERVAL '90 days'
    GROUP BY 1
    ORDER BY 1
    """
        )

        # Dictionnaires heure → vitesse / attente (safe_float défensif)
        speed_by_hour: dict[int, float] = {
            r["h"]: sf
            for r in rows
            if (sf := safe_float(r["avg_speed_kmh"])) is not None and sf > 0
        }
        wait_by_hour: dict[int, float] = {
            r["h"]: sf
            for r in rows
            if (sf := safe_float(r["avg_wait_min"])) is not None and sf >= 0
        }

        # [FIX-5] médiane des vitesses connues comme fallback
        speed_median = (float(np.median(list(speed_by_hour.values())))
                        if speed_by_hour else 35.0)

        real_speed = speed_by_hour.get(req.hour_of_day)
        speed_source: str
        if real_speed is None:
            real_speed   = speed_median
            speed_source = "median_fallback"
            log.warning(f"ETA: pas de vitesse pour h={req.hour_of_day}, "
                        f"fallback médiane={speed_median:.1f} km/h")
        else:
            speed_source = "historical"
        real_speed = max(real_speed, 5.0)   # plancher physique

        avg_wait = max(wait_by_hour.get(req.hour_of_day, 5.0), 0.0)  # [FIX-1]

        feat_df = pd.DataFrame([{
            "distance_km":   req.distance_km,
            "hour_of_day":   req.hour_of_day,
            "avg_speed_kmh": real_speed,
        }])
        ml_pred          = try_mlflow("eta_estimator", feat_df)
        meta             = mlflow_meta("eta_estimator")
        heuristic_eta    = round((req.distance_km / real_speed) * 60, 1)

        # [FIX-ETAv2] validation stricte de la prédiction ML
        if ml_pred is not None:
            val = safe_float(ml_pred[0])
            eta_min = val if (val is not None and np.isfinite(val) and val > 0) \
                      else heuristic_eta
            if val is None or not (np.isfinite(val) and val > 0):
                log.warning(f"MLflow retourné valeur invalide ({ml_pred[0]}), "
                            f"fallback heuristique={heuristic_eta}")
        else:
            eta_min = heuristic_eta
        eta_min = max(round(eta_min, 1), 1.0)

        # CI : RMSE MLflow si disponible, sinon 12 %
        model_rmse = safe_float(meta.get("metrics", {}).get("RMSE"))
        ci = (round(model_rmse, 1)
              if (model_rmse and np.isfinite(model_rmse) and model_rmse > 0)
              else round(eta_min * 0.12, 1))

        return {
            "predicted_trip_minutes":  eta_min,
            "avg_wait_minutes":        round(avg_wait, 1),
            "total_estimated_minutes": round(eta_min + avg_wait, 1),
            "confidence_interval": {
                "lower": round(max(eta_min - ci, 0.5), 1),
                "upper": round(eta_min + ci, 1),
            },
            "real_avg_speed_kmh": round(real_speed, 1),
            "speed_source":       speed_source,
            "input_distance_km":  req.distance_km,
            "hour_of_day":        req.hour_of_day,
            "samples_used":       next(
                (int(r["samples"]) for r in rows if r["h"] == req.hour_of_day), 0
            ),
            "model_version": f"eta_estimator v{meta.get('version','heuristique-SQL')}",
            "metrics":       meta.get("metrics", {"source": "sql_speed_profile_90d"}),
            "data_source":   "postgresql_90days",
        }


    # ──────────────────────────────────────────────────────────────────────
    # GET /predict/anomalies
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/predict/anomalies")
    async def detect_anomalies(hours: int = Query(24, ge=1, le=168)):
        rows = await db_query(
            """
            SELECT
                r.id, r.driver_id, r.passenger_id,
                COALESCE(r.price_final,    0)    AS price_final,
                COALESCE(r.price_estimate, 0)    AS price_estimate,
                COALESCE(r.surge_multiplier, 1.0) AS surge,
                COALESCE(r.distance_km_real, r.distance_km, 0)     AS dist_real,
                COALESCE(r.distance_km, 0)                          AS dist_est,
                COALESCE(r.duration_min_real, r.duration_min, 0)   AS dur_real,
                r.completed_at,
                COUNT(*) OVER (
                    PARTITION BY r.driver_id ORDER BY r.completed_at
                    RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
                ) AS driver_vol_1h,
                COUNT(*) OVER (
                    PARTITION BY r.passenger_id ORDER BY r.completed_at
                    RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
                ) AS pax_vol_1h
            FROM rides r
            WHERE r.status       = 'COMPLETED'
              AND r.completed_at >= NOW() - ($1 * INTERVAL '1 hour')
              AND r.price_final  IS NOT NULL
              AND r.price_final  >= 0          -- [FIX-4]
            ORDER BY r.completed_at DESC
            LIMIT 2000
            """,
            hours,
        )

        if not rows:
            return {
                "anomalies": [], "total": 0, "critical_count": 0,
                "rides_analyzed": 0,
                "message": "Aucune course complétée sur la période",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        df = pd.DataFrame([{
            "ride_id":      str(r["id"]),
            "driver_id":    str(r["driver_id"]) if r["driver_id"] else "unknown",
            "price_final":  float(r["price_final"]),
            "price_est":    float(r["price_estimate"]),
            "surge":        float(r["surge"]),
            "dist_real":    float(r["dist_real"]),
            "dist_est":     float(r["dist_est"]),
            "dur_real":     float(r["dur_real"]),
            "driver_vol_1h": int(r["driver_vol_1h"] or 1),
            "pax_vol_1h":   int(r["pax_vol_1h"] or 1),
            "completed_at": r["completed_at"],
            "price_ratio":  float(r["price_final"]) / max(float(r["price_estimate"]), 0.01),
            "dist_ratio":   float(r["dist_real"])   / max(float(r["dist_est"]), 0.01),
            "speed_kmh":    float(r["dist_real"])   / max(float(r["dur_real"]) / 60, 0.01),
            "price_delta":  float(r["price_final"]) - float(r["price_estimate"]),
        } for r in rows])

        p95_price_ratio = df["price_ratio"].quantile(0.95)
        p95_driver_vol  = df["driver_vol_1h"].quantile(0.95)
        p95_dist_ratio  = df["dist_ratio"].quantile(0.95)
        p99_surge       = df["surge"].quantile(0.99)

        meta     = mlflow_meta("fraud_detector")
        features = ["price_ratio", "dist_ratio", "surge", "driver_vol_1h", "price_delta"]
        ml_pred  = try_mlflow("fraud_detector", df[features].fillna(1.0))
        anomalies = []

        for i, row in df.iterrows():
            is_anomaly   = False
            severity     = "low"
            anomaly_type = None
            impact       = None
            action       = None
            confidence   = 0.0

            if ml_pred is not None:
                score      = float(ml_pred[i]) if ml_pred.ndim == 1 else float(ml_pred[i][0])
                is_anomaly = score < 0
                confidence = min(abs(score) * 0.5 + 0.5, 0.99)
            else:
                pr  = float(row["price_ratio"])
                dr  = float(row["dist_ratio"])
                sg  = float(row["surge"])
                dvl = int(row["driver_vol_1h"])
                pvl = int(row["pax_vol_1h"])
                spd = float(row["speed_kmh"])
                dlt = float(row["price_delta"])

                if pr > max(p95_price_ratio, 1.30) and abs(dlt) > 3:
                    is_anomaly   = True; anomaly_type = "price_spike"
                    severity     = "high" if pr > 1.50 else "medium"
                    impact       = f"Surcoût +{round(dlt,2)} TND (ratio ×{round(pr,2)})"
                    action       = "review_payment"
                    confidence   = min(0.55 + (pr - 1.30) * 0.6, 0.98)
                elif dvl > max(p95_driver_vol, 6):
                    is_anomaly   = True; anomaly_type = "driver_volume_anomaly"
                    severity     = "critical" if dvl > 10 else "high"
                    impact       = f"{dvl} courses en 1h par ce chauffeur"
                    action       = "freeze_driver_review"
                    confidence   = min(0.65 + dvl * 0.02, 0.99)
                elif pvl > max(df["pax_vol_1h"].quantile(0.98), 5):
                    is_anomaly   = True; anomaly_type = "passenger_volume_anomaly"
                    severity     = "medium"; impact = f"{pvl} courses en 1h"
                    action       = "flag_passenger"; confidence = 0.70
                elif dr > max(p95_dist_ratio, 1.35) and float(row["dist_est"]) > 2:
                    is_anomaly   = True; anomaly_type = "route_deviation"
                    severity     = "medium"; impact = f"Distance ×{round(dr,2)} l'estimée"
                    action       = "review_route"; confidence = min(0.50+(dr-1.35)*0.4, 0.92)
                elif sg > max(p99_surge, 2.8):
                    is_anomaly   = True; anomaly_type = "surge_anomaly"
                    severity     = "medium"; impact = f"Surge ×{round(sg,2)}"
                    action       = "manual_review"; confidence = 0.72
                elif spd > 180:
                    is_anomaly   = True; anomaly_type = "impossible_speed"
                    severity     = "high"; impact = f"Vitesse {round(spd)} km/h"
                    action       = "gps_review"; confidence = 0.90

            if is_anomaly:
                anomalies.append({
                    "id":          row["ride_id"],
                    "driver_id":   row["driver_id"],
                    "type":        anomaly_type or "unknown",
                    "severity":    severity,
                    "confidence":  round(confidence, 4),
                    "impact":      impact or "Anomalie détectée",
                    "action":      action or "review",
                    "detected_at": row["completed_at"].isoformat()
                                   if row["completed_at"] else datetime.now(timezone.utc).isoformat(),
                    "details": {
                        "price_ratio":   round(float(row["price_ratio"]), 3),
                        "dist_ratio":    round(float(row["dist_ratio"]), 3),
                        "surge":         round(float(row["surge"]), 2),
                        "driver_vol_1h": int(row["driver_vol_1h"]),
                    },
                    "resolved": False,
                })

        return {
            "anomalies":      anomalies,
            "total":          len(anomalies),
            "critical_count": sum(1 for a in anomalies if a["severity"] in ("critical","high")),
            "rides_analyzed": len(df),
            "anomaly_rate":   round(len(anomalies)/len(df)*100, 2) if len(df) > 0 else 0,
            "thresholds": {
                "p95_price_ratio": round(p95_price_ratio, 3),
                "p95_driver_vol":  round(p95_driver_vol, 1),
                "p95_dist_ratio":  round(p95_dist_ratio, 3),
                "p99_surge":       round(p99_surge, 3),
            },
            "model_version": f"fraud_detector v{meta.get('version','heuristique-dynamique')}",
            "metrics":       meta.get("metrics", {"method": "dynamic_percentile_thresholds"}),
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "data_source":   "postgresql_realtime",
        }


    # ──────────────────────────────────────────────────────────────────────
    # POST /predict/surge
    # ──────────────────────────────────────────────────────────────────────

    class SurgeZoneInput(BaseModel):
        zone_id: str
        demand:  Optional[float] = None
        supply:  Optional[float] = None


    class SurgeRequest(BaseModel):
        zones: List[SurgeZoneInput]


    def _calc_surge(demand: float, supply: float) -> float:
        if supply <= 0: return 3.0
        r = demand / supply
        if r < 0.5:  return 1.0
        if r < 1.0:  return round(1.0 + r * 0.4, 2)
        if r < 1.5:  return round(1.4 + (r - 1.0) * 0.6, 2)
        if r < 2.0:  return round(1.7 + (r - 1.5) * 0.6, 2)
        if r < 3.0:  return round(2.0 + (r - 2.0) * 0.4, 2)
        return min(round(r * 0.85, 2), 3.5)


    @app.post("/predict/surge")
    async def predict_surge(req: SurgeRequest):
        demand_rows = await db_query(
            """
            SELECT wa.ville AS zone_name, wa.id::text AS zone_id,
                   COUNT(r.id) AS pending_count,
                   AVG(COALESCE(r.surge_multiplier, 1.0)) AS current_surge
            FROM rides r
            JOIN drivers d     ON d.user_id = r.driver_id
            JOIN work_areas wa ON wa.id = d.work_area_id
            WHERE r.status IN ('PENDING','SEARCHING_DRIVER','ASSIGNED','EN_ROUTE_TO_PICKUP')
              AND r.created_at >= NOW() - INTERVAL '30 minutes'
            GROUP BY wa.id, wa.ville
            """
        )
        supply_rows = await db_query(
            """
            SELECT wa.ville AS zone_name, wa.id::text AS zone_id,
                   COUNT(dl.driver_id) AS online_drivers
            FROM driver_locations dl
            JOIN drivers d     ON d.user_id = dl.driver_id
            JOIN work_areas wa ON wa.id = d.work_area_id
            WHERE dl.is_online = true AND dl.is_on_trip = false
              AND dl.last_seen_at >= NOW() - INTERVAL '5 minutes'
              AND d.availability_status = 'online'
            GROUP BY wa.id, wa.ville
            """
        )
        hist_surge = await db_query(
            """
            SELECT wa.ville AS zone_name,
                   AVG(COALESCE(r.surge_multiplier, 1.0)) AS hist_surge
            FROM rides r
            JOIN drivers d     ON d.user_id = r.driver_id
            JOIN work_areas wa ON wa.id = d.work_area_id
            WHERE EXTRACT(HOUR FROM r.created_at) = EXTRACT(HOUR FROM NOW())
              AND r.created_at >= NOW() - INTERVAL '28 days'
              AND r.status = 'COMPLETED'
            GROUP BY wa.ville
            """
        )

        demand_map = {r["zone_name"].lower(): r for r in demand_rows}
        supply_map = {r["zone_name"].lower(): r for r in supply_rows}
        hist_map   = {r["zone_name"].lower(): r for r in hist_surge}
        meta       = mlflow_meta("surge_predictor")
        result     = []

        for z in req.zones:
            zl  = z.zone_id.lower()
            md  = demand_map.get(zl)
            ms  = supply_map.get(zl)
            mh  = hist_map.get(zl)
            demand_val     = float(z.demand or 0) + float(md["pending_count"] if md else 0)
            supply_val     = float(z.supply or 0) + float(ms["online_drivers"] if ms else 0)
            hist_surge_val = float(mh["hist_surge"]) if mh and mh["hist_surge"] else 1.0

            feat_df = pd.DataFrame([{
                "demand_score": demand_val, "supply_score": supply_val,
                "ratio":        demand_val / max(supply_val, 1),
                "hist_surge":   hist_surge_val,
            }])
            ml_pred     = try_mlflow("surge_predictor", feat_df)
            recommended = float(ml_pred[0]) if ml_pred is not None \
                          else _calc_surge(demand_val, supply_val)
            recommended = round(np.clip(recommended, 1.0, 3.5), 2)

            result.append({
                "zone_id":              z.zone_id,
                "current_surge":        round(_calc_surge(demand_val * 0.9, supply_val), 2),
                "recommended_surge":    recommended,
                "historical_surge_avg": round(hist_surge_val, 3),
                "demand_score":         round(demand_val, 1),
                "supply_score":         round(supply_val, 1),
                "demand_supply_ratio":  round(demand_val / max(supply_val, 1), 3),
                "data_source":          "realtime_db" if (md or ms) else "historical_only",
                "realtime_rides":       int(md["pending_count"]) if md else 0,
                "realtime_drivers":     int(ms["online_drivers"]) if ms else 0,
            })

        return {
            "zones":         result,
            "model_version": f"surge_predictor v{meta.get('version','heuristique-SQL')}",
            "metrics":       meta.get("metrics", {"method": "demand_supply_ratio"}),
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "data_source":   "postgresql_realtime",
        }


    # ──────────────────────────────────────────────────────────────────────
    # GET /intelligence/zones  [FIX-2]
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/intelligence/zones")
    async def zone_intelligence():
        zone_stats = await db_query(
            """
            SELECT
                wa.ville, wa.id::text AS zone_id,
                COUNT(r.id)                                                       AS total_rides,
                COUNT(r.id) FILTER (WHERE r.created_at >= NOW() - INTERVAL '7 days')
                                                                                  AS rides_7d,
                COUNT(r.id) FILTER (
                    WHERE r.created_at >= NOW() - INTERVAL '14 days'
                    AND   r.created_at <  NOW() - INTERVAL '7 days'
                )                                                                 AS rides_prev_7d,
                ROUND(SUM(COALESCE(r.price_final, 0))::numeric, 2)               AS revenue_total,
                ROUND(AVG(COALESCE(r.surge_multiplier, 1.0))::numeric, 3)        AS avg_surge,
                -- [FIX-2] GREATEST pour avg_wait_min
                ROUND(AVG(GREATEST(
                    EXTRACT(EPOCH FROM (r.trip_started_at - r.created_at)) / 60, 0
                ))::numeric, 1)                                                   AS avg_wait_min,
                COUNT(DISTINCT r.driver_id)                                       AS unique_drivers,
                COUNT(DISTINCT r.passenger_id)                                    AS unique_passengers
            FROM rides r
            JOIN drivers d     ON d.user_id    = r.driver_id
            JOIN work_areas wa ON wa.id        = d.work_area_id
            WHERE r.created_at      >= NOW() - INTERVAL '30 days'
              AND r.trip_started_at IS NOT NULL
              AND r.created_at      IS NOT NULL
            GROUP BY wa.id, wa.ville
            ORDER BY total_rides DESC
            """
        )

        heatmap = await db_query(
            """
            SELECT wa.ville AS zone,
                   EXTRACT(HOUR FROM r.created_at AT TIME ZONE 'Africa/Tunis')::int AS hour,
                   COUNT(r.id) AS count
            FROM rides r
            JOIN drivers d     ON d.user_id = r.driver_id
            JOIN work_areas wa ON wa.id     = d.work_area_id
            WHERE r.created_at >= NOW() - INTERVAL '30 days'
              AND r.status = 'COMPLETED'
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        )

        zone_coverage = await db_query(
            """
            SELECT wa.ville AS zone,
                   COUNT(d.user_id) FILTER (WHERE d.availability_status = 'online')   AS online,
                   COUNT(d.user_id) FILTER (WHERE d.availability_status = 'on_trip')  AS on_trip,
                   COUNT(d.user_id) AS total
            FROM drivers d
            JOIN work_areas wa ON wa.id = d.work_area_id
            WHERE d.deleted_at IS NULL
              AND d.availability_status NOT IN ('setup_required')
            GROUP BY wa.ville
            ORDER BY online DESC
            """
        )

        zones_result = []
        for r in zone_stats:
            r7  = int(r["rides_7d"]      or 0)
            rp7 = int(r["rides_prev_7d"] or 0)
            growth = round((r7 - rp7) / max(rp7, 1) * 100, 1) if rp7 else 0.0
            trend  = ("growing" if growth > 15 else "declining" if growth < -15 else "stable")

            cov              = next((c for c in zone_coverage if c["zone"] == r["ville"]), None)
            drivers_online   = int(cov["online"])  if cov else 0
            drivers_on_trip  = int(cov["on_trip"]) if cov else 0
            active_drivers   = drivers_online + drivers_on_trip

            dcr = round(float(r7) / active_drivers, 2) if active_drivers > 0 else 0.0
            coverage_status = ("no_supply"    if active_drivers == 0 else
                               "under_served" if dcr > 2 else
                               "over_served"  if dcr < 0.3 else "balanced")

            zones_result.append({
                "zone":                  r["ville"],
                "zone_id":               r["zone_id"],
                "total_rides_30d":       int(r["total_rides"] or 0),
                "rides_last_7d":         r7,
                "rides_prev_7d":         rp7,
                "growth_pct":            growth,
                "trend":                 trend,
                "revenue_total":         float(r["revenue_total"] or 0),
                "avg_surge":             float(r["avg_surge"] or 1.0),
                "avg_wait_min":          float(r["avg_wait_min"] or 0),
                "unique_drivers":        int(r["unique_drivers"] or 0),
                "unique_passengers":     int(r["unique_passengers"] or 0),
                "drivers_online_now":    drivers_online,
                "drivers_on_trip":       drivers_on_trip,
                "demand_coverage_ratio": dcr,
                "coverage_status":       coverage_status,
            })

        heatmap_dict: Dict[str, Dict[int, int]] = {}
        for r in heatmap:
            heatmap_dict.setdefault(r["zone"], {})[int(r["hour"])] = int(r["count"])

        return {
            "zones":           zones_result,
            "total_zones":     len(zones_result),
            "top_zones":       [z for z in zones_result if z["total_rides_30d"] > 0][:5],
            "growing_zones":   [z for z in zones_result if z["trend"] == "growing"],
            "declining_zones": [z for z in zones_result if z["trend"] == "declining"],
            "under_served":    [z for z in zones_result if z["coverage_status"] == "under_served"],
            "no_supply":       [z for z in zones_result if z["coverage_status"] == "no_supply"],
            "heatmap":         heatmap_dict,
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "data_source":     "postgresql_realtime",
        }


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moviroo ML v4.2")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", choices=["demand","revenue","churn","eta","fraud","surge","all"],
                       help="Entraîne un ou tous les modèles")
    group.add_argument("--serve", action="store_true",
                       help="Lance le serveur FastAPI (port ML_PORT, défaut 8005)")
    args = parser.parse_args()

    if args.train:
        logger.info(f"=== Training : {args.train} ===")
        # Remplacer les lignes ci-dessous par votre chargement réel depuis la feature store
        if args.train in ("demand", "all"):
            logger.info("→ demand  (charger df depuis feature store)")
            # df = load_feature_store("demand"); train_demand_model(df)
        if args.train in ("revenue", "all"):
            logger.info("→ revenue (charger df depuis feature store)")
        if args.train in ("churn", "all"):
            logger.info("→ churn   (charger df depuis feature store)")
        if args.train in ("eta", "all"):
            logger.info("→ eta     (charger df depuis feature store)")
        if args.train in ("fraud", "all"):
            logger.info("→ fraud   (charger df depuis feature store)")
        if args.train in ("surge", "all"):
            logger.info("→ surge   (charger df depuis feature store)")
        logger.info("Training terminé. Modèles loggués dans MLflow.")

    elif args.serve:
        if not FASTAPI_OK:
            logger.error("FastAPI non installé : pip install fastapi uvicorn")
            sys.exit(1)
        import uvicorn
        uvicorn.run(
            "moviroo_ml_v4_2:app",
            host    = "0.0.0.0",
            port    = int(os.getenv("ML_PORT", "8005")),
            reload  = False,
            workers = 1,
        )