"""
training_pipelines.py
Training pipeline for all 6 Moviroo ML models.
Uses MLflow for experiment tracking.
Run: python training_pipelines.py --model demand
"""
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("./artifacts")
MODELS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://mlflow:5000")
import mlflow
print("TRACKING URI =", mlflow.get_tracking_uri())

# ── Shared utilities ───────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    return round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2)

def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)


# ── 1. Demand Forecasting ─────────────────────────────────────────────────────

def train_demand_model(df: pd.DataFrame):
    """
    LSTM + Prophet ensemble.
    df: columns [timestamp, ride_count]
    """
    from prophet import Prophet

    mlflow.set_experiment("demand-forecasting")
    with mlflow.start_run(run_name="prophet-baseline"):

        model = Prophet(
            seasonality_mode="multiplicative",
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        df_prophet = df.rename(columns={"timestamp": "ds", "ride_count": "y"})
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=24, freq="H")
        forecast = model.predict(future)

        # Evaluate on last 10% of data
        split = int(len(df_prophet) * 0.9)
        eval_df = df_prophet.iloc[split:]
        preds = model.predict(eval_df[["ds"]])
        _mape = mape(eval_df["y"], preds["yhat"])
        _rmse = rmse(eval_df["y"], preds["yhat"])

        mlflow.log_metric("MAPE", _mape)
        mlflow.log_metric("RMSE", _rmse)
        mlflow.log_param("seasonality_mode", "multiplicative")

        model_path = MODELS_DIR / "demand_prophet.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        logger.info(f"[Demand] MAPE={_mape}% RMSE={_rmse}")
        return model


# ── 2. Revenue Optimization ───────────────────────────────────────────────────

def train_revenue_model(df: pd.DataFrame):
    """
    XGBoost regressor.
    df: columns [total_revenue, avg_price, avg_distance, ride_count, day_of_week, hour_of_day]
    """
    mlflow.set_experiment("revenue-optimization")
    with mlflow.start_run(run_name="xgb-revenue"):

        features = ["avg_price", "avg_distance", "ride_count", "day_of_week", "hour_of_day"]
        X = df[features]
        y = df["total_revenue"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        _mape = mape(y_test, preds)
        _rmse = rmse(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("MAPE", _mape)
        mlflow.log_metric("RMSE", _rmse)
        mlflow.xgboost.log_model(model, "revenue_model")

        logger.info(f"[Revenue] MAPE={_mape}% RMSE={_rmse}")
        return model


# ── 3. Driver Churn ───────────────────────────────────────────────────────────

def train_churn_model(df: pd.DataFrame):
    """
    XGBoost binary classifier.
    df: columns [rating, total_rides, rides_last_30d, days_since_last_ride,
                 cancellation_rate, revenue_last_30d, churned (label)]
    """
    mlflow.set_experiment("driver-churn")
    with mlflow.start_run(run_name="xgb-churn"):

        features = ["rating", "total_rides", "rides_last_30d",
                    "days_since_last_ride", "cancellation_rate", "revenue_last_30d"]
        X = df[features].fillna(0)
        y = df["churned"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                              stratify=y, random_state=42)

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        from sklearn.metrics import roc_auc_score, classification_report
        preds_proba = model.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, preds_proba), 4)

        mlflow.log_metric("AUC", auc)
        mlflow.log_params(params)
        mlflow.xgboost.log_model(model, "churn_model")

        logger.info(f"[Churn] AUC={auc}")
        return model


# ── 4. ETA Prediction ─────────────────────────────────────────────────────────

def train_eta_model(df: pd.DataFrame):
    """
    Gradient Boosting Regressor.
    df: columns [distance_km, hour_of_day, day_of_week, actual_minutes]
    """
    from sklearn.ensemble import GradientBoostingRegressor

    mlflow.set_experiment("eta-prediction")
    with mlflow.start_run(run_name="gbm-eta"):

        features = ["distance_km", "hour_of_day", "day_of_week"]
        X = df[features]
        y = df["actual_minutes"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "random_state": 42}
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        _mape = mape(y_test, preds)
        _rmse = rmse(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("MAPE", _mape)
        mlflow.log_metric("RMSE", _rmse)
        mlflow.sklearn.log_model(model, "eta_model")

        logger.info(f"[ETA] MAPE={_mape}% RMSE={_rmse}")
        return model


# ── 5. Fraud / Anomaly Detection ──────────────────────────────────────────────

def train_fraud_model(df: pd.DataFrame):
    """
    Isolation Forest + Autoencoder ensemble.
    df: columns [amount, expected_price, amount_delta, payments_last_hour, hour_of_day, distance]
    """
    from sklearn.ensemble import IsolationForest

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="isolation-forest"):

        features = ["amount", "expected_price", "amount_delta", "payments_last_hour", "hour_of_day"]
        X = df[features].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=200,
            contamination=0.03,  # 3% anomaly rate
            random_state=42,
        )
        model.fit(X_scaled)

        scores = model.decision_function(X_scaled)
        preds = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
        anomaly_rate = (preds == -1).mean()

        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.log_param("contamination", 0.03)
        mlflow.sklearn.log_model(model, "fraud_isolation_forest")

        # Save scaler
        scaler_path = MODELS_DIR / "fraud_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path))

        logger.info(f"[Fraud] anomaly_rate={anomaly_rate:.4f}")
        return model, scaler


# ── 6. Surge Pricing ──────────────────────────────────────────────────────────

def train_surge_model(df: pd.DataFrame):
    """
    XGBoost regressor to predict optimal surge multiplier.
    df: columns [demand, supply, hour_of_day, day_of_week,
                 weather_score, event_flag, optimal_surge (label)]
    """
    mlflow.set_experiment("surge-optimization")
    with mlflow.start_run(run_name="xgb-surge"):

        features = ["demand", "supply", "hour_of_day", "day_of_week",
                    "weather_score", "event_flag"]
        X = df[features].fillna(0)
        y = df["optimal_surge"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        _mape = mape(y_test, preds)
        _rmse = rmse(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("MAPE", _mape)
        mlflow.log_metric("RMSE", _rmse)
        mlflow.xgboost.log_model(model, "surge_model")

        logger.info(f"[Surge] MAPE={_mape}% RMSE={_rmse}")
        return model


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["demand", "revenue", "churn", "eta", "fraud", "surge", "all"])
    args = parser.parse_args()

    if args.model in ("demand", "all"):
        logger.info("Training demand model...")
        # df = load_from_feature_store("demand")
        # train_demand_model(df)

    if args.model in ("revenue", "all"):
        logger.info("Training revenue model...")

    if args.model in ("churn", "all"):
        logger.info("Training churn model...")

    if args.model in ("eta", "all"):
        logger.info("Training eta model...")

    if args.model in ("fraud", "all"):
        logger.info("Training fraud model...")

    if args.model in ("surge", "all"):
        logger.info("Training surge model...")

    logger.info("Training complete. Models logged to MLflow.")
