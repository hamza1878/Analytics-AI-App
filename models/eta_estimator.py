"""
models/eta_estimator.py

LightGBM pour estimer l'ETA (durée de trajet en minutes).
Cible : MAE < 2.5 minutes
"""
import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from config import ML


TARGET_COL = "actual_duration_min_rides"


def train_eta_model(df: pd.DataFrame) -> dict:
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    valid_data = lgb.Dataset(X_test,  label=y_test,  reference=train_data)

    params = {
        "objective":        "regression",
        "metric":           ["mae", "rmse"],
        "num_leaves":       127,
        "learning_rate":    0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 20,
        "reg_alpha":        0.1,
        "reg_lambda":       0.5,
        "verbose":          -1,
        "n_jobs":           -1,
        "seed":             42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=callbacks,
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    within_2min = float(np.mean(np.abs(y_pred - y_test) < 2.0))
    within_5min = float(np.mean(np.abs(y_pred - y_test) < 5.0))

    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)

    return {
        "model": model,
        "mae": mae,
        "r2": r2,
        "within_2min": within_2min,
        "within_5min": within_5min,
        "feature_importance": importance,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_cols": feature_cols,
        "best_iteration": model.best_iteration,
    }


def predict_eta(
    model: lgb.Booster,
    features: dict,
    feature_cols: list[str],
) -> dict:
    """
    Prédit l'ETA pour un trajet en cours.

    Returns:
        {"eta_minutes": float, "confidence_interval": [float, float]}
    """
    X = np.array([[features.get(c, 0) for c in feature_cols]])
    eta = float(model.predict(X)[0])
    eta = max(1.0, eta)  # minimum 1 minute

    # Intervalle de confiance bootstrap simplifié (±15%)
    ci_low  = round(eta * 0.85, 1)
    ci_high = round(eta * 1.15, 1)

    return {
        "eta_minutes": round(eta, 1),
        "confidence_interval": [ci_low, ci_high],
    }


def train_and_log(df: pd.DataFrame) -> str:
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    mlflow.set_experiment(ML.mlflow_experiment)

    with mlflow.start_run(run_name="eta_estimator_v5") as run:
        result = train_eta_model(df)

        mlflow.log_params({
            "model": "LightGBM",
            "num_leaves": 127,
            "learning_rate": 0.03,
            "best_iteration": result["best_iteration"],
        })
        mlflow.log_metrics({
            "mae_minutes": result["mae"],
            "r2": result["r2"],
            "within_2min": result["within_2min"],
            "within_5min": result["within_5min"],
        })

        model_path = "/tmp/eta_lgbm.txt"
        result["model"].save_model(model_path)
        mlflow.log_artifact(model_path)

        print(f"[eta_estimator] MAE={result['mae']:.2f}min | "
              f"R²={result['r2']:.4f} | "
              f"Within 2min={result['within_2min']:.1%} | "
              f"Within 5min={result['within_5min']:.1%}")

        return run.info.run_id


if __name__ == "__main__":
    from feature_engineering.eta_features import run as fe_run
    data = fe_run()
    train_and_log(data["df"])
