"""
models/surge_predictor.py

XGBoost pour prédire le surge_multiplier par zone × heure.
Cible : R² > 0.92
"""
import os
import numpy as np
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from config import ML


def train_surge_model(df: pd.DataFrame) -> dict:
    """
    Entraîne XGBoost sur les features de surge.
    df doit contenir les colonnes issues de surge_features.build_surge_features().
    """
    target = "surge_multiplier"
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols].values
    y = df[target].values

    # Adapte test_size si peu de données
    n = len(X)
    test_size = 0.20 if n >= 20 else 0.25
    if n < 10:
        print(f"[surge_predictor] ⚠️  Seulement {n} lignes — résultats indicatifs.")
        X_train, X_test, y_train, y_test = X, X, y, y   # train=test si trop petit
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Adapte n_estimators et early_stopping à la taille du dataset
    n_estimators       = min(800, max(50, n * 10))
    early_stopping     = min(50,  max(10, n_estimators // 10))
    min_child_weight   = max(1, n // 20)

    params = {
        "n_estimators":        n_estimators,
        "max_depth":           6,
        "learning_rate":       0.05,
        "subsample":           0.8,
        "colsample_bytree":    0.8,
        "min_child_weight":    min_child_weight,
        "reg_alpha":           0.1,
        "reg_lambda":          1.0,
        "objective":           "reg:squarederror",
        "eval_metric":         "mae",
        "early_stopping_rounds": early_stopping,
        "n_jobs":              -1,
        "random_state":        42,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    y_pred = model.predict(X_test)
    metrics = {
        "r2":           r2_score(y_test, y_pred),
        "mae":          mean_absolute_error(y_test, y_pred),
        "within_10pct": float(np.mean(np.abs(y_pred - y_test) / (y_test + 1e-6) < 0.10)),
        "n_samples":    n,
        "n_train":      len(X_train),
        "n_test":       len(X_test),
    }

    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return {
        "model":              model,
        "metrics":            metrics,
        "feature_importance": importance,
        "X_test":             X_test,
        "y_test":             y_test,
        "y_pred":             y_pred,
        "feature_cols":       feature_cols,
        "params":             params,
    }


def predict_zone_hour(
    model: xgb.XGBRegressor,
    zone_lat: float,
    zone_lon: float,
    hour_of_day: int,
    day_of_week: int,
    concurrent_rides: int,
    feature_cols: list[str],
) -> float:
    """
    Prédit le surge multiplier pour une zone et heure données.
    Retourne le multiplicateur clippé à [1.0, 3.5].
    """
    row = {c: 0.0 for c in feature_cols}
    row["zone_lat"]                  = zone_lat
    row["zone_lon"]                  = zone_lon
    row["hour_of_day"]               = hour_of_day
    row["day_of_week"]               = day_of_week
    row["concurrent_rides_in_hour"]  = concurrent_rides
    row["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    row["dow_sin"]  = np.sin(2 * np.pi * day_of_week / 7)
    row["dow_cos"]  = np.cos(2 * np.pi * day_of_week / 7)

    X    = np.array([[row[c] for c in feature_cols]])
    pred = float(model.predict(X)[0])
    return float(np.clip(pred, 1.0, 3.5))


def train_and_log(df: pd.DataFrame) -> str:
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    mlflow.set_experiment(ML.mlflow_experiment)

    with mlflow.start_run(run_name="surge_predictor_v2") as run:
        result = train_surge_model(df)

        mlflow.log_params({
            "model":          "XGBoostRegressor",
            "n_estimators":   result["params"]["n_estimators"],
            "max_depth":      result["params"]["max_depth"],
            "learning_rate":  result["params"]["learning_rate"],
            "n_samples":      result["metrics"]["n_samples"],
        })
        mlflow.log_metrics({
            k: v for k, v in result["metrics"].items()
            if isinstance(v, (int, float))
        })

        # ── Sauvegarde modèle ─────────────────────────────────────────────
        # Utilise le booster natif pour contourner le bug sklearn mixin XGBoost
        tmp_path = os.path.join(os.environ.get("TEMP", "/tmp"), "surge_xgb.json")
        result["model"].get_booster().save_model(tmp_path)
        mlflow.log_artifact(tmp_path, artifact_path="surge_model")

        # Top 10 feature importance
        for feat, imp in result["feature_importance"].head(10).items():
            mlflow.log_metric(f"feat_imp_{feat}", float(imp))

        print(
            f"[surge_predictor] R²={result['metrics']['r2']:.4f} | "
            f"MAE={result['metrics']['mae']:.4f} | "
            f"Within 10%={result['metrics']['within_10pct']:.1%} | "
            f"n={result['metrics']['n_samples']}"
        )

        if result["metrics"]["r2"] < 0.5:
            print("[surge_predictor] ⚠️  R² faible — collectez plus de données "
                  "pour un modèle fiable (idéal : ≥ 500 rides).")

        return run.info.run_id


if __name__ == "__main__":
    from feature_engineering.surge_features import run as fe_run
    data = fe_run()
    train_and_log(data["df"])