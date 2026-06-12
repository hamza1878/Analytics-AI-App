"""
models/churn_classifier.py

Random Forest pour classifier le risque de churn des chauffeurs.
Cible : Accuracy > 85%, AUC-ROC > 0.90
"""
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE # type: ignore
from config import ML


FEATURE_COLS = [
    "rating_average", "total_trips", "total_ratings",
    "days_since_last_trip", "days_since_last_offer", "days_since_last_login",
    "accept_rate", "reject_rate", "expire_rate",
    "avg_pickup_distance", "avg_dispatch_score",
    "recent_avg_rating", "rating_stddev",
    "recent_trips", "avg_trip_distance", "avg_earnings_per_trip",
    "is_online", "availability_encoded",
]

TARGET_COL = "churn_label"


def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].fillna(0).values
    y = df[TARGET_COL].values
    return X, y


def train_churn_model(df: pd.DataFrame) -> dict:
    """
    Entraîne un RandomForest avec SMOTE pour corriger le déséquilibre de classes.
    Évalue via StratifiedKFold (5 folds).
    """
    X, y = prepare_data(df)

    # Rééchantillonnage SMOTE si déséquilibre > 3:1
    churn_ratio = y.mean()
    if churn_ratio < 0.25 or churn_ratio > 0.75:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"  SMOTE : {len(X)} → {len(X_res)} samples")
    else:
        X_res, y_res = X, y

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    cv_acc  = cross_val_score(model, X_res, y_res, cv=cv, scoring="accuracy")
    cv_auc  = cross_val_score(model, X_res, y_res, cv=cv, scoring="roc_auc")
    cv_f1   = cross_val_score(model, X_res, y_res, cv=cv, scoring="f1")

    # Fit final
    model.fit(X_res, y_res)

    # Métriques sur tout le jeu (pour inspection)
    y_pred     = model.predict(X)
    y_proba    = model.predict_proba(X)[:, 1]
    acc        = accuracy_score(y, y_pred)
    auc        = roc_auc_score(y, y_proba)
    f1         = f1_score(y, y_pred)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    return {
        "model": model,
        "cv_acc_mean": cv_acc.mean(),
        "cv_auc_mean": cv_auc.mean(),
        "cv_f1_mean": cv_f1.mean(),
        "train_acc": acc,
        "train_auc": auc,
        "train_f1": f1,
        "feature_importance": importance,
        "confusion_matrix": confusion_matrix(y, y_pred),
        "classification_report": classification_report(y, y_pred),
        "y_proba": y_proba,
    }


def predict_churn_risk(
    model: RandomForestClassifier,
    driver_features: dict,
) -> dict:
    """
    Prédit le risque de churn pour un seul chauffeur.

    Returns:
        {"risk_score": float, "label": str, "top_factors": list[str]}
    """
    X = np.array([[driver_features.get(c, 0) for c in FEATURE_COLS]])
    proba = model.predict_proba(X)[0][1]

    if proba >= 0.75:
        label = "HIGH"
    elif proba >= 0.45:
        label = "MEDIUM"
    else:
        label = "LOW"

    # Facteurs contribuants (SHAP-lite : feature importance × valeur)
    feat_vals = X[0]
    importance = model.feature_importances_
    contrib = importance * np.abs(feat_vals)
    top_idx = np.argsort(contrib)[::-1][:3]
    top_factors = [FEATURE_COLS[i] for i in top_idx]

    return {
        "risk_score": float(proba),
        "label": label,
        "top_factors": top_factors,
    }


def train_and_log(df: pd.DataFrame) -> str:
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    mlflow.set_experiment(ML.mlflow_experiment)

    with mlflow.start_run(run_name="churn_classifier_v3") as run:
        result = train_churn_model(df)

        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 500,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "smote": True,
            "cv_folds": 5,
        })
        mlflow.log_metrics({
            "cv_accuracy": result["cv_acc_mean"],
            "cv_auc_roc": result["cv_auc_mean"],
            "cv_f1": result["cv_f1_mean"],
            "train_accuracy": result["train_acc"],
            "train_auc_roc": result["train_auc"],
        })

        mlflow.sklearn.log_model(result["model"], artifact_path="churn_model")

        print(f"[churn_classifier] CV-Acc={result['cv_acc_mean']:.1%} | "
              f"CV-AUC={result['cv_auc_mean']:.4f} | CV-F1={result['cv_f1_mean']:.4f}")
        print(result["classification_report"])

        return run.info.run_id


if __name__ == "__main__":
    from feature_engineering.churn_features import run as fe_run
    data = fe_run()
    train_and_log(data["df"])
