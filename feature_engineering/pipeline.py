"""
feature_engineering/pipeline.py

Orchestrateur principal du pipeline de feature engineering.
Exécute tous les modules dans l'ordre et sauvegarde les artefacts.
"""
import os
import uuid
import pandas as pd
from datetime import datetime


def run_full_pipeline(save_artifacts: bool = True) -> dict:
    """
    Exécute le pipeline complet :
      1. demand_features   → zone_hour_df, LSTM sequences
      2. surge_features    → feature matrix XGBoost
      3. churn_features    → feature matrix Random Forest
      4. eta_features      → feature matrix LightGBM

    Retourne un dict avec tous les DataFrames prêts pour l'entraînement.
    """
    start = datetime.now()
    print(f"\n{'='*60}")
    print(f"  Moviroo ML — Pipeline Feature Engineering")
    print(f"  Démarré à {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    results = {}

    # 1. Demand
    print("[1/4] Features demande…")
    try:
        from feature_engineering.demand_features import run as demand_run
        results["demand"] = demand_run()
        print(f"  ✓ demand OK\n")
    except Exception as e:
        print(f"  ✗ demand ERREUR : {e}\n")

    # 2. Surge
    print("[2/4] Features surge…")
    try:
        from feature_engineering.surge_features import run as surge_run
        results["surge"] = surge_run()
        print(f"  ✓ surge OK\n")
    except Exception as e:
        print(f"  ✗ surge ERREUR : {e}\n")

    # 3. Churn
    print("[3/4] Features churn…")
    try:
        from feature_engineering.churn_features import run as churn_run
        results["churn"] = churn_run()
        print(f"  ✓ churn OK\n")
    except Exception as e:
        print(f"  ✗ churn ERREUR : {e}\n")

    # 4. ETA
    print("[4/4] Features ETA…")
    try:
        from feature_engineering.eta_features import run as eta_run
        results["eta"] = eta_run()
        print(f"  ✓ eta OK\n")
    except Exception as e:
        print(f"  ✗ eta ERREUR : {e}\n")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"{'='*60}")
    print(f"  Pipeline terminé en {elapsed:.1f}s — {len(results)}/4 modules OK")
    print(f"{'='*60}\n")

    # Sauvegarde optionnelle
    if save_artifacts:
        os.makedirs("/tmp/moviroo_features", exist_ok=True)

        for name, data in results.items():
            if not data or "df" not in data:
                continue

            df = data["df"].copy()

            # Convert UUID columns to string — PyArrow cannot serialize uuid.UUID objects
            for col in df.columns:
                if df[col].dtype == object:
                    first_valid = df[col].dropna()
                    if not first_valid.empty and isinstance(first_valid.iloc[0], uuid.UUID):
                        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else x)

            path = f"/tmp/moviroo_features/{name}_features.parquet"
            df.to_parquet(path, index=False)
            print(f"  ✓ Sauvegardé : {path}")

    return results


if __name__ == "__main__":
    run_full_pipeline()