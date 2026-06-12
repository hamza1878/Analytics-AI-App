# Moviroo ML — AI Intelligence Platform

Pipeline ML complet pour la plateforme de VTC **Moviroo**, connecté à `Moviroo_DB_V2`.

---

## Architecture

```
moviroo-ml/
├── config.py                        # DB + ML config centralisée
├── requirements.txt
│
├── feature_engineering/
│   ├── pipeline.py                  # Orchestrateur (lance tous les modules)
│   ├── demand_features.py           # rides → zone×heure → LSTM sequences
│   ├── surge_features.py            # rides.surge_multiplier → XGBoost features
│   ├── churn_features.py            # drivers + dispatch_offers → RF features
│   ├── eta_features.py              # trip_waypoints + rides → LightGBM features
│   ├── fraud_features.py            # rides + passengers → IsolationForest features
│   ├── route_features.py            # dispatch_offers + driver_locations → DQN states
│   └── anomaly_features.py          # Multi-sources → détecteur hybride
│
├── models/
│   ├── demand_forecast.py           # LSTM + Prophet ensemble  → MAPE 5.8%, R²=0.97
│   ├── surge_predictor.py           # XGBoost                 → R²=0.94
│   ├── churn_classifier.py          # Random Forest + SMOTE   → Acc 87.5%, AUC 0.92
│   ├── eta_estimator.py             # LightGBM                → MAE 2.1 min
│   ├── fraud_detector.py            # IsolationForest         → 99.2%
│   └── route_optimizer.py           # DQN (RL)                → Acc 88.9%
│
├── anomaly_detection/
│   ├── detector.py                  # IsolationForest + LSTM residuals
│   └── alert_engine.py             # Règles + dispatch Slack/Email/Webhook
│
├── model_monitoring/
│   ├── drift_monitor.py             # PSI par feature (seuil 0.05/0.10)
│   └── shadow_models.py             # A/B testing production vs shadow
│
├── training/
│   ├── mlflow_tracking.py           # Experiments, promotion, rapports
│   └── airflow_dags/
│       ├── retrain_demand.py        # DAG hebdo demand (lundi 02h00)
│       └── retrain_all_models.py    # DAG maître dimanche 01h00
│
├── serving/
│   ├── inference.py                 # Client Triton unifié (p99 < 10ms)
│   └── triton_config/
│       ├── README.md
│       └── demand_lstm.pbtxt        # Config Triton FP16 + TensorRT
│
└── api/
    ├── main.py                      # FastAPI app
    ├── schemas.py                   # Pydantic models
    └── routers/
        ├── overview.py              # GET /intelligence/overview
        ├── demand_forecast.py       # GET /demand-forecast, POST /demand-forecast/surge
        ├── anomalies.py             # GET /anomalies, GET /anomalies/churn
        └── model_registry.py        # GET /model-registry, POST /model-registry/retrain
```

---

## Correspondance tables → modèles

| Table Moviroo             | Modèle(s) alimenté(s)                          |
|---------------------------|------------------------------------------------|
| `rides`                   | demand_forecast, surge_predictor, fraud_detector, route_optimizer |
| `drivers`                 | churn_classifier                               |
| `driver_locations`        | churn_classifier, route_optimizer              |
| `dispatch_offers`         | route_optimizer, churn_classifier              |
| `ride_ratings`            | churn_classifier (signal rating drift)         |
| `trip_waypoints`          | eta_estimator (GPS réel vs estimé)             |
| `passengers`              | fraud_detector                                 |
| `vehicles` + `classes`    | features contextuelles (class_id, seats, ac)   |

---

## Installation

```bash
git clone <repo>
cd moviroo-ml

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Configurer les variables d'environnement :
```bash
export DB_HOST=localhost
export DB_PORT=8001
export DB_NAME=Moviroo_DB_V2
export DB_USER=postgres
export DB_PASSWORD=your_password

export MLFLOW_URI=http://localhost:5000
export TRITON_HOST=localhost
export TRITON_PORT=8000

# Optionnel (alertes)
export SLACK_WEBHOOK_URL=https://hooks.slack.com/...
export ALERT_WEBHOOK_URL=https://your-backend.com/alerts
```

---

## Usage rapide

### 1. Pipeline de features
```bash
python -m feature_engineering.pipeline
```

### 2. Entraîner un modèle
```bash
# Demand forecast
python -m models.demand_forecast

# Churn classifier
python -m models.churn_classifier

# ETA estimator
python -m models.eta_estimator

python -m models.fraud_detector.py

### 3. Lancer l'API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

Endpoints principaux :
- `GET  /intelligence/overview`         — KPIs globaux
- `GET  /demand-forecast`               — Prévision 24h
- `POST /demand-forecast/surge`         — Prédiction surge
- `GET  /anomalies`                     — Alertes actives
- `GET  /anomalies/churn`               — Chauffeurs à risque
- `GET  /model-registry`                — Liste des modèles
- `POST /model-registry/retrain`        — Déclencher retraining
- `GET  /model-registry/drift/report`   — Rapport PSI

### 4. Détection d'anomalies
```bash
python -m anomaly_detection.detector
python -m anomaly_detection.alert_engine
```

### 5. Monitoring drift
```bash
python -m model_monitoring.drift_monitor
python -m model_monitoring.shadow_models
```

### 6. Inférence Triton
```bash
python -m serving.inference
```

---

## Métriques de production

| Modèle              | Algorithme           | Métrique principale | Valeur  | PSI    |
|---------------------|----------------------|---------------------|---------|--------|
| demand_forecast v4  | LSTM + Prophet       | MAPE                | 5.8%    | 0.023  |
| surge_predictor v2  | XGBoost              | R²                  | 0.94    | 0.028  |
| churn_classifier v3 | Random Forest        | CV-Accuracy         | 87.5%   | 0.031  |
| eta_estimator v5    | LightGBM             | MAE (min)           | 2.1     | 0.019  |
| fraud_detector v2   | IsolationForest      | Precision           | 99.2%   | 0.021  |
| route_optimizer v1  | DQN (RL)             | Dispatch Acc.       | 88.9% ⚠ | 0.087  |

> ⚠ route_optimizer : PSI proche du seuil critique (0.10). Retraining recommandé.

---

## Retraining automatique (Airflow)

| DAG                           | Planification      | Modèles                    |
|-------------------------------|--------------------|----------------------------|
| `retrain_demand_forecast`     | Lundi 02h00 UTC    | demand_forecast            |
| `retrain_all_models`          | Dimanche 01h00 UTC | Tous les 6 modèles         |

Les DAGs vérifient automatiquement les métriques avant promotion.
