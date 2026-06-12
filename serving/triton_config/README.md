# serving/triton_config/README.md
# Triton Inference Server — Configuration Moviroo ML
# =====================================================
#
# Chaque modèle a son propre répertoire dans le model_repository.
# Structure :
#   model_repository/
#   ├── demand_lstm/
#   │   ├── config.pbtxt
#   │   └── 1/model.savedmodel/   (TF SavedModel)
#   ├── surge_xgboost/
#   │   ├── config.pbtxt
#   │   └── 1/model.json          (XGBoost JSON)
#   ├── churn_rf/
#   │   ├── config.pbtxt
#   │   └── 1/model.pkl           (sklearn via python backend)
#   ├── eta_lgbm/
#   │   ├── config.pbtxt
#   │   └── 1/model.txt           (LightGBM text)
#   └── fraud_iforest/
#       ├── config.pbtxt
#       └── 1/model.pkl

# Démarrage du serveur (K8s / Docker) :
# docker run --gpus=1 --rm \
#   -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#   -v /models:/models \
#   nvcr.io/nvidia/tritonserver:24.01-py3 \
#   tritonserver --model-repository=/models
