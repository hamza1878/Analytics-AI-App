"""
Moviroo ML — Central Configuration
"""
import os
from dataclasses import dataclass


@dataclass
class DBConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "Moviroo_DB_V2")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "1878")

    @property
    def url(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class MLConfig:
    # MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_URI", "http://localhost:5000")
    mlflow_experiment: str = "moviroo-ml"

    # Triton inference server
    triton_host: str = os.getenv("TRITON_HOST", "localhost")
    triton_port: int = int(os.getenv("TRITON_PORT", "8000"))

    # Training windows
    demand_lookback_days: int = 90
    surge_lookback_days: int = 30
    churn_lookback_days: int = 60
    eta_lookback_days: int = 45

    # Drift thresholds
    psi_warning: float = 0.05
    psi_critical: float = 0.10

    # Model artifact paths
    model_dir: str = os.getenv("MODEL_DIR", "/models")


DB = DBConfig()
ML = MLConfig()
