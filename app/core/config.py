from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:1878@localhost:5432/Moviroo_DB_V2"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 300  # seconds

    # ML Service
    ML_SERVICE_URL: str = "http://ml-service:8001"
    ML_SERVICE_TIMEOUT: int = 10

    # Kafka
    KAFKA_BOOTSTRAP: str = "kafka:9092"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"

    # Security
    SECRET_KEY: str = "change-me-in-production"
    ALLOWED_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()
