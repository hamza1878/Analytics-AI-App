from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime
from enum import Enum


# ─── Shared ────────────────────────────────────────────────────────────────────

class MetricsModel(BaseModel):
    """
    Métriques ML flexibles.
    MAPE et RMSE sont optionnels car absents quand MLflow n'est pas disponible
    et qu'on utilise le fallback SQL heuristique.
    Tous les champs supplémentaires (source, auc_roc, mae_minutes…) sont acceptés.
    """
    MAPE:     Optional[float] = None
    RMSE:     Optional[float] = None
    # champs MLflow réels retournés selon le modèle
    MAE:      Optional[float] = None
    R2:       Optional[float] = None
    source:   Optional[str]   = None   # "sql_historical_profile", "sql_regression"…
    method:   Optional[str]   = None   # "demand_supply_ratio"…

    model_config = {"extra": "allow"}  # accepte n'importe quel champ supplémentaire


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total: int


# ─── Overview ──────────────────────────────────────────────────────────────────

class OverviewResponse(BaseModel):
    total_revenue_7d: float = Field(..., description="Total revenue last 7 days USD")
    total_rides_7d: int
    active_drivers: int
    avg_rating: float
    revenue_growth_pct: float = Field(..., description="vs prior 7d period")
    rides_growth_pct: float
    generated_at: datetime


# ─── Demand Forecast ───────────────────────────────────────────────────────────

class DemandPoint(BaseModel):
    timestamp: datetime
    demand: int
    lower: int
    upper: int
    # champs supplémentaires retournés par ml_server_v4 (optionnels)
    hour:       Optional[int]  = None
    dow:        Optional[int]  = None
    period:     Optional[str]  = None
    is_weekend: Optional[bool] = None

    model_config = {"extra": "allow"}


class DemandForecastResponse(BaseModel):
    predictions:   List[DemandPoint]
    metrics:       MetricsModel
    model_version: str
    generated_at:  datetime
    # champs supplémentaires retournés par ml_server (optionnels)
    summary:          Optional[Dict[str, Any]] = None
    training_samples: Optional[int]            = None
    data_source:      Optional[str]            = None

    model_config = {"extra": "allow"}


# ─── Revenue Forecast ──────────────────────────────────────────────────────────

class RevenuePoint(BaseModel):
    date:              str
    predicted_revenue: float
    baseline_revenue:  float
    uplift_pct:        float
    # champs optionnels
    day_name:    Optional[str]   = None
    dow_average: Optional[float] = None
    is_weekend:  Optional[bool]  = None

    model_config = {"extra": "allow"}


class RevenueForecastResponse(BaseModel):
    predictions:     List[RevenuePoint]
    total_predicted: float
    total_baseline:  float
    total_uplift_pct: float
    metrics:         MetricsModel
    generated_at:    datetime
    # champs optionnels
    daily_average_hist: Optional[float] = None
    trend_slope:        Optional[float] = None
    training_days:      Optional[int]   = None
    data_source:        Optional[str]   = None

    model_config = {"extra": "allow"}


# ─── Driver Risk ───────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class DriverRiskItem(BaseModel):
    driver_id:            int
    driver_name:          str
    churn_probability:    float = Field(..., ge=0, le=1)
    risk_level:           RiskLevel
    rating:               float
    total_rides:          int
    days_since_last_ride: int


class DriverRiskResponse(BaseModel):
    drivers:       List[DriverRiskItem]
    total_at_risk: int
    meta:          PaginationMeta
    generated_at:  datetime


# ─── Anomalies ─────────────────────────────────────────────────────────────────

class AnomalySeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class AnomalyItem(BaseModel):
    id:          str
    type:        str
    severity:    AnomalySeverity
    confidence:  float = Field(..., ge=0, le=1)
    impact:      str
    action:      str
    ride_id:     Optional[int]      = None
    driver_id:   Optional[int]      = None
    detected_at: datetime
    resolved:    bool = False
    # champs supplémentaires de ml_server_v4
    details:     Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class AnomaliesResponse(BaseModel):
    anomalies:      List[AnomalyItem]
    total:          int
    critical_count: int
    generated_at:   datetime
    # champs supplémentaires
    rides_analyzed: Optional[int]            = None
    anomaly_rate:   Optional[float]          = None
    thresholds:     Optional[Dict[str, Any]] = None
    model_version:  Optional[str]            = None
    metrics:        Optional[MetricsModel]   = None

    model_config = {"extra": "allow"}


class AnomalyActionRequest(BaseModel):
    action_type: str = Field(
        ...,
        description="freeze_account | flag_transaction | switch_psp | notify_driver"
    )
    notes: Optional[str] = None


class AnomalyActionResponse(BaseModel):
    anomaly_id:  str
    action_taken: str
    status:      str
    executed_at: datetime


# ─── Surge ─────────────────────────────────────────────────────────────────────

class SurgeZone(BaseModel):
    zone_id:              str
    zone_name:            Optional[str]   = None   # optionnel — absent dans ml_server_v4
    lat:                  Optional[float] = None
    lng:                  Optional[float] = None
    current_surge:        float
    recommended_surge:    float
    demand_score:         float
    supply_score:         float
    demand_supply_ratio:  float
    # champs supplémentaires de ml_server_v4
    historical_surge_avg: Optional[float] = None
    realtime_rides:       Optional[int]   = None
    realtime_drivers:     Optional[int]   = None
    data_source:          Optional[str]   = None

    model_config = {"extra": "allow"}


class SurgeResponse(BaseModel):
    zones:         List[SurgeZone]
    generated_at:  datetime
    model_version: Optional[str]          = None
    metrics:       Optional[MetricsModel] = None

    model_config = {"extra": "allow"}


# ─── ETA ───────────────────────────────────────────────────────────────────────

class ETAResponse(BaseModel):
    distance_km:           float
    predicted_minutes:     Optional[float] = None   # ancien champ
    predicted_trip_minutes: Optional[float] = None  # champ ml_server_v4
    confidence_interval:   Dict[str, float]
    traffic_factor:        Optional[float] = None
    model_version:         str
    generated_at:          datetime
    # champs supplémentaires de ml_server_v4
    avg_wait_minutes:       Optional[float] = None
    total_estimated_minutes: Optional[float] = None
    real_avg_speed_kmh:     Optional[float] = None
    speed_source:           Optional[str]   = None
    hour_of_day:            Optional[int]   = None
    samples_used:           Optional[int]   = None
    metrics:                Optional[MetricsModel] = None
    data_source:            Optional[str]   = None

    model_config = {"extra": "allow"}


# ─── Dashboard JSON ────────────────────────────────────────────────────────────

class KPICard(BaseModel):
    label:      str
    value:      Any
    unit:       str
    change_pct: float
    trend:      str   # "up" | "down" | "flat"


class ChartSeries(BaseModel):
    name: str
    data: List[Dict[str, Any]]


class AlertItem(BaseModel):
    id:        str
    message:   str
    severity:  str
    timestamp: datetime


class DashboardResponse(BaseModel):
    kpis:                 List[KPICard]
    demand_chart:         ChartSeries
    revenue_chart:        ChartSeries
    anomaly_alerts:       List[AlertItem]
    driver_risk_ranking:  List[DriverRiskItem]
    surge_heatmap:        List[SurgeZone]
    generated_at:         datetime