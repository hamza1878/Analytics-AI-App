"""
model_monitoring/shadow_models.py

A/B testing entre modèles en production et modèles shadow.
Compare les métriques en temps réel et recommande la promotion.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from scipy import stats


@dataclass
class ModelVariant:
    name:          str
    version:       str
    is_production: bool
    predictions:   list[float] = field(default_factory=list)
    actuals:       list[float] = field(default_factory=list)
    latencies_ms:  list[float] = field(default_factory=list)

    def record(self, prediction: float, actual: float, latency_ms: float):
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.latencies_ms.append(latency_ms)

    @property
    def n(self) -> int:
        return len(self.predictions)

    @property
    def mae(self) -> Optional[float]:
        if self.n == 0:
            return None
        return float(np.mean(np.abs(np.array(self.predictions) - np.array(self.actuals))))

    @property
    def mape(self) -> Optional[float]:
        if self.n == 0:
            return None
        a = np.array(self.actuals)
        p = np.array(self.predictions)
        return float(np.mean(np.abs((a - p) / (a + 1e-8))) * 100)

    @property
    def r2(self) -> Optional[float]:
        if self.n < 2:
            return None
        a = np.array(self.actuals)
        p = np.array(self.predictions)
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-8
        return float(1 - ss_res / ss_tot)

    @property
    def p99_latency_ms(self) -> Optional[float]:
        if not self.latencies_ms:
            return None
        return float(np.percentile(self.latencies_ms, 99))

    @property
    def mean_latency_ms(self) -> Optional[float]:
        if not self.latencies_ms:
            return None
        return float(np.mean(self.latencies_ms))


@dataclass
class ABTestResult:
    model_name:      str
    production:      ModelVariant
    shadow:          ModelVariant
    n_samples:       int
    mae_delta:       float        # shadow - prod (négatif = shadow meilleur)
    mape_delta:      float
    r2_delta:        float
    latency_delta:   float
    p_value:         float
    is_significant:  bool
    recommendation:  str          # "promote" | "keep_testing" | "reject"
    computed_at:     datetime

    def to_dict(self) -> dict:
        return {
            "model_name":     self.model_name,
            "n_samples":      self.n_samples,
            "production": {
                "version": self.production.version,
                "mae":     self.production.mae,
                "mape":    self.production.mape,
                "r2":      self.production.r2,
                "p99_ms":  self.production.p99_latency_ms,
            },
            "shadow": {
                "version": self.shadow.version,
                "mae":     self.shadow.mae,
                "mape":    self.shadow.mape,
                "r2":      self.shadow.r2,
                "p99_ms":  self.shadow.p99_latency_ms,
            },
            "deltas": {
                "mae":     round(self.mae_delta, 4),
                "mape":    round(self.mape_delta, 4),
                "r2":      round(self.r2_delta, 4),
                "latency": round(self.latency_delta, 2),
            },
            "statistics": {
                "p_value":        round(self.p_value, 4),
                "is_significant": self.is_significant,
            },
            "recommendation": self.recommendation,
            "computed_at":    self.computed_at.isoformat(),
        }


class ShadowTestManager:
    """
    Gère les tests A/B entre modèles production et shadow.

    Usage :
        mgr = ShadowTestManager("eta_estimator")
        mgr.register_production(version="v4")
        mgr.register_shadow(version="v5")

        # À chaque prédiction :
        mgr.record_production(pred=12.3, actual=11.8, latency_ms=6.1)
        mgr.record_shadow(pred=12.0, actual=11.8, latency_ms=5.8)

        # Analyse :
        result = mgr.evaluate(min_samples=200)
    """

    def __init__(self, model_name: str, alpha: float = 0.05):
        self.model_name  = model_name
        self.alpha       = alpha
        self._production: Optional[ModelVariant] = None
        self._shadow:     Optional[ModelVariant] = None

    def register_production(self, version: str) -> "ShadowTestManager":
        self._production = ModelVariant(
            name=self.model_name, version=version, is_production=True
        )
        return self

    def register_shadow(self, version: str) -> "ShadowTestManager":
        self._shadow = ModelVariant(
            name=self.model_name, version=version, is_production=False
        )
        return self

    def record_production(self, pred: float, actual: float, latency_ms: float):
        if self._production:
            self._production.record(pred, actual, latency_ms)

    def record_shadow(self, pred: float, actual: float, latency_ms: float):
        if self._shadow:
            self._shadow.record(pred, actual, latency_ms)

    def evaluate(self, min_samples: int = 100) -> Optional[ABTestResult]:
        if not self._production or not self._shadow:
            raise RuntimeError("Enregistrer production et shadow d'abord.")

        n = min(self._production.n, self._shadow.n)
        if n < min_samples:
            print(
                f"[shadow] {self.model_name} : {n}/{min_samples} samples "
                f"— test insuffisant"
            )
            return None

        prod_errors = np.abs(
            np.array(self._production.predictions[:n]) -
            np.array(self._production.actuals[:n])
        )
        shad_errors = np.abs(
            np.array(self._shadow.predictions[:n]) -
            np.array(self._shadow.actuals[:n])
        )

        # Test de Wilcoxon (non-paramétrique sur les erreurs absolues)
        _, p_value = stats.wilcoxon(prod_errors, shad_errors, alternative="greater")
        is_significant = p_value < self.alpha

        mae_delta   = (self._shadow.mae   or 0) - (self._production.mae   or 0)
        mape_delta  = (self._shadow.mape  or 0) - (self._production.mape  or 0)
        r2_delta    = (self._shadow.r2    or 0) - (self._production.r2    or 0)
        lat_delta   = (self._shadow.mean_latency_ms or 0) - (self._production.mean_latency_ms or 0)

        # Recommandation
        shadow_better = mae_delta < 0 and mape_delta < 0
        latency_ok    = lat_delta <= 2.0  # tolérance +2ms

        if is_significant and shadow_better and latency_ok:
            recommendation = "promote"
        elif n >= min_samples * 2 and not shadow_better:
            recommendation = "reject"
        else:
            recommendation = "keep_testing"

        result = ABTestResult(
            model_name=self.model_name,
            production=self._production,
            shadow=self._shadow,
            n_samples=n,
            mae_delta=mae_delta,
            mape_delta=mape_delta,
            r2_delta=r2_delta,
            latency_delta=lat_delta,
            p_value=p_value,
            is_significant=is_significant,
            recommendation=recommendation,
            computed_at=datetime.now(timezone.utc),
        )

        print(
            f"[shadow] {self.model_name} v{self._production.version}→v{self._shadow.version} | "
            f"n={n} | ΔMAE={mae_delta:+.3f} | p={p_value:.3f} | → {recommendation.upper()}"
        )
        return result


# ─────────────────────────────────────────────
# Simulation des résultats courants du dashboard
# ─────────────────────────────────────────────

CURRENT_SHADOW_RESULTS = [
    {
        "model_name":     "demand_forecast",
        "shadow_version": "v4-shadow",
        "prod_version":   "v3",
        "n_samples":      4320,
        "mape_delta":     -0.4,
        "r2_delta":       +0.01,
        "latency_delta":  +0.3,
        "recommendation": "promote",
        "p_value":        0.003,
    },
    {
        "model_name":     "surge_predictor",
        "shadow_version": "v2-shadow",
        "prod_version":   "v1",
        "n_samples":      2180,
        "mape_delta":     -1.1,
        "r2_delta":       +0.02,
        "latency_delta":  +0.1,
        "recommendation": "promote",
        "p_value":        0.012,
    },
    {
        "model_name":     "eta_estimator",
        "shadow_version": "v5",
        "prod_version":   "v4",
        "n_samples":      1560,
        "mae_delta":      -0.3,
        "r2_delta":       +0.015,
        "latency_delta":  -0.5,
        "recommendation": "keep_testing",
        "p_value":        0.08,
    },
]


def get_shadow_summary() -> list[dict]:
    """Retourne le résumé des tests shadow actifs (pour l'API)."""
    return CURRENT_SHADOW_RESULTS


if __name__ == "__main__":
    # Démo avec données synthétiques
    rng = np.random.default_rng(42)

    mgr = ShadowTestManager("eta_estimator_demo")
    mgr.register_production("v4")
    mgr.register_shadow("v5")

    actuals = rng.uniform(5, 30, 300)
    for a in actuals:
        prod_pred = a + rng.normal(0, 2.5)
        shad_pred = a + rng.normal(0, 2.0)    # shadow légèrement meilleur
        mgr.record_production(prod_pred, a, latency_ms=rng.uniform(6, 10))
        mgr.record_shadow(shad_pred,     a, latency_ms=rng.uniform(5, 9))

    result = mgr.evaluate(min_samples=200)
    if result:
        import json
        print(json.dumps(result.to_dict(), indent=2))
