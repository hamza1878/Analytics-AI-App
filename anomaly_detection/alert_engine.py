"""
anomaly_detection/alert_engine.py

Moteur d'alertes : reçoit les anomalies du détecteur,
applique des règles de déduplication et de throttling,
puis dispatche via webhook / email / Slack.
"""
import json
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum


class Channel(str, Enum):
    SLACK   = "slack"
    EMAIL   = "email"
    WEBHOOK = "webhook"
    LOG     = "log"          # fallback


@dataclass
class AlertRule:
    alert_type:       str
    min_severity:     str           # "LOW" | "MEDIUM" | "HIGH"
    channels:         list[Channel]
    cooldown_minutes: int = 30      # pas de doublon pendant X minutes
    min_score:        float = 0.0   # score minimum pour déclencher


@dataclass
class SentAlert:
    fingerprint:  str
    sent_at:      datetime
    alert_type:   str
    severity:     str


# ─────────────────────────────────────────────
# Règles par défaut
# ─────────────────────────────────────────────

DEFAULT_RULES: list[AlertRule] = [
    AlertRule(
        alert_type="PAYMENT_SPIKE",
        min_severity="MEDIUM",
        channels=[Channel.SLACK, Channel.WEBHOOK],
        cooldown_minutes=15,
        min_score=0.5,
    ),
    AlertRule(
        alert_type="SURGE_MISMATCH",
        min_severity="MEDIUM",
        channels=[Channel.SLACK],
        cooldown_minutes=30,
        min_score=0.4,
    ),
    AlertRule(
        alert_type="RATING_DRIFT",
        min_severity="LOW",
        channels=[Channel.EMAIL],
        cooldown_minutes=60,
        min_score=0.2,
    ),
    AlertRule(
        alert_type="DEMAND_FORECAST_RESIDUAL",
        min_severity="HIGH",
        channels=[Channel.SLACK, Channel.WEBHOOK],
        cooldown_minutes=20,
        min_score=0.6,
    ),
]

SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


# ─────────────────────────────────────────────
# Alert Engine
# ─────────────────────────────────────────────

class AlertEngine:
    """
    Reçoit des anomalies brutes, applique les règles et envoie les alertes.

    Usage :
        engine = AlertEngine(rules=DEFAULT_RULES)
        engine.process(anomalies_list)
    """

    def __init__(
        self,
        rules: list[AlertRule] = None,
        slack_webhook_url: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ):
        self.rules = rules or DEFAULT_RULES
        self.slack_webhook_url = slack_webhook_url
        self.webhook_url = webhook_url
        self._sent_cache: dict[str, SentAlert] = {}

    def _fingerprint(self, anomaly: dict) -> str:
        """Hash stable pour identifier un doublon d'alerte."""
        key = f"{anomaly['alert_type']}:{anomaly['affected_entity']}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _is_throttled(self, fingerprint: str, cooldown_min: int) -> bool:
        if fingerprint not in self._sent_cache:
            return False
        sent = self._sent_cache[fingerprint]
        elapsed = datetime.now(timezone.utc) - sent.sent_at
        return elapsed < timedelta(minutes=cooldown_min)

    def _find_rule(self, alert_type: str) -> Optional[AlertRule]:
        return next((r for r in self.rules if r.alert_type == alert_type), None)

    def _should_send(self, anomaly: dict, rule: AlertRule) -> bool:
        severity_ok = (
            SEVERITY_ORDER.get(anomaly["severity"], 0) >=
            SEVERITY_ORDER.get(rule.min_severity, 0)
        )
        score_ok = anomaly.get("score", 0) >= rule.min_score
        fp = self._fingerprint(anomaly)
        not_throttled = not self._is_throttled(fp, rule.cooldown_minutes)
        return severity_ok and score_ok and not_throttled

    def _send_slack(self, anomaly: dict) -> bool:
        """Envoie un message Slack via webhook."""
        if not self.slack_webhook_url:
            print(f"    [slack] (no webhook configured) → {anomaly['description'][:60]}")
            return True
        try:
            import urllib.request
            severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                anomaly["severity"], "⚪"
            )
            payload = {
                "text": (
                    f"{severity_emoji} *[{anomaly['severity']}] {anomaly['alert_type']}*\n"
                    f"{anomaly['description']}\n"
                    f"Score: `{anomaly['score']:.2f}` | Entity: `{anomaly['affected_entity']}`"
                )
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                self.slack_webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception as e:
            print(f"    [slack] Erreur envoi : {e}")
            return False

    def _send_webhook(self, anomaly: dict) -> bool:
        """POST l'anomalie vers un webhook générique (ex: n8n, Zapier, backend)."""
        if not self.webhook_url:
            print(f"    [webhook] (no URL configured) → {anomaly['alert_type']}")
            return True
        try:
            import urllib.request
            data = json.dumps(anomaly, default=str).encode()
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception as e:
            print(f"    [webhook] Erreur : {e}")
            return False

    def _send_email(self, anomaly: dict) -> bool:
        """Log email (implémentation SMTP à brancher en prod)."""
        print(
            f"    [email] SUJET: [{anomaly['severity']}] {anomaly['alert_type']} | "
            f"CORPS: {anomaly['description'][:80]}"
        )
        return True

    def _dispatch(self, anomaly: dict, channels: list[Channel]) -> dict[str, bool]:
        results = {}
        for channel in channels:
            if channel == Channel.SLACK:
                results["slack"] = self._send_slack(anomaly)
            elif channel == Channel.WEBHOOK:
                results["webhook"] = self._send_webhook(anomaly)
            elif channel == Channel.EMAIL:
                results["email"] = self._send_email(anomaly)
            else:
                print(f"    [log] {anomaly['severity']} | {anomaly['description'][:70]}")
                results["log"] = True
        return results

    def process(self, anomalies: list[dict]) -> dict:
        """
        Traite une liste d'anomalies et dispatche les alertes selon les règles.

        Returns:
            {
                "processed":  int,
                "sent":       int,
                "throttled":  int,
                "no_rule":    int,
                "dispatched": list[dict]
            }
        """
        stats = {"processed": 0, "sent": 0, "throttled": 0, "no_rule": 0}
        dispatched = []

        for anomaly in anomalies:
            stats["processed"] += 1
            rule = self._find_rule(anomaly["alert_type"])

            if rule is None:
                stats["no_rule"] += 1
                print(f"  [alert_engine] Pas de règle pour {anomaly['alert_type']}")
                continue

            fp = self._fingerprint(anomaly)

            if not self._should_send(anomaly, rule):
                stats["throttled"] += 1
                print(f"  [alert_engine] Throttled : {anomaly['alert_type']} ({fp})")
                continue

            print(f"  [alert_engine] → Dispatch {anomaly['severity']} {anomaly['alert_type']}")
            results = self._dispatch(anomaly, rule.channels)

            self._sent_cache[fp] = SentAlert(
                fingerprint=fp,
                sent_at=datetime.now(timezone.utc),
                alert_type=anomaly["alert_type"],
                severity=anomaly["severity"],
            )

            stats["sent"] += 1
            dispatched.append({
                "anomaly":  anomaly["alert_type"],
                "entity":   anomaly["affected_entity"],
                "channels": results,
            })

        print(
            f"\n[alert_engine] Résumé : {stats['sent']} envoyées | "
            f"{stats['throttled']} throttled | {stats['no_rule']} sans règle"
        )
        return {**stats, "dispatched": dispatched}

    def clear_cache(self):
        """Vide le cache de déduplication (utile pour les tests)."""
        self._sent_cache.clear()


# ─────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────

def run_pipeline(
    slack_webhook_url: Optional[str] = None,
    webhook_url: Optional[str] = None,
) -> dict:
    """
    Pipeline complet :
      1. Détecte les anomalies
      2. Dispatche les alertes
    """
    from anomaly_detection.detector import run_all_detectors

    print("[alert_pipeline] Détection des anomalies…")
    anomalies = run_all_detectors()

    engine = AlertEngine(
        rules=DEFAULT_RULES,
        slack_webhook_url=slack_webhook_url,
        webhook_url=webhook_url,
    )

    print(f"[alert_pipeline] {len(anomalies)} anomalies → moteur d'alertes")
    return engine.process(anomalies)


if __name__ == "__main__":
    import os
    result = run_pipeline(
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
    )
    print(json.dumps(result, indent=2, default=str))
