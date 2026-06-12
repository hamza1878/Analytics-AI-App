"""
kafka_inference_consumer.py
Real-time inference pipeline:
  Kafka topic `rides.created` → feature extraction → ML inference → Redis cache update

Run: python kafka_inference_consumer.py
"""
import kafka
from kafka import KafkaConsumer, KafkaProducer
import httpx
import redis
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

KAFKA_BOOTSTRAP = "kafka:9092"
ML_SERVICE_URL  = "http://ml-service:8001"
REDIS_URL       = "redis://redis:6379/0"

r = redis.from_url(REDIS_URL, decode_responses=True)


def process_ride_event(event: dict):
    """On each new ride: update demand cache + get fresh surge."""
    try:
        # 1. Increment rolling demand counter (15-min window)
        window_key = f"demand:window:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')[:-1]}0"
        r.incr(window_key)
        r.expire(window_key, 900)  # 15 min TTL

        # 2. Invalidate stale surge cache for the zone
        zone = event.get("zone_id", "zone_default")
        r.delete("moviroo:surge")

        logger.info(f"Processed ride event: ride_id={event.get('ride_id')} zone={zone}")
    except Exception as e:
        logger.error(f"Error processing ride event: {e}")


def process_payment_event(event: dict):
    """On each payment: run real-time fraud scoring."""
    try:
        with httpx.Client(base_url=ML_SERVICE_URL, timeout=5) as client:
            resp = client.post("/predict/anomalies", json={"payments": [event]})
            resp.raise_for_status()
            result = resp.json()

        anomalies = result.get("anomalies", [])
        if anomalies:
            for a in anomalies:
                # Push to anomaly stream
                r.lpush("stream:anomalies", json.dumps(a))
                r.ltrim("stream:anomalies", 0, 999)  # keep last 1000
                logger.warning(f"ANOMALY DETECTED: {a['type']} confidence={a['confidence']}")

    except Exception as e:
        logger.error(f"Error processing payment event: {e}")


def main():
    consumer = KafkaConsumer(
        "rides.created",
        "payments.completed",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="moviroo-inference-group",
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    logger.info("Kafka consumer started. Listening for events...")

    for msg in consumer:
        topic = msg.topic
        event = msg.value

        if topic == "rides.created":
            process_ride_event(event)
        elif topic == "payments.completed":
            process_payment_event(event)


if __name__ == "__main__":
    main()
