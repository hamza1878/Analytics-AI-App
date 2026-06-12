"""
feature_service.py
Extracts features from PostgreSQL for each ML model.
All queries are async and parameterized.
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from decimal import Decimal
import datetime
import uuid

import logging

logger = logging.getLogger(__name__)


def _sanitize(row: dict) -> dict:
    """Convert non-JSON-serializable types from PostgreSQL rows."""
    result = {}
    for k, v in row.items():
        if isinstance(v, Decimal):
            result[k] = float(v)
        elif isinstance(v, (datetime.datetime, datetime.date)):
            result[k] = v.isoformat()
        elif isinstance(v, datetime.timedelta):
            result[k] = v.total_seconds()
        else:
            result[k] = v
    return result


# ── Demand Features ────────────────────────────────────────────────────────────

async def get_demand_features(db: AsyncSession, hours: int = 168) -> List[Dict]:
    sql = text("""
        SELECT
            DATE_TRUNC('hour', created_at)  AS hour_bucket,
            COUNT(*)                         AS ride_count
        FROM rides
        WHERE created_at >= NOW() - make_interval(hours => :hours)
          AND status = 'COMPLETED'::ride_status_enum
        GROUP BY 1
        ORDER BY 1
    """)
    result = await db.execute(sql, {"hours": hours})
    return [{"timestamp": str(r.hour_bucket), "ride_count": r.ride_count} for r in result]


# ── Revenue Features ───────────────────────────────────────────────────────────

async def get_revenue_features(db: AsyncSession, days: int = 30) -> List[Dict]:
    sql = text("""
        SELECT
            DATE_TRUNC('day', r.created_at)   AS day,
            SUM(r.price_final)                 AS total_revenue,
            AVG(r.price_final)                 AS avg_price,
            AVG(r.distance_km_real)            AS avg_distance,
            COUNT(*)                           AS ride_count,
            EXTRACT(DOW FROM r.created_at)     AS day_of_week,
            EXTRACT(HOUR FROM r.created_at)    AS hour_of_day
        FROM rides r
        WHERE r.created_at >= NOW() - make_interval(days => :days)
          AND r.status = 'COMPLETED'::ride_status_enum
        GROUP BY 1, 6, 7
        ORDER BY 1
    """)
    result = await db.execute(sql, {"days": days})
    return [_sanitize(dict(r._mapping)) for r in result]


# ── Driver Churn Features ──────────────────────────────────────────────────────

async def get_driver_churn_features(db: AsyncSession) -> List[Dict]:
    sql = text("""
        SELECT
            d.id                                                          AS driver_id,
            d.name,
            d.rating_average                                              AS rating,
            d.total_rides,
            d.availability_status                                         AS status,
            COUNT(r.id)                                                   AS rides_last_30d,
            COALESCE(SUM(r.price_final), 0)                               AS revenue_last_30d,
            COALESCE(AVG(r.price_final), 0)                               AS avg_fare,
            EXTRACT(EPOCH FROM (NOW() - MAX(r.created_at)))/86400         AS days_since_last_ride,
            COUNT(r.id) FILTER (WHERE r.status = 'CANCELLED'::ride_status_enum) * 1.0
                / NULLIF(COUNT(r.id), 0)                                  AS cancellation_rate
        FROM drivers d
        LEFT JOIN rides r ON r.driver_id = d.id
            AND r.created_at >= NOW() - make_interval(days => 30)
        GROUP BY d.id, d.name, d.rating_average, d.total_rides, d.availability_status
        ORDER BY days_since_last_ride DESC NULLS LAST
    """)
    result = await db.execute(sql)
    return [_sanitize(dict(r._mapping)) for r in result]


# ── Anomaly / Fraud Features ───────────────────────────────────────────────────

# async def get_payment_features(db: AsyncSession, hours: int = 24) -> List[Dict]:
#     sql = text("""
#         SELECT
#             p.id                                             AS payment_id,
#             p.ride_id,
#             p.amount,
#             p.payment_method,
#             r.price_final                                    AS expected_price,
#             r.distance_km_real                               AS distance,
#             r.driver_id,
#             r.passenger_id,
#             p.amount - r.price_final                         AS amount_delta,
#             EXTRACT(HOUR FROM p.created_at)                  AS hour_of_day,
#             COUNT(*) OVER (
#                 PARTITION BY r.passenger_id
#                 ORDER BY p.created_at
#                 RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
#             )                                                AS payments_last_hour,
#             p.created_at
#         FROM payments p
#         JOIN rides r ON r.id = p.ride_id
#         WHERE p.created_at >= NOW() - make_interval(hours => :hours)
#         ORDER BY p.created_at DESC
#     """)
#     result = await db.execute(sql, {"hours": hours})
#     return [_sanitize(dict(r._mapping)) for r in result]
async def get_payment_features(db: AsyncSession, hours: int = 24) -> List[Dict]:
    sql = text("""
        SELECT
            r.id                                                          AS payment_id,
            r.id                                                          AS ride_id,
            r.price_final                                                 AS amount,
            'unknown'                                                     AS payment_method,
            r.price_estimate                                              AS expected_price,
            r.distance_km_real                                            AS distance,
            r.driver_id,
            r.passenger_id,
            r.price_final - r.price_estimate                             AS amount_delta,
            EXTRACT(HOUR FROM r.created_at)                              AS hour_of_day,
            COUNT(*) OVER (
                PARTITION BY r.passenger_id
                ORDER BY r.created_at
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
            )                                                             AS payments_last_hour,
            r.surge_multiplier,
            r.created_at
        FROM rides r
        WHERE r.created_at >= NOW() - make_interval(hours => :hours)
          AND r.status = 'COMPLETED'::ride_status_enum
          AND r.price_final IS NOT NULL
        ORDER BY r.created_at DESC
    """)
    result = await db.execute(sql, {"hours": hours})
    return [_sanitize(dict(r._mapping)) for r in result]

# ── Surge Features ─────────────────────────────────────────────────────────────

async def get_surge_features(db: AsyncSession) -> List[Dict]:
    sql = text("""
        WITH demand AS (
            SELECT
                'zone_' || (ROW_NUMBER() OVER ())::text  AS zone_id,
                COUNT(*)                                  AS active_requests
            FROM rides
            WHERE status = 'REQUESTED'::ride_status_enum
              AND created_at >= NOW() - INTERVAL '15 minutes'
            GROUP BY DATE_TRUNC('minute', created_at)
        ),
        supply AS (
            SELECT COUNT(*) AS available_drivers
            FROM drivers
            WHERE availability_status IN ('online', 'on_trip')
        )
        SELECT
            'zone_default'                      AS zone_id,
            COALESCE(SUM(d.active_requests), 0) AS demand,
            s.available_drivers                 AS supply
        FROM supply s
        LEFT JOIN demand d ON TRUE
        GROUP BY s.available_drivers
    """)
    result = await db.execute(sql)
    return [_sanitize(dict(r._mapping)) for r in result]


# ── Overview KPIs ──────────────────────────────────────────────────────────────

async def get_overview_kpis(db: AsyncSession) -> Dict[str, Any]:
    sql = text("""
    WITH current_period AS (
        SELECT
            COALESCE(SUM(r.price_final), 0) AS revenue,
            COUNT(r.id) AS rides
        FROM rides r
        WHERE r.created_at >= NOW() - INTERVAL '7 days'
          AND r.status = 'COMPLETED'::ride_status_enum
    ),
    prior_period AS (
        SELECT
            COALESCE(SUM(r.price_final), 0) AS revenue,
            COUNT(r.id) AS rides
        FROM rides r
        WHERE r.created_at BETWEEN NOW() - INTERVAL '14 days'
                                AND NOW() - INTERVAL '7 days'
          AND r.status = 'COMPLETED'::ride_status_enum
    ),
    drivers_kpi AS (
        SELECT
            COALESCE(
                COUNT(*) FILTER (
                    WHERE availability_status IN ('online', 'on_trip')
                ), 0
            ) AS active_drivers,
            COALESCE(ROUND(AVG(rating_average)::numeric, 2), 0) AS avg_rating
        FROM drivers
    )
    SELECT
        COALESCE(cp.revenue, 0) AS total_revenue_7d,
        COALESCE(cp.rides, 0) AS total_rides_7d,
        dk.active_drivers,
        dk.avg_rating,
        COALESCE(
            ROUND(((cp.revenue - pp.revenue) /
            NULLIF(pp.revenue, 0) * 100)::numeric, 2),
        0) AS revenue_growth_pct,
        COALESCE(
            ROUND(((cp.rides - pp.rides) /
            NULLIF(pp.rides, 0) * 100)::numeric, 2),
        0) AS rides_growth_pct
    FROM current_period cp, prior_period pp, drivers_kpi dk
    """)
    try:
        result = await db.execute(sql)
        row = result.fetchone()
        return _sanitize(dict(row._mapping)) if row else {}
    except Exception as e:
        logger.error(f"❌ KPI ERROR: {e}")
        raise
    # ── Overview KPIs ──────────────────────────────────────────────────────────────
# async def get_overview_kpis(db: AsyncSession) -> Dict[str, Any]:
#     sql = text("""
#     WITH current_period AS (
#         SELECT
#             COALESCE(SUM(r.price), 0) AS revenue,
#             COUNT(r.id)               AS rides
#         FROM rides r
#         WHERE r.created_at >= NOW() - INTERVAL '7 days'
#           AND r.status = 'completed'
#     ),
#     prior_period AS (
#         SELECT
#             COALESCE(SUM(r.price), 0) AS revenue,
#             COUNT(r.id)               AS rides
#         FROM rides r
#         WHERE r.created_at BETWEEN NOW() - INTERVAL '14 days'
#                                 AND NOW() - INTERVAL '7 days'
#           AND r.status = 'completed'
#     ),
#     drivers_kpi AS (
#         SELECT
#             COUNT(*) FILTER (WHERE status = 'active') AS active_drivers,
#             ROUND(AVG(rating)::numeric, 2)            AS avg_rating
#         FROM drivers
#     )
#     SELECT
#         cp.revenue  AS total_revenue_7d,
#         cp.rides    AS total_rides_7d,
#         dk.active_drivers,
#         dk.avg_rating,
#         ROUND(((cp.revenue - pp.revenue) / NULLIF(pp.revenue, 0) * 100)::numeric, 2) AS revenue_growth_pct,
#         ROUND(((cp.rides   - pp.rides)   / NULLIF(pp.rides,   0) * 100)::numeric, 2) AS rides_growth_pct
#     FROM current_period cp, prior_period pp, drivers_kpi dk
#     """)
#     try:
#         result = await db.execute(sql)
#         row = result.fetchone()
#         return dict(row._mapping) if row else {}
#     except Exception as e:
#         logger.error(f"KPI query failed: {e}")
#         raise

# async def get_overview_kpis(db: AsyncSession) -> Dict[str, Any]:
#     sql = text("""
#     WITH current_period AS (
#         SELECT
#             COALESCE(SUM(r.price), 0) AS revenue,
#             COUNT(r.id)               AS rides
#         FROM rides r
#         WHERE r.created_at >= NOW() - INTERVAL '7 days'
#           AND r.status = 'completed'
#     ),
#     prior_period AS (
#         SELECT
#             COALESCE(SUM(r.price), 0) AS revenue,
#             COUNT(r.id)               AS rides
#         FROM rides r
#         WHERE r.created_at BETWEEN NOW() - INTERVAL '14 days'
#                                 AND NOW() - INTERVAL '7 days'
#           AND r.status = 'completed'
#     ),
#     drivers_kpi AS (
#         SELECT
#             COUNT(*) FILTER (WHERE status = 'active') AS active_drivers,
#             ROUND(AVG(rating)::numeric, 2)            AS avg_rating
#         FROM drivers
#     )
#     SELECT
#         cp.revenue  AS total_revenue_7d,
#         cp.rides    AS total_rides_7d,
#         dk.active_drivers,
#         dk.avg_rating,
#         ROUND(((cp.revenue - pp.revenue) / NULLIF(pp.revenue, 0) * 100)::numeric, 2) AS revenue_growth_pct,
#         ROUND(((cp.rides   - pp.rides)   / NULLIF(pp.rides,   0) * 100)::numeric, 2) AS rides_growth_pct
#     FROM current_period cp, prior_period pp, drivers_kpi dk
#     """)
#     try:
#         result = await db.execute(sql)
#         row = result.fetchone()
#         return _sanitize(dict(row._mapping)) if row else {}
#     except Exception as e:
#         logger.error(f"KPI query failed: {e}")
#         raise



# async def get_overview_kpis(db: AsyncSession) -> Dict[str, Any]:
#     sql = text("""
#     WITH current_period AS (
#         SELECT
#             COALESCE(SUM(r.price), 0) AS revenue,
#             COUNT(r.id)               AS rides
#         FROM rides r
#         WHERE r.created_at >= NOW() - INTERVAL '7 days'
#           AND r.status = 'completed'
#     ),
#     prior_period AS (
#         SELECT
#             COALESCE(SUM(r.price), 0) AS revenue,
#             COUNT(r.id)               AS rides
#         FROM rides r
#         WHERE r.created_at BETWEEN NOW() - INTERVAL '14 days'
#                                 AND NOW() - INTERVAL '7 days'
#           AND r.status = 'completed'
#     ),
#     drivers_kpi AS (
#         SELECT
#             COUNT(*) FILTER (WHERE status = 'active') AS active_drivers,
#             ROUND(AVG(rating)::numeric, 2)            AS avg_rating
#         FROM drivers
#     )
#     SELECT
#         cp.revenue  AS total_revenue_7d,
#         cp.rides    AS total_rides_7d,
#         dk.active_drivers,
#         dk.avg_rating,
#         ROUND(((cp.revenue - pp.revenue) / NULLIF(pp.revenue, 0) * 100)::numeric, 2) AS revenue_growth_pct,
#         ROUND(((cp.rides   - pp.rides)   / NULLIF(pp.rides,   0) * 100)::numeric, 2) AS rides_growth_pct
#     FROM current_period cp, prior_period pp, drivers_kpi dk
#     """)
#     try:
#         result = await db.execute(sql)
#         row = result.fetchone()
#         return dict(row._mapping) if row else {}
#     except Exception as e:
#         logger.error(f"KPI query failed: {e}")
#         raise

