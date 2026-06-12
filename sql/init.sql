-- ══════════════════════════════════════════════════════════════════
-- Moviroo_DB_V2 — Schema Initialization
-- ══════════════════════════════════════════════════════════════════

-- Core tables ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS drivers (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(120) NOT NULL,
    rating     NUMERIC(3,2) DEFAULT 5.0,
    total_rides INT DEFAULT 0,
    status     VARCHAR(20) DEFAULT 'active',  -- active | inactive | suspended
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rides (
    id        SERIAL PRIMARY KEY,
    driver_id INT REFERENCES drivers(id) ON DELETE SET NULL,
    user_id   INT NOT NULL,
    price     NUMERIC(10,2) NOT NULL,
    distance  NUMERIC(8,2) NOT NULL,           -- kilometers
    status    VARCHAR(20) DEFAULT 'requested', -- requested | active | completed | cancelled
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS payments (
    id             SERIAL PRIMARY KEY,
    ride_id        INT REFERENCES rides(id) ON DELETE SET NULL,
    amount         NUMERIC(10,2) NOT NULL,
    payment_method VARCHAR(30) DEFAULT 'card', -- card | cash | wallet
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ML / Monitoring tables ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS anomaly_actions (
    id           SERIAL PRIMARY KEY,
    anomaly_id   VARCHAR(36) NOT NULL,
    action_type  VARCHAR(60) NOT NULL,
    notes        TEXT,
    executed_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (anomaly_id, action_type)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id          SERIAL PRIMARY KEY,
    model_name  VARCHAR(60) NOT NULL,
    metric_name VARCHAR(60) NOT NULL,
    value       NUMERIC(12,6) NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes ────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_rides_driver_id    ON rides(driver_id);
CREATE INDEX IF NOT EXISTS idx_rides_created_at   ON rides(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rides_status       ON rides(status);
CREATE INDEX IF NOT EXISTS idx_payments_ride_id   ON payments(ride_id);
CREATE INDEX IF NOT EXISTS idx_payments_created   ON payments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_drivers_status     ON drivers(status);

-- Seed data (dev only) ───────────────────────────────────────────

INSERT INTO drivers (name, rating, total_rides, status) VALUES
    ('Ahmed Ben Salah',   4.92, 1240, 'active'),
    ('Sana Trabelsi',     4.78,  876, 'active'),
    ('Mohamed Haddad',    4.55,  320, 'active'),
    ('Fatma Bouaziz',     4.30,   85, 'inactive'),
    ('Khalil Mansouri',   3.90,   42, 'active')
ON CONFLICT DO NOTHING;
