-- =========================================================
-- Wave 5.2 — Rejection Filter Audit Log
-- =========================================================
-- Stores every signal rejection for audit, debugging, and ML feedback.
-- Idempotent: all DDL uses IF NOT EXISTS.
-- =========================================================

CREATE TABLE IF NOT EXISTS rejection_log (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT NOW(),
  strategy_name TEXT NOT NULL,
  instrument_key TEXT NOT NULL,
  signal_type TEXT NOT NULL,
  rejection_reason TEXT NOT NULL,            -- enum: SPREAD_WIDE, EARNINGS_BLACKOUT, etc.
  details JSONB,                             -- contextual data
  signal_id BIGINT                           -- FK signals.id
);

CREATE INDEX IF NOT EXISTS idx_rejection_strategy_ts ON rejection_log(strategy_name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_rejection_reason ON rejection_log(rejection_reason, ts DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_w5_2_rejections', 'Wave 5.2 — rejection filter audit log')
ON CONFLICT (version) DO NOTHING;
