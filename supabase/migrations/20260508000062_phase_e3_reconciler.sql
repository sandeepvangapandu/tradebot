-- =========================================================
-- Phase E.3 — Position Reconciler
-- =========================================================
-- Stores audit trails for every reconciliation cycle:
--   reconciliation_runs  — one row per 30s poll cycle
--   reconciliation_log   — one row per event detected within a cycle
-- Idempotent: all DDL uses IF NOT EXISTS / ON CONFLICT DO NOTHING.
-- =========================================================

CREATE TABLE IF NOT EXISTS reconciliation_runs (
  cycle_id              BIGSERIAL PRIMARY KEY,
  started_at            TIMESTAMPTZ DEFAULT NOW(),
  ended_at              TIMESTAMPTZ,
  total_local_positions INT,
  total_broker_positions INT,
  events_count          INT         DEFAULT 0,
  status                TEXT        NOT NULL DEFAULT 'RUNNING'  -- RUNNING/COMPLETE/FAILED
);

CREATE TABLE IF NOT EXISTS reconciliation_log (
  id            BIGSERIAL PRIMARY KEY,
  ts            TIMESTAMPTZ DEFAULT NOW(),
  cycle_id      BIGINT      NOT NULL,                           -- Same cycle_id for all events from one reconcile run
  event_type    TEXT        NOT NULL,                           -- MISSED_FILL/POSITION_DRIFT/QTY_MISMATCH/PRICE_MISMATCH/BROKER_EXIT/REMOTE_NEW/CLEAN
  instrument_key TEXT       NOT NULL,
  local_state   JSONB,
  broker_state  JSONB,
  action_taken  TEXT,                                           -- INSERTED_FILL/UPDATED_POSITION/FLAGGED/NONE
  details       JSONB
);

CREATE INDEX IF NOT EXISTS idx_recon_cycle ON reconciliation_log(cycle_id, ts);
CREATE INDEX IF NOT EXISTS idx_recon_type  ON reconciliation_log(event_type, ts DESC);

-- =========================================================
-- VERSION MARKER
-- =========================================================
INSERT INTO schema_version (version, description)
VALUES ('phase_e3_reconciler', 'Phase E.3 — broker vs local position reconciliation')
ON CONFLICT (version) DO NOTHING;
