-- =========================================================
-- Phase E.2 — Pre-trade margin checks audit table
-- =========================================================

CREATE TABLE IF NOT EXISTS margin_checks (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT NOW(),
  client_order_id TEXT,
  instrument_key TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity INT NOT NULL,
  required_margin_paisa BIGINT,
  available_margin_paisa BIGINT,
  approved BOOLEAN NOT NULL,
  rejection_reason TEXT,
  raw_response JSONB
);
CREATE INDEX IF NOT EXISTS idx_margin_checks_ts ON margin_checks(ts DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_e2_validator', 'Phase E.2 — pre-trade margin checks audit')
ON CONFLICT (version) DO NOTHING;
