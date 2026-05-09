-- =========================================================
-- Wave 5.1 — Confluence Score Audit Table
-- =========================================================
-- Stores every confluence evaluation for audit + ML feedback.
-- Idempotent: all DDL uses IF NOT EXISTS.
-- =========================================================

CREATE TABLE IF NOT EXISTS confluence_scores (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT NOW(),
  strategy_name TEXT NOT NULL,
  instrument_key TEXT NOT NULL,
  signal_type TEXT NOT NULL,                 -- BUY/SELL/BUY_CE/BUY_PE/SELL_CE/SELL_PE
  composite_score NUMERIC NOT NULL,          -- 0-100
  passed BOOLEAN NOT NULL,                   -- composite_score >= threshold
  threshold NUMERIC NOT NULL,
  dimension_scores JSONB,                    -- per-dimension breakdown
  matched_conditions JSONB,                  -- which conditions contributed
  failed_conditions JSONB,
  signal_id BIGINT                           -- FK to signals.id (nullable — not all scores link to a signal)
);

CREATE INDEX IF NOT EXISTS idx_confluence_strategy_ts ON confluence_scores(strategy_name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_confluence_passed ON confluence_scores(passed, ts DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_w5_1_confluence', 'Wave 5.1 — confluence score audit')
ON CONFLICT (version) DO NOTHING;
