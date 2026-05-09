-- =========================================================
-- Phase A.3: Options Chain Snapshots + Per-Strike OI History
-- =========================================================
-- Idempotent: all statements use IF NOT EXISTS / ON CONFLICT DO NOTHING.

CREATE TABLE IF NOT EXISTS options_chain_snapshots (
  id                BIGSERIAL PRIMARY KEY,
  underlying_key    TEXT        NOT NULL,
  underlying_symbol TEXT        NOT NULL,
  expiry            DATE        NOT NULL,
  ts                TIMESTAMPTZ DEFAULT NOW(),
  spot_paisa        BIGINT      NOT NULL,
  pcr               NUMERIC,                    -- put-call OI ratio
  total_call_oi     BIGINT,
  total_put_oi      BIGINT,
  max_pain_strike   NUMERIC,
  atm_iv_call       NUMERIC,
  atm_iv_put        NUMERIC,
  iv_skew           NUMERIC,                    -- otm_put_iv - otm_call_iv
  raw_chain         JSONB                       -- full per-strike OI/IV/Greeks
);

CREATE INDEX IF NOT EXISTS idx_options_chain_underlying_ts
  ON options_chain_snapshots(underlying_key, ts DESC);

CREATE INDEX IF NOT EXISTS idx_options_chain_ts
  ON options_chain_snapshots(ts DESC);

-- =========================================================
-- Per-Strike OI History (tick-level granularity per poll)
-- =========================================================
CREATE TABLE IF NOT EXISTS options_strike_oi_history (
  underlying_key TEXT      NOT NULL,
  expiry         DATE      NOT NULL,
  strike         NUMERIC   NOT NULL,
  option_type    TEXT      NOT NULL CHECK (option_type IN ('CE','PE')),
  ts             TIMESTAMPTZ NOT NULL,
  oi             BIGINT,
  oi_change      BIGINT,
  iv             NUMERIC,
  delta          NUMERIC,
  gamma          NUMERIC,
  vega           NUMERIC,
  theta          NUMERIC,
  PRIMARY KEY (underlying_key, expiry, strike, option_type, ts)
);

CREATE INDEX IF NOT EXISTS idx_strike_oi_history_strike
  ON options_strike_oi_history(underlying_key, expiry, strike);

-- =========================================================
-- VERSION MARKER
-- =========================================================
INSERT INTO schema_version (version, description)
VALUES (
  'phase_a3_options_chain',
  'Phase A.3 — options chain snapshots + per-strike OI history'
)
ON CONFLICT (version) DO NOTHING;
