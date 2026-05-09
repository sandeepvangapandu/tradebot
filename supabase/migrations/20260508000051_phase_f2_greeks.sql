-- =========================================================
-- Phase F.2 — Portfolio Greeks Snapshot
-- =========================================================
-- Aggregates delta, gamma, vega, theta across all open option
-- positions for portfolio-level risk control.
-- Idempotent: all DDL uses IF NOT EXISTS / ON CONFLICT DO NOTHING.
-- =========================================================

CREATE TABLE IF NOT EXISTS portfolio_greeks_snapshot (
  ts                   TIMESTAMPTZ DEFAULT NOW(),
  trade_date           DATE        NOT NULL,
  total_delta          NUMERIC     NOT NULL,                 -- net portfolio delta (in equiv. shares)
  total_gamma          NUMERIC     NOT NULL,
  total_vega           NUMERIC     NOT NULL,                 -- INR per 1% vol move
  total_theta          NUMERIC     NOT NULL,                 -- INR daily decay
  total_rho            NUMERIC,
  position_count       INT,
  underlying_breakdown JSONB,                               -- {underlying_symbol: {delta, gamma, vega, theta}}
  capital_paisa        BIGINT,
  PRIMARY KEY (trade_date, ts)
);
CREATE INDEX IF NOT EXISTS idx_greeks_date ON portfolio_greeks_snapshot(trade_date DESC);

-- =========================================================
-- VERSION MARKER
-- =========================================================
INSERT INTO schema_version (version, description)
VALUES ('phase_f2_greeks', 'Phase F.2 — portfolio Greeks aggregation')
ON CONFLICT (version) DO NOTHING;
