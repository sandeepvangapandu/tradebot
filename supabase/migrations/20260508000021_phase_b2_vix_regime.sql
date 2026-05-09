-- =========================================================
-- Phase B.2: India VIX Regime Classifier
-- =========================================================

CREATE TABLE IF NOT EXISTS vix_regime_daily (
  trade_date DATE PRIMARY KEY,
  vix_open NUMERIC,
  vix_high NUMERIC,
  vix_low NUMERIC,
  vix_close NUMERIC NOT NULL,
  vix_change_pct NUMERIC,
  regime TEXT NOT NULL CHECK (regime IN ('LOW','NORMAL','HIGH','SPIKE')),
  percentile_60d NUMERIC,                -- where current VIX sits in 60d distribution
  computed_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_vix_regime_date ON vix_regime_daily(trade_date DESC);

CREATE TABLE IF NOT EXISTS vix_regime_intraday (
  ts TIMESTAMPTZ PRIMARY KEY,
  vix_value NUMERIC NOT NULL,
  regime TEXT NOT NULL,
  spike_detected BOOLEAN DEFAULT FALSE   -- > 30% intraday move
);

INSERT INTO schema_version (version, description)
VALUES ('phase_b2_vix_regime', 'Phase B.2 — VIX regime classifier daily + intraday')
ON CONFLICT (version) DO NOTHING;
