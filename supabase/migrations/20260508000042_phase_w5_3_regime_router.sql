-- =========================================================
-- Wave 5.3 — Regime Router: daily regime decisions + strategy gating
-- =========================================================
-- Stores the morning regime classification and the resulting allowed /
-- blocked strategy type lists for each trading day.
-- Idempotent: all DDL uses IF NOT EXISTS.
-- =========================================================

CREATE TABLE IF NOT EXISTS regime_decisions_daily (
  trade_date DATE PRIMARY KEY,
  vix_regime TEXT,                            -- LOW/NORMAL/HIGH/SPIKE
  sector_breadth NUMERIC,                     -- fraction of sectors with positive RS
  macro_signal TEXT,                          -- BULLISH/BEARISH/NEUTRAL
  flow_signal TEXT,                           -- TAILWIND/HEADWIND/MIXED
  market_regime TEXT NOT NULL,                -- TREND_BULL/TREND_BEAR/RANGE/CHOP/HIGH_VOL
  allowed_strategy_types JSONB NOT NULL,      -- ["trend","mean_rev","breakout","options_sell","options_buy"]
  blocked_strategy_types JSONB,               -- with reasons {type: reason}
  evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_regime_decisions_date ON regime_decisions_daily(trade_date DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_w5_3_regime_router', 'Wave 5.3 — daily regime decisions + strategy gating')
ON CONFLICT (version) DO NOTHING;
