-- =========================================================
-- Wave 5.4 — Kelly Position Allocations + Rolling Performance
-- =========================================================
-- Stores daily Kelly-based capital allocation per strategy and
-- maintains rolling 30-day performance metrics used for sizing.
-- Idempotent: all DDL uses IF NOT EXISTS.
-- =========================================================

CREATE TABLE IF NOT EXISTS kelly_allocations_daily (
  trade_date DATE NOT NULL,
  strategy_name TEXT NOT NULL,
  win_rate NUMERIC,                              -- 0-1, rolling 30-day
  avg_win_paisa BIGINT,
  avg_loss_paisa BIGINT,                         -- absolute (positive)
  expectancy NUMERIC,                            -- win_rate * avg_win - (1-win_rate) * avg_loss
  edge NUMERIC,                                  -- expectancy / avg_loss (Kelly numerator)
  full_kelly_pct NUMERIC,                        -- (b*p - q) / b where b = avg_win/avg_loss
  half_kelly_pct NUMERIC,                        -- full_kelly / 2
  recommended_pct NUMERIC NOT NULL,              -- after caps + floor
  trade_count INT,
  PRIMARY KEY (trade_date, strategy_name)
);
CREATE INDEX IF NOT EXISTS idx_kelly_alloc_date ON kelly_allocations_daily(trade_date DESC);

CREATE TABLE IF NOT EXISTS strategy_performance_rolling (
  strategy_name TEXT PRIMARY KEY,
  rolling_window_days INT NOT NULL DEFAULT 30,
  trade_count INT DEFAULT 0,
  win_count INT DEFAULT 0,
  loss_count INT DEFAULT 0,
  total_pnl_paisa BIGINT DEFAULT 0,
  total_win_paisa BIGINT DEFAULT 0,
  total_loss_paisa BIGINT DEFAULT 0,
  sharpe_ratio NUMERIC,
  max_drawdown_pct NUMERIC,
  computed_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO schema_version (version, description)
VALUES ('phase_w5_4_kelly_sizer', 'Wave 5.4 — Kelly position allocations + rolling perf')
ON CONFLICT (version) DO NOTHING;
