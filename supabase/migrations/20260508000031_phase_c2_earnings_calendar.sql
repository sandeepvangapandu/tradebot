-- Phase C.2 — Earnings Calendar
-- Stores earnings announcement dates, EPS estimates/actuals, and surprise %.
-- Used by the blackout filter (T-2 to T+1) to avoid trading around earnings.

CREATE TABLE IF NOT EXISTS earnings_calendar (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  earnings_date DATE NOT NULL,
  fiscal_quarter TEXT,                       -- e.g. "Q4 FY26"
  reporting_time TEXT,                       -- 'BMO' (before market open), 'AMC' (after market close), 'DURING'
  expected_eps NUMERIC,
  actual_eps NUMERIC,
  expected_revenue_crore NUMERIC,
  actual_revenue_crore NUMERIC,
  surprise_pct NUMERIC,                      -- (actual - expected) / abs(expected) * 100
  source TEXT DEFAULT 'MONEYCONTROL',
  scraped_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (symbol, earnings_date)
);

CREATE INDEX IF NOT EXISTS idx_earnings_symbol_date ON earnings_calendar(symbol, earnings_date DESC);
CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_calendar(earnings_date);

INSERT INTO schema_version (version, description)
VALUES ('phase_c2_earnings_calendar', 'Phase C.2 — earnings dates + actuals + surprises')
ON CONFLICT (version) DO NOTHING;
