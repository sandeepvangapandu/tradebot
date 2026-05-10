-- =========================================================
-- Phase F.1 — Portfolio Risk: Correlation, Exposure, VaR/CVaR
-- =========================================================
-- Stores rolling 60-day pairwise correlation matrix, daily
-- portfolio exposure snapshots (gross/net, beta-weighted,
-- sector/symbol breakdowns), and daily VaR/CVaR computations.
-- Idempotent: all DDL uses IF NOT EXISTS.
-- =========================================================

CREATE TABLE IF NOT EXISTS correlation_matrix_daily (
  trade_date DATE NOT NULL,
  symbol_a TEXT NOT NULL,
  symbol_b TEXT NOT NULL,
  correlation NUMERIC NOT NULL CHECK (correlation BETWEEN -1.0 AND 1.0),
  lookback_days INT NOT NULL DEFAULT 60,
  PRIMARY KEY (trade_date, symbol_a, symbol_b)
);
CREATE INDEX IF NOT EXISTS idx_correlation_date ON correlation_matrix_daily(trade_date DESC);

CREATE TABLE IF NOT EXISTS portfolio_exposure_snapshot (
  ts TIMESTAMPTZ DEFAULT NOW(),
  trade_date DATE NOT NULL,
  gross_exposure_paisa BIGINT NOT NULL,
  net_exposure_paisa BIGINT NOT NULL,
  beta_weighted_gross NUMERIC,
  beta_weighted_net NUMERIC,
  sector_breakdown JSONB,                       -- {sector: pct_of_capital}
  symbol_breakdown JSONB,                       -- {symbol: pct_of_capital}
  capital_paisa BIGINT NOT NULL,
  PRIMARY KEY (trade_date, ts)
);

CREATE TABLE IF NOT EXISTS var_daily (
  trade_date DATE PRIMARY KEY,
  capital_paisa BIGINT NOT NULL,
  var_95_paisa BIGINT NOT NULL,                 -- 1-day VaR at 95% confidence
  var_99_paisa BIGINT NOT NULL,
  cvar_95_paisa BIGINT NOT NULL,                -- expected loss given VaR breached
  cvar_99_paisa BIGINT NOT NULL,
  method TEXT NOT NULL DEFAULT 'historical',    -- 'historical'|'parametric'|'monte_carlo'
  computed_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO schema_version (version, description)
VALUES ('phase_f1_portfolio_risk', 'Phase F.1 — correlation matrix, exposure, VaR/CVaR')
ON CONFLICT (version) DO NOTHING;
