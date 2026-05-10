-- =========================================================
-- Phase B.4: FII/DII Flows + Bond Yields + Flow Regime
-- =========================================================

CREATE TABLE IF NOT EXISTS fii_dii_flows_daily (
  trade_date DATE PRIMARY KEY,
  fii_buy_value_crore NUMERIC,           -- ₹ crore
  fii_sell_value_crore NUMERIC,
  fii_net_value_crore NUMERIC,
  dii_buy_value_crore NUMERIC,
  dii_sell_value_crore NUMERIC,
  dii_net_value_crore NUMERIC,
  net_flow_total_crore NUMERIC,          -- fii + dii
  source TEXT,                           -- 'NSE'|'MONEYCONTROL'|'MANUAL'
  scraped_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fii_dii_date ON fii_dii_flows_daily(trade_date DESC);

CREATE TABLE IF NOT EXISTS bond_yield_daily (
  trade_date DATE PRIMARY KEY,
  yield_10y_pct NUMERIC NOT NULL,
  change_bps NUMERIC,                    -- bps change vs prev day
  trend_5d TEXT CHECK (trend_5d IN ('RISING','FALLING','FLAT')),
  source TEXT,
  scraped_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS flow_regime_daily (
  trade_date DATE PRIMARY KEY,
  fii_streak INT DEFAULT 0,              -- consecutive days same direction (- if selling)
  dii_streak INT DEFAULT 0,
  fii_regime TEXT CHECK (fii_regime IN ('STRONG_BUY','BUY','NEUTRAL','SELL','STRONG_SELL')),
  dii_regime TEXT CHECK (dii_regime IN ('STRONG_BUY','BUY','NEUTRAL','SELL','STRONG_SELL')),
  combined_signal TEXT CHECK (combined_signal IN ('TAILWIND','HEADWIND','MIXED'))
);

INSERT INTO schema_version (version, description)
VALUES ('phase_b4_flows_yields', 'Phase B.4 — FII/DII flows, bond yields, derived regime')
ON CONFLICT (version) DO NOTHING;
