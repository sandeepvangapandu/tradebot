-- =========================================================
-- Phase D: Universe Scanner Tables
-- =========================================================

CREATE TABLE IF NOT EXISTS universe_constituents (
  symbol TEXT PRIMARY KEY,
  instrument_key TEXT NOT NULL,
  segment TEXT NOT NULL,
  added_at TIMESTAMPTZ DEFAULT NOW(),
  active BOOLEAN DEFAULT TRUE,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS liquidity_rank_daily (
  trade_date DATE NOT NULL,
  symbol TEXT NOT NULL,
  adv_paisa BIGINT NOT NULL,             -- avg daily value (last 20 trading days)
  avg_daily_volume BIGINT NOT NULL,
  rank INT NOT NULL,
  PRIMARY KEY (trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS volatility_rank_daily (
  trade_date DATE NOT NULL,
  symbol TEXT NOT NULL,
  atr_pct NUMERIC NOT NULL,              -- ATR(14) / close * 100
  realized_vol_20d NUMERIC,              -- annualized stdev of log returns
  rank INT NOT NULL,
  PRIMARY KEY (trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS rs_rank_daily (
  trade_date DATE NOT NULL,
  symbol TEXT NOT NULL,
  rs_score NUMERIC NOT NULL,             -- relative strength vs NIFTY 50 over 20d
  rs_rank INT NOT NULL,
  PRIMARY KEY (trade_date, symbol)
);

CREATE TABLE IF NOT EXISTS universe_daily_snapshot (
  trade_date DATE NOT NULL,
  symbol TEXT NOT NULL,
  in_universe BOOLEAN NOT NULL,
  liquidity_rank INT,
  volatility_rank INT,
  rs_rank INT,
  composite_score NUMERIC,               -- weighted combo
  blackout BOOLEAN DEFAULT FALSE,        -- earnings/corp-action filter (filled by Phase C)
  blackout_reason TEXT,
  PRIMARY KEY (trade_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_universe_snapshot_date ON universe_daily_snapshot(trade_date DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_d_universe', 'Universe scanner tables: constituents, liquidity/volatility/RS ranks, daily snapshot')
ON CONFLICT (version) DO NOTHING;

-- Seed Top 10 constituents (idempotent).
-- instrument_keys verified against live NSE instrument master (2026-05-08).
-- KOTAKBANK corrected: INE237A01028 → INE237A01036 (share-split adjusted ISIN).
INSERT INTO universe_constituents (symbol, instrument_key, segment, notes) VALUES
  ('RELIANCE',   'NSE_EQ|INE002A01018', 'NSE_EQ', 'Top 10'),
  ('HDFCBANK',   'NSE_EQ|INE040A01034', 'NSE_EQ', 'Top 10'),
  ('ICICIBANK',  'NSE_EQ|INE090A01021', 'NSE_EQ', 'Top 10'),
  ('TCS',        'NSE_EQ|INE467B01029', 'NSE_EQ', 'Top 10'),
  ('INFY',       'NSE_EQ|INE009A01021', 'NSE_EQ', 'Top 10'),
  ('HINDUNILVR', 'NSE_EQ|INE030A01027', 'NSE_EQ', 'Top 10'),
  ('ITC',        'NSE_EQ|INE154A01025', 'NSE_EQ', 'Top 10'),
  ('AXISBANK',   'NSE_EQ|INE238A01034', 'NSE_EQ', 'Top 10'),
  ('KOTAKBANK',  'NSE_EQ|INE237A01036', 'NSE_EQ', 'Top 10'),
  ('SBIN',       'NSE_EQ|INE062A01020', 'NSE_EQ', 'Top 10')
ON CONFLICT (symbol) DO UPDATE SET
  instrument_key = EXCLUDED.instrument_key,
  active = TRUE,
  notes = EXCLUDED.notes;
