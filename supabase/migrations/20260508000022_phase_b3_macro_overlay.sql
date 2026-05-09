-- =========================================================
-- Phase B.3 — Macro Overlay (USDINR / Crude / Gold / Silver)
-- =========================================================

CREATE TABLE IF NOT EXISTS macro_instruments (
  symbol TEXT PRIMARY KEY,
  instrument_key TEXT NOT NULL,
  category TEXT NOT NULL CHECK (category IN ('currency','commodity','rate')),
  segment TEXT NOT NULL,
  display_name TEXT,
  active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS macro_regime_daily (
  trade_date DATE NOT NULL,
  symbol TEXT NOT NULL,
  close_paisa BIGINT NOT NULL,
  return_pct_1d NUMERIC,
  return_pct_5d NUMERIC,
  return_pct_20d NUMERIC,
  trend TEXT CHECK (trend IN ('UP','DOWN','RANGE')),
  zscore_20d NUMERIC,                    -- (close - mean_20d) / stdev_20d
  PRIMARY KEY (trade_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_macro_regime_date ON macro_regime_daily(trade_date DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_b3_macro_overlay', 'Phase B.3 — macro instruments (USDINR/Crude/Gold/Silver) + daily regime')
ON CONFLICT (version) DO NOTHING;

-- =========================================================
-- Seed macro instruments
-- Verified instrument_keys from NSE instrument master (2026-05-08):
--   USDINR : NCD_FO|7009   — USDINR FUT 22 MAY 26  (segment: NCD_FO)
--   CRUDE  : NSE_COM|121620 — CRUDEOIL FUT 18 MAY 26 (segment: NSE_COM)
--   GOLD   : NSE_COM|122886 — GOLD FUT 05 JUN 26     (segment: NSE_COM)
--   SILVER : NSE_COM|121799 — SILVER FUT 03 JUL 26   (segment: NSE_COM)
-- Note: Commodities are on NSE_COM (not MCX_FO). MCX instrument master
--       was not separately cached; NSE_COM carries the same contracts.
-- =========================================================
INSERT INTO macro_instruments (symbol, instrument_key, category, segment, display_name) VALUES
  ('USDINR', 'NCD_FO|7009',    'currency',  'NCD_FO',  'USD/INR Front-Month Futures'),
  ('CRUDE',  'NSE_COM|121620', 'commodity', 'NSE_COM', 'Crude Oil Front-Month Futures'),
  ('GOLD',   'NSE_COM|122886', 'commodity', 'NSE_COM', 'Gold Front-Month Futures'),
  ('SILVER', 'NSE_COM|121799', 'commodity', 'NSE_COM', 'Silver Front-Month Futures')
ON CONFLICT (symbol) DO UPDATE SET
  instrument_key = EXCLUDED.instrument_key,
  segment        = EXCLUDED.segment,
  display_name   = EXCLUDED.display_name,
  active         = TRUE;
