-- =========================================================
-- Phase B.1 — Sector Indices + Daily Rotation Rank
-- =========================================================

CREATE TABLE IF NOT EXISTS sector_indices (
  symbol TEXT PRIMARY KEY,
  instrument_key TEXT NOT NULL,
  display_name TEXT NOT NULL,
  active BOOLEAN DEFAULT TRUE,
  added_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sector_rank_daily (
  trade_date DATE NOT NULL,
  sector_symbol TEXT NOT NULL,
  rs_score NUMERIC NOT NULL,                -- relative strength vs NIFTY 50
  return_pct_5d NUMERIC,
  return_pct_20d NUMERIC,
  rank INT NOT NULL,
  rank_change INT,                          -- vs prev day
  PRIMARY KEY (trade_date, sector_symbol)
);
CREATE INDEX IF NOT EXISTS idx_sector_rank_date ON sector_rank_daily(trade_date DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_b1_sector_rotation', 'Phase B.1 — sector indices + daily rotation rank')
ON CONFLICT (version) DO NOTHING;

INSERT INTO sector_indices (symbol, instrument_key, display_name) VALUES
  ('NIFTY_BANK',        'NSE_INDEX|Nifty Bank',          'Nifty Bank'),
  ('NIFTY_IT',          'NSE_INDEX|Nifty IT',            'Nifty IT'),
  ('NIFTY_AUTO',        'NSE_INDEX|Nifty Auto',          'Nifty Auto'),
  ('NIFTY_PHARMA',      'NSE_INDEX|Nifty Pharma',        'Nifty Pharma'),
  ('NIFTY_METAL',       'NSE_INDEX|Nifty Metal',         'Nifty Metal'),
  ('NIFTY_FMCG',        'NSE_INDEX|Nifty FMCG',          'Nifty FMCG'),
  ('NIFTY_ENERGY',      'NSE_INDEX|Nifty Energy',        'Nifty Energy'),
  ('NIFTY_REALTY',      'NSE_INDEX|Nifty Realty',        'Nifty Realty'),
  ('NIFTY_FIN_SERVICE', 'NSE_INDEX|Nifty Fin Service',   'Nifty Financial Services')
ON CONFLICT (symbol) DO UPDATE SET active = TRUE;
