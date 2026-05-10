-- Phase C.1: Corporate Actions Calendar (dividends / splits / bonus / etc.)
-- Migration: 20260508000030_phase_c1_corporate_actions.sql

CREATE TABLE IF NOT EXISTS corporate_actions (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  instrument_key TEXT,
  action_type TEXT NOT NULL CHECK (action_type IN ('DIVIDEND','SPLIT','BONUS','RIGHTS','MERGER','BUYBACK','OTHER')),
  ex_date DATE NOT NULL,
  record_date DATE,
  details TEXT,                              -- e.g. "Dividend ₹5/share", "1:2 split", "1:1 bonus"
  ratio NUMERIC,                             -- numeric amount (₹ or ratio)
  source TEXT DEFAULT 'NSE',
  scraped_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (symbol, action_type, ex_date)
);

CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol_date ON corporate_actions(symbol, ex_date DESC);
CREATE INDEX IF NOT EXISTS idx_corp_actions_ex_date ON corporate_actions(ex_date);

INSERT INTO schema_version (version, description)
VALUES ('phase_c1_corporate_actions', 'Phase C.1 — corporate actions calendar (dividends/splits/bonus)')
ON CONFLICT (version) DO NOTHING;
