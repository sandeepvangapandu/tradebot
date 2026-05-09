-- Phase C.4: Block / Bulk Deals + Insider / Promoter Trading Disclosures
-- Migration: 20260508000033_phase_c4_block_insider.sql

CREATE TABLE IF NOT EXISTS block_deals (
  id BIGSERIAL PRIMARY KEY,
  trade_date DATE NOT NULL,
  symbol TEXT NOT NULL,
  client_name TEXT,
  side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  quantity BIGINT NOT NULL,
  price_paisa BIGINT NOT NULL,
  value_crore NUMERIC,
  exchange TEXT,                              -- 'NSE'|'BSE'
  source TEXT DEFAULT 'NSE',
  scraped_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (trade_date, symbol, client_name, side, quantity, price_paisa)
);
CREATE INDEX IF NOT EXISTS idx_block_symbol_date ON block_deals(symbol, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_block_date ON block_deals(trade_date DESC);

CREATE TABLE IF NOT EXISTS bulk_deals (
  id BIGSERIAL PRIMARY KEY,
  trade_date DATE NOT NULL,
  symbol TEXT NOT NULL,
  client_name TEXT,
  side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  quantity BIGINT NOT NULL,
  price_paisa BIGINT NOT NULL,
  exchange TEXT,
  source TEXT DEFAULT 'NSE',
  scraped_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (trade_date, symbol, client_name, side, quantity, price_paisa)
);
CREATE INDEX IF NOT EXISTS idx_bulk_symbol_date ON bulk_deals(symbol, trade_date DESC);

CREATE TABLE IF NOT EXISTS insider_trades (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  acquirer_name TEXT,
  acquirer_category TEXT,                     -- 'PROMOTER'|'PROMOTER GROUP'|'KMP'|'OTHER'
  trade_type TEXT CHECK (trade_type IN ('BUY','SELL','PLEDGE','REVOKE')),
  quantity BIGINT,
  value_crore NUMERIC,
  trade_date DATE,
  disclosure_date DATE,
  source TEXT DEFAULT 'NSE_SAST',             -- 'NSE_SAST'|'BSE_SAST'|'PIT'
  scraped_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (symbol, acquirer_name, trade_type, trade_date, quantity)
);
CREATE INDEX IF NOT EXISTS idx_insider_symbol_disclosure ON insider_trades(symbol, disclosure_date DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_c4_block_insider', 'Phase C.4 — block/bulk deals + insider/promoter trading')
ON CONFLICT (version) DO NOTHING;
