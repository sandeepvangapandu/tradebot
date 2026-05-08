-- =========================================================
-- METADATA
-- =========================================================
CREATE TABLE IF NOT EXISTS bot_config (
  key TEXT PRIMARY KEY,
  value JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS schema_version (
  version TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ DEFAULT NOW(),
  description TEXT
);

-- =========================================================
-- INSTRUMENTS
-- =========================================================
CREATE TABLE IF NOT EXISTS instruments (
  instrument_key TEXT PRIMARY KEY,
  segment TEXT NOT NULL,
  trading_symbol TEXT NOT NULL,
  name TEXT,
  underlying_key TEXT,
  underlying_symbol TEXT,
  expiry DATE,
  strike_price NUMERIC,
  option_type TEXT CHECK (option_type IN ('CE','PE') OR option_type IS NULL),
  instrument_type TEXT,
  lot_size INT DEFAULT 0,
  tick_size NUMERIC DEFAULT 0,
  freeze_quantity INT DEFAULT 0,
  isin TEXT,
  exchange_token TEXT,
  refreshed_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_instruments_segment ON instruments(segment);
CREATE INDEX IF NOT EXISTS idx_instruments_underlying ON instruments(underlying_key, expiry, option_type);

-- =========================================================
-- BARS
-- =========================================================
CREATE TABLE IF NOT EXISTS bars (
  instrument_key TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  open BIGINT NOT NULL,
  high BIGINT NOT NULL,
  low BIGINT NOT NULL,
  close BIGINT NOT NULL,
  volume BIGINT DEFAULT 0,
  oi BIGINT,
  PRIMARY KEY (instrument_key, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars(ts DESC);
CREATE INDEX IF NOT EXISTS idx_bars_instrument_tf ON bars(instrument_key, timeframe, ts DESC);

-- =========================================================
-- TICKS (hot 7d, archived after)
-- =========================================================
CREATE TABLE IF NOT EXISTS ticks (
  id BIGSERIAL PRIMARY KEY,
  instrument_key TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  ltp BIGINT NOT NULL,
  volume BIGINT DEFAULT 0,
  bid BIGINT,
  ask BIGINT,
  bid_qty INT,
  ask_qty INT,
  total_buy_qty BIGINT,
  total_sell_qty BIGINT
);
CREATE INDEX IF NOT EXISTS idx_ticks_instrument_ts ON ticks(instrument_key, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(ts DESC);

-- =========================================================
-- DEPTH (hot 7d, archived after)
-- =========================================================
CREATE TABLE IF NOT EXISTS depth_snapshots (
  id BIGSERIAL PRIMARY KEY,
  instrument_key TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  bids JSONB NOT NULL,
  asks JSONB NOT NULL,
  spread_bps NUMERIC,
  bid_wall_top5 BIGINT,
  ask_wall_top5 BIGINT
);
CREATE INDEX IF NOT EXISTS idx_depth_instrument_ts ON depth_snapshots(instrument_key, ts DESC);

-- =========================================================
-- ORDERS / FILLS / POSITIONS / DAILY PNL
-- =========================================================
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT PRIMARY KEY,
  client_order_id TEXT UNIQUE,
  strategy_name TEXT,
  instrument_key TEXT NOT NULL,
  side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  order_type TEXT NOT NULL,
  quantity INT NOT NULL,
  price BIGINT,
  trigger_price BIGINT,
  product TEXT,
  validity TEXT,
  status TEXT NOT NULL,
  filled_qty INT DEFAULT 0,
  avg_fill_price BIGINT,
  rejection_reason TEXT,
  placed_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  raw_response JSONB
);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_name, placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_instrument ON orders(instrument_key, placed_at DESC);

CREATE TABLE IF NOT EXISTS fills (
  fill_id TEXT PRIMARY KEY,
  order_id TEXT REFERENCES orders(order_id),
  instrument_key TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity INT NOT NULL,
  price BIGINT NOT NULL,
  fee_paisa BIGINT DEFAULT 0,
  filled_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);

CREATE TABLE IF NOT EXISTS positions (
  id BIGSERIAL PRIMARY KEY,
  strategy_name TEXT,
  instrument_key TEXT NOT NULL,
  side TEXT NOT NULL,
  entry_qty INT NOT NULL,
  entry_avg_price BIGINT NOT NULL,
  exit_qty INT DEFAULT 0,
  exit_avg_price BIGINT,
  realized_pnl BIGINT DEFAULT 0,
  unrealized_pnl BIGINT DEFAULT 0,
  opened_at TIMESTAMPTZ DEFAULT NOW(),
  closed_at TIMESTAMPTZ,
  status TEXT NOT NULL CHECK (status IN ('OPEN','CLOSED','PARTIAL'))
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status, opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_name, opened_at DESC);

CREATE TABLE IF NOT EXISTS daily_pnl (
  id BIGSERIAL PRIMARY KEY,
  trade_date DATE NOT NULL,
  strategy_name TEXT,
  realized_pnl BIGINT DEFAULT 0,
  unrealized_pnl BIGINT DEFAULT 0,
  fees_paid BIGINT DEFAULT 0,
  trade_count INT DEFAULT 0,
  win_count INT DEFAULT 0,
  loss_count INT DEFAULT 0,
  UNIQUE (trade_date, strategy_name)
);

-- =========================================================
-- SIGNALS
-- =========================================================
CREATE TABLE IF NOT EXISTS signals (
  id BIGSERIAL PRIMARY KEY,
  strategy_name TEXT NOT NULL,
  instrument_key TEXT NOT NULL,
  signal_type TEXT NOT NULL,
  ts TIMESTAMPTZ DEFAULT NOW(),
  conditions_met JSONB,
  conditions_failed JSONB,
  confluence_score NUMERIC,
  rejection_reason TEXT,
  acted_on BOOLEAN DEFAULT FALSE,
  order_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts DESC);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_name, ts DESC);

-- =========================================================
-- BOT RUNS
-- =========================================================
CREATE TABLE IF NOT EXISTS bot_runs (
  run_id BIGSERIAL PRIMARY KEY,
  started_at TIMESTAMPTZ DEFAULT NOW(),
  ended_at TIMESTAMPTZ,
  branch TEXT,
  git_sha TEXT,
  mode TEXT,
  capital_paisa BIGINT,
  exit_reason TEXT
);

-- =========================================================
-- VERSION MARKER
-- =========================================================
INSERT INTO schema_version (version, description)
VALUES ('phase0_foundation', 'Initial foundation tables: instruments, bars, ticks, depth, orders, fills, positions, daily_pnl, signals, bot_runs')
ON CONFLICT (version) DO NOTHING;
