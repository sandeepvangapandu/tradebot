-- =========================================================
-- Phase A.2: Per-minute Tick Aggregates + CVD
-- =========================================================

CREATE TABLE IF NOT EXISTS tick_metrics_minute (
  instrument_key TEXT NOT NULL,
  minute_ts TIMESTAMPTZ NOT NULL,
  buy_volume BIGINT DEFAULT 0,
  sell_volume BIGINT DEFAULT 0,
  delta BIGINT DEFAULT 0,                      -- buy_volume - sell_volume
  cvd_running BIGINT DEFAULT 0,                -- cumulative delta from session start
  trade_count INT DEFAULT 0,
  avg_spread_bps NUMERIC,
  PRIMARY KEY (instrument_key, minute_ts)
);
CREATE INDEX IF NOT EXISTS idx_tick_metrics_minute_ts ON tick_metrics_minute(minute_ts DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_a2_tick_metrics', 'Phase A.2 — per-minute tick aggregates + CVD')
ON CONFLICT (version) DO NOTHING;
