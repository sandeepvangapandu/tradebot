-- =========================================================
-- Phase A.1: L2 Order Book Depth Feed — Minute Aggregates
-- =========================================================
-- Idempotent: all statements use IF NOT EXISTS / ON CONFLICT DO NOTHING.

CREATE TABLE IF NOT EXISTS depth_metrics_minute (
  instrument_key   TEXT        NOT NULL,
  minute_ts        TIMESTAMPTZ NOT NULL,
  avg_spread_bps   NUMERIC,
  min_spread_bps   NUMERIC,
  max_spread_bps   NUMERIC,
  avg_bid_qty_top5 BIGINT,
  avg_ask_qty_top5 BIGINT,
  bid_wall_max_paisa BIGINT,       -- max single-level bid qty observed in the minute
  ask_wall_max_paisa BIGINT,       -- max single-level ask qty observed in the minute
  imbalance_avg    NUMERIC,        -- mean of (bid_qty-ask_qty)/(bid_qty+ask_qty) per tick
  PRIMARY KEY (instrument_key, minute_ts)
);

CREATE INDEX IF NOT EXISTS idx_depth_metrics_minute_ts
  ON depth_metrics_minute(minute_ts DESC);

-- =========================================================
-- VERSION MARKER
-- =========================================================
INSERT INTO schema_version (version, description)
VALUES (
  'phase_a1_depth_metrics',
  'Phase A.1 — depth aggregated metrics per minute'
)
ON CONFLICT (version) DO NOTHING;
