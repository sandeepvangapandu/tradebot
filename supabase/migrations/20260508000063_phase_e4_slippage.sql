-- =========================================================
-- Phase E.4 — Slippage tracking + drift detection
-- =========================================================

CREATE TABLE IF NOT EXISTS slippage_log (
  id BIGSERIAL PRIMARY KEY,
  fill_id TEXT,                                  -- FK fills.fill_id
  order_id TEXT,                                 -- FK orders.order_id
  instrument_key TEXT NOT NULL,
  side TEXT NOT NULL,
  intended_price BIGINT NOT NULL,
  actual_fill_price BIGINT NOT NULL,
  quantity INT NOT NULL,
  slippage_paisa BIGINT NOT NULL,                -- (actual - intended) * sign(side)
  slippage_bps NUMERIC,
  slippage_cost_paisa BIGINT,                    -- slippage_paisa * quantity
  mode TEXT NOT NULL,                            -- paper|live
  recorded_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_slippage_instrument ON slippage_log(instrument_key, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_slippage_mode ON slippage_log(mode, recorded_at DESC);

CREATE TABLE IF NOT EXISTS slippage_drift_alerts (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT NOW(),
  instrument_key TEXT,
  alert_type TEXT NOT NULL,                      -- DRIFT_HIGH/DRIFT_BLOCK
  paper_avg_bps NUMERIC,
  live_avg_bps NUMERIC,
  drift_pct NUMERIC,
  threshold_pct NUMERIC,
  details JSONB
);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_ts ON slippage_drift_alerts(ts DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_e4_slippage', 'Phase E.4 — slippage tracking + drift detection')
ON CONFLICT (version) DO NOTHING;
