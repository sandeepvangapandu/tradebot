-- =========================================================
-- Phase G — Kronos foundation-model forecasts (shadow mode)
-- =========================================================
-- Idempotent: all statements use IF NOT EXISTS / ON CONFLICT DO NOTHING.

CREATE TABLE IF NOT EXISTS kronos_forecasts (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT NOW(),
  instrument_key TEXT NOT NULL,
  model_size TEXT NOT NULL,                    -- 'mini'|'small'|'base'
  timeframe TEXT NOT NULL,                     -- '1m'|'5m'|'15m'|'1h'|'1d'
  context_bars INT NOT NULL,
  horizon_bars INT NOT NULL,
  current_close_paisa BIGINT NOT NULL,
  predicted_close_paisa BIGINT,                -- close at end of horizon
  predicted_high_paisa BIGINT,                 -- max high across horizon
  predicted_low_paisa BIGINT,                  -- min low across horizon
  predicted_direction TEXT CHECK (predicted_direction IN ('UP','DOWN','FLAT')),
  predicted_change_pct NUMERIC,
  predicted_range_pct NUMERIC,                 -- (high - low) / current_close
  raw_forecast JSONB,                          -- full per-bar forecast (open/high/low/close/volume/amount * horizon)
  inference_ms INT,
  model_version TEXT NOT NULL                  -- e.g. 'NeoQuasar/Kronos-small@<commit>'
);
CREATE INDEX IF NOT EXISTS idx_kronos_instrument_ts ON kronos_forecasts(instrument_key, ts DESC);
CREATE INDEX IF NOT EXISTS idx_kronos_ts ON kronos_forecasts(ts DESC);

CREATE TABLE IF NOT EXISTS kronos_accuracy_daily (
  trade_date DATE NOT NULL,
  instrument_key TEXT NOT NULL,
  horizon_bars INT NOT NULL,
  prediction_count INT DEFAULT 0,
  direction_correct INT DEFAULT 0,
  direction_accuracy_pct NUMERIC,
  close_mae_pct NUMERIC,                       -- mean abs % error of close
  range_mae_pct NUMERIC,
  PRIMARY KEY (trade_date, instrument_key, horizon_bars)
);

INSERT INTO schema_version (version, description)
VALUES ('phase_g_kronos', 'Phase G — Kronos foundation-model forecasts (shadow mode)')
ON CONFLICT (version) DO NOTHING;
