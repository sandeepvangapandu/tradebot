-- =========================================================
-- Phase A.4: Daily Volume Profile (POC / VAH / VAL)
-- =========================================================

CREATE TABLE IF NOT EXISTS volume_profile_daily (
  instrument_key TEXT NOT NULL,
  trade_date DATE NOT NULL,
  poc_paisa BIGINT NOT NULL,           -- Point of Control: price with max volume
  vah_paisa BIGINT NOT NULL,           -- Value Area High (70% volume upper bound)
  val_paisa BIGINT NOT NULL,           -- Value Area Low (70% volume lower bound)
  total_volume BIGINT NOT NULL,
  bin_size_paisa BIGINT NOT NULL,
  profile_bins JSONB,                  -- [{price, volume}, ...]
  PRIMARY KEY (instrument_key, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_vp_daily_date ON volume_profile_daily(trade_date DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_a4_volume_profile', 'Phase A.4 — daily volume profile (POC/VAH/VAL)')
ON CONFLICT (version) DO NOTHING;
