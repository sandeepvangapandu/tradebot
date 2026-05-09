-- Phase E.1 — Smart Order Router audit table
-- Tracks every parent order + child slices sent by SmartOrderRouter

CREATE TABLE IF NOT EXISTS routed_orders (
  id BIGSERIAL PRIMARY KEY,
  parent_order_id TEXT,                          -- client_order_id of original signal
  child_order_id TEXT NOT NULL,                  -- broker order_id of slice
  routing_strategy TEXT NOT NULL,                -- LIMIT_PROTECT/ICEBERG/TWAP/VWAP/MARKET
  slice_index INT DEFAULT 0,                     -- 0..N for iceberg/twap
  total_slices INT DEFAULT 1,
  intended_qty INT NOT NULL,
  filled_qty INT DEFAULT 0,
  intended_price BIGINT,                         -- paisa; NULL for MARKET
  avg_fill_price BIGINT,                         -- paisa; NULL until filled
  status TEXT NOT NULL,                          -- PENDING/PLACED/PARTIAL/FILLED/CANCELLED/FAILED
  routing_metadata JSONB,                        -- {protection_bps, retry_count, urgency, ...}
  created_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_routed_parent ON routed_orders(parent_order_id);
CREATE INDEX IF NOT EXISTS idx_routed_status ON routed_orders(status, created_at DESC);

INSERT INTO schema_version (version, description)
VALUES ('phase_e1_smart_router', 'Phase E.1 — smart order routing audit')
ON CONFLICT (version) DO NOTHING;
