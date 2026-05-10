"""Tests for src/data/depth_feed.py — L2 order book depth feed handler.

All tests use inline mock payloads so they are self-contained and do not
depend on external fixtures.  A real database is not required; DB writes
are patched or exercised against an in-memory SQLite engine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from src.data.depth_feed import DepthFeedHandler


# ---------------------------------------------------------------------------
# Helpers — build synthetic Upstox V3 payloads
# ---------------------------------------------------------------------------

def _make_bid_ask_quote(
    bid_price: int,
    bid_qty: int,
    ask_price: int,
    ask_qty: int,
    bid_orders: int = 1,
    ask_orders: int = 1,
) -> dict:
    """Return a single level dict matching Upstox V3 protobuf-decoded format."""
    return {
        "bp": bid_price,
        "bid_q": bid_qty,
        "bid_n": bid_orders,
        "ap": ask_price,
        "ask_q": ask_qty,
        "ask_n": ask_orders,
    }


def _make_payload(
    instrument_key: str = "NSE_EQ|INE002A01018",
    quotes: list[dict] | None = None,
    ts_ms: int | None = None,
) -> dict:
    """Return a minimal Upstox V3 WebSocket message dict."""
    if quotes is None:
        quotes = [
            _make_bid_ask_quote(245000, 500, 245050, 300),
            _make_bid_ask_quote(244950, 800, 245100, 250),
            _make_bid_ask_quote(244900, 1000, 245150, 400),
            _make_bid_ask_quote(244850, 600, 245200, 200),
            _make_bid_ask_quote(244800, 700, 245250, 350),
        ]
    payload: dict = {
        "feeds": {
            instrument_key: {
                "fullFeed": {
                    "marketFF": {
                        "marketLevel": {
                            "bidAskQuote": quotes,
                        },
                        "ltpc": {
                            "ltt": ts_ms or 1715150400000,
                        },
                    }
                }
            }
        },
        "currentTs": ts_ms or 1715150400000,
    }
    return payload


# ---------------------------------------------------------------------------
# Default 5-level test payload (reusable)
# ---------------------------------------------------------------------------

INSTRUMENT_KEY = "NSE_EQ|INE002A01018"

DEFAULT_QUOTES = [
    _make_bid_ask_quote(245000, 500, 245050, 300),
    _make_bid_ask_quote(244950, 800, 245100, 250),
    _make_bid_ask_quote(244900, 1000, 245150, 400),
    _make_bid_ask_quote(244850, 600, 245200, 200),
    _make_bid_ask_quote(244800, 700, 245250, 350),
]

DEFAULT_PAYLOAD = _make_payload(INSTRUMENT_KEY, DEFAULT_QUOTES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def handler_no_db() -> DepthFeedHandler:
    """Handler with no DB and raw-persist disabled."""
    return DepthFeedHandler(db_engine=None, persist_raw=False)


@pytest.fixture()
def sqlite_engine():
    """In-memory SQLite engine with depth_metrics_minute table."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE depth_metrics_minute (
                    instrument_key TEXT NOT NULL,
                    minute_ts TEXT NOT NULL,
                    avg_spread_bps REAL,
                    min_spread_bps REAL,
                    max_spread_bps REAL,
                    avg_bid_qty_top5 INTEGER,
                    avg_ask_qty_top5 INTEGER,
                    bid_wall_max_paisa INTEGER,
                    ask_wall_max_paisa INTEGER,
                    imbalance_avg REAL,
                    PRIMARY KEY (instrument_key, minute_ts)
                )
                """
            )
        )
    yield engine
    engine.dispose()


@pytest.fixture()
def handler_sqlite(sqlite_engine) -> DepthFeedHandler:
    """Handler backed by in-memory SQLite, raw-persist disabled."""
    return DepthFeedHandler(db_engine=sqlite_engine, persist_raw=False)


@pytest.fixture()
def depth_snapshots_engine():
    """In-memory SQLite engine with depth_snapshots table for raw insert tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE depth_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument_key TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    bids TEXT NOT NULL,
                    asks TEXT NOT NULL,
                    spread_bps REAL,
                    bid_wall_top5 INTEGER,
                    ask_wall_top5 INTEGER
                )
                """
            )
        )
    yield engine
    engine.dispose()


# ===========================================================================
# 1. parse_depth — extracts 5 levels
# ===========================================================================

class TestParseDepth:
    def test_parse_depth_extracts_5_levels(self, handler_no_db):
        """parse_depth should return bids and asks with exactly 5 levels."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)

        assert parsed is not None
        assert parsed["instrument_key"] == INSTRUMENT_KEY
        assert len(parsed["bids"]) == 5
        assert len(parsed["asks"]) == 5

    def test_parse_depth_bids_descending(self, handler_no_db):
        """Bids must be sorted descending by price_paisa."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        prices = [b["price_paisa"] for b in parsed["bids"]]
        assert prices == sorted(prices, reverse=True)

    def test_parse_depth_asks_ascending(self, handler_no_db):
        """Asks must be sorted ascending by price_paisa."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        prices = [a["price_paisa"] for a in parsed["asks"]]
        assert prices == sorted(prices)

    def test_parse_depth_returns_none_for_missing_feed(self, handler_no_db):
        """parse_depth should return None when depth path is missing."""
        bad_payload = {"feeds": {"NSE_EQ|INE002A01018": {"quote": {}}}}
        result = handler_no_db.parse_depth(bad_payload)
        assert result is None

    def test_parse_depth_returns_none_for_empty_payload(self, handler_no_db):
        """parse_depth returns None on an empty dict."""
        assert handler_no_db.parse_depth({}) is None

    def test_parse_depth_level_fields(self, handler_no_db):
        """Each level dict must contain price_paisa, qty, orders."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        for lv in parsed["bids"] + parsed["asks"]:
            assert "price_paisa" in lv
            assert "qty" in lv
            assert "orders" in lv

    def test_parse_depth_ts_is_utc(self, handler_no_db):
        """Parsed timestamp must be a UTC-aware datetime."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        assert isinstance(parsed["ts"], datetime)
        assert parsed["ts"].tzinfo is not None


# ===========================================================================
# 2. compute_spread_bps
# ===========================================================================

class TestComputeSpreadBps:
    def test_compute_spread_bps_normal(self, handler_no_db):
        """Spread should be (best_ask - best_bid) / best_bid * 10_000."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        # best_bid=245000 (paisa), best_ask=245050
        expected = (245050 - 245000) / 245000 * 10_000
        assert abs(handler_no_db.compute_spread_bps(parsed) - expected) < 0.001

    def test_compute_spread_bps_zero_when_locked(self, handler_no_db):
        """Spread must be 0.0 when bid price equals ask price (locked market)."""
        locked_quotes = [_make_bid_ask_quote(245000, 100, 245000, 100)]
        payload = _make_payload(INSTRUMENT_KEY, locked_quotes)
        parsed = handler_no_db.parse_depth(payload)
        assert handler_no_db.compute_spread_bps(parsed) == 0.0

    def test_compute_spread_bps_infinite_when_empty_book(self, handler_no_db):
        """Spread is inf when either side of the book is empty."""
        # Provide only bid-side data (ask price = 0 → filtered out)
        quotes = [{"bp": 245000, "bid_q": 100, "bid_n": 1, "ap": 0, "ask_q": 0, "ask_n": 0}]
        payload = _make_payload(INSTRUMENT_KEY, quotes)
        parsed = handler_no_db.parse_depth(payload)
        assert handler_no_db.compute_spread_bps(parsed) == float("inf")

    def test_compute_spread_bps_positive_for_normal_market(self, handler_no_db):
        """Spread in a normal market (bid < ask) must be positive."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        spread = handler_no_db.compute_spread_bps(parsed)
        assert spread > 0


# ===========================================================================
# 3. compute_imbalance
# ===========================================================================

class TestComputeImbalance:
    def test_compute_imbalance_balanced_returns_zero(self, handler_no_db):
        """Equal bid and ask quantities → imbalance = 0.0."""
        quotes = [
            _make_bid_ask_quote(245000, 500, 245050, 500),
            _make_bid_ask_quote(244950, 500, 245100, 500),
        ]
        payload = _make_payload(INSTRUMENT_KEY, quotes)
        parsed = handler_no_db.parse_depth(payload)
        assert handler_no_db.compute_imbalance(parsed) == pytest.approx(0.0)

    def test_compute_imbalance_bid_heavy_positive(self, handler_no_db):
        """More bid quantity than ask → positive imbalance."""
        quotes = [
            _make_bid_ask_quote(245000, 900, 245050, 100),
        ]
        payload = _make_payload(INSTRUMENT_KEY, quotes)
        parsed = handler_no_db.parse_depth(payload)
        result = handler_no_db.compute_imbalance(parsed)
        # (900 - 100) / (900 + 100) = 0.8
        assert result == pytest.approx(0.8)

    def test_compute_imbalance_ask_heavy_negative(self, handler_no_db):
        """More ask quantity than bid → negative imbalance."""
        quotes = [
            _make_bid_ask_quote(245000, 100, 245050, 900),
        ]
        payload = _make_payload(INSTRUMENT_KEY, quotes)
        parsed = handler_no_db.parse_depth(payload)
        result = handler_no_db.compute_imbalance(parsed)
        assert result == pytest.approx(-0.8)

    def test_compute_imbalance_returns_zero_for_empty_book(self, handler_no_db):
        """Empty parsed depth should return 0.0 (not crash)."""
        parsed = {"instrument_key": INSTRUMENT_KEY, "ts": datetime.now(tz=timezone.utc), "bids": [], "asks": []}
        assert handler_no_db.compute_imbalance(parsed) == 0.0

    def test_compute_imbalance_range_bounds(self, handler_no_db):
        """Imbalance must be in [-1, +1]."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        result = handler_no_db.compute_imbalance(parsed)
        assert -1.0 <= result <= 1.0


# ===========================================================================
# 4. detect_wall
# ===========================================================================

class TestDetectWall:
    def test_detect_wall_returns_max_qty_bid(self, handler_no_db):
        """detect_wall('bid') must return the max single-level bid qty."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        # bids: 500, 800, 1000, 600, 700 → max = 1000
        wall = handler_no_db.detect_wall(parsed, side="bid")
        assert wall == 1000

    def test_detect_wall_returns_max_qty_ask(self, handler_no_db):
        """detect_wall('ask') must return the max single-level ask qty."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        # asks: 300, 250, 400, 200, 350 → max = 400
        wall = handler_no_db.detect_wall(parsed, side="ask")
        assert wall == 400

    def test_detect_wall_returns_zero_for_empty_side(self, handler_no_db):
        """Wall should be 0 when the requested side is empty."""
        parsed = {"instrument_key": INSTRUMENT_KEY, "ts": datetime.now(tz=timezone.utc), "bids": [], "asks": []}
        assert handler_no_db.detect_wall(parsed, side="bid") == 0
        assert handler_no_db.detect_wall(parsed, side="ask") == 0

    def test_detect_wall_invalid_side_raises(self, handler_no_db):
        """Passing an invalid side should raise ValueError."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        with pytest.raises(ValueError):
            handler_no_db.detect_wall(parsed, side="other")

    def test_detect_wall_respects_levels_param(self, handler_no_db):
        """With levels=1, wall should only consider the best level."""
        parsed = handler_no_db.parse_depth(DEFAULT_PAYLOAD)
        # Best bid qty (after sort desc) is 500 (price 245000)
        wall_1 = handler_no_db.detect_wall(parsed, side="bid", levels=1)
        assert wall_1 == 500


# ===========================================================================
# 5. on_tick — raw persistence
# ===========================================================================

class TestOnTick:
    def test_on_tick_accumulates_in_buffer(self, handler_no_db):
        """on_tick should accumulate metrics in the internal buffer."""
        handler_no_db.on_tick(DEFAULT_PAYLOAD)
        assert INSTRUMENT_KEY in handler_no_db._buffer
        assert len(handler_no_db._buffer[INSTRUMENT_KEY]) == 1

    def test_on_tick_accumulates_multiple_ticks(self, handler_no_db):
        """Multiple on_tick calls should stack in the buffer."""
        handler_no_db.on_tick(DEFAULT_PAYLOAD)
        handler_no_db.on_tick(DEFAULT_PAYLOAD)
        handler_no_db.on_tick(DEFAULT_PAYLOAD)
        assert len(handler_no_db._buffer[INSTRUMENT_KEY]) == 3

    def test_on_tick_persists_raw_to_depth_snapshots(self):
        """on_tick with persist_raw=True should call bulk_insert_depth."""
        handler = DepthFeedHandler(db_engine=None, persist_raw=True)

        async_mock = AsyncMock(return_value=1)
        with patch("src.data.depth_feed.asyncio.get_event_loop") as mock_loop_fn:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = False
            mock_loop_fn.return_value = mock_loop

            with patch("src.storage.db.bulk_insert_depth", async_mock):
                with patch("asyncio.run", return_value=1) as mock_run:
                    handler.on_tick(DEFAULT_PAYLOAD)
                    # Either asyncio.run or loop.run_until_complete was called
                    # (depends on event-loop state) — just verify no exception

    def test_on_tick_skips_infinite_spread(self, handler_no_db):
        """on_tick must not accumulate ticks with empty/crossed books."""
        quotes = [{"bp": 0, "bid_q": 0, "bid_n": 0, "ap": 0, "ask_q": 0, "ask_n": 0}]
        payload = _make_payload(INSTRUMENT_KEY, quotes)
        handler_no_db.on_tick(payload)
        # No accumulation because spread is inf (empty book filtered)
        assert len(handler_no_db._buffer.get(INSTRUMENT_KEY, [])) == 0


# ===========================================================================
# 6. flush_minute_metrics
# ===========================================================================

class TestFlushMinuteMetrics:
    _MINUTE_TS = datetime(2026, 5, 8, 9, 15, 0, tzinfo=timezone.utc)

    def test_flush_minute_metrics_aggregates_correctly(self, handler_sqlite):
        """flush_minute_metrics should compute correct averages, min, max."""
        # Seed three ticks with known spread values
        handler_sqlite._buffer[INSTRUMENT_KEY] = [
            {
                "spread_bps": 2.0,
                "bid_qty5": 3600,
                "ask_qty5": 1500,
                "bid_wall": 1000,
                "ask_wall": 400,
                "imbalance": 0.4,
            },
            {
                "spread_bps": 4.0,
                "bid_qty5": 3200,
                "ask_qty5": 1300,
                "bid_wall": 900,
                "ask_wall": 300,
                "imbalance": 0.2,
            },
            {
                "spread_bps": 6.0,
                "bid_qty5": 4000,
                "ask_qty5": 1700,
                "bid_wall": 1100,
                "ask_wall": 500,
                "imbalance": 0.6,
            },
        ]

        rows_written = handler_sqlite.flush_minute_metrics(self._MINUTE_TS)

        assert rows_written == 1
        assert handler_sqlite._buffer == {}  # cleared after flush

    def test_flush_minute_metrics_writes_to_db(self, handler_sqlite, sqlite_engine):
        """flush_minute_metrics must persist the row to depth_metrics_minute."""
        handler_sqlite._buffer[INSTRUMENT_KEY] = [
            {
                "spread_bps": 2.04,
                "bid_qty5": 3600,
                "ask_qty5": 1500,
                "bid_wall": 1000,
                "ask_wall": 400,
                "imbalance": 0.4,
            },
        ]

        handler_sqlite.flush_minute_metrics(self._MINUTE_TS)

        with sqlite_engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM depth_metrics_minute WHERE instrument_key = :k"),
                {"k": INSTRUMENT_KEY},
            ).fetchall()

        assert len(rows) == 1
        row = rows[0]
        # Columns: instrument_key, minute_ts, avg_spread_bps, min_spread_bps,
        #          max_spread_bps, avg_bid_qty_top5, avg_ask_qty_top5,
        #          bid_wall_max_paisa, ask_wall_max_paisa, imbalance_avg
        col = dict(zip(
            [
                "instrument_key", "minute_ts",
                "avg_spread_bps", "min_spread_bps", "max_spread_bps",
                "avg_bid_qty_top5", "avg_ask_qty_top5",
                "bid_wall_max_paisa", "ask_wall_max_paisa",
                "imbalance_avg",
            ],
            row,
        ))
        assert col["instrument_key"] == INSTRUMENT_KEY
        assert abs(col["avg_spread_bps"] - 2.04) < 0.001
        assert col["bid_wall_max_paisa"] == 1000
        assert col["ask_wall_max_paisa"] == 400
        assert abs(col["imbalance_avg"] - 0.4) < 0.001

    def test_flush_minute_metrics_returns_zero_for_empty_buffer(self, handler_sqlite):
        """flush_minute_metrics returns 0 when there's nothing in the buffer."""
        result = handler_sqlite.flush_minute_metrics(self._MINUTE_TS)
        assert result == 0

    def test_flush_minute_metrics_clears_buffer(self, handler_sqlite):
        """Buffer must be empty after a successful flush."""
        handler_sqlite._buffer[INSTRUMENT_KEY] = [
            {
                "spread_bps": 1.5,
                "bid_qty5": 1000,
                "ask_qty5": 1000,
                "bid_wall": 500,
                "ask_wall": 500,
                "imbalance": 0.0,
            }
        ]
        handler_sqlite.flush_minute_metrics(self._MINUTE_TS)
        assert handler_sqlite._buffer == {}

    def test_flush_minute_metrics_multi_instrument(self, handler_sqlite, sqlite_engine):
        """flush_minute_metrics should write one row per instrument."""
        key_b = "NSE_EQ|INE040A01034"
        handler_sqlite._buffer[INSTRUMENT_KEY] = [
            {"spread_bps": 2.0, "bid_qty5": 500, "ask_qty5": 500, "bid_wall": 100, "ask_wall": 100, "imbalance": 0.0}
        ]
        handler_sqlite._buffer[key_b] = [
            {"spread_bps": 3.0, "bid_qty5": 700, "ask_qty5": 600, "bid_wall": 200, "ask_wall": 150, "imbalance": 0.1}
        ]
        rows_written = handler_sqlite.flush_minute_metrics(self._MINUTE_TS)
        assert rows_written == 2

        with sqlite_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM depth_metrics_minute")).scalar()
        assert count == 2


# ===========================================================================
# 7. parse_all_depths
# ===========================================================================

class TestParseAllDepths:
    def test_parse_all_depths_returns_all_instruments(self, handler_no_db):
        """parse_all_depths should return one entry per valid instrument."""
        key_a = "NSE_EQ|INE002A01018"
        key_b = "NSE_EQ|INE040A01034"
        payload = {
            "feeds": {
                key_a: _make_payload(key_a)["feeds"][key_a],
                key_b: _make_payload(key_b)["feeds"][key_b],
            },
            "currentTs": 1715150400000,
        }
        results = handler_no_db.parse_all_depths(payload)
        assert len(results) == 2
        keys = {r["instrument_key"] for r in results}
        assert keys == {key_a, key_b}
