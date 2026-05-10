"""Tests for src/execution/smart_router.py — Phase E.1.

All broker interactions are faked with a MagicMock that exposes:
    place_order(order_dict)  -> dict with order_id + filled_quantity
    cancel_order(order_id)   -> dict
    get_ltp(instrument_key)  -> int (paisa)

No real network / DB connections required.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call, patch

import pytest

from src.execution.smart_router import (
    RoutingConfig,
    RoutingStrategy,
    SmartOrderRouter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_broker(
    *,
    ltp: int = 2_50_000,           # 2500 rupees in paisa
    fill_qty: int | None = None,   # None → fill full quantity placed
    order_id_prefix: str = "ord",
) -> MagicMock:
    """Return a mock broker that fills orders fully by default."""
    broker = MagicMock()
    _call_count = {"n": 0}

    def _place_order(order_dict: dict) -> dict:
        _call_count["n"] += 1
        oid = f"{order_id_prefix}_{_call_count['n']}"
        qty = order_dict.get("quantity", 1)
        filled = fill_qty if fill_qty is not None else qty
        return {"order_id": oid, "filled_quantity": filled, "status": "COMPLETE"}

    broker.place_order.side_effect = _place_order
    broker.cancel_order.return_value = {"status": "CANCELLED"}
    broker.get_ltp.return_value = ltp
    return broker


def _make_order(**kwargs) -> dict:
    defaults = {
        "client_order_id": "parent-001",
        "instrument_key": "NSE_EQ|INE002A01018",
        "side": "BUY",
        "quantity": 100,
        "lot_size": 1,
        "signal_type": "ENTRY",
    }
    defaults.update(kwargs)
    return defaults


def _make_router(broker, **cfg_kwargs) -> SmartOrderRouter:
    cfg = RoutingConfig(**cfg_kwargs)
    return SmartOrderRouter(broker=broker, db_engine=None, config=cfg)


# ---------------------------------------------------------------------------
# Strategy selection tests
# ---------------------------------------------------------------------------

class TestSelectStrategy:
    """test_select_strategy_* tests."""

    def test_select_strategy_small_returns_limit_protect(self):
        """Small order (< iceberg threshold) → LIMIT_PROTECT."""
        broker = _make_broker()
        router = _make_router(broker, iceberg_threshold_lots=10, twap_threshold_lots=50)
        strategy = router.select_strategy(qty=5, lot_size=1, urgency="normal")
        assert strategy == RoutingStrategy.LIMIT_PROTECT

    def test_select_strategy_medium_returns_iceberg(self):
        """Medium order (> iceberg, < TWAP threshold) → ICEBERG."""
        broker = _make_broker()
        router = _make_router(broker, iceberg_threshold_lots=10, twap_threshold_lots=50)
        strategy = router.select_strategy(qty=20, lot_size=1, urgency="normal")
        assert strategy == RoutingStrategy.ICEBERG

    def test_select_strategy_large_returns_twap(self):
        """Large order (> TWAP threshold) → TWAP."""
        broker = _make_broker()
        router = _make_router(broker, iceberg_threshold_lots=10, twap_threshold_lots=50)
        strategy = router.select_strategy(qty=60, lot_size=1, urgency="normal")
        assert strategy == RoutingStrategy.TWAP

    def test_select_strategy_urgent_returns_market(self):
        """urgency='urgent' always returns MARKET, regardless of size."""
        broker = _make_broker()
        router = _make_router(broker)
        for qty in (1, 20, 100, 1000):
            assert router.select_strategy(qty=qty, lot_size=1, urgency="urgent") == RoutingStrategy.MARKET

    def test_select_strategy_lot_size_scaling(self):
        """Lot-size scaling: 20 contracts at lot_size=5 → 4 lots → LIMIT_PROTECT."""
        broker = _make_broker()
        router = _make_router(broker, iceberg_threshold_lots=10, twap_threshold_lots=50)
        # 20 contracts / lot_size 5 = 4 lots → below iceberg threshold
        assert router.select_strategy(qty=20, lot_size=5, urgency="normal") == RoutingStrategy.LIMIT_PROTECT

    def test_select_strategy_at_iceberg_boundary(self):
        """qty_lots exactly == iceberg_threshold → LIMIT_PROTECT (not >)."""
        broker = _make_broker()
        router = _make_router(broker, iceberg_threshold_lots=10, twap_threshold_lots=50)
        # 10 lots exactly — threshold is >, not >=
        assert router.select_strategy(qty=10, lot_size=1, urgency="normal") == RoutingStrategy.LIMIT_PROTECT

    def test_select_strategy_just_above_iceberg_boundary(self):
        """qty_lots 11 > iceberg_threshold 10 → ICEBERG."""
        broker = _make_broker()
        router = _make_router(broker, iceberg_threshold_lots=10, twap_threshold_lots=50)
        assert router.select_strategy(qty=11, lot_size=1, urgency="normal") == RoutingStrategy.ICEBERG


# ---------------------------------------------------------------------------
# execute_market tests
# ---------------------------------------------------------------------------

class TestExecuteMarket:
    """test_execute_market_* tests."""

    def test_execute_market_places_single_market_order(self):
        """execute_market() calls place_order once with order_type=MARKET."""
        broker = _make_broker(fill_qty=100)
        router = _make_router(broker)
        order = _make_order(quantity=100)

        result = router.execute_market(order)

        assert broker.place_order.call_count == 1
        placed = broker.place_order.call_args[0][0]
        assert placed["order_type"] == "MARKET"

    def test_execute_market_returns_filled_status(self):
        """execute_market() status is FILLED when broker fills full qty."""
        broker = _make_broker(fill_qty=50)
        router = _make_router(broker)
        result = router.execute_market(_make_order(quantity=50))

        assert result["status"] == "FILLED"
        assert result["total_filled_qty"] == 50

    def test_execute_market_partial_fill(self):
        """execute_market() status is PARTIAL when broker fills less than qty."""
        broker = _make_broker(fill_qty=30)
        router = _make_router(broker)
        result = router.execute_market(_make_order(quantity=100))

        assert result["status"] == "PARTIAL"
        assert result["total_filled_qty"] == 30

    def test_execute_market_returns_child_order_id(self):
        """execute_market() result includes child_order_ids list."""
        broker = _make_broker()
        router = _make_router(broker)
        result = router.execute_market(_make_order(quantity=10))

        assert len(result["child_order_ids"]) == 1
        assert result["child_order_ids"][0].startswith("ord_")


# ---------------------------------------------------------------------------
# execute_limit_protect tests
# ---------------------------------------------------------------------------

class TestExecuteLimitProtect:
    """test_execute_limit_protect_* tests."""

    def test_execute_limit_protect_places_limit_at_ltp_plus_bps(self):
        """For BUY, limit price = LTP + protect_bps * LTP / 10000."""
        ltp = 2_00_000  # 2000 rupees
        broker = _make_broker(ltp=ltp, fill_qty=100)
        router = _make_router(broker, limit_protect_bps=5.0)
        order = _make_order(quantity=100, side="BUY")

        router.execute_limit_protect(order)

        placed = broker.place_order.call_args[0][0]
        expected_bps_offset = int(ltp * 5.0 / 10_000)  # 100 paisa = 1 rupee
        assert placed["price"] == ltp + expected_bps_offset
        assert placed["order_type"] == "LIMIT"

    def test_execute_limit_protect_sell_lower_price(self):
        """For SELL, limit price = LTP - protect_bps."""
        ltp = 3_00_000  # 3000 rupees
        broker = _make_broker(ltp=ltp, fill_qty=50)
        router = _make_router(broker, limit_protect_bps=5.0)
        order = _make_order(quantity=50, side="SELL")

        router.execute_limit_protect(order)

        placed = broker.place_order.call_args[0][0]
        expected_bps_offset = int(ltp * 5.0 / 10_000)
        assert placed["price"] == ltp - expected_bps_offset

    def test_execute_limit_protect_retries_on_no_fill(self):
        """Unfilled limit orders trigger cancel + reprice up to max_retries."""
        broker = _make_broker(ltp=1_00_000, fill_qty=0)
        router = _make_router(
            broker,
            limit_protect_bps=5.0,
            limit_protect_max_retries=2,
        )
        order = _make_order(quantity=100)

        result = router.execute_limit_protect(order)

        # Should have called place_order max_retries+1 times (initial + 2 retries)
        # On the last attempt falls to MARKET
        assert broker.place_order.call_count >= 2
        # cancel_order called for each failed limit attempt
        assert broker.cancel_order.call_count >= 1

    def test_execute_limit_protect_falls_to_market_after_max_retries(self):
        """After max_retries of no-fill, falls to a MARKET order."""
        broker = _make_broker(ltp=1_00_000, fill_qty=0)
        router = _make_router(
            broker,
            limit_protect_max_retries=2,
        )
        # Override: after max_retries attempts at LIMIT (all unfilled), the last
        # call should be MARKET.  We wire the broker to fill on the MARKET call.
        call_seq = {"n": 0}
        max_retries = 2

        def _place_order(order_dict):
            call_seq["n"] += 1
            oid = f"ord_{call_seq['n']}"
            if order_dict.get("order_type") == "MARKET":
                return {"order_id": oid, "filled_quantity": 100, "status": "COMPLETE"}
            return {"order_id": oid, "filled_quantity": 0, "status": "OPEN"}

        broker.place_order.side_effect = _place_order
        order = _make_order(quantity=100)
        result = router.execute_limit_protect(order)

        # Should eventually be FILLED via the market fallback
        assert result["status"] == "FILLED"
        assert result["total_filled_qty"] == 100
        # At least one MARKET order placed
        market_calls = [
            c for c in broker.place_order.call_args_list
            if c[0][0].get("order_type") == "MARKET"
        ]
        assert len(market_calls) >= 1

    def test_execute_limit_protect_immediate_fill(self):
        """Immediate fill on first LIMIT attempt → no cancel called."""
        broker = _make_broker(ltp=50_000, fill_qty=200)
        router = _make_router(broker)
        result = router.execute_limit_protect(_make_order(quantity=200))

        assert result["status"] == "FILLED"
        assert result["total_filled_qty"] == 200
        broker.cancel_order.assert_not_called()


# ---------------------------------------------------------------------------
# execute_iceberg tests
# ---------------------------------------------------------------------------

class TestExecuteIceberg:
    """test_execute_iceberg_* tests."""

    def test_execute_iceberg_slices_into_n_chunks(self):
        """execute_iceberg() places exactly N slice orders."""
        broker = _make_broker(ltp=1_00_000, fill_qty=20)
        router = _make_router(broker, limit_protect_max_retries=0)
        order = _make_order(quantity=100)

        result = router.execute_iceberg(order, slice_count=5)

        # 5 slices, each calls execute_limit_protect → at least 5 place_order calls
        assert len(result["slices"]) == 5
        assert result["status"] == "FILLED"

    def test_execute_iceberg_last_slice_absorbs_remainder(self):
        """With 3 slices for qty=100 → [33, 33, 34]."""
        broker = _make_broker(ltp=1_00_000, fill_qty=100)  # always fills full
        # Override to capture intended_qty per slice
        slice_qtys = []

        def _place_order(order_dict):
            slice_qtys.append(order_dict["quantity"])
            return {"order_id": str(uuid.uuid4()), "filled_quantity": order_dict["quantity"]}

        broker.place_order.side_effect = _place_order
        router = _make_router(broker, limit_protect_max_retries=0)
        order = _make_order(quantity=100)

        result = router.execute_iceberg(order, slice_count=3)

        assert len(slice_qtys) == 3
        assert slice_qtys[0] == 33
        assert slice_qtys[1] == 33
        assert slice_qtys[2] == 34   # remainder absorbed in last slice
        assert sum(slice_qtys) == 100

    def test_execute_iceberg_total_filled_matches_sum(self):
        """total_filled_qty == sum of all slice fills."""
        broker = _make_broker(ltp=2_00_000, fill_qty=10)  # fills 10 per call
        router = _make_router(broker, limit_protect_max_retries=0)
        order = _make_order(quantity=50)

        result = router.execute_iceberg(order, slice_count=5)

        assert result["total_filled_qty"] == sum(s["filled_qty"] for s in result["slices"])


# ---------------------------------------------------------------------------
# execute_twap tests
# ---------------------------------------------------------------------------

class TestExecuteTwap:
    """test_execute_twap_* tests."""

    def test_execute_twap_distributes_over_duration(self):
        """execute_twap() uses iceberg_slice_count slices."""
        broker = _make_broker(ltp=1_00_000, fill_qty=20)
        router = _make_router(broker, iceberg_slice_count=4, limit_protect_max_retries=0)
        order = _make_order(quantity=80)

        result = router.execute_twap(order, duration_seconds=400)

        assert len(result["slices"]) == 4
        assert all(s["routing_strategy"] == RoutingStrategy.TWAP.value for s in result["slices"])

    def test_execute_twap_metadata_contains_interval(self):
        """Each TWAP slice metadata records duration and interval_seconds."""
        broker = _make_broker(ltp=1_00_000, fill_qty=25)
        router = _make_router(broker, iceberg_slice_count=4, limit_protect_max_retries=0)
        order = _make_order(quantity=100)

        result = router.execute_twap(order, duration_seconds=600)

        for sl in result["slices"]:
            meta = sl["routing_metadata"]
            assert meta["duration_seconds"] == 600
            assert "interval_seconds" in meta

    def test_execute_twap_last_slice_absorbs_remainder(self):
        """qty=103, 4 slices → [25, 25, 25, 28]."""
        slice_qtys = []

        broker = _make_broker(ltp=1_00_000)

        def _place_order(order_dict):
            qty = order_dict["quantity"]
            slice_qtys.append(qty)
            return {"order_id": str(uuid.uuid4()), "filled_quantity": qty}

        broker.place_order.side_effect = _place_order
        router = _make_router(broker, iceberg_slice_count=4, limit_protect_max_retries=0)
        order = _make_order(quantity=103)

        router.execute_twap(order, duration_seconds=400)

        assert slice_qtys[-1] == 28   # 103 - 3*25 = 28
        assert sum(slice_qtys) == 103

    def test_execute_twap_total_filled_accumulated(self):
        """total_filled_qty sums across all slices."""
        broker = _make_broker(ltp=5_00_000, fill_qty=25)
        router = _make_router(broker, iceberg_slice_count=4, limit_protect_max_retries=0)
        result = router.execute_twap(_make_order(quantity=100), duration_seconds=200)

        assert result["total_filled_qty"] == sum(s["filled_qty"] for s in result["slices"])


# ---------------------------------------------------------------------------
# route() integration tests
# ---------------------------------------------------------------------------

class TestRoute:
    """Integration tests for the top-level route() method."""

    def test_route_small_uses_limit_protect(self):
        """route() picks LIMIT_PROTECT for small orders."""
        broker = _make_broker(fill_qty=5)
        router = _make_router(broker, iceberg_threshold_lots=10, twap_threshold_lots=50)
        order = _make_order(quantity=5)
        result = router.route(order)

        assert result["routing_strategy"] == RoutingStrategy.LIMIT_PROTECT.value

    def test_route_returns_parent_order_id(self):
        """route() result always contains parent_order_id."""
        broker = _make_broker(fill_qty=10)
        router = _make_router(broker)
        order = _make_order(client_order_id="signal-xyz", quantity=10)
        result = router.route(order)

        assert result["parent_order_id"] == "signal-xyz"

    def test_route_urgent_uses_market(self):
        """route() with urgency='urgent' dispatches MARKET regardless of qty."""
        broker = _make_broker(fill_qty=200)
        router = _make_router(broker, iceberg_threshold_lots=10)
        order = _make_order(quantity=200)
        result = router.route(order, urgency="urgent")

        assert result["routing_strategy"] == RoutingStrategy.MARKET.value

    def test_route_persists_to_routed_orders(self):
        """route() calls _persist_result when db_engine is provided."""
        broker = _make_broker(fill_qty=10)
        cfg = RoutingConfig(iceberg_threshold_lots=10, twap_threshold_lots=50)

        db_mock = MagicMock()
        router = SmartOrderRouter(broker=broker, db_engine=db_mock, config=cfg)

        # Patch _persist_result to verify it is called
        with patch.object(router, "_persist_result") as mock_persist:
            result = router.route(_make_order(quantity=5))
            assert mock_persist.call_count == 1

    def test_route_no_db_skips_persist(self):
        """route() with db_engine=None does not call _persist_result."""
        broker = _make_broker(fill_qty=10)
        router = _make_router(broker)

        with patch.object(router, "_persist_result") as mock_persist:
            router.route(_make_order(quantity=5))
            mock_persist.assert_not_called()


# ---------------------------------------------------------------------------
# get_routing_stats tests
# ---------------------------------------------------------------------------

class TestGetRoutingStats:
    """test_get_routing_stats_* tests."""

    def test_get_routing_stats_no_db_returns_zeros(self):
        """Without a DB engine, stats return empty/zero values."""
        broker = _make_broker()
        router = _make_router(broker)
        stats = router.get_routing_stats()

        assert stats["total_routed"] == 0
        assert stats["by_strategy"] == {}
        assert stats["filled_qty"] == 0
        assert stats["partial_pct"] == 0.0

    def test_get_routing_stats_aggregates_by_strategy(self):
        """With a mock DB, stats are aggregated per routing_strategy."""
        broker = _make_broker()
        db_mock = MagicMock()

        # Simulate two rows returned: LIMIT_PROTECT×3, ICEBERG×2
        fake_rows = [
            ("LIMIT_PROTECT", 3, 300, 0),
            ("ICEBERG", 2, 200, 1),
        ]
        conn_ctx = MagicMock()
        conn_ctx.__enter__ = MagicMock(return_value=conn_ctx)
        conn_ctx.__exit__ = MagicMock(return_value=False)
        conn_ctx.execute.return_value.fetchall.return_value = fake_rows
        db_mock.connect.return_value = conn_ctx

        router = SmartOrderRouter(broker=broker, db_engine=db_mock)
        stats = router.get_routing_stats(days=7)

        assert stats["total_routed"] == 5
        assert stats["by_strategy"]["LIMIT_PROTECT"] == 3
        assert stats["by_strategy"]["ICEBERG"] == 2
        assert stats["filled_qty"] == 500

    def test_get_routing_stats_partial_pct_computed(self):
        """partial_pct is partial_count / total * 100."""
        broker = _make_broker()
        db_mock = MagicMock()

        # 4 orders, 1 partial
        fake_rows = [("MARKET", 4, 400, 1)]
        conn_ctx = MagicMock()
        conn_ctx.__enter__ = MagicMock(return_value=conn_ctx)
        conn_ctx.__exit__ = MagicMock(return_value=False)
        conn_ctx.execute.return_value.fetchall.return_value = fake_rows
        db_mock.connect.return_value = conn_ctx

        router = SmartOrderRouter(broker=broker, db_engine=db_mock)
        stats = router.get_routing_stats()

        assert stats["partial_pct"] == pytest.approx(25.0, rel=1e-3)

    def test_get_routing_stats_db_error_returns_safe_defaults(self):
        """DB errors return safe zero-filled dict without raising."""
        broker = _make_broker()
        db_mock = MagicMock()
        db_mock.connect.side_effect = Exception("connection refused")

        router = SmartOrderRouter(broker=broker, db_engine=db_mock)
        stats = router.get_routing_stats()

        assert stats["total_routed"] == 0
        assert "error" in stats
