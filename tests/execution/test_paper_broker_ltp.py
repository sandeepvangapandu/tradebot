"""Regression tests for PaperBroker LTP handling.

Locks in the fix from commit 66c19c8 (2026-05-14): the previous
implementation defaulted missing LTP to 10000 paisa (₹100), which
caused catastrophic mispricings (e.g. selling an ATM straddle for
₹95 when the option was actually worth ₹500+). The fix:
- _get_ltp returns 0 when no tick is cached
- place_order REJECTS MARKET orders when LTP is unavailable

These tests fail loudly if either behaviour regresses.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from src.execution.base_broker import (
    Order,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    ProductType,
)
from src.execution.paper_broker import PaperBroker


@pytest.fixture
def broker() -> PaperBroker:
    return PaperBroker(initial_capital=10_000_000, slippage_pct=0.0005)


def _mk_order(
    instrument_key: str,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: int = 1,
    price: int | None = None,
) -> Order:
    return Order(
        instrument_key=instrument_key,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        product_type=ProductType.MIS,
        strategy_id="TEST",
    )


class TestGetLtp:
    def test_returns_zero_for_unknown_instrument(self, broker: PaperBroker) -> None:
        """No tick cached → LTP must be 0, not a fake default."""
        assert broker.get_ltp("NSE_FO|99999") == 0

    def test_returns_cached_value(self, broker: PaperBroker) -> None:
        broker.update_ltp("NSE_FO|67204", 45_000)
        assert broker.get_ltp("NSE_FO|67204") == 45_000

    def test_zero_default_does_not_collide_with_real_zero(self, broker: PaperBroker) -> None:
        """A real explicit 0 update is indistinguishable from "no tick" — that is
        intentional: zero is not a valid market price, so place_order must
        reject in both cases."""
        broker.update_ltp("NSE_FO|67204", 0)
        assert broker.get_ltp("NSE_FO|67204") == 0


class TestPlaceOrderRejectsWithoutLtp:
    def test_market_order_rejected_when_no_ltp(self, broker: PaperBroker) -> None:
        order = _mk_order("NSE_FO|99999")
        resp: OrderResponse = broker.place_order(order)
        assert resp.status == OrderStatus.REJECTED
        assert resp.filled_quantity == 0
        assert "LTP" in (resp.message or "")

    def test_market_sell_rejected_when_no_ltp(self, broker: PaperBroker) -> None:
        order = _mk_order("NSE_FO|99999", side=OrderSide.SELL)
        resp = broker.place_order(order)
        assert resp.status == OrderStatus.REJECTED

    def test_market_order_fills_after_ltp_set(self, broker: PaperBroker) -> None:
        broker.update_ltp("NSE_FO|67204", 50_000)  # ₹500.00
        order = _mk_order("NSE_FO|67204")
        resp = broker.place_order(order)
        assert resp.status == OrderStatus.COMPLETE
        assert resp.filled_quantity == 1
        assert resp.average_price > 0
        # Fill must be near LTP (within slippage band)
        assert 49_000 < resp.average_price < 51_000

    def test_limit_order_does_not_need_cached_ltp(self, broker: PaperBroker) -> None:
        """LIMIT orders specify their own fill price; the cached LTP guard is
        only meant to catch MARKET orders firing before any tick arrives."""
        order = _mk_order(
            "NSE_FO|99999",
            order_type=OrderType.LIMIT,
            price=40_000,
        )
        resp = broker.place_order(order)
        # LIMIT should not be rejected purely for missing LTP.
        assert resp.status != OrderStatus.REJECTED or "LTP" not in (resp.message or "")
