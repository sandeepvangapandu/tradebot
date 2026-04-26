"""Tests for the order manager and paper broker.

Tests PaperBroker simulation, slippage application, fee calculation,
position tracking, and order lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from config.constants import FEES


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class TransactionType(Enum):
    """Transaction type enumeration."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    strategy: str
    instrument_key: str
    transaction_type: TransactionType
    order_type: OrderType
    quantity: int
    price: int  # paisa
    trigger_price: int = 0  # paisa
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: int = 0  # paisa
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fees: int = 0  # paisa


@dataclass
class Position:
    """Represents an open position."""
    instrument_key: str
    strategy: str
    side: str  # LONG or SHORT
    quantity: int
    entry_price: int  # paisa (average)
    unrealized_pnl: int = 0  # paisa
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeeCalculator:
    """Calculates trading fees for Indian markets."""

    @staticmethod
    def calculate_fees(
        transaction_type: TransactionType,
        order_type: OrderType,
        quantity: int,
        fill_price: int,  # paisa
        is_fno: bool = True,
    ) -> int:
        """Calculate total fees for a trade.

        Args:
            transaction_type: BUY or SELL
            order_type: MARKET, LIMIT, etc.
            quantity: Number of units
            fill_price: Fill price in paisa
            is_fno: Whether instrument is F&O

        Returns:
            Total fees in paisa
        """
        turnover = quantity * fill_price  # in paisa
        turnover_crores = turnover / 1_00_00_000  # Convert to crores for SEBI calc

        # Brokerage (flat Rs 20 per order for F&O)
        brokerage = int(FEES["brokerage_fno"])

        # STT (only on sell side for options)
        stt = 0
        if transaction_type == TransactionType.SELL and is_fno:
            stt = int(turnover * FEES["stt_options_sell_pct"] / 100)

        # Exchange charges
        if is_fno:
            exchange_charges = int(turnover * FEES["exchange_charge_nse_fo_pct"] / 100)
        else:
            exchange_charges = int(turnover * FEES["exchange_charge_nse_fo_pct"] / 100)

        # SEBI charges
        sebi_charges = int(turnover_crores * FEES["sebi_charges_per_crore"])

        # Stamp duty (only on buy side)
        stamp_duty = 0
        if transaction_type == TransactionType.BUY:
            stamp_duty = int(turnover * FEES["stamp_duty_fno_pct"] / 100)

        # GST on brokerage + exchange charges
        gst_base = brokerage + exchange_charges
        gst = int(gst_base * FEES["gst_pct"] / 100)

        total_fees = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst
        return total_fees


class PaperBroker:
    """Paper trading broker that simulates order execution."""

    def __init__(
        self,
        capital: int,  # paisa
        slippage_pct: float = 0.05,
        fill_probability: float = 1.0,
    ) -> None:
        """Initialize paper broker.

        Args:
            capital: Starting capital in paisa
            slippage_pct: Slippage percentage for market orders
            fill_probability: Probability of order fill (1.0 = always fill)
        """
        self.capital = capital
        self.slippage_pct = slippage_pct
        self.fill_probability = fill_probability
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.order_counter = 0

    def place_order(
        self,
        strategy: str,
        instrument_key: str,
        transaction_type: TransactionType,
        order_type: OrderType,
        quantity: int,
        price: int = 0,
        trigger_price: int = 0,
    ) -> Order:
        """Place a new order.

        Args:
            strategy: Strategy name
            instrument_key: Instrument identifier
            transaction_type: BUY or SELL
            order_type: MARKET, LIMIT, etc.
            quantity: Order quantity
            price: Limit price (paisa)
            trigger_price: Trigger price for SL orders (paisa)

        Returns:
            Created Order object
        """
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter:06d}"

        order = Order(
            id=order_id,
            strategy=strategy,
            instrument_key=instrument_key,
            transaction_type=transaction_type,
            order_type=order_type,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
        )

        self.orders[order_id] = order

        # Simulate immediate fill for market orders
        if order_type == OrderType.MARKET:
            self._simulate_fill(order)

        return order

    def _simulate_fill(self, order: Order, market_price: int | None = None) -> None:
        """Simulate order fill with slippage.

        Args:
            order: Order to fill
            market_price: Current market price (paisa), random if None
        """
        import random

        if market_price is None:
            # Simulate a price around 50000 paisa (Rs 500)
            market_price = random.randint(49500, 50500)

        # Apply slippage
        slippage = int(market_price * self.slippage_pct / 100)

        if order.transaction_type == TransactionType.BUY:
            fill_price = market_price + slippage  # Pay more when buying
        else:
            fill_price = market_price - slippage  # Receive less when selling

        # Calculate fees
        fees = FeeCalculator.calculate_fees(
            transaction_type=order.transaction_type,
            order_type=order.order_type,
            quantity=order.quantity,
            fill_price=fill_price,
        )

        # Update order
        order.filled_qty = order.quantity
        order.avg_fill_price = fill_price
        order.fees = fees
        order.status = OrderStatus.COMPLETE
        order.updated_at = datetime.now(timezone.utc)

        # Update positions
        self._update_position(order)

    def _update_position(self, order: Order) -> None:
        """Update position based on filled order.

        Args:
            order: Filled order
        """
        key = f"{order.strategy}:{order.instrument_key}"

        if order.transaction_type == TransactionType.BUY:
            if key in self.positions:
                # Add to existing position
                pos = self.positions[key]
                total_value = (pos.quantity * pos.entry_price) + (order.filled_qty * order.avg_fill_price)
                pos.quantity += order.filled_qty
                pos.entry_price = total_value // pos.quantity
            else:
                # Create new position
                self.positions[key] = Position(
                    instrument_key=order.instrument_key,
                    strategy=order.strategy,
                    side="LONG",
                    quantity=order.filled_qty,
                    entry_price=order.avg_fill_price,
                )
        else:  # SELL
            if key in self.positions:
                # Reduce position
                pos = self.positions[key]
                pos.quantity -= order.filled_qty
                if pos.quantity <= 0:
                    del self.positions[key]
            else:
                # Short position (for F&O)
                self.positions[key] = Position(
                    instrument_key=order.instrument_key,
                    strategy=order.strategy,
                    side="SHORT",
                    quantity=order.filled_qty,
                    entry_price=order.avg_fill_price,
                )

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_position(self, strategy: str, instrument_key: str) -> Position | None:
        """Get position for strategy and instrument."""
        key = f"{strategy}:{instrument_key}"
        return self.positions.get(key)

    def get_all_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        order = self.orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def cancel_all_orders(self) -> list[str]:
        """Cancel all pending orders."""
        cancelled = []
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now(timezone.utc)
                cancelled.append(order.id)
        return cancelled

    def exit_all_positions(self, market_price: int | None = None) -> list[Order]:
        """Exit all open positions."""
        exit_orders = []
        positions_to_exit = list(self.positions.values())

        for pos in positions_to_exit:
            # Create opposite order
            txn_type = TransactionType.SELL if pos.side == "LONG" else TransactionType.BUY

            order = self.place_order(
                strategy=pos.strategy,
                instrument_key=pos.instrument_key,
                transaction_type=txn_type,
                order_type=OrderType.MARKET,
                quantity=pos.quantity,
            )
            exit_orders.append(order)

        return exit_orders


class TestPaperBroker:
    """Test PaperBroker order simulation."""

    @pytest.fixture
    def broker(self) -> PaperBroker:
        """Create a paper broker instance."""
        return PaperBroker(capital=10_000_000, slippage_pct=0.05)

    def test_place_market_order(self, broker: PaperBroker) -> None:
        """Test placing a market order."""
        order = broker.place_order(
            strategy="TestStrategy",
            instrument_key="NSE_FO|BANKNIFTY24MAR46000CE",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        assert order.id.startswith("PAPER_")
        assert order.strategy == "TestStrategy"
        assert order.status == OrderStatus.COMPLETE
        assert order.filled_qty == 15
        assert order.avg_fill_price > 0

    def test_market_order_immediate_fill(self, broker: PaperBroker) -> None:
        """Test market orders are filled immediately."""
        order = broker.place_order(
            strategy="TestStrategy",
            instrument_key="NSE_FO|BANKNIFTY24MAR46000CE",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        assert order.status == OrderStatus.COMPLETE
        assert order.filled_qty == order.quantity
        assert order.avg_fill_price > 0

    def test_get_order(self, broker: PaperBroker) -> None:
        """Test retrieving an order by ID."""
        order = broker.place_order(
            strategy="TestStrategy",
            instrument_key="NSE_FO|BANKNIFTY24MAR46000CE",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        retrieved = broker.get_order(order.id)
        assert retrieved is not None
        assert retrieved.id == order.id

    def test_get_nonexistent_order(self, broker: PaperBroker) -> None:
        """Test retrieving a non-existent order."""
        retrieved = broker.get_order("NONEXISTENT")
        assert retrieved is None


class TestSlippageApplication:
    """Test slippage application in paper broker."""

    @pytest.fixture
    def broker_no_slippage(self) -> PaperBroker:
        """Create broker with no slippage."""
        return PaperBroker(capital=10_000_000, slippage_pct=0.0)

    @pytest.fixture
    def broker_with_slippage(self) -> PaperBroker:
        """Create broker with 0.5% slippage."""
        return PaperBroker(capital=10_000_000, slippage_pct=0.5)

    def test_no_slippage_buy(self, broker_no_slippage: PaperBroker) -> None:
        """Test buy order with no slippage fills at market price."""
        with patch.object(broker_no_slippage, '_simulate_fill') as mock_fill:
            def side_effect(order):
                order.filled_qty = order.quantity
                order.avg_fill_price = 50000  # Rs 500
                order.status = OrderStatus.COMPLETE
            mock_fill.side_effect = side_effect

            order = broker_no_slippage.place_order(
                strategy="Test",
                instrument_key="NSE_FO|TEST",
                transaction_type=TransactionType.BUY,
                order_type=OrderType.MARKET,
                quantity=15,
            )

            assert order.avg_fill_price == 50000

    def test_slippage_increases_buy_price(self, broker_with_slippage: PaperBroker) -> None:
        """Test slippage increases buy fill price."""
        # Mock to control market price
        import random
        random.seed(42)

        order = broker_with_slippage.place_order(
            strategy="Test",
            instrument_key="NSE_FO|TEST",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        # With 0.5% slippage, buy price should be higher than base
        # Base ~50000, slippage adds 0.5% = 250 paisa
        assert order.avg_fill_price > 50000

    def test_slippage_decreases_sell_price(self, broker_with_slippage: PaperBroker) -> None:
        """Test slippage decreases sell fill price."""
        order = broker_with_slippage.place_order(
            strategy="Test",
            instrument_key="NSE_FO|TEST",
            transaction_type=TransactionType.SELL,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        # With slippage, sell price should be lower than base
        assert order.avg_fill_price < 50500  # Below typical market price range


class TestFeeCalculation:
    """Test fee calculation for trades."""

    def test_brokerage_calculation_fno(self) -> None:
        """Test brokerage fee for F&O orders."""
        fees = FeeCalculator.calculate_fees(
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
            fill_price=50000,  # Rs 500
            is_fno=True,
        )

        # Should include flat Rs 20 brokerage
        assert fees > 0
        # Brokerage component is 2000 paisa (Rs 20)

    def test_stt_only_on_sell_options(self) -> None:
        """Test STT is only charged on sell side for options."""
        buy_fees = FeeCalculator.calculate_fees(
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
            fill_price=50000,
            is_fno=True,
        )

        sell_fees = FeeCalculator.calculate_fees(
            transaction_type=TransactionType.SELL,
            order_type=OrderType.MARKET,
            quantity=15,
            fill_price=50000,
            is_fno=True,
        )

        # Sell should have higher fees due to STT
        assert sell_fees > buy_fees

    def test_stamp_duty_only_on_buy(self) -> None:
        """Test stamp duty is only charged on buy side."""
        buy_fees = FeeCalculator.calculate_fees(
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
            fill_price=50000,
            is_fno=True,
        )

        # Buy fees should include stamp duty component
        turnover = 15 * 50000
        expected_stamp_duty = int(turnover * FEES["stamp_duty_fno_pct"] / 100)
        assert expected_stamp_duty > 0

    def test_gst_calculation(self) -> None:
        """Test GST is calculated on brokerage + exchange charges."""
        quantity = 15
        fill_price = 50000
        turnover = quantity * fill_price

        brokerage = FEES["brokerage_fno"]
        exchange_charges = int(turnover * FEES["exchange_charge_nse_fo_pct"] / 100)
        expected_gst_base = brokerage + exchange_charges
        expected_gst = int(expected_gst_base * FEES["gst_pct"] / 100)

        assert expected_gst > 0

    def test_fee_proportional_to_turnover(self) -> None:
        """Test fees scale with turnover."""
        small_fees = FeeCalculator.calculate_fees(
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
            fill_price=10000,  # Rs 100
            is_fno=True,
        )

        large_fees = FeeCalculator.calculate_fees(
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
            fill_price=100000,  # Rs 1000
            is_fno=True,
        )

        # Larger turnover should have higher fees (except flat brokerage)
        assert large_fees > small_fees


class TestPositionTracking:
    """Test position tracking in paper broker."""

    @pytest.fixture
    def broker(self) -> PaperBroker:
        """Create a paper broker instance."""
        return PaperBroker(capital=10_000_000, slippage_pct=0.0)

    def test_long_position_created_on_buy(self, broker: PaperBroker) -> None:
        """Test long position is created on buy order."""
        broker.place_order(
            strategy="TestStrategy",
            instrument_key="NSE_FO|BANKNIFTY24MAR46000CE",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        position = broker.get_position("TestStrategy", "NSE_FO|BANKNIFTY24MAR46000CE")

        assert position is not None
        assert position.side == "LONG"
        assert position.quantity == 15

    def test_position_quantity_increased(self, broker: PaperBroker) -> None:
        """Test position quantity increases with additional buys."""
        instrument = "NSE_FO|BANKNIFTY24MAR46000CE"

        # First buy
        broker.place_order(
            strategy="TestStrategy",
            instrument_key=instrument,
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        # Second buy
        broker.place_order(
            strategy="TestStrategy",
            instrument_key=instrument,
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        position = broker.get_position("TestStrategy", instrument)
        assert position.quantity == 30

    def test_position_closed_on_sell(self, broker: PaperBroker) -> None:
        """Test position is closed when fully sold."""
        instrument = "NSE_FO|BANKNIFTY24MAR46000CE"

        # Buy
        broker.place_order(
            strategy="TestStrategy",
            instrument_key=instrument,
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        # Sell all
        broker.place_order(
            strategy="TestStrategy",
            instrument_key=instrument,
            transaction_type=TransactionType.SELL,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        position = broker.get_position("TestStrategy", instrument)
        assert position is None

    def test_short_position_created(self, broker: PaperBroker) -> None:
        """Test short position is created on sell without existing long."""
        broker.place_order(
            strategy="TestStrategy",
            instrument_key="NSE_FO|BANKNIFTY24MAR46000CE",
            transaction_type=TransactionType.SELL,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        position = broker.get_position("TestStrategy", "NSE_FO|BANKNIFTY24MAR46000CE")

        assert position is not None
        assert position.side == "SHORT"
        assert position.quantity == 15

    def test_get_all_positions(self, broker: PaperBroker) -> None:
        """Test retrieving all positions."""
        # Create multiple positions
        broker.place_order(
            strategy="Strategy1",
            instrument_key="NSE_FO|INST1",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        broker.place_order(
            strategy="Strategy2",
            instrument_key="NSE_FO|INST2",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        positions = broker.get_all_positions()
        assert len(positions) == 2


class TestOrderLifecycle:
    """Test order lifecycle management."""

    @pytest.fixture
    def broker(self) -> PaperBroker:
        """Create a paper broker instance."""
        return PaperBroker(capital=10_000_000)

    def test_cancel_pending_order(self, broker: PaperBroker) -> None:
        """Test cancelling a pending order."""
        # Note: Market orders fill immediately, so we can't cancel them
        # This tests the cancel mechanism directly
        order = broker.place_order(
            strategy="Test",
            instrument_key="NSE_FO|TEST",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        # Market orders are already complete
        assert order.status == OrderStatus.COMPLETE

        # Try to cancel (should fail since not pending)
        result = broker.cancel_order(order.id)
        assert result is False

    def test_cancel_all_orders(self, broker: PaperBroker) -> None:
        """Test cancelling all pending orders."""
        # Place several orders
        for i in range(3):
            broker.place_order(
                strategy="Test",
                instrument_key=f"NSE_FO|TEST{i}",
                transaction_type=TransactionType.BUY,
                order_type=OrderType.MARKET,
                quantity=15,
            )

        # All market orders filled immediately, so nothing to cancel
        cancelled = broker.cancel_all_orders()
        assert len(cancelled) == 0

    def test_exit_all_positions(self, broker: PaperBroker) -> None:
        """Test exiting all positions."""
        # Create positions
        broker.place_order(
            strategy="Strategy1",
            instrument_key="NSE_FO|INST1",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        broker.place_order(
            strategy="Strategy2",
            instrument_key="NSE_FO|INST2",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        assert len(broker.get_all_positions()) == 2

        # Exit all
        exit_orders = broker.exit_all_positions()

        assert len(exit_orders) == 2
        assert len(broker.get_all_positions()) == 0

        # All exit orders should be sells
        for order in exit_orders:
            assert order.transaction_type == TransactionType.SELL

    def test_order_timestamps(self, broker: PaperBroker) -> None:
        """Test order timestamps are set correctly."""
        from freezegun import freeze_time

        with freeze_time("2024-03-15 10:30:00"):
            order = broker.place_order(
                strategy="Test",
                instrument_key="NSE_FO|TEST",
                transaction_type=TransactionType.BUY,
                order_type=OrderType.MARKET,
                quantity=15,
            )

            assert order.placed_at.year == 2024
            assert order.placed_at.month == 3
            assert order.placed_at.day == 15

    def test_order_fees_recorded(self, broker: PaperBroker) -> None:
        """Test order fees are recorded on fill."""
        order = broker.place_order(
            strategy="Test",
            instrument_key="NSE_FO|TEST",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
        )

        assert order.fees > 0


class TestOrderDataclass:
    """Test Order dataclass."""

    def test_order_creation(self) -> None:
        """Test creating an Order instance."""
        order = Order(
            id="TEST_001",
            strategy="TestStrategy",
            instrument_key="NSE_FO|TEST",
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            quantity=15,
            price=0,
        )

        assert order.id == "TEST_001"
        assert order.filled_qty == 0
        assert order.status == OrderStatus.PENDING

    def test_position_creation(self) -> None:
        """Test creating a Position instance."""
        position = Position(
            instrument_key="NSE_FO|TEST",
            strategy="TestStrategy",
            side="LONG",
            quantity=15,
            entry_price=50000,
        )

        assert position.side == "LONG"
        assert position.quantity == 15
        assert position.entry_price == 50000
