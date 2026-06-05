"""Paper trading broker implementation for backtesting and simulation.

This module provides a PaperBroker class that simulates order execution
with realistic slippage, fees, and market conditions for paper trading.
All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
"""

import threading
import uuid
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from loguru import logger

from config.constants import FEES
from src.execution.base_broker import (
    BaseBroker,
    BrokerError,
    Order,
    ComboOrder,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
)

# Indian timezone
IST = ZoneInfo("Asia/Kolkata")


class PaperBrokerConfig:
    """Configuration for paper trading simulation.

    All fee constants are sourced from config.constants.FEES for consistency.
    """

    # Slippage configuration (as decimal, e.g., 0.0005 = 0.05%)
    DEFAULT_SLIPPAGE_PCT: float = FEES["default_slippage_pct"]

    # Brokerage fees (in Rupees per order)
    BROKERAGE_EQUITY_DELIVERY: float = FEES["brokerage_equity_delivery"]  # Free
    BROKERAGE_EQUITY_INTRADAY: float = FEES["brokerage_equity_intraday"] / 100  # Rs 20
    BROKERAGE_FNO: float = FEES["brokerage_fno"] / 100  # Rs 20

    # STT (Securities Transaction Tax) - as decimal
    STT_EQUITY_DELIVERY_BUY: float = FEES["stt_equity_delivery_buy_pct"]
    STT_EQUITY_DELIVERY_SELL: float = FEES["stt_equity_delivery_sell_pct"]
    STT_EQUITY_INTRADAY_SELL: float = FEES["stt_equity_intraday_sell_pct"]
    STT_FNO_BUY: float = FEES["stt_options_buy_pct"]  # Options and futures both 0 on buy
    STT_FNO_SELL: float = FEES["stt_options_sell_pct"]

    # Exchange transaction charges - as decimal
    EXCHANGE_CHARGE_NSE: float = FEES["exchange_charge_nse_fo_pct"]
    EXCHANGE_CHARGE_BSE: float = FEES["exchange_charge_bse_eq_pct"]

    # GST - 18% on (brokerage + exchange charges)
    GST_RATE: float = FEES["gst_pct"]

    # SEBI charges - Rs 10 per crore = 0.0000001
    SEBI_CHARGE_RATE: float = FEES["sebi_charge_rate_pct"]

    # Stamp duty (on buy side only) - varies by state, using average
    STAMP_DUTY_EQUITY_DELIVERY: float = FEES["stamp_duty_equity_delivery_pct"]
    STAMP_DUTY_EQUITY_INTRADAY: float = FEES["stamp_duty_equity_intraday_pct"]
    STAMP_DUTY_FNO: float = FEES["stamp_duty_fno_pct"]

    # DP charges (for delivery sell) - Rs 13.5 per scrip per day
    DP_CHARGES: int = FEES["dp_charges_paisa"]


class PaperBroker(BaseBroker):
    """Paper trading broker that simulates order execution.

    This broker simulates realistic trading conditions including:
    - Slippage on order fills
    - Realistic brokerage and regulatory fees
    - Position tracking
    - Order book management
    - Fund management

    All monetary values are in PAISA (integer) to avoid floating-point issues.
    """

    def __init__(
        self,
        initial_capital: int = 100000000,  # 10 Lakhs in paisa
        slippage_pct: float = PaperBrokerConfig.DEFAULT_SLIPPAGE_PCT,
    ):
        """Initialize paper broker with starting capital.

        Args:
            initial_capital: Starting capital in PAISA (default 10 Lakhs)
            slippage_pct: Slippage percentage as decimal (default 0.0005 = 0.05%)
        """
        self._lock = threading.Lock()
        self._slippage_pct = slippage_pct

        # Funds tracking (all in paisa)
        self._initial_capital = initial_capital
        self._available_cash = initial_capital
        self._used_margin = 0
        self._realized_pnl = 0

        # Order and position tracking
        self._orders: dict[str, OrderResponse] = {}
        self._positions: dict[str, Position] = {}  # key: instrument_key|product_type
        self._trade_history: list[dict] = []

        # Market data cache for LTP + Level-1 bid/ask
        self._ltp_cache: dict[str, int] = {}
        self._ltp_timestamp: dict[str, Any] = {}
        self._bid_cache: dict[str, int] = {}  # best bid in paisa
        self._ask_cache: dict[str, int] = {}  # best ask in paisa

        logger.info(f"PaperBroker initialized with capital: {self._paisa_to_rupees(initial_capital):.2f} INR")

    @staticmethod
    def _paisa_to_rupees(paisa: int) -> float:
        """Convert paisa to rupees for display."""
        return paisa / 100.0

    @staticmethod
    def _rupees_to_paisa(rupees: float) -> int:
        """Convert rupees to paisa for internal storage."""
        return int(round(rupees * 100))

    def _get_ltp(self, instrument_key: str) -> int:
        """Get last traded price for an instrument.

        Returns 0 if no tick has been received yet OR if the last tick is
        stale (older than 60 seconds). Stale LTP caused the May-14 disaster
        where a ₹95 cached price was used to fill a ₹815 option.

        Returns:
            LTP in paisa, or 0 if no tick received / tick is stale.
        """
        price = self._ltp_cache.get(instrument_key, 0)
        if price <= 0:
            return 0
        ts = self._ltp_timestamp.get(instrument_key)
        if ts is None:
            return 0
        from datetime import datetime
        from zoneinfo import ZoneInfo
        age_secs = (datetime.now(ZoneInfo("Asia/Kolkata")) - ts).total_seconds()
        if age_secs > 60:
            logger.warning(
                "Stale LTP for {} ({}s old) — rejecting fill to prevent catastrophic price",
                instrument_key, int(age_secs),
            )
            return 0
        return price

    def get_ltp(self, instrument_key: str) -> int:
        """Get last traded price for an instrument (public method)."""
        return self._get_ltp(instrument_key)

    def update_ltp(self, instrument_key: str, price: int) -> None:
        """Update LTP + timestamp for an instrument (called by market data feed)."""
        if price <= 0:
            return
        from datetime import datetime
        from zoneinfo import ZoneInfo
        self._ltp_cache[instrument_key] = price
        self._ltp_timestamp[instrument_key] = datetime.now(ZoneInfo("Asia/Kolkata"))

    def update_quote(
        self,
        instrument_key: str,
        price: int,
        bid: int | None = None,
        ask: int | None = None,
    ) -> None:
        """Update LTP + bid/ask from live tick. Use for accurate paper fills."""
        self.update_ltp(instrument_key, price)
        if bid and bid > 0:
            self._bid_cache[instrument_key] = bid
        if ask and ask > 0:
            self._ask_cache[instrument_key] = ask

    def _calculate_fill_price(self, instrument_key: str, side: OrderSide, order_type: OrderType, limit_price: Optional[int] = None) -> int:
        """Calculate simulated fill price with slippage.

        When live bid/ask data is present, use it directly — the spread IS the
        transaction cost. Adding slippage on top of an ask price double-penalizes
        the trade. Slippage is only added when falling back to LTP (no quote data).

        Args:
            instrument_key: The instrument identifier
            side: BUY or SELL
            order_type: Type of order
            limit_price: Limit price for LIMIT orders

        Returns:
            Fill price in paisa
        """
        ltp = self._get_ltp(instrument_key)

        if order_type == OrderType.LIMIT and limit_price is not None:
            return limit_price

        if side == OrderSide.BUY:
            ask = self._ask_cache.get(instrument_key, 0)
            if ask > 0:
                return ask  # spread already captures cost; no additional slippage
            return ltp + int(ltp * self._slippage_pct)
        else:
            bid = self._bid_cache.get(instrument_key, 0)
            if bid > 0:
                return bid  # spread already captures cost; no additional slippage
            return max(0, ltp - int(ltp * self._slippage_pct))

    def _calculate_charges(
        self,
        instrument_key: str,
        side: OrderSide,
        product_type: ProductType,
        quantity: int,
        price: int,
    ) -> dict[str, int]:
        """Calculate all trading charges.

        Args:
            instrument_key: The instrument identifier
            side: BUY or SELL
            product_type: CNC, MIS, NRML, etc.
            quantity: Number of shares/lots
            price: Execution price in paisa

        Returns:
            Dictionary of charges in paisa
        """
        turnover = quantity * price  # in paisa
        turnover_rupees = turnover / 100.0

        # Determine segment and calculate charges
        is_fno = (
            "NSE_FO|" in instrument_key        # Upstox NSE F&O instruments
            or "BSE_FO|" in instrument_key     # Upstox BSE F&O instruments
            or "NFO|" in instrument_key        # legacy prefix fallback
            or "BFO|" in instrument_key
            or "MCX|" in instrument_key
            or "NSE_INDEX|" in instrument_key  # index straddles use F&O fee structure
        )
        is_delivery = product_type == ProductType.CNC

        charges = {}

        # Brokerage
        if is_fno:
            brokerage = PaperBrokerConfig.BROKERAGE_FNO
        elif is_delivery:
            brokerage = PaperBrokerConfig.BROKERAGE_EQUITY_DELIVERY
        else:
            brokerage = PaperBrokerConfig.BROKERAGE_EQUITY_INTRADAY

        charges["brokerage"] = self._rupees_to_paisa(brokerage)

        # STT
        if is_fno:
            stt_rate = PaperBrokerConfig.STT_FNO_SELL if side == OrderSide.SELL else PaperBrokerConfig.STT_FNO_BUY
        elif is_delivery:
            stt_rate = PaperBrokerConfig.STT_EQUITY_DELIVERY_SELL if side == OrderSide.SELL else PaperBrokerConfig.STT_EQUITY_DELIVERY_BUY
        else:
            stt_rate = PaperBrokerConfig.STT_EQUITY_INTRADAY_SELL if side == OrderSide.SELL else 0.0

        charges["stt"] = int(turnover * stt_rate)

        # Exchange charges
        exchange_rate = PaperBrokerConfig.EXCHANGE_CHARGE_NSE
        charges["exchange"] = int(turnover * exchange_rate)

        # GST on brokerage + exchange charges
        gst_base = charges["brokerage"] + charges["exchange"]
        charges["gst"] = int(gst_base * PaperBrokerConfig.GST_RATE)

        # SEBI charges
        charges["sebi"] = int(turnover_rupees * PaperBrokerConfig.SEBI_CHARGE_RATE * 100)  # Convert to paisa

        # Stamp duty (buy side only)
        if side == OrderSide.BUY:
            if is_fno:
                stamp_rate = PaperBrokerConfig.STAMP_DUTY_FNO
            elif is_delivery:
                stamp_rate = PaperBrokerConfig.STAMP_DUTY_EQUITY_DELIVERY
            else:
                stamp_rate = PaperBrokerConfig.STAMP_DUTY_EQUITY_INTRADAY
            charges["stamp_duty"] = int(turnover * stamp_rate / 100)
        else:
            charges["stamp_duty"] = 0

        # DP charges (delivery sell only)
        if is_delivery and side == OrderSide.SELL:
            charges["dp_charges"] = PaperBrokerConfig.DP_CHARGES
        else:
            charges["dp_charges"] = 0

        charges["total"] = sum(charges.values())
        return charges

    def place_order(self, order: Order) -> OrderResponse:
        """Place a simulated order.

        Args:
            order: The order to place

        Returns:
            OrderResponse with simulated fill
        """
        with self._lock:
            order_id = f"PAPER_{uuid.uuid4().hex[:12].upper()}"
            timestamp = datetime.now(IST)

            # Reject MARKET orders that arrive before the first tick for
            # this instrument — fill price would be 0/fake and produce
            # nonsensical entry levels (see _get_ltp docstring).
            if order.order_type == OrderType.MARKET and self._get_ltp(order.instrument_key) <= 0:
                logger.warning(
                    "Rejecting MARKET order — no LTP yet for {} (waiting for first tick)",
                    order.instrument_key,
                )
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    instrument_key=order.instrument_key,
                    side=order.side,
                    quantity=order.quantity,
                    filled_quantity=0,
                    message="No LTP available — first tick not yet received",
                    timestamp=timestamp,
                )

            # Calculate fill price
            fill_price = self._calculate_fill_price(
                order.instrument_key,
                order.side,
                order.order_type,
                order.price,
            )

            # Calculate charges
            charges = self._calculate_charges(
                order.instrument_key,
                order.side,
                order.product_type,
                order.quantity,
                fill_price,
            )

            # Calculate total amount including charges
            total_value = order.quantity * fill_price
            total_charges = charges["total"]

            if order.side == OrderSide.BUY:
                required_funds = total_value + total_charges
                if required_funds > self._available_cash:
                    return OrderResponse(
                        order_id=order_id,
                        status=OrderStatus.REJECTED,
                        instrument_key=order.instrument_key,
                        side=order.side,
                        quantity=order.quantity,
                        filled_quantity=0,
                        message=f"Insufficient funds. Required: {required_funds}, Available: {self._available_cash}",
                        timestamp=timestamp,
                    )

                # Deduct funds
                self._available_cash -= required_funds
                self._used_margin += required_funds
            else:
                # SELL order: deduct charges (proceeds flow through position P&L on close)
                self._available_cash -= total_charges

            # Create position key
            product_val = order.product_type.value if hasattr(order.product_type, 'value') else str(order.product_type)
            position_key = f"{order.instrument_key}|{product_val}"

            # Update or create position
            if position_key in self._positions:
                position = self._positions[position_key]
                position = self._update_position(position, order, fill_price, charges)
            else:
                position = self._create_position(order, fill_price, charges)

            self._positions[position_key] = position

            # Record order
            response = OrderResponse(
                order_id=order_id,
                status=OrderStatus.COMPLETE,
                instrument_key=order.instrument_key,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                average_price=fill_price,
                timestamp=timestamp,
            )
            self._orders[order_id] = response

            # Log trade
            side_val = order.side.value if hasattr(order.side, 'value') else str(order.side)
            trade_record = {
                "order_id": order_id,
                "timestamp": timestamp.isoformat(),
                "instrument_key": order.instrument_key,
                "side": side_val,
                "quantity": order.quantity,
                "price": fill_price,
                "charges": charges,
                "strategy_id": order.strategy_id,
            }
            self._trade_history.append(trade_record)

            logger.info(
                f"Paper order filled: {order_id} | {side_val} {order.quantity} "
                f"{order.instrument_key} @ {self._paisa_to_rupees(fill_price):.2f} | "
                f"Charges: {self._paisa_to_rupees(charges['total']):.2f}"
            )

            return response

    def _create_position(self, order: Order, fill_price: int, charges: dict) -> Position:
        """Create a new position from an order fill."""
        quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
        value = order.quantity * fill_price

        return Position(
            instrument_key=order.instrument_key,
            product_type=order.product_type,
            quantity=quantity,
            average_price=fill_price,
            last_price=fill_price,
            unrealized_pnl=0,
            realized_pnl=0,
            buy_quantity=order.quantity if order.side == OrderSide.BUY else 0,
            sell_quantity=order.quantity if order.side == OrderSide.SELL else 0,
            buy_value=value if order.side == OrderSide.BUY else 0,
            sell_value=value if order.side == OrderSide.SELL else 0,
            day_buy_quantity=order.quantity if order.side == OrderSide.BUY else 0,
            day_sell_quantity=order.quantity if order.side == OrderSide.SELL else 0,
            day_buy_value=value if order.side == OrderSide.BUY else 0,
            day_sell_value=value if order.side == OrderSide.SELL else 0,
            day_buy_average_price=fill_price if order.side == OrderSide.BUY else None,
            day_sell_average_price=fill_price if order.side == OrderSide.SELL else None,
            strategy_id=order.strategy_id,
            entry_time=datetime.now(IST),
        )

    def _update_position(self, position: Position, order: Order, fill_price: int, charges: dict) -> Position:
        """Update existing position with new trade."""
        trade_quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
        trade_value = order.quantity * fill_price

        # Check if this reduces position (opposite side)
        if (position.quantity > 0 and order.side == OrderSide.SELL) or \
           (position.quantity < 0 and order.side == OrderSide.BUY):
            # Partial or full close
            close_qty = min(abs(position.quantity), order.quantity)
            entry_value = close_qty * position.average_price
            exit_value = close_qty * fill_price

            if position.quantity > 0:  # Long position
                pnl = exit_value - entry_value
            else:  # Short position
                pnl = entry_value - exit_value

            net_pnl = pnl - charges["total"]
            position.realized_pnl += net_pnl
            self._realized_pnl += net_pnl

            # Return freed capital to available cash (entry_value = margin deployed)
            # Net proceeds = capital deployed for closed qty + any profit/loss
            freed_margin = entry_value
            self._available_cash += freed_margin + net_pnl
            self._used_margin = max(0, self._used_margin - freed_margin)

        # Update position quantities
        old_quantity = position.quantity
        position.quantity += trade_quantity

        # Update average price
        if position.quantity != 0:
            if old_quantity == 0 or (old_quantity > 0 and order.side == OrderSide.BUY) or \
               (old_quantity < 0 and order.side == OrderSide.SELL):
                # Adding to position
                total_value = abs(old_quantity) * position.average_price + trade_value
                position.average_price = total_value // abs(position.quantity)

        # Update buy/sell tracking
        if order.side == OrderSide.BUY:
            position.buy_quantity += order.quantity
            position.buy_value += trade_value
            position.day_buy_quantity += order.quantity
            position.day_buy_value += trade_value
            if position.day_buy_average_price:
                position.day_buy_average_price = (
                    (position.day_buy_average_price * (position.day_buy_quantity - order.quantity) + trade_value)
                    // position.day_buy_quantity
                )
            else:
                position.day_buy_average_price = fill_price
        else:
            position.sell_quantity += order.quantity
            position.sell_value += trade_value
            position.day_sell_quantity += order.quantity
            position.day_sell_value += trade_value
            if position.day_sell_average_price:
                position.day_sell_average_price = (
                    (position.day_sell_average_price * (position.day_sell_quantity - order.quantity) + trade_value)
                    // position.day_sell_quantity
                )
            else:
                position.day_sell_average_price = fill_price

        position.last_price = fill_price
        return position

    def place_combo_order(self, combo: ComboOrder) -> list[OrderResponse]:
        """Place a multi-leg combination order.
        
        In paper trading, this executes all legs sequentially, ensuring
        they either all succeed or fail together conceptually. Since paper
        trading rarely fails once margins are checked, we just loop through.
        """
        responses = []
        for leg in combo.legs:
            resp = self.place_order(leg)
            responses.append(resp)
        return responses

    def modify_order(self, order_id: str, changes: dict) -> OrderResponse:
        """Modify an existing order (not supported in paper trading for filled orders).

        Args:
            order_id: The order ID to modify
            changes: Dictionary of changes

        Returns:
            OrderResponse

        Raises:
            BrokerError: Always raises as paper orders fill immediately
        """
        raise BrokerError("Cannot modify orders in paper trading - orders fill immediately")

    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order (not supported in paper trading for filled orders).

        Args:
            order_id: The order ID to cancel

        Returns:
            OrderResponse

        Raises:
            BrokerError: Always raises as paper orders fill immediately
        """
        raise BrokerError("Cannot cancel orders in paper trading - orders fill immediately")

    def get_positions(self) -> list[Position]:
        """Get all current positions.

        Returns:
            List of Position objects
        """
        with self._lock:
            return list(self._positions.values())

    def get_order_book(self) -> list[OrderResponse]:
        """Get all orders for the day.

        Returns:
            List of OrderResponse objects
        """
        with self._lock:
            return list(self._orders.values())

    def get_funds(self) -> int:
        """Get available funds.

        Returns:
            Available capital in paisa
        """
        with self._lock:
            return self._available_cash

    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get status of a specific order.

        Args:
            order_id: The order ID

        Returns:
            OrderResponse

        Raises:
            BrokerError: If order not found
        """
        with self._lock:
            if order_id not in self._orders:
                raise BrokerError(f"Order not found: {order_id}")
            return self._orders[order_id]

    def is_connected(self) -> bool:
        """Check if broker is connected.

        Returns:
            Always True for paper broker
        """
        return True

    def get_trade_history(self) -> list[dict]:
        """Get trade history.

        Returns:
            List of trade records
        """
        with self._lock:
            return self._trade_history.copy()

    def sum_fees_for_instrument(self, instrument_key: str, since: Optional[datetime] = None) -> int:
        """Sum total fees (brokerage + STT + GST + ...) for an instrument.

        Used by PositionManager to attribute fees to closed-trade rows.

        Args:
            instrument_key: The instrument identifier.
            since: Optional cutoff — only include fills at-or-after this time.

        Returns:
            Total fees in paisa across all fills matching the filter.
        """
        with self._lock:
            total = 0
            for rec in self._trade_history:
                if rec.get("instrument_key") != instrument_key:
                    continue
                if since is not None:
                    try:
                        ts = datetime.fromisoformat(rec["timestamp"])
                        if ts < since:
                            continue
                    except Exception:
                        pass
                total += int(rec.get("charges", {}).get("total", 0) or 0)
            return total

    def get_portfolio_value(self) -> int:
        """Get total portfolio value including positions.

        Returns:
            Total value in paisa
        """
        with self._lock:
            position_value = 0
            for position in self._positions.values():
                if position.quantity != 0:
                    ltp = self._get_ltp(position.instrument_key)
                    position_value += abs(position.quantity) * ltp
            # available_cash already excludes deployed margin; adding position_value
            # (current market value) gives total portfolio worth without double-counting.
            return self._available_cash + position_value

    def reset(self) -> None:
        """Reset the paper broker to initial state."""
        with self._lock:
            self._available_cash = self._initial_capital
            self._used_margin = 0
            self._realized_pnl = 0
            self._orders.clear()
            self._positions.clear()
            self._trade_history.clear()
            self._ltp_cache.clear()
            logger.info("PaperBroker reset to initial state")

    def get_orderbook(self, instrument_key: str) -> Optional[dict]:
        """Get simulated orderbook for an instrument.

        Generates a simulated orderbook based on the current LTP.
        This is useful for testing entry optimization.

        Args:
            instrument_key: The instrument identifier

        Returns:
            Orderbook dict with bid/ask/vwap/ltp data, or None if LTP unavailable
        """
        ltp = self._get_ltp(instrument_key)
        if ltp <= 0:
            return None

        # Simulate a tight spread (0.05% = 5 bps)
        spread_paisa = max(5, int(ltp * 0.0005))  # Minimum 5 paisa (1 tick)
        bid = ltp - spread_paisa
        ask = ltp + spread_paisa
        vwap = ltp  # Simplified: VWAP = LTP

        # Simulate bid/ask quantities
        bid_qty = 1000
        ask_qty = 1000

        return {
            "bid": bid / 100.0,  # Convert to rupees for entry_optimizer
            "ask": ask / 100.0,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "vwap": vwap / 100.0,
            "ltp": ltp / 100.0,
        }
