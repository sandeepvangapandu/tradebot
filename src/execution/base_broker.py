"""Base broker interface for trading bot execution.

This module defines the abstract base class that all broker implementations
must inherit from, ensuring a consistent interface for order placement,
modification, cancellation, and position tracking.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Order side - BUY or SELL."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Type of order."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"  # Stop-loss limit
    SL_M = "SL-M"  # Stop-loss market
    AMO = "AMO"  # After market order


class ProductType(str, Enum):
    """Product type for orders."""
    CNC = "CNC"  # Cash and carry (delivery)
    MIS = "MIS"  # Intraday
    NRML = "NRML"  # Normal (F&O carry forward)
    BO = "BO"  # Bracket order
    CO = "CO"  # Cover order


class OrderStatus(str, Enum):
    """Status of an order."""
    PENDING = "PENDING"
    PLACED = "PLACED"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIAL_FILL = "PARTIAL_FILL"


class Order(BaseModel):
    """Represents a trading order.

    All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
    Timestamps are in IST (Asia/Kolkata).
    """
    instrument_key: str = Field(..., description="Unique instrument identifier (e.g., NSE_EQ|INE002A01018)")
    side: OrderSide = Field(..., description="BUY or SELL")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Type of order")
    product_type: ProductType = Field(default=ProductType.MIS, description="Product type")
    quantity: int = Field(..., gt=0, description="Number of shares/lots")
    price: Optional[int] = Field(default=None, description="Limit price in paisa (for LIMIT orders)")
    trigger_price: Optional[int] = Field(default=None, description="Trigger price in paisa (for SL orders)")
    disclosed_quantity: Optional[int] = Field(default=None, description="Disclosed quantity for iceberg orders")
    validity: str = Field(default="DAY", description="Order validity - DAY, IOC, etc.")
    strategy_id: Optional[str] = Field(default=None, description="ID of the strategy that generated this order")
    tag: Optional[str] = Field(default=None, description="Optional tag for tracking")

    class Config:
        use_enum_values = True


class OrderResponse(BaseModel):
    """Response from broker after placing/modifying an order."""
    order_id: str = Field(..., description="Unique order ID from broker")
    status: OrderStatus = Field(..., description="Current status of the order")
    instrument_key: str = Field(..., description="Instrument key")
    side: OrderSide = Field(..., description="Order side")
    quantity: int = Field(..., description="Total quantity")
    filled_quantity: int = Field(default=0, description="Filled quantity so far")
    average_price: Optional[int] = Field(default=None, description="Average fill price in paisa")
    message: Optional[str] = Field(default=None, description="Status message or rejection reason")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(), description="Response timestamp")

    class Config:
        use_enum_values = True


class Position(BaseModel):
    """Represents a trading position.

    All monetary values are in PAISA (integer).
    """
    instrument_key: str = Field(..., description="Unique instrument identifier")
    product_type: ProductType = Field(..., description="Product type")
    quantity: int = Field(..., description="Net quantity (positive for long, negative for short)")
    average_price: int = Field(..., description="Average entry price in paisa")
    last_price: Optional[int] = Field(default=None, description="Last traded price in paisa")
    unrealized_pnl: int = Field(default=0, description="Unrealized P&L in paisa")
    realized_pnl: int = Field(default=0, description="Realized P&L in paisa")
    buy_quantity: int = Field(default=0, description="Total buy quantity")
    sell_quantity: int = Field(default=0, description="Total sell quantity")
    buy_value: int = Field(default=0, description="Total buy value in paisa")
    sell_value: int = Field(default=0, description="Total sell value in paisa")
    day_buy_quantity: int = Field(default=0, description="Day's buy quantity")
    day_sell_quantity: int = Field(default=0, description="Day's sell quantity")
    day_buy_value: int = Field(default=0, description="Day's buy value in paisa")
    day_sell_value: int = Field(default=0, description="Day's sell value in paisa")
    day_buy_average_price: Optional[int] = Field(default=None, description="Day's average buy price")
    day_sell_average_price: Optional[int] = Field(default=None, description="Day's average sell price")
    strategy_id: Optional[str] = Field(default=None, description="Strategy that opened this position")
    entry_time: Optional[datetime] = Field(default=None, description="Position entry timestamp")

    class Config:
        use_enum_values = True


class Funds(BaseModel):
    """Represents available funds in the account.

    All monetary values are in PAISA (integer).
    """
    available_cash: int = Field(..., description="Available cash balance")
    used_margin: int = Field(default=0, description="Margin used for open positions")
    available_margin: int = Field(..., description="Available margin for trading")
    opening_balance: int = Field(..., description="Opening balance for the day")
    payin_amount: int = Field(default=0, description="Payin amount")
    payout_amount: int = Field(default=0, description="Payout amount")
    span_collateral: int = Field(default=0, description="SPAN collateral")
    non_span_collateral: int = Field(default=0, description="Non-SPAN collateral")
    available_intraday_payin: int = Field(default=0, description="Available intraday payin")
    utilized_debits: int = Field(default=0, description="Utilized debits")
    utilized_span: int = Field(default=0, description="Utilized SPAN")
    utilized_holdings: int = Field(default=0, description="Utilized holdings")
    exposure_margin: int = Field(default=0, description="Exposure margin")
    utilized_turnover: int = Field(default=0, description="Utilized turnover")


class BaseBroker(ABC):
    """Abstract base class for broker implementations.

    All broker implementations (PaperBroker, UpstoxBroker, etc.) must inherit
    from this class and implement all abstract methods. This ensures a consistent
    interface for order management across different brokers.

    All monetary values are handled in PAISA (integer) to avoid floating-point
    precision issues. 1 Rupee = 100 paisa.

    All timestamps should be in IST (Asia/Kolkata).
    """

    @abstractmethod
    def place_order(self, order: Order) -> OrderResponse:
        """Place a new order with the broker.

        Args:
            order: The order to place with all required details.

        Returns:
            OrderResponse containing the order ID and status.

        Raises:
            BrokerError: If order placement fails.
        """
        pass

    @abstractmethod
    def modify_order(self, order_id: str, changes: dict) -> OrderResponse:
        """Modify an existing open order.

        Args:
            order_id: The unique order ID to modify.
            changes: Dictionary of fields to modify (e.g., {'price': 15000, 'quantity': 100}).

        Returns:
            OrderResponse with updated order details.

        Raises:
            BrokerError: If modification fails or order cannot be modified.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an open order.

        Args:
            order_id: The unique order ID to cancel.

        Returns:
            OrderResponse with cancellation status.

        Raises:
            BrokerError: If cancellation fails.
        """
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all current positions.

        Returns:
            List of Position objects representing all open positions.

        Raises:
            BrokerError: If fetching positions fails.
        """
        pass

    @abstractmethod
    def get_order_book(self) -> list[OrderResponse]:
        """Get all orders for the day.

        Returns:
            List of OrderResponse objects representing all orders placed today.

        Raises:
            BrokerError: If fetching order book fails.
        """
        pass

    @abstractmethod
    def get_funds(self) -> int:
        """Get available funds for trading.

        Returns:
            Available capital in PAISA (integer).

        Raises:
            BrokerError: If fetching funds fails.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get the status of a specific order.

        Args:
            order_id: The unique order ID to check.

        Returns:
            OrderResponse with current order status.

        Raises:
            BrokerError: If fetching order status fails.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if broker connection is active.

        Returns:
            True if connected, False otherwise.
        """
        pass


class BrokerError(Exception):
    """Exception raised for broker-related errors."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
