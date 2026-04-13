"""Execution module for trading bot.

This module provides order execution, position management, and broker integration
for the autonomous trading bot. It supports both paper trading and live trading
through a unified interface.
"""

from src.execution.base_broker import (
    BaseBroker,
    BrokerError,
    Funds,
    Order,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    ProductType,
)
from src.execution.order_manager import OrderManager
from src.strategy.engine import Signal
from src.execution.order_tracker import OrderTracker, OrderUpdate
from src.execution.paper_broker import PaperBroker, PaperBrokerConfig
from src.execution.position_manager import (
    ManagedPosition,
    PositionConfig,
    PositionManager,
)
from src.execution.partial_profit import (
    ExitTier,
    PartialProfitManager,
    PositionDirection,
    TierAction,
    TierExitResult,
    create_partial_profit_tiers,
)
from src.execution.trailing_stop import (
    FixedTrailingStop,
    MomentumTrailingStop,
    TrailingStopPosition,
    create_trailing_stop,
)
from src.execution.broker_factory import create_broker, BrokerFactory

__all__ = [
    # Factory
    "create_broker",
    "BrokerFactory",
    # Base broker
    # Base broker
    "BaseBroker",
    "BrokerError",
    "Order",
    "OrderResponse",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "ProductType",
    "Funds",
    # Paper broker
    "PaperBroker",
    "PaperBrokerConfig",
    # Order manager
    "OrderManager",
    "Signal",
    # Position manager
    "PositionManager",
    "PositionConfig",
    "ManagedPosition",
    # Order tracker
    "OrderTracker",
    "OrderUpdate",
    # Partial profit
    "PartialProfitManager",
    "ExitTier",
    "PositionDirection",
    "TierAction",
    "TierExitResult",
    "create_partial_profit_tiers",
    # Trailing stop
    "MomentumTrailingStop",
    "FixedTrailingStop",
    "TrailingStopPosition",
    "create_trailing_stop",
]
