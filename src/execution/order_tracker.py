"""Order Tracker for monitoring order status via WebSocket updates.

This module provides the OrderTracker class that receives order updates
from the Portfolio WebSocket, maintains order status in the database,
handles partial fills, and implements retry logic for transient failures.
"""

import threading
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.execution.base_broker import BaseBroker, BrokerError, OrderResponse, OrderStatus

# Indian timezone
IST = ZoneInfo("Asia/Kolkata")

Base = declarative_base()


def _status_value(status: OrderStatus | str) -> str:
    """Return a stable string value for enum-or-string order statuses."""
    return getattr(status, "value", status)


class OrderUpdateRecord(Base):
    """Database model for order status updates."""

    __tablename__ = "order_updates"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, index=True, nullable=False)
    status = Column(String, nullable=False)
    filled_quantity = Column(Integer, default=0)
    average_price = Column(Integer)
    pending_quantity = Column(Integer, default=0)
    message = Column(String)
    exchange_order_id = Column(String)
    update_timestamp = Column(DateTime, nullable=False)


class OrderUpdate(BaseModel):
    """Order update from WebSocket or API."""

    order_id: str = Field(..., description="Order ID")
    status: OrderStatus = Field(..., description="New order status")
    filled_quantity: int = Field(default=0, description="Total filled quantity")
    average_price: Optional[int] = Field(default=None, description="Average fill price")
    pending_quantity: int = Field(default=0, description="Pending quantity")
    message: Optional[str] = Field(default=None, description="Status message")
    exchange_order_id: Optional[str] = Field(default=None, description="Exchange order ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(IST), description="Update timestamp")

    class Config:
        use_enum_values = True


class OrderTracker:
    """Tracks order status updates and maintains order state.

    The OrderTracker receives order updates from the Portfolio WebSocket,
    updates the database with current status, handles partial fills,
    and implements retry logic for transient failures.

    Attributes:
        broker: The broker for fetching order status
        db_session: Database session for persistence
        order_callbacks: Callbacks for order status changes
    """

    def __init__(
        self,
        broker: BaseBroker,
        db_url: Optional[str] = None,
        on_order_fill: Optional[Callable[[OrderUpdate], None]] = None,
        on_order_reject: Optional[Callable[[OrderUpdate], None]] = None,
        on_order_cancel: Optional[Callable[[OrderUpdate], None]] = None,
    ):
        """Initialize the OrderTracker.

        Args:
            broker: Broker implementation for order status queries
            db_url: Database URL for persistence
            on_order_fill: Callback when order fills
            on_order_reject: Callback when order is rejected
            on_order_cancel: Callback when order is cancelled
        """
        self._broker = broker
        self._on_order_fill = on_order_fill
        self._on_order_reject = on_order_reject
        self._on_order_cancel = on_order_cancel
        self._lock = threading.RLock()

        # Order state tracking
        self._orders: dict[str, OrderUpdate] = {}  # order_id -> latest update
        self._pending_orders: set[str] = set()  # Orders awaiting completion

        # Setup database
        if db_url is None:
            db_url = "sqlite:///trading.db"
        self._engine = create_engine(db_url)
        Base.metadata.create_all(bind=self._engine)
        self._SessionLocal = sessionmaker(bind=self._engine)

        logger.info("OrderTracker initialized")

    def handle_websocket_update(self, update: OrderUpdate) -> None:
        """Process an order update from WebSocket.

        Args:
            update: The order update from WebSocket
        """
        with self._lock:
            order_id = update.order_id

            # Get previous state
            previous = self._orders.get(order_id)

            # Update state
            self._orders[order_id] = update

            # Log to database
            self._log_update_to_db(update)

            # Check for state changes
            if previous:
                old_filled = previous.filled_quantity
                new_filled = update.filled_quantity

                if new_filled > old_filled:
                    # Partial or complete fill
                    fill_qty = new_filled - old_filled
                    logger.info(
                        f"Order fill update: {order_id} | Filled: {fill_qty} | "
                        f"Total: {new_filled}/{update.filled_quantity + update.pending_quantity}"
                    )

                    if update.status == OrderStatus.COMPLETE:
                        self._handle_complete_fill(update)
                    else:
                        self._handle_partial_fill(update)

                elif update.status != previous.status:
                    # Status change without fill
                    self._handle_status_change(update, previous.status)
            else:
                # New order
                self._pending_orders.add(order_id)
                logger.info(
                    f"New order tracked: {order_id} | Status: {_status_value(update.status)}"
                )

                if update.status == OrderStatus.COMPLETE:
                    self._handle_complete_fill(update)
                elif update.status == OrderStatus.REJECTED:
                    self._handle_reject(update)
                elif update.status == OrderStatus.CANCELLED:
                    self._handle_cancel(update)

    def on_order_update(self, update: dict | OrderUpdate) -> None:
        """Adapter used by portfolio-feed callbacks.

        Upstox websocket payloads are not guaranteed to arrive as our internal
        ``OrderUpdate`` model, so normalize common field names here before
        feeding the regular tracker lifecycle.
        """
        if isinstance(update, OrderUpdate):
            self.handle_websocket_update(update)
            return

        if not isinstance(update, dict):
            logger.warning("Ignoring non-dict order update: {}", type(update).__name__)
            return

        payload = update.get("data") if isinstance(update.get("data"), dict) else update

        order_id = (
            payload.get("order_id")
            or payload.get("orderId")
            or payload.get("order_reference_id")
            or payload.get("exchange_order_id")
        )
        if not order_id:
            logger.warning("Ignoring order update without order_id: {}", update)
            return

        raw_status = str(payload.get("status") or payload.get("order_status") or "").upper()
        status = self._map_status(raw_status)
        filled_quantity = int(
            payload.get("filled_quantity")
            or payload.get("filledQuantity")
            or payload.get("filled_qty")
            or 0
        )
        quantity = int(payload.get("quantity") or payload.get("order_quantity") or 0)
        pending_quantity = int(
            payload.get("pending_quantity")
            or payload.get("pendingQuantity")
            or max(quantity - filled_quantity, 0)
            or 0
        )
        avg_price_raw = (
            payload.get("average_price")
            or payload.get("averagePrice")
            or payload.get("avg_fill_price")
        )
        average_price = None
        if avg_price_raw not in (None, ""):
            avg_float = float(avg_price_raw)
            # Upstox sends rupees; internal tests/paths often send paisa.
            average_price = int(avg_float if avg_float > 10_000 else avg_float * 100)

        self.handle_websocket_update(
            OrderUpdate(
                order_id=str(order_id),
                status=status,
                filled_quantity=filled_quantity,
                average_price=average_price,
                pending_quantity=pending_quantity,
                message=payload.get("status_message") or payload.get("message"),
                exchange_order_id=payload.get("exchange_order_id"),
            )
        )

    @staticmethod
    def _map_status(status: str) -> OrderStatus:
        """Map broker status strings into internal order statuses."""
        mapping = {
            "COMPLETE": OrderStatus.COMPLETE,
            "TRADED": OrderStatus.COMPLETE,
            "FILLED": OrderStatus.COMPLETE,
            "OPEN": OrderStatus.OPEN,
            "PENDING": OrderStatus.PENDING,
            "PLACED": OrderStatus.PLACED,
            "PARTIAL": OrderStatus.PARTIAL_FILL,
            "PARTIAL_FILL": OrderStatus.PARTIAL_FILL,
            "CANCELLED": OrderStatus.CANCELLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
        }
        return mapping.get(status, OrderStatus.PENDING)

    def _handle_complete_fill(self, update: OrderUpdate) -> None:
        """Handle complete order fill.

        Args:
            update: The order update
        """
        order_id = update.order_id

        with self._lock:
            if order_id in self._pending_orders:
                self._pending_orders.discard(order_id)

        logger.info(
            f"Order complete: {order_id} | Filled: {update.filled_quantity} | "
            f"Avg Price: {update.average_price / 100 if update.average_price else 0:.2f}"
        )

        if self._on_order_fill:
            try:
                self._on_order_fill(update)
            except Exception as e:
                logger.error(f"Error in order fill callback: {e}")

    def _handle_partial_fill(self, update: OrderUpdate) -> None:
        """Handle partial order fill.

        Args:
            update: The order update
        """
        logger.info(
            f"Order partial fill: {update.order_id} | "
            f"Filled: {update.filled_quantity} | Pending: {update.pending_quantity}"
        )

        if self._on_order_fill:
            try:
                self._on_order_fill(update)
            except Exception as e:
                logger.error(f"Error in partial fill callback: {e}")

    def _handle_status_change(self, update: OrderUpdate, previous_status: OrderStatus) -> None:
        """Handle order status change.

        Args:
            update: The order update
            previous_status: Previous order status
        """
        order_id = update.order_id
        new_status = update.status

        logger.info(
            f"Order status change: {order_id} | "
            f"{_status_value(previous_status)} -> {_status_value(new_status)}"
        )

        if new_status == OrderStatus.REJECTED:
            self._handle_reject(update)
        elif new_status == OrderStatus.CANCELLED:
            self._handle_cancel(update)
        elif new_status == OrderStatus.COMPLETE:
            self._handle_complete_fill(update)

    def _handle_reject(self, update: OrderUpdate) -> None:
        """Handle order rejection.

        Args:
            update: The order update
        """
        order_id = update.order_id

        with self._lock:
            self._pending_orders.discard(order_id)

        logger.warning(f"Order rejected: {order_id} | Reason: {update.message}")

        if self._on_order_reject:
            try:
                self._on_order_reject(update)
            except Exception as e:
                logger.error(f"Error in order reject callback: {e}")

    def _handle_cancel(self, update: OrderUpdate) -> None:
        """Handle order cancellation.

        Args:
            update: The order update
        """
        order_id = update.order_id

        with self._lock:
            self._pending_orders.discard(order_id)

        logger.info(f"Order cancelled: {order_id}")

        if self._on_order_cancel:
            try:
                self._on_order_cancel(update)
            except Exception as e:
                logger.error(f"Error in order cancel callback: {e}")

    def _log_update_to_db(self, update: OrderUpdate) -> None:
        """Log order update to database.

        Args:
            update: The order update
        """
        try:
            with self._SessionLocal() as session:
                record = OrderUpdateRecord(
                    order_id=update.order_id,
                    status=_status_value(update.status),
                    filled_quantity=update.filled_quantity,
                    average_price=update.average_price,
                    pending_quantity=update.pending_quantity,
                    message=update.message,
                    exchange_order_id=update.exchange_order_id,
                    update_timestamp=update.timestamp,
                )
                session.add(record)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to log order update to database: {e}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def sync_order_status(self, order_id: str) -> Optional[OrderUpdate]:
        """Sync order status from broker API.

        Args:
            order_id: The order ID to sync

        Returns:
            OrderUpdate if successful, None otherwise
        """
        try:
            response = self._broker.get_order_status(order_id)

            update = OrderUpdate(
                order_id=response.order_id,
                status=response.status,
                filled_quantity=response.filled_quantity,
                average_price=response.average_price,
                message=response.message,
                timestamp=response.timestamp,
            )

            self.handle_websocket_update(update)
            return update

        except BrokerError as e:
            logger.error(f"Failed to sync order status for {order_id}: {e}")
            raise

    def sync_all_pending_orders(self) -> list[OrderUpdate]:
        """Sync status for all pending orders.

        Returns:
            List of updated orders
        """
        with self._lock:
            pending = list(self._pending_orders)

        updated = []
        for order_id in pending:
            try:
                update = self.sync_order_status(order_id)
                if update:
                    updated.append(update)
            except Exception as e:
                logger.error(f"Error syncing order {order_id}: {e}")

        return updated

    def get_order_status(self, order_id: str) -> Optional[OrderUpdate]:
        """Get current status of an order.

        Args:
            order_id: The order ID

        Returns:
            OrderUpdate if found, None otherwise
        """
        with self._lock:
            return self._orders.get(order_id)

    def get_pending_orders(self) -> list[OrderUpdate]:
        """Get all pending orders.

        Returns:
            List of pending order updates
        """
        with self._lock:
            return [
                self._orders[oid]
                for oid in self._pending_orders
                if oid in self._orders
            ]

    def get_completed_orders(self) -> list[OrderUpdate]:
        """Get all completed orders.

        Returns:
            List of completed order updates
        """
        with self._lock:
            return [
                update
                for update in self._orders.values()
                if update.status == OrderStatus.COMPLETE
            ]

    def get_rejected_orders(self) -> list[OrderUpdate]:
        """Get all rejected orders.

        Returns:
            List of rejected order updates
        """
        with self._lock:
            return [
                update
                for update in self._orders.values()
                if update.status == OrderStatus.REJECTED
            ]

    def is_order_pending(self, order_id: str) -> bool:
        """Check if an order is still pending.

        Args:
            order_id: The order ID

        Returns:
            True if pending, False otherwise
        """
        with self._lock:
            return order_id in self._pending_orders

    def get_fill_summary(self, order_id: str) -> dict:
        """Get fill summary for an order.

        Args:
            order_id: The order ID

        Returns:
            Dictionary with fill details
        """
        with self._lock:
            update = self._orders.get(order_id)

        if not update:
            return {"found": False}

        return {
            "found": True,
            "order_id": order_id,
            "status": _status_value(update.status),
            "filled_quantity": update.filled_quantity,
            "pending_quantity": update.pending_quantity,
            "average_price": update.average_price,
            "message": update.message,
        }

    def clear_completed_orders(self, max_age_hours: int = 24) -> int:
        """Clear old completed orders from memory.

        Args:
            max_age_hours: Maximum age in hours to keep

        Returns:
            Number of orders cleared
        """
        cutoff = datetime.now(IST).timestamp() - (max_age_hours * 3600)
        cleared = 0

        with self._lock:
            to_remove = [
                oid
                for oid, update in self._orders.items()
                if update.status in (OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.REJECTED)
                and update.timestamp.timestamp() < cutoff
            ]

            for oid in to_remove:
                del self._orders[oid]
                self._pending_orders.discard(oid)
                cleared += 1

        if cleared > 0:
            logger.info(f"Cleared {cleared} old orders from tracker")

        return cleared
