"""Live Upstox broker implementation for real order placement.

Uses the upstox-python-sdk to place actual orders via Upstox API.
All monetary values are in PAISA (integer) internally, converted to rupees for API calls.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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

IST = ZoneInfo("Asia/Kolkata")

# Mapping from internal types to Upstox API types
ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "SL",
    OrderType.SL_M: "SL-M",
}

PRODUCT_TYPE_MAP = {
    ProductType.CNC: "CNC",
    ProductType.MIS: "MIS",
    ProductType.NRML: "NRML",
    ProductType.BO: "BO",
    ProductType.CO: "CO",
}

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10  # requests per second
RATE_LIMIT_WINDOW = 1.0  # seconds


class RateLimiter:
    """Simple token bucket rate limiter for API calls."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.tokens = max_requests
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.max_requests,
                self.tokens + elapsed * (self.max_requests / self.window_seconds),
            )
            self.last_update = now

            if self.tokens < 1:
                sleep_time = (1 - self.tokens) * (self.window_seconds / self.max_requests)
                time.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class UpstoxLiveBroker(BaseBroker):
    """Live broker implementation using Upstox API.

    Places real orders with real money. Only use when TRADING_MODE=live.

    Args:
        token_manager: TokenManager instance for access token handling.
        enable_websocket: Whether to enable WebSocket for order updates.
        on_order_update: Optional callback for order updates from WebSocket.
    """

    def __init__(
        self,
        token_manager: Any,
        enable_websocket: bool = True,
        on_order_update: Callable[[OrderResponse], None] | None = None,
    ) -> None:
        self.token_manager = token_manager
        self._access_token: str | None = None
        self._api_client = None
        self._order_api = None
        self._portfolio_api = None
        self._user_api = None
        self._market_api = None
        self._websocket = None
        self._enable_websocket = enable_websocket
        self._on_order_update = on_order_update

        # Rate limiter
        self._rate_limiter = RateLimiter(
            max_requests=RATE_LIMIT_REQUESTS,
            window_seconds=RATE_LIMIT_WINDOW,
        )

        # Connection state
        self._connected = False
        self._websocket_thread: threading.Thread | None = None
        self._stop_websocket = threading.Event()

        # Safety check
        trading_mode = os.getenv("TRADING_MODE", "paper").lower()
        if trading_mode != "live":
            logger.warning(
                f"UpstoxLiveBroker initialized in {trading_mode} mode. "
                "Set TRADING_MODE=live for real trading."
            )
        else:
            logger.warning(
                "UPSTOX LIVE BROKER INITIALIZED - REAL MONEY AT RISK"
            )

    def connect(self) -> bool:
        """Establish connection to Upstox API.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._initialize_apis()
            self._connected = True

            # Test connection by fetching funds
            self.get_funds()
            logger.info("UpstoxLiveBroker connected successfully")

            # Start WebSocket if enabled
            if self._enable_websocket:
                self._start_websocket()

            return True
        except Exception as e:
            logger.error(f"Failed to connect to Upstox: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Upstox API and cleanup resources."""
        self._connected = False
        self._stop_websocket.set()

        # Stop the WebSocket thread first
        if self._websocket_thread and self._websocket_thread.is_alive():
            self._websocket_thread.join(timeout=5)

        # Close the WebSocket connection
        if self._websocket:
            try:
                self._websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self._websocket = None

        # Explicitly close the ApiClient urllib3 connection pool.
        # This prevents [Errno 9] Bad file descriptor errors during GC
        # which occur when ApiClient.__del__ tries to close already-reclaimed
        # OS file descriptors at interpreter shutdown.
        if self._api_client is not None:
            try:
                self._api_client.close()
            except (OSError, IOError):
                # Socket already closed by the OS at shutdown — safe to ignore
                pass
            except Exception as e:
                logger.debug(f"Error during ApiClient cleanup: {e}")
            finally:
                self._api_client = None
                self._order_api = None
                self._portfolio_api = None
                self._user_api = None
                self._market_api = None

        logger.info("UpstoxLiveBroker disconnected")

    def _initialize_apis(self) -> None:
        """Initialize Upstox API clients."""
        try:
            import upstox_client

            self._access_token = self.token_manager.get_valid_token()

            # Create API configuration
            config = upstox_client.Configuration()
            config.access_token = self._access_token

            # Initialize API clients
            self._api_client = upstox_client.ApiClient(config)
            self._order_api = upstox_client.OrderApi(self._api_client)
            self._portfolio_api = upstox_client.PortfolioApi(self._api_client)
            self._user_api = upstox_client.UserApi(self._api_client)
            self._market_api = upstox_client.MarketQuoteApi(self._api_client)

            logger.info("Upstox API clients initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Upstox APIs: {e}")
            raise BrokerError(f"API initialization failed: {e}")

    def _refresh_token_if_needed(self) -> None:
        """Refresh access token if expired."""
        try:
            if not self.token_manager.is_token_valid():
                logger.info("Access token expired, refreshing...")
                self._access_token = self.token_manager.refresh_token()
                self._initialize_apis()
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise BrokerError(f"Token refresh failed: {e}")

    def _handle_auth_error(self, error_msg: str) -> bool:
        """Check if error is auth-related and refresh token if needed.

        Args:
            error_msg: The error message to check.

        Returns:
            True if token was refreshed, False otherwise.
        """
        auth_error_keywords = [
            "unauthorized",
            "401",
            "invalid token",
            "token expired",
            "authentication failed",
        ]
        error_lower = error_msg.lower()

        if any(keyword in error_lower for keyword in auth_error_keywords):
            logger.warning("Auth error detected, refreshing token...")
            try:
                self._access_token = self.token_manager.refresh_token()
                self._initialize_apis()
                return True
            except Exception as e:
                logger.error(f"Token refresh after auth error failed: {e}")
                return False
        return False

    def _api_call_with_retry(self, api_func, *args, **kwargs):
        """Make API call with rate limiting and retry logic.

        Args:
            api_func: The API function to call.
            *args: Positional arguments for the API function.
            **kwargs: Keyword arguments for the API function.

        Returns:
            API response.

        Raises:
            BrokerError: If API call fails after retries.
        """
        self._rate_limiter.acquire()
        self._refresh_token_if_needed()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                return api_func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {error_msg}")

                # Check for auth error and refresh token
                if self._handle_auth_error(error_msg) and attempt < max_retries - 1:
                    continue

                # Check for rate limit error
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Check for timeout
                if "timeout" in error_msg.lower():
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue

                if attempt == max_retries - 1:
                    raise BrokerError(f"API call failed after {max_retries} attempts: {error_msg}")

        raise BrokerError("Unexpected end of retry loop")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def place_order(self, order: Order) -> OrderResponse:
        """Place a real order with Upstox.

        Args:
            order: The order to place.

        Returns:
            OrderResponse with broker order ID.

        Raises:
            BrokerError: If order placement fails.
        """
        # Safety check
        trading_mode = os.getenv("TRADING_MODE", "paper").lower()
        if trading_mode != "live":
            raise BrokerError(
                f"Cannot place live order in {trading_mode} mode. "
                "Set TRADING_MODE=live"
            )

        # Convert paisa to rupees for API
        price_rupees = order.price / 100.0 if order.price else None
        trigger_price_rupees = order.trigger_price / 100.0 if order.trigger_price else None

        # Map order type
        order_type = ORDER_TYPE_MAP.get(order.order_type, "MARKET")
        product_type = PRODUCT_TYPE_MAP.get(order.product_type, "MIS")

        # Handle both enum and string values (Pydantic use_enum_values config)
        side_str = order.side.value if hasattr(order.side, 'value') else order.side
        logger.info(
            f"Placing {side_str} {order_type} order for {order.instrument_key} "
            f"Qty: {order.quantity} Price: {price_rupees}"
        )

        try:
            # Handle both enum and string values (Pydantic use_enum_values config)
            side_str = order.side.value if hasattr(order.side, 'value') else order.side
            response = self._api_call_with_retry(
                self._order_api.place_order,
                order_body={
                    "instrument_token": order.instrument_key,
                    "transaction_type": side_str,
                    "order_type": order_type,
                    "product": product_type,
                    "quantity": order.quantity,
                    "price": price_rupees,
                    "trigger_price": trigger_price_rupees,
                    "validity": order.validity,
                    "tag": order.tag or "bot",
                }
            )

            order_id = response.data.order_id if response.data else None

            if not order_id:
                raise BrokerError("Order placement failed: No order ID returned")

            logger.info(f"Order placed successfully: {order_id}")

            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.PLACED,
                instrument_key=order.instrument_key,
                side=order.side,
                quantity=order.quantity,
                message=f"Order placed: {order_id}",
                timestamp=datetime.now(IST),
            )

        except BrokerError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Order placement failed: {error_msg}")
            raise BrokerError(f"Order placement failed: {error_msg}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def modify_order(self, order_id: str, changes: dict) -> OrderResponse:
        """Modify an existing order.

        Args:
            order_id: The order ID to modify.
            changes: Dictionary of changes (price, quantity, trigger_price).

        Returns:
            OrderResponse with updated status.

        Raises:
            BrokerError: If modification fails.
        """
        # Convert paisa to rupees
        if "price" in changes:
            changes["price"] = changes["price"] / 100.0
        if "trigger_price" in changes:
            changes["trigger_price"] = changes["trigger_price"] / 100.0

        logger.info(f"Modifying order {order_id}: {changes}")

        try:
            response = self._api_call_with_retry(
                self._order_api.modify_order,
                order_id=order_id,
                order_body=changes,
            )

            logger.info(f"Order {order_id} modified successfully")

            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.OPEN,
                instrument_key="",  # Will be fetched from order book
                side=OrderSide.BUY,  # Will be fetched from order book
                quantity=changes.get("quantity", 0),
                message="Order modified",
                timestamp=datetime.now(IST),
            )

        except BrokerError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Order modification failed: {error_msg}")
            raise BrokerError(f"Order modification failed: {error_msg}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an open order.

        Args:
            order_id: The order ID to cancel.

        Returns:
            OrderResponse with cancellation status.

        Raises:
            BrokerError: If cancellation fails.
        """
        logger.info(f"Cancelling order {order_id}")

        try:
            self._api_call_with_retry(
                self._order_api.cancel_order,
                order_id=order_id
            )

            logger.info(f"Order {order_id} cancelled successfully")

            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.CANCELLED,
                instrument_key="",
                side=OrderSide.BUY,
                quantity=0,
                message="Order cancelled",
                timestamp=datetime.now(IST),
            )

        except BrokerError:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Order cancellation failed: {error_msg}")
            raise BrokerError(f"Order cancellation failed: {error_msg}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_order_book(self) -> list[OrderResponse]:
        """Get all orders for the day.

        Returns:
            List of OrderResponse objects.

        Raises:
            BrokerError: If fetching order book fails.
        """
        try:
            response = self._api_call_with_retry(self._order_api.get_order_book)
            orders = []

            for order_data in response.data or []:
                orders.append(
                    OrderResponse(
                        order_id=order_data.order_id,
                        status=self._map_order_status(order_data.status),
                        instrument_key=order_data.instrument_token,
                        side=OrderSide(order_data.transaction_type)
                        if hasattr(order_data, "transaction_type")
                        else OrderSide.BUY,
                        quantity=order_data.quantity,
                        filled_quantity=getattr(order_data, "filled_quantity", 0),
                        average_price=int(order_data.average_price * 100)
                        if getattr(order_data, "average_price", None)
                        else None,
                        message=getattr(order_data, "status_message", "") or "",
                        timestamp=datetime.now(IST),
                    )
                )

            return orders

        except BrokerError:
            raise
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            raise BrokerError(f"Failed to get order book: {e}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_trade_book(self) -> list[dict]:
        """Get all trades for the day.

        Returns:
            List of trade dictionaries.

        Raises:
            BrokerError: If fetching trade book fails.
        """
        try:
            response = self._api_call_with_retry(self._order_api.get_trade_book)
            trades = []

            for trade_data in response.data or []:
                trades.append({
                    "trade_id": getattr(trade_data, "trade_id", ""),
                    "order_id": getattr(trade_data, "order_id", ""),
                    "instrument_key": getattr(trade_data, "instrument_token", ""),
                    "transaction_type": getattr(trade_data, "transaction_type", ""),
                    "quantity": getattr(trade_data, "quantity", 0),
                    "price": int(getattr(trade_data, "price", 0) * 100)
                    if getattr(trade_data, "price", None)
                    else 0,
                    "timestamp": getattr(trade_data, "trade_timestamp", ""),
                })

            return trades

        except BrokerError:
            raise
        except Exception as e:
            logger.error(f"Failed to get trade book: {e}")
            raise BrokerError(f"Failed to get trade book: {e}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_positions(self) -> list[Position]:
        """Get current positions from Upstox.

        Returns:
            List of Position objects.

        Raises:
            BrokerError: If fetching positions fails.
        """
        try:
            response = self._api_call_with_retry(self._portfolio_api.get_positions)
            positions = []

            for pos_data in response.data or []:
                positions.append(
                    Position(
                        instrument_key=pos_data.instrument_token,
                        product_type=ProductType(pos_data.product)
                        if hasattr(pos_data, "product")
                        else ProductType.MIS,
                        quantity=pos_data.quantity,
                        average_price=int(pos_data.average_price * 100)
                        if getattr(pos_data, "average_price", None)
                        else 0,
                        last_price=int(pos_data.last_price * 100)
                        if getattr(pos_data, "last_price", None)
                        else None,
                        unrealized_pnl=int(pos_data.pnl * 100)
                        if getattr(pos_data, "pnl", None)
                        else 0,
                        realized_pnl=int(pos_data.realized_profit * 100)
                        if getattr(pos_data, "realized_profit", None)
                        else 0,
                        buy_quantity=getattr(pos_data, "buy_quantity", 0) or 0,
                        sell_quantity=getattr(pos_data, "sell_quantity", 0) or 0,
                        buy_value=int(pos_data.buy_value * 100)
                        if getattr(pos_data, "buy_value", None)
                        else 0,
                        sell_value=int(pos_data.sell_value * 100)
                        if getattr(pos_data, "sell_value", None)
                        else 0,
                    )
                )

            logger.debug(f"Fetched {len(positions)} positions")
            return positions

        except BrokerError:
            raise
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_holdings(self) -> list[dict]:
        """Get holdings (for CNC orders).

        Returns:
            List of holding dictionaries.

        Raises:
            BrokerError: If fetching holdings fails.
        """
        try:
            response = self._api_call_with_retry(self._portfolio_api.get_holdings)
            holdings = []

            for holding in response.data or []:
                holdings.append({
                    "instrument_key": holding.instrument_token,
                    "quantity": holding.quantity,
                    "average_price": int(holding.average_price * 100)
                    if getattr(holding, "average_price", None)
                    else 0,
                    "last_price": int(holding.last_price * 100)
                    if getattr(holding, "last_price", None)
                    else None,
                    "pnl": int(holding.pnl * 100) if getattr(holding, "pnl", None) else 0,
                })

            return holdings

        except BrokerError:
            raise
        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            raise BrokerError(f"Failed to get holdings: {e}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Get status of a specific order.

        Args:
            order_id: Order ID to check.

        Returns:
            OrderResponse with current status.

        Raises:
            BrokerError: If fetching order status fails.
        """
        try:
            response = self._api_call_with_retry(
                self._order_api.get_order_details,
                order_id=order_id
            )
            order_data = response.data

            return OrderResponse(
                order_id=order_data.order_id,
                status=self._map_order_status(order_data.status),
                instrument_key=order_data.instrument_token,
                side=OrderSide(order_data.transaction_type)
                if hasattr(order_data, "transaction_type")
                else OrderSide.BUY,
                quantity=order_data.quantity,
                filled_quantity=getattr(order_data, "filled_quantity", 0),
                average_price=int(order_data.average_price * 100)
                if getattr(order_data, "average_price", None)
                else None,
                message=getattr(order_data, "status_message", "") or "",
                timestamp=datetime.now(IST),
            )

        except BrokerError:
            raise
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            raise BrokerError(f"Failed to get order status: {e}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_funds(self) -> Funds:
        """Get available funds for trading.

        Returns:
            Funds object with detailed fund information.

        Raises:
            BrokerError: If fetching funds fails.
        """
        try:
            response = self._api_call_with_retry(self._user_api.get_user_funds_margin)

            # Get equity segment data (default)
            segment_data = None
            if response.data:
                if hasattr(response.data, "equity"):
                    segment_data = response.data.equity
                else:
                    # Fallback to first available segment
                    for segment in ["equity", "commodity", "futures"]:
                        if hasattr(response.data, segment):
                            seg_data = getattr(response.data, segment)
                            if seg_data:
                                segment_data = seg_data
                                break

            if not segment_data:
                logger.warning("No fund data available, returning zeros")
                return Funds(
                    available_cash=0,
                    available_margin=0,
                    opening_balance=0,
                )

            def to_paisa(value):
                return int((value or 0) * 100)

            return Funds(
                available_cash=to_paisa(getattr(segment_data, "available_cash", 0)),
                used_margin=to_paisa(getattr(segment_data, "used_margin", 0)),
                available_margin=to_paisa(getattr(segment_data, "available_margin", 0)),
                opening_balance=to_paisa(getattr(segment_data, "opening_balance", 0)),
                payin_amount=to_paisa(getattr(segment_data, "payin_amount", 0)),
                payout_amount=to_paisa(getattr(segment_data, "payout_amount", 0)),
                span_collateral=to_paisa(getattr(segment_data, "span_collateral", 0)),
                non_span_collateral=to_paisa(getattr(segment_data, "non_span_collateral", 0)),
                available_intraday_payin=to_paisa(
                    getattr(segment_data, "available_intraday_payin", 0)
                ),
                utilized_debits=to_paisa(getattr(segment_data, "utilized_debits", 0)),
                utilized_span=to_paisa(getattr(segment_data, "utilized_span", 0)),
                utilized_holdings=to_paisa(getattr(segment_data, "utilized_holdings", 0)),
                exposure_margin=to_paisa(getattr(segment_data, "exposure_margin", 0)),
                utilized_turnover=to_paisa(getattr(segment_data, "utilized_turnover", 0)),
            )

        except BrokerError:
            raise
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            raise BrokerError(f"Failed to get funds: {e}")

    @retry(
        retry=retry_if_exception_type(BrokerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_ltp(self, instrument_key: str) -> int:
        """Get last traded price for an instrument.

        Args:
            instrument_key: The instrument identifier.

        Returns:
            Last traded price in paisa.

        Raises:
            BrokerError: If fetching LTP fails.
        """
        try:
            response = self._api_call_with_retry(
                self._market_api.get_market_quote_ohlc,
                instrument_key=instrument_key,
                interval="1d",
            )

            if response.data and hasattr(response.data, "ohlc"):
                close_price = response.data.ohlc.close
                return int(close_price * 100)

            raise BrokerError(f"No OHLC data available for {instrument_key}")

        except BrokerError:
            raise
        except Exception as e:
            logger.error(f"Failed to get LTP for {instrument_key}: {e}")
            raise BrokerError(f"Failed to get LTP: {e}")

    def is_connected(self) -> bool:
        """Check if broker connection is active.

        Returns:
            True if connected, False otherwise.
        """
        if not self._connected:
            return False
        try:
            self.get_funds()
            return True
        except Exception:
            return False

    def _start_websocket(self) -> None:
        """Start WebSocket connection for order updates."""
        if not self._enable_websocket:
            return

        self._stop_websocket.clear()
        self._websocket_thread = threading.Thread(
            target=self._websocket_loop,
            daemon=True,
        )
        self._websocket_thread.start()
        logger.info("Order update WebSocket started")

    def _websocket_loop(self) -> None:
        """WebSocket connection loop with reconnection logic."""
        while not self._stop_websocket.is_set():
            try:
                self._connect_websocket()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if not self._stop_websocket.is_set():
                    time.sleep(5)  # Wait before reconnecting

    def _connect_websocket(self) -> None:
        """Establish WebSocket connection for order updates."""
        try:
            import websocket
            import json

            ws_url = f"wss://api.upstox.com/v2/feed/order-updates?access_token={self._access_token}"

            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._handle_order_update(data)
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")

            def on_close(ws, close_status_code, close_msg):
                logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")

            def on_open(ws):
                logger.info("WebSocket connection opened")

            self._websocket = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            self._websocket.run_forever()

        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise

    def _handle_order_update(self, data: dict) -> None:
        """Handle order update from WebSocket.

        Args:
            data: Order update data from WebSocket.
        """
        try:
            order_id = data.get("order_id")
            status = data.get("status")

            if not order_id:
                return

            order_response = OrderResponse(
                order_id=order_id,
                status=self._map_order_status(status),
                instrument_key=data.get("instrument_token", ""),
                side=OrderSide(data.get("transaction_type", "BUY")),
                quantity=data.get("quantity", 0),
                filled_quantity=data.get("filled_quantity", 0),
                average_price=int(data.get("average_price", 0) * 100)
                if data.get("average_price")
                else None,
                message=data.get("status_message", ""),
                timestamp=datetime.now(IST),
            )

            logger.info(f"Order update via WebSocket: {order_id} - {status}")

            if self._on_order_update:
                self._on_order_update(order_response)

        except Exception as e:
            logger.error(f"Error handling order update: {e}")

    def _map_order_status(self, status: str) -> OrderStatus:
        """Map Upstox order status to internal OrderStatus.

        Args:
            status: Upstox status string.

        Returns:
            Internal OrderStatus enum.
        """
        if not status:
            return OrderStatus.PENDING

        status_map = {
            "PENDING": OrderStatus.PENDING,
            "OPEN": OrderStatus.OPEN,
            "COMPLETE": OrderStatus.COMPLETE,
            "REJECTED": OrderStatus.REJECTED,
            "CANCELLED": OrderStatus.CANCELLED,
            "PARTIAL_FILL": OrderStatus.PARTIAL_FILL,
            "CANCEL_PENDING": OrderStatus.PENDING,
            "MODIFY_PENDING": OrderStatus.PENDING,
        }
        return status_map.get(status.upper(), OrderStatus.PENDING)

    def get_pnl(self) -> dict:
        """Get realized and unrealized P&L.

        Returns:
            Dictionary with 'realized', 'unrealized' keys in paisa.
        """
        positions = self.get_positions()
        realized = sum(p.realized_pnl for p in positions)
        unrealized = sum(p.unrealized_pnl for p in positions)

        return {
            "realized": realized,
            "unrealized": unrealized,
            "total": realized + unrealized,
        }
