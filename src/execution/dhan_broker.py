"""Dhan live broker implementation.

Implements BaseBroker using the dhanhq SDK (pip install dhanhq).
Used for live trading through Dhan brokerage.

All monetary values are in PAISA internally; converted to Rupees only
when calling the Dhan API (which expects float Rupee values).

Authentication: Dhan uses a Client ID + Access Token (JWT, 30-day TTL).
Generate tokens at: https://dhanhq.co/ → Developer Portal.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from loguru import logger

from src.execution.base_broker import (
    BaseBroker,
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

# ─── Dhan exchange segment mapping ───────────────────────────────────────────
# Dhan uses numeric codes internally; the REST API uses string segment names.
_EXCHANGE_MAP: dict[str, str] = {
    # Our instrument_key prefix → Dhan exchange_segment string
    "NSE_EQ": "NSE_EQ",
    "BSE_EQ": "BSE_EQ",
    "NSE_FNO": "NSE_FNO",
    "NSE_CURRENCY": "NSE_CURRENCY",
    "BSE_CURRENCY": "BSE_CURRENCY",
    "MCX_COMM": "MCX_COMM",
    # Common shorthands used in our codebase
    "NSE": "NSE_EQ",
    "BSE": "BSE_EQ",
    "NFO": "NSE_FNO",
}

# ─── Order type mapping ───────────────────────────────────────────────────────
_ORDER_TYPE_MAP: dict[str, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "STOP_LOSS",
    OrderType.SL_M: "STOP_LOSS_MARKET",
    OrderType.AMO: "MARKET",  # AMO treated as market
}

# ─── Product type mapping ─────────────────────────────────────────────────────
_PRODUCT_TYPE_MAP: dict[str, str] = {
    ProductType.CNC: "CNC",
    ProductType.MIS: "INTRADAY",
    ProductType.NRML: "MARGIN",
    ProductType.BO: "BO",
    ProductType.CO: "CO",
}


class DhanBroker(BaseBroker):
    """Live broker implementation using Dhan HQ API v2.

    Args:
        client_id: Dhan account client ID (10-digit account number).
        access_token: JWT access token from Dhan Developer Portal.
        sandbox: If True, use Dhan sandbox base URL.

    Example:
        broker = DhanBroker(
            client_id="1111361185",
            access_token="eyJ...",
        )
    """

    def __init__(
        self,
        client_id: str,
        access_token: str,
        sandbox: bool = False,
    ) -> None:
        self._client_id = client_id
        self._access_token = access_token
        self._sandbox = sandbox

        try:
            from dhanhq import dhanhq as DhanHQ
            self._dhan = DhanHQ(client_id=client_id, access_token=access_token)
            logger.info(
                "DhanBroker connected | client_id={} | sandbox={}",
                client_id, sandbox,
            )
        except ImportError as e:
            raise ImportError(
                "dhanhq package not installed. Run: pip install dhanhq"
            ) from e

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _parse_instrument(self, instrument_key: str) -> tuple[str, str]:
        """Split 'NSE_EQ|INE002A01018' → ('NSE_EQ', 'INE002A01018').

        Also handles plain 'NSE_EQ:RELIANCE' or 'RELIANCE' formats by
        using the first segment identifier found.

        Returns:
            (exchange_segment, security_id) tuple.
        """
        # Format: "NSE_EQ|INE002A01018" (preferred)
        if "|" in instrument_key:
            prefix, sec_id = instrument_key.split("|", 1)
            exchange = _EXCHANGE_MAP.get(prefix.upper(), "NSE_EQ")
            return exchange, sec_id

        # Format: "NSE_EQ:RELIANCE" (symbol-based)
        if ":" in instrument_key:
            prefix, symbol = instrument_key.split(":", 1)
            exchange = _EXCHANGE_MAP.get(prefix.upper(), "NSE_EQ")
            return exchange, symbol

        # Bare symbol — default to NSE equity
        return "NSE_EQ", instrument_key

    @staticmethod
    def _paisa_to_rupees(paisa: int) -> float:
        """Convert internal paisa (int) to Rupees (float) for Dhan API."""
        return round(paisa / 100.0, 2)

    @staticmethod
    def _rupees_to_paisa(rupees: float) -> int:
        """Convert Dhan API Rupees (float) to internal paisa (int)."""
        return int(round(rupees * 100))

    def _map_order_status(self, dhan_status: str) -> OrderStatus:
        """Map Dhan order status string to our OrderStatus enum."""
        mapping = {
            "PENDING": OrderStatus.PENDING,
            "TRANSIT": OrderStatus.PLACED,
            "OPEN": OrderStatus.OPEN,
            "TRADED": OrderStatus.COMPLETE,
            "COMPLETE": OrderStatus.COMPLETE,
            "REJECTED": OrderStatus.REJECTED,
            "CANCELLED": OrderStatus.CANCELLED,
            "PART_TRADED": OrderStatus.PARTIAL_FILL,
            "PART_TRADED_NOW": OrderStatus.PARTIAL_FILL,
        }
        return mapping.get(dhan_status.upper(), OrderStatus.PENDING)

    # ─── BaseBroker interface ─────────────────────────────────────────────────

    def place_order(self, order: Order) -> OrderResponse:
        """Place a new order via Dhan API.

        Args:
            order: Order to place (monetary values in paisa).

        Returns:
            OrderResponse with Dhan order_id and status.
        """
        exchange, security_id = self._parse_instrument(order.instrument_key)
        transaction_type = "BUY" if order.side == OrderSide.BUY else "SELL"
        order_type = _ORDER_TYPE_MAP.get(order.order_type, "MARKET")
        product_type = _PRODUCT_TYPE_MAP.get(order.product_type, "INTRADAY")

        price = self._paisa_to_rupees(order.price) if order.price else 0.0
        trigger_price = (
            self._paisa_to_rupees(order.trigger_price) if order.trigger_price else 0.0
        )

        try:
            response = self._dhan.place_order(
                security_id=security_id,
                exchange_segment=exchange,
                transaction_type=transaction_type,
                quantity=order.quantity,
                order_type=order_type,
                product_type=product_type,
                price=price,
                trigger_price=trigger_price,
                validity="DAY",
                disclosed_quantity=order.disclosed_quantity or 0,
                after_market_order=order.order_type == OrderType.AMO,
                tag=order.tag or "",
            )

            logger.info(
                "DhanBroker.place_order | {} {} {} qty={} | response={}",
                transaction_type,
                order.instrument_key,
                order_type,
                order.quantity,
                response,
            )

            # Dhan returns: {"status": "success", "data": {"orderId": "...", "orderStatus": "..."}}
            if response.get("status") == "success":
                data = response.get("data", {})
                order_id = str(data.get("orderId", uuid.uuid4()))
                order_status = self._map_order_status(
                    data.get("orderStatus", "PENDING")
                )
                return OrderResponse(
                    order_id=order_id,
                    status=order_status,
                    instrument_key=order.instrument_key,
                    side=order.side,
                    quantity=order.quantity,
                    filled_quantity=int(data.get("filledQty", 0)),
                    average_price=self._rupees_to_paisa(
                        float(data.get("price", 0))
                    ),
                    message="Order placed successfully",
                    timestamp=datetime.now(tz=IST),
                )
            else:
                error_msg = response.get("remarks", "Unknown error")
                logger.error("Dhan order rejected: {}", error_msg)
                return OrderResponse(
                    order_id=str(uuid.uuid4()),
                    status=OrderStatus.REJECTED,
                    instrument_key=order.instrument_key,
                    side=order.side,
                    quantity=order.quantity,
                    message=str(error_msg),
                    timestamp=datetime.now(tz=IST),
                )

        except Exception as e:
            logger.exception("DhanBroker.place_order failed: {}", e)
            return OrderResponse(
                order_id=str(uuid.uuid4()),
                status=OrderStatus.REJECTED,
                instrument_key=order.instrument_key,
                side=order.side,
                quantity=order.quantity,
                message=str(e),
                timestamp=datetime.now(tz=IST),
            )

    def modify_order(self, order_id: str, changes: dict[str, Any]) -> OrderResponse:
        """Modify an existing open order.

        Args:
            order_id: Dhan order ID.
            changes: Dict with keys: price, quantity, trigger_price, order_type.

        Returns:
            OrderResponse with updated details.
        """
        try:
            modify_kwargs: dict[str, Any] = {"order_id": order_id}
            if "price" in changes:
                modify_kwargs["price"] = self._paisa_to_rupees(changes["price"])
            if "quantity" in changes:
                modify_kwargs["quantity"] = changes["quantity"]
            if "trigger_price" in changes:
                modify_kwargs["trigger_price"] = self._paisa_to_rupees(
                    changes["trigger_price"]
                )
            if "order_type" in changes:
                modify_kwargs["order_type"] = _ORDER_TYPE_MAP.get(
                    changes["order_type"], "MARKET"
                )

            response = self._dhan.modify_order(**modify_kwargs)
            logger.info("DhanBroker.modify_order | order_id={} | response={}", order_id, response)

            if response.get("status") == "success":
                data = response.get("data", {})
                return OrderResponse(
                    order_id=order_id,
                    status=self._map_order_status(data.get("orderStatus", "OPEN")),
                    instrument_key=data.get("tradingSymbol", ""),
                    side=OrderSide.BUY,  # not returned in modify response
                    quantity=int(data.get("quantity", 0)),
                    message="Order modified",
                    timestamp=datetime.now(tz=IST),
                )
            else:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    instrument_key="",
                    side=OrderSide.BUY,
                    quantity=0,
                    message=str(response.get("remarks", "Modify failed")),
                    timestamp=datetime.now(tz=IST),
                )

        except Exception as e:
            logger.exception("DhanBroker.modify_order failed: {}", e)
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                instrument_key="",
                side=OrderSide.BUY,
                quantity=0,
                message=str(e),
                timestamp=datetime.now(tz=IST),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID.

        Args:
            order_id: Dhan order ID.

        Returns:
            True if cancellation succeeded, False otherwise.
        """
        try:
            response = self._dhan.cancel_order(order_id=order_id)
            logger.info("DhanBroker.cancel_order | order_id={} | response={}", order_id, response)
            return response.get("status") == "success"
        except Exception as e:
            logger.exception("DhanBroker.cancel_order failed: {}", e)
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderResponse]:
        """Fetch the current status of a specific order.

        Args:
            order_id: Dhan order ID.

        Returns:
            OrderResponse or None if not found.
        """
        try:
            response = self._dhan.get_order_by_id(order_id=order_id)
            if response.get("status") == "success":
                data = response.get("data", {})
                return OrderResponse(
                    order_id=order_id,
                    status=self._map_order_status(data.get("orderStatus", "PENDING")),
                    instrument_key=data.get("tradingSymbol", ""),
                    side=OrderSide.BUY if data.get("transactionType") == "BUY" else OrderSide.SELL,
                    quantity=int(data.get("quantity", 0)),
                    filled_quantity=int(data.get("filledQty", 0)),
                    average_price=self._rupees_to_paisa(float(data.get("price", 0))),
                    message=data.get("omsErrorDescription", ""),
                    timestamp=datetime.now(tz=IST),
                )
            return None
        except Exception as e:
            logger.exception("DhanBroker.get_order_status failed: {}", e)
            return None

    def get_positions(self) -> list[Position]:
        """Fetch all current open positions from Dhan.

        Returns:
            List of Position objects (monetary values in paisa).
        """
        try:
            response = self._dhan.get_positions()
            positions: list[Position] = []

            if response.get("status") == "success":
                for item in response.get("data", []):
                    net_qty = int(item.get("netQty", 0))
                    if net_qty == 0:
                        continue  # skip flat positions

                    exchange = item.get("exchangeSegment", "NSE_EQ")
                    symbol = item.get("tradingSymbol", "")
                    instrument_key = f"{exchange}|{symbol}"

                    positions.append(
                        Position(
                            instrument_key=instrument_key,
                            product_type=self._map_product_type(item.get("productType", "INTRADAY")),
                            quantity=net_qty,
                            average_price=self._rupees_to_paisa(float(item.get("buyAvg", 0))),
                            last_price=self._rupees_to_paisa(float(item.get("lastTradedPrice", 0))),
                            unrealized_pnl=self._rupees_to_paisa(float(item.get("unrealizedProfit", 0))),
                            realized_pnl=self._rupees_to_paisa(float(item.get("realizedProfit", 0))),
                            buy_quantity=int(item.get("buyQty", 0)),
                            sell_quantity=int(item.get("sellQty", 0)),
                            buy_value=self._rupees_to_paisa(float(item.get("buyValue", 0))),
                            sell_value=self._rupees_to_paisa(float(item.get("sellValue", 0))),
                        )
                    )
            return positions

        except Exception as e:
            logger.exception("DhanBroker.get_positions failed: {}", e)
            return []

    def get_holdings(self) -> list[dict[str, Any]]:
        """Fetch long-term equity holdings from Dhan.

        Returns:
            List of holding dicts with monetary values in paisa.
        """
        try:
            response = self._dhan.get_holdings()
            holdings: list[dict[str, Any]] = []

            if response.get("status") == "success":
                for item in response.get("data", []):
                    holdings.append({
                        "instrument_key": f"NSE_EQ|{item.get('tradingSymbol', '')}",
                        "isin": item.get("isin", ""),
                        "quantity": int(item.get("totalQty", 0)),
                        "average_price": self._rupees_to_paisa(float(item.get("avgCostPrice", 0))),
                        "last_price": self._rupees_to_paisa(float(item.get("lastTradedPrice", 0))),
                        "pnl": self._rupees_to_paisa(float(item.get("totalPnl", 0))),
                        "collateral_quantity": int(item.get("collateralQty", 0)),
                        "t1_quantity": int(item.get("t1Qty", 0)),
                    })
            return holdings

        except Exception as e:
            logger.exception("DhanBroker.get_holdings failed: {}", e)
            return []

    def get_funds(self) -> Optional[Funds]:
        """Fetch available funds / margin from Dhan.

        Returns:
            Funds object with monetary values in paisa, or None on error.
        """
        try:
            response = self._dhan.get_fund_limits()
            if response.get("status") == "success":
                data = response.get("data", {})
                return Funds(
                    available_cash=self._rupees_to_paisa(
                        float(data.get("availabelBalance", 0))
                    ),
                    used_margin=self._rupees_to_paisa(
                        float(data.get("utilizedAmount", 0))
                    ),
                    available_margin=self._rupees_to_paisa(
                        float(data.get("availabelBalance", 0))
                    ),
                    opening_balance=self._rupees_to_paisa(
                        float(data.get("openingBalance", 0))
                    ),
                    span_collateral=self._rupees_to_paisa(
                        float(data.get("spanMargin", 0))
                    ),
                    exposure_margin=self._rupees_to_paisa(
                        float(data.get("exposureMargin", 0))
                    ),
                )
            return None
        except Exception as e:
            logger.exception("DhanBroker.get_funds failed: {}", e)
            return None

    def get_order_book(self) -> list[dict[str, Any]]:
        """Fetch today's order book from Dhan.

        Returns:
            List of order dicts.
        """
        try:
            response = self._dhan.get_order_list()
            if response.get("status") == "success":
                return response.get("data", [])
            return []
        except Exception as e:
            logger.exception("DhanBroker.get_order_book failed: {}", e)
            return []

    def get_trade_book(self) -> list[dict[str, Any]]:
        """Fetch today's trade book (executed trades) from Dhan.

        Returns:
            List of trade dicts.
        """
        try:
            response = self._dhan.get_trade_book()
            if response.get("status") == "success":
                return response.get("data", [])
            return []
        except Exception as e:
            logger.exception("DhanBroker.get_trade_book failed: {}", e)
            return []

    def is_connected(self) -> bool:
        """Check if the Dhan API client is initialised and reachable.

        Performs a lightweight fund-limits ping to verify connectivity.

        Returns:
            True if API responds successfully, False otherwise.
        """
        try:
            response = self._dhan.get_fund_limits()
            return response.get("status") == "success"
        except Exception:
            return False

    # ─── Helper: reverse product type mapping ────────────────────────────────

    @staticmethod
    def _map_product_type(dhan_product: str) -> str:
        """Map Dhan product type string to our ProductType."""
        mapping = {
            "CNC": ProductType.CNC,
            "INTRADAY": ProductType.MIS,
            "MARGIN": ProductType.NRML,
            "BO": ProductType.BO,
            "CO": ProductType.CO,
        }
        return mapping.get(dhan_product.upper(), ProductType.MIS)
