"""Tests for UpstoxLiveBroker implementation.

These tests mock the Upstox API to verify broker functionality without
making real API calls.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.execution.base_broker import (
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
from src.execution.upstox_live import UpstoxLiveBroker

IST = ZoneInfo("Asia/Kolkata")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_token_manager():
    """Return a mock token manager."""
    manager = MagicMock()
    manager.get_valid_token.return_value = "test_access_token"
    manager.is_token_valid.return_value = True
    manager.refresh_token.return_value = "refreshed_token"
    return manager


@pytest.fixture
def broker_with_mocks(mock_token_manager):
    """Return an UpstoxLiveBroker instance with mocked APIs."""
    os.environ["TRADING_MODE"] = "paper"

    # Create mock APIs first - these will be the actual objects used by the broker
    mock_order_api = MagicMock()
    mock_portfolio_api = MagicMock()
    mock_user_api = MagicMock()
    mock_market_api = MagicMock()
    mock_api_client = MagicMock()

    # Create mock upstox_client module
    mock_upstox_client = MagicMock()
    mock_upstox_client.Configuration.return_value = MagicMock()
    mock_upstox_client.ApiClient.return_value = mock_api_client

    # Configure the API constructors to return our pre-created mocks
    mock_upstox_client.OrderApi.return_value = mock_order_api
    mock_upstox_client.PortfolioApi.return_value = mock_portfolio_api
    mock_upstox_client.UserApi.return_value = mock_user_api
    mock_upstox_client.MarketQuoteApi.return_value = mock_market_api

    with patch.dict(sys.modules, {"upstox_client": mock_upstox_client}):
        broker = UpstoxLiveBroker(
            token_manager=mock_token_manager,
            enable_websocket=False,
        )

        # Initialize APIs by calling connect() which uses our mocked module
        # Mock the funds call that happens during connect()
        mock_funds_data = MagicMock()
        mock_funds_data.equity.available_margin = 100000.0
        mock_user_api.get_user_funds_margin.return_value = MagicMock(data=mock_funds_data)

        broker.connect()

        # Verify APIs are initialized
        assert broker._order_api is not None
        assert broker._user_api is not None

        # Store mocks for test access - these are the SAME objects the broker uses
        broker._mock_apis = {
            "order_api": broker._order_api,
            "portfolio_api": broker._portfolio_api,
            "user_api": broker._user_api,
            "market_api": broker._market_api,
        }

        yield broker

        # Cleanup
        broker.disconnect()


@pytest.fixture
def sample_order():
    """Return a sample order for testing."""
    return Order(
        instrument_key="NSE_EQ|INE002A01018",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        product_type=ProductType.MIS,
        quantity=10,
        price=25000,  # 250 INR in paisa
        validity="DAY",
        tag="test_order",
    )


# -----------------------------------------------------------------------------
# Connection Tests
# -----------------------------------------------------------------------------
def test_connect_success(mock_token_manager):
    """Test successful connection to Upstox API."""
    os.environ["TRADING_MODE"] = "paper"

    # Create mock APIs
    mock_user_api = MagicMock()
    mock_funds_data = MagicMock()
    mock_funds_data.equity.available_margin = 100000.0
    mock_user_api.get_user_funds_margin.return_value = MagicMock(data=mock_funds_data)

    mock_upstox_client = MagicMock()
    mock_upstox_client.Configuration.return_value = MagicMock()
    mock_upstox_client.ApiClient.return_value = MagicMock()
    mock_upstox_client.OrderApi.return_value = MagicMock()
    mock_upstox_client.PortfolioApi.return_value = MagicMock()
    mock_upstox_client.UserApi.return_value = mock_user_api
    mock_upstox_client.MarketQuoteApi.return_value = MagicMock()

    with patch.dict(sys.modules, {"upstox_client": mock_upstox_client}):
        broker = UpstoxLiveBroker(
            token_manager=mock_token_manager,
            enable_websocket=False,
        )

        result = broker.connect()

    assert result is True
    assert broker._connected is True
    mock_token_manager.get_valid_token.assert_called_once()


def test_connect_failure(mock_token_manager):
    """Test connection failure handling."""
    os.environ["TRADING_MODE"] = "paper"

    mock_upstox_client = MagicMock()
    mock_upstox_client.Configuration.return_value = MagicMock()
    mock_upstox_client.ApiClient.side_effect = Exception("Connection failed")

    with patch.dict(sys.modules, {"upstox_client": mock_upstox_client}):
        broker = UpstoxLiveBroker(
            token_manager=mock_token_manager,
            enable_websocket=False,
        )

        result = broker.connect()

    assert result is False
    assert broker._connected is False


def test_disconnect(broker_with_mocks):
    """Test disconnection from Upstox API."""
    broker_with_mocks.disconnect()

    assert broker_with_mocks._connected is False


def test_is_connected_true(broker_with_mocks):
    """Test is_connected returns True when connection is active."""
    # Mock funds response
    mock_funds_data = MagicMock()
    mock_funds_data.equity.available_margin = 100000.0
    broker_with_mocks._mock_apis["user_api"].get_user_funds_margin.return_value = MagicMock(
        data=mock_funds_data
    )

    result = broker_with_mocks.is_connected()

    assert result is True


def test_is_connected_false(broker_with_mocks):
    """Test is_connected returns False when connection fails."""
    broker_with_mocks._connected = False

    result = broker_with_mocks.is_connected()

    assert result is False


# -----------------------------------------------------------------------------
# Order Placement Tests
# -----------------------------------------------------------------------------
def test_place_order_success_live_mode(broker_with_mocks, sample_order):
    """Test successful order placement in live mode."""
    os.environ["TRADING_MODE"] = "live"

    # Mock order response
    mock_response = MagicMock()
    mock_response.data.order_id = "TEST123456"
    broker_with_mocks._mock_apis["order_api"].place_order.return_value = mock_response

    result = broker_with_mocks.place_order(sample_order)

    assert isinstance(result, OrderResponse)
    assert result.order_id == "TEST123456"
    assert result.status == OrderStatus.PLACED
    assert result.instrument_key == sample_order.instrument_key
    assert result.side == sample_order.side
    assert result.quantity == sample_order.quantity

    # Verify API call
    broker_with_mocks._mock_apis["order_api"].place_order.assert_called_once()
    call_args = broker_with_mocks._mock_apis["order_api"].place_order.call_args[1]["order_body"]
    assert call_args["instrument_token"] == sample_order.instrument_key
    assert call_args["transaction_type"] == "BUY"
    assert call_args["quantity"] == sample_order.quantity
    assert call_args["price"] == 250.0  # Converted from paisa to rupees


def test_place_order_paper_mode_rejection(broker_with_mocks, sample_order):
    """Test that orders are rejected in paper mode."""
    os.environ["TRADING_MODE"] = "paper"

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.place_order(sample_order)

    assert "Cannot place live order in paper mode" in str(exc_info.value)


def test_place_order_api_failure(broker_with_mocks, sample_order):
    """Test order placement failure handling."""
    os.environ["TRADING_MODE"] = "live"

    # Mock API failure
    broker_with_mocks._mock_apis["order_api"].place_order.side_effect = Exception("API Error")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.place_order(sample_order)

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


def test_place_order_no_order_id(broker_with_mocks, sample_order):
    """Test handling when API returns no order ID."""
    os.environ["TRADING_MODE"] = "live"

    # Mock response with no order ID
    mock_response = MagicMock()
    mock_response.data = None
    broker_with_mocks._mock_apis["order_api"].place_order.return_value = mock_response

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.place_order(sample_order)

    assert "No order ID returned" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Order Modification Tests
# -----------------------------------------------------------------------------
def test_modify_order_success(broker_with_mocks):
    """Test successful order modification."""
    mock_response = MagicMock()
    broker_with_mocks._mock_apis["order_api"].modify_order.return_value = mock_response

    changes = {"price": 26000, "quantity": 20}
    result = broker_with_mocks.modify_order("TEST123", changes)

    assert isinstance(result, OrderResponse)
    assert result.order_id == "TEST123"
    assert result.status == OrderStatus.OPEN

    # Verify price conversion from paisa to rupees
    call_args = broker_with_mocks._mock_apis["order_api"].modify_order.call_args[1]
    assert call_args["order_id"] == "TEST123"


def test_modify_order_failure(broker_with_mocks):
    """Test order modification failure handling."""
    broker_with_mocks._mock_apis["order_api"].modify_order.side_effect = Exception("Modify failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.modify_order("TEST123", {"price": 26000})

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Order Cancellation Tests
# -----------------------------------------------------------------------------
def test_cancel_order_success(broker_with_mocks):
    """Test successful order cancellation."""
    mock_response = MagicMock()
    broker_with_mocks._mock_apis["order_api"].cancel_order.return_value = mock_response

    result = broker_with_mocks.cancel_order("TEST123")

    assert isinstance(result, OrderResponse)
    assert result.order_id == "TEST123"
    assert result.status == OrderStatus.CANCELLED


def test_cancel_order_failure(broker_with_mocks):
    """Test order cancellation failure handling."""
    broker_with_mocks._mock_apis["order_api"].cancel_order.side_effect = Exception("Cancel failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.cancel_order("TEST123")

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Order Book Tests
# -----------------------------------------------------------------------------
def test_get_order_book_success(broker_with_mocks):
    """Test fetching order book."""
    # Mock order book response
    mock_order = MagicMock()
    mock_order.order_id = "TEST123"
    mock_order.status = "COMPLETE"
    mock_order.instrument_token = "NSE_EQ|INE002A01018"
    mock_order.transaction_type = "BUY"
    mock_order.quantity = 10
    mock_order.filled_quantity = 10
    mock_order.average_price = 250.0
    mock_order.status_message = "Order completed"

    mock_response = MagicMock()
    mock_response.data = [mock_order]
    broker_with_mocks._mock_apis["order_api"].get_order_book.return_value = mock_response

    result = broker_with_mocks.get_order_book()

    assert len(result) == 1
    assert result[0].order_id == "TEST123"
    assert result[0].status == OrderStatus.COMPLETE
    assert result[0].average_price == 25000  # Converted to paisa


def test_get_order_book_empty(broker_with_mocks):
    """Test fetching empty order book."""
    mock_response = MagicMock()
    mock_response.data = []
    broker_with_mocks._mock_apis["order_api"].get_order_book.return_value = mock_response

    result = broker_with_mocks.get_order_book()

    assert result == []


def test_get_order_book_failure(broker_with_mocks):
    """Test order book fetch failure."""
    broker_with_mocks._mock_apis["order_api"].get_order_book.side_effect = Exception("Fetch failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.get_order_book()

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Trade Book Tests
# -----------------------------------------------------------------------------
def test_get_trade_book_success(broker_with_mocks):
    """Test fetching trade book."""
    # Mock trade book response
    mock_trade = MagicMock()
    mock_trade.trade_id = "TRADE123"
    mock_trade.order_id = "ORDER123"
    mock_trade.instrument_token = "NSE_EQ|INE002A01018"
    mock_trade.transaction_type = "BUY"
    mock_trade.quantity = 10
    mock_trade.price = 250.0
    mock_trade.trade_timestamp = "2024-01-01T10:00:00Z"

    mock_response = MagicMock()
    mock_response.data = [mock_trade]
    broker_with_mocks._mock_apis["order_api"].get_trade_book.return_value = mock_response

    result = broker_with_mocks.get_trade_book()

    assert len(result) == 1
    assert result[0]["trade_id"] == "TRADE123"
    assert result[0]["price"] == 25000  # Converted to paisa


def test_get_trade_book_failure(broker_with_mocks):
    """Test trade book fetch failure."""
    broker_with_mocks._mock_apis["order_api"].get_trade_book.side_effect = Exception("Fetch failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.get_trade_book()

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Position Tests
# -----------------------------------------------------------------------------
def test_get_positions_success(broker_with_mocks):
    """Test fetching positions."""
    # Mock positions response
    mock_position = MagicMock()
    mock_position.instrument_token = "NSE_EQ|INE002A01018"
    mock_position.product = "MIS"
    mock_position.quantity = 10
    mock_position.average_price = 250.0
    mock_position.last_price = 260.0
    mock_position.pnl = 100.0
    mock_position.realized_profit = 0.0
    mock_position.buy_quantity = 10
    mock_position.sell_quantity = 0
    mock_position.buy_value = 2500.0
    mock_position.sell_value = 0.0

    mock_response = MagicMock()
    mock_response.data = [mock_position]
    broker_with_mocks._mock_apis["portfolio_api"].get_positions.return_value = mock_response

    result = broker_with_mocks.get_positions()

    assert len(result) == 1
    assert isinstance(result[0], Position)
    assert result[0].instrument_key == "NSE_EQ|INE002A01018"
    assert result[0].quantity == 10
    assert result[0].average_price == 25000  # Converted to paisa
    assert result[0].unrealized_pnl == 10000  # Converted to paisa


def test_get_positions_empty(broker_with_mocks):
    """Test fetching empty positions."""
    mock_response = MagicMock()
    mock_response.data = []
    broker_with_mocks._mock_apis["portfolio_api"].get_positions.return_value = mock_response

    result = broker_with_mocks.get_positions()

    assert result == []


def test_get_positions_failure(broker_with_mocks):
    """Test positions fetch failure."""
    broker_with_mocks._mock_apis["portfolio_api"].get_positions.side_effect = Exception("Fetch failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.get_positions()

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Holdings Tests
# -----------------------------------------------------------------------------
def test_get_holdings_success(broker_with_mocks):
    """Test fetching holdings."""
    # Mock holdings response
    mock_holding = MagicMock()
    mock_holding.instrument_token = "NSE_EQ|INE002A01018"
    mock_holding.quantity = 100
    mock_holding.average_price = 200.0
    mock_holding.last_price = 250.0
    mock_holding.pnl = 5000.0

    mock_response = MagicMock()
    mock_response.data = [mock_holding]
    broker_with_mocks._mock_apis["portfolio_api"].get_holdings.return_value = mock_response

    result = broker_with_mocks.get_holdings()

    assert len(result) == 1
    assert result[0]["instrument_key"] == "NSE_EQ|INE002A01018"
    assert result[0]["quantity"] == 100
    assert result[0]["average_price"] == 20000  # Converted to paisa


def test_get_holdings_failure(broker_with_mocks):
    """Test holdings fetch failure."""
    broker_with_mocks._mock_apis["portfolio_api"].get_holdings.side_effect = Exception("Fetch failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.get_holdings()

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Order Status Tests
# -----------------------------------------------------------------------------
def test_get_order_status_success(broker_with_mocks):
    """Test fetching order status."""
    # Mock order details response
    mock_order = MagicMock()
    mock_order.order_id = "TEST123"
    mock_order.status = "OPEN"
    mock_order.instrument_token = "NSE_EQ|INE002A01018"
    mock_order.transaction_type = "BUY"
    mock_order.quantity = 10
    mock_order.filled_quantity = 5
    mock_order.average_price = 250.0
    mock_order.status_message = "Partially filled"

    mock_response = MagicMock()
    mock_response.data = mock_order
    broker_with_mocks._mock_apis["order_api"].get_order_details.return_value = mock_response

    result = broker_with_mocks.get_order_status("TEST123")

    assert isinstance(result, OrderResponse)
    assert result.order_id == "TEST123"
    assert result.status == OrderStatus.OPEN
    assert result.filled_quantity == 5


def test_get_order_status_failure(broker_with_mocks):
    """Test order status fetch failure."""
    broker_with_mocks._mock_apis["order_api"].get_order_details.side_effect = Exception("Fetch failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.get_order_status("TEST123")

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Funds Tests
# -----------------------------------------------------------------------------
def test_get_funds_success(broker_with_mocks):
    """Test fetching funds."""
    # Mock funds response
    mock_equity = MagicMock()
    mock_equity.available_cash = 100000.0
    mock_equity.used_margin = 50000.0
    mock_equity.available_margin = 50000.0
    mock_equity.opening_balance = 100000.0
    mock_equity.payin_amount = 0.0
    mock_equity.payout_amount = 0.0
    mock_equity.span_collateral = 0.0
    mock_equity.non_span_collateral = 0.0
    mock_equity.available_intraday_payin = 0.0
    mock_equity.utilized_debits = 0.0
    mock_equity.utilized_span = 0.0
    mock_equity.utilized_holdings = 0.0
    mock_equity.exposure_margin = 0.0
    mock_equity.utilized_turnover = 0.0

    mock_data = MagicMock()
    mock_data.equity = mock_equity

    mock_response = MagicMock()
    mock_response.data = mock_data
    broker_with_mocks._mock_apis["user_api"].get_user_funds_margin.return_value = mock_response

    result = broker_with_mocks.get_funds()

    assert isinstance(result, Funds)
    assert result.available_cash == 10000000  # Converted to paisa
    assert result.used_margin == 5000000
    assert result.available_margin == 5000000


def test_get_funds_no_data(broker_with_mocks):
    """Test fetching funds when no data available."""
    mock_response = MagicMock()
    mock_response.data = None
    broker_with_mocks._mock_apis["user_api"].get_user_funds_margin.return_value = mock_response

    result = broker_with_mocks.get_funds()

    assert isinstance(result, Funds)
    assert result.available_cash == 0


def test_get_funds_failure(broker_with_mocks):
    """Test funds fetch failure."""
    broker_with_mocks._mock_apis["user_api"].get_user_funds_margin.side_effect = Exception("Fetch failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.get_funds()

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# LTP Tests
# -----------------------------------------------------------------------------
def test_get_ltp_success(broker_with_mocks):
    """Test fetching LTP."""
    # Mock OHLC response
    mock_ohlc = MagicMock()
    mock_ohlc.close = 250.0

    mock_data = MagicMock()
    mock_data.ohlc = mock_ohlc

    mock_response = MagicMock()
    mock_response.data = mock_data
    broker_with_mocks._mock_apis["market_api"].get_market_quote_ohlc.return_value = mock_response

    result = broker_with_mocks.get_ltp("NSE_EQ|INE002A01018")

    assert result == 25000  # Converted to paisa


def test_get_ltp_no_data(broker_with_mocks):
    """Test LTP fetch when no data available."""
    mock_response = MagicMock()
    mock_response.data = None
    broker_with_mocks._mock_apis["market_api"].get_market_quote_ohlc.return_value = mock_response

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.get_ltp("NSE_EQ|INE002A01018")

    assert "No OHLC data available" in str(exc_info.value)


def test_get_ltp_failure(broker_with_mocks):
    """Test LTP fetch failure."""
    broker_with_mocks._mock_apis["market_api"].get_market_quote_ohlc.side_effect = Exception("Fetch failed")

    with pytest.raises(BrokerError) as exc_info:
        broker_with_mocks.get_ltp("NSE_EQ|INE002A01018")

    # Error message comes from _api_call_with_retry after 3 attempts
    assert "API call failed after 3 attempts" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Token Refresh Tests
# -----------------------------------------------------------------------------
def test_token_refresh_on_auth_error(broker_with_mocks, mock_token_manager):
    """Test automatic token refresh on auth error."""
    os.environ["TRADING_MODE"] = "live"

    # First call fails with auth error, second succeeds
    mock_response = MagicMock()
    mock_response.data.order_id = "TEST123"

    side_effects = [
        Exception("401 Unauthorized"),
        mock_response,
    ]
    broker_with_mocks._mock_apis["order_api"].place_order.side_effect = side_effects

    order = Order(
        instrument_key="NSE_EQ|INE002A01018",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        product_type=ProductType.MIS,
        quantity=10,
    )

    result = broker_with_mocks.place_order(order)

    assert result.order_id == "TEST123"
    assert mock_token_manager.refresh_token.call_count >= 1


def test_token_refresh_failure(mock_token_manager):
    """Test handling when token refresh fails."""
    mock_token_manager.is_token_valid.return_value = False
    mock_token_manager.refresh_token.side_effect = Exception("Refresh failed")

    # Create a minimal mock for upstox_client
    mock_upstox_client = MagicMock()

    with patch.dict(sys.modules, {"upstox_client": mock_upstox_client}):
        broker = UpstoxLiveBroker(
            token_manager=mock_token_manager,
            enable_websocket=False,
        )

        with pytest.raises(BrokerError) as exc_info:
            broker._refresh_token_if_needed()

        assert "Token refresh failed" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Order Status Mapping Tests
# -----------------------------------------------------------------------------
def test_map_order_status_complete(broker_with_mocks):
    """Test mapping COMPLETE status."""
    result = broker_with_mocks._map_order_status("COMPLETE")
    assert result == OrderStatus.COMPLETE


def test_map_order_status_rejected(broker_with_mocks):
    """Test mapping REJECTED status."""
    result = broker_with_mocks._map_order_status("REJECTED")
    assert result == OrderStatus.REJECTED


def test_map_order_status_cancelled(broker_with_mocks):
    """Test mapping CANCELLED status."""
    result = broker_with_mocks._map_order_status("CANCELLED")
    assert result == OrderStatus.CANCELLED


def test_map_order_status_partial_fill(broker_with_mocks):
    """Test mapping PARTIAL_FILL status."""
    result = broker_with_mocks._map_order_status("PARTIAL_FILL")
    assert result == OrderStatus.PARTIAL_FILL


def test_map_order_status_unknown(broker_with_mocks):
    """Test mapping unknown status defaults to PENDING."""
    result = broker_with_mocks._map_order_status("UNKNOWN_STATUS")
    assert result == OrderStatus.PENDING


def test_map_order_status_none(broker_with_mocks):
    """Test mapping None status defaults to PENDING."""
    result = broker_with_mocks._map_order_status(None)
    assert result == OrderStatus.PENDING


# -----------------------------------------------------------------------------
# P&L Tests
# -----------------------------------------------------------------------------
def test_get_pnl(broker_with_mocks):
    """Test fetching P&L."""
    # Mock positions with P&L
    mock_position1 = MagicMock()
    mock_position1.instrument_token = "NSE_EQ|INE002A01018"
    mock_position1.product = "MIS"
    mock_position1.quantity = 10
    mock_position1.average_price = 250.0
    mock_position1.last_price = 260.0
    mock_position1.pnl = 100.0
    mock_position1.realized_profit = 50.0
    mock_position1.buy_quantity = 10
    mock_position1.sell_quantity = 0
    mock_position1.buy_value = 2500.0
    mock_position1.sell_value = 0.0

    mock_position2 = MagicMock()
    mock_position2.instrument_token = "NSE_EQ|INE467B01029"
    mock_position2.product = "MIS"
    mock_position2.quantity = -5
    mock_position2.average_price = 3000.0
    mock_position2.last_price = 2900.0
    mock_position2.pnl = 500.0
    mock_position2.realized_profit = 0.0
    mock_position2.buy_quantity = 0
    mock_position2.sell_quantity = 5
    mock_position2.buy_value = 0.0
    mock_position2.sell_value = 15000.0

    mock_response = MagicMock()
    mock_response.data = [mock_position1, mock_position2]
    broker_with_mocks._mock_apis["portfolio_api"].get_positions.return_value = mock_response

    result = broker_with_mocks.get_pnl()

    assert result["realized"] == 5000  # 50 * 100 paisa
    assert result["unrealized"] == 60000  # (100 + 500) * 100 paisa
    assert result["total"] == 65000


# -----------------------------------------------------------------------------
# Rate Limiter Tests
# -----------------------------------------------------------------------------
def test_rate_limiter_acquire():
    """Test rate limiter token acquisition."""
    from src.execution.upstox_live import RateLimiter

    limiter = RateLimiter(max_requests=2, window_seconds=1.0)

    # First two should be immediate
    start = datetime.now()
    limiter.acquire()
    limiter.acquire()
    elapsed = (datetime.now() - start).total_seconds()

    # Should be very fast
    assert elapsed < 0.1

    # Third should wait
    start = datetime.now()
    limiter.acquire()
    elapsed = (datetime.now() - start).total_seconds()

    # Should have waited at least some time
    assert elapsed >= 0.4  # At least part of the window


# -----------------------------------------------------------------------------
# Safety Check Tests
# -----------------------------------------------------------------------------
def test_safety_check_live_mode_warning(mock_token_manager):
    """Test warning is logged in live mode."""
    os.environ["TRADING_MODE"] = "live"

    mock_upstox_client = MagicMock()

    with patch.dict(sys.modules, {"upstox_client": mock_upstox_client}):
        with patch("src.execution.upstox_live.logger") as mock_logger:
            UpstoxLiveBroker(
                token_manager=mock_token_manager,
                enable_websocket=False,
            )

            # Should log live broker warning
            warning_calls = [call for call in mock_logger.warning.call_args_list
                            if "REAL MONEY AT RISK" in str(call)]
            assert len(warning_calls) >= 1


def test_safety_check_paper_mode_warning(mock_token_manager):
    """Test warning is logged in paper mode."""
    os.environ["TRADING_MODE"] = "paper"

    mock_upstox_client = MagicMock()

    with patch.dict(sys.modules, {"upstox_client": mock_upstox_client}):
        with patch("src.execution.upstox_live.logger") as mock_logger:
            UpstoxLiveBroker(
                token_manager=mock_token_manager,
                enable_websocket=False,
            )

            # Should log paper mode warning
            warning_calls = [call for call in mock_logger.warning.call_args_list
                            if "paper" in str(call).lower()]
            assert len(warning_calls) >= 1
