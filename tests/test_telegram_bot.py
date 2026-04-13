"""Tests for Telegram bot notifications and commands.

Uses pytest-asyncio for async test support and mocks python-telegram-bot API.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.notifications.telegram_bot import (
    IST,
    TelegramBot,
    TelegramNotifier,
    create_bot_from_settings,
    create_notifier_from_settings,
)
from src.persistence.models import Base, DailyPnL, PositionRecord, TradeRecord


# Fixtures
@pytest.fixture
def mock_env_token():
    """Set up mock environment variables for Telegram."""
    os.environ["TELEGRAM_BOT_TOKEN"] = "test_token_12345"
    os.environ["TELEGRAM_CHAT_ID"] = "123456789"
    yield
    del os.environ["TELEGRAM_BOT_TOKEN"]
    del os.environ["TELEGRAM_CHAT_ID"]


@pytest.fixture
def mock_env_empty():
    """Ensure no Telegram environment variables are set."""
    old_token = os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    old_chat = os.environ.pop("TELEGRAM_CHAT_ID", None)
    yield
    if old_token:
        os.environ["TELEGRAM_BOT_TOKEN"] = old_token
    if old_chat:
        os.environ["TELEGRAM_CHAT_ID"] = old_chat


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def sample_position(db_session):
    """Create a sample position in the database."""
    position = PositionRecord(
        strategy="EMA_CROSSOVER",
        instrument_key="NSE_EQ:RELIANCE",
        side="BUY",
        entry_price=250000,  # 2500.00 in paisa
        quantity=100,
        status="open",
        opened_at=datetime.now(timezone.utc),
    )
    db_session.add(position)
    db_session.commit()
    return position


@pytest.fixture
def sample_trade(db_session):
    """Create a sample trade in the database."""
    trade = TradeRecord(
        strategy="EMA_CROSSOVER",
        instrument_key="NSE_EQ:RELIANCE",
        side="BUY",
        entry_price=250000,
        exit_price=255000,
        quantity=100,
        realized_pnl=50000,  # 500.00 in paisa
        fees=100,
        entry_time=datetime.now(timezone.utc) - timedelta(hours=1),
        exit_time=datetime.now(timezone.utc),
        holding_duration_seconds=3600,
    )
    db_session.add(trade)
    db_session.commit()
    return trade


@pytest.fixture
def sample_daily_pnl(db_session):
    """Create a sample daily PnL record."""
    pnl = DailyPnL(
        date=date.today(),
        realized_pnl=50000,
        unrealized_pnl=25000,
        total_pnl=75000,
        trades_count=5,
        win_count=3,
    )
    db_session.add(pnl)
    db_session.commit()
    return pnl


# TelegramNotifier Tests
class TestTelegramNotifier:
    """Tests for TelegramNotifier class."""

    def test_init_with_token(self, mock_env_token):
        """Test initialization with valid token."""
        notifier = TelegramNotifier()
        assert notifier.is_enabled() is True
        assert notifier.bot_token == "test_token_12345"
        assert notifier.chat_id == "123456789"

    def test_init_without_token(self, mock_env_empty):
        """Test initialization without token disables notifier."""
        notifier = TelegramNotifier()
        assert notifier.is_enabled() is False

    def test_init_with_explicit_params(self):
        """Test initialization with explicit parameters."""
        notifier = TelegramNotifier(
            bot_token="explicit_token",
            chat_id="explicit_chat",
        )
        assert notifier.is_enabled() is True
        assert notifier.bot_token == "explicit_token"
        assert notifier.chat_id == "explicit_chat"

    @pytest.mark.asyncio
    async def test_send_trade_entry_success(self, mock_env_token):
        """Test successful trade entry notification."""
        notifier = TelegramNotifier()

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_bot_class.return_value = mock_bot

            trade_data = {
                "symbol": "NSE_EQ:RELIANCE",
                "side": "BUY",
                "price": 2500.50,
                "quantity": 100,
                "strategy": "EMA_CROSSOVER",
            }

            result = await notifier.send_trade_entry(trade_data)

            assert result is True
            mock_bot.send_message.assert_called_once()
            call_args = mock_bot.send_message.call_args
            assert call_args.kwargs["chat_id"] == "123456789"
            text = call_args.kwargs.get("text", "")
            assert "TRADE ENTRY" in text
            assert "RELIANCE" in text
            assert "2,500.50" in text

    @pytest.mark.asyncio
    async def test_send_trade_entry_disabled(self, mock_env_empty):
        """Test trade entry notification when disabled."""
        notifier = TelegramNotifier()
        trade_data = {"symbol": "NSE_EQ:RELIANCE", "side": "BUY", "price": 2500.0, "quantity": 100}

        result = await notifier.send_trade_entry(trade_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_trade_exit_success(self, mock_env_token):
        """Test successful trade exit notification with P&L."""
        notifier = TelegramNotifier()

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_bot_class.return_value = mock_bot

            trade_data = {
                "symbol": "NSE_EQ:RELIANCE",
                "side": "BUY",
                "entry_price": 2500.00,
                "exit_price": 2550.00,
                "quantity": 100,
                "pnl": 5000.00,
                "pnl_pct": 2.0,
                "strategy": "EMA_CROSSOVER",
                "duration": "1h 30m",
            }

            result = await notifier.send_trade_exit(trade_data)

            assert result is True
            mock_bot.send_message.assert_called_once()
            call_args = mock_bot.send_message.call_args
            text = call_args.kwargs.get("text", "")
            assert "TRADE EXIT" in text
            assert "5,000.00" in text
            assert "+2.00%" in text  # Format includes + and 2 decimal places
            assert "1h 30m" in text

    @pytest.mark.asyncio
    async def test_send_trade_exit_loss(self, mock_env_token):
        """Test trade exit notification for losing trade."""
        notifier = TelegramNotifier()

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_bot_class.return_value = mock_bot

            trade_data = {
                "symbol": "NSE_EQ:TCS",
                "side": "SELL",
                "entry_price": 3500.00,
                "exit_price": 3550.00,
                "quantity": 50,
                "pnl": -2500.00,
                "pnl_pct": -1.43,
            }

            result = await notifier.send_trade_exit(trade_data)

            assert result is True
            call_args = mock_bot.send_message.call_args
            text = call_args.kwargs.get("text", "")
            assert "-2,500.00" in text
            assert "-1.43%" in text  # Format includes - and 2 decimal places

    @pytest.mark.asyncio
    async def test_send_daily_summary_success(self, mock_env_token):
        """Test successful daily summary notification."""
        notifier = TelegramNotifier()

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_bot_class.return_value = mock_bot

            pnl_data = {
                "realized_pnl": 5000.00,
                "unrealized_pnl": 2500.00,
                "total_pnl": 7500.00,
                "trades_count": 5,
                "win_count": 3,
                "loss_count": 2,
                "win_rate": 0.60,
            }

            result = await notifier.send_daily_summary(pnl_data)

            assert result is True
            mock_bot.send_message.assert_called_once()
            call_args = mock_bot.send_message.call_args
            text = call_args.kwargs.get("text", "")
            assert "DAILY P&L SUMMARY" in text
            assert "5,000.00" in text
            assert "60.0%" in text

    @pytest.mark.asyncio
    async def test_send_risk_alert(self, mock_env_token):
        """Test risk alert notification."""
        notifier = TelegramNotifier()

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_bot_class.return_value = mock_bot

            result = await notifier.send_risk_alert(
                alert_message="Daily loss limit exceeded!",
                alert_type="DAILY_LIMIT",
            )

            assert result is True
            mock_bot.send_message.assert_called_once()
            call_args = mock_bot.send_message.call_args
            text = call_args.kwargs.get("text", "")
            assert "RISK ALERT" in text
            assert "DAILY_LIMIT" in text
            assert "exceeded" in text

    @pytest.mark.asyncio
    async def test_send_circuit_breaker_alert(self, mock_env_token):
        """Test circuit breaker risk alert."""
        notifier = TelegramNotifier()

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_bot_class.return_value = mock_bot

            result = await notifier.send_risk_alert(
                alert_message="3 consecutive losses detected. Trading paused.",
                alert_type="CIRCUIT_BREAKER",
            )

            assert result is True
            call_args = mock_bot.send_message.call_args
            text = call_args.kwargs.get("text", "")
            assert "🚨" in text  # Circuit breaker emoji

    @pytest.mark.asyncio
    async def test_send_error_alert(self, mock_env_token):
        """Test system error notification."""
        notifier = TelegramNotifier()

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_bot_class.return_value = mock_bot

            result = await notifier.send_error_alert(
                error_message="Connection to broker failed!",
            )

            assert result is True
            mock_bot.send_message.assert_called_once()
            call_args = mock_bot.send_message.call_args
            text = call_args.kwargs.get("text", "")
            assert "SYSTEM ERROR" in text
            assert "Connection to broker failed" in text

    @pytest.mark.asyncio
    async def test_send_message_failure(self, mock_env_token):
        """Test handling of send message failure."""
        notifier = TelegramNotifier()

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock(side_effect=Exception("Network error"))
            mock_bot_class.return_value = mock_bot

            result = await notifier.send_trade_entry({
                "symbol": "NSE_EQ:RELIANCE",
                "side": "BUY",
                "price": 2500.0,
                "quantity": 100,
            })

            assert result is False

    def test_get_status_text_with_positions(self, mock_env_token, db_session, sample_position):
        """Test status text generation with open positions."""
        notifier = TelegramNotifier(db_session=db_session)
        status_text = notifier.get_status_text()

        assert "CURRENT POSITIONS" in status_text
        assert "RELIANCE" in status_text
        assert "BUY" in status_text
        assert "2,500.00" in status_text  # Formatted price

    def test_get_status_text_no_positions(self, mock_env_token, db_session):
        """Test status text generation with no positions."""
        notifier = TelegramNotifier(db_session=db_session)
        status_text = notifier.get_status_text()

        assert "No open positions" in status_text

    def test_get_status_text_no_db(self, mock_env_token):
        """Test status text generation without database."""
        notifier = TelegramNotifier()
        status_text = notifier.get_status_text()

        assert "Database not connected" in status_text

    def test_get_pnl_text_with_data(self, mock_env_token, db_session, sample_daily_pnl):
        """Test P&L text generation with data."""
        notifier = TelegramNotifier(db_session=db_session)
        pnl_text = notifier.get_pnl_text()

        assert "TODAY'S P&L" in pnl_text
        assert "500.00" in pnl_text  # realized_pnl / 100
        assert "250.00" in pnl_text  # unrealized_pnl / 100
        assert "750.00" in pnl_text  # total_pnl / 100

    def test_get_pnl_text_no_data(self, mock_env_token, db_session):
        """Test P&L text generation without data."""
        notifier = TelegramNotifier(db_session=db_session)
        pnl_text = notifier.get_pnl_text()

        assert "No trades today" in pnl_text

    def test_get_trades_text(self, mock_env_token, db_session, sample_trade):
        """Test trades text generation."""
        notifier = TelegramNotifier(db_session=db_session)
        trades_text = notifier.get_trades_text(n=5)

        assert "LAST 1 TRADES" in trades_text
        assert "RELIANCE" in trades_text
        assert "EMA_CROSSOVER" in trades_text
        assert "500.00" in trades_text or "+500.00" in trades_text  # P&L in rupees


# TelegramBot Tests
class TestTelegramBot:
    """Tests for TelegramBot command handlers."""

    def test_init_with_token(self, mock_env_token):
        """Test initialization with valid token."""
        bot = TelegramBot()
        assert bot.is_enabled() is True
        assert bot.bot_token == "test_token_12345"

    def test_init_without_token(self, mock_env_empty):
        """Test initialization without token disables bot."""
        bot = TelegramBot()
        assert bot.is_enabled() is False

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_env_token):
        """Test bot start and stop."""
        bot = TelegramBot()

        with patch("telegram.ext.Application") as mock_app_class:
            mock_app = MagicMock()
            mock_app.initialize = AsyncMock()
            mock_app.start = AsyncMock()
            mock_app.stop = AsyncMock()
            mock_app.shutdown = AsyncMock()
            mock_app.add_handler = MagicMock()
            mock_app_class.builder.return_value.token.return_value.build.return_value = mock_app

            # Test start
            result = await bot.start()
            assert result is True
            mock_app.initialize.assert_called_once()
            mock_app.start.assert_called_once()

            # Test stop
            result = await bot.stop()
            assert result is True
            mock_app.stop.assert_called_once()
            mock_app.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_disabled(self, mock_env_empty):
        """Test start when bot is disabled."""
        bot = TelegramBot()
        result = await bot.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_cmd_start(self, mock_env_token):
        """Test /start command handler."""
        bot = TelegramBot()

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()

        await bot._cmd_start(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "Welcome to Trading Bot" in text
        assert "/status" in text
        assert "/help" in text

    @pytest.mark.asyncio
    async def test_cmd_help(self, mock_env_token):
        """Test /help command handler."""
        bot = TelegramBot()

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()

        await bot._cmd_help(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "Trading Bot Commands" in text
        assert "/status" in text
        assert "/kill" in text

    @pytest.mark.asyncio
    async def test_cmd_status(self, mock_env_token, db_session, sample_position):
        """Test /status command handler."""
        bot = TelegramBot(db_session=db_session)

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()

        await bot._cmd_status(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "CURRENT POSITIONS" in text
        assert "RELIANCE" in text

    @pytest.mark.asyncio
    async def test_cmd_pnl(self, mock_env_token, db_session, sample_daily_pnl):
        """Test /pnl command handler."""
        bot = TelegramBot(db_session=db_session)

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()

        await bot._cmd_pnl(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "TODAY'S P&L" in text

    @pytest.mark.asyncio
    async def test_cmd_positions(self, mock_env_token, db_session, sample_position):
        """Test /positions command handler."""
        bot = TelegramBot(db_session=db_session)

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()

        await bot._cmd_positions(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "CURRENT POSITIONS" in text

    @pytest.mark.asyncio
    async def test_cmd_trades_default(self, mock_env_token, db_session, sample_trade):
        """Test /trades command with default count."""
        bot = TelegramBot(db_session=db_session)

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = []

        await bot._cmd_trades(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "LAST 1 TRADES" in text

    @pytest.mark.asyncio
    async def test_cmd_trades_custom_count(self, mock_env_token, db_session, sample_trade):
        """Test /trades command with custom count."""
        bot = TelegramBot(db_session=db_session)

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = ["5"]

        await bot._cmd_trades(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "LAST 1 TRADES" in text  # Only 1 trade in DB

    @pytest.mark.asyncio
    async def test_cmd_trades_invalid_count(self, mock_env_token, db_session):
        """Test /trades command with invalid count."""
        bot = TelegramBot(db_session=db_session)

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = ["invalid"]

        await bot._cmd_trades(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "Invalid number" in text

    @pytest.mark.asyncio
    async def test_cmd_alert_success(self, mock_env_token):
        """Test /alert command with valid arguments."""
        bot = TelegramBot()

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = ["NSE_EQ:RELIANCE", "2500.50"]

        await bot._cmd_alert(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "Price Alert Set" in text
        assert "RELIANCE" in text
        assert "2,500.50" in text

        # Verify alert was stored
        assert "NSE_EQ:RELIANCE" in bot._price_alerts
        assert bot._price_alerts["NSE_EQ:RELIANCE"] == 2500.50

    @pytest.mark.asyncio
    async def test_cmd_alert_missing_args(self, mock_env_token):
        """Test /alert command with missing arguments."""
        bot = TelegramBot()

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = ["NSE_EQ:RELIANCE"]  # Missing price

        await bot._cmd_alert(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "Usage" in text

    @pytest.mark.asyncio
    async def test_cmd_alert_invalid_price(self, mock_env_token):
        """Test /alert command with invalid price."""
        bot = TelegramBot()

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = ["NSE_EQ:RELIANCE", "not_a_number"]

        await bot._cmd_alert(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "Invalid price" in text

    @pytest.mark.asyncio
    async def test_cmd_kill_request_confirmation(self, mock_env_token):
        """Test /kill command requests confirmation."""
        bot = TelegramBot()

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = []

        await bot._cmd_kill(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "EMERGENCY STOP" in text
        assert "confirm" in text.lower()

    @pytest.mark.asyncio
    async def test_cmd_kill_confirmed_no_callback(self, mock_env_token):
        """Test /kill confirm without callback configured."""
        bot = TelegramBot()

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = ["confirm"]

        await bot._cmd_kill(mock_update, mock_context)

        assert mock_update.message.reply_text.call_count == 2
        # First call: activation message
        # Second call: no callback configured message

    @pytest.mark.asyncio
    async def test_cmd_kill_confirmed_with_callback(self, mock_env_token):
        """Test /kill confirm with callback configured."""
        kill_callback = AsyncMock()
        bot = TelegramBot(kill_switch_callback=kill_callback)

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = ["confirm"]

        await bot._cmd_kill(mock_update, mock_context)

        kill_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_kill_callback_failure(self, mock_env_token):
        """Test /kill confirm when callback fails."""
        kill_callback = AsyncMock(side_effect=Exception("Kill failed"))
        bot = TelegramBot(kill_switch_callback=kill_callback)

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock()
        mock_context.args = ["confirm"]

        await bot._cmd_kill(mock_update, mock_context)

        # Should still send error message
        assert mock_update.message.reply_text.call_count >= 2

    def test_check_price_alerts_triggered(self, mock_env_token):
        """Test price alert checking when triggered."""
        bot = TelegramBot()
        bot._price_alerts["NSE_EQ:RELIANCE"] = 2500.00

        result = bot.check_price_alerts("NSE_EQ:RELIANCE", 2505.00)

        assert len(result) == 1
        assert "PRICE ALERT TRIGGERED" in result[0]
        assert "NSE_EQ:RELIANCE" not in bot._price_alerts  # Should be removed

    def test_check_price_alerts_not_triggered(self, mock_env_token):
        """Test price alert checking when not triggered."""
        bot = TelegramBot()
        bot._price_alerts["NSE_EQ:RELIANCE"] = 2500.00

        result = bot.check_price_alerts("NSE_EQ:RELIANCE", 2400.00)

        assert len(result) == 0
        assert "NSE_EQ:RELIANCE" in bot._price_alerts  # Should still be there

    def test_check_price_alerts_no_alerts(self, mock_env_token):
        """Test price alert checking with no alerts set."""
        bot = TelegramBot()

        result = bot.check_price_alerts("NSE_EQ:RELIANCE", 2500.00)

        assert len(result) == 0


# Factory Function Tests
class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_notifier_from_settings(self, mock_env_token, db_session):
        """Test factory function for TelegramNotifier."""
        notifier = create_notifier_from_settings(db_session=db_session)

        assert isinstance(notifier, TelegramNotifier)
        assert notifier.is_enabled() is True
        assert notifier.bot_token == "test_token_12345"
        assert notifier.db_session is db_session

    def test_create_bot_from_settings(self, mock_env_token, db_session):
        """Test factory function for TelegramBot."""
        kill_callback = lambda: None
        bot = create_bot_from_settings(
            db_session=db_session,
            kill_switch_callback=kill_callback,
        )

        assert isinstance(bot, TelegramBot)
        assert bot.is_enabled() is True
        assert bot.db_session is db_session
        assert bot.kill_switch_callback is kill_callback

    def test_create_notifier_disabled(self, mock_env_empty):
        """Test factory function when not configured."""
        notifier = create_notifier_from_settings()

        assert notifier.is_enabled() is False

    def test_create_bot_disabled(self, mock_env_empty):
        """Test factory function when not configured."""
        bot = create_bot_from_settings()

        assert bot.is_enabled() is False
