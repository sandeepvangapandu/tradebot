"""Telegram bot for trade notifications and commands.

Uses python-telegram-bot v20+ for async operations.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from loguru import logger

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

IST = ZoneInfo("Asia/Kolkata")


class TelegramNotifier:
    """Telegram notification handler for trade alerts and system notifications.

    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.
    Gracefully disables itself if configuration is missing.

    Args:
        bot_token: Telegram bot token from @BotFather.
        chat_id: Telegram chat ID for notifications.
        db_session: Optional SQLAlchemy session for command handlers.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        db_session: Any | None = None,
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.db_session = db_session
        self._enabled = bool(self.bot_token and self.chat_id)
        self._app = None

        if self._enabled:
            logger.info("Telegram notifier initialized")
        else:
            logger.warning(
                "Telegram notifier disabled - missing bot_token or chat_id"
            )

    def is_enabled(self) -> bool:
        """Check if Telegram bot is properly configured."""
        return self._enabled

    async def send_trade_entry(self, trade_data: dict) -> bool:
        """Send a trade entry notification.

        Args:
            trade_data: Dictionary with keys:
                - symbol: Instrument symbol (e.g., 'NSE_EQ:RELIANCE')
                - side: 'BUY' or 'SELL'
                - price: Entry price in rupees
                - quantity: Number of units
                - strategy: Strategy name
                - timestamp: Entry timestamp (optional)

        Returns:
            True if message sent successfully.
        """
        if not self._enabled:
            return False

        try:
            symbol = trade_data.get("symbol", "Unknown")
            side = trade_data.get("side", "BUY")
            price = trade_data.get("price", 0.0)
            quantity = trade_data.get("quantity", 0)
            strategy = trade_data.get("strategy", "")
            timestamp = trade_data.get("timestamp", datetime.now(IST))

            side_emoji = "📈" if side == "BUY" else "📉"

            message = f"🟢 *TRADE ENTRY*\n\n"
            message += f"*Symbol:* `{symbol}`\n"
            message += f"*Side:* {side_emoji} {side}\n"
            message += f"*Price:* ₹{price:,.2f}\n"
            message += f"*Quantity:* {quantity}\n"
            if strategy:
                message += f"*Strategy:* {strategy}\n"
            message += f"*Time:* {timestamp.strftime('%H:%M:%S IST') if isinstance(timestamp, datetime) else timestamp}"

            success = await self._send_message(message)
            if success:
                logger.info(f"Trade entry notification sent for {symbol}")
            return success

        except Exception as e:
            logger.error(f"Failed to send trade entry notification: {e}")
            return False

    async def send_trade_exit(self, trade_data: dict) -> bool:
        """Send a trade exit notification with P&L.

        Args:
            trade_data: Dictionary with keys:
                - symbol: Instrument symbol
                - side: 'BUY' or 'SELL'
                - entry_price: Entry price in rupees
                - exit_price: Exit price in rupees
                - quantity: Number of units
                - pnl: Realized P&L in rupees
                - pnl_pct: P&L percentage (optional)
                - strategy: Strategy name
                - timestamp: Exit timestamp (optional)
                - duration: Trade duration string (optional)

        Returns:
            True if message sent successfully.
        """
        if not self._enabled:
            return False

        try:
            symbol = trade_data.get("symbol", "Unknown")
            side = trade_data.get("side", "BUY")
            entry_price = trade_data.get("entry_price", 0.0)
            exit_price = trade_data.get("exit_price", 0.0)
            quantity = trade_data.get("quantity", 0)
            pnl = trade_data.get("pnl", 0.0)
            pnl_pct = trade_data.get("pnl_pct")
            strategy = trade_data.get("strategy", "")
            timestamp = trade_data.get("timestamp", datetime.now(IST))
            duration = trade_data.get("duration", "")

            side_emoji = "📈" if side == "BUY" else "📉"
            pnl_emoji = "✅" if pnl >= 0 else "❌"
            pnl_sign = "+" if pnl >= 0 else ""

            message = f"🔴 *TRADE EXIT*\n\n"
            message += f"*Symbol:* `{symbol}`\n"
            message += f"*Side:* {side_emoji} {side}\n"
            message += f"*Entry:* ₹{entry_price:,.2f}\n"
            message += f"*Exit:* ₹{exit_price:,.2f}\n"
            message += f"*Quantity:* {quantity}\n"
            if strategy:
                message += f"*Strategy:* {strategy}\n"
            message += f"\n*P&L:* {pnl_emoji} ₹{pnl_sign}{pnl:,.2f}"
            if pnl_pct is not None:
                message += f" ({pnl_sign}{pnl_pct:.2f}%)"
            if duration:
                message += f"\n*Duration:* {duration}"
            message += f"\n*Time:* {timestamp.strftime('%H:%M:%S IST') if isinstance(timestamp, datetime) else timestamp}"

            success = await self._send_message(message)
            if success:
                logger.info(f"Trade exit notification sent for {symbol}, P&L: ₹{pnl:,.2f}")
            return success

        except Exception as e:
            logger.error(f"Failed to send trade exit notification: {e}")
            return False

    async def send_daily_summary(self, pnl_data: dict) -> bool:
        """Send daily P&L summary at market close.

        Args:
            pnl_data: Dictionary with keys:
                - realized_pnl: float
                - unrealized_pnl: float (optional)
                - total_pnl: float (optional)
                - trades_count: int
                - win_count: int
                - loss_count: int
                - win_rate: float
                - date: date object (optional, defaults to today)

        Returns:
            True if message sent successfully.
        """
        if not self._enabled:
            return False

        try:
            realized_pnl = pnl_data.get("realized_pnl", 0.0)
            unrealized_pnl = pnl_data.get("unrealized_pnl", 0.0)
            total_pnl = pnl_data.get("total_pnl", realized_pnl + unrealized_pnl)
            trades_count = pnl_data.get("trades_count", 0)
            win_count = pnl_data.get("win_count", 0)
            loss_count = pnl_data.get("loss_count", 0)
            win_rate = pnl_data.get("win_rate", 0.0)
            summary_date = pnl_data.get("date", date.today())

            pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
            pnl_sign = "+" if total_pnl >= 0 else ""

            message = f"📊 *DAILY P&L SUMMARY*\n\n"
            message += f"*Date:* {summary_date.strftime('%Y-%m-%d')}\n"
            message += f"*Realized P&L:* ₹{realized_pnl:+,.2f}\n"
            if unrealized_pnl != 0:
                message += f"*Unrealized P&L:* ₹{unrealized_pnl:+,.2f}\n"
            message += f"*Total P&L:* {pnl_emoji} {pnl_sign}₹{total_pnl:,.2f}\n"
            message += f"\n*Total Trades:* {trades_count}\n"
            message += f"*Wins:* {win_count} ✅\n"
            message += f"*Losses:* {loss_count} ❌\n"
            if trades_count > 0:
                message += f"*Win Rate:* {win_rate:.1%}\n"

            success = await self._send_message(message)
            if success:
                logger.info(f"Daily summary sent for {summary_date}")
            return success

        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False

    async def send_risk_alert(self, alert_message: str, alert_type: str = "RISK") -> bool:
        """Send a risk limit breach alert.

        Args:
            alert_message: Alert message text.
            alert_type: Type of alert (e.g., 'CIRCUIT_BREAKER', 'DAILY_LIMIT').

        Returns:
            True if message sent successfully.
        """
        if not self._enabled:
            return False

        try:
            emoji = "🚨" if "BREAKER" in alert_type or "KILL" in alert_type else "⚠️"
            timestamp = datetime.now(IST).strftime("%H:%M:%S IST")

            message = f"{emoji} *RISK ALERT: {alert_type}*\n\n"
            message += f"{alert_message}\n\n"
            message += f"*Time:* {timestamp}"

            success = await self._send_message(message)
            if success:
                logger.warning(f"Risk alert sent: {alert_type}")
            return success

        except Exception as e:
            logger.error(f"Failed to send risk alert: {e}")
            return False

    async def send_error_alert(self, error_message: str) -> bool:
        """Send a system error notification.

        Args:
            error_message: Error message text.

        Returns:
            True if message sent successfully.
        """
        if not self._enabled:
            return False

        try:
            timestamp = datetime.now(IST).strftime("%H:%M:%S IST")

            message = f"🔥 *SYSTEM ERROR*\n\n"
            message += f"{error_message}\n\n"
            message += f"*Time:* {timestamp}"

            success = await self._send_message(message)
            if success:
                logger.error(f"Error alert sent: {error_message[:100]}...")
            return success

        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")
            return False

    async def _send_message(self, text: str) -> bool:
        """Send a message to the configured chat.

        Args:
            text: Message text (Markdown format supported).

        Returns:
            True if message sent successfully.
        """
        if not self._enabled:
            return False

        try:
            from telegram import Bot
            bot = Bot(token=self.bot_token)
            await bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown",
            )
            logger.debug(f"Telegram message sent: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def get_status_text(self) -> str:
        """Get formatted status text for /status command.

        Returns:
            Status message with current positions.
        """
        if not self.db_session:
            return "❌ Database not connected"

        try:
            from src.persistence.models import PositionRecord

            positions = (
                self.db_session.query(PositionRecord)
                .filter(PositionRecord.status == "open")
                .all()
            )

            if not positions:
                return "📭 No open positions"

            message = f"📊 *CURRENT POSITIONS* ({len(positions)})\n\n"

            for pos in positions:
                side_emoji = "📈" if pos.side == "BUY" else "📉"
                entry_price_rupees = pos.entry_price / 100.0
                message += (
                    f"*{pos.instrument_key}*\n"
                    f"  {side_emoji} {pos.side} {pos.quantity} @ ₹{entry_price_rupees:,.2f}\n"
                    f"  Strategy: {pos.strategy}\n\n"
                )

            return message

        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return f"❌ Error fetching status: {e}"

    def get_pnl_text(self) -> str:
        """Get formatted P&L text for /pnl command.

        Returns:
            P&L summary message.
        """
        if not self.db_session:
            return "❌ Database not connected"

        try:
            from src.persistence.models import DailyPnL

            today = date.today()
            today_pnl = (
                self.db_session.query(DailyPnL)
                .filter(DailyPnL.date == today)
                .first()
            )

            if not today_pnl:
                return "📊 No trades today yet"

            total_pnl_rupees = today_pnl.total_pnl / 100.0
            pnl_emoji = "🟢" if total_pnl_rupees >= 0 else "🔴"

            message = f"📊 *TODAY'S P&L*\n\n"
            message += f"*Realized:* ₹{today_pnl.realized_pnl / 100.0:,.2f}\n"
            message += f"*Unrealized:* ₹{today_pnl.unrealized_pnl / 100.0:,.2f}\n"
            message += f"*Total:* {pnl_emoji} ₹{total_pnl_rupees:,.2f}\n"
            message += f"\n*Trades:* {today_pnl.trades_count} ({today_pnl.win_count} wins)"

            return message

        except Exception as e:
            logger.error(f"Failed to get P&L: {e}")
            return f"❌ Error fetching P&L: {e}"

    def get_positions_text(self) -> str:
        """Get formatted positions text for /positions command.

        Returns:
            Open positions message.
        """
        return self.get_status_text()

    def get_trades_text(self, n: int = 10) -> str:
        """Get formatted trades text for /trades command.

        Args:
            n: Number of recent trades to show.

        Returns:
            Recent trades message.
        """
        if not self.db_session:
            return "❌ Database not connected"

        try:
            from src.persistence.models import TradeRecord

            trades = (
                self.db_session.query(TradeRecord)
                .order_by(TradeRecord.exit_time.desc())
                .limit(n)
                .all()
            )

            if not trades:
                return "📋 No trades found"

            message = f"📋 *LAST {len(trades)} TRADES*\n\n"

            for trade in trades:
                pnl_rupees = trade.realized_pnl / 100.0
                pnl_emoji = "✅" if pnl_rupees >= 0 else "❌"
                entry_price = trade.entry_price / 100.0
                exit_price = trade.exit_price / 100.0

                message += (
                    f"*{trade.instrument_key}* ({trade.strategy})\n"
                    f"  {trade.side} {trade.quantity} @ ₹{entry_price:,.2f} → ₹{exit_price:,.2f}\n"
                    f"  P&L: {pnl_emoji} ₹{pnl_rupees:+,.2f}\n\n"
                )

            return message

        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return f"❌ Error fetching trades: {e}"


class TelegramBot:
    """Interactive Telegram bot for trading commands.

    Handles user commands for checking status, P&L, positions, and
    emergency kill switch. Runs as a separate async service.

    Args:
        bot_token: Telegram bot token from @BotFather.
        db_session: Optional SQLAlchemy session for database queries.
        kill_switch_callback: Optional callback function for /kill command.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        db_session: Any | None = None,
        kill_switch_callback: callable | None = None,
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.db_session = db_session
        self.kill_switch_callback = kill_switch_callback
        self._enabled = bool(self.bot_token)
        self._app = None
        self._price_alerts: dict[str, float] = {}

        if self._enabled:
            logger.info("Telegram bot initialized")
        else:
            logger.warning("Telegram bot disabled - missing bot_token")

    def is_enabled(self) -> bool:
        """Check if Telegram bot is properly configured."""
        return self._enabled

    async def start(self) -> bool:
        """Start the Telegram bot.

        Returns:
            True if bot started successfully.
        """
        if not self._enabled:
            logger.warning("Cannot start Telegram bot - not enabled")
            return False

        try:
            from telegram.ext import Application

            self._app = Application.builder().token(self.bot_token).build()
            self._register_handlers()

            await self._app.initialize()
            await self._app.start()
            logger.info("Telegram bot started")
            return True

        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the Telegram bot.

        Returns:
            True if bot stopped successfully.
        """
        if not self._app:
            return True

        try:
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram bot stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop Telegram bot: {e}")
            return False

    def _register_handlers(self) -> None:
        """Register command handlers."""
        from telegram.ext import CommandHandler

        handlers = [
            CommandHandler("start", self._cmd_start),
            CommandHandler("help", self._cmd_help),
            CommandHandler("status", self._cmd_status),
            CommandHandler("pnl", self._cmd_pnl),
            CommandHandler("positions", self._cmd_positions),
            CommandHandler("trades", self._cmd_trades),
            CommandHandler("alert", self._cmd_alert),
            CommandHandler("kill", self._cmd_kill),
        ]

        for handler in handlers:
            self._app.add_handler(handler)

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        welcome_message = (
            "👋 *Welcome to Trading Bot!*\n\n"
            "I'm your trading assistant. Here are the available commands:\n\n"
            "📊 *Status Commands*\n"
            "  /status - Current positions and P&L\n"
            "  /pnl - Today's P&L summary\n"
            "  /positions - List open positions\n"
            "  /trades [n] - Show last n trades (default 10)\n\n"
            "⚙️ *Control Commands*\n"
            "  /alert <symbol> <price> - Set price alert\n"
            "  /kill - Emergency stop (requires confirmation)\n\n"
            "❓ *Help*\n"
            "  /help - Show this help message\n\n"
            "Stay profitable! 📈"
        )
        await update.message.reply_text(welcome_message, parse_mode="Markdown")
        logger.info(f"User {update.effective_user.id} started the bot")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_message = (
            "📚 *Trading Bot Commands*\n\n"
            "*Status Commands:*\n"
            "  `/status` - Show current positions and overall status\n"
            "  `/pnl` - Display today's realized and unrealized P&L\n"
            "  `/positions` - List all open positions with entry prices\n"
            "  `/trades [n]` - Show last n completed trades (default: 10)\n\n"
            "*Control Commands:*\n"
            "  `/alert <symbol> <price>` - Set a price alert\n"
            "    Example: `/alert NSE_EQ:RELIANCE 2500.50`\n"
            "  `/kill` - Emergency stop - closes all positions immediately\n"
            "    ⚠️ Requires confirmation!\n\n"
            "*Getting Started:*\n"
            "  `/start` - Show welcome message\n"
            "  `/help` - Show this help message\n\n"
            "All times are in IST (Asia/Kolkata)."
        )
        await update.message.reply_text(help_message, parse_mode="Markdown")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        notifier = TelegramNotifier(
            bot_token=self.bot_token,
            db_session=self.db_session
        )
        status_text = notifier.get_status_text()

        # Add system status
        timestamp = datetime.now(IST).strftime("%H:%M:%S IST")
        status_text += f"\n🕐 *Last Updated:* {timestamp}"

        await update.message.reply_text(status_text, parse_mode="Markdown")
        logger.debug(f"Status requested by user {update.effective_user.id}")

    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pnl command."""
        notifier = TelegramNotifier(
            bot_token=self.bot_token,
            db_session=self.db_session
        )
        pnl_text = notifier.get_pnl_text()
        await update.message.reply_text(pnl_text, parse_mode="Markdown")
        logger.debug(f"P&L requested by user {update.effective_user.id}")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command."""
        notifier = TelegramNotifier(
            bot_token=self.bot_token,
            db_session=self.db_session
        )
        positions_text = notifier.get_positions_text()
        await update.message.reply_text(positions_text, parse_mode="Markdown")
        logger.debug(f"Positions requested by user {update.effective_user.id}")

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /trades command."""
        # Parse number of trades from arguments
        n = 10
        if context.args:
            try:
                n = int(context.args[0])
                n = max(1, min(50, n))  # Limit between 1 and 50
            except ValueError:
                await update.message.reply_text(
                    "⚠️ Invalid number. Usage: `/trades [n]` where n is 1-50",
                    parse_mode="Markdown"
                )
                return

        notifier = TelegramNotifier(
            bot_token=self.bot_token,
            db_session=self.db_session
        )
        trades_text = notifier.get_trades_text(n)
        await update.message.reply_text(trades_text, parse_mode="Markdown")
        logger.debug(f"Last {n} trades requested by user {update.effective_user.id}")

    async def _cmd_alert(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /alert command."""
        if len(context.args) < 2:
            await update.message.reply_text(
                "⚠️ Usage: `/alert <symbol> <price>`\n"
                "Example: `/alert NSE_EQ:RELIANCE 2500.50`",
                parse_mode="Markdown"
            )
            return

        symbol = context.args[0]
        try:
            price = float(context.args[1])
        except ValueError:
            await update.message.reply_text(
                "⚠️ Invalid price. Please provide a valid number."
            )
            return

        self._price_alerts[symbol] = price

        await update.message.reply_text(
            f"🔔 *Price Alert Set*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Target Price: ₹{price:,.2f}\n\n"
            f"You'll be notified when the price is reached.",
            parse_mode="Markdown"
        )
        logger.info(f"Price alert set by user {update.effective_user.id}: {symbol} @ ₹{price}")

    async def _cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /kill command (emergency stop)."""
        # Check for confirmation
        if context.args and context.args[0].lower() == "confirm":
            await update.message.reply_text(
                "🚨 *EMERGENCY STOP ACTIVATED* 🚨\n\n"
                "Closing all positions immediately...",
                parse_mode="Markdown"
            )
            logger.critical(f"KILL SWITCH activated by user {update.effective_user.id}")

            if self.kill_switch_callback:
                try:
                    await self.kill_switch_callback()
                    await update.message.reply_text(
                        "✅ All positions closed. Trading bot stopped."
                    )
                except Exception as e:
                    logger.error(f"Kill switch failed: {e}")
                    await update.message.reply_text(
                        f"❌ *Error during kill switch:* {e}\n"
                        f"Please check the system immediately!",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text(
                    "⚠️ No kill switch callback configured. "
                    "Please stop the bot manually."
                )
        else:
            # Request confirmation
            await update.message.reply_text(
                "🚨 *EMERGENCY STOP* 🚨\n\n"
                "This will immediately close ALL positions and stop trading.\n\n"
                "⚠️ *This action cannot be undone!*\n\n"
                "To confirm, type:\n"
                "`/kill confirm`",
                parse_mode="Markdown"
            )
            logger.warning(f"Kill switch requested by user {update.effective_user.id} (awaiting confirmation)")

    def check_price_alerts(self, symbol: str, current_price: float) -> list[str]:
        """Check if any price alerts are triggered.

        Args:
            symbol: Instrument symbol.
            current_price: Current market price.

        Returns:
            List of triggered alert messages.
        """
        triggered = []

        if symbol in self._price_alerts:
            target_price = self._price_alerts[symbol]

            # Check if price crossed the target (either direction)
            if abs(current_price - target_price) / target_price < 0.01:  # Within 1%
                message = (
                    f"🔔 *PRICE ALERT TRIGGERED*\n\n"
                    f"Symbol: `{symbol}`\n"
                    f"Target: ₹{target_price:,.2f}\n"
                    f"Current: ₹{current_price:,.2f}\n\n"
                    f"Time: {datetime.now(IST).strftime('%H:%M:%S IST')}"
                )
                triggered.append(message)
                del self._price_alerts[symbol]  # Remove triggered alert

        return triggered


def create_notifier_from_settings(db_session: Any | None = None) -> TelegramNotifier:
    """Factory function to create TelegramNotifier from environment variables.

    Args:
        db_session: Optional database session for command handlers.

    Returns:
        Configured TelegramNotifier instance.
    """
    return TelegramNotifier(db_session=db_session)


def create_bot_from_settings(
    db_session: Any | None = None,
    kill_switch_callback: callable | None = None,
) -> TelegramBot:
    """Factory function to create TelegramBot from environment variables.

    Args:
        db_session: Optional database session for command handlers.
        kill_switch_callback: Optional callback for kill switch command.

    Returns:
        Configured TelegramBot instance.
    """
    return TelegramBot(
        db_session=db_session,
        kill_switch_callback=kill_switch_callback,
    )
