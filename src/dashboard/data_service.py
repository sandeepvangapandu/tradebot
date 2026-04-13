"""Dashboard data service - queries database and formats for display."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.persistence.models import DailyPnL, PositionRecord, TradeRecord

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class PositionView:
    """Formatted position for dashboard display."""

    instrument_key: str
    strategy: str
    side: str
    entry_price: float  # in rupees
    quantity: int
    status: str
    opened_at: datetime
    unrealized_pnl: float = 0.0  # in rupees


@dataclass
class TradeView:
    """Formatted trade for dashboard display."""

    instrument_key: str
    strategy: str
    side: str
    entry_price: float  # in rupees
    exit_price: float  # in rupees
    quantity: int
    realized_pnl: float  # in rupees
    fees: float  # in rupees
    entry_time: datetime
    exit_time: datetime
    duration_minutes: float


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""

    strategy_name: str
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    total_pnl: float  # in rupees
    avg_win: float  # in rupees
    avg_loss: float  # in rupees
    profit_factor: float


@dataclass
class RiskMetrics:
    """Current risk usage metrics."""

    open_positions: int
    max_positions: int
    position_usage_pct: float
    deployed_capital: float  # in rupees
    max_capital: float  # in rupees
    capital_usage_pct: float
    daily_pnl: float  # in rupees
    daily_loss_limit: float  # in rupees
    daily_loss_usage_pct: float


class DashboardDataService:
    """Service layer for dashboard data queries.

    Converts paisa (integer) values to rupees (float) for display.
    """

    def __init__(self, session: Session):
        self.session = session

    @staticmethod
    def _paisa_to_rupees(paisa: int) -> float:
        """Convert paisa to rupees."""
        return paisa / 100.0

    def get_current_positions(
        self, current_prices: Optional[dict] = None
    ) -> List[PositionView]:
        """Get all open positions with unrealized P&L.

        Args:
            current_prices: Optional dict of instrument_key -> current price (paisa)

        Returns:
            List of PositionView objects.
        """
        positions = (
            self.session.query(PositionRecord)
            .filter(PositionRecord.status == "open")
            .order_by(PositionRecord.opened_at.desc())
            .all()
        )

        result = []
        for pos in positions:
            unrealized_pnl = 0.0
            if current_prices and pos.instrument_key in current_prices:
                current_price = current_prices[pos.instrument_key]
                price_diff = current_price - pos.entry_price
                if pos.side == "SELL":
                    price_diff = -price_diff
                unrealized_pnl = self._paisa_to_rupees(price_diff * pos.quantity)

            result.append(
                PositionView(
                    instrument_key=pos.instrument_key,
                    strategy=pos.strategy,
                    side=pos.side,
                    entry_price=self._paisa_to_rupees(pos.entry_price),
                    quantity=pos.quantity,
                    status=pos.status,
                    opened_at=pos.opened_at,
                    unrealized_pnl=unrealized_pnl,
                )
            )

        return result

    def get_today_trades(self) -> List[TradeView]:
        """Get today's completed trades.

        Returns:
            List of TradeView objects.
        """
        today = datetime.now(IST).date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())

        trades = (
            self.session.query(TradeRecord)
            .filter(TradeRecord.exit_time >= today_start)
            .filter(TradeRecord.exit_time <= today_end)
            .order_by(TradeRecord.exit_time.desc())
            .all()
        )

        return [
            TradeView(
                instrument_key=t.instrument_key,
                strategy=t.strategy,
                side=t.side,
                entry_price=self._paisa_to_rupees(t.entry_price),
                exit_price=self._paisa_to_rupees(t.exit_price),
                quantity=t.quantity,
                realized_pnl=self._paisa_to_rupees(t.realized_pnl),
                fees=self._paisa_to_rupees(t.fees),
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                duration_minutes=t.holding_duration_seconds / 60.0,
            )
            for t in trades
        ]

    def get_trade_history(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        strategy: Optional[str] = None,
        instrument: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get filtered trade history.

        Args:
            start_date: Filter trades from this date (inclusive)
            end_date: Filter trades up to this date (inclusive)
            strategy: Filter by strategy name
            instrument: Filter by instrument key

        Returns:
            DataFrame with trade data.
        """
        query = self.session.query(TradeRecord)

        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            query = query.filter(TradeRecord.exit_time >= start_dt)

        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            query = query.filter(TradeRecord.exit_time <= end_dt)

        if strategy:
            query = query.filter(TradeRecord.strategy == strategy)

        if instrument:
            query = query.filter(TradeRecord.instrument_key == instrument)

        trades = query.order_by(TradeRecord.exit_time.desc()).all()

        if not trades:
            return pd.DataFrame()

        data = {
            "instrument_key": [t.instrument_key for t in trades],
            "strategy": [t.strategy for t in trades],
            "side": [t.side for t in trades],
            "entry_price": [self._paisa_to_rupees(t.entry_price) for t in trades],
            "exit_price": [self._paisa_to_rupees(t.exit_price) for t in trades],
            "quantity": [t.quantity for t in trades],
            "realized_pnl": [self._paisa_to_rupees(t.realized_pnl) for t in trades],
            "fees": [self._paisa_to_rupees(t.fees) for t in trades],
            "entry_time": [t.entry_time for t in trades],
            "exit_time": [t.exit_time for t in trades],
            "duration_minutes": [t.holding_duration_seconds / 60.0 for t in trades],
        }

        return pd.DataFrame(data)

    def get_strategy_performance(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> List[StrategyPerformance]:
        """Get performance metrics per strategy.

        Args:
            start_date: Filter trades from this date
            end_date: Filter trades up to this date

        Returns:
            List of StrategyPerformance objects.
        """
        query = self.session.query(TradeRecord)

        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            query = query.filter(TradeRecord.exit_time >= start_dt)

        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            query = query.filter(TradeRecord.exit_time <= end_dt)

        trades = query.all()

        # Group by strategy
        strategy_stats = {}
        for trade in trades:
            name = trade.strategy
            if name not in strategy_stats:
                strategy_stats[name] = {
                    "trades": [],
                    "wins": [],
                    "losses": [],
                }

            pnl = trade.realized_pnl
            strategy_stats[name]["trades"].append(trade)
            if pnl > 0:
                strategy_stats[name]["wins"].append(pnl)
            else:
                strategy_stats[name]["losses"].append(pnl)

        result = []
        for name, stats in strategy_stats.items():
            wins = stats["wins"]
            losses = stats["losses"]
            total_trades = len(stats["trades"])

            win_count = len(wins)
            loss_count = len(losses)
            win_rate = win_count / total_trades if total_trades > 0 else 0.0

            total_pnl = sum(wins) + sum(losses)
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            profit_factor = (
                sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0.0
            )

            result.append(
                StrategyPerformance(
                    strategy_name=name,
                    total_trades=total_trades,
                    win_count=win_count,
                    loss_count=loss_count,
                    win_rate=win_rate,
                    total_pnl=self._paisa_to_rupees(total_pnl),
                    avg_win=self._paisa_to_rupees(avg_win),
                    avg_loss=self._paisa_to_rupees(avg_loss),
                    profit_factor=profit_factor,
                )
            )

        return sorted(result, key=lambda x: x.total_pnl, reverse=True)

    def get_daily_pnl_summary(self, days: int = 30) -> pd.DataFrame:
        """Get daily P&L summary for the last N days.

        Args:
            days: Number of days to fetch

        Returns:
            DataFrame with daily P&L data.
        """
        end_date = datetime.now(IST).date()
        start_date = end_date - timedelta(days=days)

        records = (
            self.session.query(DailyPnL)
            .filter(DailyPnL.date >= start_date)
            .filter(DailyPnL.date <= end_date)
            .order_by(DailyPnL.date.asc())
            .all()
        )

        if not records:
            return pd.DataFrame()

        data = {
            "date": [r.date for r in records],
            "realized_pnl": [self._paisa_to_rupees(r.realized_pnl) for r in records],
            "unrealized_pnl": [self._paisa_to_rupees(r.unrealized_pnl) for r in records],
            "total_pnl": [self._paisa_to_rupees(r.total_pnl) for r in records],
            "trades_count": [r.trades_count for r in records],
            "win_count": [r.win_count for r in records],
        }

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def get_risk_metrics(
        self,
        max_positions: int = 10,
        max_capital_pct: int = 95,
        daily_loss_limit: float = 5000.0,
        total_capital: float = 100000.0,
    ) -> RiskMetrics:
        """Get current risk metrics.

        Args:
            max_positions: Maximum allowed open positions
            max_capital_pct: Maximum capital deployment percentage
            daily_loss_limit: Daily loss limit in rupees
            total_capital: Total trading capital in rupees

        Returns:
            RiskMetrics object.
        """
        # Count open positions
        open_count = (
            self.session.query(PositionRecord)
            .filter(PositionRecord.status == "open")
            .count()
        )

        # Calculate deployed capital
        positions = (
            self.session.query(PositionRecord)
            .filter(PositionRecord.status == "open")
            .all()
        )

        deployed_paisa = sum(p.entry_price * p.quantity for p in positions)
        deployed_capital = self._paisa_to_rupees(deployed_paisa)
        max_capital = total_capital * (max_capital_pct / 100)

        # Get today's P&L
        today = datetime.now(IST).date()
        today_pnl = (
            self.session.query(DailyPnL).filter(DailyPnL.date == today).first()
        )

        daily_pnl = (
            self._paisa_to_rupees(today_pnl.total_pnl) if today_pnl else 0.0
        )

        return RiskMetrics(
            open_positions=open_count,
            max_positions=max_positions,
            position_usage_pct=(open_count / max_positions * 100)
            if max_positions > 0
            else 0.0,
            deployed_capital=deployed_capital,
            max_capital=max_capital,
            capital_usage_pct=(deployed_capital / max_capital * 100)
            if max_capital > 0
            else 0.0,
            daily_pnl=daily_pnl,
            daily_loss_limit=daily_loss_limit,
            daily_loss_usage_pct=(abs(daily_pnl) / daily_loss_limit * 100)
            if daily_pnl < 0
            else 0.0,
        )

    def get_total_pnl(self) -> float:
        """Get total realized P&L across all trades.

        Returns:
            Total P&L in rupees.
        """
        result = self.session.query(func.sum(TradeRecord.realized_pnl)).scalar()
        return self._paisa_to_rupees(result or 0)

    def get_today_pnl(self) -> float:
        """Get today's P&L.

        Returns:
            Today's P&L in rupees.
        """
        today = datetime.now(IST).date()
        today_pnl = (
            self.session.query(DailyPnL).filter(DailyPnL.date == today).first()
        )
        return self._paisa_to_rupees(today_pnl.total_pnl) if today_pnl else 0.0
