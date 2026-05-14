"""High-level persistence helper for logging orders, trades, and strategy state.

``TradeLogger`` wraps raw SQLAlchemy operations behind a clean interface
that the rest of the bot can call without touching the ORM directly.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Callable, Generator

from loguru import logger
from sqlalchemy.orm import Session

from src.persistence.models import (
    DailyPnL,
    OrderRecord,
    PositionRecord,
    StrategyState,
    TradeRecord,
)


class TradeLogger:
    """Facade for persisting trading artefacts to the database.

    Args:
        session_factory: A callable (typically ``database.get_session``)
            that returns a context manager yielding a ``Session``.
    """

    def __init__(
        self,
        session_factory: Callable[[], contextmanager[Generator[Session, None, None]]],
    ) -> None:
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def log_order(self, order_data: dict) -> OrderRecord:
        """Persist a new order record.

        Args:
            order_data: Dictionary whose keys match ``OrderRecord`` column
                names.  At minimum ``strategy``, ``instrument_key``,
                ``transaction_type``, ``order_type``, ``quantity``, and
                ``price`` must be present.

        Returns:
            The newly created ``OrderRecord`` with its ``id`` populated.
        """
        with self._session_factory() as session:
            record = OrderRecord(**order_data)
            session.add(record)
            session.flush()  # populate record.id before commit
            logger.info(
                "ORDER | Logged order id={} {} {} {} qty={} price={}p",
                record.id,
                record.strategy,
                record.transaction_type,
                record.instrument_key,
                record.quantity,
                record.price,
            )
            return record

    def update_order_status(
        self,
        order_id: int,
        status: str,
        filled_qty: int = 0,
        avg_price: int = 0,
    ) -> None:
        """Update the status of an existing order.

        Args:
            order_id: Primary key of the ``OrderRecord`` to update.
            status: New status string (e.g. ``FILLED``, ``CANCELLED``).
            filled_qty: Cumulative filled quantity.
            avg_price: Volume-weighted average fill price in paisa.

        Raises:
            ValueError: If no order with *order_id* exists.
        """
        with self._session_factory() as session:
            record = session.get(OrderRecord, order_id)
            if record is None:
                raise ValueError(f"OrderRecord with id={order_id} not found.")
            record.status = status
            record.filled_qty = filled_qty
            record.avg_fill_price = avg_price
            record.updated_at = datetime.now(timezone.utc)
            logger.info(
                "ORDER | Updated order id={} status={} filled_qty={} avg_price={}p",
                order_id,
                status,
                filled_qty,
                avg_price,
            )

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def log_trade(self, trade_data: dict) -> TradeRecord:
        """Persist a completed round-trip trade.

        Args:
            trade_data: Dictionary whose keys match ``TradeRecord`` column
                names.

        Returns:
            The newly created ``TradeRecord``.
        """
        with self._session_factory() as session:
            record = TradeRecord(**trade_data)
            session.add(record)
            session.flush()
            logger.info(
                "TRADE | id={} {} {} {} pnl={}p fees={}p",
                record.id,
                record.strategy,
                record.side,
                record.instrument_key,
                record.realized_pnl,
                record.fees,
            )
            return record

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[PositionRecord]:
        """Return all positions whose status is ``open``.

        Returns:
            List of ``PositionRecord`` instances (may be empty).
        """
        with self._session_factory() as session:
            positions = (
                session.query(PositionRecord)
                .filter(PositionRecord.status == "open")
                .all()
            )
            # Expunge so the caller can use them outside the session.
            for pos in positions:
                session.expunge(pos)
            return positions

    def log_position(self, position_data: dict) -> PositionRecord:
        """Persist a new open position record.

        Args:
            position_data: Dictionary whose keys match ``PositionRecord`` columns.
        """
        with self._session_factory() as session:
            record = PositionRecord(**position_data)
            session.add(record)
            session.flush()
            logger.info(
                "POSITION | Logged new open position id={} {} {} {} qty={} price={}p",
                record.id,
                record.strategy,
                record.side,
                record.instrument_key,
                record.quantity,
                record.entry_price,
            )
            return record

    def close_position(self, instrument_key: str, status: str = "closed") -> None:
        """Mark a position as closed.

        Args:
            instrument_key: The instrument key of the position.
            status: New status, defaults to "closed".
        """
        from zoneinfo import ZoneInfo
        # opened_at is recorded in IST-naive (matches PositionManager._now()).
        # Use the same convention for closed_at so timeline ordering is sane.
        ist_now = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
        with self._session_factory() as session:
            record = (
                session.query(PositionRecord)
                .filter(PositionRecord.instrument_key == instrument_key, PositionRecord.status == "open")
                .first()
            )
            if record:
                record.status = status
                record.closed_at = ist_now
                logger.info("POSITION | Marked position closed for {}", instrument_key)

    # ------------------------------------------------------------------
    # Daily P&L
    # ------------------------------------------------------------------

    def log_daily_summary(
        self,
        date: date,
        realized: int,
        unrealized: int,
        trades: int,
        wins: int,
    ) -> None:
        """Insert or update the daily P&L summary for *date*.

        Args:
            date: Calendar date for the summary.
            realized: Realised P&L in paisa.
            unrealized: Unrealised (mark-to-market) P&L in paisa.
            trades: Total number of completed trades.
            wins: Number of winning trades.
        """
        total = realized + unrealized
        with self._session_factory() as session:
            existing: DailyPnL | None = (
                session.query(DailyPnL).filter(DailyPnL.date == date).first()
            )
            if existing is not None:
                existing.realized_pnl = realized
                existing.unrealized_pnl = unrealized
                existing.total_pnl = total
                existing.trades_count = trades
                existing.win_count = wins
                logger.debug("Updated DailyPnL for {}.", date)
            else:
                record = DailyPnL(
                    date=date,
                    realized_pnl=realized,
                    unrealized_pnl=unrealized,
                    total_pnl=total,
                    trades_count=trades,
                    win_count=wins,
                )
                session.add(record)
                logger.debug("Inserted DailyPnL for {}.", date)

    # ------------------------------------------------------------------
    # Strategy State
    # ------------------------------------------------------------------

    def save_strategy_state(
        self,
        strategy_name: str,
        set_name: str,
        state: dict,
    ) -> None:
        """Persist (upsert) the serialised state for a strategy/set pair.

        Args:
            strategy_name: Strategy identifier.
            set_name: Logical partition key (e.g. instrument key).
            state: Arbitrary dictionary that will be JSON-serialised.
        """
        with self._session_factory() as session:
            existing: StrategyState | None = (
                session.query(StrategyState)
                .filter(
                    StrategyState.strategy_name == strategy_name,
                    StrategyState.set_name == set_name,
                )
                .first()
            )
            serialised = json.dumps(state, default=str)
            now = datetime.now(timezone.utc)

            if existing is not None:
                existing.state_json = serialised
                existing.updated_at = now
            else:
                session.add(
                    StrategyState(
                        strategy_name=strategy_name,
                        set_name=set_name,
                        state_json=serialised,
                        updated_at=now,
                    )
                )
            logger.debug(
                "Saved strategy state for {}/{}.", strategy_name, set_name
            )

    def load_strategy_state(
        self,
        strategy_name: str,
        set_name: str,
    ) -> dict | None:
        """Load previously saved strategy state.

        Args:
            strategy_name: Strategy identifier.
            set_name: Logical partition key.

        Returns:
            The deserialised dictionary, or ``None`` if no state exists.
        """
        with self._session_factory() as session:
            record: StrategyState | None = (
                session.query(StrategyState)
                .filter(
                    StrategyState.strategy_name == strategy_name,
                    StrategyState.set_name == set_name,
                )
                .first()
            )
            if record is None or record.state_json is None:
                return None
            return json.loads(record.state_json)
