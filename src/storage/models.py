"""SQLAlchemy 2.x ORM models matching the Phase 0 migration schema.

All monetary values stored in paisa (BIGINT). Timestamps in UTC with timezone.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Metadata tables
# ---------------------------------------------------------------------------

class BotConfig(Base):
    """Key-value configuration store."""

    __tablename__ = "bot_config"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMPTZ, default=datetime.utcnow
    )


class SchemaVersion(Base):
    """Migration version tracking."""

    __tablename__ = "schema_version"

    version: Mapped[str] = mapped_column(Text, primary_key=True)
    applied_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMPTZ, default=datetime.utcnow
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

class Instrument(Base):
    """Instrument master record from Upstox instrument dump."""

    __tablename__ = "instruments"

    instrument_key: Mapped[str] = mapped_column(Text, primary_key=True)
    segment: Mapped[str] = mapped_column(Text, nullable=False)
    trading_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    underlying_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    underlying_symbol: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expiry: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    strike_price: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    option_type: Mapped[Optional[str]] = mapped_column(String(2), nullable=True)
    instrument_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lot_size: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    tick_size: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    freeze_quantity: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    isin: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    exchange_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    refreshed_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMPTZ, default=datetime.utcnow
    )


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

class Bar(Base):
    """OHLCV bar (candlestick). All prices in paisa."""

    __tablename__ = "bars"

    instrument_key: Mapped[str] = mapped_column(Text, primary_key=True)
    timeframe: Mapped[str] = mapped_column(Text, primary_key=True)
    ts: Mapped[datetime] = mapped_column(TIMESTAMPTZ, primary_key=True)
    open: Mapped[int] = mapped_column(BigInteger, nullable=False)
    high: Mapped[int] = mapped_column(BigInteger, nullable=False)
    low: Mapped[int] = mapped_column(BigInteger, nullable=False)
    close: Mapped[int] = mapped_column(BigInteger, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    oi: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)


class Tick(Base):
    """Individual tick record. All prices in paisa. Hot for 7 days."""

    __tablename__ = "ticks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    instrument_key: Mapped[str] = mapped_column(Text, nullable=False)
    ts: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False)
    ltp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, default=0)
    bid: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    ask: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    bid_qty: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ask_qty: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_buy_qty: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    total_sell_qty: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)


class DepthSnapshot(Base):
    """Order book depth snapshot. Hot for 7 days."""

    __tablename__ = "depth_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    instrument_key: Mapped[str] = mapped_column(Text, nullable=False)
    ts: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False)
    bids: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    asks: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    spread_bps: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    bid_wall_top5: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    ask_wall_top5: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)


# ---------------------------------------------------------------------------
# Orders / Fills / Positions / P&L
# ---------------------------------------------------------------------------

class Order(Base):
    """Order record. Prices in paisa."""

    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(Text, primary_key=True)
    client_order_id: Mapped[Optional[str]] = mapped_column(Text, unique=True, nullable=True)
    strategy_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    instrument_key: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    order_type: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    trigger_price: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    product: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    validity: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    filled_qty: Mapped[int] = mapped_column(Integer, default=0)
    avg_fill_price: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    placed_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMPTZ, default=datetime.utcnow
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMPTZ, default=datetime.utcnow
    )
    raw_response: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    fills: Mapped[list["Fill"]] = relationship("Fill", back_populates="order")


class Fill(Base):
    """Trade fill record. Prices in paisa."""

    __tablename__ = "fills"

    fill_id: Mapped[str] = mapped_column(Text, primary_key=True)
    order_id: Mapped[Optional[str]] = mapped_column(
        Text, ForeignKey("orders.order_id"), nullable=True
    )
    instrument_key: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fee_paisa: Mapped[int] = mapped_column(BigInteger, default=0)
    filled_at: Mapped[datetime] = mapped_column(TIMESTAMPTZ, nullable=False)

    order: Mapped[Optional["Order"]] = relationship("Order", back_populates="fills")


class Position(Base):
    """Position record. Prices and P&L in paisa."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strategy_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    instrument_key: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    entry_qty: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_avg_price: Mapped[int] = mapped_column(BigInteger, nullable=False)
    exit_qty: Mapped[int] = mapped_column(Integer, default=0)
    exit_avg_price: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    realized_pnl: Mapped[int] = mapped_column(BigInteger, default=0)
    unrealized_pnl: Mapped[int] = mapped_column(BigInteger, default=0)
    opened_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMPTZ, default=datetime.utcnow
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMPTZ, nullable=True)
    status: Mapped[str] = mapped_column(String(10), nullable=False)


class DailyPnl(Base):
    """Daily aggregated P&L per strategy. Values in paisa."""

    __tablename__ = "daily_pnl"

    __table_args__ = (UniqueConstraint("trade_date", "strategy_name"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    strategy_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    realized_pnl: Mapped[int] = mapped_column(BigInteger, default=0)
    unrealized_pnl: Mapped[int] = mapped_column(BigInteger, default=0)
    fees_paid: Mapped[int] = mapped_column(BigInteger, default=0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    win_count: Mapped[int] = mapped_column(Integer, default=0)
    loss_count: Mapped[int] = mapped_column(Integer, default=0)


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class Signal(Base):
    """Strategy signal record."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(Text, nullable=False)
    instrument_key: Mapped[str] = mapped_column(Text, nullable=False)
    signal_type: Mapped[str] = mapped_column(Text, nullable=False)
    ts: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMPTZ, default=datetime.utcnow
    )
    conditions_met: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    conditions_failed: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    confluence_score: Mapped[Optional[Decimal]] = mapped_column(Numeric, nullable=True)
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    acted_on: Mapped[bool] = mapped_column(Boolean, default=False)
    order_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Bot runs
# ---------------------------------------------------------------------------

class BotRun(Base):
    """Bot lifecycle run record."""

    __tablename__ = "bot_runs"

    run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMPTZ, default=datetime.utcnow
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMPTZ, nullable=True)
    branch: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    git_sha: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mode: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    capital_paisa: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
