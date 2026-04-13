"""SQLAlchemy ORM models for the trading bot persistence layer.

All monetary values are stored in **paisa** (1/100 INR) as integers
to avoid floating-point precision issues.
"""

from datetime import date, datetime, timezone

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class OrderRecord(Base):
    """Persisted representation of a trading order (paper or live)."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(128), nullable=False)
    instrument_key = Column(String(64), nullable=False)
    transaction_type = Column(String(8), nullable=False)
    order_type = Column(String(16), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    trigger_price = Column(Integer, nullable=True)
    status = Column(String(24), nullable=False, default="PENDING")
    filled_qty = Column(Integer, nullable=False, default=0)
    avg_fill_price = Column(Integer, nullable=False, default=0)
    placed_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)
    tag = Column(String(64), nullable=True)
    broker_order_id = Column(String(64), nullable=True)


class TradeRecord(Base):
    """A completed round-trip trade (entry + exit)."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(128), nullable=False)
    instrument_key = Column(String(64), nullable=False)
    side = Column(String(8), nullable=False)
    entry_price = Column(Integer, nullable=False)
    exit_price = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    realized_pnl = Column(Integer, nullable=False)
    fees = Column(Integer, nullable=False)
    entry_time = Column(DateTime(timezone=True), nullable=False)
    exit_time = Column(DateTime(timezone=True), nullable=False)
    holding_duration_seconds = Column(Integer, nullable=False)


class PositionRecord(Base):
    """An open or closed position tracked by the bot."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(128), nullable=False)
    instrument_key = Column(String(64), nullable=False)
    side = Column(String(8), nullable=False)
    entry_price = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False, default="open")
    opened_at = Column(DateTime(timezone=True), nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)


class DailyPnL(Base):
    """Aggregated profit-and-loss summary for a single trading day."""

    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, nullable=False)
    realized_pnl = Column(Integer, nullable=False)
    unrealized_pnl = Column(Integer, nullable=False)
    total_pnl = Column(Integer, nullable=False)
    trades_count = Column(Integer, nullable=False)
    win_count = Column(Integer, nullable=False)


class StrategyState(Base):
    """Serialised snapshot of a strategy's internal state."""

    __tablename__ = "strategy_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(128), nullable=False)
    set_name = Column(String(128), nullable=False)
    last_evaluated_bar = Column(DateTime(timezone=True), nullable=True)
    state_json = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (UniqueConstraint("strategy_name", "set_name", name="uq_strategy_set"),)


class ResearchReport(Base):
    """Persisted trade research analysis report."""

    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    instrument_key = Column(String(64), nullable=False)
    direction = Column(String(8), nullable=False)
    strategy_name = Column(String(128), nullable=False)
    final_score = Column(Float, nullable=False)
    verdict = Column(String(16), nullable=False)
    signal_strength = Column(String(16), nullable=False)
    market_regime = Column(String(24), nullable=False)
    volatility_regime = Column(String(16), nullable=False)
    components_json = Column(Text, nullable=True)
    trend_data_json = Column(Text, nullable=True)
    momentum_data_json = Column(Text, nullable=True)
    sr_data_json = Column(Text, nullable=True)
    volume_data_json = Column(Text, nullable=True)
    options_data_json = Column(Text, nullable=True)
    suggested_quantity_multiplier = Column(Float, default=1.0, nullable=False)
    suggested_sl_adjustment = Column(Float, default=1.0, nullable=False)
    suggested_target_adjustment = Column(Float, default=1.0, nullable=False)
    optimal_entry_type = Column(String(24), default="MARKET", nullable=False)
    key_risks_json = Column(Text, nullable=True)
    key_supports_json = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    was_executed = Column(Integer, nullable=True)  # Boolean as int
    execution_result_json = Column(Text, nullable=True)
    execution_timestamp = Column(DateTime(timezone=True), nullable=True)


class AnalyzerPerformance(Base):
    """Tracks analyzer predictions vs actual trade outcomes."""

    __tablename__ = "analyzer_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(64), index=True, nullable=False)
    analyzer_name = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    prediction_score = Column(Float, nullable=False)
    prediction_contribution = Column(Float, nullable=False)
    market_regime = Column(String(24), nullable=False)
    volatility_regime = Column(String(16), nullable=False)
    actual_pnl = Column(Integer, nullable=True)
    trade_result = Column(String(16), nullable=True)
    trade_duration_seconds = Column(Integer, nullable=True)
    was_executed = Column(Integer, default=0, nullable=False)  # Boolean as int
    outcome_recorded_at = Column(DateTime(timezone=True), nullable=True)


class OptimizedWeights(Base):
    """Stores optimized analyzer weights by market regime."""

    __tablename__ = "optimized_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    regime = Column(String(32), index=True, nullable=False, default="global")
    weights_json = Column(Text, nullable=False)
    sample_count = Column(Integer, default=0, nullable=False)
    accuracy_by_analyzer_json = Column(Text, nullable=True)
    correlation_by_analyzer_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (UniqueConstraint("regime", name="uq_regime_weights"),)


class LessonLearnedRecord(Base):
    """Persisted lessons learned from trade analysis.

    Stores insights extracted from winning and losing trades
    for continuous strategy improvement.
    """

    __tablename__ = "lessons_learned"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lesson_id = Column(String(64), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    strategy_name = Column(String(128), nullable=False, index=True)
    category = Column(String(32), nullable=False)  # strategy, timing, market_condition, risk
    description = Column(Text, nullable=False)
    trigger_pattern_json = Column(Text, nullable=True)  # JSON of trigger conditions
    impact_paisa = Column(Integer, nullable=False, default=0)  # P&L impact
    confidence = Column(Float, nullable=False, default=0.5)  # 0-1
    applied = Column(Integer, default=0, nullable=False)  # Boolean as int
    applied_at = Column(DateTime(timezone=True), nullable=True)


class StrategyPerformanceRecord(Base):
    """Historical performance metrics for strategies.

    Tracks strategy performance over time for trend analysis
    and optimization decisions.
    """

    __tablename__ = "strategy_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(128), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    total_trades = Column(Integer, default=0, nullable=False)
    winning_trades = Column(Integer, default=0, nullable=False)
    losing_trades = Column(Integer, default=0, nullable=False)
    total_pnl_paisa = Column(Integer, default=0, nullable=False)
    win_rate = Column(Float, default=0.0, nullable=False)
    profit_factor = Column(Float, default=0.0, nullable=False)
    avg_win_pct = Column(Float, default=0.0, nullable=False)
    avg_loss_pct = Column(Float, default=0.0, nullable=False)
    max_consecutive_losses = Column(Integer, default=0, nullable=False)
    hourly_performance_json = Column(Text, nullable=True)  # JSON of hourly stats
    recommendations_json = Column(Text, nullable=True)  # JSON of recommendations
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (UniqueConstraint("strategy_name", "date", name="uq_strategy_date"),)


class AgentDecisionRecord(Base):
    """Records every AI agent decision for observability."""

    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String(64), nullable=False, index=True)
    instrument_key = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False, default=0.5)
    used_fallback = Column(Integer, default=0, nullable=False)
    latency_ms = Column(Integer, default=0, nullable=False)
    lessons_injected = Column(Integer, default=0, nullable=False)


class MemoryLessonRecord(Base):
    """Persistent storage for memory lessons with decay tracking."""

    __tablename__ = "memory_lessons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lesson_id = Column(String(64), unique=True, nullable=False, index=True)
    category = Column(String(32), nullable=False, index=True)
    strategy = Column(String(128), nullable=False, index=True)
    regime = Column(String(32), nullable=False, index=True)
    description = Column(Text, nullable=False)
    severity = Column(String(16), nullable=False)
    base_score = Column(Float, nullable=False, default=1.0)
    times_injected = Column(Integer, default=0, nullable=False)
    times_useful = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


class MistakeRecord(Base):
    """Records classified mistakes for analysis."""

    __tablename__ = "trade_mistakes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), nullable=False, index=True)
    category = Column(String(32), nullable=False, index=True)
    severity = Column(String(16), nullable=False)
    description = Column(Text, nullable=False)
    lesson = Column(Text, nullable=False)
    regime = Column(String(32), nullable=False)
    strategy = Column(String(128), nullable=False)
    score = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
