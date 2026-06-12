"""Regression tests for live/paper execution pipeline blockers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import create_engine, inspect, text

from src.execution.base_broker import OrderSide, OrderType, ProductType
from src.execution.order_tracker import OrderTracker
from src.execution.paper_broker import PaperBroker
from src.execution.position_manager import PositionManager
from src.main import TradingBot
from src.persistence.database import _run_sqlite_compat_migrations
from src.risk.risk_manager import RiskManager
from scripts.replay_upstox_paper_24mo import _charge_total


def test_order_tracker_normalizes_portfolio_update_and_emits_fill(tmp_path: Path) -> None:
    """A COMPLETE websocket dict should reach the fill callback once."""
    fills = []
    tracker = OrderTracker(
        broker=MagicMock(),
        db_url=f"sqlite:///{tmp_path / 'orders.db'}",
        on_order_fill=fills.append,
    )

    tracker.on_order_update(
        {
            "data": {
                "order_id": "OID-1",
                "status": "complete",
                "filled_quantity": 15,
                "quantity": 15,
                "average_price": 200.5,
            }
        }
    )

    assert len(fills) == 1
    assert fills[0].order_id == "OID-1"
    assert fills[0].status == "COMPLETE"
    assert fills[0].average_price == 20_050


def test_unrealized_daily_loss_breach_invokes_flatten_callback() -> None:
    """Mark-to-market loss should trip the flatten path, not only block entries."""
    risk = RiskManager(
        capital=1_000_000,
        max_daily_loss=50_000,
        max_open_positions=5,
        max_position_size_pct=20,
    )
    flatten = MagicMock()
    risk.set_daily_loss_breach_callback(flatten)

    risk.update_pnl(realized=0, unrealized=-50_000)

    flatten.assert_called_once()


def test_sqlite_daily_pnl_migration_rebuilds_legacy_unique_constraint(tmp_path: Path) -> None:
    """Legacy daily_pnl(date) tables must support per-strategy rows after startup."""
    db_path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE daily_pnl ("
                "id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, "
                "date DATE NOT NULL UNIQUE, "
                "realized_pnl BIGINT NOT NULL, "
                "unrealized_pnl BIGINT NOT NULL, "
                "total_pnl BIGINT NOT NULL, "
                "trades_count INTEGER NOT NULL, "
                "win_count INTEGER NOT NULL"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO daily_pnl "
                "(date, realized_pnl, unrealized_pnl, total_pnl, trades_count, win_count) "
                "VALUES ('2026-06-01', 100, 20, 120, 2, 1)"
            )
        )

    _run_sqlite_compat_migrations(engine)

    columns = {col["name"] for col in inspect(engine).get_columns("daily_pnl")}
    assert "strategy" in columns
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO daily_pnl "
                "(date, strategy, realized_pnl, unrealized_pnl, total_pnl, trades_count, win_count) "
                "VALUES ('2026-06-01', 'ema', 1, 2, 3, 1, 1)"
            )
        )
        rows = conn.execute(text("SELECT date, strategy, total_pnl FROM daily_pnl")).all()

    assert ("2026-06-01", "__total__", 120) in rows
    assert ("2026-06-01", "ema", 3) in rows


def test_position_manager_can_disable_generated_price_exits() -> None:
    """Combo legs that rely on paired emergency exits should not get solo targets."""
    manager = PositionManager(broker=MagicMock())

    position = manager.add_position(
        instrument_key="NSE_FO|BANKNIFTY26JUN50000CE",
        strategy_id="straddle_proxy",
        side=OrderSide.SELL,
        quantity=35,
        entry_price=20_000,
        product_type=ProductType.MIS,
        enable_price_exits=False,
    )

    assert position.stop_loss_price is None
    assert position.target_price is None
    assert position.price_exits_enabled is False


def test_position_lookup_handles_multiple_entries_per_instrument() -> None:
    """Instrument lookup should not crash when index stores multiple position IDs."""
    manager = PositionManager(broker=MagicMock())
    first = manager.add_position(
        instrument_key="NSE_EQ|INE002A01018",
        strategy_id="ema",
        side=OrderSide.BUY,
        quantity=1,
        entry_price=100_00,
    )
    second = manager.add_position(
        instrument_key="NSE_EQ|INE009A01021",
        strategy_id="rsi",
        side=OrderSide.BUY,
        quantity=1,
        entry_price=101_00,
    )
    manager._instrument_to_position["NSE_EQ|INE002A01018"] = [
        first.position_id,
        second.position_id,
    ]

    assert manager.get_position_by_instrument("NSE_EQ|INE002A01018") == first


def test_tick_updates_paper_quote_before_exit_order_uses_it() -> None:
    """Exit orders triggered by a tick must fill at that tick's bid/ask."""
    broker = PaperBroker(initial_capital=1_000_000_00, slippage_pct=0.0005)
    manager = PositionManager(broker=broker)
    bot = TradingBot()
    bot.position_manager = manager
    bot.paper_broker = broker
    bot.depth_feed = None
    bot.tick_metrics = None
    bot.vix_regime = None
    bot.micro_feature_engine = None

    instrument = "NSE_EQ|INE002A01018"
    broker.update_quote(instrument, 2500_00, 2499_95, 2500_05)
    manager.add_position(
        instrument_key=instrument,
        strategy_id="smoke",
        side=OrderSide.BUY,
        quantity=10,
        entry_price=2500_05,
        product_type=ProductType.MIS,
        stop_loss_price=2450_00,
        target_price=2510_00,
    )

    bot._on_market_tick(
        {
            "feeds": {
                instrument: {
                    "fullFeed": {
                        "marketFF": {
                            "ltpc": {"ltp": 2511.0, "v": 1000},
                            "depth": {
                                "buyDepth": [{"price": 2510.95}],
                                "sellDepth": [{"price": 2511.05}],
                            },
                        }
                    }
                }
            }
        }
    )

    orders = broker.get_order_book()
    assert len(orders) == 1
    assert orders[0].side == "SELL"
    assert orders[0].average_price == 2510_95
    assert manager.get_open_positions() == []


def test_paper_broker_stamp_duty_uses_decimal_rates() -> None:
    """Stamp duty constants are decimals, so no extra divide-by-100 is applied."""
    broker = PaperBroker()

    charges = broker._calculate_charges(
        instrument_key="NSE_FO|BANKNIFTY26JUN50000PE",
        side=OrderSide.BUY,
        product_type=ProductType.MIS,
        quantity=35,
        price=20_000,
    )

    assert charges["stamp_duty"] == int(35 * 20_000 * 0.00002)
    assert charges["stamp_duty"] == 14


def test_replay_report_handles_paper_broker_charge_dict() -> None:
    """PaperBroker trade history stores detailed charge dicts, not scalars."""
    assert _charge_total({"charges": {"brokerage": 2, "total": 123}}) == 123
    assert _charge_total({"charges": 45}) == 45
    assert _charge_total({"fees": 67}) == 67
