"""Example usage of the self-learning trading system.

This script demonstrates how to use the learning system to:
1. Process trades and extract lessons
2. Generate recommendations
3. Adjust position sizes based on performance
4. Create reports
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.learning import (
    LearningEngine,
    LearningIntegration,
    LessonLearned,
    create_learning_report,
)
from src.persistence.models import Base, TradeRecord


def create_sample_trades(db_session):
    """Create sample trade records for demonstration."""
    trades = []

    # Winning trades - EMA Crossover
    for i in range(5):
        trade = TradeRecord(
            strategy="ema_crossover",
            instrument_key="NSE_EQ:RELIANCE",
            side="BUY",
            entry_price=250000,  # 2500.00
            exit_price=255000,   # 2550.00
            quantity=10,
            realized_pnl=50000,  # 500.00 profit
            fees=500,
            entry_time=datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(days=i, hours=2),
            exit_time=datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(days=i),
            holding_duration_seconds=7200,
        )
        trades.append(trade)

    # Losing trades - RSI Reversal (during lunch hours)
    for i in range(4):
        trade = TradeRecord(
            strategy="rsi_reversal",
            instrument_key="NSE_EQ:TCS",
            side="BUY",
            entry_price=350000,
            exit_price=348000,
            quantity=5,
            realized_pnl=-10000,  # -100.00 loss
            fees=500,
            entry_time=datetime.now(ZoneInfo("Asia/Kolkata")).replace(hour=12, minute=0) - timedelta(days=i),
            exit_time=datetime.now(ZoneInfo("Asia/Kolkata")).replace(hour=12, minute=30) - timedelta(days=i),
            holding_duration_seconds=1800,
        )
        trades.append(trade)

    # Quick exit trades - suggesting tight SL
    for i in range(3):
        trade = TradeRecord(
            strategy="vwap_breakout",
            instrument_key="NSE_EQ:INFY",
            side="BUY",
            entry_price=150000,
            exit_price=149800,
            quantity=20,
            realized_pnl=-4000,  # Small loss
            fees=400,
            entry_time=datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(days=i, hours=1),
            exit_time=datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(days=i, minutes=55),
            holding_duration_seconds=300,  # 5 minutes - very quick
        )
        trades.append(trade)

    for trade in trades:
        db_session.add(trade)

    db_session.commit()
    print(f"Created {len(trades)} sample trades")


def main():
    """Run the learning system example."""
    # Setup database
    engine = create_engine("sqlite:///example_learning.db")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # Create sample trades
        create_sample_trades(db)

        # Initialize learning engine
        print("\n" + "=" * 70)
        print("INITIALIZING LEARNING ENGINE")
        print("=" * 70)

        engine_learn = LearningEngine(db)

        # Process all trades
        trades = db.query(TradeRecord).all()
        print(f"\nProcessing {len(trades)} trades...")

        for trade in trades:
            insights = engine_learn.process_trade(trade)
            print(
                f"  {trade.strategy}: P&L={insights['pnl_rupees']:.2f}, "
                f"Win={insights['was_win']}, Hour={insights['hour_of_entry']}"
            )

        # Generate recommendations
        print("\n" + "=" * 70)
        print("GENERATING RECOMMENDATIONS")
        print("=" * 70)

        recommendations = engine_learn.generate_recommendations()
        for rec in recommendations:
            print(f"\n[{rec['priority'].upper()}] {rec['strategy']}")
            print(f"  Issue: {rec['issue']}")
            print(f"  Action: {rec['action']}")

        # Test position sizing adjustments
        print("\n" + "=" * 70)
        print("POSITION SIZE ADJUSTMENTS")
        print("=" * 70)

        strategies = ["ema_crossover", "rsi_reversal", "vwap_breakout"]
        base_size = 100

        for strategy in strategies:
            adjusted = engine_learn.adjust_position_size(strategy, base_size)
            perf = engine_learn._analyzer.get_strategy_performance(strategy)
            if perf:
                print(
                    f"{strategy}: {base_size} -> {adjusted} "
                    f"(Win Rate: {perf.win_rate:.1%}, PF: {perf.profit_factor:.2f})"
                )

        # Generate full report
        print("\n" + "=" * 70)
        print("FULL LEARNING REPORT")
        print("=" * 70)

        report = engine_learn.get_lessons_report()
        print(report)

        # Export lessons
        print("\n" + "=" * 70)
        print("EXPORTING LESSONS")
        print("=" * 70)

        export_path = "example_lessons.json"
        engine_learn.export_lessons(export_path)
        print(f"Lessons exported to: {export_path}")

        # Demonstrate integration layer
        print("\n" + "=" * 70)
        print("INTEGRATION LAYER DEMO")
        print("=" * 70)

        integration = LearningIntegration(db)

        # Check if strategies should trade
        for strategy in strategies:
            should_trade, reason = integration.should_trade_strategy(strategy)
            status = "✓ TRADE" if should_trade else "✗ BLOCK"
            print(f"{strategy}: {status} - {reason}")

        # Get best trading hours
        print("\nBest Trading Hours Analysis:")
        for strategy in strategies:
            best_hours = integration.get_best_trading_hours(strategy)
            if best_hours:
                print(f"  {strategy}: {best_hours[:3]}")  # Top 3 hours

    finally:
        db.close()

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
