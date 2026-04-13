"""Unit tests for Kelly Criterion position sizer."""

import pytest

from src.risk.position_sizer import (
    KellyPositionSizer,
    KellyStats,
    TradeResult,
    create_kelly_sizer_from_trades,
)


class TestTradeResult:
    """Tests for TradeResult dataclass."""

    def test_trade_result_win(self):
        """Test winning trade detection."""
        trade = TradeResult(pnl=1000, entry_price=10000, exit_price=10100, quantity=100)
        assert trade.is_win is True
        assert trade.is_loss is False
        assert trade.absolute_pnl == 1000

    def test_trade_result_loss(self):
        """Test losing trade detection."""
        trade = TradeResult(pnl=-500, entry_price=10000, exit_price=9950, quantity=100)
        assert trade.is_win is False
        assert trade.is_loss is True
        assert trade.absolute_pnl == 500

    def test_trade_result_breakeven(self):
        """Test breakeven trade."""
        trade = TradeResult(pnl=0, entry_price=10000, exit_price=10000, quantity=100)
        assert trade.is_win is False
        assert trade.is_loss is False
        assert trade.absolute_pnl == 0


class TestKellyPositionSizerBasics:
    """Tests for basic KellyPositionSizer functionality."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        sizer = KellyPositionSizer()
        assert sizer.half_kelly is True
        assert sizer.max_kelly_pct == 0.25
        assert sizer.min_trades_for_kelly == 20
        assert sizer.fallback_pct == 0.02

    def test_initialization_custom(self):
        """Test custom initialization."""
        sizer = KellyPositionSizer(
            half_kelly=False,
            max_kelly_pct=0.50,
            min_trades_for_kelly=30,
            fallback_pct=0.05,
        )
        assert sizer.half_kelly is False
        assert sizer.max_kelly_pct == 0.50
        assert sizer.min_trades_for_kelly == 30
        assert sizer.fallback_pct == 0.05

    def test_max_kelly_minimum(self):
        """Test that max_kelly_pct has a minimum of 1%."""
        sizer = KellyPositionSizer(max_kelly_pct=0.005)  # 0.5%
        assert sizer.max_kelly_pct == 0.01  # Capped at 1%

    def test_min_trades_minimum(self):
        """Test that min_trades has a minimum of 5."""
        sizer = KellyPositionSizer(min_trades_for_kelly=2)
        assert sizer.min_trades_for_kelly == 5  # Capped at 5


class TestKellyCalculation:
    """Tests for Kelly Criterion calculation."""

    def test_insufficient_trades(self):
        """Test Kelly returns 0 with insufficient trades."""
        sizer = KellyPositionSizer(min_trades_for_kelly=20)

        # Add only 10 trades
        for i in range(10):
            sizer.add_trade(TradeResult(pnl=1000, entry_price=10000, exit_price=10100, quantity=100))

        kelly = sizer.calculate_kelly_fraction()
        assert kelly == 0.0
        assert sizer.has_sufficient_history() is False

    def test_no_losses(self):
        """Test Kelly returns 0 when no losses (edge case)."""
        sizer = KellyPositionSizer()

        # Add 20 winning trades
        for i in range(20):
            sizer.add_trade(TradeResult(pnl=1000, entry_price=10000, exit_price=10100, quantity=100))

        kelly = sizer.calculate_kelly_fraction()
        assert kelly == 0.0  # Cannot calculate without losses

    def test_no_wins(self):
        """Test Kelly returns 0 when no wins (edge case)."""
        sizer = KellyPositionSizer()

        # Add 20 losing trades
        for i in range(20):
            sizer.add_trade(TradeResult(pnl=-1000, entry_price=10000, exit_price=9900, quantity=100))

        kelly = sizer.calculate_kelly_fraction()
        assert kelly == 0.0  # Cannot calculate without wins

    def test_kelly_calculation_example(self):
        """Test Kelly calculation with known example.

        Win rate 60%, avg win 200, avg loss 100
        b = 200/100 = 2
        Kelly = (0.6*2 - 0.4) / 2 = 0.4 (40%)
        """
        sizer = KellyPositionSizer()

        # 12 wins of +200 paisa each (60% win rate)
        for _ in range(12):
            sizer.add_trade(TradeResult(pnl=200, entry_price=10000, exit_price=10200, quantity=100))

        # 8 losses of -100 paisa each (40% loss rate)
        for _ in range(8):
            sizer.add_trade(TradeResult(pnl=-100, entry_price=10000, exit_price=9900, quantity=100))

        kelly = sizer.calculate_kelly_fraction()
        assert kelly == pytest.approx(0.4, abs=0.01)  # 40% Kelly

    def test_kelly_negative_edge(self):
        """Test Kelly returns negative when no edge."""
        sizer = KellyPositionSizer()

        # 40% win rate, 60% loss rate
        # Avg win 100, avg loss 200
        # b = 0.5
        # Kelly = (0.4*0.5 - 0.6) / 0.5 = -0.8 (negative, don't trade)

        for _ in range(8):  # 40% wins
            sizer.add_trade(TradeResult(pnl=100, entry_price=10000, exit_price=10100, quantity=100))

        for _ in range(12):  # 60% losses
            sizer.add_trade(TradeResult(pnl=-200, entry_price=10000, exit_price=9800, quantity=100))

        kelly = sizer.calculate_kelly_fraction()
        assert kelly < 0  # Negative edge


class TestSafeKelly:
    """Tests for safe Kelly constraints."""

    def test_negative_kelly_returns_zero(self):
        """Test that negative Kelly returns 0 (no trade)."""
        sizer = KellyPositionSizer()
        safe = sizer.get_safe_kelly(-0.2)
        assert safe == 0.0

    def test_half_kelly_applied(self):
        """Test Half-Kelly safety is applied."""
        sizer = KellyPositionSizer(half_kelly=True)
        safe = sizer.get_safe_kelly(0.40)
        assert safe == pytest.approx(0.20, abs=0.001)  # Half of 40%

    def test_full_kelly_when_disabled(self):
        """Test Full Kelly when half_kelly is False."""
        sizer = KellyPositionSizer(half_kelly=False, max_kelly_pct=1.0)
        safe = sizer.get_safe_kelly(0.40)
        assert safe == pytest.approx(0.40, abs=0.001)  # Full 40%

    def test_max_kelly_cap(self):
        """Test maximum Kelly percentage cap."""
        sizer = KellyPositionSizer(half_kelly=False, max_kelly_pct=0.20)
        safe = sizer.get_safe_kelly(0.50)  # 50% Kelly
        assert safe == 0.20  # Capped at 20%

    def test_zero_kelly(self):
        """Test zero Kelly returns zero."""
        sizer = KellyPositionSizer()
        safe = sizer.get_safe_kelly(0.0)
        assert safe == 0.0


class TestPositionSizeCalculation:
    """Tests for position size calculation."""

    def test_basic_position_size(self):
        """Test basic position size calculation."""
        sizer = KellyPositionSizer(half_kelly=True, max_kelly_pct=0.25)

        # Add trades: 60% win rate, 2:1 win/loss ratio
        for _ in range(12):
            sizer.add_trade(TradeResult(pnl=200, entry_price=10000, exit_price=10200, quantity=100))
        for _ in range(8):
            sizer.add_trade(TradeResult(pnl=-100, entry_price=10000, exit_price=9900, quantity=100))

        # Capital: 100,000 INR (10,000,000 paisa)
        # Price: 100 INR (10,000 paisa)
        # Kelly: 40%, Half-Kelly: 20%
        # Position value: 20,000 INR (2,000,000 paisa)
        # Quantity: 2,000,000 / 10,000 = 200

        capital = 10_000_000  # 100,000 INR in paisa
        price = 10_000  # 100 INR in paisa

        quantity = sizer.get_position_size(capital, price)
        assert quantity == 200

    def test_position_size_with_risk_cap(self):
        """Test position size capped by max risk per trade."""
        sizer = KellyPositionSizer(half_kelly=False, max_kelly_pct=1.0)

        # Add trades with high Kelly
        for _ in range(15):
            sizer.add_trade(TradeResult(pnl=500, entry_price=10000, exit_price=10500, quantity=100))
        for _ in range(5):
            sizer.add_trade(TradeResult(pnl=-100, entry_price=10000, exit_price=9900, quantity=100))

        capital = 10_000_000  # 100,000 INR
        price = 10_000  # 100 INR

        # With 1% max risk per trade
        quantity = sizer.get_position_size(capital, price, max_risk_per_trade_pct=0.01)
        position_value = quantity * price

        # Should be capped at 1% of capital = 1,000 INR = 100,000 paisa
        assert position_value <= 100_000

    def test_zero_capital(self):
        """Test position size with zero capital."""
        sizer = KellyPositionSizer()
        quantity = sizer.get_position_size(0, 10000)
        assert quantity == 0

    def test_zero_price(self):
        """Test position size with zero price."""
        sizer = KellyPositionSizer()
        quantity = sizer.get_position_size(10_000_000, 0)
        assert quantity == 0

    def test_fallback_position_size(self):
        """Test fallback sizing when insufficient history."""
        sizer = KellyPositionSizer(fallback_pct=0.02)

        # Only 5 trades (less than default 20)
        for _ in range(5):
            sizer.add_trade(TradeResult(pnl=100, entry_price=10000, exit_price=10100, quantity=100))

        capital = 10_000_000  # 100,000 INR
        price = 10_000  # 100 INR

        quantity = sizer.get_position_size(capital, price)

        # Should use fallback: 2% of capital = 2,000 INR = 200,000 paisa
        # Quantity: 200,000 / 10,000 = 20
        expected_qty = int(capital * 0.02) // price
        assert quantity == expected_qty


class TestKellyStats:
    """Tests for Kelly statistics."""

    def test_stats_with_sufficient_trades(self):
        """Test stats calculation with sufficient trades."""
        sizer = KellyPositionSizer()

        # 60% win rate, 2:1 ratio
        for _ in range(12):
            sizer.add_trade(TradeResult(pnl=200, entry_price=10000, exit_price=10200, quantity=100))
        for _ in range(8):
            sizer.add_trade(TradeResult(pnl=-100, entry_price=10000, exit_price=9900, quantity=100))

        stats = sizer.get_kelly_stats()

        assert stats.sample_size == 20
        assert stats.win_rate == pytest.approx(0.60, abs=0.01)
        assert stats.loss_rate == pytest.approx(0.40, abs=0.01)
        assert stats.avg_win == 200.0
        assert stats.avg_loss == 100.0
        assert stats.win_loss_ratio == 2.0
        assert stats.kelly_fraction == pytest.approx(0.40, abs=0.01)

    def test_stats_with_insufficient_trades(self):
        """Test stats with insufficient trades."""
        sizer = KellyPositionSizer()

        for _ in range(5):
            sizer.add_trade(TradeResult(pnl=100, entry_price=10000, exit_price=10100, quantity=100))

        stats = sizer.get_kelly_stats()

        assert stats.sample_size == 5
        assert stats.win_rate == 0.0
        assert stats.kelly_fraction == 0.0


class TestTradeHistoryManagement:
    """Tests for trade history management."""

    def test_add_single_trade(self):
        """Test adding a single trade."""
        sizer = KellyPositionSizer()
        trade = TradeResult(pnl=1000, entry_price=10000, exit_price=10100, quantity=100)

        sizer.add_trade(trade)
        assert sizer.get_trade_count() == 1

    def test_add_multiple_trades(self):
        """Test adding multiple trades."""
        sizer = KellyPositionSizer()
        trades = [
            TradeResult(pnl=1000, entry_price=10000, exit_price=10100, quantity=100),
            TradeResult(pnl=-500, entry_price=10000, exit_price=9950, quantity=100),
        ]

        sizer.add_trades(trades)
        assert sizer.get_trade_count() == 2

    def test_clear_history(self):
        """Test clearing trade history."""
        sizer = KellyPositionSizer()
        sizer.add_trade(TradeResult(pnl=1000, entry_price=10000, exit_price=10100, quantity=100))

        sizer.clear_history()
        assert sizer.get_trade_count() == 0


class TestLeverageRecommendation:
    """Tests for leverage recommendation."""

    def test_leverage_with_good_performance(self):
        """Test leverage recommendation with good performance."""
        sizer = KellyPositionSizer(half_kelly=True)

        # High win rate, good ratio
        for _ in range(15):
            sizer.add_trade(TradeResult(pnl=300, entry_price=10000, exit_price=10300, quantity=100))
        for _ in range(5):
            sizer.add_trade(TradeResult(pnl=-100, entry_price=10000, exit_price=9900, quantity=100))

        leverage = sizer.get_recommended_leverage()
        assert leverage >= 1.0
        assert leverage <= 3.0  # Capped at 3x

    def test_leverage_with_insufficient_data(self):
        """Test leverage returns 1.0 with insufficient data."""
        sizer = KellyPositionSizer()

        for _ in range(5):
            sizer.add_trade(TradeResult(pnl=100, entry_price=10000, exit_price=10100, quantity=100))

        leverage = sizer.get_recommended_leverage()
        assert leverage == 1.0


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_from_trades(self):
        """Test creating sizer from raw trade tuples."""
        trades = [
            (200, 10000, 10200, 100),  # Win
            (200, 10000, 10200, 100),  # Win
            (-100, 10000, 9900, 100),  # Loss
            (-100, 10000, 9900, 100),  # Loss
        ]

        sizer = create_kelly_sizer_from_trades(trades, half_kelly=True, max_kelly_pct=0.25)

        assert sizer.get_trade_count() == 4
        assert sizer.half_kelly is True
        assert sizer.max_kelly_pct == 0.25


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_trade_addition(self):
        """Test thread-safe trade addition."""
        import threading

        sizer = KellyPositionSizer()

        def add_trades():
            for _ in range(10):
                trade = TradeResult(pnl=100, entry_price=10000, exit_price=10100, quantity=100)
                sizer.add_trade(trade)

        threads = [threading.Thread(target=add_trades) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sizer.get_trade_count() == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
