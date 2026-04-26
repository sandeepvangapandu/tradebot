"""Tests for risk limits including daily trade limits and position limits."""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.risk.risk_manager import RiskManager, RiskCheckResult

IST = ZoneInfo("Asia/Kolkata")


class TestDailyTradeLimit:
    """Test max trades per day limit."""

    def test_default_max_trades_per_day(self):
        """Test default max trades per day is 3."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )
        assert risk_mgr.max_trades_per_day == 3

    def test_custom_max_trades_per_day(self):
        """Test custom max trades per day."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=5
        )
        assert risk_mgr.max_trades_per_day == 5

    def test_can_take_new_trade_initially(self):
        """Test that new trade is allowed initially."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        allowed, reason = risk_mgr.can_take_new_trade("BANKNIFTY")
        assert allowed is True
        assert reason == "OK"

    def test_trade_limit_blocks_after_max_reached(self):
        """Test that trade is blocked after max trades reached."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        # Record 3 trades
        for i in range(3):
            risk_mgr.record_trade("BANKNIFTY")

        # 4th trade should be blocked
        allowed, reason = risk_mgr.can_take_new_trade("BANKNIFTY")
        assert allowed is False
        assert "Daily trade limit reached" in reason

    def test_trade_limit_per_symbol_isolated(self):
        """Test that trade limit is per symbol."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        # Record 3 trades for BANKNIFTY
        for _ in range(3):
            risk_mgr.record_trade("BANKNIFTY")

        # Should still be able to trade NIFTY
        allowed, _ = risk_mgr.can_take_new_trade("NIFTY")
        assert allowed is True

        # Should be able to trade RELIANCE
        allowed, _ = risk_mgr.can_take_new_trade("RELIANCE")
        assert allowed is True

    def test_get_daily_trade_count(self):
        """Test getting daily trade count."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        assert risk_mgr.get_daily_trade_count("BANKNIFTY") == 0

        risk_mgr.record_trade("BANKNIFTY")
        assert risk_mgr.get_daily_trade_count("BANKNIFTY") == 1

        risk_mgr.record_trade("BANKNIFTY")
        assert risk_mgr.get_daily_trade_count("BANKNIFTY") == 2

    def test_trade_count_with_timestamp(self):
        """Test recording trades with specific timestamps."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        now = datetime(2024, 1, 15, 10, 30, tzinfo=IST)
        risk_mgr.record_trade("BANKNIFTY", now)

        assert risk_mgr.get_daily_trade_count("BANKNIFTY") == 1

    def test_trade_limit_resets_next_day(self):
        """Test that trade limit resets on next day."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        # Record 3 trades yesterday
        yesterday = datetime(2024, 1, 15, 14, 30, tzinfo=IST)
        for _ in range(3):
            risk_mgr.record_trade("BANKNIFTY", yesterday)

        # Should be blocked yesterday
        allowed, _ = risk_mgr.can_take_new_trade("BANKNIFTY", yesterday)
        assert allowed is False

        # Should be allowed today
        today = datetime(2024, 1, 16, 10, 0, tzinfo=IST)
        allowed, _ = risk_mgr.can_take_new_trade("BANKNIFTY", today)
        assert allowed is True

    def test_old_trades_cleaned_up(self):
        """Test that old trades are cleaned up when checking."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        # Record trades yesterday
        yesterday = datetime(2024, 1, 15, 10, 0, tzinfo=IST)
        risk_mgr.record_trade("BANKNIFTY", yesterday)
        risk_mgr.record_trade("BANKNIFTY", yesterday)

        # Check today - should trigger cleanup
        today = datetime(2024, 1, 16, 10, 0, tzinfo=IST)
        risk_mgr._get_todays_trades("BANKNIFTY", today)

        # Old trades should be cleaned up
        assert len(risk_mgr._daily_trades.get("BANKNIFTY", [])) == 0

    def test_multiple_symbols_tracked_separately(self):
        """Test that multiple symbols are tracked separately."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=2
        )

        # Record 2 trades for BANKNIFTY
        risk_mgr.record_trade("BANKNIFTY")
        risk_mgr.record_trade("BANKNIFTY")

        # Record 1 trade for NIFTY
        risk_mgr.record_trade("NIFTY")

        # BANKNIFTY should be blocked
        assert risk_mgr.can_take_new_trade("BANKNIFTY")[0] is False

        # NIFTY should be allowed (1/2)
        assert risk_mgr.can_take_new_trade("NIFTY")[0] is True

        # RELIANCE should be allowed (0/2)
        assert risk_mgr.can_take_new_trade("RELIANCE")[0] is True


class TestDailyLossLimit:
    """Test daily loss limit enforcement."""

    def test_daily_loss_limit_not_breached(self):
        """Test that trading is allowed when loss is below limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # Small loss
        risk_mgr.update_pnl(realized=-10_000_00, unrealized=-5_000_00)

        result = risk_mgr.check_daily_loss_limit()
        assert result.approved is True

    def test_daily_loss_limit_breached(self):
        """Test that trading is blocked when loss limit is breached."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # Loss at limit
        risk_mgr.update_pnl(realized=-30_000_00, unrealized=0)

        result = risk_mgr.check_daily_loss_limit()
        assert result.approved is False
        assert "Daily loss limit breached" in result.reason

    def test_daily_loss_limit_exceeded(self):
        """Test that trading is blocked when loss exceeds limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # Loss exceeds limit
        risk_mgr.update_pnl(realized=-35_000_00, unrealized=0)

        result = risk_mgr.check_daily_loss_limit()
        assert result.approved is False

    def test_unrealized_loss_counts(self):
        """Test that unrealized loss counts toward daily limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # Only unrealized loss
        risk_mgr.update_pnl(realized=0, unrealized=-30_000_00)

        result = risk_mgr.check_daily_loss_limit()
        assert result.approved is False

    def test_combined_loss_counts(self):
        """Test that combined realized + unrealized loss counts."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # Combined loss at limit
        risk_mgr.update_pnl(realized=-20_000_00, unrealized=-10_000_00)

        result = risk_mgr.check_daily_loss_limit()
        assert result.approved is False


class TestPositionCountLimit:
    """Test max open positions limit."""

    def test_position_count_below_limit(self):
        """Test that trading is allowed when positions below limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        risk_mgr.update_positions(count=4, deployed=400_000_00)

        result = risk_mgr.check_position_count()
        assert result.approved is True

    def test_position_count_at_limit(self):
        """Test that trading is blocked when positions at limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        risk_mgr.update_positions(count=5, deployed=500_000_00)

        result = risk_mgr.check_position_count()
        assert result.approved is False
        assert "Max open positions reached" in result.reason

    def test_position_count_above_limit(self):
        """Test that trading is blocked when positions above limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        risk_mgr.update_positions(count=6, deployed=600_000_00)

        result = risk_mgr.check_position_count()
        assert result.approved is False


class TestPositionSizeLimit:
    """Test max position size limit."""

    def test_position_size_within_limit(self):
        """Test that order is allowed when size within limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # 20% of 1L = 20,000
        result = risk_mgr.check_position_size(200_000_00)
        assert result.approved is True

    def test_position_size_exceeds_limit(self):
        """Test that order is blocked when size exceeds limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # 25% of 1L = 25,000
        result = risk_mgr.check_position_size(250_000_00)
        assert result.approved is False
        assert "exceeds" in result.reason

    def test_position_size_limit_calculation(self):
        """Test that position size limit is calculated correctly."""
        risk_mgr = RiskManager(
            capital=2_000_000_00,  # 2L
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=25
        )

        # 25% of 2L = 50,000
        result = risk_mgr.check_position_size(500_000_00)
        assert result.approved is True

        result = risk_mgr.check_position_size(501_000_00)
        assert result.approved is False


class TestCapitalDeploymentLimit:
    """Test max capital deployment limit."""

    def test_capital_deployment_within_limit(self):
        """Test that order is allowed when deployment within limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_capital_deployment_pct=80
        )

        # Already deployed 70%, adding 10% = 80%
        risk_mgr.update_positions(count=3, deployed=700_000_00)

        result = risk_mgr.check_capital_deployment(100_000_00)
        assert result.approved is True

    def test_capital_deployment_exceeds_limit(self):
        """Test that order is blocked when deployment would exceed limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_capital_deployment_pct=80
        )

        # Already deployed 70%, adding 20% = 90% (exceeds 80%)
        risk_mgr.update_positions(count=3, deployed=700_000_00)

        result = risk_mgr.check_capital_deployment(200_000_00)
        assert result.approved is False
        assert "Capital deployment limit breached" in result.reason

    def test_capital_deployment_at_limit(self):
        """Test that order is blocked when deployment at limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_capital_deployment_pct=80
        )

        # Already deployed 80%
        risk_mgr.update_positions(count=4, deployed=800_000_00)

        result = risk_mgr.check_capital_deployment(100_000_00)
        assert result.approved is False


class TestPreTradeCheck:
    """Test comprehensive pre-trade check."""

    def test_pre_trade_check_passes(self):
        """Test that pre-trade check passes when all limits OK."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_capital_deployment_pct=80
        )

        result = risk_mgr.pre_trade_check(100_000_00, "NSE_EQ:RELIANCE")
        assert result.approved is True

    def test_pre_trade_check_fails_on_daily_loss(self):
        """Test that pre-trade check fails when daily loss limit breached."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        risk_mgr.update_pnl(realized=-30_000_00, unrealized=0)

        result = risk_mgr.pre_trade_check(100_000_00, "NSE_EQ:RELIANCE")
        assert result.approved is False
        assert "Daily loss limit breached" in result.reason

    def test_pre_trade_check_fails_on_position_count(self):
        """Test that pre-trade check fails when max positions reached."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=2,
            max_position_size_pct=20
        )

        risk_mgr.update_positions(count=2, deployed=200_000_00)

        result = risk_mgr.pre_trade_check(100_000_00, "NSE_EQ:RELIANCE")
        assert result.approved is False
        assert "Max open positions reached" in result.reason

    def test_pre_trade_check_fails_on_position_size(self):
        """Test that pre-trade check fails when position size too large."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=10
        )

        result = risk_mgr.pre_trade_check(150_000_00, "NSE_EQ:RELIANCE")
        assert result.approved is False
        assert "exceeds" in result.reason

    def test_pre_trade_check_fails_on_capital_deployment(self):
        """Test that pre-trade check fails when capital deployment would exceed limit."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_capital_deployment_pct=50
        )

        risk_mgr.update_positions(count=2, deployed=500_000_00)

        result = risk_mgr.pre_trade_check(100_000_00, "NSE_EQ:RELIANCE")
        assert result.approved is False
        assert "Capital deployment limit breached" in result.reason


class TestResetDaily:
    """Test daily reset functionality."""

    def test_reset_daily_clears_pnl(self):
        """Test that reset_daily clears P&L counters."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        risk_mgr.update_pnl(realized=-10000, unrealized=-5000)
        risk_mgr.reset_daily()

        assert risk_mgr.realized_pnl == 0
        assert risk_mgr.unrealized_pnl == 0

    def test_reset_daily_clears_trade_history(self):
        """Test that reset_daily clears trade history."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        risk_mgr.record_trade("BANKNIFTY")
        risk_mgr.record_trade("BANKNIFTY")

        assert risk_mgr.get_daily_trade_count("BANKNIFTY") == 2

        risk_mgr.reset_daily()

        assert risk_mgr.get_daily_trade_count("BANKNIFTY") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
