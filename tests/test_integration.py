"""Integration tests for the complete trading system."""

import pytest
from datetime import datetime, time
from zoneinfo import ZoneInfo
from unittest.mock import Mock, MagicMock
import pandas as pd

from src.execution.partial_profit import (
    PartialProfitManager,
    FOUR_TIER_CONFIG,
    PositionDirection,
    PositionSnapshot,
    ExitTier,
)
from src.risk.risk_manager import RiskManager, RiskCheckResult
from src.risk.position_sizer import KellyPositionSizer, TradeResult
from src.utils.validation import InputValidator, ValidationResult

IST = ZoneInfo("Asia/Kolkata")


class TestPartialProfitSystem:
    """Test 4-tier partial profit system."""

    def test_tier_initialization(self):
        """Test that tiers are correctly initialized."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(
            entry=100000,  # ₹1000
            stop_loss=99000,  # ₹990
            quantity=100,
            direction=PositionDirection.LONG
        )

        assert len(tiers) == 4
        assert tiers[0].rr_level == 1.0
        assert tiers[1].rr_level == 2.0
        assert tiers[2].rr_level == 3.0
        assert tiers[3].rr_level is None  # Trailing

    def test_tier_exit_prices(self):
        """Test tier exit price calculations."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        # R = 100 paisa
        # Tier 1 @ 1:1 = ₹1010
        assert tiers[0].exit_price == 101000
        # Tier 2 @ 1:2 = ₹1020
        assert tiers[1].exit_price == 102000
        # Tier 3 @ 1:3 = ₹1030
        assert tiers[2].exit_price == 103000

    def test_tier_exit_prices_short(self):
        """Test tier exit price calculations for SHORT positions."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 101000, 100, PositionDirection.SHORT)

        # R = 100 paisa (stop loss above entry)
        # Tier 1 @ 1:1 = ₹990
        assert tiers[0].exit_price == 99000
        # Tier 2 @ 1:2 = ₹980
        assert tiers[1].exit_price == 98000
        # Tier 3 @ 1:3 = ₹970
        assert tiers[2].exit_price == 97000

    def test_partial_exit_detection(self):
        """Test that partial exits are detected at correct prices."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        position = PositionSnapshot(
            instrument_key="NSE_EQ|RELIANCE",
            direction=PositionDirection.LONG,
            entry_price=100000,
            stop_loss_price=99000,
            current_price=101000,  # Hits tier 1
            total_quantity=100,
            remaining_quantity=100
        )

        # Price hits tier 1
        exits = ppm.check_tiers(position, tiers)

        assert len(exits) >= 1
        assert exits[0].tier == 1
        assert exits[0].quantity == 25  # 25% of 100

    def test_tier_marking_exited(self):
        """Test that tiers are properly marked as exited."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        # Mark tier 1 as exited
        ppm.mark_tier_exited(tiers[0], 101000, 2500)  # 25 qty * 100 paisa profit

        assert tiers[0].exited is True
        assert tiers[0].exited_price == 101000
        assert tiers[0].realized_pnl == 2500

    def test_remaining_quantity_calculation(self):
        """Test remaining quantity after partial exits."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        # Mark tier 1 as exited (25%)
        ppm.mark_tier_exited(tiers[0], 101000, 2500)

        remaining = ppm.get_remaining_quantity(tiers, 100)
        assert remaining == 75

    def test_get_realized_pnl(self):
        """Test realized P&L calculation from exited tiers."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        # Exit tier 1 with 2500 paisa profit
        ppm.mark_tier_exited(tiers[0], 101000, 2500)

        total_pnl = ppm.get_realized_pnl(tiers)
        assert total_pnl == 2500

    def test_trailing_stop_calculation(self):
        """Test trailing stop price calculation for tier 4."""
        ppm = PartialProfitManager()

        # Long position, activated at 1:1 R:R
        trailing_price = ppm.calculate_trailing_stop_price(
            entry=100000,
            current_price=102000,  # At 2:1 R:R
            highest_price=102000,
            stop_loss_distance=1000,  # 100 paisa * 10 for scaling
            direction=PositionDirection.LONG,
            activation_rr=1.0
        )

        assert trailing_price is not None
        # Trailing stop should be below highest price by stop_loss_distance
        assert trailing_price == 102000 - 1000

    def test_trailing_stop_not_activated(self):
        """Test that trailing stop is not activated before threshold."""
        ppm = PartialProfitManager()

        # Long position, not yet at activation R:R
        trailing_price = ppm.calculate_trailing_stop_price(
            entry=100000,
            current_price=100500,  # Only 0.5:1 R:R
            highest_price=100500,
            stop_loss_distance=1000,
            direction=PositionDirection.LONG,
            activation_rr=1.0
        )

        assert trailing_price is None


class TestDailyTradeLimit:
    """Test max trades per day limit."""

    def test_trade_limit_enforced(self):
        """Test that trade limit is enforced."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        # Record 3 trades
        for _ in range(3):
            risk_mgr.record_trade("BANKNIFTY")

        # 4th trade should be blocked
        allowed, reason = risk_mgr.can_take_new_trade("BANKNIFTY")
        assert not allowed
        assert "Daily trade limit reached" in reason

    def test_trade_limit_resets_daily(self):
        """Test that limit resets next day."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_trades_per_day=3
        )

        # Record 3 trades yesterday
        yesterday = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
        for _ in range(3):
            risk_mgr.record_trade("BANKNIFTY", yesterday)

        # Should be allowed today
        today = datetime(2024, 1, 2, 10, 0, tzinfo=IST)
        allowed, _ = risk_mgr.can_take_new_trade("BANKNIFTY", today)
        assert allowed

    def test_trade_limit_per_symbol(self):
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
        assert allowed

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


class TestRiskManagerChecks:
    """Test RiskManager pre-trade checks."""

    def test_daily_loss_limit_check(self):
        """Test daily loss limit enforcement."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # Set realized loss at limit
        risk_mgr.update_pnl(realized=-30_000_00, unrealized=0)

        result = risk_mgr.check_daily_loss_limit()
        assert not result.approved
        assert "Daily loss limit breached" in result.reason

    def test_position_count_check(self):
        """Test max position count enforcement."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=2,
            max_position_size_pct=20
        )

        risk_mgr.update_positions(count=2, deployed=100_000_00)

        result = risk_mgr.check_position_count()
        assert not result.approved
        assert "Max open positions reached" in result.reason

    def test_position_size_check(self):
        """Test max position size enforcement."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        # Try to place order larger than 20% of capital
        order_value = 250_000_00  # 25% of capital

        result = risk_mgr.check_position_size(order_value)
        assert not result.approved
        assert "Position size" in result.reason and "exceeds" in result.reason

    def test_capital_deployment_check(self):
        """Test capital deployment limit enforcement."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20,
            max_capital_deployment_pct=80
        )

        # Already deployed 70%, trying to add 20% more
        risk_mgr.update_positions(count=3, deployed=700_000_00)

        result = risk_mgr.check_capital_deployment(200_000_00)
        assert not result.approved
        assert "Capital deployment limit breached" in result.reason

    def test_pre_trade_check_passes(self):
        """Test that pre-trade check passes when all limits are OK."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        result = risk_mgr.pre_trade_check(100_000_00, "NSE_EQ:RELIANCE")
        assert result.approved
        assert result.reason is None


class TestInputValidation:
    """Test input validation."""

    def test_invalid_symbol_rejected(self):
        """Test invalid symbols are rejected."""
        validator = InputValidator()

        result = validator.validate_symbol("")
        assert not result.is_valid

        result = validator.validate_symbol(None)
        assert not result.is_valid

        result = validator.validate_symbol("INVALID@SYMBOL")
        assert not result.is_valid

    def test_valid_symbol_accepted(self):
        """Test valid symbols are accepted."""
        validator = InputValidator()

        result = validator.validate_symbol("RELIANCE")
        assert result.is_valid
        assert result.sanitized_value == "RELIANCE"

        result = validator.validate_symbol("nifty 50")
        assert result.is_valid
        assert result.sanitized_value == "NIFTY 50"

    def test_invalid_price_rejected(self):
        """Test invalid prices are rejected."""
        validator = InputValidator()

        result = validator.validate_price(-100)
        assert not result.is_valid

        result = validator.validate_price(None)
        assert not result.is_valid

        result = validator.validate_price(0)
        assert not result.is_valid

    def test_invalid_quantity_rejected(self):
        """Test invalid quantities are rejected."""
        validator = InputValidator()

        result = validator.validate_quantity(0)
        assert not result.is_valid

        result = validator.validate_quantity(-10)
        assert not result.is_valid

        result = validator.validate_quantity(None)
        assert not result.is_valid

    def test_quantity_lot_size_multiple(self):
        """Test that quantity must be lot size multiple."""
        validator = InputValidator()

        result = validator.validate_quantity(100, lot_size=75)
        assert not result.is_valid
        assert "multiple of lot_size" in result.errors[0]

        result = validator.validate_quantity(150, lot_size=75)
        assert result.is_valid


class TestKellyPositionSizing:
    """Test Kelly criterion position sizing."""

    def test_kelly_calculation(self):
        """Test Kelly fraction calculation."""
        sizer = KellyPositionSizer()

        # Add winning trades (2:1 win/loss ratio, 60% win rate)
        for _ in range(12):
            sizer.add_trade(TradeResult(pnl=20000, entry_price=10000, exit_price=10200, quantity=100))

        # Add losing trades
        for _ in range(8):
            sizer.add_trade(TradeResult(pnl=-10000, entry_price=10000, exit_price=9900, quantity=100))

        stats = sizer.get_kelly_stats()
        assert stats.win_rate == 0.6
        assert stats.win_loss_ratio == 2.0
        assert stats.kelly_fraction > 0

    def test_safe_kelly_half_kelly(self):
        """Test Half-Kelly safety constraint."""
        sizer = KellyPositionSizer(half_kelly=True, max_kelly_pct=0.25)

        # Add winning trades
        for _ in range(20):
            sizer.add_trade(TradeResult(pnl=20000, entry_price=10000, exit_price=10200, quantity=100))
        for _ in range(10):
            sizer.add_trade(TradeResult(pnl=-10000, entry_price=10000, exit_price=9900, quantity=100))

        raw_kelly = sizer.calculate_kelly_fraction()
        safe_kelly = sizer.get_safe_kelly(raw_kelly)

        # Half-Kelly should be half of raw Kelly
        assert safe_kelly == raw_kelly / 2

    def test_position_size_with_kelly(self):
        """Test position sizing uses Kelly when enough history."""
        sizer = KellyPositionSizer(min_trades_for_kelly=20, half_kelly=True)

        # Add 30 trades
        for _ in range(18):
            sizer.add_trade(TradeResult(pnl=20000, entry_price=10000, exit_price=10200, quantity=100))
        for _ in range(12):
            sizer.add_trade(TradeResult(pnl=-10000, entry_price=10000, exit_price=9900, quantity=100))

        size = sizer.get_position_size(
            capital=1_000_000_00,  # 1L INR
            current_price=10000,
            max_risk_per_trade_pct=0.01
        )

        assert size > 0

    def test_position_size_fallback(self):
        """Test position sizing uses fallback when insufficient history."""
        sizer = KellyPositionSizer(min_trades_for_kelly=20, fallback_pct=0.02)

        # Add only 5 trades (less than min_trades_for_kelly)
        for _ in range(3):
            sizer.add_trade(TradeResult(pnl=20000, entry_price=10000, exit_price=10200, quantity=100))
        for _ in range(2):
            sizer.add_trade(TradeResult(pnl=-10000, entry_price=10000, exit_price=9900, quantity=100))

        size = sizer.get_position_size(
            capital=1_000_000_00,
            current_price=10000,
            max_risk_per_trade_pct=0.01
        )

        # Should use fallback percentage
        assert size > 0

    def test_negative_kelly_no_trade(self):
        """Test that negative Kelly returns zero position size."""
        sizer = KellyPositionSizer()

        # Add mostly losing trades (negative edge)
        for _ in range(15):
            sizer.add_trade(TradeResult(pnl=-20000, entry_price=10000, exit_price=9800, quantity=100))
        for _ in range(5):
            sizer.add_trade(TradeResult(pnl=10000, entry_price=10000, exit_price=10100, quantity=100))

        kelly = sizer.calculate_kelly_fraction()
        safe_kelly = sizer.get_safe_kelly(kelly)

        # Negative Kelly should result in no trade
        assert safe_kelly == 0.0


class TestRiskManagerReset:
    """Test RiskManager daily reset functionality."""

    def test_reset_daily_clears_pnl(self):
        """Test that reset_daily clears P&L counters."""
        risk_mgr = RiskManager(
            capital=1_000_000_00,
            max_daily_loss=30_000_00,
            max_open_positions=5,
            max_position_size_pct=20
        )

        risk_mgr.update_pnl(realized=-10000, unrealized=-5000)
        risk_mgr.record_trade("BANKNIFTY")

        risk_mgr.reset_daily()

        assert risk_mgr.realized_pnl == 0
        assert risk_mgr.unrealized_pnl == 0
        assert risk_mgr.get_daily_trade_count("BANKNIFTY") == 0


class TestPartialProfitEdgeCases:
    """Test edge cases for partial profit system."""

    def test_zero_quantity_raises_error(self):
        """Test that zero quantity raises error."""
        ppm = PartialProfitManager()

        with pytest.raises(ValueError, match="Quantity .* must be positive"):
            ppm.initialize_tiers(100000, 99000, 0, PositionDirection.LONG)

    def test_negative_prices_raise_error(self):
        """Test that negative prices raise error."""
        ppm = PartialProfitManager()

        with pytest.raises(ValueError, match="Entry .* and stop_loss .* must be positive"):
            ppm.initialize_tiers(-100000, 99000, 100, PositionDirection.LONG)

    def test_invalid_tier_number_raises_error(self):
        """Test that invalid tier number raises error."""
        with pytest.raises(ValueError, match="Tier must be between 1 and 4"):
            ExitTier(tier=5, rr_level=1.0, exit_percent=25)

    def test_invalid_exit_percent_raises_error(self):
        """Test that invalid exit percent raises error."""
        with pytest.raises(ValueError, match="Exit percent must be between 0 and 100"):
            ExitTier(tier=1, rr_level=1.0, exit_percent=0)

    def test_get_next_tier_returns_correct_tier(self):
        """Test getting next tier to execute."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        next_tier = ppm.get_next_tier(tiers)
        assert next_tier.tier == 1

        # Mark tier 1 as exited
        ppm.mark_tier_exited(tiers[0], 101000, 2500)

        next_tier = ppm.get_next_tier(tiers)
        assert next_tier.tier == 2

    def test_is_complete_all_tiers_exited(self):
        """Test is_complete when all tiers are exited."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        assert not ppm.is_complete(tiers)

        for tier in tiers:
            ppm.mark_tier_exited(tier, 101000, 2500)

        assert ppm.is_complete(tiers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
