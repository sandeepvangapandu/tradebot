"""Tests for the 4-tier partial profit system."""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.execution.partial_profit import (
    PartialProfitManager,
    FOUR_TIER_CONFIG,
    PositionDirection,
    PositionSnapshot,
    ExitTier,
    TierAction,
    TierExitResult,
)

IST = ZoneInfo("Asia/Kolkata")


class TestPartialProfitManagerInitialization:
    """Test PartialProfitManager initialization."""

    def test_default_initialization(self):
        """Test default initialization with 4-tier config."""
        ppm = PartialProfitManager()
        assert ppm._config == FOUR_TIER_CONFIG
        assert len(ppm._config) == 4

    def test_custom_config(self):
        """Test initialization with custom tier configuration."""
        custom_config = [
            {"tier": 1, "rr": 1.0, "percent": 50, "action": TierAction.EXIT},
            {"tier": 2, "rr": None, "percent": 50, "action": TierAction.TRAIL},
        ]
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(
            entry=100000,
            stop_loss=99000,
            quantity=100,
            direction=PositionDirection.LONG,
            custom_config=custom_config
        )
        assert len(tiers) == 2
        assert tiers[0].exit_percent == 50
        assert tiers[1].exit_percent == 50


class TestTierInitialization:
    """Test tier initialization and price calculations."""

    def test_long_position_tiers(self):
        """Test tier setup for long position."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(
            entry=100000,  # ₹1000
            stop_loss=99000,  # ₹990 (100 paisa risk)
            quantity=100,
            direction=PositionDirection.LONG
        )

        # Tier 1: 1:1 R:R = ₹1010
        assert tiers[0].tier == 1
        assert tiers[0].rr_level == 1.0
        assert tiers[0].exit_price == 101000
        assert tiers[0].exit_percent == 25

        # Tier 2: 1:2 R:R = ₹1020
        assert tiers[1].tier == 2
        assert tiers[1].rr_level == 2.0
        assert tiers[1].exit_price == 102000

        # Tier 3: 1:3 R:R = ₹1030
        assert tiers[2].tier == 3
        assert tiers[2].rr_level == 3.0
        assert tiers[2].exit_price == 103000

        # Tier 4: Trailing (no fixed price)
        assert tiers[3].tier == 4
        assert tiers[3].rr_level is None
        assert tiers[3].exit_price is None

    def test_short_position_tiers(self):
        """Test tier setup for short position."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(
            entry=100000,  # ₹1000
            stop_loss=101000,  # ₹1010 (100 paisa risk above entry)
            quantity=100,
            direction=PositionDirection.SHORT
        )

        # Tier 1: 1:1 R:R = ₹990
        assert tiers[0].exit_price == 99000

        # Tier 2: 1:2 R:R = ₹980
        assert tiers[1].exit_price == 98000

        # Tier 3: 1:3 R:R = ₹970
        assert tiers[2].exit_price == 97000

    def test_larger_risk_distance(self):
        """Test tier calculations with larger risk distance."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(
            entry=100000,  # ₹1000
            stop_loss=95000,  # ₹950 (500 paisa risk)
            quantity=100,
            direction=PositionDirection.LONG
        )

        # Tier 1: 1:1 R:R = ₹1050
        assert tiers[0].exit_price == 105000

        # Tier 2: 1:2 R:R = ₹1100
        assert tiers[1].exit_price == 110000

        # Tier 3: 1:3 R:R = ₹1150
        assert tiers[2].exit_price == 115000

    def test_invalid_entry_price(self):
        """Test that invalid entry price raises error."""
        ppm = PartialProfitManager()
        with pytest.raises(ValueError, match="Entry .* and stop_loss .* must be positive"):
            ppm.initialize_tiers(entry=0, stop_loss=99000, quantity=100, direction=PositionDirection.LONG)

    def test_invalid_stop_loss(self):
        """Test that invalid stop loss raises error."""
        ppm = PartialProfitManager()
        with pytest.raises(ValueError, match="Entry .* and stop_loss .* must be positive"):
            ppm.initialize_tiers(entry=100000, stop_loss=-1000, quantity=100, direction=PositionDirection.LONG)

    def test_invalid_quantity(self):
        """Test that invalid quantity raises error."""
        ppm = PartialProfitManager()
        with pytest.raises(ValueError, match="Quantity .* must be positive"):
            ppm.initialize_tiers(entry=100000, stop_loss=99000, quantity=0, direction=PositionDirection.LONG)


class TestTierChecking:
    """Test tier checking and exit detection."""

    def test_no_exit_when_price_below_tier(self):
        """Test that no exit is triggered when price is below tier level."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        position = PositionSnapshot(
            instrument_key="NSE_EQ|RELIANCE",
            direction=PositionDirection.LONG,
            entry_price=100000,
            stop_loss_price=99000,
            current_price=100500,  # Below tier 1 (101000)
            total_quantity=100,
            remaining_quantity=100
        )

        exits = ppm.check_tiers(position, tiers)
        assert len(exits) == 0

    def test_tier_1_exit_triggered(self):
        """Test that tier 1 exit is triggered when price reaches level."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        position = PositionSnapshot(
            instrument_key="NSE_EQ|RELIANCE",
            direction=PositionDirection.LONG,
            entry_price=100000,
            stop_loss_price=99000,
            current_price=101000,  # At tier 1 level
            total_quantity=100,
            remaining_quantity=100
        )

        exits = ppm.check_tiers(position, tiers)
        assert len(exits) == 1
        assert exits[0].tier == 1
        assert exits[0].quantity == 25  # 25% of 100

    def test_multiple_tiers_triggered(self):
        """Test that multiple tiers can trigger if price jumps."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        position = PositionSnapshot(
            instrument_key="NSE_EQ|RELIANCE",
            direction=PositionDirection.LONG,
            entry_price=100000,
            stop_loss_price=99000,
            current_price=102500,  # Above tier 2 (102000)
            total_quantity=100,
            remaining_quantity=100
        )

        exits = ppm.check_tiers(position, tiers)
        # Should trigger tier 1 and tier 2
        assert len(exits) == 2
        assert exits[0].tier == 1
        assert exits[1].tier == 2

    def test_short_position_exit_triggered(self):
        """Test exit detection for short positions."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 101000, 100, PositionDirection.SHORT)

        position = PositionSnapshot(
            instrument_key="NSE_EQ|RELIANCE",
            direction=PositionDirection.SHORT,
            entry_price=100000,
            stop_loss_price=101000,
            current_price=99000,  # At tier 1 level for short
            total_quantity=100,
            remaining_quantity=100
        )

        exits = ppm.check_tiers(position, tiers)
        assert len(exits) == 1
        assert exits[0].tier == 1

    def test_exited_tier_not_triggered_again(self):
        """Test that already exited tiers are not triggered again."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        # Mark tier 1 as already exited
        ppm.mark_tier_exited(tiers[0], 101000, 2500)

        position = PositionSnapshot(
            instrument_key="NSE_EQ|RELIANCE",
            direction=PositionDirection.LONG,
            entry_price=100000,
            stop_loss_price=99000,
            current_price=101000,
            total_quantity=100,
            remaining_quantity=75  # After tier 1 exit
        )

        exits = ppm.check_tiers(position, tiers)
        # Tier 1 should not trigger again
        assert all(e.tier != 1 for e in exits)


class TestTierMarking:
    """Test tier marking and state management."""

    def test_mark_tier_exited(self):
        """Test marking a tier as exited."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        ppm.mark_tier_exited(tiers[0], 101000, 2500)

        assert tiers[0].exited is True
        assert tiers[0].exited_price == 101000
        assert tiers[0].realized_pnl == 2500
        assert tiers[0].exited_at is not None

    def test_get_remaining_quantity(self):
        """Test remaining quantity calculation."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        # No tiers exited
        assert ppm.get_remaining_quantity(tiers, 100) == 100

        # Tier 1 exited (25%)
        ppm.mark_tier_exited(tiers[0], 101000, 2500)
        assert ppm.get_remaining_quantity(tiers, 100) == 75

        # Tier 2 exited (50% total)
        ppm.mark_tier_exited(tiers[1], 102000, 2500)
        assert ppm.get_remaining_quantity(tiers, 100) == 50

    def test_get_realized_pnl(self):
        """Test realized P&L calculation."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        # No tiers exited
        assert ppm.get_realized_pnl(tiers) == 0

        # Exit tier 1 with profit
        ppm.mark_tier_exited(tiers[0], 101000, 2500)
        assert ppm.get_realized_pnl(tiers) == 2500

        # Exit tier 2 with profit
        ppm.mark_tier_exited(tiers[1], 102000, 2500)
        assert ppm.get_realized_pnl(tiers) == 5000

    def test_get_unexited_tiers(self):
        """Test getting list of unexited tiers."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        unexited = ppm.get_unexited_tiers(tiers)
        assert len(unexited) == 4

        ppm.mark_tier_exited(tiers[0], 101000, 2500)

        unexited = ppm.get_unexited_tiers(tiers)
        assert len(unexited) == 3
        assert all(t.tier != 1 for t in unexited)

    def test_get_next_tier(self):
        """Test getting next tier to execute."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        next_tier = ppm.get_next_tier(tiers)
        assert next_tier.tier == 1

        ppm.mark_tier_exited(tiers[0], 101000, 2500)
        next_tier = ppm.get_next_tier(tiers)
        assert next_tier.tier == 2

        ppm.mark_tier_exited(tiers[1], 102000, 2500)
        next_tier = ppm.get_next_tier(tiers)
        assert next_tier.tier == 3

    def test_is_complete(self):
        """Test checking if all tiers are complete."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        assert ppm.is_complete(tiers) is False

        for tier in tiers:
            ppm.mark_tier_exited(tier, 101000, 2500)

        assert ppm.is_complete(tiers) is True

    def test_get_tier_summary(self):
        """Test tier summary generation."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        ppm.mark_tier_exited(tiers[0], 101000, 2500)

        summary = ppm.get_tier_summary(tiers)
        assert summary["total_tiers"] == 4
        assert summary["exited_tiers"] == 1
        assert summary["remaining_tiers"] == 3
        assert summary["total_realized_pnl_paisa"] == 2500
        assert summary["is_complete"] is False


class TestTrailingStop:
    """Test trailing stop functionality."""

    def test_trailing_stop_activates_at_threshold(self):
        """Test trailing stop activates when R:R threshold is reached."""
        ppm = PartialProfitManager()

        # Long position at 1.5:1 R:R (above 1.0 activation)
        trailing_price = ppm.calculate_trailing_stop_price(
            entry=100000,
            current_price=101500,
            highest_price=101500,
            stop_loss_distance=1000,
            direction=PositionDirection.LONG,
            activation_rr=1.0
        )

        assert trailing_price is not None
        assert trailing_price == 100500  # highest - stop_loss_distance

    def test_trailing_stop_not_activated_below_threshold(self):
        """Test trailing stop does not activate below R:R threshold."""
        ppm = PartialProfitManager()

        # Long position at 0.5:1 R:R (below 1.0 activation)
        trailing_price = ppm.calculate_trailing_stop_price(
            entry=100000,
            current_price=100500,
            highest_price=100500,
            stop_loss_distance=1000,
            direction=PositionDirection.LONG,
            activation_rr=1.0
        )

        assert trailing_price is None

    def test_trailing_stop_for_short_position(self):
        """Test trailing stop for short positions."""
        ppm = PartialProfitManager()

        # Short position at 1.5:1 R:R
        trailing_price = ppm.calculate_trailing_stop_price(
            entry=100000,
            current_price=98500,
            highest_price=98500,  # Actually lowest for short
            stop_loss_distance=1000,
            direction=PositionDirection.SHORT,
            activation_rr=1.0
        )

        assert trailing_price is not None
        # For short, trail above lowest price
        assert trailing_price == 99500  # lowest + stop_loss_distance

    def test_trailing_stop_does_not_go_below_entry_long(self):
        """Test trailing stop for long doesn't go below entry."""
        ppm = PartialProfitManager()

        # Long position with high highest_price
        trailing_price = ppm.calculate_trailing_stop_price(
            entry=100000,
            current_price=102000,
            highest_price=102000,
            stop_loss_distance=5000,  # Large distance
            direction=PositionDirection.LONG,
            activation_rr=1.0
        )

        # Should be capped at entry price
        assert trailing_price == 100000

    def test_trailing_stop_does_not_go_above_entry_short(self):
        """Test trailing stop for short doesn't go above entry."""
        ppm = PartialProfitManager()

        # Short position with low lowest_price
        trailing_price = ppm.calculate_trailing_stop_price(
            entry=100000,
            current_price=95000,
            highest_price=95000,
            stop_loss_distance=10000,  # Large distance
            direction=PositionDirection.SHORT,
            activation_rr=1.0
        )

        # Should be capped at entry price
        assert trailing_price == 100000


class TestResetTiers:
    """Test tier reset functionality."""

    def test_reset_tiers(self):
        """Test resetting all tiers to unexited state."""
        ppm = PartialProfitManager()
        tiers = ppm.initialize_tiers(100000, 99000, 100, PositionDirection.LONG)

        # Mark some tiers as exited
        ppm.mark_tier_exited(tiers[0], 101000, 2500)
        ppm.mark_tier_exited(tiers[1], 102000, 2500)

        assert tiers[0].exited is True
        assert tiers[1].exited is True

        ppm.reset_tiers(tiers)

        for tier in tiers:
            assert tier.exited is False
            assert tier.exited_at is None
            assert tier.exited_price is None
            assert tier.realized_pnl == 0


class TestConvenienceFunction:
    """Test convenience factory function."""

    def test_create_partial_profit_tiers(self):
        """Test factory function for creating tiers."""
        from src.execution.partial_profit import create_partial_profit_tiers

        tiers = create_partial_profit_tiers(
            entry=100000,
            stop_loss=99000,
            quantity=100,
            direction=PositionDirection.LONG
        )

        assert len(tiers) == 4
        assert tiers[0].exit_price == 101000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
