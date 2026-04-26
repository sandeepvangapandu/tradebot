"""Tests for the risk management module.

Tests daily loss limits, position count limits, position size limits,
capital deployment limits, and circuit breaker logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from freezegun import freeze_time

from src.risk.circuit_breaker import CircuitBreaker
from src.risk.risk_manager import RiskCheckResult, RiskManager

IST = ZoneInfo("Asia/Kolkata")


class TestDailyLossLimit:
    """Test daily loss limit enforcement."""

    def test_daily_loss_limit_not_breached(self, risk_manager: RiskManager) -> None:
        """Test trading allowed when daily loss is within limit."""
        # Set P&L to small loss
        risk_manager.update_pnl(realized=-10_000, unrealized=-5_000)  # 150 INR loss

        result = risk_manager.check_daily_loss_limit()

        assert result.approved is True
        assert result.reason is None

    def test_daily_loss_limit_breached(self, risk_manager: RiskManager) -> None:
        """Test trading blocked when daily loss exceeds limit."""
        # Set P&L to large loss (exceeds 50,000 paisa limit)
        risk_manager.update_pnl(realized=-60_000, unrealized=0)

        result = risk_manager.check_daily_loss_limit()

        assert result.approved is False
        assert result.reason is not None
        assert "Daily loss limit breached" in result.reason

    def test_daily_loss_includes_unrealized(self, risk_manager: RiskManager) -> None:
        """Test unrealized P&L is included in daily loss calculation."""
        # Realized loss + unrealized loss exceeds limit
        risk_manager.update_pnl(realized=-30_000, unrealized=-25_000)

        result = risk_manager.check_daily_loss_limit()

        assert result.approved is False

    def test_daily_loss_limit_exactly_at_threshold(self, risk_manager: RiskManager) -> None:
        """Test behavior when loss is exactly at the limit."""
        # Exactly at limit (50,000 paisa)
        risk_manager.update_pnl(realized=-50_000, unrealized=0)

        result = risk_manager.check_daily_loss_limit()

        # Should be blocked when at or exceeding limit
        assert result.approved is False

    def test_profits_not_blocked(self, risk_manager: RiskManager) -> None:
        """Test that profits don't trigger loss limit."""
        risk_manager.update_pnl(realized=100_000, unrealized=50_000)

        result = risk_manager.check_daily_loss_limit()

        assert result.approved is True


class TestPositionCountLimit:
    """Test position count limit enforcement."""

    def test_position_count_within_limit(self, risk_manager: RiskManager) -> None:
        """Test trading allowed when position count is within limit."""
        risk_manager.update_positions(count=3, deployed=1_000_000)

        result = risk_manager.check_position_count()

        assert result.approved is True

    def test_position_count_at_limit(self, risk_manager: RiskManager) -> None:
        """Test trading blocked when position count reaches limit."""
        risk_manager.update_positions(count=5, deployed=1_000_000)

        result = risk_manager.check_position_count()

        assert result.approved is False
        assert "Max open positions reached" in result.reason

    def test_position_count_over_limit(self, risk_manager: RiskManager) -> None:
        """Test trading blocked when position count exceeds limit."""
        risk_manager.update_positions(count=6, deployed=1_000_000)

        result = risk_manager.check_position_count()

        assert result.approved is False

    def test_zero_positions_allowed(self, risk_manager: RiskManager) -> None:
        """Test trading allowed with zero positions."""
        risk_manager.update_positions(count=0, deployed=0)

        result = risk_manager.check_position_count()

        assert result.approved is True


class TestPositionSizeLimit:
    """Test position size limit enforcement."""

    def test_position_size_within_limit(self, risk_manager: RiskManager) -> None:
        """Test order allowed when size is within limit."""
        # 20% of 1,00,000 INR = 20,000 INR = 2,000,000 paisa
        order_value = 1_500_000  # 15,000 INR

        result = risk_manager.check_position_size(order_value)

        assert result.approved is True

    def test_position_size_at_limit(self, risk_manager: RiskManager) -> None:
        """Test order at exact limit is allowed."""
        # Exactly 20% of capital
        order_value = 2_000_000  # 20,000 INR

        result = risk_manager.check_position_size(order_value)

        assert result.approved is True

    def test_position_size_over_limit(self, risk_manager: RiskManager) -> None:
        """Test order blocked when size exceeds limit."""
        order_value = 2_500_000  # 25,000 INR

        result = risk_manager.check_position_size(order_value)

        assert result.approved is False
        assert "exceeds" in result.reason

    def test_position_size_calculation(self, risk_manager: RiskManager) -> None:
        """Test position size limit is calculated correctly."""
        # 20% of 1,00,000 INR capital
        max_allowed = risk_manager.max_position_size_pct * risk_manager.capital // 100
        assert max_allowed == 2_000_000  # 20,000 INR in paisa


class TestCapitalDeploymentLimit:
    """Test capital deployment limit enforcement."""

    def test_capital_deployment_within_limit(self, risk_manager: RiskManager) -> None:
        """Test order allowed when deployment stays within limit."""
        # Currently deployed: 50,000 INR, trying to add 20,000 INR
        # Total: 70,000 INR (70% of 1,00,000 INR)
        risk_manager.update_positions(count=2, deployed=5_000_000)
        order_value = 2_000_000

        result = risk_manager.check_capital_deployment(order_value)

        assert result.approved is True

    def test_capital_deployment_at_limit(self, risk_manager: RiskManager) -> None:
        """Test order allowed when deployment reaches exactly at limit."""
        # Currently deployed: 60,000 INR, trying to add 20,000 INR
        # Total: 80,000 INR (80% of 1,00,000 INR)
        risk_manager.update_positions(count=3, deployed=6_000_000)
        order_value = 2_000_000

        result = risk_manager.check_capital_deployment(order_value)

        assert result.approved is True

    def test_capital_deployment_over_limit(self, risk_manager: RiskManager) -> None:
        """Test order blocked when deployment would exceed limit."""
        # Currently deployed: 70,000 INR, trying to add 20,000 INR
        # Total: 90,000 INR (90% of 1,00,000 INR) - exceeds 80% limit
        risk_manager.update_positions(count=3, deployed=7_000_000)
        order_value = 2_000_000

        result = risk_manager.check_capital_deployment(order_value)

        assert result.approved is False
        assert "Capital deployment limit breached" in result.reason

    def test_capital_deployment_empty_portfolio(self, risk_manager: RiskManager) -> None:
        """Test order allowed with empty portfolio."""
        risk_manager.update_positions(count=0, deployed=0)
        order_value = 5_000_000  # 50,000 INR

        result = risk_manager.check_capital_deployment(order_value)

        assert result.approved is True


class TestPreTradeCheck:
    """Test comprehensive pre-trade risk check."""

    def test_pre_trade_all_checks_pass(self, risk_manager: RiskManager) -> None:
        """Test pre-trade check passes when all limits are respected."""
        risk_manager.update_pnl(realized=0, unrealized=0)
        risk_manager.update_positions(count=1, deployed=1_000_000)

        result = risk_manager.pre_trade_check(
            order_value=1_000_000,
            instrument_key="NSE_EQ:RELIANCE"
        )

        assert result.approved is True

    def test_pre_trade_fails_on_first_violation(self, risk_manager: RiskManager) -> None:
        """Test pre-trade check returns first failing check."""
        # Set up multiple violations
        risk_manager.update_pnl(realized=-100_000, unrealized=0)  # Daily loss exceeded
        risk_manager.update_positions(count=10, deployed=9_000_000)  # Position count exceeded

        result = risk_manager.pre_trade_check(
            order_value=5_000_000,
            instrument_key="NSE_EQ:RELIANCE"
        )

        assert result.approved is False
        # Should return the first failure (daily loss)
        assert "Daily loss limit breached" in result.reason

    def test_pre_trade_order_value_used_correctly(self, risk_manager: RiskManager) -> None:
        """Test order value is correctly passed to all checks."""
        risk_manager.update_positions(count=0, deployed=0)

        # Order value that would exceed position size limit
        order_value = 3_000_000  # 30,000 INR (exceeds 20% limit)

        result = risk_manager.pre_trade_check(
            order_value=order_value,
            instrument_key="NSE_EQ:RELIANCE"
        )

        assert result.approved is False
        assert "exceeds" in result.reason


class TestCircuitBreaker:
    """Test circuit breaker consecutive loss logic."""

    def test_circuit_breaker_initial_state(self, circuit_breaker: CircuitBreaker) -> None:
        """Test circuit breaker starts in non-halted state."""
        assert circuit_breaker.is_halted() is False
        assert circuit_breaker.consecutive_losses == 0
        assert circuit_breaker.halted is False

    def test_record_single_loss(self, circuit_breaker: CircuitBreaker) -> None:
        """Test recording a single loss."""
        circuit_breaker.record_loss()

        assert circuit_breaker.consecutive_losses == 1
        assert circuit_breaker.is_halted() is False

    def test_record_multiple_losses_below_threshold(self, circuit_breaker: CircuitBreaker) -> None:
        """Test recording losses below threshold doesn't halt."""
        circuit_breaker.record_loss()
        circuit_breaker.record_loss()

        assert circuit_breaker.consecutive_losses == 2
        assert circuit_breaker.is_halted() is False

    def test_record_losses_at_threshold_activates_pause(self, circuit_breaker: CircuitBreaker) -> None:
        """Test recording losses at threshold activates pause."""
        with freeze_time("2024-03-15 10:00:00", tz_offset=5.5):
            circuit_breaker.record_loss()
            circuit_breaker.record_loss()
            circuit_breaker.record_loss()

            assert circuit_breaker.consecutive_losses == 3
            assert circuit_breaker.is_halted() is True
            assert circuit_breaker.halt_until is not None

    def test_record_win_resets_counter(self, circuit_breaker: CircuitBreaker) -> None:
        """Test recording a win resets consecutive loss counter."""
        circuit_breaker.record_loss()
        circuit_breaker.record_loss()
        circuit_breaker.record_win()

        assert circuit_breaker.consecutive_losses == 0
        assert circuit_breaker.is_halted() is False

    def test_record_win_after_halt_doesnt_clear_halt(self, circuit_breaker: CircuitBreaker) -> None:
        """Test win doesn't clear active halt (time-based only)."""
        with freeze_time("2024-03-15 10:00:00", tz_offset=5.5):
            circuit_breaker.record_loss()
            circuit_breaker.record_loss()
            circuit_breaker.record_loss()

            assert circuit_breaker.is_halted() is True

            # Recording a win shouldn't clear the halt
            circuit_breaker.record_win()
            assert circuit_breaker.is_halted() is True

    def test_auto_resume_after_pause_period(self, circuit_breaker: CircuitBreaker) -> None:
        """Test auto-resume after pause period expires."""
        initial_time = datetime(2024, 3, 15, 10, 0, 0, tzinfo=IST)

        with freeze_time(initial_time) as frozen_time:
            # Trigger halt
            circuit_breaker.record_loss()
            circuit_breaker.record_loss()
            circuit_breaker.record_loss()

            assert circuit_breaker.is_halted() is True

            # Move time forward past pause period (30 minutes)
            frozen_time.move_to(initial_time + timedelta(minutes=31))

            # Should auto-resume
            assert circuit_breaker.is_halted() is False
            assert circuit_breaker.consecutive_losses == 0

    def test_kill_switch_permanent_halt(self, circuit_breaker: CircuitBreaker) -> None:
        """Test kill switch creates permanent halt."""
        mock_order_manager = MagicMock()

        circuit_breaker.kill_switch(mock_order_manager)

        assert circuit_breaker.is_halted() is True
        assert circuit_breaker.halt_until is None  # Permanent halt

        # Cancel and exit should have been called
        mock_order_manager.cancel_all_orders.assert_called_once()
        mock_order_manager.exit_all_positions.assert_called_once()

    def test_kill_switch_without_order_manager(self, circuit_breaker: CircuitBreaker) -> None:
        """Test kill switch works without order manager."""
        circuit_breaker.kill_switch()

        assert circuit_breaker.is_halted() is True
        assert circuit_breaker.halt_until is None

    def test_manual_resume(self, circuit_breaker: CircuitBreaker) -> None:
        """Test manual resume clears halt."""
        with freeze_time("2024-03-15 10:00:00", tz_offset=5.5):
            circuit_breaker.record_loss()
            circuit_breaker.record_loss()
            circuit_breaker.record_loss()

            assert circuit_breaker.is_halted() is True

            circuit_breaker.resume()

            assert circuit_breaker.is_halted() is False
            assert circuit_breaker.consecutive_losses == 0
            assert circuit_breaker.halt_until is None

    def test_kill_switch_requires_manual_resume(self, circuit_breaker: CircuitBreaker) -> None:
        """Test kill switch halt doesn't auto-resume."""
        initial_time = datetime(2024, 3, 15, 10, 0, 0, tzinfo=IST)

        with freeze_time(initial_time) as frozen_time:
            circuit_breaker.kill_switch()

            assert circuit_breaker.is_halted() is True

            # Move time forward (shouldn't auto-resume)
            frozen_time.move_to(initial_time + timedelta(hours=24))

            assert circuit_breaker.is_halted() is True

            # Must manually resume
            circuit_breaker.resume()
            assert circuit_breaker.is_halted() is False

    def test_loss_win_loss_sequence(self, circuit_breaker: CircuitBreaker) -> None:
        """Test loss-win-loss sequence resets counter correctly."""
        circuit_breaker.record_loss()
        circuit_breaker.record_loss()
        assert circuit_breaker.consecutive_losses == 2

        circuit_breaker.record_win()
        assert circuit_breaker.consecutive_losses == 0

        circuit_breaker.record_loss()
        assert circuit_breaker.consecutive_losses == 1
        assert circuit_breaker.is_halted() is False


class TestRiskManagerStateUpdates:
    """Test RiskManager state update methods."""

    def test_update_pnl(self, risk_manager: RiskManager) -> None:
        """Test updating P&L values."""
        risk_manager.update_pnl(realized=50_000, unrealized=25_000)

        assert risk_manager.realized_pnl == 50_000
        assert risk_manager.unrealized_pnl == 25_000

    def test_update_positions(self, risk_manager: RiskManager) -> None:
        """Test updating position counters."""
        risk_manager.update_positions(count=3, deployed=2_000_000)

        assert risk_manager.open_position_count == 3
        assert risk_manager.deployed_capital == 2_000_000

    def test_reset_daily(self, risk_manager: RiskManager) -> None:
        """Test daily reset of counters."""
        risk_manager.update_pnl(realized=-30_000, unrealized=-20_000)
        risk_manager.reset_daily()

        assert risk_manager.realized_pnl == 0
        assert risk_manager.unrealized_pnl == 0

    def test_thread_safety(self, risk_manager: RiskManager) -> None:
        """Test that state updates are thread-safe."""
        import threading

        def update_pnl():
            for _ in range(100):
                risk_manager.update_pnl(realized=1000, unrealized=500)

        threads = [threading.Thread(target=update_pnl) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without race conditions
        assert risk_manager.realized_pnl == 1000  # Last value set


class TestRiskCheckResult:
    """Test RiskCheckResult dataclass."""

    def test_approved_result(self) -> None:
        """Test creating an approved result."""
        result = RiskCheckResult(approved=True)

        assert result.approved is True
        assert result.reason is None

    def test_rejected_result(self) -> None:
        """Test creating a rejected result with reason."""
        result = RiskCheckResult(approved=False, reason="Test rejection")

        assert result.approved is False
        assert result.reason == "Test rejection"
