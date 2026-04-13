"""Tests for Opening Range Breakout (ORB) strategy."""

import pytest
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from src.strategy.conditions import ConditionEvaluator, FilterResult

IST = ZoneInfo("Asia/Kolkata")


def create_mock_15min_data(
    start_date=None,
    orb_high=45000,
    orb_low=44800,
    breakout_price=45100,
    num_bars=30
):
    """Create mock 15-minute OHLCV data for ORB testing.

    Args:
        start_date: Start date for data (defaults to today 9:15 AM)
        orb_high: High of the ORB period
        orb_low: Low of the ORB period
        breakout_price: Price after breakout
        num_bars: Number of 15-min bars to generate

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    if start_date is None:
        start_date = datetime.now(IST).replace(hour=9, minute=15, second=0, microsecond=0)

    # Generate timestamps (15-min intervals)
    timestamps = [start_date + timedelta(minutes=15*i) for i in range(num_bars)]

    data = []
    for i, ts in enumerate(timestamps):
        if i < 1:  # First 15 min (ORB formation)
            open_p = orb_low + 50
            high_p = orb_high
            low_p = orb_low
            close_p = orb_high - 25
        elif i < 2:  # Still in ORB
            open_p = close_p
            high_p = orb_high + 10
            low_p = orb_low - 10
            close_p = orb_high
        else:  # After ORB - breakout
            open_p = breakout_price - 20 if i == 2 else breakout_price
            high_p = breakout_price + 50
            low_p = breakout_price - 20
            close_p = breakout_price + 30

        data.append({
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': 10000 + i * 1000
        })

    df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name='timestamp'))
    return {'BANKNIFTY': df}


def create_mock_breakout_data():
    """Create mock data with breakout above ORB high."""
    return create_mock_15min_data(
        orb_high=45000,
        orb_low=44800,
        breakout_price=45100  # Above ORB high
    )


def create_mock_breakdown_data():
    """Create mock data with breakdown below ORB low."""
    return create_mock_15min_data(
        orb_high=45000,
        orb_low=44800,
        breakout_price=44700  # Below ORB low
    )


def create_mock_no_breakout_data():
    """Create mock data with no breakout (price within range)."""
    return create_mock_15min_data(
        orb_high=45000,
        orb_low=44800,
        breakout_price=44900  # Within ORB range
    )


class TestORBConditionEvaluator:
    """Test ORB condition evaluation."""

    def test_orb_high_low_calculation(self):
        """Test ORB range calculation from first 15 min."""
        evaluator = ConditionEvaluator()
        data = create_mock_15min_data()

        result = evaluator.get_orb_levels(data, symbol='BANKNIFTY', orb_minutes=15)

        assert result is not None
        assert result['orb_high'] > 0
        assert result['orb_low'] > 0
        assert result['orb_high'] > result['orb_low']
        assert 'orb_midpoint' in result
        assert 'orb_range' in result

    def test_orb_breakout_detection(self):
        """Test breakout above ORB high detected."""
        evaluator = ConditionEvaluator()
        data = create_mock_breakout_data()

        result = evaluator.evaluate_orb_condition(
            data,
            direction='breakout',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert isinstance(result, FilterResult)
        assert result.passed is True
        assert result.details['breakout_triggered'] is True
        assert 'orb_high' in result.details
        assert 'orb_low' in result.details
        assert 'suggested_stop_loss' in result.details

    def test_orb_breakdown_detection(self):
        """Test breakdown below ORB low detected."""
        evaluator = ConditionEvaluator()
        data = create_mock_breakdown_data()

        result = evaluator.evaluate_orb_condition(
            data,
            direction='breakdown',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert isinstance(result, FilterResult)
        assert result.passed is True
        assert result.details['breakdown_triggered'] is True

    def test_no_breakout_when_price_in_range(self):
        """Test no breakout when price within ORB range."""
        evaluator = ConditionEvaluator()
        data = create_mock_no_breakout_data()

        result = evaluator.evaluate_orb_condition(
            data,
            direction='breakout',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert isinstance(result, FilterResult)
        assert result.passed is False
        assert result.details['breakout_triggered'] is False

    def test_invalid_direction(self):
        """Test that invalid direction returns error."""
        evaluator = ConditionEvaluator()
        data = create_mock_breakout_data()

        result = evaluator.evaluate_orb_condition(
            data,
            direction='invalid',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert isinstance(result, FilterResult)
        assert result.passed is False
        assert 'error' in result.details

    def test_missing_symbol(self):
        """Test that missing symbol returns error."""
        evaluator = ConditionEvaluator()
        data = create_mock_breakout_data()

        result = evaluator.evaluate_orb_condition(
            data,
            direction='breakout',
            symbol='NONEXISTENT',
            orb_minutes=15
        )

        assert isinstance(result, FilterResult)
        assert result.passed is False
        assert 'error' in result.details

    def test_empty_data(self):
        """Test that empty data returns error."""
        evaluator = ConditionEvaluator()

        result = evaluator.evaluate_orb_condition(
            {},
            direction='breakout',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert isinstance(result, FilterResult)
        assert result.passed is False
        assert 'error' in result.details

    def test_still_in_orb_formation(self):
        """Test that condition fails during ORB formation period."""
        evaluator = ConditionEvaluator()

        # Create data only within ORB window
        start = datetime.now(IST).replace(hour=9, minute=15)
        timestamps = [start + timedelta(minutes=5*i) for i in range(3)]

        data = []
        for ts in timestamps:
            data.append({
                'open': 44900,
                'high': 45000,
                'low': 44800,
                'close': 44950,
                'volume': 10000
            })

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name='timestamp'))

        result = evaluator.evaluate_orb_condition(
            {'BANKNIFTY': df},
            direction='breakout',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert isinstance(result, FilterResult)
        assert result.passed is False
        assert 'Still in ORB formation' in result.details.get('error', '')

    def test_suggested_stop_loss_at_midpoint(self):
        """Test that suggested stop loss is at ORB midpoint."""
        evaluator = ConditionEvaluator()
        data = create_mock_breakout_data()

        result = evaluator.evaluate_orb_condition(
            data,
            direction='breakout',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        orb_high = result.details['orb_high']
        orb_low = result.details['orb_low']
        expected_midpoint = (orb_high + orb_low) / 2

        assert result.details['suggested_stop_loss'] == expected_midpoint

    def test_orb_range_calculation(self):
        """Test that ORB range is calculated correctly."""
        evaluator = ConditionEvaluator()
        data = create_mock_15min_data(orb_high=45000, orb_low=44800)

        result = evaluator.get_orb_levels(data, symbol='BANKNIFTY', orb_minutes=15)

        expected_range = 45000 - 44800
        assert result['orb_range'] == expected_range


class TestORBScoreCalculation:
    """Test ORB score/score calculation."""

    def test_breakout_score_is_one_when_triggered(self):
        """Test that score is 1.0 when breakout is triggered."""
        evaluator = ConditionEvaluator()
        data = create_mock_breakout_data()

        result = evaluator.evaluate_orb_condition(
            data,
            direction='breakout',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert result.score == 1.0

    def test_score_between_zero_and_one_when_not_triggered(self):
        """Test that score is between 0 and 1 when not triggered."""
        evaluator = ConditionEvaluator()
        data = create_mock_no_breakout_data()

        result = evaluator.evaluate_orb_condition(
            data,
            direction='breakout',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert 0.0 <= result.score < 1.0


class TestORBEdgeCases:
    """Test ORB edge cases."""

    def test_orb_with_no_bars_in_window(self):
        """Test handling when no bars in ORB window."""
        evaluator = ConditionEvaluator()

        # Create data starting after ORB window
        start = datetime.now(IST).replace(hour=10, minute=0)  # After 9:30
        timestamps = [start + timedelta(minutes=15*i) for i in range(5)]

        data = []
        for ts in timestamps:
            data.append({
                'open': 44900,
                'high': 45000,
                'low': 44800,
                'close': 44950,
                'volume': 10000
            })

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name='timestamp'))

        result = evaluator.evaluate_orb_condition(
            {'BANKNIFTY': df},
            direction='breakout',
            symbol='BANKNIFTY',
            orb_minutes=15
        )

        assert isinstance(result, FilterResult)
        assert result.passed is False
        assert 'No bars in ORB window' in result.details.get('error', '')

    def test_orb_with_single_bar(self):
        """Test ORB calculation with single bar."""
        evaluator = ConditionEvaluator()

        start = datetime.now(IST).replace(hour=9, minute=15)
        timestamps = [start, start + timedelta(minutes=30)]

        data = [
            {'open': 44900, 'high': 45000, 'low': 44800, 'close': 44950, 'volume': 10000},
            {'open': 44950, 'high': 45100, 'low': 44900, 'close': 45050, 'volume': 15000}
        ]

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name='timestamp'))

        result = evaluator.get_orb_levels({'BANKNIFTY': df}, symbol='BANKNIFTY', orb_minutes=15)

        assert result is not None
        assert result['orb_high'] == 45000
        assert result['orb_low'] == 44800

    def test_custom_orb_minutes(self):
        """Test ORB with custom time window."""
        evaluator = ConditionEvaluator()

        start = datetime.now(IST).replace(hour=9, minute=15)
        timestamps = [start + timedelta(minutes=5*i) for i in range(10)]

        data = []
        for i, ts in enumerate(timestamps):
            if i < 2:  # First 10 minutes (custom ORB)
                high_p = 45000
                low_p = 44800
            else:
                high_p = 45100
                low_p = 44900

            data.append({
                'open': 44900,
                'high': high_p,
                'low': low_p,
                'close': 44950,
                'volume': 10000
            })

        df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name='timestamp'))

        result = evaluator.get_orb_levels(
            {'BANKNIFTY': df},
            symbol='BANKNIFTY',
            orb_minutes=10  # Custom 10-min ORB
        )

        assert result is not None
        assert result['orb_high'] == 45000
        assert result['orb_low'] == 44800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
