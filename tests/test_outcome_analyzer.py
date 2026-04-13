"""Tests for trade outcome analyzer with MAE/MFE metrics."""

import pytest
from src.memory.outcome_analyzer import TradeOutcome, OutcomeAnalyzer


class TestOutcomeAnalyzer:
    def test_winning_trade_metrics(self):
        outcome = TradeOutcome(
            trade_id="t001",
            strategy="ema_crossover",
            instrument_key="NSE_EQ:RELIANCE",
            direction="BUY",
            entry_price=50000_00,
            exit_price=51000_00,
            stop_loss=49000_00,
            target=52000_00,
            quantity=10,
            realized_pnl=100000_00,  # +1000 INR
            max_adverse_excursion=300_00,  # Price went 3 INR against
            max_favorable_excursion=1200_00,  # Price went 12 INR in favor
            holding_seconds=3600,
        )
        analyzer = OutcomeAnalyzer()
        analysis = analyzer.analyze(outcome)

        assert analysis["was_win"]
        assert analysis["mae_pct"] == pytest.approx(0.6, rel=0.01)  # 30000/5000000 * 100
        assert analysis["mfe_pct"] == pytest.approx(2.4, rel=0.01)  # 120000/5000000 * 100
        assert analysis["efficiency"] == pytest.approx(0.833, rel=0.01)  # 100000/120000

    def test_losing_trade_metrics(self):
        outcome = TradeOutcome(
            trade_id="t002",
            strategy="rsi_reversal",
            instrument_key="NSE_EQ:TCS",
            direction="BUY",
            entry_price=40000_00,
            exit_price=39000_00,
            stop_loss=38500_00,
            target=42000_00,
            quantity=5,
            realized_pnl=-50000_00,  # -500 INR
            max_adverse_excursion=1200_00,
            max_favorable_excursion=300_00,
            holding_seconds=600,
        )
        analyzer = OutcomeAnalyzer()
        analysis = analyzer.analyze(outcome)

        assert not analysis["was_win"]
        assert analysis["efficiency"] < 0  # Negative for losers

    def test_premature_exit_detection(self):
        """Exit captured < 50% of MFE = premature exit."""
        outcome = TradeOutcome(
            trade_id="t003",
            strategy="vwap_breakout",
            instrument_key="NSE_EQ:INFY",
            direction="BUY",
            entry_price=15000_00,
            exit_price=15200_00,
            stop_loss=14800_00,
            target=16000_00,
            quantity=20,
            realized_pnl=40000_00,  # +200 gain
            max_adverse_excursion=100_00,
            max_favorable_excursion=800_00,  # MFE = 800, captured only 200 = 25%
            holding_seconds=1800,
        )
        analyzer = OutcomeAnalyzer()
        analysis = analyzer.analyze(outcome)

        assert analysis["was_win"]
        assert analysis["premature_exit"]
        assert analysis["efficiency"] < 0.5

    def test_stop_loss_hit_detection(self):
        """MAE approximately equals entry - stop_loss = stop hit."""
        outcome = TradeOutcome(
            trade_id="t004",
            strategy="supertrend",
            instrument_key="NSE_EQ:RELIANCE",
            direction="BUY",
            entry_price=50000_00,
            exit_price=49050_00,
            stop_loss=49000_00,
            target=52000_00,
            quantity=10,
            realized_pnl=-95000_00,
            max_adverse_excursion=1000_00,  # ~equal to SL distance of 1000
            max_favorable_excursion=200_00,
            holding_seconds=300,
        )
        analyzer = OutcomeAnalyzer()
        analysis = analyzer.analyze(outcome)

        assert analysis["stop_loss_hit"]
