"""Tests for rule-based mistake classifier."""

import pytest
from src.memory.mistake_classifier import MistakeClassifier, MistakeCategory


class TestMistakeClassifier:
    def _make_analysis(self, **overrides):
        base = {
            "trade_id": "t001",
            "strategy": "ema_crossover",
            "instrument_key": "NSE_EQ:RELIANCE",
            "direction": "BUY",
            "was_win": False,
            "pnl_paisa": -50000_00,
            "mae_pct": 2.0,
            "mfe_pct": 0.5,
            "efficiency": -0.8,
            "premature_exit": False,
            "late_exit": False,
            "stop_loss_hit": True,
            "quick_stop": False,
            "holding_seconds": 1800,
        }
        base.update(overrides)
        return base

    def test_quick_stop_classified_as_stop_too_tight(self):
        analysis = self._make_analysis(
            quick_stop=True,
            stop_loss_hit=True,
            holding_seconds=300,
        )
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="trending_up")
        categories = [m.category for m in mistakes]
        assert MistakeCategory.STOP_TOO_TIGHT in categories

    def test_regime_mismatch(self):
        analysis = self._make_analysis(direction="BUY")
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="trending_down")
        categories = [m.category for m in mistakes]
        assert MistakeCategory.REGIME_MISMATCH in categories

    def test_premature_exit_classified(self):
        analysis = self._make_analysis(was_win=True, premature_exit=True, pnl_paisa=10000_00)
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="trending_up")
        categories = [m.category for m in mistakes]
        assert MistakeCategory.PREMATURE_EXIT in categories

    def test_late_exit_classified(self):
        analysis = self._make_analysis(was_win=True, late_exit=True, pnl_paisa=10000_00)
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="ranging")
        categories = [m.category for m in mistakes]
        assert MistakeCategory.LATE_EXIT in categories

    def test_winning_trade_no_mistakes(self):
        analysis = self._make_analysis(
            was_win=True,
            pnl_paisa=100000_00,
            premature_exit=False,
            late_exit=False,
            stop_loss_hit=False,
        )
        classifier = MistakeClassifier()
        mistakes = classifier.classify(analysis, regime="trending_up")
        assert len(mistakes) == 0

    def test_severity_levels(self):
        classifier = MistakeClassifier()
        analysis = self._make_analysis(direction="BUY", quick_stop=True, holding_seconds=200)
        mistakes = classifier.classify(analysis, regime="trending_down")
        # Should have regime_mismatch (high) and stop_too_tight (medium)
        severities = {m.category: m.severity for m in mistakes}
        assert severities[MistakeCategory.REGIME_MISMATCH] == "high"
        assert severities[MistakeCategory.STOP_TOO_TIGHT] == "medium"
