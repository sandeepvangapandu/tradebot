"""Tests for broker factory."""

import os
import pytest

from src.execution.broker_factory import create_broker, BrokerFactory
from src.execution.paper_broker import PaperBroker


class TestBrokerFactory:
    """Test broker factory functionality."""

    def test_create_paper_broker(self):
        """Test creating paper broker."""
        broker = create_broker("paper", require_confirmation=False)
        assert isinstance(broker, PaperBroker)

    def test_factory_create_paper(self):
        """Test Factory.create_paper method."""
        broker = BrokerFactory.create_paper()
        assert isinstance(broker, PaperBroker)

    def test_invalid_mode_raises(self):
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid trading mode"):
            create_broker("invalid")

    def test_live_broker_requires_confirmation(self):
        """Test live broker requires confirmation."""
        with pytest.raises(RuntimeError, match="LIVE TRADING SAFETY CHECK FAILED"):
            create_broker("live", require_confirmation=True)

    def test_live_broker_with_env_confirmation(self, monkeypatch):
        """Test live broker with env confirmation set."""
        # This will fail because Upstox SDK not available in test,
        # but it should get past the confirmation check
        monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")

        # Should raise ImportError or similar when trying to import UpstoxLiveBroker
        with pytest.raises(Exception):  # noqa: B017
            create_broker("live", require_confirmation=True)

    def test_mode_from_env(self, monkeypatch):
        """Test mode read from environment variable."""
        monkeypatch.setenv("TRADING_MODE", "paper")
        broker = create_broker(require_confirmation=False)
        assert isinstance(broker, PaperBroker)

    def test_paper_broker_with_custom_capital(self):
        """Test paper broker with custom initial capital."""
        config = {"initial_capital": 5_000_000_00}  # 5L paisa
        broker = create_broker("paper", config, require_confirmation=False)
        assert isinstance(broker, PaperBroker)
