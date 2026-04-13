"""Pytest fixtures for the trading bot test suite."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import Settings
from src.data.instruments import InstrumentManager
from src.persistence.models import Base
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def settings() -> Settings:
    """Return a Settings instance with test values."""
    return Settings(
        upstox_client_id="test_client_id",
        upstox_client_secret="test_client_secret",
        upstox_redirect_uri="http://localhost:8000/callback",
        trading_mode="paper",
        default_exchange="NSE",
        max_daily_loss=50_000,  # 500 INR in paisa
        max_open_positions=5,
        max_position_size_pct=20,
        capital=10_000_000,  # 1,00,000 INR in paisa
        max_capital_deployment_pct=80,
        consecutive_loss_pause=3,
        pause_minutes=30,
        slippage_pct=0.05,
        database_url="sqlite:///:memory:",
        log_level="DEBUG",
    )


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def db_engine() -> Generator:
    """Create a SQLite in-memory engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    """Create a SQLAlchemy session for testing."""
    SessionLocal = sessionmaker(bind=db_engine, expire_on_commit=False)
    session = SessionLocal()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Risk management fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def risk_manager() -> RiskManager:
    """Return a RiskManager with test capital."""
    return RiskManager(
        capital=10_000_000,  # 1,00,000 INR
        max_daily_loss=50_000,  # 500 INR
        max_open_positions=5,
        max_position_size_pct=20,
        max_capital_deployment_pct=80,
    )


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    """Return a CircuitBreaker with test settings."""
    return CircuitBreaker(
        max_consecutive_losses=3,
        pause_minutes=30,
    )


# ---------------------------------------------------------------------------
# Instrument manager fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def instrument_manager() -> Generator[InstrumentManager, None, None]:
    """Return an InstrumentManager with a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield InstrumentManager(cache_dir=tmp_dir)


@pytest.fixture
def sample_instrument_data() -> pd.DataFrame:
    """Return sample instrument data for testing."""
    data = {
        "segment": ["NSE_EQ", "NSE_EQ", "NSE_FO", "NSE_FO", "NSE_FO", "NSE_FO"],
        "name": ["RELIANCE", "TCS", "BANKNIFTY", "BANKNIFTY", "BANKNIFTY", "BANKNIFTY"],
        "instrument_key": [
            "NSE_EQ|INE002A01018",
            "NSE_EQ|INE467B01029",
            "NSE_FO|BANKNIFTY24MAR46000CE",
            "NSE_FO|BANKNIFTY24MAR46000PE",
            "NSE_FO|BANKNIFTY24MAR46100CE",
            "NSE_FO|BANKNIFTY24MAR46100PE",
        ],
        "trading_symbol": ["RELIANCE", "TCS", "BANKNIFTY24MAR46000CE", "BANKNIFTY24MAR46000PE", "BANKNIFTY24MAR46100CE", "BANKNIFTY24MAR46100PE"],
        "lot_size": [1, 1, 15, 15, 15, 15],
        "tick_size": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "expiry": [None, None, "2024-03-28", "2024-03-28", "2024-03-28", "2024-03-28"],
        "strike_price": [0.0, 0.0, 46000.0, 46000.0, 46100.0, 46100.0],
        "option_type": [None, None, "CE", "PE", "CE", "PE"],
        "underlying_key": [None, None, "NSE_INDEX|Nifty Bank", "NSE_INDEX|Nifty Bank", "NSE_INDEX|Nifty Bank", "NSE_INDEX|Nifty Bank"],
        "underlying_symbol": [None, None, "BANKNIFTY", "BANKNIFTY", "BANKNIFTY", "BANKNIFTY"],
        "exchange_token": ["1234", "5678", "9001", "9002", "9003", "9004"],
        "isin": ["INE002A01018", "INE467B01029", None, None, None, None],
        "instrument_type": ["EQUITY", "EQUITY", "OPTION", "OPTION", "OPTION", "OPTION"],
        "short_name": ["RELIANCE", "TCS", "BANKNIFTY", "BANKNIFTY", "BANKNIFTY", "BANKNIFTY"],
        "freeze_quantity": [0, 0, 900, 900, 900, 900],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_instrument_manager(instrument_manager, sample_instrument_data) -> InstrumentManager:
    """Return an InstrumentManager with pre-loaded sample data."""
    instrument_manager._instruments = sample_instrument_data
    return instrument_manager


# ---------------------------------------------------------------------------
# Paper broker fixture (mock)
# ---------------------------------------------------------------------------
@pytest.fixture
def paper_broker(settings) -> MagicMock:
    """Return a mocked PaperBroker instance."""
    mock_broker = MagicMock()
    mock_broker.settings = settings
    mock_broker.slippage_pct = settings.slippage_pct
    mock_broker.capital = settings.capital
    mock_broker.positions = {}
    mock_broker.orders = []

    # Mock methods
    mock_broker.place_order.return_value = {
        "order_id": "paper_001",
        "status": "COMPLETE",
        "filled_qty": 15,
        "avg_fill_price": 50000,
    }
    mock_broker.get_positions.return_value = []
    mock_broker.get_orders.return_value = []
    mock_broker.cancel_all_orders.return_value = None
    mock_broker.exit_all_positions.return_value = None

    return mock_broker


# ---------------------------------------------------------------------------
# Sample market data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """Return sample OHLC data for indicator testing."""
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")

    # Generate trending data
    trend = np.linspace(45000, 47000, 100)
    noise = np.random.normal(0, 100, 100)

    close = trend + noise
    high = close + np.abs(np.random.normal(50, 20, 100))
    low = close - np.abs(np.random.normal(50, 20, 100))
    open_price = close + np.random.normal(0, 30, 100)
    volume = np.random.randint(1000, 10000, 100)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def sample_strategy_config() -> dict:
    """Return a sample strategy configuration for testing."""
    return {
        "name": "Test_Strategy",
        "description": "Test strategy for unit tests",
        "enabled": True,
        "underlying": {
            "instrument_key": "NSE_INDEX|Nifty Bank",
            "symbol": "BANKNIFTY",
            "segment": "NSE_FO"
        },
        "trading_hours": {
            "start_time": "09:20:00",
            "end_time": "15:10:00",
            "timezone": "Asia/Kolkata"
        },
        "instrument_selection": {
            "type": "options",
            "expiry_type": "weekly_current",
            "strike_selection": "atm",
            "option_types": ["CE", "PE"]
        },
        "entry_sets": [
            {
                "name": "Test_Bullish",
                "signal": "CE",
                "conditions": [
                    {
                        "indicator": "RSI",
                        "comparison": "<",
                        "reference": "constant",
                        "constant_value": 30,
                        "timeframe": "5min",
                        "parameters": {"length": 14}
                    }
                ]
            }
        ],
        "exit_rules": {
            "stop_loss_pct": 20,
            "target_pct": 40,
            "trailing_stop_loss": {
                "enabled": True,
                "activation_pct": 20,
                "trail_pct": 10
            },
            "time_based_exit": "15:10:00",
            "max_holding_minutes": 60
        },
        "position_sizing": {
            "method": "fixed_quantity",
            "quantity": 15,
            "max_risk_per_trade_pct": 2
        }
    }


@pytest.fixture
def temp_strategy_config_file(sample_strategy_config) -> Generator[str, None, None]:
    """Create a temporary strategy config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_strategy_config, f)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Time mocking fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_market_hours():
    """Context manager for mocking market hours."""
    from freezegun import freeze_time

    def _freeze_market_time(time_str: str = "2024-03-15 10:00:00"):
        return freeze_time(time_str, tz_offset=5.5)  # IST offset

    return _freeze_market_time
