"""Data module for the trading bot.

Provides market data feeds, instrument management, and historical data fetching.
"""

from src.data.bar_builder import BarBuilder, OHLCVBar
from src.data.historical import HistoricalDataFetcher
from src.data.instruments import InstrumentManager
from src.data.portfolio_feed import PortfolioFeed
from src.data.websocket_feed import MarketDataFeed

__all__ = [
    "BarBuilder",
    "OHLCVBar",
    "HistoricalDataFetcher",
    "InstrumentManager",
    "PortfolioFeed",
    "MarketDataFeed",
]
