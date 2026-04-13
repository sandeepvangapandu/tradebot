"""RSI Divergence Multi-Timeframe Strategy.

Implements a reversal trading strategy based on RSI divergence patterns
with multi-timeframe confirmation for trading BankNifty options.

Strategy Logic:
- Bullish Divergence: Price makes lower low, RSI makes higher low
- Bearish Divergence: Price makes higher high, RSI makes lower high
- Requires higher timeframe trend alignment
- Entry near support/resistance levels for better R:R

All prices are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps in IST (Asia/Kolkata).
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from src.strategy.builder import (
    Condition,
    EntrySet,
    StrategyConfig,
)
from src.strategy.conditions import ConditionEvaluator
from src.strategy.engine import Signal

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class RSIDivergenceSignal:
    """Internal signal structure for RSI divergence detection."""

    signal_type: str  # "bullish" or "bearish"
    strength: float  # 0.0 - 1.0
    price_swing: Optional[float] = None
    rsi_swing: Optional[float] = None
    htf_aligned: bool = False  # Higher timeframe alignment
    near_support_resistance: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(IST))


class RSIDivergenceMTFStrategy:
    """RSI Divergence Multi-Timeframe Reversal Strategy.

    This strategy detects RSI divergence patterns on the primary timeframe (5min)
    and confirms with higher timeframe trend (15min) for higher probability
    reversal trades on BankNifty options.

    Entry Conditions - Bullish Divergence (CE):
    1. RSI Divergence detected (bullish) on 5min
    2. RSI < 40 (oversold zone)
    3. Price > EMA 20 on 15min (higher timeframe uptrend)
    4. Near support level (< 0.3% distance)

    Entry Conditions - Bearish Divergence (PE):
    1. RSI Divergence detected (bearish) on 5min
    2. RSI > 60 (overbought zone)
    3. Price < EMA 20 on 15min (higher timeframe downtrend)
    4. Near resistance level (< 0.3% distance)

    Exit Rules:
    - Stop Loss: 25% of premium
    - Target: 50% of premium (2:1 R:R)
    - Trailing Stop: Activates at 25% profit, trails at 10%

    Attributes:
        config: Strategy configuration loaded from JSON
        evaluator: Condition evaluator for entry conditions
        primary_timeframe: Primary entry timeframe (default "5min")
        higher_timeframe: Higher timeframe for trend (default "15min")
        lower_timeframe: Lower timeframe for fine-tuning (default "3min")
    """

    def __init__(self, config: StrategyConfig):
        """Initialize the RSI Divergence strategy.

        Args:
            config: Validated StrategyConfig instance
        """
        self.config = config
        self.evaluator = ConditionEvaluator()

        # Timeframe configuration
        self.primary_timeframe = config.timeframes.get("primary", "5min")
        self.higher_timeframe = config.timeframes.get("higher", "15min")
        self.lower_timeframe = config.timeframes.get("lower", "3min")

        # Strategy parameters
        self.divergence_lookback = config.params.get("divergence_lookback", 10)
        self.min_swing_points = config.params.get("min_swing_points", 2)
        self.htf_trend_alignment = config.params.get("htf_trend_alignment", True)
        self.rsi_oversold = config.params.get("rsi_oversold", 40)
        self.rsi_overbought = config.params.get("rsi_overbought", 60)
        self.proximity_threshold = config.params.get("proximity_threshold_pct", 0.3)

        # Exit rules
        self.stop_loss_pct = config.exit_rules.stop_loss_pct or 25
        self.target_pct = config.exit_rules.target_pct or 50
        self.trailing_config = config.exit_rules.trailing_stop_loss or {}

        # Risk management
        self.max_open_positions = config.risk_management.get("max_open_positions", 2)
        self.max_trades_per_day = config.risk_management.get("max_trades_per_day", 3)
        self.require_htf_alignment = config.risk_management.get("require_htf_alignment", True)

        # Position sizing
        self.position_quantity = config.position_sizing.quantity or 2

        # State tracking
        self._signals_generated_today = 0
        self._last_signal_date: Optional[datetime.date] = None
        self._active_signals: list[RSIDivergenceSignal] = []

        logger.info(
            f"RSI Divergence MTF Strategy initialized: {config.name}"
        )
        logger.info(
            f"  Timeframes: {self.primary_timeframe} (primary), "
            f"{self.higher_timeframe} (higher), {self.lower_timeframe} (lower)"
        )
        logger.info(
            f"  RSI thresholds: oversold < {self.rsi_oversold}, "
            f"overbought > {self.rsi_overbought}"
        )

    def evaluate_entry_sets(
        self,
        bars_by_timeframe: dict[str, dict[str, pd.DataFrame]],
        symbol: str,
    ) -> list[Signal]:
        """Evaluate all entry sets and generate signals.

        Args:
            bars_by_timeframe: Dict mapping timeframe to bars dict
                e.g., {"5min": {"BANKNIFTY": df}, "15min": {...}}
            symbol: Trading symbol

        Returns:
            List of Signal objects for triggered entry sets
        """
        signals = []

        # Reset daily counters if new day
        current_date = datetime.now(IST).date()
        if self._last_signal_date != current_date:
            self._signals_generated_today = 0
            self._last_signal_date = current_date

        # Check max trades per day limit
        if (
            self.max_trades_per_day
            and self._signals_generated_today >= self.max_trades_per_day
        ):
            logger.debug(
                f"Max trades per day ({self.max_trades_per_day}) reached"
            )
            return signals

        # Get primary timeframe bars
        primary_bars = bars_by_timeframe.get(self.primary_timeframe, {})
        if symbol not in primary_bars:
            logger.warning(f"No primary timeframe data for {symbol}")
            return signals

        # Get higher timeframe bars for trend alignment
        higher_bars = bars_by_timeframe.get(self.higher_timeframe, {})

        for entry_set in self.config.entry_sets:
            if self._evaluate_entry_set(entry_set, bars_by_timeframe, symbol):
                signal = self._create_signal(entry_set, symbol, primary_bars)
                if signal:
                    signals.append(signal)
                    self._signals_generated_today += 1

        return signals

    def _evaluate_entry_set(
        self,
        entry_set: EntrySet,
        bars_by_timeframe: dict[str, dict[str, pd.DataFrame]],
        symbol: str,
    ) -> bool:
        """Evaluate a single entry set's conditions.

        Args:
            entry_set: Entry set to evaluate
            bars_by_timeframe: Bars organized by timeframe
            symbol: Trading symbol

        Returns:
            True if all conditions are met
        """
        all_conditions_met = True

        for condition in entry_set.conditions:
            # Determine which timeframe to use for this condition
            timeframe = condition.timeframe or self.primary_timeframe
            bars = bars_by_timeframe.get(timeframe, {})

            if not bars or symbol not in bars:
                logger.debug(f"No data for {symbol} on {timeframe}")
                return False

            # Evaluate the condition
            condition_met = self.evaluator.evaluate(condition, bars, symbol)

            if not condition_met:
                all_conditions_met = False
                logger.debug(
                    f"Entry set '{entry_set.name}' condition failed: "
                    f"{condition.indicator} {condition.comparison} "
                    f"{condition.against or condition.value}"
                )
                break

        if all_conditions_met:
            logger.info(
                f"Entry set '{entry_set.name}' ALL conditions met for {symbol}"
            )

        return all_conditions_met

    def _create_signal(
        self,
        entry_set: EntrySet,
        symbol: str,
        bars: dict[str, pd.DataFrame],
    ) -> Optional[Signal]:
        """Create a trading signal from entry set.

        Args:
            entry_set: Triggered entry set
            symbol: Trading symbol
            bars: Price data for calculating levels

        Returns:
            Signal object or None if invalid
        """
        if symbol not in bars:
            return None

        df = bars[symbol]
        if df.empty:
            return None

        current_price = int(df["close"].iloc[-1])

        # Determine signal type from entry set
        signal_type = self._map_signal_type(entry_set.signal)
        if signal_type is None:
            logger.error(f"Unknown signal type: {entry_set.signal}")
            return None

        # Calculate stop loss and target based on option premium
        # For options, SL and target are percentages of entry premium
        # We'll use the underlying price as reference for metadata

        # Create the signal
        signal = Signal(
            strategy_name=self.config.name,
            set_name=entry_set.name,
            instrument_key=self._get_instrument_key(signal_type),
            signal_type=signal_type,
            quantity=self.position_quantity,
            price=None,  # Market order
            stop_loss=None,  # Will be set by position manager based on option premium
            target=None,  # Will be set by position manager
            order_type="MARKET",
            product_type="I",  # Intraday
            timestamp=datetime.now(IST),
            underlying=symbol,
            timeframe=self.primary_timeframe,
            metadata={
                "entry_price_underlying": current_price,
                "stop_loss_pct": self.stop_loss_pct,
                "target_pct": self.target_pct,
                "trailing_config": self.trailing_config,
                "divergence_lookback": self.divergence_lookback,
                "htf_aligned": self.require_htf_alignment,
            },
        )

        logger.info(
            f"Generated {signal_type.value} signal for {symbol} "
            f"via entry set '{entry_set.name}'"
        )

        return signal

    def _map_signal_type(self, signal_str: str) -> Optional[Any]:
        """Map string signal type to SignalType enum.

        Args:
            signal_str: Signal string from config (e.g., "CE", "PE")

        Returns:
            SignalType enum value or None
        """
        from src.strategy.builder import SignalType

        mapping = {
            "CE": SignalType.BUY_CE,
            "PE": SignalType.BUY_PE,
            "BUY": SignalType.BUY,
            "SELL": SignalType.SELL,
        }
        return mapping.get(signal_str.upper())

    def _get_instrument_key(self, signal_type: Any) -> str:
        """Get instrument key for the signal.

        For options strategies, this would typically be resolved by the
        instrument selection logic based on ATM/ITM/OTM preference.

        Args:
            signal_type: SignalType enum value

        Returns:
            Instrument key string
        """
        # This is a placeholder - actual instrument selection happens
        # in the strategy engine based on instrument_selection config
        underlying = self.config.underlying or {}
        symbol = underlying.get("symbol", "BANKNIFTY")
        segment = underlying.get("segment", "NSE_FO")

        # Return underlying key - actual option instrument will be selected
        # by the instrument selection module
        return f"{segment}|{symbol}"

    def is_trading_time(self, current_time: Optional[time] = None) -> bool:
        """Check if current time is within trading hours.

        Args:
            current_time: Time to check (default: now)

        Returns:
            True if within trading hours
        """
        if current_time is None:
            current_time = datetime.now(IST).time()

        start = self.config.trading_hours.start
        end = self.config.trading_hours.end

        return start <= current_time <= end

    def get_required_timeframes(self) -> list[str]:
        """Get list of timeframes required by this strategy.

        Returns:
            List of timeframe strings
        """
        return [
            self.primary_timeframe,
            self.higher_timeframe,
            self.lower_timeframe,
        ]

    def get_exit_levels(
        self,
        entry_price: float,
        signal_type: str,
    ) -> dict[str, float]:
        """Calculate exit levels for a position.

        Args:
            entry_price: Entry price (option premium in PAISA)
            signal_type: "CE" or "PE"

        Returns:
            Dictionary with stop_loss, target, and trailing levels
        """
        # For options, SL and target are percentages of premium
        stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
        target = entry_price * (1 + self.target_pct / 100)

        result = {
            "stop_loss": stop_loss,
            "target": target,
            "entry_price": entry_price,
        }

        # Add trailing stop configuration if enabled
        if self.trailing_config.get("enabled", False):
            result["trailing_enabled"] = True
            result["trailing_activation"] = entry_price * (
                1 + self.trailing_config.get("activation_pct", 25) / 100
            )
            result["trailing_step"] = self.trailing_config.get("trail_pct", 10)

        return result

    def reset_daily_state(self) -> None:
        """Reset daily state counters."""
        self._signals_generated_today = 0
        self._last_signal_date = datetime.now(IST).date()
        self._active_signals.clear()
        logger.debug("RSI Divergence strategy daily state reset")


def create_rsi_divergence_strategy(config_path: str) -> RSIDivergenceMTFStrategy:
    """Factory function to create RSI Divergence strategy from config file.

    Args:
        config_path: Path to strategy JSON config file

    Returns:
        Initialized RSIDivergenceMTFStrategy

    Example:
        >>> strategy = create_rsi_divergence_strategy(
        ...     "config/strategies/rsi_divergence_mtf_banknifty.json"
        ... )
        >>> print(strategy.config.name)
        'RSI_Divergence_MTF_BankNifty'
    """
    from src.strategy.builder import StrategyBuilder

    config = StrategyBuilder.load_strategy(config_path)
    return RSIDivergenceMTFStrategy(config)
