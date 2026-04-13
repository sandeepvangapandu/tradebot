"""Graceful degradation utilities for the trading bot.

This module provides decorators and utilities for handling component failures
gracefully, allowing the system to continue operating with reduced functionality
rather than crashing.

All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


def graceful_degrade(
    default_return: Any = None,
    log_level: str = "warning",
    include_self: bool = True,
) -> Callable[[F], F]:
    """Decorator that catches exceptions and returns a default value.

    This decorator wraps functions to catch any exceptions and return a
    predefined default value instead of crashing. It logs the failure
    at the specified log level for monitoring.

    Args:
        default_return: The value to return if the function fails.
            Should match the expected return type of the decorated function.
        log_level: Log level for failure messages. Options: "debug", "info",
            "warning", "error", "critical". Default is "warning".
        include_self: Whether to include 'self' in the function name when
            logging (for class methods). Default is True.

    Returns:
        Decorated function that returns default_return on exception.

    Example:
        @graceful_degrade(default_return={"score": 50, "confidence": 0.3})
        def analyze(self, data: pd.DataFrame) -> dict:
            # Analysis logic that might fail
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Determine function name for logging
                if include_self and args and hasattr(args[0], "__class__"):
                    # It's a method call, include class name
                    class_name = args[0].__class__.__name__
                    func_name = f"{class_name}.{func.__name__}"
                else:
                    func_name = func.__name__

                # Log at appropriate level
                log_message = f"{func_name} failed: {type(e).__name__}: {e}"
                if log_level == "debug":
                    logger.debug(log_message)
                elif log_level == "info":
                    logger.info(log_message)
                elif log_level == "warning":
                    logger.warning(log_message)
                elif log_level == "error":
                    logger.error(log_message)
                elif log_level == "critical":
                    logger.critical(log_message)
                else:
                    logger.warning(log_message)

                return default_return

        # Mark the function as having graceful degradation
        wrapper._graceful_degrade = True  # type: ignore
        wrapper._default_return = default_return  # type: ignore

        return wrapper  # type: ignore

    return decorator


# Default fallback values for common analyzer return types

# Trend/Momentum analyzers
DEFAULT_TREND_RETURN = {"score": 50, "confidence": 0.3, "regime": "unknown"}
DEFAULT_MOMENTUM_RETURN = {"score": 50, "confidence": 0.3, "regime": "neutral"}

# Volume analyzer
DEFAULT_VOLUME_RETURN = {"confirmed": True, "score": 50}

# Options analyzer
DEFAULT_OPTIONS_RETURN = {"signal": "neutral", "score": 50}

# Support/Resistance analyzer
DEFAULT_SR_RETURN = {"near_support": False, "near_resistance": False, "score": 50}

# Market Regime analyzer
DEFAULT_REGIME_RETURN = {"regime": "unknown", "confidence": 0.5}

# Correlation analyzer
DEFAULT_CORRELATION_RETURN = {"diversified": True, "score": 50}

# Event Calendar analyzer
DEFAULT_EVENT_RETURN = {"avoid": False, "score": 50}

# Time of Day analyzer
DEFAULT_TIME_RETURN = {"score": 60}

# Candle Patterns analyzer
DEFAULT_PATTERN_RETURN = {"pattern": None, "score": 50}


def is_degraded_result(result: Any, default_return: Any) -> bool:
    """Check if a result is the default/degraded value.

    Args:
        result: The actual result returned by an analyzer.
        default_return: The expected default return value.

    Returns:
        True if the result matches the default (indicating degradation).
    """
    if result is default_return:
        return True

    # For dict results, check if key fields match defaults
    if isinstance(result, dict) and isinstance(default_return, dict):
        # Check if all default keys have default values
        for key, default_val in default_return.items():
            if key not in result or result[key] != default_val:
                return False
        return True

    return result == default_return


def get_degraded_analyzers(components: list[Any]) -> list[str]:
    """Extract list of analyzer names that returned degraded results.

    Args:
        components: List of analysis components to check.

    Returns:
        List of analyzer names that appear to be degraded.
    """
    degraded = []

    for component in components:
        # Check if component has a name and appears degraded
        if hasattr(component, "name") and hasattr(component, "confidence"):
            # Low confidence often indicates degraded/fallback result
            if component.confidence < 30:
                degraded.append(component.name)
        elif hasattr(component, "score") and hasattr(component, "name"):
            # Score of exactly 50 with low confidence suggests default
            if component.score == 50 and getattr(component, "confidence", 100) < 50:
                degraded.append(component.name)

    return degraded
