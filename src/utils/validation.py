"""Input validation layer for the trading bot.

This module provides comprehensive validation for all inputs to prevent crashes
from malformed data. It validates data from:
- Upstox WebSocket (market data)
- User strategy configs (YAML/JSON)
- Internal signal generation

All inputs must be validated before processing.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field, validator

from src.utils.logger import logger

# Constants for Indian markets
MAX_SYMBOL_LENGTH = 25
MAX_PRICE_PAISA = 1_000_000 * 100  # 10,00,000 INR in paisa
MAX_QUANTITY = 10_000
VALID_ORDER_TYPES = {"MARKET", "LIMIT", "SL", "SL-M"}
VALID_SIGNAL_TYPES = {"BUY_CE", "BUY_PE", "BUY", "SELL"}
VALID_SIDES = {"BUY", "SELL"}
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9\-&|_ ]+$")

T = TypeVar("T")


class ValidationError(Exception):
    """Exception raised when validation fails in strict mode."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid: Whether the validation passed.
        errors: List of error messages if validation failed.
        warnings: List of warning messages for non-critical issues.
        sanitized_value: The sanitized/cleaned value (if applicable).
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_value: Any = None

    def __bool__(self) -> bool:
        """Allow boolean evaluation of validation result."""
        return self.is_valid


class OrderValidationModel(BaseModel):
    """Pydantic model for order validation."""

    symbol: str = Field(..., min_length=1, max_length=MAX_SYMBOL_LENGTH)
    quantity: int = Field(..., gt=0, le=MAX_QUANTITY)
    side: str = Field(...)
    order_type: str = Field(default="MARKET")
    price: int | None = Field(default=None, ge=0, le=MAX_PRICE_PAISA)
    trigger_price: int | None = Field(default=None, ge=0, le=MAX_PRICE_PAISA)
    product_type: str = Field(default="MIS")

    @validator("side")
    def validate_side(cls, v: str) -> str:
        """Validate order side."""
        v_upper = v.upper()
        if v_upper not in VALID_SIDES:
            raise ValueError(f"side must be one of {VALID_SIDES}, got '{v}'")
        return v_upper

    @validator("order_type")
    def validate_order_type(cls, v: str) -> str:
        """Validate order type."""
        v_upper = v.upper()
        if v_upper not in VALID_ORDER_TYPES:
            raise ValueError(f"order_type must be one of {VALID_ORDER_TYPES}, got '{v}'")
        return v_upper

    @validator("price", always=True)
    def validate_price_for_order_type(cls, v: int | None, values: dict) -> int | None:
        """Validate price is provided for LIMIT and SL orders."""
        order_type = values.get("order_type", "MARKET")
        if order_type in ("LIMIT", "SL") and v is None:
            raise ValueError(f"price is required for {order_type} orders")
        return v

    @validator("trigger_price", always=True)
    def validate_trigger_for_sl(cls, v: int | None, values: dict) -> int | None:
        """Validate trigger_price is provided for SL orders."""
        order_type = values.get("order_type", "MARKET")
        if order_type in ("SL", "SL-M") and v is None:
            raise ValueError(f"trigger_price is required for {order_type} orders")
        return v


class SignalValidationModel(BaseModel):
    """Pydantic model for trading signal validation."""

    symbol: str = Field(..., min_length=1, max_length=MAX_SYMBOL_LENGTH)
    signal_type: str = Field(...)
    entry_price: int = Field(..., gt=0, le=MAX_PRICE_PAISA)
    stop_loss: int = Field(..., gt=0, le=MAX_PRICE_PAISA)
    target: int = Field(..., gt=0, le=MAX_PRICE_PAISA)
    quantity: int = Field(..., gt=0, le=MAX_QUANTITY)
    strategy_name: str = Field(..., min_length=1)

    @validator("signal_type")
    def validate_signal_type(cls, v: str) -> str:
        """Validate signal type."""
        v_upper = v.upper()
        if v_upper not in VALID_SIGNAL_TYPES:
            raise ValueError(f"signal_type must be one of {VALID_SIGNAL_TYPES}, got '{v}'")
        return v_upper

    @validator("target")
    def validate_risk_reward(cls, v: int, values: dict) -> int:
        """Validate stop_loss < entry_price < target for BUY signals."""
        entry = values.get("entry_price")
        sl = values.get("stop_loss")
        signal_type = values.get("signal_type", "").upper()

        if entry is None or sl is None:
            return v

        if "BUY" in signal_type:
            if not (sl < entry < v):
                raise ValueError(
                    f"For BUY signals, stop_loss ({sl}) < entry_price ({entry}) < target ({v}) required"
                )
        elif "SELL" in signal_type:
            if not (v < entry < sl):
                raise ValueError(
                    f"For SELL signals, target ({v}) < entry_price ({entry}) < stop_loss ({sl}) required"
                )
        return v


class ConfigValidationModel(BaseModel):
    """Pydantic model for configuration validation."""

    capital: int = Field(default=1_000_000, gt=0, le=1_000_000_000)
    max_risk_per_trade: float = Field(default=0.01, gt=0, le=0.5)
    daily_loss_limit: float = Field(default=0.03, gt=0, le=1.0)
    max_positions: int = Field(default=5, gt=0, le=100)
    max_capital_deployment: float = Field(default=0.8, gt=0, le=1.0)
    per_instrument_limit: float = Field(default=0.2, gt=0, le=1.0)
    log_level: str = Field(default="INFO")

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got '{v}'")
        return v_upper

    @validator("per_instrument_limit")
    def validate_per_instrument_limit(cls, v: float, values: dict) -> float:
        """Validate per_instrument_limit doesn't exceed max_capital_deployment."""
        max_deploy = values.get("max_capital_deployment", 1.0)
        if v > max_deploy:
            raise ValueError(
                f"per_instrument_limit ({v}) cannot exceed max_capital_deployment ({max_deploy})"
            )
        return v


class InputValidator:
    """Comprehensive input validator for trading bot data.

    This class provides methods to validate various types of inputs including
    symbols, prices, quantities, order parameters, and trading signals.

    All methods return a ValidationResult that contains:
    - is_valid: Boolean indicating if validation passed
    - errors: List of error messages (empty if valid)
    - warnings: List of warning messages for non-critical issues
    - sanitized_value: Cleaned/transformed value (if applicable)
    """

    @staticmethod
    def validate_symbol(symbol: Any) -> ValidationResult:
        """Validate a trading symbol.

        Args:
            symbol: The symbol to validate (should be non-empty string).

        Returns:
            ValidationResult with validation status and any errors.
        """
        errors: list[str] = []
        warnings: list[str] = []
        sanitized: str | None = None

        # Check None
        if symbol is None:
            errors.append("Symbol cannot be None")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check type
        if not isinstance(symbol, str):
            errors.append(f"Symbol must be a string, got {type(symbol).__name__}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Sanitize
        sanitized = symbol.strip().upper()

        # Check empty after strip
        if not sanitized:
            errors.append("Symbol cannot be empty or whitespace only")
            return ValidationResult(
                is_valid=False, errors=errors, warnings=warnings, sanitized_value=sanitized
            )

        # Check length
        if len(sanitized) > MAX_SYMBOL_LENGTH:
            errors.append(f"Symbol '{sanitized}' exceeds max length of {MAX_SYMBOL_LENGTH}")

        # Check pattern
        if not SYMBOL_PATTERN.match(sanitized):
            errors.append(
                f"Symbol '{sanitized}' contains invalid characters. "
                f"Only uppercase letters, numbers, hyphens, and ampersands are allowed"
            )

        # Warning for unusual patterns
        if "-" in sanitized and sanitized.count("-") > 2:
            warnings.append(f"Symbol '{sanitized}' has multiple hyphens - verify this is correct")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized,
        )

    @staticmethod
    def validate_price(
        price: Any,
        field_name: str = "price",
        allow_zero: bool = False,
        allow_negative: bool = False,
    ) -> ValidationResult:
        """Validate a price value.

        Args:
            price: The price to validate (must be numeric).
            field_name: Name of the field for error messages.
            allow_zero: Whether to allow zero as a valid price.
            allow_negative: Whether to allow negative prices.

        Returns:
            ValidationResult with validation status and any errors.
        """
        errors: list[str] = []
        warnings: list[str] = []
        sanitized: int | None = None

        # Check None
        if price is None:
            errors.append(f"{field_name} cannot be None")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Convert to numeric
        numeric_price = sanitize_numeric(price)
        if numeric_price is None:
            errors.append(
                f"{field_name} must be numeric (int, float, or numeric string), "
                f"got {type(price).__name__}: {price!r}"
            )
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        sanitized = numeric_price

        # Check bounds
        if not allow_negative and numeric_price < 0:
            errors.append(f"{field_name} cannot be negative: {numeric_price}")
        elif not allow_zero and numeric_price == 0:
            errors.append(f"{field_name} must be positive (non-zero)")

        if numeric_price > MAX_PRICE_PAISA:
            errors.append(
                f"{field_name} ({numeric_price} paisa) exceeds maximum allowed ({MAX_PRICE_PAISA} paisa)"
            )

        # Warning for very small prices
        if 0 < numeric_price < 100:  # Less than 1 rupee
            warnings.append(f"{field_name} is very small ({numeric_price} paisa = {numeric_price/100:.2f} INR)")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized,
        )

    @staticmethod
    def validate_quantity(qty: Any, lot_size: int = 1) -> ValidationResult:
        """Validate a quantity value.

        Args:
            qty: The quantity to validate (must be positive integer).
            lot_size: The lot size multiplier (default 1 for equity).

        Returns:
            ValidationResult with validation status and any errors.
        """
        errors: list[str] = []
        warnings: list[str] = []
        sanitized: int | None = None

        # Check None
        if qty is None:
            errors.append("Quantity cannot be None")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check lot_size
        if not isinstance(lot_size, int) or lot_size < 1:
            errors.append(f"lot_size must be a positive integer, got {lot_size}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Convert to integer
        numeric_qty = sanitize_numeric(qty)
        if numeric_qty is None:
            errors.append(
                f"Quantity must be numeric (int or numeric string), got {type(qty).__name__}: {qty!r}"
            )
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        sanitized = numeric_qty

        # Check positive
        if numeric_qty <= 0:
            errors.append(f"Quantity must be positive, got {numeric_qty}")

        # Check max quantity (fat-finger protection)
        if numeric_qty > MAX_QUANTITY:
            errors.append(
                f"Quantity ({numeric_qty}) exceeds maximum allowed ({MAX_QUANTITY}) - "
                "possible fat-finger error"
            )

        # Check lot size multiple
        if numeric_qty > 0 and numeric_qty % lot_size != 0:
            errors.append(
                f"Quantity ({numeric_qty}) must be a multiple of lot_size ({lot_size})"
            )

        # Warning for large quantities
        if numeric_qty > 1000:
            warnings.append(f"Quantity ({numeric_qty}) is large - verify this is intentional")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized,
        )

    @staticmethod
    def validate_order_params(params: dict) -> ValidationResult:
        """Validate order parameters dictionary.

        Required fields: symbol, quantity, side
        Optional fields: order_type, price, trigger_price, product_type

        Args:
            params: Dictionary containing order parameters.

        Returns:
            ValidationResult with validation status and any errors.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check None
        if params is None:
            errors.append("Order params cannot be None")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check type
        if not isinstance(params, dict):
            errors.append(f"Order params must be a dict, got {type(params).__name__}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check required fields
        required_fields = {"symbol", "quantity", "side"}
        missing_fields = required_fields - set(params.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {sorted(missing_fields)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Validate using Pydantic model
        try:
            OrderValidationModel(**params)
        except Exception as e:
            # Extract validation errors from Pydantic
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                # Clean up pydantic error message
                lines = error_msg.split("\n")
                for line in lines[1:] if len(lines) > 1 else lines:
                    if line.strip():
                        errors.append(line.strip())
            else:
                errors.append(f"Order validation failed: {error_msg}")

        # Additional validation for price/trigger consistency
        order_type = str(params.get("order_type", "MARKET")).upper()
        price = params.get("price")
        trigger_price = params.get("trigger_price")

        if order_type in ("SL", "SL-M") and trigger_price is not None and price is not None:
            side = str(params.get("side", "")).upper()
            trigger_val = sanitize_numeric(trigger_price)
            price_val = sanitize_numeric(price)

            if trigger_val is not None and price_val is not None:
                if side == "BUY" and order_type == "SL" and price_val > trigger_val:
                    warnings.append(
                        f"For BUY SL order, limit price ({price_val}) should typically be >= trigger ({trigger_val})"
                    )
                elif side == "SELL" and order_type == "SL" and price_val < trigger_val:
                    warnings.append(
                        f"For SELL SL order, limit price ({price_val}) should typically be <= trigger ({trigger_val})"
                    )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_signal(signal: dict) -> ValidationResult:
        """Validate a trading signal dictionary.

        Required fields from Signal dataclass:
        - symbol: Trading symbol
        - signal_type: One of BUY_CE, BUY_PE, BUY, SELL
        - entry_price: Entry price in paisa
        - stop_loss: Stop loss price in paisa
        - target: Target price in paisa
        - quantity: Position size
        - strategy_name: Name of the generating strategy

        Args:
            signal: Dictionary containing signal parameters.

        Returns:
            ValidationResult with validation status and any errors.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check None
        if signal is None:
            errors.append("Signal cannot be None")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check type
        if not isinstance(signal, dict):
            errors.append(f"Signal must be a dict, got {type(signal).__name__}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check required fields
        required_fields = {"symbol", "signal_type", "entry_price", "stop_loss", "target", "quantity", "strategy_name"}
        missing_fields = required_fields - set(signal.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {sorted(missing_fields)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Validate using Pydantic model
        try:
            SignalValidationModel(**signal)
        except Exception as e:
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                lines = error_msg.split("\n")
                for line in lines[1:] if len(lines) > 1 else lines:
                    if line.strip():
                        errors.append(line.strip())
            else:
                errors.append(f"Signal validation failed: {error_msg}")

        # Calculate and warn about risk/reward ratio
        entry = sanitize_numeric(signal.get("entry_price"))
        sl = sanitize_numeric(signal.get("stop_loss"))
        target = sanitize_numeric(signal.get("target"))
        signal_type = str(signal.get("signal_type", "")).upper()

        if entry is not None and sl is not None and target is not None and entry > 0:
            if "BUY" in signal_type:
                risk = entry - sl
                reward = target - entry
            else:  # SELL
                risk = sl - entry
                reward = entry - target

            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < 1.0:
                    warnings.append(f"Risk/Reward ratio is unfavorable ({rr_ratio:.2f}:1) - consider better setup")
                elif rr_ratio < 1.5:
                    warnings.append(f"Risk/Reward ratio is marginal ({rr_ratio:.2f}:1)")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_config(config: dict) -> ValidationResult:
        """Validate trading configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters.

        Returns:
            ValidationResult with validation status and any errors.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check None
        if config is None:
            errors.append("Config cannot be None")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check type
        if not isinstance(config, dict):
            errors.append(f"Config must be a dict, got {type(config).__name__}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Validate using Pydantic model
        try:
            ConfigValidationModel(**config)
        except Exception as e:
            error_msg = str(e)
            if "validation error" in error_msg.lower():
                lines = error_msg.split("\n")
                for line in lines[1:] if len(lines) > 1 else lines:
                    if line.strip():
                        errors.append(line.strip())
            else:
                errors.append(f"Config validation failed: {error_msg}")

        # Additional warnings
        max_risk = config.get("max_risk_per_trade", 0.01)
        if isinstance(max_risk, (int, float)) and max_risk > 0.02:
            warnings.append(f"max_risk_per_trade ({max_risk:.1%}) is higher than recommended 2%")

        daily_loss = config.get("daily_loss_limit", 0.03)
        if isinstance(daily_loss, (int, float)) and daily_loss > 0.05:
            warnings.append(f"daily_loss_limit ({daily_loss:.1%}) is higher than recommended 5%")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def sanitize_symbol(symbol: Any) -> str:
    """Sanitize a trading symbol.

    Args:
        symbol: The symbol to sanitize.

    Returns:
        Uppercase, stripped symbol string.
    """
    if symbol is None:
        return ""
    if not isinstance(symbol, str):
        symbol = str(symbol)
    return symbol.strip().upper()


def sanitize_timestamp(ts: Any) -> datetime | None:
    """Sanitize and convert various timestamp formats to datetime.

    Handles:
    - datetime objects
    - ISO format strings
    - Unix timestamps (int/float)
    - Common date/time strings

    Args:
        ts: The timestamp to sanitize.

    Returns:
        Timezone-aware datetime or None if conversion fails.
    """
    if ts is None:
        return None

    # Already a datetime
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            from src.utils.helpers import IST
            return ts.replace(tzinfo=IST)
        return ts

    # String formats
    if isinstance(ts, str):
        ts = ts.strip()
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]
        for fmt in formats:
            try:
                result = datetime.strptime(ts, fmt)
                if result.tzinfo is None:
                    from src.utils.helpers import IST
                    result = result.replace(tzinfo=IST)
                return result
            except ValueError:
                continue

    # Unix timestamp
    try:
        numeric_ts = float(ts)
        if numeric_ts > 1e12:  # Milliseconds
            numeric_ts = numeric_ts / 1000
        result = datetime.fromtimestamp(numeric_ts)
        from src.utils.helpers import IST
        return result.replace(tzinfo=IST)
    except (ValueError, TypeError, OSError):
        pass

    logger.warning(f"Could not parse timestamp: {ts!r}")
    return None


def sanitize_numeric(val: Any) -> int | None:
    """Convert a value to integer paisa amount.

    Handles:
    - Integers (returned as-is)
    - Floats (converted to paisa)
    - Numeric strings
    - Decimal values

    Args:
        val: The value to convert.

    Returns:
        Integer value or None if conversion fails.
    """
    if val is None:
        return None

    # Already an integer
    if isinstance(val, int) and not isinstance(val, bool):
        return val

    # Float - convert to paisa (multiply by 100 and round)
    if isinstance(val, float):
        return round(val * 100)

    # String - try to parse
    if isinstance(val, str):
        val = val.strip().replace(",", "")  # Remove thousand separators
        try:
            # Try integer first
            if "." not in val:
                return int(val)
            # Parse as float then convert
            return round(float(val) * 100)
        except ValueError:
            pass

    # Decimal
    if isinstance(val, Decimal):
        return int(val * 100)

    return None


def validate_inputs(
    **validators: Callable[[Any], ValidationResult]
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for automatic input validation.

    Usage:
        @validate_inputs(symbol=InputValidator.validate_symbol)
        def process_trade(symbol: str, quantity: int):
            pass

    Args:
        **validators: Mapping of argument name to validator function.

    Returns:
        Decorator function that validates inputs before calling the wrapped function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get function signature to map args
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            all_errors: list[str] = []
            all_warnings: list[str] = []

            for arg_name, validator_func in validators.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    result = validator_func(value)

                    if not result.is_valid:
                        all_errors.extend(
                            [f"{arg_name}: {err}" for err in result.errors]
                        )
                    if result.warnings:
                        all_warnings.extend(
                            [f"{arg_name}: {warn}" for warn in result.warnings]
                        )

                    # Update with sanitized value if available
                    if result.sanitized_value is not None:
                        bound_args.arguments[arg_name] = result.sanitized_value

            # Log warnings
            for warning in all_warnings:
                logger.warning(f"Validation warning in {func.__name__}: {warning}")

            # Handle errors
            if all_errors:
                error_msg = f"Input validation failed for {func.__name__}"
                for error in all_errors:
                    logger.error(f"Validation error in {func.__name__}: {error}")
                raise ValidationError(error_msg, all_errors)

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator


# Convenience functions for common validation patterns

def is_valid_symbol(symbol: Any) -> bool:
    """Quick check if a symbol is valid.

    Args:
        symbol: The symbol to check.

    Returns:
        True if valid, False otherwise.
    """
    return InputValidator.validate_symbol(symbol).is_valid


def is_valid_price(price: Any, allow_zero: bool = False) -> bool:
    """Quick check if a price is valid.

    Args:
        price: The price to check.
        allow_zero: Whether to allow zero as valid.

    Returns:
        True if valid, False otherwise.
    """
    return InputValidator.validate_price(price, allow_zero=allow_zero).is_valid


def is_valid_quantity(qty: Any, lot_size: int = 1) -> bool:
    """Quick check if a quantity is valid.

    Args:
        qty: The quantity to check.
        lot_size: The lot size multiplier.

    Returns:
        True if valid, False otherwise.
    """
    return InputValidator.validate_quantity(qty, lot_size).is_valid


def get_sanitized_symbol(symbol: Any, default: str = "") -> str:
    """Get sanitized symbol or default if invalid.

    Args:
        symbol: The symbol to sanitize.
        default: Default value if invalid.

    Returns:
        Sanitized symbol or default.
    """
    result = InputValidator.validate_symbol(symbol)
    if result.is_valid and result.sanitized_value:
        return result.sanitized_value
    return default
