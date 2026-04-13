"""General-purpose utility functions for the trading bot."""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def now_ist() -> datetime:
    """Return the current datetime in IST (Asia/Kolkata).

    Returns:
        Current IST-aware datetime.
    """
    return datetime.now(tz=IST)


def is_market_open(holidays: set[date] | None = None) -> bool:
    """Check whether the Indian equity market is currently open.

    Checks that today is a weekday (Mon-Fri), not a listed holiday,
    and the current IST time falls between 09:15 and 15:30.

    Args:
        holidays: Optional set of holiday dates to exclude.

    Returns:
        True if the market is open right now, False otherwise.
    """
    current = now_ist()

    # Weekend check (Monday=0 ... Friday=4).
    if current.weekday() > 4:
        return False

    # Holiday check.
    if holidays and current.date() in holidays:
        return False

    # Time window check.
    return MARKET_OPEN <= current.time() < MARKET_CLOSE


def rupees_to_paisa(r: float) -> int:
    """Convert a rupee amount to paisa (integer).

    Args:
        r: Amount in rupees.

    Returns:
        Equivalent amount in paisa.
    """
    return round(r * 100)


def paisa_to_rupees(p: int) -> float:
    """Convert a paisa amount to rupees.

    Args:
        p: Amount in paisa.

    Returns:
        Equivalent amount in rupees.
    """
    return p / 100.0


def format_instrument_key(segment: str, identifier: str) -> str:
    """Build a canonical instrument key string.

    Args:
        segment: Market segment (e.g. ``"NSE_EQ"``).
        identifier: Instrument identifier (e.g. ``"RELIANCE"``).

    Returns:
        Formatted key like ``"NSE_EQ|RELIANCE"``.
    """
    return f"{segment}|{identifier}"


def parse_time(t: str) -> time:
    """Parse an ``"HH:MM:SS"`` string into a :class:`datetime.time` object.

    Args:
        t: Time string in ``HH:MM:SS`` format.

    Returns:
        Parsed time object.

    Raises:
        ValueError: If the string does not match the expected format.
    """
    parts = t.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS format, got '{t}'")
    return time(int(parts[0]), int(parts[1]), int(parts[2]))
