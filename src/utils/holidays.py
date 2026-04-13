"""NSE Trading Holiday Calendar Utility."""

from datetime import date
from typing import Set

# List of NSE trading holidays for 2024-2026 (partial list - need to be updated)
# Source: https://www.nseindia.com/get-data/equity-trading-holiday
NSE_HOLIDAYS = {
    date(2024, 1, 1),  # New Year's Day
    date(2024, 1, 26),  # Republic Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 1),  # May Day
    date(2024, 8, 15),  # Independence Day
    date(2024, 10, 2),  # Gandhi Jayanti
    date(2024, 11, 1),  # Diwali (Lakshmi Pujan)
    date(2024, 12, 25),  # Christmas
    date(2025, 1, 1),  # New Year's Day
    date(2025, 1, 26),  # Republic Day
    date(2025, 3, 18),  # Holi
    date(2025, 3, 29),  # Good Friday
    date(2025, 5, 1),  # May Day
    date(2025, 8, 15),  # Independence Day
    date(2025, 10, 2),  # Gandhi Jayanti
    date(2025, 10, 20),  # Diwali (Lakshmi Pujan)
    date(2025, 12, 25),  # Christmas
    date(2026, 1, 1),  # New Year's Day
    date(2026, 1, 26),  # Republic Day
    date(2026, 3, 3),  # Maha Shivratri
    date(2026, 3, 18),  # Holi
    date(2026, 3, 30),  # Good Friday
    date(2026, 5, 1),  # May Day
    date(2026, 8, 15),  # Independence Day
    date(2026, 10, 2),  # Gandhi Jayanti
    date(2026, 11, 10),  # Diwali (Lakshmi Pujan)
    date(2026, 12, 25),  # Christmas
}


def is_nse_holiday(date_obj: date) -> bool:
    """Check if a given date is an NSE trading holiday."""
    return date_obj in NSE_HOLIDAYS


def is_trading_day(date_obj: date) -> bool:
    """Check if a given date is a valid trading day (not weekend or holiday)."""
    # Check if weekend (Saturday=5, Sunday=6)
    if date_obj.weekday() >= 5:
        return False
    return not is_nse_holiday(date_obj)


def get_next_trading_day(date_obj: date) -> date:
    """Get the next trading day after the given date."""
    next_day = date_obj + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day


from datetime import timedelta  # noqa: E402
