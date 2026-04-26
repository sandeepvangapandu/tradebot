"""NSE Trading Holiday Calendar Utility.

2026 list verified against official NSE circular CMTR71775 and corroborated
against Zerodha + Groww calendars (2026-04-26).

Muhurat trading on 2026-11-08 (Sunday) is intentionally NOT in this set —
that is a special trading session, not a holiday.

2024/2025 entries are retained for backtests; they are partial and should
not be relied on for live trading on those dates.

When the next year's circular is published (typically December), append the
new dates to ``NSE_HOLIDAYS`` below. Source of truth:
https://www.nseindia.com/resources/exchange-communication-holidays
"""

from datetime import date, timedelta

NSE_HOLIDAYS: set[date] = {
    # ---- 2024 (partial — for backtest reference only) ----
    date(2024, 1, 1),    # New Year's Day
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 29),   # Good Friday
    date(2024, 5, 1),    # May Day
    date(2024, 8, 15),   # Independence Day
    date(2024, 10, 2),   # Gandhi Jayanti
    date(2024, 11, 1),   # Diwali (Lakshmi Pujan)
    date(2024, 12, 25),  # Christmas

    # ---- 2025 (partial — for backtest reference only) ----
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 26),   # Republic Day (Sunday — already weekend)
    date(2025, 3, 18),   # Holi
    date(2025, 3, 29),   # Good Friday
    date(2025, 5, 1),    # May Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 20),  # Diwali (Lakshmi Pujan)
    date(2025, 12, 25),  # Christmas

    # ---- 2026 (FULL list — verified for live trading) ----
    date(2026, 1, 15),   # Municipal Corporation Elections (Maharashtra)
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Eid
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali-Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}


def is_nse_holiday(date_obj: date) -> bool:
    """Check if a given date is an NSE trading holiday."""
    return date_obj in NSE_HOLIDAYS


def is_trading_day(date_obj: date) -> bool:
    """Check if a given date is a valid trading day (not weekend or holiday)."""
    # Saturday=5, Sunday=6
    if date_obj.weekday() >= 5:
        return False
    return not is_nse_holiday(date_obj)


def get_next_trading_day(date_obj: date) -> date:
    """Get the next trading day after the given date."""
    next_day = date_obj + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day
