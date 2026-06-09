"""Simulated clock for replay/backtest mode.

When sim time is set, now_ist() returns the simulated time instead of
wall-clock. This lets the strategy engine's trading-hours gate work
correctly when replaying historical data outside Indian market hours.

Usage:
    from src.utils.sim_clock import set_sim_time, reset_sim_time, now_ist

    set_sim_time(some_historical_datetime)   # enter replay mode
    # ... replay runs ...
    reset_sim_time()                          # back to wall-clock
"""
from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

_lock = threading.Lock()
_sim_time: datetime | None = None


def set_sim_time(dt: datetime) -> None:
    global _sim_time
    with _lock:
        _sim_time = dt.astimezone(IST) if dt.tzinfo else dt.replace(tzinfo=IST)


def reset_sim_time() -> None:
    global _sim_time
    with _lock:
        _sim_time = None


def now_ist() -> datetime:
    with _lock:
        if _sim_time is not None:
            return _sim_time
    return datetime.now(IST)
