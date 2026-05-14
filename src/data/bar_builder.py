"""OHLCV bar builder from WebSocket ticks.

Consumes ticks from a queue and builds 1min, 5min, 15min candles.
Notifies strategy threads on bar close via threading.Event.
"""

from __future__ import annotations

import queue
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# Timeframes in minutes
TIMEFRAMES = [1, 5, 15]
MAX_BARS = 500

# Lock for current bars access
_current_bars_lock = threading.Lock()


@dataclass
class OHLCVBar:
    """Single OHLCV bar."""

    timestamp: datetime
    open: int  # paisa
    high: int  # paisa
    low: int  # paisa
    close: int  # paisa
    volume: int

    def to_list(self) -> list[Any]:
        """Convert to list format for storage."""
        return [self.timestamp, self.open, self.high, self.low, self.close, self.volume]


@dataclass
class CurrentBar:
    """In-progress bar data."""

    open: int = 0
    high: int = 0
    low: int = 0
    close: int = 0
    volume: int = 0
    timestamp: datetime | None = None


class BarBuilder:
    """Builds OHLCV bars from ticks and notifies on bar close.

    Runs in a separate thread consuming from tick_queue.
    Maintains bar history per instrument and timeframe.

    Args:
        tick_queue: Queue to consume ticks from.
        bar_close_event: Event to signal when bars close.
    """

    def __init__(
        self,
        tick_queue: queue.Queue[dict] | None = None,
        bar_close_event: threading.Event | None = None,
        db_engine: Any | None = None,
    ) -> None:
        self.tick_queue = tick_queue or queue.Queue()
        self.bar_close_event = bar_close_event or threading.Event()
        self._db_engine = db_engine

        # {instrument_key: {timeframe: CurrentBar}}
        self._current_bars: dict[str, dict[int, CurrentBar]] = defaultdict(
            lambda: defaultdict(CurrentBar)
        )

        # {instrument_key: {timeframe: deque[OHLCVBar]}}
        self._bar_history: dict[str, dict[int, deque[OHLCVBar]]] = defaultdict(
            lambda: {tf: deque(maxlen=MAX_BARS) for tf in TIMEFRAMES}
        )

        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the bar builder thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, name="BarBuilder", daemon=True)
        self._thread.start()
        logger.info("BarBuilder started")

    def stop(self) -> None:
        """Stop the bar builder thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("BarBuilder stopped")

    def _run(self) -> None:
        """Main loop consuming ticks from queue."""
        while self._running:
            try:
                tick = self.tick_queue.get(timeout=1.0)
                self._process_tick(tick)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error("Error processing tick: {}", exc)

    def _process_tick(self, tick: dict[str, Any]) -> None:
        """Process a single tick update.

        Handles both legacy flat format and Upstox V3 feed format:
            V3:  {"type": ..., "feeds": {<instrument_key>: {...}}, "currentTs": ...}
            Legacy: {"instrument_key": ..., "last_price": ..., ...}
        """
        # --- V3 format: iterate `feeds` dict ---
        feeds = tick.get("feeds")
        if isinstance(feeds, dict) and feeds:
            top_ts = tick.get("currentTs")
            bars_closed = False
            for instrument_key, feed in feeds.items():
                ltp_paisa, vol, ts_ms = self._extract_ltp_v3(feed)
                if ltp_paisa is None:
                    continue
                ts_source = ts_ms or top_ts
                timestamp = self._parse_timestamp(ts_source)
                if timestamp.year < 2000 and top_ts:
                    # Upstox sends negative magic numbers (like -6050019840) for ltt if no trades happened
                    timestamp = self._parse_timestamp(top_ts)
                
                # If still invalid, fallback to current time
                if timestamp.year < 2000:
                    timestamp = datetime.now(IST)

                for tf in TIMEFRAMES:
                    if self._update_bar(instrument_key, tf, ltp_paisa, vol, timestamp):
                        bars_closed = True
            if bars_closed:
                self.bar_close_event.set()
            return

        # --- Legacy flat format ---
        instrument_key = tick.get("instrument_key")
        if not instrument_key:
            return
        ltp = tick.get("last_price", 0)
        if isinstance(ltp, (int, float)):
            ltp = int(float(ltp) * 100)
        volume = tick.get("volume_traded", 0)
        timestamp = self._parse_timestamp(tick.get("timestamp"))

        bars_closed = False
        for tf in TIMEFRAMES:
            if self._update_bar(instrument_key, tf, ltp, volume, timestamp):
                bars_closed = True
        if bars_closed:
            self.bar_close_event.set()

    def _extract_ltp_v3(
        self, feed: Any
    ) -> tuple[int | None, int, Any]:
        """Extract LTP (paisa), volume, and timestamp from a V3 feed entry.

        V3 feed shapes seen on Upstox:
          - {"ltpc": {"ltp": 50000.5, "ltt": "..."}}
          - {"fullFeed": {"marketFF": {"ltpc": {...}, "vtt": ...}}}
          - {"fullFeed": {"indexFF": {"ltpc": {...}}}}     (indices)
          - {"firstLevelWithGreeks": {"ltpc": {...}}}      (options)
        """
        if not isinstance(feed, dict):
            return None, 0, None

        ltpc = feed.get("ltpc")
        vol = 0
        ts = None

        if ltpc is None:
            ff = feed.get("fullFeed", {}) if isinstance(feed.get("fullFeed"), dict) else {}
            inner = ff.get("marketFF") or ff.get("indexFF") or {}
            if isinstance(inner, dict):
                ltpc = inner.get("ltpc")
                vol = inner.get("vtt") or inner.get("volume") or 0

        if ltpc is None:
            inner = feed.get("firstLevelWithGreeks") or {}
            if isinstance(inner, dict):
                ltpc = inner.get("ltpc")

        if not isinstance(ltpc, dict):
            return None, 0, None

        ltp_raw = ltpc.get("ltp")
        if ltp_raw is None:
            return None, 0, None

        try:
            ltp_paisa = int(float(ltp_raw) * 100)
        except (TypeError, ValueError):
            return None, 0, None

        ts = ltpc.get("ltt")
        try:
            vol = int(vol) if vol else 0
        except (TypeError, ValueError):
            vol = 0
        return ltp_paisa, vol, ts

    def _update_bar(
        self,
        instrument_key: str,
        timeframe: int,
        price: int,
        volume: int,
        timestamp: datetime,
    ) -> bool:
        """Update a bar for given instrument and timeframe.

        Returns:
            True if bar closed on this update.
        """
        with _current_bars_lock:
            bar = self._current_bars[instrument_key][timeframe]
            bar_time = self._get_bar_time(timestamp, timeframe)

            # New bar started
            if bar.timestamp != bar_time:
                # Close previous bar if exists
                if bar.timestamp is not None:
                    closed_bar = OHLCVBar(
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                    )
                    with self._lock:
                        self._bar_history[instrument_key][timeframe].append(closed_bar)
                    logger.debug(
                        "Bar closed: {} {} {} O:{} H:{} L:{} C:{} V:{}",
                        instrument_key,
                        timeframe,
                        bar.timestamp,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                    )
                    self._persist_bar_to_postgres(instrument_key, timeframe, closed_bar)

                # Start new bar
                bar.timestamp = bar_time
                bar.open = price
                bar.high = price
                bar.low = price
                bar.close = price
                bar.volume = volume
                return True

            # Update existing bar
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)
            bar.close = price
            bar.volume += volume
            return False

    def _get_bar_time(self, timestamp: datetime, timeframe: int) -> datetime:
        """Get the bar start time for a given timestamp and timeframe."""
        minutes = timestamp.hour * 60 + timestamp.minute
        bar_minutes = (minutes // timeframe) * timeframe
        bar_hour = bar_minutes // 60
        bar_min = bar_minutes % 60
        return timestamp.replace(hour=bar_hour, minute=bar_min, second=0, microsecond=0)

    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp to datetime in IST.

        Accepts:
          - datetime instance
          - ISO 8601 string
          - int / numeric-string epoch (seconds or milliseconds)
        """
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=IST)
            return ts.astimezone(IST)
        if isinstance(ts, (int, float)):
            # Heuristic: > 1e12 → milliseconds, else seconds.
            seconds = ts / 1000.0 if ts > 1e12 else float(ts)
            return datetime.fromtimestamp(seconds, tz=IST)
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.astimezone(IST)
            except ValueError:
                pass
            try:
                ts_num = float(ts)
                seconds = ts_num / 1000.0 if ts_num > 1e12 else ts_num
                return datetime.fromtimestamp(seconds, tz=IST)
            except ValueError:
                pass
        return datetime.now(IST)

    def get_bars(
        self,
        instrument_key: str,
        timeframe: int,
        include_current: bool = False,
    ) -> pd.DataFrame:
        """Get bar history as DataFrame.

        Args:
            instrument_key: Instrument key.
            timeframe: Bar timeframe in minutes.
            include_current: Whether to include in-progress bar.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        with self._lock:
            history = list(self._bar_history[instrument_key].get(timeframe, []))

        data = [bar.to_list() for bar in history]

        # Include current bar if requested
        if include_current and timeframe in self._current_bars.get(instrument_key, {}):
            current = self._current_bars[instrument_key][timeframe]
            if current.timestamp:
                data.append(
                    [
                        current.timestamp,
                        current.open,
                        current.high,
                        current.low,
                        current.close,
                        current.volume,
                    ]
                )

        if not data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df.set_index("timestamp", inplace=True)
        return df

    def seed_bars(
        self,
        instrument_key: str,
        timeframe: int,
        df: pd.DataFrame,
    ) -> int:
        """Seed bar history from a historical OHLCV DataFrame.

        Used at startup to warm up indicators with prior trading days
        instead of waiting for live ticks to accumulate enough samples.

        Args:
            instrument_key: Instrument key.
            timeframe: Bar timeframe in minutes (must be in TIMEFRAMES).
            df: DataFrame indexed by timestamp with columns
                open/high/low/close/volume (prices in PAISA).

        Returns:
            Number of bars seeded.
        """
        if timeframe not in TIMEFRAMES:
            logger.warning(
                "seed_bars skipped: timeframe {} not in {}", timeframe, TIMEFRAMES
            )
            return 0
        if df is None or df.empty:
            return 0

        seeded = 0
        with self._lock:
            history = self._bar_history[instrument_key][timeframe]
            for ts, row in df.iterrows():
                bar = OHLCVBar(
                    timestamp=ts,
                    open=int(row["open"]),
                    high=int(row["high"]),
                    low=int(row["low"]),
                    close=int(row["close"]),
                    volume=int(row.get("volume", 0)),
                )
                history.append(bar)
                seeded += 1
        return seeded

    def get_all_instrument_keys(self) -> list[str]:
        """Get all instrument keys with bar data."""
        return list(self._bar_history.keys())

    def clear(self, instrument_key: str | None = None) -> None:
        """Clear bar history.

        Args:
            instrument_key: Specific instrument to clear, or None for all.
        """
        with self._lock:
            if instrument_key:
                self._bar_history.pop(instrument_key, None)
                self._current_bars.pop(instrument_key, None)
            else:
                self._bar_history.clear()
                self._current_bars.clear()

    def _persist_bar_to_postgres(
        self,
        instrument_key: str,
        timeframe: int,
        bar: OHLCVBar,
    ) -> None:
        """Write closed bar to Postgres ``bars`` table.

        Idempotent via PRIMARY KEY (instrument_key, timeframe, ts) — uses
        INSERT ON CONFLICT DO NOTHING.  Silently skips if no engine
        configured or if the DB call fails (bot operation must not be
        interrupted by a storage error).

        Args:
            instrument_key: Upstox instrument key string.
            timeframe: Bar width in minutes (1, 5, 15, …).
            bar: Closed OHLCVBar dataclass instance.
        """
        if self._db_engine is None:
            return
        try:
            from sqlalchemy import text as _text

            # Map int timeframe (minutes) → string used in Phase 0 schema
            tf_str = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}.get(
                int(timeframe), f"{timeframe}m"
            )
            with self._db_engine.begin() as conn:
                conn.execute(
                    _text(
                        "INSERT INTO bars "
                        "(instrument_key, timeframe, ts, open, high, low, close, volume) "
                        "VALUES (:ik, :tf, :ts, :o, :h, :l, :c, :v) "
                        "ON CONFLICT (instrument_key, timeframe, ts) DO NOTHING"
                    ),
                    {
                        "ik": instrument_key,
                        "tf": tf_str,
                        "ts": bar.timestamp,
                        "o": int(bar.open),
                        "h": int(bar.high),
                        "l": int(bar.low),
                        "c": int(bar.close),
                        "v": int(bar.volume),
                    },
                )
        except Exception as exc:
            logger.debug(
                "bar persist to postgres failed for {} {}: {}",
                instrument_key,
                timeframe,
                exc,
            )
