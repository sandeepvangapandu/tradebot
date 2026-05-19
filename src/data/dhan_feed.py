"""Dhan real-time market data feed via WebSocket.

Connects to the Dhan market feed WebSocket (wss://api-feed.dhan.co)
and streams live LTP, OHLC, and depth data for subscribed instruments.

The feed runs in a background thread and delivers ticks via a callback
function, matching the same contract used by the Upstox market feed so
strategies don't need to change.

Usage:
    feed = DhanMarketFeed(
        client_id="1111361185",
        access_token="eyJ...",
        instruments=["NSE_EQ|26009", "NSE_FNO|35001"],  # security_id format
        on_tick=my_callback,
    )
    feed.start()
    # ... trading ...
    feed.stop()
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# ─── Dhan exchange segment constants ─────────────────────────────────────────
_SEGMENT_MAP: dict[str, int] = {
    "IDX": 0,
    "NSE_EQ": 1,
    "NSE_FNO": 2,
    "NSE_CURRENCY": 3,
    "BSE_EQ": 4,
    "MCX_COMM": 5,
    "BSE_CURRENCY": 7,
    "BSE_FNO": 8,
    # Shorthands
    "NSE": 1,
    "BSE": 4,
    "NFO": 2,
    "IDX_I": 0,
}

# Feed subscription types
_TICKER = 15    # LTP only
_QUOTE = 17     # LTP + OHLC + volume
_FULL = 21      # LTP + OHLC + depth


class DhanTick:
    """Normalised tick data from the Dhan market feed.

    Attributes:
        instrument_key: Our internal format e.g. 'NSE_EQ|26009'.
        ltp: Last traded price in PAISA (int).
        open: Open price in PAISA.
        high: High price in PAISA.
        low: Low price in PAISA.
        close: Previous close in PAISA.
        volume: Traded volume.
        oi: Open interest (for F&O).
        timestamp: Tick timestamp in IST.
        raw: Raw dict from Dhan for debug use.
    """

    __slots__ = (
        "instrument_key", "ltp", "open", "high", "low",
        "close", "volume", "oi", "timestamp", "raw",
    )

    def __init__(
        self,
        instrument_key: str,
        ltp: int,
        open_: int = 0,
        high: int = 0,
        low: int = 0,
        close: int = 0,
        volume: int = 0,
        oi: int = 0,
        timestamp: Optional[datetime] = None,
        raw: Optional[dict] = None,
    ) -> None:
        self.instrument_key = instrument_key
        self.ltp = ltp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.oi = oi
        self.timestamp = timestamp or datetime.now(tz=IST)
        self.raw = raw or {}

    def to_dict(self) -> dict[str, Any]:
        """Return tick as dict (compatible with Upstox feed contract)."""
        return {
            "instrument_key": self.instrument_key,
            "ltp": self.ltp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "oi": self.oi,
            "timestamp": self.timestamp.isoformat(),
        }


def _rupees_to_paisa(val: float) -> int:
    """Convert float Rupees to integer paisa."""
    return int(round(val * 100))


def _parse_instruments(
    instrument_keys: list[str],
) -> list[tuple[int, str]]:
    """Convert our instrument_key format to Dhan (segment_int, security_id) tuples.

    Args:
        instrument_keys: List of 'SEGMENT|security_id' strings.

    Returns:
        List of (segment_code, security_id) tuples for DhanFeed.
    """
    result: list[tuple[int, str]] = []
    for key in instrument_keys:
        if "|" in key:
            seg, sec_id = key.split("|", 1)
            code = _SEGMENT_MAP.get(seg.upper(), 1)
            result.append((code, sec_id))
        else:
            result.append((1, key))  # default NSE_EQ
    return result


class DhanMarketFeed:
    """Dhan real-time WebSocket market feed.

    Streams live ticks for subscribed instruments and calls
    on_tick(tick: DhanTick) for each update.

    Args:
        client_id: Dhan account client ID.
        access_token: Dhan JWT access token.
        instruments: List of instrument keys in 'SEGMENT|security_id' format.
                     Example: ["NSE_EQ|1333", "NSE_FNO|35001"]
        on_tick: Callback called with each DhanTick.
        subscription_type: 'ticker' (LTP only), 'quote' (OHLC), 'full' (depth).
        reconnect_delay_s: Seconds between reconnection attempts.
    """

    def __init__(
        self,
        client_id: str,
        access_token: str,
        instruments: list[str],
        on_tick: Callable[[DhanTick], None],
        subscription_type: str = "quote",
        reconnect_delay_s: float = 5.0,
    ) -> None:
        self._client_id = client_id
        self._access_token = access_token
        self._instrument_keys = instruments
        self._on_tick = on_tick
        self._reconnect_delay = reconnect_delay_s

        _sub_map = {"ticker": _TICKER, "quote": _QUOTE, "full": _FULL}
        self._sub_code = _sub_map.get(subscription_type, _QUOTE)

        # (segment_code, security_id) pairs for DhanFeed
        self._dhan_instruments = _parse_instruments(instruments)

        # Map security_id back to our instrument_key for tick resolution
        self._sec_to_key: dict[str, str] = {}
        for key, (_, sec_id) in zip(instruments, self._dhan_instruments):
            self._sec_to_key[sec_id] = key

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._feed = None
        self._running = False
        self._last_ticks: dict[str, DhanTick] = {}

    # ─── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the feed in a background thread."""
        if self._running:
            logger.warning("DhanMarketFeed already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="dhan-market-feed",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "DhanMarketFeed started | instruments={} | type={}",
            len(self._instrument_keys),
            self._sub_code,
        )

    def stop(self) -> None:
        """Stop the feed and close the WebSocket connection."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            if self._feed:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._feed.disconnect(), self._loop
                    ).result(timeout=3.0)
                except Exception:
                    pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("DhanMarketFeed stopped")

    def get_ltp(self, instrument_key: str) -> Optional[int]:
        """Return the last LTP in paisa for the given instrument."""
        tick = self._last_ticks.get(instrument_key)
        return tick.ltp if tick else None

    def get_last_tick(self, instrument_key: str) -> Optional[DhanTick]:
        """Return the full last tick for the given instrument."""
        return self._last_ticks.get(instrument_key)

    @property
    def is_running(self) -> bool:
        """True if the feed thread is alive and connected."""
        return self._running and (self._thread is not None) and self._thread.is_alive()

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Run the asyncio event loop in a background thread with auto-reconnect."""
        while self._running:
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.run_until_complete(self._feed_session())
            except Exception as e:
                logger.warning("DhanMarketFeed error: {} | reconnecting in {}s", e, self._reconnect_delay)
            finally:
                if self._loop and not self._loop.is_closed():
                    self._loop.close()

            if self._running:
                time.sleep(self._reconnect_delay)

    async def _feed_session(self) -> None:
        """Single WebSocket session — runs until disconnect or error."""
        from dhanhq import marketfeed

        # Build instrument list: list of (exchange_code, security_id, sub_type)
        instruments_with_type = [
            (seg, sec_id, self._sub_code)
            for seg, sec_id in self._dhan_instruments
        ]

        self._feed = marketfeed.DhanFeed(
            client_id=self._client_id,
            access_token=self._access_token,
            instruments=instruments_with_type,
            version="v2",
        )

        await self._feed.connect()
        logger.info("DhanMarketFeed WebSocket connected")

        while self._running:
            try:
                raw_data = await asyncio.wait_for(
                    self._feed.get_instrument_data(),
                    timeout=30.0,
                )
                if raw_data:
                    tick = self._parse_tick(raw_data)
                    if tick:
                        self._last_ticks[tick.instrument_key] = tick
                        try:
                            self._on_tick(tick)
                        except Exception as cb_err:
                            logger.warning("on_tick callback error: {}", cb_err)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await self._feed.ws.ping()
                except Exception:
                    break
            except Exception as e:
                logger.warning("DhanMarketFeed recv error: {}", e)
                break

    def _parse_tick(self, data: dict[str, Any]) -> Optional[DhanTick]:
        """Parse raw Dhan data dict into a normalised DhanTick.

        Dhan returns different packet formats based on subscription type.
        We normalise everything to DhanTick with values in PAISA.
        """
        if not isinstance(data, dict):
            return None

        sec_id = str(data.get("security_id", data.get("securityId", "")))
        instrument_key = self._sec_to_key.get(sec_id, f"NSE_EQ|{sec_id}")

        # LTP — could be 'LTP', 'last_price', etc.
        ltp_raw = data.get("LTP") or data.get("last_price") or data.get("ltp") or 0
        ltp = _rupees_to_paisa(float(ltp_raw))

        if ltp == 0:
            return None  # skip empty ticks

        open_ = _rupees_to_paisa(float(data.get("open", 0) or 0))
        high = _rupees_to_paisa(float(data.get("high", 0) or 0))
        low = _rupees_to_paisa(float(data.get("low", 0) or 0))
        close = _rupees_to_paisa(float(data.get("close", 0) or data.get("prev_close", 0) or 0))
        volume = int(data.get("volume", 0) or 0)
        oi = int(data.get("OI", 0) or data.get("oi", 0) or 0)

        return DhanTick(
            instrument_key=instrument_key,
            ltp=ltp,
            open_=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            oi=oi,
            timestamp=datetime.now(tz=IST),
            raw=data,
        )


# ─── NSE index security IDs (common instruments) ──────────────────────────────
# These are the Dhan security IDs for popular instruments.
# Used to quickly subscribe to index data.
DHAN_SECURITY_IDS: dict[str, str] = {
    "NIFTY_50":    "13",
    "BANKNIFTY":   "25",
    "FINNIFTY":    "27",
    "MIDCPNIFTY":  "442",
    "SENSEX":      "1",
    "RELIANCE":    "2885",
    "TCS":         "11536",
    "INFY":        "1594",
    "HDFCBANK":    "1333",
    "ICICIBANK":   "4963",
    "SBIN":        "3045",
}
