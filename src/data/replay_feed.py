"""Drop-in replacement for MarketDataFeed that replays historical 1-min CSV bars.

Emits Upstox V3-format tick dicts into the live trading pipeline so the
full TradingBot component tree (BarBuilder → StrategyEngine → OrderManager)
runs on historical data with the same code path as live trading.

For F&O straddle instruments (NSE_INDEX|*), synthetic ATM option prices are
computed via an ATR model and injected as additional ticks whenever the
OrderManager needs a fill price for a straddle leg.
"""
from __future__ import annotations

import queue
import threading
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from src.utils.sim_clock import set_sim_time

IST = ZoneInfo("Asia/Kolkata")

# Upstox V3 tick format helpers
def _make_tick(instrument_key: str, ltp_rs: float, ts: datetime, volume: int = 0) -> dict:
    """Build a minimal Upstox V3 tick dict."""
    ts_ms = str(int(ts.timestamp() * 1000))
    return {
        "feeds": {
            instrument_key: {
                "ltpc": {
                    "ltp": ltp_rs,
                    "v": volume,
                    "cp": ltp_rs,
                },
                "ff": {
                    "marketFF": {
                        "ltpc": {"ltp": ltp_rs, "v": volume},
                    }
                },
            }
        },
        "currentTs": ts_ms,
    }


class ReplayInstrument:
    """Configuration for a single instrument in the replay."""

    def __init__(
        self,
        instrument_key: str,
        csv_path: str,
        straddle_proxy: bool = False,
        prices_in_paisa: bool = True,
        label: str = "REAL_PRICE",
    ):
        self.instrument_key = instrument_key
        self.csv_path = csv_path
        self.straddle_proxy = straddle_proxy  # True for NSE_INDEX → synthetic options
        self.prices_in_paisa = prices_in_paisa
        self.label = label


class LiveReplayFeed:
    """Replaces MarketDataFeed for historical replay mode.

    Interface mirrors MarketDataFeed so TradingBot wiring stays unchanged:
      - register_callback(cb)
      - subscribe(keys, mode)
      - start(symbols, mode)   ← begins replay in a background thread
      - stop()
    """

    def __init__(
        self,
        instruments: list[ReplayInstrument],
        tick_queue: queue.Queue,
        start_date: date | None = None,
        end_date: date | None = None,
        on_day_start: Callable[[date], None] | None = None,
        on_day_end: Callable[[date], None] | None = None,
        inter_bar_sleep: float = 0.0,  # 0 = as-fast-as-possible
    ):
        self._instruments = instruments
        self._tick_queue = tick_queue
        self._start_date = start_date
        self._end_date = end_date
        self._on_day_start = on_day_start
        self._on_day_end = on_day_end
        self._inter_bar_sleep = inter_bar_sleep

        self._callbacks: list[Callable] = []
        self._subscribed_keys: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Synthetic option price cache: {option_key → last_price_rs}
        self._option_prices: dict[str, float] = {}
        # Map: underlying_key → list of subscribed option keys
        self._option_subscriptions: dict[str, list[str]] = defaultdict(list)

        self._bars_processed = 0
        self._active = False

    # ------------------------------------------------------------------ #
    # MarketDataFeed interface                                             #
    # ------------------------------------------------------------------ #

    def register_callback(self, cb: Callable) -> None:
        self._callbacks.append(cb)

    def subscribe(self, keys: list[str], mode: str = "full") -> None:
        """Called by TradingBot when it wants to receive prices for option keys."""
        for key in keys:
            if key not in self._subscribed_keys:
                self._subscribed_keys.add(key)
                logger.info("ReplayFeed: subscribed to {}", key)
                # Track which underlying this option belongs to
                # key format: NSE_FO|BANKNIFTY24DEC50000CE
                # We match it to NSE_INDEX|Nifty Bank etc. via symbol prefix
                for instr in self._instruments:
                    if instr.straddle_proxy and self._key_matches_underlying(key, instr.instrument_key):
                        self._option_subscriptions[instr.instrument_key].append(key)
                        break

    def start(self, symbols: list[str], mode: str = "full") -> None:
        """Start replay in a background thread."""
        self._active = True
        self._thread = threading.Thread(
            target=self._replay_loop,
            daemon=True,
            name="LiveReplayFeed",
        )
        self._thread.start()
        logger.info("LiveReplayFeed started | {} instruments", len(self._instruments))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._active = False

    @property
    def bars_processed(self) -> int:
        return self._bars_processed

    # ------------------------------------------------------------------ #
    # Internal replay logic                                                #
    # ------------------------------------------------------------------ #

    def _replay_loop(self) -> None:
        """Main loop: merge all instruments and replay chronologically."""
        try:
            # Load all CSVs into a merged time-indexed structure
            all_bars: list[tuple[datetime, str, ReplayInstrument, pd.Series]] = []
            for instr in self._instruments:
                df = self._load_csv(instr)
                if df is None or df.empty:
                    logger.warning("No data loaded for {}", instr.instrument_key)
                    continue
                for ts, row in df.iterrows():
                    all_bars.append((ts, instr.instrument_key, instr, row))

            if not all_bars:
                logger.error("LiveReplayFeed: no bars to replay")
                return

            # Sort chronologically (all instruments merged)
            all_bars.sort(key=lambda x: x[0])
            logger.info("LiveReplayFeed: {} total bars to replay", len(all_bars))

            current_day: date | None = None
            bar_day_cache: dict[str, list] = defaultdict(list)

            for ts, ik, instr, row in all_bars:
                if self._stop_event.is_set():
                    break

                bar_date = ts.date()

                # Day boundary: SOD / EOD hooks
                if bar_date != current_day:
                    if current_day is not None and self._on_day_end:
                        try:
                            self._on_day_end(current_day)
                        except Exception as exc:
                            logger.warning("on_day_end error: {}", exc)
                        # Small pause to let EOD square-off settle
                        threading.Event().wait(0.05)

                    current_day = bar_date
                    bar_day_cache.clear()

                    if self._on_day_start:
                        try:
                            self._on_day_start(current_day)
                        except Exception as exc:
                            logger.warning("on_day_start error: {}", exc)

                # Update simulated clock to bar timestamp
                set_sim_time(ts)

                # Skip out-of-window bars
                if not self._is_market_hours(ts):
                    continue

                # Convert price to rupees (feed always in rupees)
                close_rs = row["close"] / 100.0 if instr.prices_in_paisa else row["close"]
                open_rs  = row["open"]  / 100.0 if instr.prices_in_paisa else row["open"]
                high_rs  = row["high"]  / 100.0 if instr.prices_in_paisa else row["high"]
                low_rs   = row["low"]   / 100.0 if instr.prices_in_paisa else row["low"]
                vol      = int(row.get("volume", 0))

                # Emit OHLC ticks for this bar
                # Open at bar start, then high/low interleaved, close at bar end
                bar_start = ts
                bar_end   = ts + timedelta(seconds=59)
                mid       = ts + timedelta(seconds=30)

                ticks = [
                    (bar_start, open_rs,  vol // 4),
                    (mid - timedelta(seconds=10), high_rs, vol // 4),
                    (mid,                          low_rs,  vol // 4),
                    (bar_end,                      close_rs, vol - 3 * (vol // 4)),
                ]

                for tick_ts, price_rs, tick_vol in ticks:
                    tick = _make_tick(ik, price_rs, tick_ts, tick_vol)
                    self._emit_tick(tick)

                # For straddle proxies: also emit synthetic option prices
                if instr.straddle_proxy:
                    self._emit_proxy_options(ik, close_rs, ts, row, instr)

                self._bars_processed += 1

                if self._inter_bar_sleep > 0:
                    threading.Event().wait(self._inter_bar_sleep)

            # Final EOD
            if current_day is not None and self._on_day_end:
                try:
                    self._on_day_end(current_day)
                except Exception as exc:
                    logger.warning("Final on_day_end error: {}", exc)

            logger.info(
                "LiveReplayFeed replay complete | {} bars processed",
                self._bars_processed,
            )

        except Exception as exc:
            logger.error("LiveReplayFeed fatal error: {}", exc)
            import traceback
            logger.debug(traceback.format_exc())

    def _load_csv(self, instr: ReplayInstrument) -> pd.DataFrame | None:
        try:
            df = pd.read_csv(instr.csv_path, parse_dates=["timestamp"])
            df = df.rename(columns={"timestamp": "timestamp"})
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
                # Ensure IST timezone
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
                else:
                    df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
                df = df.set_index("timestamp")
            elif df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    df.index = df.index.tz_localize(IST)
                else:
                    df.index = df.index.tz_convert(IST)
            else:
                # First column is timestamp
                df.index = pd.to_datetime(df.iloc[:, 0])
                df = df.iloc[:, 1:]
                if df.index.tz is None:
                    df.index = df.index.tz_localize(IST)
                else:
                    df.index = df.index.tz_convert(IST)

            df = df.sort_index()

            # Date filtering
            if self._start_date:
                df = df[df.index.date >= self._start_date]
            if self._end_date:
                df = df[df.index.date <= self._end_date]

            logger.info("Loaded {} bars from {} for {}", len(df), instr.csv_path, instr.instrument_key)
            return df
        except Exception as exc:
            logger.error("Failed to load {}: {}", instr.csv_path, exc)
            return None

    def _emit_tick(self, tick: dict) -> None:
        """Put tick in queue AND fire all registered callbacks."""
        self._tick_queue.put(tick)
        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception as exc:
                logger.debug("Replay callback error: {}", exc)

    def _emit_proxy_options(
        self,
        underlying_key: str,
        underlying_close_rs: float,
        ts: datetime,
        row: pd.Series,
        instr: ReplayInstrument,
    ) -> None:
        """Emit synthetic ATM option prices for subscribed option keys."""
        subscribed = self._option_subscriptions.get(underlying_key, [])
        if not subscribed:
            return

        # ATR proxy: use high-low range as single-bar pseudo-ATR
        high_rs = row["high"] / 100.0 if instr.prices_in_paisa else row["high"]
        low_rs  = row["low"]  / 100.0 if instr.prices_in_paisa else row["low"]
        bar_range = high_rs - low_rs

        # Synthetic ATM premium ≈ max(bar_range * 1.5, underlying_close * 0.005)
        atm_premium_rs = max(bar_range * 1.5, underlying_close_rs * 0.005)

        # Intraday decay: premium decays linearly from 10:00 to 14:00
        decay_factor = self._intraday_decay(ts)
        premium_rs = atm_premium_rs * decay_factor

        for opt_key in subscribed:
            tick = _make_tick(opt_key, max(premium_rs, 0.05), ts)
            self._emit_tick(tick)
            self._option_prices[opt_key] = premium_rs

    @staticmethod
    def _intraday_decay(ts: datetime) -> float:
        """Linear decay factor from 1.0 at 10:00 to 0.2 at 14:00."""
        t = ts.time()
        start = time(10, 0)
        end   = time(14, 0)
        if t <= start:
            return 1.0
        if t >= end:
            return 0.2
        elapsed = (t.hour * 60 + t.minute) - (start.hour * 60 + start.minute)
        total   = (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
        return 1.0 - 0.8 * (elapsed / total)

    @staticmethod
    def _is_market_hours(ts: datetime) -> bool:
        t = ts.time()
        return time(9, 15) <= t <= time(15, 30)

    @staticmethod
    def _key_matches_underlying(option_key: str, underlying_key: str) -> bool:
        """Rough check: BANKNIFTY option belongs to NSE_INDEX|Nifty Bank."""
        mapping = {
            "NSE_INDEX|Nifty Bank": "BANKNIFTY",
            "NSE_INDEX|Nifty 50":   "NIFTY",
            "NSE_INDEX|Nifty Fin Service": "FINNIFTY",
        }
        symbol = mapping.get(underlying_key, "")
        return bool(symbol) and symbol in option_key.upper()
