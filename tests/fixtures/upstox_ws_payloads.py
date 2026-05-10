"""Upstox V3 WebSocket payload factories.

All payloads match the structure consumed by ``src/data/websocket_feed.py``
and the Upstox MarketDataStreamerV3 wire format, so they parse cleanly
in integration tests without hitting the real API.

All price values are in **paisa** (1/100 of a Rupee) to match the
internal convention of the trading bot.
"""

from __future__ import annotations

import math
import time
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts_ms() -> int:
    """Current epoch time in milliseconds (IST-agnostic)."""
    return int(time.time() * 1000)


def _depth_entry(bid_paisa: int, ask_paisa: int, qty: int = 100) -> dict[str, Any]:
    return {
        "bid_quantity": qty,
        "bid_price": bid_paisa,
        "ask_quantity": qty,
        "ask_price": ask_paisa,
    }


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def make_ltpc_tick(
    instrument_key: str,
    ltp_paisa: int,
    ts_epoch_ms: int | None = None,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Return a minimal LTPC tick payload (Last Traded Price + Change).

    Args:
        instrument_key: Upstox instrument key, e.g. ``"NSE_EQ|INE002A01018"``.
        ltp_paisa: Last-traded price in paisa.
        ts_epoch_ms: Epoch timestamp in milliseconds.  Defaults to now.
        seed: Unused — kept for API symmetry with other factories.

    Returns:
        Dict matching the Upstox V3 LTPC feed format.
    """
    ts = ts_epoch_ms if ts_epoch_ms is not None else _ts_ms()
    ltp_rupees = ltp_paisa / 100.0
    return {
        "feeds": {
            instrument_key: {
                "feedType": "ltpc",
                "ltpc": {
                    "ltp": ltp_rupees,
                    "ltt": ts,
                    "ltq": 1,
                    "cp": ltp_rupees * 0.995,  # close_price ~0.5% below LTP
                },
            }
        },
        "currentTs": ts,
    }


def make_full_tick(
    instrument_key: str,
    ltp_paisa: int,
    ohlc_paisa: dict[str, int] | None = None,
    depth_5_levels: dict[str, list[dict[str, int]]] | None = None,
    ts_epoch_ms: int | None = None,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Return a full-mode tick payload with OHLC, volume, and 5-level depth.

    Args:
        instrument_key: Upstox instrument key.
        ltp_paisa: Last-traded price in paisa.
        ohlc_paisa: Dict with keys ``open``, ``high``, ``low``, ``close``
            in paisa.  Defaults to OHLC synthesised around ``ltp_paisa``.
        depth_5_levels: Dict with keys ``buy`` and ``sell``, each a list of
            5 depth entries ``{"price": int, "quantity": int, "orders": int}``.
            Values are in **paisa**.  Defaults to a synthetic spread.
        ts_epoch_ms: Epoch timestamp in milliseconds.
        seed: Unused — kept for API symmetry.

    Returns:
        Dict matching the Upstox V3 full-mode feed format.
    """
    ts = ts_epoch_ms if ts_epoch_ms is not None else _ts_ms()

    if ohlc_paisa is None:
        ohlc_paisa = {
            "open": int(ltp_paisa * 0.998),
            "high": int(ltp_paisa * 1.004),
            "low": int(ltp_paisa * 0.994),
            "close": int(ltp_paisa * 0.999),
        }

    ltp_r = ltp_paisa / 100.0

    if depth_5_levels is None:
        spread_step = max(5, int(ltp_paisa * 0.0001))  # ~0.01% of price
        depth_5_levels = {
            "buy": [
                {"price": (ltp_paisa - spread_step * (i + 1)) / 100.0, "quantity": 100 * (i + 1), "orders": i + 1}
                for i in range(5)
            ],
            "sell": [
                {"price": (ltp_paisa + spread_step * (i + 1)) / 100.0, "quantity": 100 * (i + 1), "orders": i + 1}
                for i in range(5)
            ],
        }

    return {
        "feeds": {
            instrument_key: {
                "feedType": "full",
                "ff": {
                    "marketFF": {
                        "ltpc": {
                            "ltp": ltp_r,
                            "ltt": ts,
                            "ltq": 50,
                            "cp": ohlc_paisa["close"] / 100.0,
                        },
                        "marketOHLC": {
                            "ohlc": [
                                {
                                    "interval": "1d",
                                    "open": ohlc_paisa["open"] / 100.0,
                                    "high": ohlc_paisa["high"] / 100.0,
                                    "low": ohlc_paisa["low"] / 100.0,
                                    "close": ohlc_paisa["close"] / 100.0,
                                    "volume": 1_200_000,
                                    "ts": ts - 3600_000,  # 1h ago
                                }
                            ]
                        },
                        "eFeedDetails": {
                            "atp": ltp_r * 1.001,
                            "cp": ohlc_paisa["close"] / 100.0,
                            "vtt": 5_000_000,
                            "tbq": 2_000_000,
                            "tsq": 1_800_000,
                            "oi": 0,
                            "change": ltp_r - ohlc_paisa["close"] / 100.0,
                            "pChange": (
                                (ltp_paisa - ohlc_paisa["close"]) / ohlc_paisa["close"] * 100
                                if ohlc_paisa["close"] != 0 else 0.0
                            ),
                        },
                        "depth": {
                            "buy": depth_5_levels["buy"],
                            "sell": depth_5_levels["sell"],
                        },
                    }
                },
            }
        },
        "currentTs": ts,
    }


def make_index_tick(
    instrument_key: str,
    ltp_paisa: int,
    ts_epoch_ms: int | None = None,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Return an index tick payload (no depth, no OI).

    Args:
        instrument_key: Upstox index instrument key,
            e.g. ``"NSE_INDEX|Nifty 50"``.
        ltp_paisa: Index value in paisa.
        ts_epoch_ms: Epoch timestamp in milliseconds.
        seed: Unused — kept for API symmetry.

    Returns:
        Dict matching the Upstox V3 index feed format.
    """
    ts = ts_epoch_ms if ts_epoch_ms is not None else _ts_ms()
    ltp_r = ltp_paisa / 100.0
    prev_close = ltp_r * 0.997

    return {
        "feeds": {
            instrument_key: {
                "feedType": "ltpc",
                "ltpc": {
                    "ltp": ltp_r,
                    "ltt": ts,
                    "ltq": 0,
                    "cp": prev_close,
                },
            }
        },
        "type": "live_feed",
        "currentTs": ts,
    }


def stream_ticks(
    instrument_key: str,
    start_paisa: int,
    count: int,
    drift: float = 0.001,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a time-ordered list of LTPC ticks with a geometric-Brownian walk.

    Args:
        instrument_key: Upstox instrument key.
        start_paisa: Starting price in paisa.
        count: Number of ticks to generate.
        drift: Per-tick proportional drift (e.g. 0.001 = 0.1%).
            Positive for upward bias, negative for downward bias.
            Noise standard deviation is ``abs(drift) * 2``.
        seed: Random seed for reproducibility.

    Returns:
        List of LTPC tick dicts ordered by ascending timestamp.
    """
    import random

    rng = random.Random(seed)
    sigma = abs(drift) * 2.0 if drift != 0 else 0.001

    price = float(start_paisa)
    base_ts = _ts_ms()
    ticks: list[dict[str, Any]] = []

    for i in range(count):
        # Geometric-Brownian-ish: price * exp(drift + noise)
        z = rng.gauss(0.0, 1.0)
        price = price * math.exp(drift + sigma * z)
        price = max(price, 100.0)  # never go below 1 rupee

        ts = base_ts + i * 500  # 500 ms between ticks
        ticks.append(make_ltpc_tick(instrument_key, int(price), ts, seed=seed))

    return ticks
