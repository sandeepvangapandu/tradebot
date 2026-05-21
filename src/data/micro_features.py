"""Tick-level microstructure feature engine for Indian options trading.

Computes five microstructure features from the rolling WebSocket tick buffer
maintained per instrument key:

1. bid_ask_spread_pct  — percentage spread between best bid and ask.
2. order_imbalance     — signed buy/sell pressure from market depth.
3. volume_burst        — current-tick volume relative to 20-tick rolling mean.
4. tick_momentum       — price return over the last 5 ticks (percentage).
5. oi_change_rate      — open-interest growth over the last 5 ticks (percentage).

All monetary values are carried in **paisa** (int) internally.  Only the
feature output values are floats.

Thread safety: a ``threading.Lock`` is held per instrument while the buffer is
mutated or read.  Cross-instrument operations (``get_all_features``) snapshot
the key list first so as not to hold multiple locks simultaneously.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum ticks before features can be computed.
_MIN_TICKS = 5

# Lookback for tick momentum and OI change rate.
_LOOKBACK_TICKS = 5

# Lookback for volume rolling average.
_VOLUME_LOOKBACK = 20

# Fallback spread when bid/ask are unavailable (liquid options estimate).
_FALLBACK_SPREAD_PCT: float = 0.1

# Feature clipping bounds.
_SPREAD_MAX: float = 5.0
_VOL_BURST_MAX: float = 5.0
_MOMENTUM_CLIP: float = 2.0   # percent
_OI_RATE_CLIP: float = 5.0    # percent


# ---------------------------------------------------------------------------
# Internal data structure
# ---------------------------------------------------------------------------

@dataclass
class _TickRecord:
    """A single normalised tick snapshot stored in the rolling buffer.

    All prices are in paisa (int).  Volume and OI are raw integers from the
    Upstox WebSocket payload.
    """

    ltp: int                          # last traded price, paisa
    volume: int                       # traded volume at this tick
    oi: int                           # open interest

    # Bid/ask — None when the tick does not carry Level 1 data.
    bid: int | None = None            # best bid price, paisa
    ask: int | None = None            # best ask price, paisa

    # Market depth — list of {"bid_q": int, "ask_q": int, ...} dicts (L2)
    depth: list[dict[str, Any]] | None = None


@dataclass
class _InstrumentBuffer:
    """Per-instrument rolling tick buffer with a dedicated lock.

    Args:
        maxlen: Maximum number of ticks to retain.
    """

    maxlen: int
    ticks: deque[_TickRecord] = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.ticks = deque(maxlen=self.maxlen)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MicroFeatureEngine:
    """Rolling-buffer microstructure feature engine.

    Maintains a per-instrument deque of the most recent *buffer_size* ticks
    and exposes five normalised float features derived from that buffer.

    Args:
        buffer_size: Number of ticks to retain per instrument.  Must be >= 20
            (the volume rolling-average lookback).  Defaults to 50.
    """

    def __init__(self, buffer_size: int = 50) -> None:
        if buffer_size < _VOLUME_LOOKBACK:
            raise ValueError(
                f"buffer_size must be >= {_VOLUME_LOOKBACK} "
                f"(volume rolling-average lookback); got {buffer_size}."
            )
        self._buffer_size = buffer_size
        # Outer dict is protected by _registry_lock only during key insertion.
        # Per-instrument locks guard individual buffer mutations.
        self._registry_lock = threading.Lock()
        self._buffers: dict[str, _InstrumentBuffer] = {}

    # ------------------------------------------------------------------
    # Tick ingestion
    # ------------------------------------------------------------------

    def on_tick(self, instrument_key: str, tick: dict) -> None:
        """Consume a single market tick and append it to the rolling buffer.

        Args:
            instrument_key: Upstox instrument identifier
                (e.g. ``"NSE_FO|OPTIDX..."``).
            tick: WebSocket tick dict.  Expected keys:

                * ``ltp`` (int, paisa) — **required**
                * ``volume`` (int) — **required**
                * ``oi`` (int) — open interest; defaults to 0 if absent
                * ``best_bid_price`` (int, paisa) — optional Level 1 bid
                * ``best_ask_price`` (int, paisa) — optional Level 1 ask
                * ``depth`` (list[dict]) — optional Level 2 depth rows
        """
        ltp = tick.get("ltp")
        volume = tick.get("volume")

        if ltp is None or volume is None:
            logger.warning(
                "micro_features: skipping tick for %s — missing ltp or volume",
                instrument_key,
            )
            return

        record = _TickRecord(
            ltp=int(ltp),
            volume=int(volume),
            oi=int(tick.get("oi", 0)),
            bid=tick.get("best_bid_price"),
            ask=tick.get("best_ask_price"),
            depth=tick.get("depth"),
        )

        buf = self._get_or_create_buffer(instrument_key)
        with buf.lock:
            buf.ticks.append(record)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_features(self, instrument_key: str) -> dict[str, float] | None:
        """Compute the five microstructure features for an instrument.

        Returns ``None`` if fewer than :data:`_MIN_TICKS` ticks have been
        buffered (no partial / unreliable results are surfaced).

        Args:
            instrument_key: Upstox instrument identifier.

        Returns:
            Dict with keys:
                * ``bid_ask_spread_pct``
                * ``order_imbalance``
                * ``volume_burst``
                * ``tick_momentum``
                * ``oi_change_rate``

            All values are ``float``.  Returns ``None`` when insufficient data.
        """
        buf = self._buffers.get(instrument_key)
        if buf is None:
            return None

        with buf.lock:
            ticks = list(buf.ticks)  # snapshot inside the lock

        if len(ticks) < _MIN_TICKS:
            return None

        return {
            "bid_ask_spread_pct": self._bid_ask_spread_pct(ticks),
            "order_imbalance": self._order_imbalance(ticks),
            "volume_burst": self._volume_burst(ticks),
            "tick_momentum": self._tick_momentum(ticks),
            "oi_change_rate": self._oi_change_rate(ticks),
        }

    # ------------------------------------------------------------------
    # Entry confirmation gate
    # ------------------------------------------------------------------

    def is_entry_confirmed(
        self, instrument_key: str, signal_type: str
    ) -> tuple[bool, str]:
        """Microstructure gate for entry confirmation.

        Criteria:
            **CE / BUY**:
                * ``order_imbalance > 0.1``   (buying pressure present)
                * ``volume_burst > 1.3``       (above-average participation)
                * ``tick_momentum > 0``        (recent upward price movement)

            **PE / SELL**:
                * ``order_imbalance < -0.1``  (selling pressure present)
                * ``volume_burst > 1.3``      (above-average participation)
                * ``tick_momentum < 0``       (recent downward price movement)

        When fewer than :data:`_MIN_TICKS` ticks are buffered the gate returns
        ``(True, 'insufficient data')`` so as not to block signals during
        session warm-up.

        Args:
            instrument_key: Upstox instrument identifier.
            signal_type: ``'CE'`` / ``'BUY'`` or ``'PE'`` / ``'SELL'``
                (case-insensitive).

        Returns:
            ``(confirmed: bool, reason: str)``
        """
        features = self.get_features(instrument_key)
        if features is None:
            return True, "insufficient data"

        sig = signal_type.upper()
        imbalance = features["order_imbalance"]
        burst = features["volume_burst"]
        momentum = features["tick_momentum"]

        if sig in ("CE", "BUY"):
            if imbalance <= 0.1:
                return False, (
                    f"order_imbalance={imbalance:.3f} not > 0.1 for CE/BUY"
                )
            if burst <= 1.3:
                return False, (
                    f"volume_burst={burst:.3f} not > 1.3 for CE/BUY"
                )
            if momentum <= 0:
                return False, (
                    f"tick_momentum={momentum:.4f} not > 0 for CE/BUY"
                )
            return True, (
                f"confirmed CE/BUY: imbalance={imbalance:.3f} "
                f"burst={burst:.3f} momentum={momentum:.4f}"
            )

        if sig in ("PE", "SELL"):
            if imbalance >= -0.1:
                return False, (
                    f"order_imbalance={imbalance:.3f} not < -0.1 for PE/SELL"
                )
            if burst <= 1.3:
                return False, (
                    f"volume_burst={burst:.3f} not > 1.3 for PE/SELL"
                )
            if momentum >= 0:
                return False, (
                    f"tick_momentum={momentum:.4f} not < 0 for PE/SELL"
                )
            return True, (
                f"confirmed PE/SELL: imbalance={imbalance:.3f} "
                f"burst={burst:.3f} momentum={momentum:.4f}"
            )

        logger.warning(
            "micro_features.is_entry_confirmed: unknown signal_type=%r — "
            "defaulting to not confirmed",
            signal_type,
        )
        return False, f"unknown signal_type={signal_type!r}"

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def get_all_features(self) -> dict[str, dict[str, float]]:
        """Return features for every tracked instrument that has enough data.

        Instruments with fewer than :data:`_MIN_TICKS` ticks are omitted.

        Returns:
            Mapping of ``instrument_key`` to feature dict.  May be empty.
        """
        with self._registry_lock:
            keys = list(self._buffers.keys())

        result: dict[str, dict[str, float]] = {}
        for key in keys:
            features = self.get_features(key)
            if features is not None:
                result[key] = features
        return result

    # ------------------------------------------------------------------
    # Private helpers — feature computation
    # ------------------------------------------------------------------

    @staticmethod
    def _bid_ask_spread_pct(ticks: list[_TickRecord]) -> float:
        """Compute the bid-ask spread percentage from the most recent tick.

        Formula: ``(ask - bid) / mid * 100``.

        Falls back to :data:`_FALLBACK_SPREAD_PCT` (0.1%) when the latest
        tick carries no bid/ask information.

        Args:
            ticks: Snapshot of the instrument's tick buffer (non-empty).

        Returns:
            Spread percentage in range [0, :data:`_SPREAD_MAX`].
        """
        latest = ticks[-1]
        bid = latest.bid
        ask = latest.ask

        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return _FALLBACK_SPREAD_PCT

        mid = (bid + ask) / 2.0
        if mid <= 0:
            return _FALLBACK_SPREAD_PCT

        spread_pct = (ask - bid) / mid * 100.0
        return float(min(max(spread_pct, 0.0), _SPREAD_MAX))

    @staticmethod
    def _order_imbalance(ticks: list[_TickRecord]) -> float:
        """Compute order-book imbalance from the most recent tick's depth.

        Formula: ``(buy_qty - sell_qty) / (buy_qty + sell_qty)``.

        Aggregates across all available Level-2 depth rows.  Falls back to
        ``0.0`` when depth is unavailable.

        Args:
            ticks: Snapshot of the instrument's tick buffer (non-empty).

        Returns:
            Imbalance in range [-1.0, +1.0].  Positive indicates net buying
            pressure.
        """
        latest = ticks[-1]
        depth = latest.depth

        if not depth:
            return 0.0

        buy_qty: int = 0
        sell_qty: int = 0

        for row in depth:
            # Upstox V3 depth rows use bid_q / ask_q keys (matching depth_feed.py
            # conventions); accept 0 when a key is absent.
            buy_qty += int(row.get("bid_q", 0))
            sell_qty += int(row.get("ask_q", 0))

        total = buy_qty + sell_qty
        if total == 0:
            return 0.0

        return float((buy_qty - sell_qty) / total)

    @staticmethod
    def _volume_burst(ticks: list[_TickRecord]) -> float:
        """Ratio of current-tick volume to the rolling 20-tick average.

        A value of ``1.0`` means volume is exactly at its recent average.
        Values ``> 2.0`` indicate unusual activity.

        Args:
            ticks: Snapshot of the instrument's tick buffer (non-empty).
                Must contain at least 1 tick; fewer than 20 ticks yield a
                ratio relative to whatever is available.

        Returns:
            Ratio in range [0.0, :data:`_VOL_BURST_MAX`].
        """
        current_vol = ticks[-1].volume

        # Use up to the last _VOLUME_LOOKBACK ticks (excluding current tick) for
        # the rolling average baseline; include the current tick if fewer than 20
        # historical ticks are available.
        lookback = ticks[-_VOLUME_LOOKBACK:]  # up to 20 ticks including current
        volumes = [t.volume for t in lookback]

        if not volumes:
            return 1.0

        avg_vol = float(np.mean(volumes))
        if avg_vol <= 0:
            return 1.0

        ratio = current_vol / avg_vol
        return float(min(ratio, _VOL_BURST_MAX))

    @staticmethod
    def _tick_momentum(ticks: list[_TickRecord]) -> float:
        """Price return over the past :data:`_LOOKBACK_TICKS` ticks.

        Formula: ``(ltp_now - ltp_5_ago) / ltp_5_ago * 100``.

        Args:
            ticks: Snapshot of the instrument's tick buffer.  Must contain at
                least :data:`_MIN_TICKS` entries.

        Returns:
            Percentage return clipped to
            [−:data:`_MOMENTUM_CLIP`, +:data:`_MOMENTUM_CLIP`].
        """
        ltp_now = ticks[-1].ltp
        # ticks[-_LOOKBACK_TICKS] is the tick 5 steps behind the current one.
        ltp_past = ticks[-_LOOKBACK_TICKS].ltp

        if ltp_past <= 0:
            return 0.0

        momentum = (ltp_now - ltp_past) / ltp_past * 100.0
        return float(np.clip(momentum, -_MOMENTUM_CLIP, _MOMENTUM_CLIP))

    @staticmethod
    def _oi_change_rate(ticks: list[_TickRecord]) -> float:
        """Open-interest growth rate over the past :data:`_LOOKBACK_TICKS` ticks.

        Formula: ``(oi_now - oi_5_ago) / oi_5_ago * 100``.

        A positive value means new positions are being opened (OI expanding).
        A negative value means positions are being closed (OI contracting).

        Args:
            ticks: Snapshot of the instrument's tick buffer.  Must contain at
                least :data:`_MIN_TICKS` entries.

        Returns:
            Percentage change clipped to
            [−:data:`_OI_RATE_CLIP`, +:data:`_OI_RATE_CLIP`].
        """
        oi_now = ticks[-1].oi
        oi_past = ticks[-_LOOKBACK_TICKS].oi

        if oi_past <= 0:
            return 0.0

        rate = (oi_now - oi_past) / oi_past * 100.0
        return float(np.clip(rate, -_OI_RATE_CLIP, _OI_RATE_CLIP))

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _get_or_create_buffer(self, instrument_key: str) -> _InstrumentBuffer:
        """Return the existing buffer for an instrument or create a new one.

        Creation is serialised through :attr:`_registry_lock` to prevent
        duplicate-buffer races under concurrent first ticks.

        Args:
            instrument_key: Upstox instrument identifier.

        Returns:
            The :class:`_InstrumentBuffer` for that instrument.
        """
        # Fast path — no lock needed once the key exists.
        buf = self._buffers.get(instrument_key)
        if buf is not None:
            return buf

        with self._registry_lock:
            # Re-check after acquiring the lock (another thread may have won).
            buf = self._buffers.get(instrument_key)
            if buf is None:
                buf = _InstrumentBuffer(maxlen=self._buffer_size)
                self._buffers[instrument_key] = buf
                logger.debug(
                    "micro_features: created buffer for %s (size=%d)",
                    instrument_key,
                    self._buffer_size,
                )
        return buf
