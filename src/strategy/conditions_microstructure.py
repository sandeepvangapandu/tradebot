"""Microstructure condition evaluators (CVD divergence, aggressor flow).

These helpers wrap :class:`~src.data.tick_metrics.TickMetricsAggregator` into
callable boolean/numeric conditions that strategy configs can reference.
They are intentionally kept **separate** from ``conditions.py`` so that
microstructure dependencies can be added or removed without touching the
existing condition evaluation pipeline.

Wiring into ``conditions.py`` is left to the conditions-wiring agent.

All functions are pure (no side-effects) and thread-safe provided the
underlying aggregator is thread-safe (which it is — it uses an internal lock).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.tick_metrics import TickMetricsAggregator


def cvd_divergence(
    instrument_key: str,
    aggregator: "TickMetricsAggregator",
    divergence_type: str = "bearish",
    lookback: int = 30,
) -> bool:
    """Return True when the detected divergence matches *divergence_type*.

    Args:
        instrument_key: Upstox instrument identifier (e.g. ``"NSE_EQ|INE002A01018"``).
        aggregator: Live :class:`~src.data.tick_metrics.TickMetricsAggregator` instance.
        divergence_type: ``'bearish'`` or ``'bullish'``.
        lookback: How many recent minute buckets to consider (passed to
            :meth:`~src.data.tick_metrics.TickMetricsAggregator.detect_divergence`).

    Returns:
        ``True`` when a divergence of *divergence_type* is currently detected.
    """
    result = aggregator.detect_divergence(instrument_key, lookback_minutes=lookback)
    return bool(result.get("divergent") and result.get("type") == divergence_type)


def cvd_above_threshold(
    instrument_key: str,
    aggregator: "TickMetricsAggregator",
    threshold: int,
) -> bool:
    """Return True when running CVD exceeds *threshold*.

    A positive threshold tests for net buy pressure; a negative threshold
    tests for net sell pressure (i.e. ``cvd_above_threshold(key, agg, -500)``
    is ``True`` when CVD > -500, which includes all positive CVD values).

    Args:
        instrument_key: Upstox instrument identifier.
        aggregator: Live aggregator instance.
        threshold: CVD level in volume units (shares or contracts).

    Returns:
        ``True`` when ``running_cvd > threshold``.
    """
    return aggregator.get_running_cvd(instrument_key) > threshold


def aggressor_imbalance_pct(
    instrument_key: str,
    aggregator: "TickMetricsAggregator",
    lookback_min: int = 5,
) -> float:
    """Compute the buy/sell imbalance as a signed percentage over recent minutes.

    Formula::

        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) * 100

    A result of **+100** means all volume was buy-aggressed; **-100** means
    all volume was sell-aggressed; **0** means perfectly balanced.

    The metric is computed from the *in-memory unflushed buckets* of the
    last *lookback_min* minutes.  Flushed minutes are not re-read from the
    DB here; for historical accuracy query ``tick_metrics_minute`` directly.

    Args:
        instrument_key: Upstox instrument identifier.
        aggregator: Live aggregator instance.
        lookback_min: How many recent minute buckets to include.

    Returns:
        Signed imbalance percentage in ``[-100.0, +100.0]``.  Returns ``0.0``
        when there is insufficient data.
    """
    # Access the internal state via the public API where possible; we need
    # the raw buckets, so we rely on the aggregator's lock-protected state.
    # We call detect_divergence which reads all buckets, but for imbalance we
    # need raw buy/sell volumes.  Access internal state directly (same module
    # package) — guarded by the aggregator's internal lock.

    with aggregator._lock:  # noqa: SLF001  (legitimate internal access)
        state = aggregator._state.get(instrument_key)  # noqa: SLF001
        if state is None or not state.buckets:
            return 0.0

        sorted_ts = sorted(state.buckets.keys())
        window = sorted_ts[-lookback_min:]

        total_buy = sum(state.buckets[ts].buy_volume for ts in window)
        total_sell = sum(state.buckets[ts].sell_volume for ts in window)

    total = total_buy + total_sell
    if total == 0:
        return 0.0

    return (total_buy - total_sell) / total * 100.0
