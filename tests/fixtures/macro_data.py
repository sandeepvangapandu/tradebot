"""Macro-economic data fixture factories.

Generates deterministic synthetic time-series DataFrames for VIX, USD/INR,
Crude oil, and FII/DII flow data. Designed for backtesting and scenario
harnesses that need realistic macro context without live data feeds.

All values use realistic Indian-market calibration:
  - VIX  : typically 12–30, spikes to 40+ in crisis
  - USDINR: typically 82–86 range
  - Crude : WTI equivalent, typically USD 65–90 per barrel
  - FII/DII flows: in INR crore (1 crore = 10 million)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VIX_PARAMS: dict[str, dict] = {
    "low":    {"base": 11.5, "drift": -0.0005, "vol": 0.008},
    "normal": {"base": 16.0, "drift":  0.0000, "vol": 0.015},
    "high":   {"base": 24.0, "drift":  0.0003, "vol": 0.025},
    "spike":  {"base": 34.0, "drift":  0.0010, "vol": 0.060},
}


def _trading_dates(start: datetime, end: datetime) -> list[datetime]:
    """Return weekday dates between start and end (inclusive)."""
    dates: list[datetime] = []
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_d = end.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= end_d:
        if current.weekday() < 5:  # Monday–Friday
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _gbm_series(
    dates: list[datetime],
    base: float,
    drift: float,
    vol: float,
    rng: np.random.Generator,
    floor: float = 0.1,
) -> np.ndarray:
    """Generate a GBM price series."""
    n = len(dates)
    log_returns = (drift - 0.5 * vol**2) + vol * rng.standard_normal(n)
    prices = base * np.exp(np.cumsum(log_returns))
    prices = np.concatenate([[base], prices])[:-1]
    return np.maximum(prices, floor)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

def make_vix_series(
    start: datetime,
    end: datetime,
    regime: str = "normal",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic India VIX daily time-series.

    Args:
        start: Start date.
        end: End date.
        regime: One of ``"low"``, ``"normal"``, ``"high"``, ``"spike"``.
            Controls the base level and volatility of the VIX path.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``date`` (datetime), ``vix`` (float).
        Rows are trading days only (Mon–Fri).
    """
    if regime not in _VIX_PARAMS:
        raise ValueError(f"regime must be one of {list(_VIX_PARAMS)}, got {regime!r}")

    params = _VIX_PARAMS[regime]
    rng = np.random.default_rng(seed)
    dates = _trading_dates(start, end)
    if not dates:
        return pd.DataFrame(columns=["date", "vix"])

    vix = _gbm_series(
        dates,
        base=params["base"],
        drift=params["drift"],
        vol=params["vol"],
        rng=rng,
        floor=5.0,  # VIX rarely goes below 5
    )

    # For spike regime, add a one-time spike in the middle
    if regime == "spike" and len(dates) > 5:
        spike_idx = len(dates) // 3
        vix[spike_idx] = min(80.0, vix[spike_idx] * 1.8)

    return pd.DataFrame({"date": dates, "vix": np.round(vix, 2)})


def make_usdinr_series(
    start: datetime,
    end: datetime,
    trend: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic USD/INR exchange rate daily time-series.

    Args:
        start: Start date.
        end: End date.
        trend: Daily log drift (e.g. 0.0002 for mild depreciation,
            -0.0002 for mild appreciation). Defaults to 0.0 (random walk).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``date`` (datetime), ``usdinr`` (float).
    """
    rng = np.random.default_rng(seed)
    dates = _trading_dates(start, end)
    if not dates:
        return pd.DataFrame(columns=["date", "usdinr"])

    usdinr = _gbm_series(
        dates,
        base=84.20,   # current realistic rate
        drift=trend,
        vol=0.004,    # ~0.4% daily vol — typical for INR
        rng=rng,
        floor=60.0,
    )
    return pd.DataFrame({"date": dates, "usdinr": np.round(usdinr, 4)})


def make_crude_series(
    start: datetime,
    end: datetime,
    trend: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic Crude Oil (WTI) daily price time-series in USD/barrel.

    Args:
        start: Start date.
        end: End date.
        trend: Daily log drift (positive = rising oil prices).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``date`` (datetime), ``crude_usd`` (float).
    """
    rng = np.random.default_rng(seed)
    dates = _trading_dates(start, end)
    if not dates:
        return pd.DataFrame(columns=["date", "crude_usd"])

    crude = _gbm_series(
        dates,
        base=76.50,   # typical WTI level
        drift=trend,
        vol=0.018,    # ~1.8% daily vol
        rng=rng,
        floor=20.0,
    )
    return pd.DataFrame({"date": dates, "crude_usd": np.round(crude, 2)})


def make_fii_dii_flows(
    start_date: datetime,
    end_date: datetime,
    regime: str = "neutral",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic FII and DII net flows (in INR crore) by trading day.

    Args:
        start_date: Start date.
        end_date: End date.
        regime: Market regime affecting flow bias.
            ``"bullish"`` — strong FII buying, moderate DII;
            ``"bearish"`` — heavy FII selling, DII buying;
            ``"neutral"`` — mixed flows around zero;
            ``"panic"``   — extreme FII selling, frantic DII buying.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
        ``date`` (datetime), ``fii_net`` (float, crore),
        ``dii_net`` (float, crore), ``net_total`` (float, crore).
        Positive = buying, negative = selling.
    """
    _REGIME_PARAMS: dict[str, dict] = {
        "bullish": {"fii_mu": 1500,  "fii_sigma": 1000, "dii_mu":  400, "dii_sigma": 600},
        "bearish": {"fii_mu": -1200, "fii_sigma": 900,  "dii_mu":  800, "dii_sigma": 700},
        "neutral": {"fii_mu":    50, "fii_sigma": 1200, "dii_mu":  200, "dii_sigma": 500},
        "panic":   {"fii_mu": -3500, "fii_sigma": 2000, "dii_mu": 2500, "dii_sigma": 1500},
    }
    if regime not in _REGIME_PARAMS:
        raise ValueError(f"regime must be one of {list(_REGIME_PARAMS)}, got {regime!r}")

    params = _REGIME_PARAMS[regime]
    rng = np.random.default_rng(seed)
    dates = _trading_dates(start_date, end_date)
    if not dates:
        return pd.DataFrame(columns=["date", "fii_net", "dii_net", "net_total"])

    n = len(dates)
    fii_net = rng.normal(params["fii_mu"], params["fii_sigma"], n)
    dii_net = rng.normal(params["dii_mu"], params["dii_sigma"], n)
    net_total = fii_net + dii_net

    return pd.DataFrame({
        "date": dates,
        "fii_net": np.round(fii_net, 2),
        "dii_net": np.round(dii_net, 2),
        "net_total": np.round(net_total, 2),
    })
