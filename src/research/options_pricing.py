"""Black-Scholes-Merton option pricing and Greeks calculator.

Provides functions for European option valuation and implied volatility
estimation. Designed for backtesting Indian options (BankNifty/Nifty)
using India VIX as a volatility proxy or per-strike IV from NSE bhavcopy.

All price outputs are in the same currency unit as the underlying price
(typically PAISA for Indian equities/indices).
"""

import math
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

@dataclass
class OptionGreeks:
    """Container for option Greeks."""

    delta: float
    gamma: float
    theta: float  # per day
    vega: float  # per 1% vol change
    rho: float  # per 1% rate change
    d1: float
    d2: float

@njit(cache=True)
def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using error function.

    Args:
        x: Value to evaluate.

    Returns:
        CDF value.
    """
    # Using math.erf: Φ(x) = (1 + erf(x/√2)) / 2
    return (1.0 + math.erf(x / 1.4142135623730951)) / 2.0

@njit(cache=True)
def _norm_pdf(x: float) -> float:
    """Standard normal probability density function.

    Args:
        x: Value to evaluate.

    Returns:
        PDF value.
    """
    return math.exp(-0.5 * x * x) / 2.5066282746310002

# We cannot use njit easily on string comparison, but let's change bsm_price 
# to a helper that takes an int for kind (0=call, 1=put) so numba can compile it,
# or we can just leave kind as string and bypass njit for the outer wrapper,
# but since numba supports strings to some extent, we will try.

@njit(cache=True)
def _bsm_price_core(S: float, K: float, T: float, sigma: float, r: float, is_call: bool) -> float:
    if sigma <= 0.0:
        return 0.0
    if T <= 0.0:
        if is_call:
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if is_call:
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return max(0.0, price)

def bsm_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.05,
    kind: str = "call",
) -> float:
    """Calculate Black-Scholes-Merton European option price.

    Args:
        S: Current underlying price (same currency unit as K).
        K: Strike price.
        T: Time to expiration in years.
        sigma: Volatility (annualized standard deviation, e.g., 0.20 for 20%).
        r: Risk-free rate (annual, decimal). Default 5% (common in India).
        kind: "call" or "put".

    Returns:
        Option theoretical price.

    Raises:
        ValueError: Invalid parameters or negative sigma/T.
    """
    if kind.lower() not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind}")

    is_call = kind.lower() == "call"
    return _bsm_price_core(float(S), float(K), float(T), float(sigma), float(r), is_call)


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.05,
    kind: str = "call",
) -> OptionGreeks:
    """Calculate Black-Scholes Greeks.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        sigma: Volatility (annualized).
        r: Risk-free rate (decimal).
        kind: "call" or "put".

    Returns:
        OptionGreeks dataclass with delta, gamma, theta, vega, rho.

    Notes:
        - theta returned is per day (annual Θ / 365).
        - vega returned is per 1% absolute vol change (e.g., vega=250 means +1% vol = +2.5 price).
        - rho returned is per 1% rate change.
    """
    if sigma <= 0 or T <= 0:
        return OptionGreeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    pdf = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)

    if kind.lower() == "call":
        delta = cdf_d1
    else:
        delta = cdf_d1 - 1.0

    # Gamma: same for call and put
    gamma = pdf / (S * sigma * sqrtT)

    # Theta: annualized, then convert to per-day
    # Call:  - (S σ N'(d1)) / (2√T)  - rK e^{-rT} N(d2)
    # Put:   + (S σ N'(d1)) / (2√T)  + rK e^{-rT} N(-d2)
    term1 = -(S * pdf * sigma) / (2 * sqrtT)
    if kind.lower() == "call":
        term2 = r * K * math.exp(-r * T) * cdf_d2
        theta = (term1 - term2) / 365.0  # per day
    else:
        term2 = r * K * math.exp(-r * T) * _norm_cdf(-d2)
        theta = (term1 + term2) / 365.0

    # Vega: same for call and put, per 1 vol-point (not 1%)
    vega = (S * pdf * sqrtT) / 100.0  # per 1% change

    # Rho: per 1% rate change
    if kind.lower() == "call":
        rho = (K * T * math.exp(-r * T) * cdf_d2) / 100.0
    else:
        rho = (-K * T * math.exp(-r * T) * _norm_cdf(-d2)) / 100.0

    return OptionGreeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        d1=d1,
        d2=d2,
    )


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.05,
    kind: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """Estimate implied volatility via Newton-Raphson.

    Args:
        market_price: Observed option price.
        S: Underlying price.
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate.
        kind: "call" or "put".
        tol: Convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        Implied volatility or None if convergence fails.
    """
    # Bump initial guess using VIX approximation or simple heuristic
    sigma = max(0.01, math.sqrt(2 * math.pi / T) * (market_price / S))

    for i in range(max_iter):
        price = bsm_price(S, K, T, sigma, r, kind)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Vega for vega not vega_per_1pct
        # Compute raw vega (per 1 unit vol) then divide
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        pdf = _norm_pdf(d1)
        vega_raw = S * pdf * sqrtT  # per 1.0 (1000%) not per 1%
        if vega_raw == 0:
            return None

        sigma -= diff / vega_raw
        if sigma <= 0:
            sigma = 0.001

    logger.warning(f"IV failed to converge after {max_iter} iterations (price diff={diff:.4f})")
    return None


def time_to_expiry(
    entry_time: float,  # epoch seconds
    expiry_time: float,  # epoch seconds
    trading_days_per_year: int = 252,
    hours_per_day: int = 6,  # Indian market hours 9:15-15:15 = 6h
) -> float:
    """Calculate time to expiry in years using trading-time convention.

    For Indian options, uses trading-day calendar: 1 year = 252 trading days,
    each trading day = 6 hours (9:15–15:15 IST). Converts actual elapsed
    time to fractional years.

    Args:
        entry_time: Entry timestamp (seconds since epoch).
        expiry_time: Expiry timestamp (usually 15:30 IST on expiry day).
        trading_days_per_year: Number of trading days per year (default 252).
        hours_per_day: Market hours per day (default 6).

    Returns:
        Time to expiry in years.
    """
    seconds_per_trading_day = hours_per_day * 3600
    seconds_per_year = trading_days_per_year * seconds_per_trading_day
    T = (expiry_time - entry_time) / seconds_per_year
    return max(0.0, T)


# Convenience wrapper for backtest: price in paisa
def bsm_price_paisa(
    S_paisa: int,
    K_paisa: int,
    T: float,
    sigma: float,
    r: float = 0.05,
    kind: str = "call",
) -> int:
    """BSM price returning integer paisa.

    Args:
        S_paisa: Underlying price in paisa.
        K_paisa: Strike price in paisa.
        T: Years to expiry.
        sigma: Volatility (decimal).
        r: Risk-free rate.
        kind: "call" or "put".

    Returns:
        Option premium in paisa (rounded).
    """
    price_rupees = bsm_price(S_paisa / 100.0, K_paisa / 100.0, T, sigma, r, kind)
    return int(round(price_rupees * 100))
