"""Volume profile (TPO/POC/VAH/VAL) indicator.

Computes per-session volume profiles from intraday OHLCV bars.
POC = price bin with the highest volume.
Value Area (VA) = price range covering 70% of total session volume,
  expanded outward from POC using next-highest-volume bins.
VAH = upper bound of the value area.
VAL = lower bound of the value area.

All price values are stored/returned in paisa (integer, 100 paisa = 1 INR).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Default tick size in paisa (NSE equity: ₹0.05 = 5 paisa).
_DEFAULT_TICK_PAISA: int = 5
# Default bin multiplier: bin_size = tick_size * 10.
_BIN_MULTIPLIER: int = 10


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_volume_profile(
    bars: pd.DataFrame,
    bin_size_paisa: int | None = None,
    value_area_pct: float = 0.70,
) -> dict[str, Any] | None:
    """Compute volume profile from intraday OHLCV bars.

    Distributes bar volume evenly across its high-low price range, then
    aggregates into price bins. POC is the bin with the maximum volume.
    The value area expands outward from POC (by next-highest-volume bin)
    until cumulative volume >= ``value_area_pct`` of total volume.

    Args:
        bars: DataFrame with columns ``open``, ``high``, ``low``, ``close``,
            ``volume``. Prices may be in rupees (float) or paisa (int/float).
            If ``high.max() < 10_000`` the values are treated as rupees and
            converted to paisa automatically.
        bin_size_paisa: Width of each price bucket in paisa.
            Auto-computed as ``tick_size * 10`` when ``None``.
        value_area_pct: Fraction of total volume to capture in the value area.
            Default 0.70 (70%).

    Returns:
        Dictionary with keys:
            - ``poc_paisa`` (int)
            - ``vah_paisa`` (int)
            - ``val_paisa`` (int)
            - ``total_volume`` (int)
            - ``bin_size_paisa`` (int)
            - ``bins`` (list[dict]): [{``price``: int, ``volume``: int}, ...]
        Returns ``None`` when ``bars`` is empty or has no volume.
    """
    if bars is None or bars.empty:
        logger.debug("compute_volume_profile: empty bars — returning None")
        return None

    required_cols = {"high", "low", "volume"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing required columns: {missing}")

    # --- Convert to paisa if prices look like rupees (< ₹10,000 avg) -------
    avg_close = float(bars["high"].mean())
    if avg_close < 10_000:
        # Treat as rupees → convert to paisa
        multiplier = 100
    else:
        multiplier = 1  # already in paisa

    highs = (bars["high"] * multiplier).round().astype(int)
    lows = (bars["low"] * multiplier).round().astype(int)
    volumes = bars["volume"].astype(int)

    total_volume = int(volumes.sum())
    if total_volume == 0:
        logger.debug("compute_volume_profile: zero total volume — returning None")
        return None

    # --- Determine bin size -------------------------------------------------
    if bin_size_paisa is None:
        bin_size_paisa = _DEFAULT_TICK_PAISA * _BIN_MULTIPLIER  # 50 paisa default

    bin_size_paisa = max(1, int(bin_size_paisa))

    # --- Distribute volume into price bins ----------------------------------
    price_volume: dict[int, int] = {}

    for high_p, low_p, vol in zip(highs, lows, volumes):
        if high_p < low_p:
            high_p, low_p = low_p, high_p

        # Align to bin boundaries
        low_bin = (low_p // bin_size_paisa) * bin_size_paisa
        high_bin = (high_p // bin_size_paisa) * bin_size_paisa

        # Number of bins covered
        n_bins = max(1, (high_bin - low_bin) // bin_size_paisa + 1)
        vol_per_bin = vol / n_bins

        bin_price = low_bin
        while bin_price <= high_bin:
            price_volume[bin_price] = price_volume.get(bin_price, 0) + vol_per_bin
            bin_price += bin_size_paisa

    # --- Convert accumulated floats to int (rounding) -----------------------
    price_volume_int: dict[int, int] = {
        p: round(v) for p, v in price_volume.items()
    }

    # --- POC: bin with maximum volume ---------------------------------------
    poc_paisa = max(price_volume_int, key=lambda p: price_volume_int[p])

    # --- Value area expansion from POC -------------------------------------
    target_volume = value_area_pct * total_volume

    # Sort bins by volume descending (for greedy expansion).
    sorted_by_vol = sorted(
        price_volume_int.items(), key=lambda kv: kv[1], reverse=True
    )

    selected_prices: set[int] = {poc_paisa}
    cum_vol = price_volume_int[poc_paisa]

    for bin_price, bin_vol in sorted_by_vol:
        if cum_vol >= target_volume:
            break
        if bin_price not in selected_prices:
            selected_prices.add(bin_price)
            cum_vol += bin_vol

    vah_paisa = max(selected_prices)
    val_paisa = min(selected_prices)

    # Adjust boundaries to include the full bin width
    vah_paisa = vah_paisa + bin_size_paisa - 1  # upper edge of top bin
    # val_paisa remains at the lower edge of the bottom bin

    # --- Build bins list (sorted by price) ----------------------------------
    bins = [
        {"price": p, "volume": v}
        for p, v in sorted(price_volume_int.items())
    ]

    return {
        "poc_paisa": poc_paisa,
        "vah_paisa": vah_paisa,
        "val_paisa": val_paisa,
        "total_volume": total_volume,
        "bin_size_paisa": bin_size_paisa,
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# Builder class: session persistence + live queries
# ---------------------------------------------------------------------------

class VolumeProfileBuilder:
    """Builds and persists volume profiles for strategy use.

    Args:
        db_engine: SQLAlchemy engine for persistence. If ``None``, DB
            operations are skipped (in-memory mode only).
    """

    def __init__(self, db_engine=None) -> None:
        self._engine = db_engine

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def compute_session(
        self,
        instrument_key: str,
        bars: pd.DataFrame,
        trade_date: date | str,
    ) -> dict[str, Any] | None:
        """Compute volume profile for a trading session and persist to DB.

        Args:
            instrument_key: Upstox instrument identifier.
            bars: Intraday OHLCV bars for the session.
            trade_date: Session date (``date`` object or ISO string).

        Returns:
            Profile dict (same as :func:`compute_volume_profile`) or ``None``
            when bars are empty or have zero volume.
        """
        profile = compute_volume_profile(bars)
        if profile is None:
            return None

        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)

        if self._engine is not None:
            self._persist(instrument_key, trade_date, profile)

        return profile

    def _persist(
        self,
        instrument_key: str,
        trade_date: date,
        profile: dict[str, Any],
    ) -> None:
        """Insert or replace profile row in ``volume_profile_daily``.

        Args:
            instrument_key: Instrument identifier.
            trade_date: Trading date.
            profile: Profile dict from :func:`compute_volume_profile`.
        """
        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO volume_profile_daily
                (instrument_key, trade_date, poc_paisa, vah_paisa, val_paisa,
                 total_volume, bin_size_paisa, profile_bins)
            VALUES
                (:instrument_key, :trade_date, :poc_paisa, :vah_paisa, :val_paisa,
                 :total_volume, :bin_size_paisa, CAST(:profile_bins AS jsonb))
            ON CONFLICT (instrument_key, trade_date)
            DO UPDATE SET
                poc_paisa      = EXCLUDED.poc_paisa,
                vah_paisa      = EXCLUDED.vah_paisa,
                val_paisa      = EXCLUDED.val_paisa,
                total_volume   = EXCLUDED.total_volume,
                bin_size_paisa = EXCLUDED.bin_size_paisa,
                profile_bins   = EXCLUDED.profile_bins
            """
        )
        with self._engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "instrument_key": instrument_key,
                    "trade_date": trade_date.isoformat(),
                    "poc_paisa": profile["poc_paisa"],
                    "vah_paisa": profile["vah_paisa"],
                    "val_paisa": profile["val_paisa"],
                    "total_volume": profile["total_volume"],
                    "bin_size_paisa": profile["bin_size_paisa"],
                    "profile_bins": json.dumps(profile["bins"]),
                },
            )
        logger.debug(
            "Persisted volume profile: %s %s POC=%d VAH=%d VAL=%d",
            instrument_key,
            trade_date,
            profile["poc_paisa"],
            profile["vah_paisa"],
            profile["val_paisa"],
        )

    # ------------------------------------------------------------------
    # Live intraday (in-memory, no persist)
    # ------------------------------------------------------------------

    def compute_today_intraday(
        self,
        instrument_key: str,
        bars: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Compute profile for today's partial session (no DB write).

        Used by strategies during the live session to get the current
        intraday POC/VAH/VAL without persisting incomplete data.

        Args:
            instrument_key: Upstox instrument identifier (used for logging).
            bars: Partial intraday OHLCV bars accumulated so far today.

        Returns:
            Profile dict or ``None``.
        """
        profile = compute_volume_profile(bars)
        if profile is None:
            logger.debug("compute_today_intraday: no profile for %s", instrument_key)
        return profile

    # ------------------------------------------------------------------
    # DB read
    # ------------------------------------------------------------------

    def get_profile(
        self,
        instrument_key: str,
        trade_date: date | str,
    ) -> dict[str, Any] | None:
        """Fetch a persisted profile from the database.

        Args:
            instrument_key: Instrument identifier.
            trade_date: Trading date.

        Returns:
            Profile dict with keys ``poc_paisa``, ``vah_paisa``, ``val_paisa``,
            ``total_volume``, ``bin_size_paisa``, ``bins``, or ``None`` if not found.
        """
        if self._engine is None:
            logger.warning("get_profile: no db_engine configured")
            return None

        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)

        from sqlalchemy import text

        sql = text(
            """
            SELECT poc_paisa, vah_paisa, val_paisa, total_volume,
                   bin_size_paisa, profile_bins
            FROM volume_profile_daily
            WHERE instrument_key = :instrument_key
              AND trade_date = :trade_date
            LIMIT 1
            """
        )
        with self._engine.connect() as conn:
            row = conn.execute(
                sql,
                {
                    "instrument_key": instrument_key,
                    "trade_date": trade_date.isoformat(),
                },
            ).fetchone()

        if row is None:
            return None

        poc, vah, val, total_vol, bin_sz, bins_raw = row
        bins = bins_raw if isinstance(bins_raw, list) else json.loads(bins_raw or "[]")

        return {
            "poc_paisa": int(poc),
            "vah_paisa": int(vah),
            "val_paisa": int(val),
            "total_volume": int(total_vol),
            "bin_size_paisa": int(bin_sz),
            "bins": bins,
        }
