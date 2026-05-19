"""Instrument master download, caching, and lookup utilities.

Downloads gzipped JSON instrument files from the Upstox CDN, caches them
locally, and provides fast search / option-chain helpers.
"""

from __future__ import annotations

import gzip
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")

# Upstox CDN base URL pattern for instrument dumps.
_CDN_BASE = "https://assets.upstox.com/market-quote/instruments/exchange"

# Exchanges supported by the Upstox CDN.
_SUPPORTED_EXCHANGES: list[str] = ["NSE", "BSE", "MCX"]

# Columns we keep from the raw JSON (in order).
_INSTRUMENT_COLUMNS: list[str] = [
    "segment",
    "name",
    "instrument_key",
    "trading_symbol",
    "lot_size",
    "tick_size",
    "expiry",
    "strike_price",
    "option_type",
    "underlying_key",
    "underlying_symbol",
    "exchange_token",
    "isin",
    "instrument_type",
    "short_name",
    "freeze_quantity",
]


class InstrumentManager:
    """Manages downloading, caching, and querying the Upstox instrument master.

    Args:
        cache_dir: Directory where cached instrument JSON files are stored.
    """

    def __init__(self, cache_dir: str = "data/instruments") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._instruments: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Download & cache
    # ------------------------------------------------------------------

    def download_instruments(
        self,
        exchanges: list[str] | None = None,
    ) -> None:
        """Fetch gzipped JSON instrument files from Upstox CDN and cache locally.

        Downloads ``{exchange}.json.gz`` for each requested exchange, decompresses,
        and saves the raw JSON to ``cache_dir/{exchange}.json``.

        Args:
            exchanges: List of exchange codes (e.g. ``["NSE", "BSE"]``).
                Defaults to all supported exchanges.
        """
        targets = exchanges or _SUPPORTED_EXCHANGES

        for exchange in targets:
            exchange_upper = exchange.upper()
            url = f"{_CDN_BASE}/{exchange_upper}.json.gz"
            dest = self._cache_dir / f"{exchange_upper}.json"

            logger.info("Downloading instrument file for {} from {}", exchange_upper, url)
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.get(url)
                    resp.raise_for_status()

                # Decompress gzip payload.
                raw_json = gzip.decompress(resp.content)
                dest.write_bytes(raw_json)

                logger.info(
                    "Saved {} instrument file to {} ({:.1f} KB)",
                    exchange_upper,
                    dest,
                    len(raw_json) / 1024,
                )
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "HTTP {} when downloading {} instruments: {}",
                    exc.response.status_code,
                    exchange_upper,
                    exc,
                )
            except Exception as exc:
                logger.error(
                    "Failed to download {} instruments: {}",
                    exchange_upper,
                    exc,
                )

        # Invalidate in-memory cache so next access re-loads from disk.
        self._instruments = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_instruments(
        self,
        exchanges: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load cached JSON instrument files into a single DataFrame.

        If instruments have already been loaded they are returned from the
        in-memory cache.

        Args:
            exchanges: List of exchange codes to load.
                Defaults to all supported exchanges.

        Returns:
            DataFrame with standardised instrument columns.
        """
        if self._instruments is not None:
            return self._instruments

        targets = exchanges or _SUPPORTED_EXCHANGES
        frames: list[pd.DataFrame] = []

        for exchange in targets:
            path = self._cache_dir / f"{exchange.upper()}.json"
            if not path.exists():
                logger.warning("Instrument cache file not found: {}", path)
                continue

            logger.debug("Loading instruments from {}", path)
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            df = pd.DataFrame(data)
            frames.append(df)

        if not frames:
            logger.warning("No instrument data loaded — returning empty DataFrame")
            self._instruments = pd.DataFrame(columns=_INSTRUMENT_COLUMNS)
            return self._instruments

        combined = pd.concat(frames, ignore_index=True)

        # Keep only the columns we care about (missing columns become NaN).
        for col in _INSTRUMENT_COLUMNS:
            if col not in combined.columns:
                combined[col] = None
        combined = combined[_INSTRUMENT_COLUMNS]

        # Derive option_type from instrument_type (Upstox uses instrument_type for CE/PE/FUT/EQ).
        combined["option_type"] = combined["instrument_type"].where(
            combined["instrument_type"].isin(["CE", "PE"]),
            other=None,
        )

        # Convert numeric columns.
        combined["lot_size"] = pd.to_numeric(combined["lot_size"], errors="coerce").fillna(0).astype(int)
        combined["tick_size"] = pd.to_numeric(combined["tick_size"], errors="coerce").fillna(0)
        combined["strike_price"] = pd.to_numeric(combined["strike_price"], errors="coerce").fillna(0)
        combined["freeze_quantity"] = pd.to_numeric(combined["freeze_quantity"], errors="coerce").fillna(0).astype(int)

        # Normalise expiry to date strings (YYYY-MM-DD).
        # Upstox provides expiry as epoch milliseconds (large int); non-numeric / null rows → NaT.
        expiry_numeric = pd.to_numeric(combined["expiry"], errors="coerce")
        combined["expiry"] = (
            pd.to_datetime(expiry_numeric, unit="ms", errors="coerce")
            .dt.strftime("%Y-%m-%d")
        )

        self._instruments = combined
        logger.info("Loaded {} instruments across {} exchange(s)", len(combined), len(frames))
        return self._instruments

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def search(
        self,
        segment: str | None = None,
        symbol: str | None = None,
        instrument_type: str | None = None,
    ) -> pd.DataFrame:
        """Filter loaded instruments by segment, symbol, and/or instrument type.

        Args:
            segment: Market segment filter (e.g. ``"NSE_EQ"``).
            symbol: Substring match on ``trading_symbol`` (case-insensitive).
            instrument_type: Exact match on ``instrument_type`` (e.g. ``"EQUITY"``).

        Returns:
            Filtered DataFrame.
        """
        df = self.load_instruments()
        mask = pd.Series(True, index=df.index)

        if segment is not None:
            mask &= df["segment"].str.upper() == segment.upper()
        if symbol is not None:
            mask &= df["trading_symbol"].str.contains(symbol, case=False, na=False)
        if instrument_type is not None:
            mask &= df["instrument_type"].str.upper() == instrument_type.upper()

        return df.loc[mask].reset_index(drop=True)

    def get_option_chain(
        self,
        underlying_key: str,
        expiry_date: str,
    ) -> pd.DataFrame:
        """Retrieve the option chain for a given underlying and expiry.

        Args:
            underlying_key: Instrument key of the underlying
                (e.g. ``"NSE_INDEX|Nifty 50"``).
            expiry_date: Expiry date as ``"YYYY-MM-DD"`` string.

        Returns:
            DataFrame of CE and PE options matching the underlying and expiry,
            sorted by strike price.
        """
        df = self.load_instruments()
        mask = (
            (df["underlying_key"] == underlying_key)
            & (df["expiry"] == expiry_date)
            & (df["option_type"].isin(["CE", "PE"]))
        )
        chain = df.loc[mask].copy()
        chain.sort_values("strike_price", inplace=True)
        return chain.reset_index(drop=True)

    def get_atm_strike(
        self,
        spot_price: int,
        option_chain: pd.DataFrame,
    ) -> int:
        """Find the at-the-money strike nearest to a given spot price.

        Args:
            spot_price: Current spot price in **paisa**.
            option_chain: Option chain DataFrame (as returned by
                :meth:`get_option_chain`). ``strike_price`` column is
                expected in rupees (as provided by Upstox); the method
                converts internally.

        Returns:
            Strike price in **paisa** that is closest to ``spot_price``.
        """
        if option_chain.empty:
            raise ValueError("Cannot determine ATM strike from an empty option chain")

        # Upstox provides strike_price in rupees; convert to paisa for comparison.
        strikes_paisa = (option_chain["strike_price"] * 100).astype(int)
        unique_strikes = strikes_paisa.unique()
        idx = (abs(unique_strikes - spot_price)).argmin()
        return int(unique_strikes[idx])

    def get_weekly_expiry(
        self,
        underlying_symbol: str,
    ) -> str | None:
        """Find the nearest weekly expiry for an underlying symbol.

        Args:
            underlying_symbol: Underlying symbol string
                (e.g. ``"NIFTY"``, ``"BANKNIFTY"``).

        Returns:
            Expiry date as ``"YYYY-MM-DD"`` or ``None`` if nothing found.
        """
        df = self.load_instruments()
        mask = (
            (df["underlying_symbol"].str.upper() == underlying_symbol.upper())
            & (df["option_type"].isin(["CE", "PE"]))
            & (df["expiry"].notna())
        )
        options = df.loc[mask]
        if options.empty:
            return None

        today_str = date.today().strftime("%Y-%m-%d")
        future_expiries = options.loc[options["expiry"] >= today_str, "expiry"].unique()
        if len(future_expiries) == 0:
            return None

        # Return the nearest expiry.
        return str(sorted(future_expiries)[0])

    def get_instrument_key(
        self,
        segment: str,
        trading_symbol: str,
    ) -> str | None:
        """Look up the Upstox instrument key by segment and trading symbol.

        Args:
            segment: Market segment (e.g. ``"NSE_EQ"``).
            trading_symbol: Exact trading symbol (e.g. ``"RELIANCE"``).

        Returns:
            Instrument key string or ``None`` if not found.
        """
        df = self.load_instruments()
        mask = (
            (df["segment"].str.upper() == segment.upper())
            & (df["trading_symbol"] == trading_symbol)
        )
        matches = df.loc[mask, "instrument_key"]
        if matches.empty:
            return None
        return str(matches.iloc[0])

    def resolve(self, instrument_key: str) -> str:
        """Resolve a trading symbol or instrument key to a full instrument key.

        If the input already contains ``"|"`` it is assumed to be a fully
        qualified instrument key and is returned as-is.  Otherwise, the
        loaded instruments are searched by ``trading_symbol`` and the first
        matching ``instrument_key`` is returned.  If no match is found the
        input is returned unchanged as a fallback.

        Args:
            instrument_key: An instrument key (e.g. ``"NSE_EQ|INE002A01018"``)
                or a bare trading symbol (e.g. ``"RELIANCE"``).

        Returns:
            The resolved instrument key string.
        """
        if "|" in instrument_key:
            return instrument_key

        df = self.load_instruments()
        mask = df["trading_symbol"] == instrument_key
        matches = df.loc[mask, "instrument_key"]
        if not matches.empty:
            return str(matches.iloc[0])

        return instrument_key
