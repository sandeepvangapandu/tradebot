"""Options chain REST poller + OI/IV/PCR computation.

Polls Upstox ``/v2/option/chain`` every N seconds for a set of
active underlyings, computes derived metrics (OI change, IV
percentile, PCR, max-pain, ATM IV, IV skew) and persists snapshots
to Postgres.

All monetary values are stored in **paisa** (int).  Times are in IST.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from loguru import logger
from sqlalchemy import text

IST = ZoneInfo("Asia/Kolkata")

_UPSTOX_API_BASE = "https://api.upstox.com"


class OptionsChainFeed:
    """Polls the Upstox option chain REST endpoint and persists snapshots.

    Args:
        access_token: Valid Upstox OAuth2 access token.
        db_engine: SQLAlchemy sync engine (``get_sync_engine()``).
                   If ``None`` persistence calls are skipped (useful for tests).
        poll_interval_seconds: How often (in seconds) to refresh each chain.
    """

    def __init__(
        self,
        access_token: str,
        db_engine=None,
        poll_interval_seconds: int = 30,
    ) -> None:
        self._token = access_token
        self._engine = db_engine
        self._poll_interval = poll_interval_seconds
        # Cache of previous OI per (underlying_key, expiry, strike, option_type)
        # used to compute oi_change between polls.
        self._prev_oi: dict[tuple[str, str, float, str], int] = {}

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Public: fetch chain
    # ------------------------------------------------------------------

    def fetch_chain(self, underlying_key: str, expiry: str) -> pd.DataFrame:
        """Fetch the full option chain from Upstox REST API.

        Makes a GET request to ``/v2/option/chain?instrument_key=...&expiry_date=...``.

        Args:
            underlying_key: Upstox instrument key for the underlying
                (e.g. ``"NSE_INDEX|Nifty 50"``).
            expiry: Expiry date string in ``YYYY-MM-DD`` format.

        Returns:
            DataFrame with columns:
            ``strike, option_type, ltp, oi, volume, iv, delta, gamma, vega, theta``.
            Empty DataFrame if the request fails or returns no data.
        """
        url = f"{_UPSTOX_API_BASE}/v2/option/chain"
        params = {"instrument_key": underlying_key, "expiry_date": expiry}

        try:
            response = httpx.get(url, headers=self._headers(), params=params, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("[OptionsChainFeed] fetch_chain failed for {} {}: {}", underlying_key, expiry, exc)
            return pd.DataFrame()

        data = payload.get("data", [])
        if not data:
            logger.debug("[OptionsChainFeed] empty chain returned for {} {}", underlying_key, expiry)
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for entry in data:
            # Each entry has "call_options" and "put_options" sub-dicts
            for opt_type, key in (("CE", "call_options"), ("PE", "put_options")):
                opt = entry.get(key)
                if opt is None:
                    continue
                md = opt.get("market_data") or {}
                greek = opt.get("option_greeks") or {}
                rows.append(
                    {
                        "strike": float(entry.get("strike_price", 0)),
                        "option_type": opt_type,
                        "ltp": float(md.get("ltp", 0) or 0),
                        "oi": int(md.get("oi", 0) or 0),
                        "volume": int(md.get("volume", 0) or 0),
                        "iv": float(greek.get("iv", 0) or 0),
                        "delta": float(greek.get("delta", 0) or 0),
                        "gamma": float(greek.get("gamma", 0) or 0),
                        "vega": float(greek.get("vega", 0) or 0),
                        "theta": float(greek.get("theta", 0) or 0),
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Upstox nests IV inside option_greeks.iv (not always vega); fix if present
        # Re-parse for IV specifically
        iv_rows: dict[tuple[float, str], float] = {}
        for entry in data:
            strike = float(entry.get("strike_price", 0))
            for opt_type, key in (("CE", "call_options"), ("PE", "put_options")):
                opt = entry.get(key)
                if opt is None:
                    continue
                greek = opt.get("option_greeks") or {}
                iv_val = greek.get("iv", None)
                if iv_val is not None:
                    iv_rows[(strike, opt_type)] = float(iv_val)

        if iv_rows:
            df["iv"] = df.apply(
                lambda r: iv_rows.get((r["strike"], r["option_type"]), r["iv"]),
                axis=1,
            )

        return df

    # ------------------------------------------------------------------
    # Public: derived metrics
    # ------------------------------------------------------------------

    def compute_pcr(self, chain: pd.DataFrame) -> float:
        """Compute the Put-Call OI Ratio.

        Args:
            chain: DataFrame returned by :meth:`fetch_chain`.

        Returns:
            ``total_put_oi / total_call_oi``.
            Returns ``float('inf')`` when total call OI is 0.
        """
        if chain.empty or "option_type" not in chain.columns:
            return float("nan")

        total_call_oi = chain.loc[chain["option_type"] == "CE", "oi"].sum()
        total_put_oi = chain.loc[chain["option_type"] == "PE", "oi"].sum()

        if total_call_oi == 0:
            return float("inf")

        return float(total_put_oi / total_call_oi)

    def compute_max_pain(self, chain: pd.DataFrame) -> float:
        """Compute the max-pain strike.

        The max-pain strike is the one at which total intrinsic value
        (i.e. the loss) across all outstanding options is maximised.
        This is equivalent to the strike where option writers lose the
        least / holders lose the most.

        For each candidate expiry price (every strike in the chain):
          - total call intrinsic  = sum(max(0, candidate - call_strike) * call_oi)
          - total put  intrinsic  = sum(max(0, put_strike - candidate) * put_oi)

        The strike with the minimum total intrinsic is the max-pain point.

        Args:
            chain: DataFrame returned by :meth:`fetch_chain`.

        Returns:
            Strike price (float).  Returns NaN if chain is empty.
        """
        if chain.empty or "strike" not in chain.columns:
            return float("nan")

        calls = chain[chain["option_type"] == "CE"][["strike", "oi"]].copy()
        puts = chain[chain["option_type"] == "PE"][["strike", "oi"]].copy()

        strikes = sorted(chain["strike"].unique())
        min_pain: float = float("inf")
        max_pain_strike: float = float("nan")

        for candidate in strikes:
            call_pain = float(
                calls.apply(lambda r: max(0.0, candidate - r["strike"]) * r["oi"], axis=1).sum()
            )
            put_pain = float(
                puts.apply(lambda r: max(0.0, r["strike"] - candidate) * r["oi"], axis=1).sum()
            )
            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = candidate

        return max_pain_strike

    def compute_atm_iv(self, chain: pd.DataFrame, spot: float) -> tuple[float, float]:
        """Compute ATM call IV and ATM put IV.

        Finds the nearest strike to *spot* and returns the IV for its
        CE and PE legs.

        Args:
            chain: DataFrame returned by :meth:`fetch_chain`.
            spot: Underlying spot price (in the same unit as chain strikes).

        Returns:
            Tuple ``(atm_call_iv, atm_put_iv)``.
            Returns ``(nan, nan)`` if chain is empty.
        """
        if chain.empty or "strike" not in chain.columns:
            return float("nan"), float("nan")

        strikes = chain["strike"].unique()
        atm_strike = float(min(strikes, key=lambda s: abs(s - spot)))

        atm_call = chain[(chain["strike"] == atm_strike) & (chain["option_type"] == "CE")]
        atm_put = chain[(chain["strike"] == atm_strike) & (chain["option_type"] == "PE")]

        atm_call_iv = float(atm_call["iv"].iloc[0]) if not atm_call.empty else float("nan")
        atm_put_iv = float(atm_put["iv"].iloc[0]) if not atm_put.empty else float("nan")

        return atm_call_iv, atm_put_iv

    def compute_iv_skew(self, chain: pd.DataFrame, spot: float) -> float:
        """Compute the IV skew: OTM put IV minus OTM call IV.

        Uses strikes approximately 5% OTM on each side.

        Args:
            chain: DataFrame returned by :meth:`fetch_chain`.
            spot: Underlying spot price (same unit as strikes).

        Returns:
            ``otm_put_iv - otm_call_iv``.  Positive values indicate a
            fear/downside skew (typical in equity markets).
            Returns ``nan`` if either leg cannot be found.
        """
        if chain.empty or "strike" not in chain.columns:
            return float("nan")

        target_otm_call_strike = spot * 1.05
        target_otm_put_strike = spot * 0.95

        call_chain = chain[chain["option_type"] == "CE"]
        put_chain = chain[chain["option_type"] == "PE"]

        if call_chain.empty or put_chain.empty:
            return float("nan")

        otm_call_strike = float(
            min(call_chain["strike"].unique(), key=lambda s: abs(s - target_otm_call_strike))
        )
        otm_put_strike = float(
            min(put_chain["strike"].unique(), key=lambda s: abs(s - target_otm_put_strike))
        )

        otm_call_row = call_chain[call_chain["strike"] == otm_call_strike]
        otm_put_row = put_chain[put_chain["strike"] == otm_put_strike]

        if otm_call_row.empty or otm_put_row.empty:
            return float("nan")

        otm_call_iv = float(otm_call_row["iv"].iloc[0])
        otm_put_iv = float(otm_put_row["iv"].iloc[0])

        return otm_put_iv - otm_call_iv

    # ------------------------------------------------------------------
    # Public: snapshot (fetch + compute + persist)
    # ------------------------------------------------------------------

    def snapshot(
        self,
        underlying_key: str,
        underlying_symbol: str,
        expiry: str,
        spot_paisa: int,
    ) -> dict[str, Any] | None:
        """Fetch chain, compute metrics, persist to DB, and return summary dict.

        Writes one row to ``options_chain_snapshots`` and one row per
        (strike, option_type) to ``options_strike_oi_history``.

        Args:
            underlying_key: Upstox instrument key for the underlying.
            underlying_symbol: Human-readable symbol (e.g. ``"NIFTY"``).
            expiry: Expiry date string ``YYYY-MM-DD``.
            spot_paisa: Current spot price in paisa.

        Returns:
            Dict with computed metrics, or ``None`` if chain is empty.
        """
        chain = self.fetch_chain(underlying_key, expiry)
        if chain.empty:
            logger.warning(
                "[OptionsChainFeed] snapshot: empty chain for {} {}", underlying_key, expiry
            )
            return None

        spot_price = spot_paisa / 100.0  # convert to rupees for strike comparison

        pcr = self.compute_pcr(chain)
        max_pain = self.compute_max_pain(chain)
        atm_call_iv, atm_put_iv = self.compute_atm_iv(chain, spot_price)
        iv_skew = self.compute_iv_skew(chain, spot_price)

        total_call_oi = int(chain.loc[chain["option_type"] == "CE", "oi"].sum())
        total_put_oi = int(chain.loc[chain["option_type"] == "PE", "oi"].sum())

        ts = datetime.now(IST)
        raw_chain_json = chain.to_dict(orient="records")

        result: dict[str, Any] = {
            "underlying_key": underlying_key,
            "underlying_symbol": underlying_symbol,
            "expiry": expiry,
            "ts": ts,
            "spot_paisa": spot_paisa,
            "pcr": None if (isinstance(pcr, float) and (pcr != pcr)) else pcr,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "max_pain_strike": None if (isinstance(max_pain, float) and (max_pain != max_pain)) else max_pain,
            "atm_iv_call": None if (isinstance(atm_call_iv, float) and (atm_call_iv != atm_call_iv)) else atm_call_iv,
            "atm_iv_put": None if (isinstance(atm_put_iv, float) and (atm_put_iv != atm_put_iv)) else atm_put_iv,
            "iv_skew": None if (isinstance(iv_skew, float) and (iv_skew != iv_skew)) else iv_skew,
        }

        if self._engine is not None:
            self._persist_snapshot(result, raw_chain_json, chain, underlying_key, expiry, ts)

        return result

    # ------------------------------------------------------------------
    # Private: persistence
    # ------------------------------------------------------------------

    def _persist_snapshot(
        self,
        result: dict[str, Any],
        raw_chain_json: list[dict],
        chain: pd.DataFrame,
        underlying_key: str,
        expiry: str,
        ts: datetime,
    ) -> None:
        """Write snapshot and per-strike rows to Postgres.

        Args:
            result: Computed metrics dict from :meth:`snapshot`.
            raw_chain_json: Raw chain records for JSONB storage.
            chain: The full chain DataFrame for per-strike rows.
            underlying_key: Upstox underlying key.
            expiry: Expiry date string.
            ts: Timestamp of this snapshot (IST).
        """
        snap_sql = text(
            """
            INSERT INTO options_chain_snapshots
              (underlying_key, underlying_symbol, expiry, ts, spot_paisa,
               pcr, total_call_oi, total_put_oi, max_pain_strike,
               atm_iv_call, atm_iv_put, iv_skew, raw_chain)
            VALUES
              (:underlying_key, :underlying_symbol, :expiry, :ts, :spot_paisa,
               :pcr, :total_call_oi, :total_put_oi, :max_pain_strike,
               :atm_iv_call, :atm_iv_put, :iv_skew, CAST(:raw_chain AS jsonb))
            RETURNING id
            """
        )

        strike_sql = text(
            """
            INSERT INTO options_strike_oi_history
              (underlying_key, expiry, strike, option_type, ts,
               oi, oi_change, iv, delta, gamma, vega, theta)
            VALUES
              (:underlying_key, :expiry, :strike, :option_type, :ts,
               :oi, :oi_change, :iv, :delta, :gamma, :vega, :theta)
            ON CONFLICT (underlying_key, expiry, strike, option_type, ts) DO NOTHING
            """
        )

        with self._engine.begin() as conn:
            conn.execute(
                snap_sql,
                {
                    **result,
                    "underlying_symbol": result["underlying_symbol"],
                    "expiry": result["expiry"],
                    "ts": ts,
                    "raw_chain": json.dumps(raw_chain_json),
                },
            )

            strike_rows: list[dict[str, Any]] = []
            for _, row in chain.iterrows():
                cache_key = (underlying_key, expiry, float(row["strike"]), str(row["option_type"]))
                prev_oi = self._prev_oi.get(cache_key)
                current_oi = int(row["oi"])
                oi_change = (current_oi - prev_oi) if prev_oi is not None else None
                self._prev_oi[cache_key] = current_oi

                strike_rows.append(
                    {
                        "underlying_key": underlying_key,
                        "expiry": expiry,
                        "strike": float(row["strike"]),
                        "option_type": str(row["option_type"]),
                        "ts": ts,
                        "oi": current_oi,
                        "oi_change": oi_change,
                        "iv": float(row["iv"]) if row.get("iv") is not None else None,
                        "delta": float(row["delta"]) if row.get("delta") is not None else None,
                        "gamma": float(row["gamma"]) if row.get("gamma") is not None else None,
                        "vega": float(row["vega"]) if row.get("vega") is not None else None,
                        "theta": float(row["theta"]) if row.get("theta") is not None else None,
                    }
                )

            if strike_rows:
                conn.execute(strike_sql, strike_rows)

        logger.info(
            "[OptionsChainFeed] persisted snapshot for {} {} — {} strikes",
            underlying_key,
            expiry,
            len(chain),
        )

    # ------------------------------------------------------------------
    # Public: async poll loop
    # ------------------------------------------------------------------

    async def poll_loop(self, subscriptions: list[dict[str, str]]) -> None:
        """Continuously poll every underlying and persist snapshots.

        Args:
            subscriptions: List of dicts, each containing:
                ``underlying_key`` (str), ``underlying_symbol`` (str),
                ``expiry`` (str ``YYYY-MM-DD``), ``spot_paisa`` (int).
                If ``spot_paisa`` is absent it defaults to 0.

        The loop runs until the task is cancelled.
        """
        logger.info(
            "[OptionsChainFeed] Starting poll loop — {} subscriptions, interval={}s",
            len(subscriptions),
            self._poll_interval,
        )

        while True:
            for sub in subscriptions:
                underlying_key: str = sub["underlying_key"]
                underlying_symbol: str = sub["underlying_symbol"]
                expiry: str = sub["expiry"]
                spot_paisa: int = int(sub.get("spot_paisa", 0))

                try:
                    await asyncio.to_thread(
                        self.snapshot, underlying_key, underlying_symbol, expiry, spot_paisa
                    )
                except Exception as exc:
                    logger.error(
                        "[OptionsChainFeed] poll_loop error for {} {}: {}",
                        underlying_key,
                        expiry,
                        exc,
                    )

            await asyncio.sleep(self._poll_interval)
