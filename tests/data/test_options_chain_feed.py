"""Tests for src/data/options_chain_feed.py — Options chain REST poller.

All tests use inline mock chain DataFrames and httpx HTTP call mocks so they
are self-contained and do not depend on external services or a real database.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.data.options_chain_feed import OptionsChainFeed

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Helpers — build synthetic chain DataFrames
# ---------------------------------------------------------------------------

KEY = "NSE_INDEX|Nifty 50"
SYMBOL = "NIFTY"
EXPIRY = "2026-05-15"


def _make_chain(
    strikes: list[float] | None = None,
    call_oi_multiplier: float = 1.0,
    put_oi_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Return a minimal synthetic options chain DataFrame."""
    if strikes is None:
        strikes = [24800.0, 24850.0, 24900.0, 24950.0, 25000.0,
                   25050.0, 25100.0, 25150.0, 25200.0]

    rows = []
    for strike in strikes:
        call_iv = 0.18 + (strike - 25000.0) / 100000.0  # slight smile
        put_iv = 0.20 + (25000.0 - strike) / 100000.0   # put skew
        rows.append({
            "strike": strike,
            "option_type": "CE",
            "ltp": max(0.05, 25100.0 - strike),
            "oi": int(1000 * call_oi_multiplier + (25200.0 - strike) * 2),
            "volume": 500,
            "iv": max(0.01, call_iv),
            "delta": 0.5 - (strike - 25000.0) / 1000.0,
            "gamma": 0.001,
            "vega": 0.5,
            "theta": -0.3,
        })
        rows.append({
            "strike": strike,
            "option_type": "PE",
            "ltp": max(0.05, strike - 24900.0),
            "oi": int(1000 * put_oi_multiplier + (strike - 24800.0) * 2),
            "volume": 600,
            "iv": max(0.01, put_iv),
            "delta": -0.5 + (strike - 25000.0) / 1000.0,
            "gamma": 0.001,
            "vega": 0.5,
            "theta": -0.3,
        })

    return pd.DataFrame(rows)


def _make_upstox_api_response(chain: pd.DataFrame) -> dict:
    """Reconstruct an Upstox API JSON payload from a synthetic chain DataFrame."""
    # Group CE/PE by strike
    grouped: dict[float, dict] = {}
    for _, row in chain.iterrows():
        strike = row["strike"]
        if strike not in grouped:
            grouped[strike] = {}
        key = "call_options" if row["option_type"] == "CE" else "put_options"
        grouped[strike][key] = {
            "market_data": {
                "ltp": row["ltp"],
                "oi": row["oi"],
                "volume": row["volume"],
            },
            "option_greeks": {
                "iv": row["iv"],
                "delta": row["delta"],
                "gamma": row["gamma"],
                "vega": row["vega"],
                "theta": row["theta"],
            },
        }

    data = []
    for strike, opts in sorted(grouped.items()):
        entry = {"strike_price": strike}
        entry.update(opts)
        data.append(entry)

    return {"status": "success", "data": data}


# ---------------------------------------------------------------------------
# compute_pcr
# ---------------------------------------------------------------------------

class TestComputePcr:
    """Tests for OptionsChainFeed.compute_pcr."""

    def test_compute_pcr_normal(self):
        """PCR = total_put_oi / total_call_oi for a balanced chain."""
        feed = OptionsChainFeed(access_token="tok")
        chain = _make_chain(strikes=[24900.0, 25000.0, 25100.0])

        pcr = feed.compute_pcr(chain)

        total_call_oi = chain.loc[chain["option_type"] == "CE", "oi"].sum()
        total_put_oi = chain.loc[chain["option_type"] == "PE", "oi"].sum()
        expected = total_put_oi / total_call_oi

        assert math.isfinite(pcr)
        assert abs(pcr - expected) < 1e-9

    def test_compute_pcr_zero_calls_returns_inf(self):
        """Safe division: PCR returns inf when all call OI is zero."""
        feed = OptionsChainFeed(access_token="tok")
        chain = _make_chain(strikes=[24900.0, 25000.0], call_oi_multiplier=0)
        # Force all CE oi to 0
        chain.loc[chain["option_type"] == "CE", "oi"] = 0

        pcr = feed.compute_pcr(chain)

        assert pcr == float("inf")

    def test_compute_pcr_empty_chain_returns_nan(self):
        """PCR returns NaN for an empty DataFrame."""
        feed = OptionsChainFeed(access_token="tok")
        assert math.isnan(feed.compute_pcr(pd.DataFrame()))


# ---------------------------------------------------------------------------
# compute_max_pain
# ---------------------------------------------------------------------------

class TestComputeMaxPain:
    """Tests for OptionsChainFeed.compute_max_pain."""

    def test_compute_max_pain_returns_strike_with_max_intrinsic(self):
        """Max-pain returns the strike minimising total holder intrinsic loss.

        We construct a deliberately lopsided chain where one strike is clearly
        the max-pain point and assert the method returns it.
        """
        feed = OptionsChainFeed(access_token="tok")

        # Simple chain: 3 strikes, large OI concentration at 25000.
        # If expiry is at 25000, call holders with K<25000 collect intrinsic,
        # put holders with K>25000 collect intrinsic.  With most OI at 25000
        # the writer loss is minimised there.
        data = [
            {"strike": 24900.0, "option_type": "CE", "oi": 10000},
            {"strike": 24900.0, "option_type": "PE", "oi": 100},
            {"strike": 25000.0, "option_type": "CE", "oi": 50000},
            {"strike": 25000.0, "option_type": "PE", "oi": 50000},
            {"strike": 25100.0, "option_type": "CE", "oi": 100},
            {"strike": 25100.0, "option_type": "PE", "oi": 10000},
        ]
        chain = pd.DataFrame(data)

        max_pain = feed.compute_max_pain(chain)

        # The balanced OI at 25000 should be the max-pain strike.
        assert max_pain == 25000.0

    def test_compute_max_pain_empty_chain_returns_nan(self):
        """Max-pain returns NaN for an empty DataFrame."""
        feed = OptionsChainFeed(access_token="tok")
        assert math.isnan(feed.compute_max_pain(pd.DataFrame()))


# ---------------------------------------------------------------------------
# compute_atm_iv
# ---------------------------------------------------------------------------

class TestComputeAtmIv:
    """Tests for OptionsChainFeed.compute_atm_iv."""

    def test_compute_atm_iv_picks_nearest_strike(self):
        """ATM IV is read from the strike closest to spot."""
        feed = OptionsChainFeed(access_token="tok")
        chain = _make_chain(strikes=[24900.0, 24950.0, 25000.0, 25050.0, 25100.0])

        # spot = 24960, nearest strike is 24950
        atm_call_iv, atm_put_iv = feed.compute_atm_iv(chain, spot=24960.0)

        atm_strike = 24950.0
        expected_call_iv = chain.loc[
            (chain["strike"] == atm_strike) & (chain["option_type"] == "CE"), "iv"
        ].iloc[0]
        expected_put_iv = chain.loc[
            (chain["strike"] == atm_strike) & (chain["option_type"] == "PE"), "iv"
        ].iloc[0]

        assert abs(atm_call_iv - expected_call_iv) < 1e-9
        assert abs(atm_put_iv - expected_put_iv) < 1e-9

    def test_compute_atm_iv_empty_chain_returns_nan_pair(self):
        """ATM IV returns (nan, nan) for an empty chain."""
        feed = OptionsChainFeed(access_token="tok")
        call_iv, put_iv = feed.compute_atm_iv(pd.DataFrame(), spot=25000.0)
        assert math.isnan(call_iv) and math.isnan(put_iv)


# ---------------------------------------------------------------------------
# compute_iv_skew
# ---------------------------------------------------------------------------

class TestComputeIvSkew:
    """Tests for OptionsChainFeed.compute_iv_skew."""

    def test_compute_iv_skew_positive_in_fear_regime(self):
        """IV skew (OTM put IV - OTM call IV) is positive in a fear regime.

        We manually set OTM put IV higher than OTM call IV to simulate a
        downside-skewed market and assert skew > 0.
        """
        feed = OptionsChainFeed(access_token="tok")
        # strikes spanning 5% either side of 25000 spot
        strikes = [23700.0, 23750.0, 25000.0, 26250.0, 26300.0]
        chain = _make_chain(strikes=strikes)

        spot = 25000.0
        # 5% OTM call target = 26250, 5% OTM put target = 23750

        # Force OTM put IV to 0.40, OTM call IV to 0.15 (fear skew)
        chain.loc[(chain["strike"] == 23750.0) & (chain["option_type"] == "PE"), "iv"] = 0.40
        chain.loc[(chain["strike"] == 26250.0) & (chain["option_type"] == "CE"), "iv"] = 0.15

        skew = feed.compute_iv_skew(chain, spot=spot)

        assert skew > 0, f"Expected positive skew in fear regime, got {skew}"

    def test_compute_iv_skew_empty_chain_returns_nan(self):
        """IV skew returns NaN for an empty chain."""
        feed = OptionsChainFeed(access_token="tok")
        assert math.isnan(feed.compute_iv_skew(pd.DataFrame(), spot=25000.0))


# ---------------------------------------------------------------------------
# snapshot — persistence
# ---------------------------------------------------------------------------

class TestSnapshot:
    """Tests for OptionsChainFeed.snapshot."""

    def _make_mock_response(self, chain: pd.DataFrame) -> MagicMock:
        """Return a mock httpx.Response wrapping the synthetic chain data."""
        payload = _make_upstox_api_response(chain)
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_snapshot_persists_chain_plus_strike_history(self):
        """snapshot() writes one row to each DB table and returns a metrics dict."""
        chain = _make_chain(strikes=[24900.0, 25000.0, 25100.0])
        mock_resp = self._make_mock_response(chain)

        # Build a minimal in-memory SQLite engine with the required tables.
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE options_chain_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    underlying_key TEXT,
                    underlying_symbol TEXT,
                    expiry TEXT,
                    ts TEXT,
                    spot_paisa INTEGER,
                    pcr REAL,
                    total_call_oi INTEGER,
                    total_put_oi INTEGER,
                    max_pain_strike REAL,
                    atm_iv_call REAL,
                    atm_iv_put REAL,
                    iv_skew REAL,
                    raw_chain TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE options_strike_oi_history (
                    underlying_key TEXT,
                    expiry TEXT,
                    strike REAL,
                    option_type TEXT,
                    ts TEXT,
                    oi INTEGER,
                    oi_change INTEGER,
                    iv REAL,
                    delta REAL,
                    gamma REAL,
                    vega REAL,
                    theta REAL,
                    PRIMARY KEY (underlying_key, expiry, strike, option_type, ts)
                )
            """))

        feed = OptionsChainFeed(access_token="tok", db_engine=engine)

        with patch("httpx.get", return_value=mock_resp):
            result = feed.snapshot(
                underlying_key=KEY,
                underlying_symbol=SYMBOL,
                expiry=EXPIRY,
                spot_paisa=2500000,
            )

        assert result is not None, "snapshot() should return a metrics dict"
        assert result["underlying_key"] == KEY
        assert result["total_call_oi"] > 0
        assert result["total_put_oi"] > 0
        assert math.isfinite(result["pcr"])

        # Verify DB rows were inserted
        with engine.connect() as conn:
            snap_count = conn.execute(
                text("SELECT COUNT(*) FROM options_chain_snapshots")
            ).scalar()
            strike_count = conn.execute(
                text("SELECT COUNT(*) FROM options_strike_oi_history")
            ).scalar()

        assert snap_count == 1, f"Expected 1 snapshot row, got {snap_count}"
        # 3 strikes × 2 option types = 6 strike rows
        assert strike_count == 6, f"Expected 6 strike rows, got {strike_count}"

    def test_snapshot_handles_empty_chain(self):
        """snapshot() returns None when the API returns no data."""
        empty_resp = MagicMock()
        empty_resp.json.return_value = {"status": "success", "data": []}
        empty_resp.raise_for_status.return_value = None

        feed = OptionsChainFeed(access_token="tok")

        with patch("httpx.get", return_value=empty_resp):
            result = feed.snapshot(
                underlying_key=KEY,
                underlying_symbol=SYMBOL,
                expiry=EXPIRY,
                spot_paisa=2500000,
            )

        assert result is None

    def test_snapshot_no_db_engine_skips_persistence(self):
        """snapshot() returns metrics dict even when no db_engine is supplied."""
        chain = _make_chain(strikes=[24900.0, 25000.0, 25100.0])
        mock_resp = self._make_mock_response(chain)

        feed = OptionsChainFeed(access_token="tok", db_engine=None)

        with patch("httpx.get", return_value=mock_resp):
            result = feed.snapshot(
                underlying_key=KEY,
                underlying_symbol=SYMBOL,
                expiry=EXPIRY,
                spot_paisa=2500000,
            )

        assert result is not None
        assert "pcr" in result


# ---------------------------------------------------------------------------
# OI change tracking across polls
# ---------------------------------------------------------------------------

class TestOiChangeTracking:
    """Verify that oi_change is computed correctly between successive polls."""

    def test_oi_change_computed_on_second_poll(self):
        """oi_change is tracked correctly in _prev_oi between successive polls.

        We call _persist_snapshot directly with controlled timestamps to bypass
        the datetime.now() indeterminism and verify the oi_change cache logic.
        """
        chain = _make_chain(strikes=[25000.0])
        initial_oi = int(chain.loc[
            (chain["strike"] == 25000.0) & (chain["option_type"] == "CE"), "oi"
        ].iloc[0])

        # Build minimal SQLite engine
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE options_chain_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    underlying_key TEXT, underlying_symbol TEXT, expiry TEXT, ts TEXT,
                    spot_paisa INTEGER, pcr REAL, total_call_oi INTEGER, total_put_oi INTEGER,
                    max_pain_strike REAL, atm_iv_call REAL, atm_iv_put REAL,
                    iv_skew REAL, raw_chain TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE options_strike_oi_history (
                    underlying_key TEXT, expiry TEXT, strike REAL, option_type TEXT, ts TEXT,
                    oi INTEGER, oi_change INTEGER, iv REAL, delta REAL, gamma REAL,
                    vega REAL, theta REAL,
                    PRIMARY KEY (underlying_key, expiry, strike, option_type, ts)
                )
            """))

        feed = OptionsChainFeed(access_token="tok", db_engine=engine)

        ts1 = datetime(2026, 5, 15, 9, 15, 0, tzinfo=IST)
        ts2 = datetime(2026, 5, 15, 9, 15, 30, tzinfo=IST)

        # Build a minimal result dict for _persist_snapshot
        def _make_result(oi_chain):
            return {
                "underlying_key": KEY,
                "underlying_symbol": SYMBOL,
                "expiry": EXPIRY,
                "spot_paisa": 2500000,
                "pcr": 1.0,
                "total_call_oi": int(oi_chain.loc[oi_chain["option_type"] == "CE", "oi"].sum()),
                "total_put_oi": int(oi_chain.loc[oi_chain["option_type"] == "PE", "oi"].sum()),
                "max_pain_strike": 25000.0,
                "atm_iv_call": 0.18,
                "atm_iv_put": 0.20,
                "iv_skew": 0.02,
            }

        raw_json = chain.to_dict(orient="records")

        # First persist — oi_change should be NULL (no previous OI cached)
        result1 = _make_result(chain)
        feed._persist_snapshot(result1, raw_json, chain, KEY, EXPIRY, ts1)

        with engine.connect() as conn:
            oi_change_first = conn.execute(
                text("""
                    SELECT oi_change FROM options_strike_oi_history
                    WHERE strike=25000.0 AND option_type='CE'
                    ORDER BY ts ASC LIMIT 1
                """)
            ).scalar()

        assert oi_change_first is None, "First poll oi_change should be NULL"

        # Second persist with increased OI
        chain2 = chain.copy()
        delta = 300
        chain2.loc[
            (chain2["strike"] == 25000.0) & (chain2["option_type"] == "CE"), "oi"
        ] += delta

        result2 = _make_result(chain2)
        raw_json2 = chain2.to_dict(orient="records")
        feed._persist_snapshot(result2, raw_json2, chain2, KEY, EXPIRY, ts2)

        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT oi_change FROM options_strike_oi_history
                    WHERE strike=25000.0 AND option_type='CE'
                    ORDER BY rowid ASC
                """)
            ).fetchall()

        assert len(rows) >= 2, f"Expected at least 2 rows, got {len(rows)}"
        second_oi_change = rows[1][0]
        assert second_oi_change == delta, (
            f"Expected oi_change={delta} on second poll, got {second_oi_change}"
        )
