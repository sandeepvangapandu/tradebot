"""Smoke tests for all fixture modules and scenario harnesses.

Validates that:
  - Every fixture module imports cleanly.
  - Factory functions return non-empty, correctly typed results.
  - Every scenario ``setup()`` returns a dict with all required keys.
  - Runs well under 1 second (all data is synthetic / in-memory).

Run with::

    python -m pytest tests/test_fixtures_smoke.py -v
"""

from __future__ import annotations

import importlib
from datetime import datetime

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Required keys that every scenario dict must contain
# ---------------------------------------------------------------------------

_SCENARIO_REQUIRED_KEYS = {
    "name",
    "description",
    "ticks",
    "warmup_bars",
    "option_chains",
    "news",
    "macro_state",
    "expected_outcome",
}

_EXPECTED_OUTCOME_REQUIRED_KEYS = {
    "signal_count_min",
    "orders_placed_min",
    "rejection_reason",
}

# ---------------------------------------------------------------------------
# Fixture module smoke tests
# ---------------------------------------------------------------------------


class TestUpstoxWSPayloads:
    """Smoke tests for tests.fixtures.upstox_ws_payloads."""

    def setup_method(self):
        from tests.fixtures import upstox_ws_payloads as mod
        self.mod = mod

    def test_make_ltpc_tick_returns_dict_with_feeds(self):
        tick = self.mod.make_ltpc_tick("NSE_EQ|INE002A01018", 290_000)
        assert isinstance(tick, dict)
        assert "feeds" in tick
        assert "NSE_EQ|INE002A01018" in tick["feeds"]

    def test_make_full_tick_returns_feed_with_depth(self):
        tick = self.mod.make_full_tick("NSE_EQ|INE002A01018", 290_000)
        assert isinstance(tick, dict)
        assert "feeds" in tick
        feed = tick["feeds"]["NSE_EQ|INE002A01018"]
        assert "ff" in feed
        depth = feed["ff"]["marketFF"]["depth"]
        assert len(depth["buy"]) == 5
        assert len(depth["sell"]) == 5

    def test_make_index_tick_returns_dict(self):
        tick = self.mod.make_index_tick("NSE_INDEX|Nifty 50", 2_450_000)
        assert isinstance(tick, dict)
        assert "NSE_INDEX|Nifty 50" in tick["feeds"]

    def test_stream_ticks_returns_correct_count(self):
        ticks = self.mod.stream_ticks("NSE_EQ|INE002A01018", 290_000, count=20)
        assert len(ticks) == 20
        # Verify timestamps are ascending
        timestamps = [t["currentTs"] for t in ticks]
        assert timestamps == sorted(timestamps)

    def test_stream_ticks_deterministic_with_seed(self):
        ticks_a = self.mod.stream_ticks("NSE_EQ|INE002A01018", 290_000, count=5, seed=99)
        ticks_b = self.mod.stream_ticks("NSE_EQ|INE002A01018", 290_000, count=5, seed=99)
        prices_a = [list(t["feeds"].values())[0]["ltpc"]["ltp"] for t in ticks_a]
        prices_b = [list(t["feeds"].values())[0]["ltpc"]["ltp"] for t in ticks_b]
        assert prices_a == prices_b


class TestOptionChain:
    """Smoke tests for tests.fixtures.option_chain."""

    def setup_method(self):
        from tests.fixtures import option_chain as mod
        self.mod = mod

    def test_make_option_chain_returns_dataframe(self):
        df = self.mod.make_option_chain("NIFTY", 24_500.0, "2026-05-15")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_option_chain_has_required_columns(self):
        df = self.mod.make_option_chain("RELIANCE", 2_900.0, "2026-05-15")
        required = {"strike_price", "option_type", "ltp", "iv", "delta", "oi", "volume"}
        assert required.issubset(set(df.columns))

    def test_option_chain_has_both_ce_and_pe(self):
        df = self.mod.make_option_chain("NIFTY", 24_500.0, "2026-05-15")
        assert set(df["option_type"].unique()) == {"CE", "PE"}

    def test_atm_strike_rounds_correctly(self):
        # 24500 / 50 = 490.0 → ATM = 24500
        assert self.mod.atm_strike(24_500.0, 50) == 24_500.0
        # 24525 / 50 = 490.5 → rounds to 490 (banker's rounding) → 24500
        # Use an unambiguous case: 24560 / 50 = 491.2 → rounds to 491 → 24550
        assert self.mod.atm_strike(24_560.0, 50) == 24_550.0
        # 24600 / 50 = 492.0 → ATM = 24600
        assert self.mod.atm_strike(24_600.0, 50) == 24_600.0

    def test_realistic_iv_clamped(self):
        iv = self.mod.realistic_iv(24_500.0, 24_500.0, 7)
        assert 0.05 <= iv <= 2.0

    def test_top_10_underlyings_constant(self):
        assert len(self.mod.TOP_10_UNDERLYINGS) == 10
        assert "RELIANCE" in self.mod.TOP_10_UNDERLYINGS

    def test_indices_constant(self):
        assert "NIFTY" in self.mod.INDICES
        assert "BANKNIFTY" in self.mod.INDICES


class TestInstrumentMasterMini:
    """Smoke tests for tests.fixtures.instrument_master_mini."""

    def setup_method(self):
        from tests.fixtures import instrument_master_mini as mod
        self.mod = mod

    def test_make_mini_master_returns_dataframe(self):
        df = self.mod.make_mini_master()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 12

    def test_master_has_required_columns(self):
        df = self.mod.make_mini_master()
        required = {
            "segment", "name", "instrument_key", "trading_symbol",
            "lot_size", "tick_size", "expiry", "strike_price",
            "option_type", "underlying_key", "underlying_symbol",
            "exchange_token", "isin", "instrument_type",
            "short_name", "freeze_quantity",
        }
        assert required.issubset(set(df.columns))

    def test_master_contains_nifty_and_banknifty(self):
        df = self.mod.make_mini_master()
        syms = df["trading_symbol"].tolist()
        assert "NIFTY" in syms
        assert "BANKNIFTY" in syms

    def test_master_contains_equity_instruments(self):
        df = self.mod.make_mini_master()
        eq = df[df["segment"] == "NSE_EQ"]
        assert len(eq) >= 10


class TestHistoricalBars:
    """Smoke tests for tests.fixtures.historical_bars."""

    def setup_method(self):
        from tests.fixtures import historical_bars as mod
        self.mod = mod

    def test_make_ohlcv_bars_returns_dataframe(self):
        start = datetime(2026, 3, 1)
        end = datetime(2026, 3, 5)
        df = self.mod.make_ohlcv_bars("RELIANCE", start, end, timeframe_minutes=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_ohlcv_columns_present(self):
        start = datetime(2026, 3, 3)
        end = datetime(2026, 3, 4)
        df = self.mod.make_ohlcv_bars("INFY", start, end, timeframe_minutes=15)
        assert {"timestamp", "symbol", "open", "high", "low", "close", "volume"}.issubset(set(df.columns))

    def test_ohlcv_high_gte_low(self):
        start = datetime(2026, 3, 3)
        end = datetime(2026, 3, 4)
        df = self.mod.make_ohlcv_bars("TCS", start, end, timeframe_minutes=5)
        assert (df["high"] >= df["low"]).all()

    def test_make_warmup_bars_returns_dict(self):
        keys = ["NSE_EQ|INE002A01018", "NSE_EQ|INE009A01021"]
        result = self.mod.make_warmup_bars(keys, days=5, timeframe_minutes=15)
        assert isinstance(result, dict)
        assert len(result) == 2
        for key in keys:
            assert isinstance(result[key], pd.DataFrame)
            assert len(result[key]) > 0

    def test_make_warmup_bars_deterministic(self):
        keys = ["NSE_EQ|INE002A01018"]
        r1 = self.mod.make_warmup_bars(keys, days=5, timeframe_minutes=15, seed=7)
        r2 = self.mod.make_warmup_bars(keys, days=5, timeframe_minutes=15, seed=7)
        pd.testing.assert_frame_equal(r1[keys[0]], r2[keys[0]])


class TestNewsRSS:
    """Smoke tests for tests.fixtures.news_rss."""

    def setup_method(self):
        from tests.fixtures import news_rss as mod
        self.mod = mod

    def test_make_rss_xml_returns_string(self):
        articles = self.mod.sample_articles_neutral(count=3)
        xml_str = self.mod.make_rss_xml(articles)
        assert isinstance(xml_str, str)
        assert "<rss" in xml_str
        assert "version=" in xml_str

    def test_rss_xml_is_valid_xml(self):
        import xml.etree.ElementTree as ET
        articles = self.mod.sample_articles_positive_for("TCS", count=2)
        xml_str = self.mod.make_rss_xml(articles)
        # Should not raise
        root = ET.fromstring(xml_str)
        assert root.tag == "rss"

    def test_negative_articles_mention_symbol(self):
        articles = self.mod.sample_articles_negative_for("INFY", count=3)
        assert len(articles) == 3
        for art in articles:
            assert "INFY" in art["title"]

    def test_positive_articles_mention_symbol(self):
        articles = self.mod.sample_articles_positive_for("RELIANCE", count=2)
        assert len(articles) == 2
        for art in articles:
            assert "RELIANCE" in art["title"]

    def test_neutral_articles_count(self):
        articles = self.mod.sample_articles_neutral(count=7)
        assert len(articles) == 7

    def test_articles_have_required_keys(self):
        articles = self.mod.sample_articles_neutral(count=3)
        for art in articles:
            assert "title" in art
            assert "link" in art
            assert "description" in art
            assert "pubDate" in art


class TestMacroData:
    """Smoke tests for tests.fixtures.macro_data."""

    def setup_method(self):
        from tests.fixtures import macro_data as mod
        self.mod = mod
        self.start = datetime(2026, 3, 1)
        self.end = datetime(2026, 3, 31)

    def test_make_vix_series_normal(self):
        df = self.mod.make_vix_series(self.start, self.end, regime="normal")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "vix" in df.columns
        assert (df["vix"] > 0).all()

    def test_make_vix_series_spike_regime(self):
        df = self.mod.make_vix_series(self.start, self.end, regime="spike")
        assert df["vix"].max() > 20  # spike regime should produce high VIX

    def test_make_vix_series_invalid_regime_raises(self):
        with pytest.raises(ValueError, match="regime must be one of"):
            self.mod.make_vix_series(self.start, self.end, regime="unknown")

    def test_make_usdinr_series(self):
        df = self.mod.make_usdinr_series(self.start, self.end)
        assert isinstance(df, pd.DataFrame)
        assert "usdinr" in df.columns
        assert (df["usdinr"] > 60).all()
        assert (df["usdinr"] < 120).all()

    def test_make_crude_series(self):
        df = self.mod.make_crude_series(self.start, self.end)
        assert isinstance(df, pd.DataFrame)
        assert "crude_usd" in df.columns
        assert (df["crude_usd"] > 0).all()

    def test_make_fii_dii_flows_neutral(self):
        df = self.mod.make_fii_dii_flows(self.start, self.end, regime="neutral")
        assert isinstance(df, pd.DataFrame)
        required_cols = {"date", "fii_net", "dii_net", "net_total"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) > 0

    def test_make_fii_dii_flows_panic_extreme(self):
        df = self.mod.make_fii_dii_flows(self.start, self.end, regime="panic")
        assert df["fii_net"].mean() < -1000  # heavy selling

    def test_macro_series_trading_days_only(self):
        df = self.mod.make_vix_series(self.start, self.end, regime="normal")
        # All dates should be weekdays
        weekdays = df["date"].apply(lambda d: d.weekday())
        assert (weekdays < 5).all()

    def test_macro_deterministic_with_seed(self):
        df1 = self.mod.make_vix_series(self.start, self.end, regime="normal", seed=77)
        df2 = self.mod.make_vix_series(self.start, self.end, regime="normal", seed=77)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Scenario smoke tests
# ---------------------------------------------------------------------------

_SCENARIO_MODULES = [
    "tests.fixtures.scenarios.bull_open_breakout",
    "tests.fixtures.scenarios.earnings_blackout",
    "tests.fixtures.scenarios.liquidity_collapse",
    "tests.fixtures.scenarios.correlated_overload",
    "tests.fixtures.scenarios.vix_spike_regime",
    "tests.fixtures.scenarios.partial_fill_recovery",
    "tests.fixtures.scenarios.connection_drop",
    "tests.fixtures.scenarios.eod_squareoff",
    "tests.fixtures.scenarios.news_negative_reject",
    "tests.fixtures.scenarios.sector_rotation_winner",
]


@pytest.mark.parametrize("module_path", _SCENARIO_MODULES)
def test_scenario_imports(module_path: str):
    """Each scenario module must import without error."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "setup"), f"{module_path} must expose a setup() function"


@pytest.mark.parametrize("module_path", _SCENARIO_MODULES)
def test_scenario_setup_returns_dict(module_path: str):
    """Each scenario setup() must return a dict with all required keys."""
    mod = importlib.import_module(module_path)
    result = mod.setup()
    assert isinstance(result, dict), f"{module_path}: setup() must return dict"
    missing = _SCENARIO_REQUIRED_KEYS - set(result.keys())
    assert not missing, f"{module_path}: missing keys {missing}"


@pytest.mark.parametrize("module_path", _SCENARIO_MODULES)
def test_scenario_expected_outcome_keys(module_path: str):
    """Each scenario expected_outcome must have all required sub-keys."""
    mod = importlib.import_module(module_path)
    result = mod.setup()
    eo = result["expected_outcome"]
    assert isinstance(eo, dict)
    missing = _EXPECTED_OUTCOME_REQUIRED_KEYS - set(eo.keys())
    assert not missing, f"{module_path}: expected_outcome missing keys {missing}"


@pytest.mark.parametrize("module_path", _SCENARIO_MODULES)
def test_scenario_ticks_non_empty_or_list(module_path: str):
    """Each scenario ticks field must be a list."""
    mod = importlib.import_module(module_path)
    result = mod.setup()
    assert isinstance(result["ticks"], list), f"{module_path}: ticks must be a list"
    assert len(result["ticks"]) > 0, f"{module_path}: ticks must not be empty"


@pytest.mark.parametrize("module_path", _SCENARIO_MODULES)
def test_scenario_warmup_bars_is_dict(module_path: str):
    """Each scenario warmup_bars must be a dict."""
    mod = importlib.import_module(module_path)
    result = mod.setup()
    assert isinstance(result["warmup_bars"], dict), f"{module_path}: warmup_bars must be dict"


@pytest.mark.parametrize("module_path", _SCENARIO_MODULES)
def test_scenario_news_is_list(module_path: str):
    """Each scenario news must be a list."""
    mod = importlib.import_module(module_path)
    result = mod.setup()
    assert isinstance(result["news"], list), f"{module_path}: news must be list"


@pytest.mark.parametrize("module_path", _SCENARIO_MODULES)
def test_scenario_name_is_string(module_path: str):
    """Each scenario name must be a non-empty string."""
    mod = importlib.import_module(module_path)
    result = mod.setup()
    assert isinstance(result["name"], str) and result["name"], \
        f"{module_path}: name must be non-empty string"


@pytest.mark.parametrize("module_path", _SCENARIO_MODULES)
def test_scenario_macro_state_has_vix(module_path: str):
    """Each scenario macro_state must contain a 'vix' key."""
    mod = importlib.import_module(module_path)
    result = mod.setup()
    assert "vix" in result["macro_state"], \
        f"{module_path}: macro_state must contain 'vix'"


# ---------------------------------------------------------------------------
# Individual scenario sanity checks
# ---------------------------------------------------------------------------

def test_bull_open_breakout_has_upward_drift():
    """Bull breakout ticks should show rising prices overall."""
    from tests.fixtures.scenarios.bull_open_breakout import setup
    s = setup()
    ticks = s["ticks"]
    # Extract LTP values from ltpc ticks
    prices = [
        list(t["feeds"].values())[0]["ltpc"]["ltp"]
        for t in ticks
        if "ltpc" in list(t["feeds"].values())[0]
    ]
    assert prices[-1] > prices[0], "Bull breakout: final price should be higher than first"


def test_earnings_blackout_rejection_reason():
    """Earnings blackout must specify rejection_reason."""
    from tests.fixtures.scenarios.earnings_blackout import setup
    s = setup()
    assert s["expected_outcome"]["rejection_reason"] == "earnings_blackout"
    assert "blackout_instruments" in s["macro_state"]


def test_liquidity_collapse_rejection_reason():
    """Liquidity collapse must specify rejection_reason."""
    from tests.fixtures.scenarios.liquidity_collapse import setup
    s = setup()
    assert s["expected_outcome"]["rejection_reason"] == "insufficient_liquidity"


def test_vix_spike_scenario_has_two_phases():
    """VIX spike should have both calm and spike ticks."""
    from tests.fixtures.scenarios.vix_spike_regime import setup
    s = setup()
    assert len(s["ticks"]) >= 20
    assert s["macro_state"]["vix_post_spike"] > s["macro_state"]["vix_pre_spike"]


def test_connection_drop_has_sentinel():
    """Connection drop scenario must contain the drop sentinel dict."""
    from tests.fixtures.scenarios.connection_drop import setup
    s = setup()
    sentinels = [t for t in s["ticks"] if isinstance(t, dict) and t.get("event") == "connection_drop"]
    assert len(sentinels) == 1, "Must contain exactly one connection_drop sentinel"


def test_eod_squareoff_has_open_positions():
    """EOD squareoff must describe open MIS positions in macro_state."""
    from tests.fixtures.scenarios.eod_squareoff import setup
    s = setup()
    assert "open_mis_positions" in s["macro_state"]
    assert len(s["macro_state"]["open_mis_positions"]) == 2
    assert s["expected_outcome"]["orders_placed_min"] == 2


def test_news_negative_reject_has_rss_xml():
    """Negative news rejection must expose raw RSS XML in macro_state."""
    import xml.etree.ElementTree as ET
    from tests.fixtures.scenarios.news_negative_reject import setup
    s = setup()
    assert "news_rss_xml" in s["macro_state"]
    xml_str = s["macro_state"]["news_rss_xml"]
    root = ET.fromstring(xml_str)
    assert root.tag == "rss"


def test_correlated_overload_signal_count():
    """Correlated overload must have >= 5 signals expected."""
    from tests.fixtures.scenarios.correlated_overload import setup
    s = setup()
    assert s["expected_outcome"]["signal_count_min"] >= 5
    assert len(s["ticks"]) >= 50  # 5 instruments * 10 ticks each


def test_sector_rotation_has_it_option_chains():
    """Sector rotation must include INFY and TCS option chains."""
    from tests.fixtures.scenarios.sector_rotation_winner import setup
    s = setup()
    assert "INFY" in s["option_chains"]
    assert "TCS" in s["option_chains"]
