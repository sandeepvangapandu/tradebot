"""Tests for src.risk.kelly_sizer.KellySizer.

All tests use inline mocks — no external database or network required.
DB interactions are validated through mock engine/connection stubs.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch, call
import pytest

from src.risk.kelly_sizer import KellyConfig, KellySizer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _sizer(
    db_engine=None,
    rolling_window_days: int = 30,
    min_trades_for_kelly: int = 20,
    fallback_pct: float = 0.005,
    half_kelly: bool = True,
    min_pct: float = 0.001,
    max_pct: float = 0.05,
    confluence_multiplier: bool = True,
) -> KellySizer:
    cfg = KellyConfig(
        rolling_window_days=rolling_window_days,
        min_trades_for_kelly=min_trades_for_kelly,
        fallback_pct=fallback_pct,
        half_kelly=half_kelly,
        min_pct=min_pct,
        max_pct=max_pct,
        confluence_multiplier=confluence_multiplier,
    )
    return KellySizer(db_engine=db_engine, config=cfg)


def _perf(
    trade_count: int = 30,
    win_count: int = 18,
    loss_count: int = 12,
    win_rate: float = 0.60,
    avg_win_paisa: int = 20_000,
    avg_loss_paisa: int = 10_000,
    expectancy: float = 8_000.0,
    sharpe: float = 1.5,
    max_drawdown_pct: float = 5.0,
    total_pnl_paisa: int = 240_000,
    total_win_paisa: int = 360_000,
    total_loss_paisa: int = 120_000,
) -> dict:
    return {
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "avg_win_paisa": avg_win_paisa,
        "avg_loss_paisa": avg_loss_paisa,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "total_pnl_paisa": total_pnl_paisa,
        "total_win_paisa": total_win_paisa,
        "total_loss_paisa": total_loss_paisa,
    }


TODAY = date(2026, 5, 9)


# ===========================================================================
# compute_full_kelly
# ===========================================================================


class TestComputeFullKelly:
    """Tests for KellySizer.compute_full_kelly."""

    def test_compute_full_kelly_positive_edge(self):
        """60% win rate with 2:1 R:R should produce Kelly > 0.

        Kelly = (2*0.6 - 0.4) / 2 = 0.4
        """
        s = _sizer()
        result = s.compute_full_kelly(win_rate=0.60, avg_win=200, avg_loss=100)
        assert result == pytest.approx(0.40, abs=1e-9)

    def test_compute_full_kelly_no_edge_returns_zero(self):
        """50% win rate with 1:1 R:R is break-even — Kelly returns 0.

        Kelly = (1*0.5 - 0.5) / 1 = 0.0
        """
        s = _sizer()
        result = s.compute_full_kelly(win_rate=0.50, avg_win=100, avg_loss=100)
        assert result == pytest.approx(0.0)

    def test_compute_full_kelly_50_50_returns_zero(self):
        """Exact 50/50 with equal win/loss returns 0 (no edge)."""
        s = _sizer()
        result = s.compute_full_kelly(win_rate=0.5, avg_win=1000, avg_loss=1000)
        assert result == pytest.approx(0.0)

    def test_compute_full_kelly_high_winrate_high_edge(self):
        """75% win rate with 3:1 R:R should produce a large Kelly fraction.

        Kelly = (3*0.75 - 0.25) / 3 = 2.0/3 ≈ 0.6667
        Clamped to [0, 1] → 0.6667
        """
        s = _sizer()
        result = s.compute_full_kelly(win_rate=0.75, avg_win=300, avg_loss=100)
        assert result == pytest.approx(2 / 3, abs=1e-6)

    def test_compute_full_kelly_negative_edge_returns_zero(self):
        """40% win rate with 1:1 R:R is negative edge — returns 0."""
        s = _sizer()
        result = s.compute_full_kelly(win_rate=0.40, avg_win=100, avg_loss=100)
        assert result == pytest.approx(0.0)

    def test_compute_full_kelly_zero_avg_loss_returns_zero(self):
        """avg_loss == 0 is degenerate; must return 0 without ZeroDivisionError."""
        s = _sizer()
        result = s.compute_full_kelly(win_rate=0.6, avg_win=200, avg_loss=0)
        assert result == 0.0


# ===========================================================================
# compute_recommended_pct
# ===========================================================================


class TestComputeRecommendedPct:
    """Tests for KellySizer.compute_recommended_pct."""

    def test_compute_recommended_pct_applies_half_kelly(self):
        """Half-Kelly halves the full Kelly fraction before clamping."""
        s = _sizer(half_kelly=True, max_pct=1.0, min_pct=0.0)
        # full Kelly = 0.40 → half = 0.20
        perf = _perf(win_rate=0.60, avg_win_paisa=200, avg_loss_paisa=100)
        result = s.compute_recommended_pct(perf, confluence_score=None)
        assert result == pytest.approx(0.20, abs=1e-6)

    def test_compute_recommended_pct_caps_at_max(self):
        """Recommendation must not exceed max_pct even with very high Kelly."""
        s = _sizer(half_kelly=False, max_pct=0.05, min_pct=0.001)
        # Win rate 80%, R:R 5:1 → full Kelly ≈ 0.64  (way above 5% cap)
        perf = _perf(win_rate=0.80, avg_win_paisa=500, avg_loss_paisa=100)
        result = s.compute_recommended_pct(perf, confluence_score=None)
        assert result <= 0.05

    def test_compute_recommended_pct_floors_at_min(self):
        """When Kelly is extremely small, recommendation must be at least min_pct."""
        s = _sizer(half_kelly=True, min_pct=0.001, max_pct=0.05)
        # Barely positive edge: 51% win, 1:1 R:R → tiny Kelly
        perf = _perf(
            trade_count=20,
            win_rate=0.51,
            avg_win_paisa=100,
            avg_loss_paisa=100,
        )
        result = s.compute_recommended_pct(perf, confluence_score=None)
        assert result >= 0.001

    def test_compute_recommended_pct_uses_fallback_when_few_trades(self):
        """With fewer trades than min_trades_for_kelly, use fallback_pct."""
        s = _sizer(min_trades_for_kelly=20, fallback_pct=0.005)
        perf = _perf(trade_count=5)  # Only 5 trades — insufficient
        result = s.compute_recommended_pct(perf)
        assert result == pytest.approx(0.005)

    def test_compute_recommended_pct_scales_by_confluence(self):
        """confluence_score=50 should halve the base Kelly recommendation."""
        s = _sizer(
            half_kelly=True,
            confluence_multiplier=True,
            max_pct=1.0,
            min_pct=0.0,
        )
        # full Kelly = 0.40 → half = 0.20 → × 0.5 = 0.10
        perf = _perf(win_rate=0.60, avg_win_paisa=200, avg_loss_paisa=100)
        result = s.compute_recommended_pct(perf, confluence_score=50.0)
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_compute_recommended_pct_full_confluence_no_scaling(self):
        """confluence_score=100 applies scale factor of 1 (no reduction)."""
        s = _sizer(
            half_kelly=True,
            confluence_multiplier=True,
            max_pct=1.0,
            min_pct=0.0,
        )
        perf = _perf(win_rate=0.60, avg_win_paisa=200, avg_loss_paisa=100)
        result_100 = s.compute_recommended_pct(perf, confluence_score=100.0)
        result_none = s.compute_recommended_pct(perf, confluence_score=None)
        assert result_100 == pytest.approx(result_none, abs=1e-6)

    def test_compute_recommended_pct_zero_confluence_uses_fallback(self):
        """confluence_score=0 drives pct to 0, so fallback_pct is returned."""
        s = _sizer(
            half_kelly=True,
            confluence_multiplier=True,
            fallback_pct=0.005,
            min_pct=0.001,
            max_pct=0.05,
        )
        perf = _perf(win_rate=0.60, avg_win_paisa=200, avg_loss_paisa=100)
        result = s.compute_recommended_pct(perf, confluence_score=0.0)
        # scale → 0 → fallback triggered
        assert result == pytest.approx(0.005)


# ===========================================================================
# size_position
# ===========================================================================


class TestSizePosition:
    """Tests for KellySizer.size_position."""

    def test_size_position_returns_qty_from_risk_div_sl(self):
        """Quantity = (capital * pct) // sl_distance."""
        s = _sizer(min_trades_for_kelly=5)  # low threshold so Kelly kicks in

        # Patch evaluate_strategy to return known pct
        with patch.object(s, "evaluate_strategy") as mock_eval:
            mock_eval.return_value = {"recommended_pct": 0.02}
            result = s.size_position(
                strategy_name="test_strat",
                trade_date=TODAY,
                capital_paisa=100_000_000,  # 1 lakh INR in paisa (₹1,00,000)
                sl_distance_paisa=5_000,    # ₹50 SL in paisa
                confluence_score=80.0,
            )

        # risk_paisa = 100_000_000 * 0.02 = 2_000_000
        # position_qty = 2_000_000 // 5_000 = 400
        assert result["risk_paisa"] == 2_000_000
        assert result["position_qty"] == 400
        assert result["pct_used"] == pytest.approx(0.02)
        assert result["lot_count"] == 0  # Caller handles lot rounding

    def test_size_position_zero_sl_returns_zero_qty(self):
        """SL distance of 0 must not raise ZeroDivisionError — returns qty 0."""
        s = _sizer()
        with patch.object(s, "evaluate_strategy") as mock_eval:
            mock_eval.return_value = {"recommended_pct": 0.02}
            result = s.size_position(
                strategy_name="test_strat",
                trade_date=TODAY,
                capital_paisa=50_000_000,
                sl_distance_paisa=0,
            )
        assert result["position_qty"] == 0

    def test_size_position_lot_count_always_zero(self):
        """lot_count is always 0 — caller must round via instrument lot_size."""
        s = _sizer()
        with patch.object(s, "evaluate_strategy") as mock_eval:
            mock_eval.return_value = {"recommended_pct": 0.01}
            result = s.size_position("s", TODAY, 10_000_000, 1_000)
        assert result["lot_count"] == 0


# ===========================================================================
# compute_rolling_perf (mocked DB reads)
# ===========================================================================


class TestComputeRollingPerf:
    """Tests for KellySizer.compute_rolling_perf — mocked DB."""

    def _mock_engine(self, pnl_values: list[int]) -> MagicMock:
        """Return a mock engine whose connection returns ``pnl_values``."""
        engine = MagicMock()
        conn_cm = MagicMock()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            (v,) for v in pnl_values
        ]
        conn_cm.__enter__ = MagicMock(return_value=conn)
        conn_cm.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = conn_cm
        return engine

    def test_compute_rolling_perf_aggregates_from_daily_pnl(self):
        """With 30 synthetic trade rows, perf stats are aggregated correctly."""
        # 18 wins of 20_000 paisa, 12 losses of 10_000 paisa
        pnl_values = [20_000] * 18 + [-10_000] * 12
        engine = self._mock_engine(pnl_values)
        s = _sizer(db_engine=engine)

        perf = s.compute_rolling_perf("strat_a", TODAY)

        assert perf["trade_count"] == 30
        assert perf["win_count"] == 18
        assert perf["loss_count"] == 12
        assert perf["win_rate"] == pytest.approx(0.60)
        assert perf["avg_win_paisa"] == 20_000
        assert perf["avg_loss_paisa"] == 10_000
        assert perf["total_pnl_paisa"] == 18 * 20_000 - 12 * 10_000

    def test_compute_rolling_perf_empty_returns_defaults(self):
        """Empty DB result returns a safe zero-filled dict."""
        engine = self._mock_engine([])
        s = _sizer(db_engine=engine)

        perf = s.compute_rolling_perf("no_trades", TODAY)
        assert perf["trade_count"] == 0
        assert perf["win_rate"] == 0.0
        assert perf["avg_win_paisa"] == 0
        assert perf["avg_loss_paisa"] == 0

    def test_compute_rolling_perf_no_engine_returns_defaults(self):
        """When db_engine is None, returns safe defaults without error."""
        s = _sizer(db_engine=None)
        perf = s.compute_rolling_perf("strat_x", TODAY)
        assert perf["trade_count"] == 0


# ===========================================================================
# evaluate_strategy — DB persist path
# ===========================================================================


class TestEvaluateStrategy:
    """Tests for KellySizer.evaluate_strategy — verifies DB persist call."""

    def _make_engine_with_pnl(self, pnl_values: list[int]) -> MagicMock:
        engine = MagicMock()

        # connect() context manager (for _fetch_pnl_rows)
        conn_ctx = MagicMock()
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [(v,) for v in pnl_values]
        conn_ctx.__enter__ = MagicMock(return_value=conn)
        conn_ctx.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = conn_ctx

        # begin() context manager (for _upsert_allocation)
        begin_ctx = MagicMock()
        begin_conn = MagicMock()
        begin_ctx.__enter__ = MagicMock(return_value=begin_conn)
        begin_ctx.__exit__ = MagicMock(return_value=False)
        engine.begin.return_value = begin_ctx

        return engine

    def test_evaluate_strategy_persists_to_db(self):
        """evaluate_strategy must call engine.begin() to upsert the allocation."""
        pnl_values = [20_000] * 18 + [-10_000] * 12
        engine = self._make_engine_with_pnl(pnl_values)
        s = _sizer(db_engine=engine, min_trades_for_kelly=20)

        record = s.evaluate_strategy("strat_b", TODAY, confluence_score=80.0)

        # Persistence was attempted
        engine.begin.assert_called_once()
        # Returned record has the right structure
        assert "recommended_pct" in record
        assert "strategy_name" in record
        assert record["strategy_name"] == "strat_b"
        assert record["trade_date"] == TODAY
        assert record["recommended_pct"] > 0

    def test_evaluate_strategy_returns_fallback_when_few_trades(self):
        """With < min_trades_for_kelly trades, recommended_pct == fallback_pct."""
        pnl_values = [20_000] * 5  # Only 5 trades
        engine = self._make_engine_with_pnl(pnl_values)
        s = _sizer(
            db_engine=engine,
            min_trades_for_kelly=20,
            fallback_pct=0.005,
        )

        record = s.evaluate_strategy("strat_c", TODAY)
        assert record["recommended_pct"] == pytest.approx(0.005)

    def test_evaluate_strategy_no_edge_uses_fallback(self):
        """50% win rate 1:1 → no Kelly edge → recommended_pct == fallback_pct."""
        # 10 wins of 100, 10 losses of 100 → full_kelly = 0
        pnl_values = [100] * 10 + [-100] * 10
        engine = self._make_engine_with_pnl(pnl_values)
        s = _sizer(
            db_engine=engine,
            min_trades_for_kelly=20,
            fallback_pct=0.005,
        )

        record = s.evaluate_strategy("strat_d", TODAY)
        assert record["recommended_pct"] == pytest.approx(0.005)


# ===========================================================================
# get_today_allocation round-trip
# ===========================================================================


class TestGetTodayAllocation:
    """Tests for KellySizer.get_today_allocation."""

    def _make_fetch_engine(self, row: tuple | None) -> MagicMock:
        engine = MagicMock()
        conn_ctx = MagicMock()
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = row
        conn_ctx.__enter__ = MagicMock(return_value=conn)
        conn_ctx.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = conn_ctx
        return engine

    def test_get_today_allocation_roundtrip(self):
        """get_today_allocation returns a dict mapping column names to values."""
        row = (
            TODAY,           # trade_date
            "strat_e",       # strategy_name
            0.60,            # win_rate
            20_000,          # avg_win_paisa
            10_000,          # avg_loss_paisa
            8_000.0,         # expectancy
            0.80,            # edge
            0.40,            # full_kelly_pct
            0.20,            # half_kelly_pct
            0.02,            # recommended_pct
            30,              # trade_count
        )
        engine = self._make_fetch_engine(row)
        s = _sizer(db_engine=engine)

        result = s.get_today_allocation("strat_e", TODAY)

        assert result is not None
        assert result["strategy_name"] == "strat_e"
        assert result["trade_date"] == TODAY
        assert result["win_rate"] == pytest.approx(0.60)
        assert result["recommended_pct"] == pytest.approx(0.02)
        assert result["trade_count"] == 30

    def test_get_today_allocation_returns_none_when_missing(self):
        """Returns None when the DB row doesn't exist."""
        engine = self._make_fetch_engine(None)
        s = _sizer(db_engine=engine)

        result = s.get_today_allocation("strat_missing", TODAY)
        assert result is None

    def test_get_today_allocation_no_engine_returns_none(self):
        """Returns None without error when db_engine is None."""
        s = _sizer(db_engine=None)
        result = s.get_today_allocation("strat_x", TODAY)
        assert result is None


# ===========================================================================
# _aggregate_pnl_rows (pure maths, no DB)
# ===========================================================================


class TestAggregatePnlRows:
    """Tests for KellySizer._aggregate_pnl_rows static helper."""

    def test_aggregate_all_wins(self):
        """All winning rows → loss_count = 0, win_rate = 1.0."""
        result = KellySizer._aggregate_pnl_rows([100, 200, 150])
        assert result["win_count"] == 3
        assert result["loss_count"] == 0
        assert result["win_rate"] == pytest.approx(1.0)
        assert result["avg_loss_paisa"] == 0

    def test_aggregate_all_losses(self):
        """All losing rows → win_count = 0, win_rate = 0.0."""
        result = KellySizer._aggregate_pnl_rows([-100, -200])
        assert result["win_count"] == 0
        assert result["loss_count"] == 2
        assert result["win_rate"] == pytest.approx(0.0)
        assert result["avg_win_paisa"] == 0

    def test_aggregate_mixed(self):
        """Mixed results should correctly split wins and losses."""
        pnl = [200, -100, 200, -100, 200, -100]
        result = KellySizer._aggregate_pnl_rows(pnl)
        assert result["trade_count"] == 6
        assert result["win_count"] == 3
        assert result["loss_count"] == 3
        assert result["win_rate"] == pytest.approx(0.50)
        assert result["avg_win_paisa"] == 200
        assert result["avg_loss_paisa"] == 100

    def test_aggregate_empty_list(self):
        """Empty list returns zero-filled dict without error."""
        result = KellySizer._aggregate_pnl_rows([])
        assert result["trade_count"] == 0
        assert result["win_rate"] == 0.0
        assert result["sharpe"] is None
        assert result["max_drawdown_pct"] is None

    def test_aggregate_sharpe_computed_for_sufficient_data(self):
        """Sharpe ratio is non-None when there are at least 2 data points."""
        result = KellySizer._aggregate_pnl_rows([100, 200, -50, 300])
        assert result["sharpe"] is not None

    def test_aggregate_max_drawdown_computed(self):
        """Max drawdown is computed from cumulative PnL series."""
        # Rising then falling → drawdown from peak
        pnl = [1000, 1000, 1000, -5000, 1000]
        result = KellySizer._aggregate_pnl_rows(pnl)
        assert result["max_drawdown_pct"] is not None
        assert result["max_drawdown_pct"] >= 0
