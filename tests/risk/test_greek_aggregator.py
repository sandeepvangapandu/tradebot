"""Tests for src.risk.greek_aggregator.GreekAggregator.

All tests use inline mocks — no external database or network required.
DB interactions are validated through mock engine/connection stubs.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch, call

import pytest

from src.risk.greek_aggregator import GreekCapConfig, GreekAggregator


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TODAY = date(2026, 5, 9)

# ₹5 lakh capital expressed in paisa
CAPITAL_PAISA = 5_00_000 * 100  # 5,00,00,000 paisa = ₹5,00,000


def _aggregator(
    db_engine=None,
    max_abs_delta_pct: float = 30.0,
    max_gamma_per_lakh: float = 0.5,
    max_vega_per_lakh_per_pct: float = 500.0,
    max_theta_decay_pct: float = 1.0,
    delta_neutral_tolerance: float = 10.0,
) -> GreekAggregator:
    cfg = GreekCapConfig(
        max_abs_delta_pct=max_abs_delta_pct,
        max_gamma_per_lakh=max_gamma_per_lakh,
        max_vega_per_lakh_per_pct=max_vega_per_lakh_per_pct,
        max_theta_decay_pct=max_theta_decay_pct,
        delta_neutral_tolerance=delta_neutral_tolerance,
    )
    return GreekAggregator(db_engine=db_engine, config=cfg)


def _mock_engine_returning(greeks_rows: list[dict | None]):
    """Build a SQLAlchemy engine mock that returns Greek rows in sequence.

    Each element of ``greeks_rows`` is either:
      - a dict with keys {delta, gamma, vega, theta} → one fetchone() result
      - None → fetchone() returns None (no data found)
    """
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    # Build mock rows: tuple (delta, gamma, vega, theta) or None
    mock_rows = []
    for g in greeks_rows:
        if g is None:
            mock_rows.append(None)
        else:
            mock_rows.append((g["delta"], g["gamma"], g["vega"], g["theta"]))

    # fetchone() returns rows in sequence
    execute_result = MagicMock()
    execute_result.fetchone.side_effect = mock_rows
    conn.execute.return_value = execute_result

    return engine


def _mock_write_engine():
    """Build a SQLAlchemy engine mock that supports begin() writes."""
    engine = MagicMock()
    # connect() for reads
    read_conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=read_conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    read_conn.execute.return_value.fetchone.return_value = None

    # begin() for writes
    write_conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=write_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    return engine, write_conn, read_conn


# ---------------------------------------------------------------------------
# Empty positions
# ---------------------------------------------------------------------------


def test_aggregate_portfolio_zero_positions_returns_zero_greeks():
    """Aggregating an empty position list returns all-zero Greeks."""
    ag = _aggregator()
    result = ag.aggregate_portfolio([])

    assert result["total_delta"] == 0.0
    assert result["total_gamma"] == 0.0
    assert result["total_vega"] == 0.0
    assert result["total_theta"] == 0.0
    assert result["position_count"] == 0
    assert result["underlying_breakdown"] == {}


# ---------------------------------------------------------------------------
# Long call → positive delta
# ---------------------------------------------------------------------------


def test_aggregate_portfolio_long_call_positive_delta():
    """BUY 50 CE with delta=0.5 → portfolio delta = +25."""
    greeks_row = {"delta": 0.5, "gamma": 0.02, "vega": 30.0, "theta": -5.0}
    engine = _mock_engine_returning([greeks_row])
    ag = _aggregator(db_engine=engine)

    positions = [
        {
            "instrument_key": "NSE_INDEX|Nifty 50",
            "expiry": "2026-05-29",
            "strike": 24000.0,
            "option_type": "CE",
            "side": "BUY",
            "quantity": 50,
        }
    ]
    result = ag.aggregate_portfolio(positions)

    assert result["total_delta"] == pytest.approx(25.0)  # +1 * 50 * 0.5
    assert result["total_gamma"] == pytest.approx(1.0)   # 50 * 0.02
    assert result["total_vega"] == pytest.approx(1500.0) # 50 * 30
    assert result["total_theta"] == pytest.approx(-250.0) # 50 * -5
    assert result["position_count"] == 1


# ---------------------------------------------------------------------------
# Short put → positive delta
# ---------------------------------------------------------------------------


def test_aggregate_portfolio_short_put_positive_delta():
    """SELL 25 PE with delta=-0.4 → portfolio delta = +10 (short PE is positive delta)."""
    greeks_row = {"delta": -0.4, "gamma": 0.015, "vega": 20.0, "theta": -4.0}
    engine = _mock_engine_returning([greeks_row])
    ag = _aggregator(db_engine=engine)

    positions = [
        {
            "instrument_key": "NSE_INDEX|Nifty Bank",
            "expiry": "2026-05-29",
            "strike": 52000.0,
            "option_type": "PE",
            "side": "SELL",
            "quantity": 25,
        }
    ]
    result = ag.aggregate_portfolio(positions)

    # side_sign = -1; delta contrib = (-1) * 25 * (-0.4) = +10
    assert result["total_delta"] == pytest.approx(10.0)
    assert result["total_gamma"] == pytest.approx(-0.375)  # -1 * 25 * 0.015
    assert result["total_vega"] == pytest.approx(-500.0)   # -1 * 25 * 20
    assert result["total_theta"] == pytest.approx(100.0)   # -1 * 25 * -4


# ---------------------------------------------------------------------------
# Short straddle → negative gamma and vega
# ---------------------------------------------------------------------------


def test_aggregate_portfolio_short_straddle_negative_gamma_vega():
    """SELL CE + SELL PE at ATM → gamma and vega both negative."""
    ce_greeks = {"delta": 0.5, "gamma": 0.02, "vega": 40.0, "theta": -8.0}
    pe_greeks = {"delta": -0.5, "gamma": 0.02, "vega": 40.0, "theta": -8.0}
    engine = _mock_engine_returning([ce_greeks, pe_greeks])
    ag = _aggregator(db_engine=engine)

    positions = [
        {
            "instrument_key": "NSE_INDEX|Nifty 50",
            "expiry": "2026-05-29",
            "strike": 24000.0,
            "option_type": "CE",
            "side": "SELL",
            "quantity": 50,
        },
        {
            "instrument_key": "NSE_INDEX|Nifty 50",
            "expiry": "2026-05-29",
            "strike": 24000.0,
            "option_type": "PE",
            "side": "SELL",
            "quantity": 50,
        },
    ]
    result = ag.aggregate_portfolio(positions)

    # CE: sign=-1; delta = -1*50*0.5=-25; gamma=-1; vega=-2000; theta=+400
    # PE: sign=-1; delta = -1*50*(-0.5)=+25; gamma=-1; vega=-2000; theta=+400
    assert result["total_delta"] == pytest.approx(0.0)
    assert result["total_gamma"] == pytest.approx(-2.0)    # -1 + -1
    assert result["total_vega"] == pytest.approx(-4000.0)  # -2000 + -2000
    assert result["total_theta"] == pytest.approx(800.0)   # +400 + +400
    assert result["position_count"] == 2


# ---------------------------------------------------------------------------
# Underlying breakdown
# ---------------------------------------------------------------------------


def test_aggregate_portfolio_underlying_breakdown_sums_correctly():
    """Breakdown dict should separately track two different underlyings."""
    nifty_greeks = {"delta": 0.5, "gamma": 0.01, "vega": 20.0, "theta": -3.0}
    bnf_greeks = {"delta": 0.6, "gamma": 0.015, "vega": 25.0, "theta": -4.0}
    engine = _mock_engine_returning([nifty_greeks, bnf_greeks])
    ag = _aggregator(db_engine=engine)

    positions = [
        {
            "instrument_key": "NSE_INDEX|Nifty 50",
            "expiry": "2026-05-29",
            "strike": 24000.0,
            "option_type": "CE",
            "side": "BUY",
            "quantity": 50,
        },
        {
            "instrument_key": "NSE_INDEX|Nifty Bank",
            "expiry": "2026-05-29",
            "strike": 52000.0,
            "option_type": "CE",
            "side": "BUY",
            "quantity": 15,
        },
    ]
    result = ag.aggregate_portfolio(positions)
    bd = result["underlying_breakdown"]

    assert "NSE_INDEX|Nifty 50" in bd
    assert "NSE_INDEX|Nifty Bank" in bd

    # Nifty 50: +1 * 50 * 0.5 = 25
    assert bd["NSE_INDEX|Nifty 50"]["delta"] == pytest.approx(25.0)
    # BankNifty: +1 * 15 * 0.6 = 9
    assert bd["NSE_INDEX|Nifty Bank"]["delta"] == pytest.approx(9.0)
    # Totals
    assert result["total_delta"] == pytest.approx(34.0)


# ---------------------------------------------------------------------------
# snapshot() persists to DB
# ---------------------------------------------------------------------------


def test_snapshot_persists_to_db():
    """snapshot() calls engine.begin() and inserts a row."""
    engine, write_conn, read_conn = _mock_write_engine()

    # Make fetch_position_greeks return a zero-row (no DB data for positions)
    # by having read_conn.execute return None fetchone
    ag = _aggregator(db_engine=engine)

    # No open positions → aggregate returns all zeros; we still want INSERT called
    result = ag.snapshot(TODAY, [], CAPITAL_PAISA)

    assert result["total_delta"] == 0.0
    assert result["position_count"] == 0

    # Verify begin() was called (write path)
    engine.begin.assert_called()
    write_conn.execute.assert_called_once()

    # Inspect the call args: first positional arg is the text() object,
    # second is the params dict
    call_args = write_conn.execute.call_args
    params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
    # Params may be positional arg list or keyword — handle both
    if isinstance(params, dict):
        assert "trade_date" in params
        assert str(params["trade_date"]) == str(TODAY)


# ---------------------------------------------------------------------------
# is_delta_neutral
# ---------------------------------------------------------------------------


def test_is_delta_neutral_true_within_tolerance():
    """Portfolio delta within 10% of capital is neutral."""
    ag = _aggregator(delta_neutral_tolerance=10.0)
    # capital = ₹5L = 5,00,000 INR. 10% = 50,000 INR equiv delta
    # If total_delta = 1000 that is 1000/500000 = 0.2% << 10%
    greeks = {"total_delta": 1000.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    assert ag.is_delta_neutral(greeks, CAPITAL_PAISA) is True


def test_is_delta_neutral_false_outside_tolerance():
    """Portfolio delta outside 10% of capital is not neutral."""
    ag = _aggregator(delta_neutral_tolerance=10.0)
    # capital = ₹5L = 5,00,000 INR. 10% band = 50,000 INR delta
    # total_delta = 60,000 → 60000/500000 = 12% > 10%
    greeks = {"total_delta": 60_000.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    assert ag.is_delta_neutral(greeks, CAPITAL_PAISA) is False


# ---------------------------------------------------------------------------
# check_delta_cap
# ---------------------------------------------------------------------------


def test_check_delta_cap_false_when_exceeds_30_pct():
    """Adding delta that pushes |delta/capital| above 30% is rejected."""
    ag = _aggregator(max_abs_delta_pct=30.0)
    # capital = ₹5L = 5,00,000 INR. 30% = 1,50,000 INR delta limit
    # current delta = 0, add 2,00,000 → 40% > 30%
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    assert ag.check_delta_cap(200_000.0, current, CAPITAL_PAISA) is False


def test_check_delta_cap_true_when_within_limit():
    """Adding delta within 30% limit is allowed."""
    ag = _aggregator(max_abs_delta_pct=30.0)
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    # 100,000 / 500,000 = 20% < 30%
    assert ag.check_delta_cap(100_000.0, current, CAPITAL_PAISA) is True


# ---------------------------------------------------------------------------
# check_gamma_cap
# ---------------------------------------------------------------------------


def test_check_gamma_cap_false_when_exceeds():
    """Adding gamma that pushes |gamma/lakh| above cap is rejected."""
    ag = _aggregator(max_gamma_per_lakh=0.5)
    # capital = ₹5L → 5 lakhs. Cap = 0.5 * 5 = 2.5 total gamma
    # current = 0; add 3.0 → 3.0 / 5 = 0.6 > 0.5
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    assert ag.check_gamma_cap(3.0, current, CAPITAL_PAISA) is False


def test_check_gamma_cap_true_when_within():
    """Adding gamma within cap is allowed."""
    ag = _aggregator(max_gamma_per_lakh=0.5)
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    # 1.0 / 5 lakhs = 0.2 < 0.5
    assert ag.check_gamma_cap(1.0, current, CAPITAL_PAISA) is True


# ---------------------------------------------------------------------------
# check_vega_cap
# ---------------------------------------------------------------------------


def test_check_vega_cap_false_when_exceeds():
    """Adding vega that pushes |vega/lakh| above cap is rejected."""
    ag = _aggregator(max_vega_per_lakh_per_pct=500.0)
    # capital = ₹5L → 5 lakhs. Cap = 500 * 5 = 2,500 total vega
    # Add 3,000 → 3000/5 = 600 > 500
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    assert ag.check_vega_cap(3_000.0, current, CAPITAL_PAISA) is False


def test_check_vega_cap_true_when_within():
    """Adding vega within cap is allowed."""
    ag = _aggregator(max_vega_per_lakh_per_pct=500.0)
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    # 1,000 / 5 lakhs = 200 < 500
    assert ag.check_vega_cap(1_000.0, current, CAPITAL_PAISA) is True


# ---------------------------------------------------------------------------
# check_theta_budget
# ---------------------------------------------------------------------------


def test_check_theta_budget_false_when_decay_too_high():
    """Adding theta that pushes |theta/capital| above 1% is rejected."""
    ag = _aggregator(max_theta_decay_pct=1.0)
    # capital = ₹5L = 5,00,000 INR. 1% = ₹5,000 max daily decay
    # Add -6,000 → |6000|/500000 = 1.2% > 1%
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    assert ag.check_theta_budget(-6_000.0, current, CAPITAL_PAISA) is False


def test_check_theta_budget_true_when_within():
    """Adding theta within budget is allowed."""
    ag = _aggregator(max_theta_decay_pct=1.0)
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    # -3,000 → 3000/500000 = 0.6% < 1%
    assert ag.check_theta_budget(-3_000.0, current, CAPITAL_PAISA) is True


# ---------------------------------------------------------------------------
# can_add_options_position — pass
# ---------------------------------------------------------------------------


def test_can_add_options_position_passes_when_all_clear():
    """Proposed position well within all caps returns (True, None)."""
    greeks_row = {"delta": 0.5, "gamma": 0.001, "vega": 5.0, "theta": -1.0}
    engine = _mock_engine_returning([greeks_row])
    ag = _aggregator(db_engine=engine)

    proposed = {
        "instrument_key": "NSE_INDEX|Nifty 50",
        "expiry": "2026-05-29",
        "strike": 24000.0,
        "option_type": "CE",
        "side": "BUY",
        "quantity": 1,
    }
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    allowed, reason = ag.can_add_options_position(proposed, current, CAPITAL_PAISA)

    assert allowed is True
    assert reason is None


# ---------------------------------------------------------------------------
# can_add_options_position — DELTA_CAP rejection
# ---------------------------------------------------------------------------


def test_can_add_options_position_rejects_delta_cap():
    """Proposed position that breaches delta cap returns (False, 'DELTA_CAP')."""
    # delta = 1.0 per unit, quantity = 2,00,001 would push far past 30%
    greeks_row = {"delta": 1.0, "gamma": 0.001, "vega": 1.0, "theta": -0.1}
    engine = _mock_engine_returning([greeks_row])
    ag = _aggregator(db_engine=engine)

    proposed = {
        "instrument_key": "NSE_INDEX|Nifty 50",
        "expiry": "2026-05-29",
        "strike": 24000.0,
        "option_type": "CE",
        "side": "BUY",
        "quantity": 300_000,  # 300,000 * 1.0 delta = 300,000 >> 30% of ₹5L
    }
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    allowed, reason = ag.can_add_options_position(proposed, current, CAPITAL_PAISA)

    assert allowed is False
    assert reason == "DELTA_CAP"


# ---------------------------------------------------------------------------
# can_add_options_position — VEGA_CAP rejection
# ---------------------------------------------------------------------------


def test_can_add_options_position_rejects_vega_cap():
    """Proposed position that breaches vega cap returns (False, 'VEGA_CAP')."""
    # Small delta (OK), tiny gamma (OK), huge vega → VEGA_CAP
    greeks_row = {"delta": 0.01, "gamma": 0.00001, "vega": 1_000.0, "theta": -0.1}
    engine = _mock_engine_returning([greeks_row])
    ag = _aggregator(db_engine=engine, max_abs_delta_pct=30.0, max_gamma_per_lakh=0.5, max_vega_per_lakh_per_pct=500.0)

    proposed = {
        "instrument_key": "NSE_INDEX|Nifty 50",
        "expiry": "2026-05-29",
        "strike": 24000.0,
        "option_type": "CE",
        "side": "BUY",
        "quantity": 10,  # 10 * 1000 = 10,000 vega → 10000/5 lakhs = 2000 per lakh >> 500
    }
    current = {"total_delta": 0.0, "total_gamma": 0.0, "total_vega": 0.0, "total_theta": 0.0}
    allowed, reason = ag.can_add_options_position(proposed, current, CAPITAL_PAISA)

    assert allowed is False
    assert reason == "VEGA_CAP"


# ---------------------------------------------------------------------------
# get_today_snapshot roundtrip
# ---------------------------------------------------------------------------


def test_get_today_snapshot_roundtrip():
    """get_today_snapshot returns None when engine returns no row."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.fetchone.return_value = None

    ag = _aggregator(db_engine=engine)
    result = ag.get_today_snapshot(TODAY)
    assert result is None


def test_get_today_snapshot_returns_data_when_row_exists():
    """get_today_snapshot parses DB row into expected dict format."""
    from datetime import datetime, timezone, timedelta

    IST = timezone(timedelta(hours=5, minutes=30))
    ts_now = datetime(2026, 5, 9, 10, 30, 0, tzinfo=IST)
    fake_row = (
        ts_now,          # ts
        TODAY,           # trade_date
        25.0,            # total_delta
        -2.0,            # total_gamma
        -4000.0,         # total_vega
        800.0,           # total_theta
        None,            # total_rho
        2,               # position_count
        {"NSE_INDEX|Nifty 50": {"delta": 25.0}},  # underlying_breakdown
        CAPITAL_PAISA,   # capital_paisa
    )

    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.fetchone.return_value = fake_row

    ag = _aggregator(db_engine=engine)
    result = ag.get_today_snapshot(TODAY)

    assert result is not None
    assert result["total_delta"] == pytest.approx(25.0)
    assert result["total_gamma"] == pytest.approx(-2.0)
    assert result["total_vega"] == pytest.approx(-4000.0)
    assert result["total_theta"] == pytest.approx(800.0)
    assert result["position_count"] == 2
    assert result["capital_paisa"] == CAPITAL_PAISA
    assert result["total_rho"] is None
