"""Tests for src.execution.order_validator — Phase E.2.

All tests use inline mocks.  No live DB connection required.
IST timestamps, paisa for monetary values.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.execution.order_validator import OrderValidator, ValidationConfig


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

INSTRUMENT_KEY = "NSE_FO|NIFTY25APR24000CE"
INSTRUMENT_KEY_EQ = "NSE_EQ|INE002A01018"


def _make_validator(
    lot_size: int = 50,
    freeze_qty: int = 1800,
    available_paisa: int = 10_000_000,
    enable_margin_check: bool = True,
) -> OrderValidator:
    """Build an OrderValidator backed entirely by mocks."""
    # instrument_manager mock
    im = MagicMock()
    im.get_instrument.return_value = {
        "lot_size": lot_size,
        "freeze_quantity": freeze_qty,
    }

    # broker mock — get_funds returns int (paisa)
    broker = MagicMock()
    broker.get_funds.return_value = available_paisa

    cfg = ValidationConfig(
        margin_buffer_pct=5.0,
        enable_margin_check=enable_margin_check,
    )
    return OrderValidator(
        instrument_manager=im,
        broker=broker,
        db_engine=None,
        config=cfg,
    )


def _make_order(
    qty: int = 50,
    instrument_key: str = INSTRUMENT_KEY,
    side: str = "BUY",
    order_type: str = "MARKET",
    price: int | None = None,
    client_order_id: str = "TEST-001",
) -> dict:
    return {
        "client_order_id": client_order_id,
        "instrument_key": instrument_key,
        "side": side,
        "quantity": qty,
        "order_type": order_type,
        "price": price,
    }


# ---------------------------------------------------------------------------
# get_lot_size
# ---------------------------------------------------------------------------


class TestGetLotSize:
    def test_get_lot_size_reads_from_db(self):
        """Validator falls back to DB query when instrument_manager returns None."""
        im = MagicMock()
        im.get_instrument.return_value = None  # manager can't resolve it

        # Mock DB engine that returns lot_size=75, freeze_quantity=0
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, i: [75, 0][i]
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (75, 0)
        mock_conn.execute.return_value = mock_result
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        v = OrderValidator(instrument_manager=im, db_engine=mock_engine)
        lot = v.get_lot_size(INSTRUMENT_KEY)
        assert lot == 75

    def test_get_lot_size_returns_one_for_equity_default(self):
        """When neither manager nor DB can resolve, lot_size defaults to 1."""
        im = MagicMock()
        im.get_instrument.return_value = None
        # DB engine that returns nothing
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        v = OrderValidator(instrument_manager=im, db_engine=mock_engine)
        lot = v.get_lot_size(INSTRUMENT_KEY_EQ)
        assert lot == 1


# ---------------------------------------------------------------------------
# get_freeze_quantity
# ---------------------------------------------------------------------------


class TestGetFreezeQuantity:
    def test_get_freeze_quantity_returns_zero_when_unset(self):
        """freeze_quantity of 0 (or NULL-like None) maps to 0."""
        im = MagicMock()
        im.get_instrument.return_value = {"lot_size": 1, "freeze_quantity": None}
        v = OrderValidator(instrument_manager=im)
        fq = v.get_freeze_quantity(INSTRUMENT_KEY_EQ)
        assert fq == 0

    def test_get_freeze_quantity_returns_configured_value(self):
        im = MagicMock()
        im.get_instrument.return_value = {"lot_size": 50, "freeze_quantity": 1800}
        v = OrderValidator(instrument_manager=im)
        assert v.get_freeze_quantity(INSTRUMENT_KEY) == 1800


# ---------------------------------------------------------------------------
# round_to_lot
# ---------------------------------------------------------------------------


class TestRoundToLot:
    def test_round_to_lot_down(self):
        v = _make_validator()
        assert v.round_to_lot(175, 50, mode="down") == 150

    def test_round_to_lot_up(self):
        v = _make_validator()
        assert v.round_to_lot(151, 50, mode="up") == 200

    def test_round_to_lot_nearest_rounds_to_floor(self):
        """174 is equidistant between 150 and 200; nearest picks floor."""
        v = _make_validator()
        # 174 - 150 = 24, 200 - 174 = 26  → floor wins
        assert v.round_to_lot(174, 50, mode="nearest") == 150

    def test_round_to_lot_nearest_rounds_to_ceil(self):
        """176 is closer to 200."""
        v = _make_validator()
        # 176 - 150 = 26, 200 - 176 = 24 → ceil wins
        assert v.round_to_lot(176, 50, mode="nearest") == 200

    def test_round_to_lot_exact_multiple_unchanged(self):
        v = _make_validator()
        assert v.round_to_lot(200, 50, mode="down") == 200

    def test_round_to_lot_lot_size_one_is_identity(self):
        v = _make_validator()
        assert v.round_to_lot(137, 1, mode="down") == 137

    def test_round_to_lot_unknown_mode_raises(self):
        v = _make_validator()
        with pytest.raises(ValueError, match="Unknown rounding mode"):
            v.round_to_lot(100, 50, mode="closest")


# ---------------------------------------------------------------------------
# split_for_freeze
# ---------------------------------------------------------------------------


class TestSplitForFreeze:
    def test_split_for_freeze_returns_single_when_under(self):
        """qty ≤ freeze_qty → single-element list."""
        v = _make_validator()
        result = v.split_for_freeze(1800, 1800, 50)
        assert result == [1800]

    def test_split_for_freeze_splits_into_multiple_chunks(self):
        """qty = 3600, freeze = 1800 → two equal chunks."""
        v = _make_validator()
        result = v.split_for_freeze(3600, 1800, 50)
        assert result == [1800, 1800]
        assert sum(result) == 3600

    def test_split_for_freeze_chunks_are_lot_multiples(self):
        """Each chunk must be a multiple of lot_size."""
        v = _make_validator()
        result = v.split_for_freeze(4000, 1800, 50)
        for chunk in result:
            assert chunk % 50 == 0, f"Chunk {chunk} not a multiple of 50"
        assert sum(result) == 4000

    def test_split_for_freeze_zero_freeze_returns_single(self):
        """freeze_qty=0 means no limit; return the full qty unchanged."""
        v = _make_validator()
        result = v.split_for_freeze(9999, 0, 50)
        assert result == [9999]

    def test_split_for_freeze_already_within_limit(self):
        v = _make_validator()
        result = v.split_for_freeze(50, 1800, 50)
        assert result == [50]

    def test_split_for_freeze_odd_total(self):
        """Total qty that isn't a clean multiple of freeze_qty."""
        v = _make_validator(lot_size=50, freeze_qty=1800)
        # 2700 = 1800 + 900
        result = v.split_for_freeze(2700, 1800, 50)
        assert len(result) == 2
        assert result[0] == 1800
        assert result[1] == 900
        assert sum(result) == 2700


# ---------------------------------------------------------------------------
# check_margin
# ---------------------------------------------------------------------------


class TestCheckMargin:
    def test_check_margin_persists_record(self):
        """check_margin must call db_engine.begin() to persist a row."""
        # Set up DB mock
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_conn

        im = MagicMock()
        im.get_instrument.return_value = {"lot_size": 50, "freeze_quantity": 1800}
        broker = MagicMock()
        broker.get_funds.return_value = 50_000_000  # 5 lakh

        v = OrderValidator(instrument_manager=im, broker=broker, db_engine=mock_engine)
        order = _make_order(qty=50)
        result = v.check_margin(order)

        # DB should have been used (begin called)
        mock_engine.begin.assert_called_once()
        assert "approved" in result
        assert "required_paisa" in result
        assert "available_paisa" in result

    def test_check_margin_rejects_when_insufficient(self):
        """Margin check must reject when available < required + buffer."""
        im = MagicMock()
        im.get_instrument.return_value = {"lot_size": 50, "freeze_quantity": 1800}

        # We'll override _fetch_margin so required > available
        broker = MagicMock()
        broker.get_funds.return_value = 100  # near-zero available

        cfg = ValidationConfig(margin_buffer_pct=0.0, enable_margin_check=True)
        v = OrderValidator(instrument_manager=im, broker=broker, db_engine=None, config=cfg)

        # Patch _fetch_margin to return required=1_000_000, available=100
        with patch.object(v, "_fetch_margin", return_value=(1_000_000, 100)):
            result = v.check_margin(_make_order(qty=50))

        assert result["approved"] is False
        assert result["rejection_reason"] is not None
        assert "Insufficient" in result["rejection_reason"]

    def test_check_margin_approved_when_sufficient(self):
        """Margin check must approve when available > required + buffer."""
        im = MagicMock()
        im.get_instrument.return_value = {"lot_size": 50, "freeze_quantity": 1800}
        broker = MagicMock()
        broker.get_funds.return_value = 50_000_000

        cfg = ValidationConfig(margin_buffer_pct=5.0, enable_margin_check=True)
        v = OrderValidator(instrument_manager=im, broker=broker, db_engine=None, config=cfg)

        with patch.object(v, "_fetch_margin", return_value=(100_000, 50_000_000)):
            result = v.check_margin(_make_order(qty=50))

        assert result["approved"] is True
        assert result["rejection_reason"] is None


# ---------------------------------------------------------------------------
# validate — core integration
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_rejects_when_qty_zero_after_lot_round(self):
        """qty=10 with lot_size=50 rounds to 0 → rejected."""
        v = _make_validator(lot_size=50, freeze_qty=1800, available_paisa=50_000_000)
        order = _make_order(qty=10)
        result = v.validate(order)

        assert result["valid"] is False
        assert "0" in result["rejection_reason"] or "lot" in result["rejection_reason"].lower()
        assert result["split_orders"] == []

    def test_validate_returns_split_orders_for_large_qty(self):
        """qty=3600, freeze=1800 → valid=True with 2 slices."""
        v = _make_validator(lot_size=50, freeze_qty=1800, available_paisa=50_000_000)
        order = _make_order(qty=3600)
        result = v.validate(order)

        assert result["valid"] is True
        assert len(result["split_orders"]) == 2
        assert all(so["quantity"] == 1800 for so in result["split_orders"])

    def test_validate_rejects_when_total_margin_exceeds_available(self):
        """Aggregated margin across slices exceeds available → rejected."""
        im = MagicMock()
        im.get_instrument.return_value = {"lot_size": 50, "freeze_quantity": 1800}
        broker = MagicMock()
        broker.get_funds.return_value = 100  # very low

        cfg = ValidationConfig(margin_buffer_pct=0.0, enable_margin_check=True)
        v = OrderValidator(instrument_manager=im, broker=broker, db_engine=None, config=cfg)

        # _fetch_margin: required=1_000_000, available=100 for each slice
        with patch.object(v, "_fetch_margin", return_value=(1_000_000, 100)):
            result = v.validate(_make_order(qty=3600))

        assert result["valid"] is False
        assert "margin" in result["rejection_reason"].lower()

    def test_validate_passes_when_all_clear(self):
        """qty=50, lot=50, no freeze limit, ample margin → valid=True, one slice."""
        v = _make_validator(lot_size=50, freeze_qty=0, available_paisa=50_000_000)
        order = _make_order(qty=50)
        result = v.validate(order)

        assert result["valid"] is True
        assert result["rejection_reason"] is None
        assert len(result["split_orders"]) == 1
        assert result["split_orders"][0]["quantity"] == 50
        assert result["margin_check"]["approved"] is True

    def test_validate_rounds_qty_before_splitting(self):
        """qty=175 with lot=50 → rounded to 150, then split if needed."""
        v = _make_validator(lot_size=50, freeze_qty=1800, available_paisa=50_000_000)
        order = _make_order(qty=175)
        result = v.validate(order)

        assert result["valid"] is True
        total_qty = sum(so["quantity"] for so in result["split_orders"])
        assert total_qty == 150  # rounded down from 175

    def test_validate_no_split_when_under_freeze(self):
        """qty=1800 exactly at freeze limit → single slice, no split."""
        v = _make_validator(lot_size=50, freeze_qty=1800, available_paisa=50_000_000)
        order = _make_order(qty=1800)
        result = v.validate(order)

        assert result["valid"] is True
        assert len(result["split_orders"]) == 1
        assert result["split_orders"][0]["quantity"] == 1800

    def test_validate_margin_check_disabled(self):
        """With enable_margin_check=False, even 0 available → valid."""
        im = MagicMock()
        im.get_instrument.return_value = {"lot_size": 50, "freeze_quantity": 0}
        broker = MagicMock()
        broker.get_funds.return_value = 0

        cfg = ValidationConfig(margin_buffer_pct=5.0, enable_margin_check=False)
        v = OrderValidator(instrument_manager=im, broker=broker, db_engine=None, config=cfg)
        result = v.validate(_make_order(qty=50))

        assert result["valid"] is True


# ---------------------------------------------------------------------------
# get_recent_rejections
# ---------------------------------------------------------------------------


class TestGetRecentRejections:
    def test_returns_empty_when_no_db(self):
        v = OrderValidator()
        assert v.get_recent_rejections() == []

    def test_returns_rows_from_db(self):
        """Queries margin_checks and returns rejected rows."""
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        # Simulate one rejected row
        from datetime import datetime, timezone as tz

        ts_now = datetime.now(tz=tz.utc)
        fake_row = (
            1, ts_now, "TEST-001", INSTRUMENT_KEY, "BUY",
            50, 1_000_000, 100, False, "Insufficient margin"
        )
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [fake_row]
        mock_conn.execute.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        v = OrderValidator(db_engine=mock_engine)
        rows = v.get_recent_rejections(days=7)

        assert len(rows) == 1
        assert rows[0]["instrument_key"] == INSTRUMENT_KEY
        assert rows[0]["approved"] is False
        assert rows[0]["rejection_reason"] == "Insufficient margin"
