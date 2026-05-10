"""Tests for src/execution/reconciler.py — Phase E.3.

Strategy
--------
- All DB calls are exercised against a real local Postgres (LOCAL_DB_URL).
- The broker is always a MagicMock with a ``get_positions`` method.
- A module-scoped ``module_db`` fixture runs migrations and cleans up
  reconciliation_runs / reconciliation_log after the module finishes.
- Each test that hits the DB gets its own ``reconciler`` fixture with an
  isolated engine so tests don't bleed into each other.

Naming convention mirrors the spec:
  test_diff_*         — pure diff logic (no DB)
  test_apply_actions_* — action application (DB)
  test_run_cycle_*    — full cycle (DB)
  test_get_*          — query helpers (DB)
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# DB / engine helpers
# ---------------------------------------------------------------------------

_DB_URL_ENV = os.environ.get(
    "LOCAL_DB_URL", "postgresql://sandeepvangapandu@localhost:5432/tradebot"
)


def _sqla_url(url: str) -> str:
    if "+psycopg" in url:
        return url
    return url.replace("postgresql://", "postgresql+psycopg://", 1).replace(
        "postgres://", "postgresql+psycopg://", 1
    )


def _make_engine():
    from sqlalchemy import create_engine
    return create_engine(_sqla_url(_DB_URL_ENV), pool_pre_ping=True)


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def module_db():
    """Run migrations once; clean up reconciliation tables after module."""
    # Run migrations to ensure tables exist
    result = subprocess.run(
        ["python3", "-m", "src.storage.migrate", "up"],
        capture_output=True,
        text=True,
        cwd="/Users/sandeepvangapandu/Downloads/Trading",
    )
    assert result.returncode == 0, f"Migration failed: {result.stderr}"

    engine = _make_engine()
    yield engine

    # Cleanup
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM reconciliation_log WHERE TRUE"))
        conn.execute(text("DELETE FROM reconciliation_runs WHERE TRUE"))
    engine.dispose()


@pytest.fixture
def engine(module_db):
    """Per-test engine (reuses module_db connection pool)."""
    return module_db


@pytest.fixture
def mock_broker():
    """MagicMock broker with get_positions returning an empty list by default."""
    broker = MagicMock()
    broker.get_positions.return_value = []
    return broker


@pytest.fixture
def reconciler(mock_broker, engine):
    """PositionReconciler wired to the real test DB and a mock broker."""
    from src.execution.reconciler import PositionReconciler, ReconcilerConfig

    cfg = ReconcilerConfig(
        auto_insert_missed_fills=True,
        auto_update_position_qty=False,
        halt_on_remote_new=True,
        price_tolerance_pct=0.5,
    )
    return PositionReconciler(broker=mock_broker, db_engine=engine, config=cfg)


@pytest.fixture
def reconciler_no_auto(mock_broker, engine):
    """Reconciler with all auto-repair disabled."""
    from src.execution.reconciler import PositionReconciler, ReconcilerConfig

    cfg = ReconcilerConfig(
        auto_insert_missed_fills=False,
        auto_update_position_qty=False,
        halt_on_remote_new=False,
        price_tolerance_pct=0.5,
    )
    return PositionReconciler(broker=mock_broker, db_engine=engine, config=cfg)


# ---------------------------------------------------------------------------
# Sample position builders
# ---------------------------------------------------------------------------

def _local_pos(instrument_key="NSE_EQ|INE002A01018", entry_qty=100, exit_qty=0,
               entry_avg_price=240000, side="BUY", status="OPEN", pos_id=1):
    return {
        "id": pos_id,
        "instrument_key": instrument_key,
        "side": side,
        "entry_qty": entry_qty,
        "exit_qty": exit_qty,
        "entry_avg_price": entry_avg_price,
        "exit_avg_price": None,
        "status": status,
        "realized_pnl": 0,
    }


def _broker_pos(instrument_key="NSE_EQ|INE002A01018", quantity=100,
                average_price=240000, side="BUY"):
    return {
        "instrument_key": instrument_key,
        "quantity": quantity,
        "average_price": average_price,
        "side": side,
        "last_price": average_price,
    }


# ===========================================================================
# DIFF TESTS — pure logic, no DB
# ===========================================================================

class TestDiffClean:
    def test_diff_clean_when_local_matches_broker(self, reconciler):
        """Matching qty + price within tolerance → CLEAN event."""
        local  = [_local_pos()]
        broker = [_broker_pos()]

        events = reconciler.diff(local, broker)

        assert len(events) == 1
        assert events[0]["event_type"] == "CLEAN"
        assert events[0]["instrument_key"] == "NSE_EQ|INE002A01018"


class TestDiffMissedFill:
    def test_diff_detects_missed_fill_when_broker_qty_higher(self, reconciler):
        """Broker holds 150 but local only recorded 100 → MISSED_FILL."""
        local  = [_local_pos(entry_qty=100)]
        broker = [_broker_pos(quantity=150)]

        events = reconciler.diff(local, broker)

        assert any(e["event_type"] == "MISSED_FILL" for e in events)
        missed = next(e for e in events if e["event_type"] == "MISSED_FILL")
        assert missed["details"]["delta_qty"] == 50
        assert missed["details"]["local_net_qty"] == 100
        assert missed["details"]["broker_qty"] == 150


class TestDiffBrokerExit:
    def test_diff_detects_broker_exit_when_local_has_position_broker_doesnt(self, reconciler):
        """Position in local DB but absent from broker → BROKER_EXIT."""
        local  = [_local_pos()]
        broker = []  # broker returned nothing

        events = reconciler.diff(local, broker)

        assert len(events) == 1
        assert events[0]["event_type"] == "BROKER_EXIT"
        assert events[0]["broker_state"] is None


class TestDiffRemoteNew:
    def test_diff_detects_remote_new_when_broker_has_position_local_doesnt(self, reconciler):
        """Broker shows a position we have no local record for → REMOTE_NEW."""
        local  = []
        broker = [_broker_pos()]

        events = reconciler.diff(local, broker)

        assert len(events) == 1
        assert events[0]["event_type"] == "REMOTE_NEW"
        assert events[0]["local_state"] is None


class TestDiffPriceMismatch:
    def test_diff_detects_price_mismatch_above_tolerance(self, reconciler):
        """Same qty but avg price differs by more than 0.5% → PRICE_MISMATCH."""
        # local avg = 240000 paisa, broker avg = 241300 paisa → drift ~0.54%
        local  = [_local_pos(entry_avg_price=240000)]
        broker = [_broker_pos(average_price=241300)]

        events = reconciler.diff(local, broker)

        assert any(e["event_type"] == "PRICE_MISMATCH" for e in events), events
        pm = next(e for e in events if e["event_type"] == "PRICE_MISMATCH")
        assert pm["details"]["drift_pct"] > 0.5

    def test_diff_clean_when_price_within_tolerance(self, reconciler):
        """Price drift < 0.5% should still yield CLEAN."""
        # local = 240000, broker = 240100 → drift ~0.04%
        local  = [_local_pos(entry_avg_price=240000)]
        broker = [_broker_pos(average_price=240100)]

        events = reconciler.diff(local, broker)

        assert any(e["event_type"] == "CLEAN" for e in events)


class TestDiffQtyMismatch:
    def test_diff_detects_qty_mismatch_same_direction(self, reconciler):
        """Local holds 100, broker holds 80 → QTY_MISMATCH."""
        local  = [_local_pos(entry_qty=100)]
        broker = [_broker_pos(quantity=80)]

        events = reconciler.diff(local, broker)

        assert any(e["event_type"] == "QTY_MISMATCH" for e in events)
        qm = next(e for e in events if e["event_type"] == "QTY_MISMATCH")
        assert qm["details"]["delta_qty"] == 20


# ===========================================================================
# APPLY_ACTIONS TESTS — require DB
# ===========================================================================

def _insert_test_position(engine, *, instrument_key, entry_qty=100, entry_avg_price=240000,
                           side="BUY", status="OPEN") -> int:
    """Insert a synthetic position row and return its id."""
    from sqlalchemy import text
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "INSERT INTO positions (instrument_key, side, entry_qty, entry_avg_price, "
                "status) "
                "VALUES (:ikey, :side, :qty, :price, :status) "
                "RETURNING id"
            ),
            {"ikey": instrument_key, "side": side, "qty": entry_qty,
             "price": entry_avg_price, "status": status},
        ).fetchone()
    return int(row.id)


def _cleanup_position(engine, pos_id: int) -> None:
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM fills WHERE fill_id LIKE 'RECON_%'"))
        conn.execute(text("DELETE FROM positions WHERE id = :pid"), {"pid": pos_id})


class TestApplyActionsMissedFill:
    def test_apply_actions_inserts_fill_for_missed_fill_when_auto_enabled(
        self, reconciler, engine
    ):
        """With auto_insert_missed_fills=True a fill row is inserted and position qty updated."""
        from sqlalchemy import text

        ikey = "NSE_EQ|TESTMISSED1"
        pos_id = _insert_test_position(engine, instrument_key=ikey, entry_qty=100)

        try:
            # Create a reconciliation_runs row first
            with engine.begin() as conn:
                cycle_id = conn.execute(
                    text(
                        "INSERT INTO reconciliation_runs (total_local_positions, total_broker_positions, status) "
                        "VALUES (1, 1, 'RUNNING') RETURNING cycle_id"
                    )
                ).fetchone().cycle_id

            lp = _local_pos(instrument_key=ikey, pos_id=pos_id, entry_qty=100)
            bp = _broker_pos(instrument_key=ikey, quantity=150, average_price=241000)
            event = {
                "event_type": "MISSED_FILL",
                "instrument_key": ikey,
                "local_state": lp,
                "broker_state": bp,
                "details": {"delta_qty": 50, "local_net_qty": 100, "broker_qty": 150},
            }

            actions = reconciler.apply_actions([event], cycle_id)

            assert actions == 1

            # Verify fill inserted
            with engine.connect() as conn:
                fill = conn.execute(
                    text("SELECT * FROM fills WHERE fill_id LIKE 'RECON_%' AND instrument_key = :ikey"),
                    {"ikey": ikey},
                ).fetchone()
            assert fill is not None
            assert fill.quantity == 50

            # Verify position entry_qty updated
            with engine.connect() as conn:
                pos = conn.execute(
                    text("SELECT entry_qty, entry_avg_price FROM positions WHERE id = :pid"),
                    {"pid": pos_id},
                ).fetchone()
            assert pos.entry_qty == 150
        finally:
            _cleanup_position(engine, pos_id)

    def test_apply_actions_only_logs_when_auto_disabled(
        self, reconciler_no_auto, engine
    ):
        """With auto_insert_missed_fills=False no fill is inserted; action_taken=FLAGGED."""
        from sqlalchemy import text

        ikey = "NSE_EQ|TESTMISSED2"
        pos_id = _insert_test_position(engine, instrument_key=ikey, entry_qty=100)

        try:
            with engine.begin() as conn:
                cycle_id = conn.execute(
                    text(
                        "INSERT INTO reconciliation_runs (total_local_positions, total_broker_positions, status) "
                        "VALUES (1, 1, 'RUNNING') RETURNING cycle_id"
                    )
                ).fetchone().cycle_id

            lp = _local_pos(instrument_key=ikey, pos_id=pos_id, entry_qty=100)
            bp = _broker_pos(instrument_key=ikey, quantity=150)
            event = {
                "event_type": "MISSED_FILL",
                "instrument_key": ikey,
                "local_state": lp,
                "broker_state": bp,
                "details": {"delta_qty": 50, "local_net_qty": 100, "broker_qty": 150},
            }

            actions = reconciler_no_auto.apply_actions([event], cycle_id)

            assert actions == 0  # No auto action

            # position qty should NOT have changed
            with engine.connect() as conn:
                pos = conn.execute(
                    text("SELECT entry_qty FROM positions WHERE id = :pid"),
                    {"pid": pos_id},
                ).fetchone()
            assert pos.entry_qty == 100

            # Log row should still be written
            with engine.connect() as conn:
                log_row = conn.execute(
                    text(
                        "SELECT action_taken FROM reconciliation_log "
                        "WHERE cycle_id = :cid AND event_type = 'MISSED_FILL'"
                    ),
                    {"cid": cycle_id},
                ).fetchone()
            assert log_row is not None
            assert log_row.action_taken == "FLAGGED"
        finally:
            _cleanup_position(engine, pos_id)


class TestApplyActionsBrokerExit:
    def test_apply_actions_closes_local_position_on_broker_exit(self, reconciler, engine):
        """BROKER_EXIT should update status=CLOSED and set exit_avg_price + realized_pnl."""
        from sqlalchemy import text

        ikey = "NSE_EQ|TESTEXIT1"
        pos_id = _insert_test_position(
            engine, instrument_key=ikey, entry_qty=100, entry_avg_price=240000
        )

        try:
            with engine.begin() as conn:
                cycle_id = conn.execute(
                    text(
                        "INSERT INTO reconciliation_runs (total_local_positions, total_broker_positions, status) "
                        "VALUES (1, 0, 'RUNNING') RETURNING cycle_id"
                    )
                ).fetchone().cycle_id

            lp = _local_pos(instrument_key=ikey, pos_id=pos_id, entry_qty=100, entry_avg_price=240000)
            event = {
                "event_type": "BROKER_EXIT",
                "instrument_key": ikey,
                "local_state": lp,
                "broker_state": None,
                "details": {"reason": "absent from broker"},
            }

            actions = reconciler.apply_actions([event], cycle_id)

            assert actions == 1

            with engine.connect() as conn:
                pos = conn.execute(
                    text("SELECT status, closed_at FROM positions WHERE id = :pid"),
                    {"pid": pos_id},
                ).fetchone()
            assert pos.status == "CLOSED"
            assert pos.closed_at is not None
        finally:
            _cleanup_position(engine, pos_id)


class TestApplyActionsRemoteNew:
    def test_apply_actions_logs_critical_on_remote_new(self, reconciler, engine, caplog):
        """REMOTE_NEW should set halt_new_orders=True and emit CRITICAL log."""
        import logging
        from sqlalchemy import text

        with engine.begin() as conn:
            cycle_id = conn.execute(
                text(
                    "INSERT INTO reconciliation_runs (total_local_positions, total_broker_positions, status) "
                    "VALUES (0, 1, 'RUNNING') RETURNING cycle_id"
                )
            ).fetchone().cycle_id

        ikey = "NSE_EQ|TESTREMOTE1"
        bp = _broker_pos(instrument_key=ikey)
        event = {
            "event_type": "REMOTE_NEW",
            "instrument_key": ikey,
            "local_state": None,
            "broker_state": bp,
            "details": {"reason": "manual trade detected"},
        }

        with caplog.at_level(logging.CRITICAL, logger="src.execution.reconciler"):
            reconciler.apply_actions([event], cycle_id)

        assert reconciler.halt_new_orders is True
        assert any("REMOTE_NEW" in r.message for r in caplog.records)


# ===========================================================================
# RUN_CYCLE TESTS
# ===========================================================================

class TestRunCycle:
    def test_run_cycle_creates_run_record(self, reconciler, engine):
        """run_cycle() must insert a reconciliation_runs row."""
        from sqlalchemy import text

        result = reconciler.run_cycle()

        cycle_id = result["cycle_id"]
        assert cycle_id is not None

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM reconciliation_runs WHERE cycle_id = :cid"),
                {"cid": cycle_id},
            ).fetchone()
        assert row is not None

    def test_run_cycle_marks_complete_on_success(self, reconciler, engine):
        """A successful run must end with status=COMPLETE."""
        from sqlalchemy import text

        result = reconciler.run_cycle()

        assert result["status"] == "COMPLETE"

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT status FROM reconciliation_runs WHERE cycle_id = :cid"),
                {"cid": result["cycle_id"]},
            ).fetchone()
        assert row.status == "COMPLETE"

    def test_run_cycle_marks_failed_on_exception(self, mock_broker, engine):
        """If fetch_broker_positions raises, status should be FAILED in reconciliation_runs."""
        from sqlalchemy import text
        from src.execution.reconciler import PositionReconciler, ReconcilerConfig

        mock_broker.get_positions.side_effect = RuntimeError("connection timeout")

        rec = PositionReconciler(broker=mock_broker, db_engine=engine)
        result = rec.run_cycle()

        # Even on failure a run row must be present
        assert result["cycle_id"] is not None

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT status FROM reconciliation_runs WHERE cycle_id = :cid"),
                {"cid": result["cycle_id"]},
            ).fetchone()
        assert row.status == "FAILED"


# ===========================================================================
# QUERY HELPER TESTS
# ===========================================================================

class TestGetRecentEvents:
    def test_get_recent_events_filters_by_type(self, reconciler, engine):
        """get_recent_events(event_type='CLEAN') should only return CLEAN rows."""
        from sqlalchemy import text

        # Insert a run and a couple of log rows with different types
        with engine.begin() as conn:
            cycle_id = conn.execute(
                text(
                    "INSERT INTO reconciliation_runs (total_local_positions, total_broker_positions, status) "
                    "VALUES (1, 1, 'COMPLETE') RETURNING cycle_id"
                )
            ).fetchone().cycle_id
            conn.execute(
                text(
                    "INSERT INTO reconciliation_log "
                    "(cycle_id, event_type, instrument_key, action_taken) "
                    "VALUES (:cid, 'CLEAN', 'NSE_EQ|X', 'NONE'), "
                    "       (:cid, 'MISSED_FILL', 'NSE_EQ|Y', 'FLAGGED')"
                ),
                {"cid": cycle_id},
            )

        events = reconciler.get_recent_events(event_type="CLEAN", hours=1)

        assert all(e["event_type"] == "CLEAN" for e in events)
        assert any(e["instrument_key"] == "NSE_EQ|X" for e in events)
        assert not any(e["instrument_key"] == "NSE_EQ|Y" for e in events)

    def test_get_recent_events_returns_all_when_no_type_filter(self, reconciler, engine):
        """get_recent_events() with no filter returns all event types."""
        from sqlalchemy import text

        with engine.begin() as conn:
            cycle_id = conn.execute(
                text(
                    "INSERT INTO reconciliation_runs (total_local_positions, total_broker_positions, status) "
                    "VALUES (1, 1, 'COMPLETE') RETURNING cycle_id"
                )
            ).fetchone().cycle_id
            conn.execute(
                text(
                    "INSERT INTO reconciliation_log "
                    "(cycle_id, event_type, instrument_key, action_taken) "
                    "VALUES (:cid, 'BROKER_EXIT', 'NSE_EQ|Z', 'UPDATED_POSITION')"
                ),
                {"cid": cycle_id},
            )

        events = reconciler.get_recent_events(hours=1)
        types = {e["event_type"] for e in events}
        # Should include the row we just inserted
        assert "BROKER_EXIT" in types


class TestGetDriftSummary:
    def test_get_drift_summary_returns_dict_with_expected_keys(self, reconciler):
        """get_drift_summary() should return a dict with required keys."""
        summary = reconciler.get_drift_summary(days=1)

        required_keys = {
            "total_cycles", "total_events", "events_by_type",
            "auto_fixed", "flagged", "last_cycle_id", "last_cycle_status",
        }
        assert required_keys.issubset(set(summary.keys())), summary
