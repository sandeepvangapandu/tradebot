"""Position reconciler — detect drift between local and broker state.

Every 30 seconds this module compares the local ``positions`` table (status
IN ('OPEN','PARTIAL')) against the broker's live positions endpoint.  When it
finds drift it either auto-repairs (inserts missing fills, closes orphaned
positions) or flags the mismatch for human review, depending on the
``ReconcilerConfig`` settings.

All monetary values are in **Paisa** (1 INR = 100 paisa).
All timestamps are in **IST (Asia/Kolkata)**.

Tables written:
  - ``reconciliation_runs``  — one row per poll cycle.
  - ``reconciliation_log``   — one row per event within a cycle.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Enumerations & Config
# ---------------------------------------------------------------------------

class ReconciliationEventType(str):
    """Event types produced by the reconciler diff step.

    String constants (not Enum subclass) so they can round-trip through
    Postgres TEXT columns without any special serialisation.
    """
    MISSED_FILL    = "MISSED_FILL"     # Broker has fill, local doesn't
    POSITION_DRIFT = "POSITION_DRIFT"  # Different qty between local/broker
    QTY_MISMATCH   = "QTY_MISMATCH"   # Same direction, different qty
    PRICE_MISMATCH = "PRICE_MISMATCH"  # Avg price differs > tolerance
    BROKER_EXIT    = "BROKER_EXIT"     # Local has position, broker doesn't (closed externally)
    REMOTE_NEW     = "REMOTE_NEW"      # Broker has position, no local record (manual broker-side trade)
    CLEAN          = "CLEAN"           # Local and broker agree

    # Expose all names for iteration / validation
    _ALL = ("MISSED_FILL", "POSITION_DRIFT", "QTY_MISMATCH",
            "PRICE_MISMATCH", "BROKER_EXIT", "REMOTE_NEW", "CLEAN")


@dataclass
class ReconcilerConfig:
    """Tuning knobs for reconciler behaviour.

    Attributes:
        poll_interval_seconds:    How often to run a full cycle (default 30s).
        price_tolerance_pct:      Max acceptable avg-price deviation in % (default 0.5%).
        auto_insert_missed_fills: When True, auto-create fill rows for MISSED_FILL events.
        auto_update_position_qty: When True, auto-adjust position qty for QTY_MISMATCH.
                                  Default False — requires manual review.
        halt_on_remote_new:       When True, sets an internal flag that prevents new
                                  orders being dispatched until the anomaly is reviewed.
    """
    poll_interval_seconds: int   = 30
    price_tolerance_pct: float   = 0.5
    auto_insert_missed_fills: bool  = True
    auto_update_position_qty: bool  = False
    halt_on_remote_new: bool        = True


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PositionReconciler:
    """Compares local position table against live broker positions.

    Args:
        broker:     Any broker object that exposes ``get_positions() -> list[Position]``.
                    In production this is PaperBroker or UpstoxLiveBroker.
                    In tests this is a MagicMock.
        db_engine:  SQLAlchemy engine (sync). If None, the reconciler will
                    construct one from ``LOCAL_DB_URL`` env var.
        config:     Optional tuning configuration.
    """

    def __init__(
        self,
        broker: Any,
        db_engine: Any = None,
        config: Optional[ReconcilerConfig] = None,
    ) -> None:
        self.broker = broker
        self.config = config or ReconcilerConfig()
        self._halt_new_orders: bool = False  # set True on REMOTE_NEW when halt_on_remote_new

        if db_engine is not None:
            self._engine = db_engine
        else:
            db_url = os.environ.get(
                "LOCAL_DB_URL",
                "sqlite:///data/trading_bot.db",
            )
            try:
                from sqlalchemy import create_engine as _ce
                if db_url.startswith("postgresql"):
                    # Ensure SQLAlchemy driver suffix for PostgreSQL
                    if "+psycopg" not in db_url:
                        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1).replace(
                            "postgres://", "postgresql+psycopg://", 1
                        )
                    self._engine = _ce(db_url, pool_pre_ping=True)
                else:
                    self._engine = _ce(db_url)
            except Exception as exc:  # pragma: no cover
                logger.warning("PositionReconciler: could not create DB engine — %s", exc)
                self._engine = None

        self._ensure_reconciler_tables()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def halt_new_orders(self) -> bool:
        """True when a REMOTE_NEW event was detected and halt_on_remote_new is set."""
        return self._halt_new_orders

    # ------------------------------------------------------------------
    # Private setup
    # ------------------------------------------------------------------

    def _ensure_reconciler_tables(self) -> None:
        """Create reconciliation tracking tables if they don't exist (SQLite-compatible DDL)."""
        if self._engine is None:
            return
        from sqlalchemy import text
        ddl = [
            """CREATE TABLE IF NOT EXISTS reconciliation_runs (
                cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ended_at TEXT,
                total_local_positions INTEGER NOT NULL DEFAULT 0,
                total_broker_positions INTEGER NOT NULL DEFAULT 0,
                events_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'RUNNING'
            )""",
            """CREATE TABLE IF NOT EXISTS reconciliation_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                cycle_id INTEGER,
                event_type TEXT,
                instrument_key TEXT,
                local_state TEXT,
                broker_state TEXT,
                action_taken TEXT,
                details TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT,
                instrument_key TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price INTEGER NOT NULL,
                filled_at TEXT NOT NULL
            )""",
        ]
        try:
            with self._engine.begin() as conn:
                for stmt in ddl:
                    conn.execute(text(stmt))
        except Exception as exc:
            logger.warning("reconciler: could not create tracking tables: %s", exc)

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    def fetch_local_positions(self) -> list[dict]:
        """Read all OPEN/PARTIAL positions from the local ``positions`` table.

        Returns:
            List of dicts with keys: instrument_key, side, entry_qty, exit_qty,
            entry_avg_price, exit_avg_price, status, id.
        """
        if self._engine is None:
            return []
        try:
            from sqlalchemy import text
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT id, instrument_key, side, "
                        "quantity AS entry_qty, "
                        "0 AS exit_qty, "
                        "entry_price AS entry_avg_price, "
                        "0 AS exit_avg_price, "
                        "status, "
                        "0 AS realized_pnl "
                        "FROM positions "
                        "WHERE status IN ('open','partial') "
                        "ORDER BY opened_at"
                    )
                ).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as exc:
            logger.error("fetch_local_positions failed: %s", exc)
            return []

    def fetch_broker_positions(self) -> list[dict]:
        """Call broker.get_positions() and normalise to plain dicts.

        The broker returns ``Position`` Pydantic objects.  We normalise them
        so the diff logic only deals with plain dicts regardless of broker
        implementation.

        Returns:
            List of dicts with keys: instrument_key, quantity, average_price, side.
        """
        # Let exceptions propagate — reconciler must fail loud if broker is
        # unreachable rather than silently proceeding with stale local data.
        raw: list[Any] = self.broker.get_positions()

        normalised: list[dict] = []
        for pos in raw:
            if isinstance(pos, dict):
                normalised.append(pos)
            else:
                # Pydantic / dataclass
                d = pos.dict() if hasattr(pos, "dict") else vars(pos)
                normalised.append(d)
        return normalised

    # ------------------------------------------------------------------
    # Diff logic
    # ------------------------------------------------------------------

    def diff(self, local: list[dict], broker: list[dict]) -> list[dict]:
        """Compare local and broker positions and produce a list of events.

        Matching key: ``instrument_key``.

        Rules applied per instrument:
          - In local but NOT in broker  → BROKER_EXIT
          - In broker but NOT in local  → REMOTE_NEW
          - In both with matching qty   → check price → PRICE_MISMATCH or CLEAN
          - In both with different qty  → QTY_MISMATCH (and possibly MISSED_FILL
            when broker qty > local net qty)

        Args:
            local:  Output of ``fetch_local_positions()``.
            broker: Output of ``fetch_broker_positions()``.

        Returns:
            List of event dicts. Each dict contains:
              event_type, instrument_key, local_state, broker_state, details.
        """
        events: list[dict] = []

        # Index by instrument_key
        local_by_key: dict[str, dict] = {p["instrument_key"]: p for p in local}
        broker_by_key: dict[str, dict] = {}
        for bp in broker:
            key = bp.get("instrument_key", "")
            if key:
                broker_by_key[key] = bp

        # --- instruments in local only ---
        for key, lp in local_by_key.items():
            if key not in broker_by_key:
                events.append(
                    {
                        "event_type": "BROKER_EXIT",
                        "instrument_key": key,
                        "local_state": lp,
                        "broker_state": None,
                        "details": {
                            "reason": "Position present locally but absent from broker (possible external close)"
                        },
                    }
                )

        # --- instruments in broker only ---
        for key, bp in broker_by_key.items():
            if key not in local_by_key:
                events.append(
                    {
                        "event_type": "REMOTE_NEW",
                        "instrument_key": key,
                        "local_state": None,
                        "broker_state": bp,
                        "details": {
                            "reason": "Position present at broker but absent locally (possible manual trade)"
                        },
                    }
                )

        # --- instruments in both ---
        for key in set(local_by_key) & set(broker_by_key):
            lp = local_by_key[key]
            bp = broker_by_key[key]

            local_net_qty: int = lp.get("entry_qty", 0) - lp.get("exit_qty", 0)
            broker_qty: int = abs(int(bp.get("quantity", 0)))
            local_avg_price: int = int(lp.get("entry_avg_price", 0))
            broker_avg_price: int = int(bp.get("average_price", 0))

            qty_match = local_net_qty == broker_qty

            if not qty_match:
                if broker_qty > local_net_qty:
                    # Broker has more shares → we missed some fills
                    events.append(
                        {
                            "event_type": "MISSED_FILL",
                            "instrument_key": key,
                            "local_state": lp,
                            "broker_state": bp,
                            "details": {
                                "local_net_qty": local_net_qty,
                                "broker_qty": broker_qty,
                                "delta_qty": broker_qty - local_net_qty,
                            },
                        }
                    )
                else:
                    # Local has more than broker → quantity mismatch (partial close?)
                    events.append(
                        {
                            "event_type": "QTY_MISMATCH",
                            "instrument_key": key,
                            "local_state": lp,
                            "broker_state": bp,
                            "details": {
                                "local_net_qty": local_net_qty,
                                "broker_qty": broker_qty,
                                "delta_qty": local_net_qty - broker_qty,
                            },
                        }
                    )
            else:
                # Qty matches — check average price
                if broker_avg_price > 0 and local_avg_price > 0:
                    price_drift_pct = abs(broker_avg_price - local_avg_price) / local_avg_price * 100
                    if price_drift_pct > self.config.price_tolerance_pct:
                        events.append(
                            {
                                "event_type": "PRICE_MISMATCH",
                                "instrument_key": key,
                                "local_state": lp,
                                "broker_state": bp,
                                "details": {
                                    "local_avg_price_paisa": local_avg_price,
                                    "broker_avg_price_paisa": broker_avg_price,
                                    "drift_pct": round(price_drift_pct, 4),
                                    "tolerance_pct": self.config.price_tolerance_pct,
                                },
                            }
                        )
                        continue

                # Everything agrees
                events.append(
                    {
                        "event_type": "CLEAN",
                        "instrument_key": key,
                        "local_state": lp,
                        "broker_state": bp,
                        "details": {"qty": local_net_qty, "avg_price_paisa": local_avg_price},
                    }
                )

        return events

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_actions(self, events: list[dict], cycle_id: int) -> int:
        """Persist events to reconciliation_log and auto-repair where configured.

        Args:
            events:   Output of ``diff()``.
            cycle_id: The ``reconciliation_runs.cycle_id`` for this cycle.

        Returns:
            Count of actions actually taken (not just logged).
        """
        actions_taken = 0
        now_ist = datetime.now(IST)

        for event in events:
            etype = event["event_type"]
            key   = event["instrument_key"]
            lp    = event.get("local_state")
            bp    = event.get("broker_state")
            details = event.get("details", {})

            action_taken = "NONE"

            # ---- MISSED_FILL ----
            if etype == "MISSED_FILL":
                if self.config.auto_insert_missed_fills and self._engine is not None:
                    try:
                        self._insert_missed_fill(lp, bp, details, now_ist)
                        action_taken = "INSERTED_FILL"
                        actions_taken += 1
                        logger.info(
                            "RECONCILER [MISSED_FILL] auto-inserted fill for %s "
                            "(delta_qty=%s)",
                            key, details.get("delta_qty"),
                        )
                    except Exception as exc:
                        logger.error("RECONCILER [MISSED_FILL] insert failed for %s: %s", key, exc)
                        action_taken = "FLAGGED"
                else:
                    action_taken = "FLAGGED"
                    logger.warning(
                        "RECONCILER [MISSED_FILL] %s — delta_qty=%s (auto-insert disabled)",
                        key, details.get("delta_qty"),
                    )

            # ---- QTY_MISMATCH ----
            elif etype == "QTY_MISMATCH":
                if self.config.auto_update_position_qty and self._engine is not None:
                    try:
                        self._update_position_qty(lp, bp, details, now_ist)
                        action_taken = "UPDATED_POSITION"
                        actions_taken += 1
                        logger.warning(
                            "RECONCILER [QTY_MISMATCH] auto-updated qty for %s", key
                        )
                    except Exception as exc:
                        logger.error("RECONCILER [QTY_MISMATCH] update failed for %s: %s", key, exc)
                        action_taken = "FLAGGED"
                else:
                    action_taken = "FLAGGED"
                    logger.warning(
                        "RECONCILER [QTY_MISMATCH] %s local=%s broker=%s — flagged for review",
                        key, details.get("local_net_qty"), details.get("broker_qty"),
                    )

            # ---- PRICE_MISMATCH ----
            elif etype == "PRICE_MISMATCH":
                action_taken = "FLAGGED"
                logger.warning(
                    "RECONCILER [PRICE_MISMATCH] %s drift=%.2f%% (tol=%.2f%%)",
                    key, details.get("drift_pct", 0), self.config.price_tolerance_pct,
                )

            # ---- BROKER_EXIT ----
            elif etype == "BROKER_EXIT":
                if self._engine is not None and lp is not None:
                    try:
                        self._close_local_position(lp, bp, now_ist)
                        action_taken = "UPDATED_POSITION"
                        actions_taken += 1
                        logger.info(
                            "RECONCILER [BROKER_EXIT] closed local position for %s", key
                        )
                    except Exception as exc:
                        logger.error("RECONCILER [BROKER_EXIT] close failed for %s: %s", key, exc)
                        action_taken = "FLAGGED"
                else:
                    action_taken = "FLAGGED"

            # ---- REMOTE_NEW ----
            elif etype == "REMOTE_NEW":
                action_taken = "FLAGGED"
                if self.config.halt_on_remote_new:
                    self._halt_new_orders = True
                logger.critical(
                    "RECONCILER [REMOTE_NEW] %s — broker has a position with no local record! "
                    "Possible manual trade. halt_new_orders=%s",
                    key, self._halt_new_orders,
                )

            # ---- CLEAN ----
            elif etype == "CLEAN":
                action_taken = "NONE"
                logger.debug("RECONCILER [CLEAN] %s", key)

            # Persist to reconciliation_log
            if self._engine is not None:
                try:
                    self._log_event(
                        cycle_id=cycle_id,
                        event_type=etype,
                        instrument_key=key,
                        local_state=lp,
                        broker_state=bp,
                        action_taken=action_taken,
                        details=details,
                    )
                except Exception as exc:
                    logger.error("RECONCILER log_event failed for %s/%s: %s", etype, key, exc)

        return actions_taken

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict:
        """Execute a complete reconciliation cycle.

        Steps:
          1. Insert a ``reconciliation_runs`` row (status=RUNNING).
          2. Fetch local + broker positions.
          3. Diff them.
          4. Apply actions.
          5. Update run row to COMPLETE (or FAILED).

        Returns:
            Dict with keys: cycle_id, events_count, status, actions_taken.
        """
        cycle_id: Optional[int] = None
        status = "FAILED"
        events_count = 0
        actions_taken = 0

        try:
            local = self.fetch_local_positions()
            # Create the run row BEFORE fetching broker so we always have a
            # cycle_id even if the broker call fails. broker count starts at 0
            # and is updated implicitly via the events_count tally.
            cycle_id = self._create_run(len(local), 0)
            broker = self.fetch_broker_positions()

            events = self.diff(local, broker)
            events_count = len(events)
            actions_taken = self.apply_actions(events, cycle_id)

            status = "COMPLETE"
            logger.info(
                "RECONCILER cycle %s complete — local=%d broker=%d events=%d actions=%d",
                cycle_id, len(local), len(broker), events_count, actions_taken,
            )
        except Exception as exc:
            logger.error("RECONCILER cycle failed: %s", exc)
            status = "FAILED"
        finally:
            if cycle_id is not None:
                self._finish_run(cycle_id, events_count, status)

        return {
            "cycle_id": cycle_id,
            "events_count": events_count,
            "status": status,
            "actions_taken": actions_taken,
        }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_recent_events(
        self, event_type: Optional[str] = None, hours: int = 24
    ) -> list[dict]:
        """Fetch recent events from reconciliation_log.

        Args:
            event_type: Optional filter (e.g. 'MISSED_FILL'). None returns all types.
            hours:      Look-back window in hours (default 24).

        Returns:
            List of event dicts ordered by ts DESC.
        """
        if self._engine is None:
            return []
        try:
            from sqlalchemy import text

            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            if event_type:
                query = text(
                    "SELECT * FROM reconciliation_log "
                    "WHERE event_type = :etype AND ts >= :since "
                    "ORDER BY ts DESC"
                )
                params: dict = {"etype": event_type, "since": since}
            else:
                query = text(
                    "SELECT * FROM reconciliation_log "
                    "WHERE ts >= :since "
                    "ORDER BY ts DESC"
                )
                params = {"since": since}

            with self._engine.connect() as conn:
                rows = conn.execute(query, params).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as exc:
            logger.error("get_recent_events failed: %s", exc)
            return []

    def get_drift_summary(self, days: int = 7) -> dict:
        """Aggregate reconciliation stats over the last N days.

        Returns:
            Dict with keys: total_cycles, total_events, events_by_type,
            auto_fixed, flagged, last_cycle_id, last_cycle_status.
        """
        if self._engine is None:
            return {}
        try:
            from sqlalchemy import text

            since = datetime.now(timezone.utc) - timedelta(days=days)
            with self._engine.connect() as conn:
                # Run summary
                run_row = conn.execute(
                    text(
                        "SELECT COUNT(*) AS total_cycles, "
                        "MAX(cycle_id) AS last_cycle_id "
                        "FROM reconciliation_runs WHERE started_at >= :since"
                    ),
                    {"since": since},
                ).fetchone()

                # Event breakdown
                event_rows = conn.execute(
                    text(
                        "SELECT event_type, COUNT(*) AS cnt, "
                        "SUM(CASE WHEN action_taken IN ('INSERTED_FILL','UPDATED_POSITION') THEN 1 ELSE 0 END) AS auto_fixed, "
                        "SUM(CASE WHEN action_taken = 'FLAGGED' THEN 1 ELSE 0 END) AS flagged "
                        "FROM reconciliation_log WHERE ts >= :since "
                        "GROUP BY event_type"
                    ),
                    {"since": since},
                ).fetchall()

                last_run = None
                if run_row and run_row.last_cycle_id:
                    last_run = conn.execute(
                        text("SELECT status FROM reconciliation_runs WHERE cycle_id = :cid"),
                        {"cid": run_row.last_cycle_id},
                    ).fetchone()

            events_by_type: dict[str, int] = {}
            total_auto_fixed = 0
            total_flagged    = 0
            total_events     = 0
            for er in event_rows:
                em = dict(er._mapping)
                etype = em["event_type"]
                cnt   = int(em["cnt"])
                events_by_type[etype] = cnt
                total_events += cnt
                total_auto_fixed += int(em.get("auto_fixed") or 0)
                total_flagged    += int(em.get("flagged") or 0)

            return {
                "total_cycles": int(run_row.total_cycles) if run_row else 0,
                "total_events": total_events,
                "events_by_type": events_by_type,
                "auto_fixed": total_auto_fixed,
                "flagged": total_flagged,
                "last_cycle_id": int(run_row.last_cycle_id) if run_row and run_row.last_cycle_id else None,
                "last_cycle_status": last_run.status if last_run else None,
            }
        except Exception as exc:
            logger.error("get_drift_summary failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Private DB helpers
    # ------------------------------------------------------------------

    def _create_run(self, n_local: int, n_broker: int) -> int:
        """Insert a new reconciliation_runs row and return its cycle_id."""
        if self._engine is None:
            return 0
        from sqlalchemy import text
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(
                        "INSERT INTO reconciliation_runs "
                        "(started_at, total_local_positions, total_broker_positions, status) "
                        "VALUES (CURRENT_TIMESTAMP, :nl, :nb, 'RUNNING')"
                    ),
                    {"nl": n_local, "nb": n_broker},
                )
                return int(result.lastrowid)
        except Exception as exc:
            logger.error("_create_run failed: %s", exc)
            return 0

    def _finish_run(self, cycle_id: int, events_count: int, status: str) -> None:
        """Update reconciliation_runs on cycle completion."""
        from sqlalchemy import text
        if self._engine is None:
            return
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE reconciliation_runs "
                        "SET ended_at = CURRENT_TIMESTAMP, "
                        "    events_count = :ec, "
                        "    status = :st "
                        "WHERE cycle_id = :cid"
                    ),
                    {"ec": events_count, "st": status, "cid": cycle_id},
                )
        except Exception as exc:
            logger.error("_finish_run failed: %s", exc)

    def _log_event(
        self,
        *,
        cycle_id: int,
        event_type: str,
        instrument_key: str,
        local_state: Any,
        broker_state: Any,
        action_taken: str,
        details: Any,
    ) -> None:
        """Insert one row into reconciliation_log."""
        from sqlalchemy import text

        def _to_json(obj: Any) -> Optional[str]:
            if obj is None:
                return None
            if isinstance(obj, str):
                return obj
            try:
                return json.dumps(obj, default=str)
            except Exception:
                return str(obj)

        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO reconciliation_log "
                        "(ts, cycle_id, event_type, instrument_key, "
                        " local_state, broker_state, action_taken, details) "
                        "VALUES (CURRENT_TIMESTAMP, :cid, :etype, :ikey, "
                        "        :ls, :bs, :at, :det)"
                    ),
                    {
                        "cid":   cycle_id,
                        "etype": event_type,
                        "ikey":  instrument_key,
                        "ls":    _to_json(local_state),
                        "bs":    _to_json(broker_state),
                        "at":    action_taken,
                        "det":   _to_json(details),
                    },
                )
        except Exception as exc:
            logger.error("_log_event failed: %s", exc)

    def _insert_missed_fill(
        self, lp: dict, bp: dict, details: dict, now_ist: datetime
    ) -> None:
        """Insert a synthetic fill row and bump the position entry_qty."""
        from sqlalchemy import text

        pos_id = lp.get("id")
        delta_qty = int(details.get("delta_qty", 0))
        broker_avg_price = int(bp.get("average_price", 0))
        instrument_key = lp.get("instrument_key", "")
        side = lp.get("side", "BUY")

        # Build a synthetic fill_id
        fill_id = f"RECON_{now_ist.strftime('%Y%m%d%H%M%S')}_{instrument_key.replace('|','_')}"

        # We need a corresponding order_id — use None (nullable FK)
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT OR IGNORE INTO fills (fill_id, order_id, instrument_key, side, "
                    "quantity, price, filled_at) "
                    "VALUES (:fid, NULL, :ikey, :side, :qty, :price, :ts)"
                ),
                {
                    "fid":   fill_id,
                    "ikey":  instrument_key,
                    "side":  side,
                    "qty":   delta_qty,
                    "price": broker_avg_price,
                    "ts":    now_ist,
                },
            )
            if pos_id is not None:
                # Recalculate weighted average price
                local_qty  = int(lp.get("entry_qty", 0))
                local_price = int(lp.get("entry_avg_price", 0))
                new_total_qty = local_qty + delta_qty
                new_avg = (
                    (local_qty * local_price + delta_qty * broker_avg_price) // new_total_qty
                    if new_total_qty > 0 else local_price
                )
                conn.execute(
                    text(
                        "UPDATE positions "
                        "SET quantity = :newqty, entry_price = :newprice "
                        "WHERE id = :pid"
                    ),
                    {"newqty": new_total_qty, "newprice": new_avg, "pid": pos_id},
                )

    def _update_position_qty(
        self, lp: dict, bp: dict, details: dict, now_ist: datetime
    ) -> None:
        """Force-update local position qty to match broker (for QTY_MISMATCH)."""
        from sqlalchemy import text

        pos_id = lp.get("id")
        if pos_id is None:
            return
        broker_qty = int(details.get("broker_qty", 0))
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE positions SET quantity = :qty WHERE id = :pid"
                ),
                {"qty": broker_qty, "pid": pos_id},
            )

    def _close_local_position(
        self, lp: dict, bp: Optional[dict], now_ist: datetime
    ) -> None:
        """Mark a local position CLOSED after a broker-side exit."""
        from sqlalchemy import text

        pos_id = lp.get("id")
        if pos_id is None:
            return

        # Use broker last_price as exit price if available, else zero
        exit_price = 0
        if bp and "last_price" in bp:
            exit_price = int(bp.get("last_price") or 0)
        elif bp and "average_price" in bp:
            exit_price = int(bp.get("average_price") or 0)

        entry_qty   = int(lp.get("entry_qty", 0))
        entry_price = int(lp.get("entry_avg_price", 0))
        side        = lp.get("side", "BUY")

        if side == "BUY":
            realized_pnl = (exit_price - entry_price) * entry_qty
        else:
            realized_pnl = (entry_price - exit_price) * entry_qty

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE positions "
                    "SET status = 'CLOSED', "
                    "    closed_at = :ts "
                    "WHERE id = :pid"
                ),
                {
                    "ts":  now_ist,
                    "pid": pos_id,
                },
            )
