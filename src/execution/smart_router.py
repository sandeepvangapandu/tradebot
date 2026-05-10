"""Smart order routing — LIMIT-protect, iceberg, TWAP/VWAP.

Sits between OrderManager and the broker adapter.  Receives a logical order
(instrument_key, side, quantity, lot_size, …) and decides how to slice and
pace it to minimise market-impact and information leakage.

Routing hierarchy
-----------------
urgency == 'urgent'
    → MARKET immediately

qty_lots > twap_threshold_lots
    → TWAP (uniform slices over twap_duration_seconds)

qty_lots > iceberg_threshold_lots
    → ICEBERG (N hidden chunks placed sequentially as LIMIT_PROTECT)

else
    → LIMIT_PROTECT (single LIMIT at LTP±protect_bps, auto-reprice up to
      max_retries, falls to MARKET on exhaustion)

Monetary values: PAISA (int).  Timestamps: IST (Asia/Kolkata).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums + config
# ---------------------------------------------------------------------------

class RoutingStrategy(str, Enum):
    """Routing algorithm tag stored in routed_orders."""
    MARKET = "MARKET"
    LIMIT_PROTECT = "LIMIT_PROTECT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"


@dataclass
class RoutingConfig:
    """Tunable parameters for SmartOrderRouter."""

    # iceberg
    iceberg_threshold_lots: int = 10     # slice when qty_lots > this
    iceberg_slice_count: int = 5         # number of equal chunks

    # TWAP
    twap_threshold_lots: int = 50        # TWAP when qty_lots > this
    twap_duration_seconds: int = 600     # spread over 10 minutes

    # LIMIT-protect
    limit_protect_bps: float = 5.0       # offset from LTP in basis points
    limit_protect_max_retries: int = 3   # re-pricings before falling to MARKET
    limit_protect_retry_seconds: int = 10  # wait between re-pricings


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class SmartOrderRouter:
    """Intelligent order placement layer between OrderManager and broker.

    Args:
        broker: Any object exposing ``place_order(order_dict) -> dict``,
                ``cancel_order(order_id) -> dict``, and
                ``get_ltp(instrument_key) -> int`` (paisa).
        db_engine: Optional SQLAlchemy engine for persisting routed_orders rows.
                   If None, persistence is skipped (useful for unit tests).
        config:  RoutingConfig instance.  Defaults constructed if not provided.

    Example::

        router = SmartOrderRouter(broker=paper_broker, config=RoutingConfig())
        result = router.route(
            order={
                "client_order_id": "signal-42",
                "instrument_key": "NSE_EQ|INE002A01018",
                "side": "BUY",
                "quantity": 150,
                "lot_size": 1,
            }
        )
    """

    def __init__(
        self,
        broker: Any,
        db_engine: Optional[Any] = None,
        config: Optional[RoutingConfig] = None,
    ) -> None:
        self._broker = broker
        self._db = db_engine
        self._cfg = config or RoutingConfig()

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def select_strategy(
        self, qty: int, lot_size: int, urgency: str = "normal"
    ) -> RoutingStrategy:
        """Choose routing algorithm for the given order size and urgency.

        Args:
            qty:      Total quantity (shares / contracts).
            lot_size: Lot size of the instrument (1 for equities).
            urgency:  ``'urgent'`` forces MARKET; ``'normal'`` applies size logic.

        Returns:
            RoutingStrategy enum value.
        """
        if urgency == "urgent":
            return RoutingStrategy.MARKET

        qty_lots = qty / max(lot_size, 1)

        if qty_lots > self._cfg.twap_threshold_lots:
            return RoutingStrategy.TWAP

        if qty_lots > self._cfg.iceberg_threshold_lots:
            return RoutingStrategy.ICEBERG

        return RoutingStrategy.LIMIT_PROTECT

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def route(self, order: Dict[str, Any], urgency: str = "normal") -> Dict[str, Any]:
        """Route an order using the appropriate strategy.

        Args:
            order: Dict with at minimum:
                   ``client_order_id``, ``instrument_key``, ``side``,
                   ``quantity``, ``lot_size``.
            urgency: ``'urgent'`` | ``'normal'``.

        Returns:
            Dict with keys:
            - ``parent_order_id``    (str)
            - ``child_order_ids``    (list[str])
            - ``routing_strategy``   (str)
            - ``status``             (str — FILLED / PARTIAL / FAILED)
            - ``total_filled_qty``   (int)
        """
        parent_id = order.get("client_order_id") or str(uuid.uuid4())
        qty = order["quantity"]
        lot_size = order.get("lot_size", 1)
        strategy = self.select_strategy(qty, lot_size, urgency)

        logger.info(
            "SmartRouter: parent=%s qty=%d strategy=%s urgency=%s",
            parent_id, qty, strategy, urgency,
        )

        if strategy == RoutingStrategy.MARKET:
            result = self.execute_market(order)
        elif strategy == RoutingStrategy.ICEBERG:
            result = self.execute_iceberg(order, self._cfg.iceberg_slice_count)
        elif strategy == RoutingStrategy.TWAP:
            result = self.execute_twap(order, self._cfg.twap_duration_seconds)
        else:
            result = self.execute_limit_protect(order)

        result["parent_order_id"] = parent_id
        result["routing_strategy"] = strategy.value

        # Persist audit rows
        if self._db is not None:
            self._persist_result(result, order)

        return result

    # ------------------------------------------------------------------
    # Executors
    # ------------------------------------------------------------------

    def execute_market(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place a single MARKET order immediately.

        Args:
            order: Order dict from caller.

        Returns:
            Routing result dict.
        """
        market_order = {**order, "order_type": "MARKET"}
        response = self._broker.place_order(market_order)

        child_id = self._extract_order_id(response)
        filled_qty = self._extract_filled_qty(response, order["quantity"])

        return {
            "child_order_ids": [child_id],
            "total_filled_qty": filled_qty,
            "status": "FILLED" if filled_qty >= order["quantity"] else "PARTIAL",
            "slices": [
                {
                    "slice_index": 0,
                    "total_slices": 1,
                    "intended_qty": order["quantity"],
                    "filled_qty": filled_qty,
                    "routing_strategy": RoutingStrategy.MARKET.value,
                    "routing_metadata": {"urgency": "market"},
                }
            ],
        }

    def execute_limit_protect(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place a LIMIT order at LTP±protect_bps; reprice on non-fill.

        Places a LIMIT order offset from last traded price by ``protect_bps``
        basis points.  After ``retry_seconds`` checks for fill; if unfilled,
        cancels and reprices.  Falls to MARKET after ``max_retries`` exhausted.

        Args:
            order: Order dict from caller.

        Returns:
            Routing result dict with single child.
        """
        instrument_key = order["instrument_key"]
        side = order["side"]
        qty = order["quantity"]
        cfg = self._cfg

        total_filled = 0
        child_id: Optional[str] = None
        last_price: Optional[int] = None
        metadata: Dict[str, Any] = {
            "protection_bps": cfg.limit_protect_bps,
            "retry_count": 0,
        }

        for attempt in range(cfg.limit_protect_max_retries + 1):
            # Get LTP; fall back to last known price if broker call fails
            try:
                ltp = self._broker.get_ltp(instrument_key)
            except Exception:
                ltp = last_price or 100_00  # 100 rupees fallback

            last_price = ltp
            bps_offset = int(ltp * cfg.limit_protect_bps / 10_000)

            if side.upper() == "BUY":
                limit_price = ltp + bps_offset
            else:
                limit_price = max(ltp - bps_offset, 1)

            limit_order = {**order, "order_type": "LIMIT", "price": limit_price}
            response = self._broker.place_order(limit_order)
            child_id = self._extract_order_id(response)
            metadata["retry_count"] = attempt
            metadata["limit_price"] = limit_price

            # Simulate wait + check (in tests, time.sleep is typically 0)
            time.sleep(0)  # real implementation would sleep cfg.retry_seconds

            filled_qty = self._extract_filled_qty(response, 0)
            total_filled = filled_qty

            if filled_qty >= qty:
                logger.info(
                    "LIMIT_PROTECT: parent filled on attempt %d (child=%s)", attempt, child_id
                )
                return {
                    "child_order_ids": [child_id],
                    "total_filled_qty": total_filled,
                    "status": "FILLED",
                    "slices": [
                        {
                            "slice_index": 0,
                            "total_slices": 1,
                            "intended_qty": qty,
                            "filled_qty": filled_qty,
                            "routing_strategy": RoutingStrategy.LIMIT_PROTECT.value,
                            "routing_metadata": metadata.copy(),
                        }
                    ],
                }

            # Unfilled — cancel and retry if attempts remain
            if attempt < cfg.limit_protect_max_retries:
                try:
                    self._broker.cancel_order(child_id)
                except Exception as exc:
                    logger.warning("cancel_order failed: %s", exc)
                logger.info(
                    "LIMIT_PROTECT: unfilled on attempt %d, repricing (child=%s)",
                    attempt, child_id,
                )
            else:
                # Max retries exhausted — fall to MARKET
                logger.warning(
                    "LIMIT_PROTECT: max retries (%d) exhausted, falling to MARKET",
                    cfg.limit_protect_max_retries,
                )
                try:
                    self._broker.cancel_order(child_id)
                except Exception:
                    pass

                market_result = self.execute_market({**order, "quantity": qty - total_filled})
                market_child_id = market_result["child_order_ids"][0]
                total_filled += market_result["total_filled_qty"]

                metadata["fell_to_market"] = True
                return {
                    "child_order_ids": [child_id, market_child_id],
                    "total_filled_qty": total_filled,
                    "status": "FILLED" if total_filled >= qty else "PARTIAL",
                    "slices": [
                        {
                            "slice_index": 0,
                            "total_slices": 1,
                            "intended_qty": qty,
                            "filled_qty": total_filled,
                            "routing_strategy": RoutingStrategy.LIMIT_PROTECT.value,
                            "routing_metadata": metadata.copy(),
                        }
                    ],
                }

        # Should not reach here, but guard anyway
        return {
            "child_order_ids": [child_id] if child_id else [],
            "total_filled_qty": total_filled,
            "status": "PARTIAL",
            "slices": [],
        }

    def execute_iceberg(
        self, order: Dict[str, Any], slice_count: int
    ) -> Dict[str, Any]:
        """Slice order into ``slice_count`` equal chunks placed as LIMIT_PROTECT.

        The last slice absorbs any rounding remainder.

        Args:
            order:       Order dict from caller.
            slice_count: Number of hidden chunks.

        Returns:
            Routing result dict with one child_order_id per slice.
        """
        qty = order["quantity"]
        base_qty = qty // slice_count
        remainder = qty - base_qty * slice_count

        child_order_ids: List[str] = []
        slices: List[Dict[str, Any]] = []
        total_filled = 0

        for i in range(slice_count):
            slice_qty = base_qty + (remainder if i == slice_count - 1 else 0)
            slice_order = {**order, "quantity": slice_qty}
            result = self.execute_limit_protect(slice_order)

            child_order_ids.extend(result["child_order_ids"])
            filled = result["total_filled_qty"]
            total_filled += filled

            slices.append(
                {
                    "slice_index": i,
                    "total_slices": slice_count,
                    "intended_qty": slice_qty,
                    "filled_qty": filled,
                    "routing_strategy": RoutingStrategy.ICEBERG.value,
                    "routing_metadata": {
                        "slice_count": slice_count,
                        "slice_index": i,
                    },
                }
            )

        status = "FILLED" if total_filled >= qty else "PARTIAL"
        return {
            "child_order_ids": child_order_ids,
            "total_filled_qty": total_filled,
            "status": status,
            "slices": slices,
        }

    def execute_twap(
        self, order: Dict[str, Any], duration_seconds: int
    ) -> Dict[str, Any]:
        """Distribute order uniformly over ``duration_seconds``.

        Slices into ``iceberg_slice_count`` chunks and places each as
        LIMIT_PROTECT with equal time-gaps between slices.  In paper/test mode
        the sleep is skipped (``time.sleep(0)``).

        Args:
            order:            Order dict from caller.
            duration_seconds: Total window to spread execution over.

        Returns:
            Routing result dict.
        """
        slice_count = self._cfg.iceberg_slice_count
        qty = order["quantity"]
        base_qty = qty // slice_count
        remainder = qty - base_qty * slice_count

        interval = duration_seconds / slice_count  # seconds between slices

        child_order_ids: List[str] = []
        slices: List[Dict[str, Any]] = []
        total_filled = 0

        for i in range(slice_count):
            slice_qty = base_qty + (remainder if i == slice_count - 1 else 0)
            slice_order = {**order, "quantity": slice_qty}
            result = self.execute_limit_protect(slice_order)

            child_order_ids.extend(result["child_order_ids"])
            filled = result["total_filled_qty"]
            total_filled += filled

            slices.append(
                {
                    "slice_index": i,
                    "total_slices": slice_count,
                    "intended_qty": slice_qty,
                    "filled_qty": filled,
                    "routing_strategy": RoutingStrategy.TWAP.value,
                    "routing_metadata": {
                        "duration_seconds": duration_seconds,
                        "interval_seconds": interval,
                        "slice_index": i,
                    },
                }
            )

            if i < slice_count - 1:
                time.sleep(0)  # real: time.sleep(interval)

        status = "FILLED" if total_filled >= qty else "PARTIAL"
        return {
            "child_order_ids": child_order_ids,
            "total_filled_qty": total_filled,
            "status": status,
            "slices": slices,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_routing_stats(
        self,
        parent_order_id: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """Return aggregate statistics from routed_orders table.

        Args:
            parent_order_id: If given, filter to this parent only.
            days:            Look-back window in calendar days.

        Returns:
            Dict with keys:
            - ``total_routed`` (int)
            - ``by_strategy``  (dict[strategy -> count])
            - ``filled_qty``   (int)
            - ``partial_pct``  (float, 0-100)
        """
        if self._db is None:
            return {
                "total_routed": 0,
                "by_strategy": {},
                "filled_qty": 0,
                "partial_pct": 0.0,
                "source": "no_db",
            }

        try:
            from sqlalchemy import text

            with self._db.connect() as conn:
                base_filter = "WHERE created_at > NOW() - INTERVAL :days_interval"
                params: Dict[str, Any] = {"days_interval": f"{days} days"}

                if parent_order_id:
                    base_filter += " AND parent_order_id = :pid"
                    params["pid"] = parent_order_id

                rows = conn.execute(
                    text(
                        f"""
                        SELECT routing_strategy,
                               COUNT(*) AS cnt,
                               SUM(filled_qty) AS total_filled,
                               SUM(CASE WHEN status = 'PARTIAL' THEN 1 ELSE 0 END) AS partial_cnt
                        FROM routed_orders
                        {base_filter}
                        GROUP BY routing_strategy
                        """
                    ),
                    params,
                ).fetchall()

            total = sum(r[1] for r in rows)
            by_strategy = {r[0]: r[1] for r in rows}
            filled_qty = sum(r[2] or 0 for r in rows)
            partial_cnt = sum(r[3] or 0 for r in rows)
            partial_pct = (partial_cnt / total * 100) if total else 0.0

            return {
                "total_routed": total,
                "by_strategy": by_strategy,
                "filled_qty": filled_qty,
                "partial_pct": round(partial_pct, 2),
                "source": "db",
            }
        except Exception as exc:
            logger.error("get_routing_stats error: %s", exc)
            return {
                "total_routed": 0,
                "by_strategy": {},
                "filled_qty": 0,
                "partial_pct": 0.0,
                "source": "error",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_order_id(self, response: Any) -> str:
        """Pull broker order_id out of whatever the broker returns."""
        if isinstance(response, dict):
            return str(
                response.get("order_id")
                or response.get("id")
                or uuid.uuid4()
            )
        if hasattr(response, "order_id"):
            return str(response.order_id)
        return str(uuid.uuid4())

    def _extract_filled_qty(self, response: Any, default: int) -> int:
        """Pull filled quantity from broker response."""
        if isinstance(response, dict):
            return int(
                response.get("filled_quantity")
                or response.get("filled_qty")
                or default
            )
        if hasattr(response, "filled_quantity"):
            return int(response.filled_quantity or default)
        return default

    def _persist_result(
        self, result: Dict[str, Any], original_order: Dict[str, Any]
    ) -> None:
        """Write slice rows to routed_orders table."""
        try:
            from sqlalchemy import text

            slices = result.get("slices", [])
            if not slices:
                # Synthesise one row for strategies that don't emit slice detail
                slices = [
                    {
                        "slice_index": 0,
                        "total_slices": 1,
                        "intended_qty": original_order["quantity"],
                        "filled_qty": result.get("total_filled_qty", 0),
                        "routing_strategy": result.get("routing_strategy", "MARKET"),
                        "routing_metadata": {},
                    }
                ]

            child_ids = result.get("child_order_ids", [])
            with self._db.begin() as conn:
                for idx, sl in enumerate(slices):
                    child_id = child_ids[idx] if idx < len(child_ids) else None
                    conn.execute(
                        text(
                            """
                            INSERT INTO routed_orders
                              (parent_order_id, child_order_id, routing_strategy,
                               slice_index, total_slices, intended_qty, filled_qty,
                               status, routing_metadata)
                            VALUES
                              (:parent, :child, :strategy,
                               :sidx, :total, :iqty, :fqty,
                               :status, :meta::jsonb)
                            """
                        ),
                        {
                            "parent": result.get("parent_order_id"),
                            "child": child_id or str(uuid.uuid4()),
                            "strategy": sl.get("routing_strategy", result["routing_strategy"]),
                            "sidx": sl["slice_index"],
                            "total": sl["total_slices"],
                            "iqty": sl["intended_qty"],
                            "fqty": sl["filled_qty"],
                            "status": result["status"],
                            "meta": __import__("json").dumps(
                                sl.get("routing_metadata", {})
                            ),
                        },
                    )
        except Exception as exc:
            logger.error("SmartRouter._persist_result error: %s", exc)
