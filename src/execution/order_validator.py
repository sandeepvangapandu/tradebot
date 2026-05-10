"""Pre-trade order validation: lot rounding, freeze split, SPAN margin.

This module provides the OrderValidator class which performs pre-flight
validation on orders before they reach the broker.  Three key checks:

1. Lot rounding — quantity must be a multiple of the instrument's lot_size.
2. Freeze split — Upstox rejects single orders > freeze_quantity; we split.
3. SPAN margin — verify sufficient margin is available for all slices.

All monetary values are in PAISA (1 Rupee = 100 paisa).
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ValidationConfig:
    """Configuration for OrderValidator behaviour.

    Attributes:
        margin_buffer_pct: Extra margin cushion required beyond the raw SPAN
            requirement.  Default 5.0 means 105% of required must be available.
        enable_margin_check: If False, skip the margin HTTP call entirely
            (useful for paper-only runs where margin is unlimited).
    """

    margin_buffer_pct: float = 5.0
    enable_margin_check: bool = True


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------


class OrderValidator:
    """Pre-trade validator: lot-round → freeze-split → SPAN margin check.

    Args:
        instrument_manager: Object with ``get_instrument(instrument_key) -> dict``
            that returns at least ``lot_size`` and ``freeze_quantity`` fields.
            May be None; validator falls back to DB query.
        broker: Broker object used to fetch current funds (``get_funds()``).
            May be None; margin check is skipped when both broker and db_engine
            are unavailable.
        db_engine: SQLAlchemy engine for persisting margin_checks rows.
            Optional — if None, persistence is silently skipped.
        config: ValidationConfig instance.  Defaults are used when None.
    """

    def __init__(
        self,
        instrument_manager=None,
        broker=None,
        db_engine=None,
        config: Optional[ValidationConfig] = None,
    ) -> None:
        self._im = instrument_manager
        self._broker = broker
        self._db = db_engine
        self._cfg = config or ValidationConfig()

    # ------------------------------------------------------------------
    # Instrument helpers
    # ------------------------------------------------------------------

    def get_lot_size(self, instrument_key: str) -> int:
        """Return the lot size for *instrument_key*.

        Resolution order:
        1. instrument_manager.get_instrument() if manager is available.
        2. Direct DB query against the ``instruments`` table.
        3. Fallback: 1 (equity — effectively "every share is its own lot").

        Args:
            instrument_key: Upstox instrument key, e.g. ``NSE_FO|NIFTY25APR24000CE``.

        Returns:
            Lot size as a positive integer (≥ 1).
        """
        row = self._fetch_instrument(instrument_key)
        lot = (row or {}).get("lot_size", 1)
        return max(int(lot or 1), 1)

    def get_freeze_quantity(self, instrument_key: str) -> int:
        """Return the freeze quantity for *instrument_key*.

        A freeze quantity of 0 (or NULL in DB) means "no limit" — orders of
        any size are accepted in a single shot.

        Args:
            instrument_key: Upstox instrument key.

        Returns:
            Freeze quantity as non-negative integer.  0 means no limit.
        """
        row = self._fetch_instrument(instrument_key)
        fq = (row or {}).get("freeze_quantity", 0)
        return max(int(fq or 0), 0)

    def _fetch_instrument(self, instrument_key: str) -> Optional[dict]:
        """Resolve instrument row from manager or DB.

        Args:
            instrument_key: Instrument identifier.

        Returns:
            Dict with at least ``lot_size`` and ``freeze_quantity``, or None.
        """
        # Try instrument_manager first
        if self._im is not None:
            try:
                row = self._im.get_instrument(instrument_key)
                if row:
                    return row if isinstance(row, dict) else dict(row)
            except Exception:
                logger.debug(
                    "instrument_manager lookup failed for %s", instrument_key
                )

        # Fall back to direct DB query
        if self._db is not None:
            try:
                from sqlalchemy import text

                with self._db.connect() as conn:
                    result = conn.execute(
                        text(
                            "SELECT lot_size, freeze_quantity FROM instruments "
                            "WHERE instrument_key = :key"
                        ),
                        {"key": instrument_key},
                    )
                    row = result.fetchone()
                    if row:
                        return {"lot_size": row[0], "freeze_quantity": row[1]}
            except Exception:
                logger.debug(
                    "DB lookup failed for instrument %s", instrument_key
                )

        return None

    # ------------------------------------------------------------------
    # Lot rounding
    # ------------------------------------------------------------------

    def round_to_lot(self, qty: int, lot_size: int, mode: str = "down") -> int:
        """Round *qty* to the nearest multiple of *lot_size*.

        Args:
            qty: Requested quantity (may not be a multiple of lot_size).
            lot_size: Lot size for the instrument (must be ≥ 1).
            mode: Rounding mode — one of ``'down'``, ``'up'``, ``'nearest'``.
                ``'down'`` is the conservative default (never over-order).

        Returns:
            Rounded quantity.  May be 0 if qty < lot_size and mode is 'down'.
        """
        lot_size = max(int(lot_size), 1)
        if lot_size == 1:
            return max(qty, 0)

        if mode == "down":
            return (qty // lot_size) * lot_size
        elif mode == "up":
            return ((qty + lot_size - 1) // lot_size) * lot_size
        elif mode == "nearest":
            floor = (qty // lot_size) * lot_size
            ceil_ = floor + lot_size
            if (qty - floor) <= (ceil_ - qty):
                return floor
            return ceil_
        else:
            raise ValueError(f"Unknown rounding mode: {mode!r}")

    # ------------------------------------------------------------------
    # Freeze splitting
    # ------------------------------------------------------------------

    def split_for_freeze(
        self, qty: int, freeze_qty: int, lot_size: int
    ) -> list[int]:
        """Split *qty* into chunks each ≤ *freeze_qty* and a multiple of *lot_size*.

        Upstox (and some other Indian brokers) reject F&O orders whose
        quantity exceeds a per-instrument freeze threshold in a single shot.
        This method slices the total quantity into legal-sized pieces.

        Args:
            qty: Total quantity to split.  Must already be lot-rounded.
            freeze_qty: Maximum quantity per order slice.  0 means no limit —
                returns ``[qty]`` unchanged.
            lot_size: Lot size to use when computing per-slice sizes.

        Returns:
            List of chunk quantities that sum to *qty*.  Each chunk is a
            multiple of *lot_size* and ≤ *freeze_qty*.  Returns ``[qty]``
            when no splitting is required.
        """
        lot_size = max(int(lot_size), 1)
        qty = max(int(qty), 0)
        freeze_qty = max(int(freeze_qty), 0)

        if qty == 0:
            return [0]

        # No freeze limit or already within limit
        if freeze_qty == 0 or qty <= freeze_qty:
            return [qty]

        # Determine max lots per slice (floor to nearest lot)
        max_per_slice = self.round_to_lot(freeze_qty, lot_size, mode="down")
        if max_per_slice == 0:
            # freeze_qty smaller than lot_size — can't slice sensibly; return as-is
            logger.warning(
                "freeze_qty %d < lot_size %d; returning unsplit qty %d",
                freeze_qty,
                lot_size,
                qty,
            )
            return [qty]

        slices: list[int] = []
        remaining = qty
        while remaining > 0:
            chunk = min(remaining, max_per_slice)
            slices.append(chunk)
            remaining -= chunk

        return slices

    # ------------------------------------------------------------------
    # Margin check
    # ------------------------------------------------------------------

    def check_margin(self, order: dict) -> dict:
        """Check whether sufficient SPAN margin exists for *order*.

        In production this would call Upstox ``/v3/charges/margin`` (or the
        v2 fallback).  For now the HTTP call is stubbed to return approved=True
        so the validator can be used without live broker credentials.

        Persists a row to the ``margin_checks`` table regardless of approval.

        Args:
            order: Order dict with keys: instrument_key, side, quantity,
                order_type, price, and optionally client_order_id.

        Returns:
            Dict with keys:
                ``required_paisa`` (int): Required margin in paisa.
                ``available_paisa`` (int): Available margin in paisa.
                ``approved`` (bool): True if margin is sufficient.
                ``rejection_reason`` (str | None): Set when approved is False.
                ``raw_response`` (dict): Raw API/stub response.
        """
        instrument_key = order.get("instrument_key", "")
        side = order.get("side", "")
        quantity = int(order.get("quantity", 0))
        client_order_id = order.get("client_order_id")

        # ------------------------------------------------------------------
        # Margin calculation — stub returns ok.
        # Real implementation: POST to Upstox /v3/charges/margin with
        # {"instrument_key": ..., "quantity": ..., "side": ..., "order_type": ...}
        # ------------------------------------------------------------------
        required_paisa, available_paisa = self._fetch_margin(order)

        # Apply buffer: need required * (1 + buffer_pct/100) ≤ available
        buffered = int(required_paisa * (1 + self._cfg.margin_buffer_pct / 100))
        approved = available_paisa >= buffered if self._cfg.enable_margin_check else True
        rejection_reason: Optional[str] = None
        if not approved:
            rejection_reason = (
                f"Insufficient margin: required {buffered} paisa "
                f"(incl {self._cfg.margin_buffer_pct}% buffer), "
                f"available {available_paisa} paisa"
            )

        raw_response: dict = {
            "stub": True,
            "required_paisa": required_paisa,
            "available_paisa": available_paisa,
        }

        result = {
            "required_paisa": required_paisa,
            "available_paisa": available_paisa,
            "approved": approved,
            "rejection_reason": rejection_reason,
            "raw_response": raw_response,
        }

        # Persist to DB
        self._persist_margin_check(
            client_order_id=client_order_id,
            instrument_key=instrument_key,
            side=side,
            quantity=quantity,
            required_paisa=required_paisa,
            available_paisa=available_paisa,
            approved=approved,
            rejection_reason=rejection_reason,
            raw_response=raw_response,
        )

        return result

    def _fetch_margin(self, order: dict) -> tuple[int, int]:
        """Return (required_paisa, available_paisa) for the order.

        Currently a stub.  Real wiring connects to Upstox margin API.

        Args:
            order: Order dict.

        Returns:
            Tuple of (required_margin_paisa, available_margin_paisa).
        """
        # Try broker.get_funds() to get available margin
        available_paisa = 0
        if self._broker is not None:
            try:
                funds = self._broker.get_funds()
                # get_funds() returns int (paisa) or a Funds object
                if isinstance(funds, int):
                    available_paisa = funds
                elif hasattr(funds, "available_margin"):
                    available_paisa = int(funds.available_margin)
            except Exception:
                logger.debug("broker.get_funds() failed; defaulting available to 0")

        # Stub: required margin = 0 (always approved unless broker says 0 available)
        # When wiring: replace with POST /v3/charges/margin response
        required_paisa = 0

        return required_paisa, available_paisa

    def _persist_margin_check(
        self,
        *,
        client_order_id: Optional[str],
        instrument_key: str,
        side: str,
        quantity: int,
        required_paisa: int,
        available_paisa: int,
        approved: bool,
        rejection_reason: Optional[str],
        raw_response: dict,
    ) -> None:
        """Insert a row into the margin_checks table.

        Silently skips if db_engine is None or the insert fails.

        Args:
            client_order_id: Caller-supplied order identifier (may be None).
            instrument_key: Instrument being ordered.
            side: BUY or SELL.
            quantity: Number of units (post lot-round).
            required_paisa: Required SPAN margin in paisa.
            available_paisa: Available margin in paisa at check time.
            approved: Whether the margin check passed.
            rejection_reason: Human-readable reason when approved is False.
            raw_response: Raw response dict from margin API / stub.
        """
        if self._db is None:
            return
        try:
            import json

            from sqlalchemy import text

            sql = text(
                """
                INSERT INTO margin_checks (
                    ts, client_order_id, instrument_key, side, quantity,
                    required_margin_paisa, available_margin_paisa,
                    approved, rejection_reason, raw_response
                ) VALUES (
                    NOW(), :client_order_id, :instrument_key, :side, :quantity,
                    :required_margin_paisa, :available_margin_paisa,
                    :approved, :rejection_reason, :raw_response
                )
                """
            )
            with self._db.begin() as conn:
                conn.execute(
                    sql,
                    {
                        "client_order_id": client_order_id,
                        "instrument_key": instrument_key,
                        "side": side,
                        "quantity": quantity,
                        "required_margin_paisa": required_paisa,
                        "available_margin_paisa": available_paisa,
                        "approved": approved,
                        "rejection_reason": rejection_reason,
                        "raw_response": json.dumps(raw_response),
                    },
                )
        except Exception as exc:
            logger.warning("Failed to persist margin_check row: %s", exc)

    # ------------------------------------------------------------------
    # Master validate
    # ------------------------------------------------------------------

    def validate(self, order: dict) -> dict:
        """Run full pre-trade validation pipeline on *order*.

        Steps:
            1. Lot-round the requested quantity (mode=down).
            2. Reject if rounded quantity is 0.
            3. Split the rounded quantity for freeze limit.
            4. Run margin check for each slice.
            5. Aggregate required margin; reject if total exceeds available.

        Args:
            order: Dict with keys:
                ``instrument_key`` (str): Instrument identifier.
                ``side`` (str): ``'BUY'`` or ``'SELL'``.
                ``quantity`` (int): Requested number of units.
                ``order_type`` (str): e.g. ``'MARKET'``, ``'LIMIT'``.
                ``price`` (int | None): Limit price in paisa.
                ``client_order_id`` (str | None): Optional caller ID.

        Returns:
            Dict with keys:
                ``valid`` (bool): True if the order can proceed.
                ``rejection_reason`` (str | None): Human-readable reason if
                    valid is False; absent (None) when valid.
                ``split_orders`` (list[dict]): Post-split order dicts, each
                    ≤ freeze_qty and a multiple of lot_size.  Empty list when
                    valid is False.
                ``margin_check`` (dict): Result from the last ``check_margin``
                    call (or aggregated result when multiple slices).
        """
        instrument_key = order.get("instrument_key", "")
        requested_qty = int(order.get("quantity", 0))

        # 1. Lot round (down — conservative)
        lot_size = self.get_lot_size(instrument_key)
        rounded_qty = self.round_to_lot(requested_qty, lot_size, mode="down")

        # 2. Reject zero quantity
        if rounded_qty == 0:
            return {
                "valid": False,
                "rejection_reason": (
                    f"Quantity {requested_qty} rounds to 0 with lot_size={lot_size}"
                ),
                "split_orders": [],
                "margin_check": {
                    "required_paisa": 0,
                    "available_paisa": 0,
                    "approved": False,
                    "rejection_reason": "zero quantity after lot rounding",
                    "raw_response": {},
                },
            }

        # 3. Split for freeze
        freeze_qty = self.get_freeze_quantity(instrument_key)
        slice_qtys = self.split_for_freeze(rounded_qty, freeze_qty, lot_size)

        # Build split order dicts
        base_order = dict(order)
        base_order["quantity"] = rounded_qty  # updated rounded qty

        split_orders: list[dict] = []
        for sq in slice_qtys:
            slice_order = dict(base_order)
            slice_order["quantity"] = sq
            split_orders.append(slice_order)

        # 4 & 5. Margin check per slice + aggregate
        total_required = 0
        last_margin_result: dict = {}
        available_paisa = 0

        for slice_order in split_orders:
            margin_result = self.check_margin(slice_order)
            total_required += margin_result["required_paisa"]
            available_paisa = margin_result["available_paisa"]
            last_margin_result = margin_result

        # Re-evaluate approval with aggregated required
        if self._cfg.enable_margin_check:
            buffered_total = int(
                total_required * (1 + self._cfg.margin_buffer_pct / 100)
            )
            if available_paisa < buffered_total:
                agg_margin = dict(last_margin_result)
                agg_margin["required_paisa"] = total_required
                agg_margin["approved"] = False
                agg_margin["rejection_reason"] = (
                    f"Insufficient total margin: required {buffered_total} paisa "
                    f"(incl {self._cfg.margin_buffer_pct}% buffer for {len(split_orders)} slices), "
                    f"available {available_paisa} paisa"
                )
                return {
                    "valid": False,
                    "rejection_reason": agg_margin["rejection_reason"],
                    "split_orders": [],
                    "margin_check": agg_margin,
                }

        # All clear
        agg_margin = dict(last_margin_result)
        agg_margin["required_paisa"] = total_required
        agg_margin["approved"] = True
        agg_margin["rejection_reason"] = None

        return {
            "valid": True,
            "rejection_reason": None,
            "split_orders": split_orders,
            "margin_check": agg_margin,
        }

    # ------------------------------------------------------------------
    # Audit query
    # ------------------------------------------------------------------

    def get_recent_rejections(self, days: int = 7) -> list[dict]:
        """Return margin check records that were rejected in the last *days* days.

        Args:
            days: Look-back window in calendar days.

        Returns:
            List of dicts with margin_checks columns.  Empty list if DB is
            unavailable or no rejections exist.
        """
        if self._db is None:
            return []

        try:
            from sqlalchemy import text

            cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
            with self._db.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT id, ts, client_order_id, instrument_key, side,
                               quantity, required_margin_paisa, available_margin_paisa,
                               approved, rejection_reason
                        FROM margin_checks
                        WHERE approved = FALSE AND ts >= :cutoff
                        ORDER BY ts DESC
                        """
                    ),
                    {"cutoff": cutoff},
                )
                rows = result.fetchall()
                cols = [
                    "id", "ts", "client_order_id", "instrument_key", "side",
                    "quantity", "required_margin_paisa", "available_margin_paisa",
                    "approved", "rejection_reason",
                ]
                return [dict(zip(cols, row)) for row in rows]
        except Exception as exc:
            logger.warning("get_recent_rejections failed: %s", exc)
            return []
