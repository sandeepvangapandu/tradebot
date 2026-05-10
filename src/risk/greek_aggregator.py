"""Portfolio Greeks aggregation + caps.

Aggregates delta, gamma, vega, theta across all open option positions and
exposes per-cap checks so strategies can gate new positions before opening.

All monetary values are in **paisa** (1 INR = 100 paisa).
All times are in IST (Asia/Kolkata).

Typical usage by a strategy:
    aggregator = GreekAggregator(db_engine=engine)
    current_greeks = aggregator.aggregate_portfolio(open_positions)
    allowed, reason = aggregator.can_add_options_position(
        proposed, current_greeks, capital_paisa
    )
    if not allowed:
        log.warning("Rejected: %s", reason)

Wiring agent is responsible for calling:
  - ``snapshot()`` once per bar close (or on-demand) to persist to DB.
  - ``can_add_options_position()`` before each options order is sized.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class GreekCapConfig:
    """Runtime configuration for GreekAggregator caps.

    Attributes:
        max_abs_delta_pct: |portfolio delta| / capital_paisa ≤ this fraction
            expressed as a *percentage*.  Default 30%.
        max_gamma_per_lakh: Maximum |portfolio gamma| per ₹1 lakh capital.
            Default 0.5.
        max_vega_per_lakh_per_pct: Maximum |portfolio vega| per ₹1 lakh capital
            per 1% volatility move (i.e. raw vega ₹).  Default 500.
        max_theta_decay_pct: Maximum daily |theta decay| / capital as a
            *percentage*.  Default 1.0%.
        delta_neutral_tolerance: Portfolio is "delta-neutral" when
            |delta| / capital_paisa ≤ this fraction expressed as a percentage.
            Default 10%.
    """

    max_abs_delta_pct: float = 30.0
    max_gamma_per_lakh: float = 0.5
    max_vega_per_lakh_per_pct: float = 500.0
    max_theta_decay_pct: float = 1.0
    delta_neutral_tolerance: float = 10.0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class GreekAggregator:
    """Portfolio-level Greek aggregator backed by options_strike_oi_history.

    Args:
        db_engine: A SQLAlchemy sync engine pointing at the tradebot DB.
            When ``None`` (unit tests), DB calls are skipped or mocked.
        config: Optional ``GreekCapConfig``; defaults are safe.
    """

    def __init__(
        self,
        db_engine: Any = None,
        config: GreekCapConfig | None = None,
    ) -> None:
        self._engine = db_engine
        self.config = config or GreekCapConfig()

    # ------------------------------------------------------------------
    # Greek fetch
    # ------------------------------------------------------------------

    def fetch_position_greeks(
        self,
        instrument_key: str,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> dict:
        """Fetch the latest Greeks from options_strike_oi_history for a single leg.

        Args:
            instrument_key: Upstox underlying instrument key (e.g.
                ``NSE_INDEX|Nifty 50``).
            expiry: ISO-8601 date string (``YYYY-MM-DD``) or ``YYYY-MM-DD``
                ``date`` / ``str``.
            strike: Numeric strike price.
            option_type: ``"CE"`` or ``"PE"``.

        Returns:
            Dict with keys ``{delta, gamma, vega, theta}`` (float values), or
            an empty dict if no row is found in the DB.
        """
        if self._engine is None:
            return {}

        params = {
            "underlying_key": instrument_key,
            "expiry": str(expiry),
            "strike": float(strike),
            "option_type": option_type.upper(),
        }

        try:
            from sqlalchemy import text as sa_text

            named_sql = sa_text(
                """
                SELECT delta, gamma, vega, theta
                FROM options_strike_oi_history
                WHERE underlying_key = :underlying_key
                  AND expiry         = :expiry
                  AND strike         = :strike
                  AND option_type    = :option_type
                ORDER BY ts DESC
                LIMIT 1
                """
            )
            with self._engine.connect() as conn:
                row = conn.execute(named_sql, params).fetchone()
        except Exception as exc:
            log.warning("fetch_position_greeks DB error: %s", exc)
            return {}

        if row is None:
            return {}

        return {
            "delta": float(row[0]) if row[0] is not None else 0.0,
            "gamma": float(row[1]) if row[1] is not None else 0.0,
            "vega": float(row[2]) if row[2] is not None else 0.0,
            "theta": float(row[3]) if row[3] is not None else 0.0,
        }

    # ------------------------------------------------------------------
    # Portfolio aggregation
    # ------------------------------------------------------------------

    def aggregate_portfolio(self, open_positions: list[dict]) -> dict:
        """Aggregate Greeks across all open option positions.

        Each element of ``open_positions`` must have:
            - ``instrument_key`` (str): underlying key
            - ``expiry`` (str): ``YYYY-MM-DD``
            - ``strike`` (float): strike price
            - ``option_type`` (str): ``"CE"`` or ``"PE"``
            - ``side`` (str): ``"BUY"`` or ``"SELL"``
            - ``quantity`` (int): number of contracts/units

        Greek contribution per position:
            side_sign = +1 if BUY, -1 if SELL
            contribution = side_sign * quantity * greek_value

        Args:
            open_positions: List of position dicts as above.

        Returns:
            Dict with keys:
                ``total_delta``, ``total_gamma``, ``total_vega``,
                ``total_theta``, ``position_count``,
                ``underlying_breakdown`` (dict: underlying → {delta, …}).
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        breakdown: dict[str, dict[str, float]] = {}

        for pos in open_positions:
            ikey = pos.get("instrument_key", "")
            expiry = pos.get("expiry", "")
            strike = float(pos.get("strike", 0))
            otype = pos.get("option_type", "CE")
            side = pos.get("side", "BUY").upper()
            qty = int(pos.get("quantity", 0))

            side_sign = 1 if side == "BUY" else -1

            greeks = self.fetch_position_greeks(ikey, expiry, strike, otype)
            if not greeks:
                # No Greeks available — treat as zero contribution but still log
                log.debug("No Greeks for %s %s %s %s — skipping", ikey, expiry, strike, otype)
                greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

            d = side_sign * qty * greeks["delta"]
            g = side_sign * qty * greeks["gamma"]
            v = side_sign * qty * greeks["vega"]
            t = side_sign * qty * greeks["theta"]

            total_delta += d
            total_gamma += g
            total_vega += v
            total_theta += t

            # Per-underlying breakdown — use instrument_key as key
            if ikey not in breakdown:
                breakdown[ikey] = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
            breakdown[ikey]["delta"] += d
            breakdown[ikey]["gamma"] += g
            breakdown[ikey]["vega"] += v
            breakdown[ikey]["theta"] += t

        return {
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_vega": total_vega,
            "total_theta": total_theta,
            "position_count": len(open_positions),
            "underlying_breakdown": breakdown,
        }

    # ------------------------------------------------------------------
    # Snapshot persistence
    # ------------------------------------------------------------------

    def snapshot(
        self,
        trade_date: Any,
        open_positions: list[dict],
        capital_paisa: int,
    ) -> dict:
        """Aggregate Greeks and persist a snapshot row to portfolio_greeks_snapshot.

        Args:
            trade_date: A ``date`` or ISO-8601 ``str`` (``YYYY-MM-DD``).
            open_positions: Same format as ``aggregate_portfolio``.
            capital_paisa: Total capital in paisa.

        Returns:
            The aggregated Greeks dict (same as ``aggregate_portfolio`` output).
        """
        greeks = self.aggregate_portfolio(open_positions)
        ts_now = datetime.now(tz=IST)

        if self._engine is not None:
            try:
                from sqlalchemy import text as sa_text

                sql = sa_text(
                    """
                    INSERT INTO portfolio_greeks_snapshot
                        (ts, trade_date, total_delta, total_gamma, total_vega,
                         total_theta, total_rho, position_count,
                         underlying_breakdown, capital_paisa)
                    VALUES
                        (:ts, :trade_date, :total_delta, :total_gamma, :total_vega,
                         :total_theta, :total_rho, :position_count,
                         :underlying_breakdown, :capital_paisa)
                    ON CONFLICT DO NOTHING
                    """
                )
                params = {
                    "ts": ts_now,
                    "trade_date": str(trade_date),
                    "total_delta": greeks["total_delta"],
                    "total_gamma": greeks["total_gamma"],
                    "total_vega": greeks["total_vega"],
                    "total_theta": greeks["total_theta"],
                    "total_rho": None,
                    "position_count": greeks["position_count"],
                    "underlying_breakdown": json.dumps(greeks["underlying_breakdown"]),
                    "capital_paisa": capital_paisa,
                }
                with self._engine.begin() as conn:
                    conn.execute(sql, params)
                log.info(
                    "snapshot: date=%s delta=%.4f vega=%.2f theta=%.2f positions=%d",
                    trade_date,
                    greeks["total_delta"],
                    greeks["total_vega"],
                    greeks["total_theta"],
                    greeks["position_count"],
                )
            except Exception as exc:
                log.error("snapshot DB write failed: %s", exc)

        return greeks

    # ------------------------------------------------------------------
    # Cap checks
    # ------------------------------------------------------------------

    def is_delta_neutral(self, greeks: dict, capital_paisa: int) -> bool:
        """Return True if |portfolio delta| is within delta_neutral_tolerance.

        The check is: |total_delta| / (capital_paisa / 100) ≤ tolerance_pct / 100.
        Capital is converted from paisa to INR (÷ 100) for a meaningful ratio.

        Args:
            greeks: Output of ``aggregate_portfolio``.
            capital_paisa: Total capital in paisa.

        Returns:
            True if delta is within neutral band.
        """
        if capital_paisa <= 0:
            return True
        capital_inr = capital_paisa / 100.0
        ratio_pct = abs(greeks.get("total_delta", 0.0)) / capital_inr * 100.0
        return ratio_pct <= self.config.delta_neutral_tolerance

    def check_delta_cap(
        self,
        additional_delta: float,
        current_greeks: dict,
        capital_paisa: int,
    ) -> bool:
        """Return True if adding ``additional_delta`` keeps delta within cap.

        Cap: |total_delta + additional_delta| / capital_INR ≤ max_abs_delta_pct / 100.

        Args:
            additional_delta: Proposed delta contribution from the new position.
            current_greeks: Current aggregated Greeks.
            capital_paisa: Total capital in paisa.

        Returns:
            True if within cap (position is allowed from a delta perspective).
        """
        if capital_paisa <= 0:
            return True
        capital_inr = capital_paisa / 100.0
        new_delta = current_greeks.get("total_delta", 0.0) + additional_delta
        ratio_pct = abs(new_delta) / capital_inr * 100.0
        return ratio_pct <= self.config.max_abs_delta_pct

    def check_gamma_cap(
        self,
        additional_gamma: float,
        current_greeks: dict,
        capital_paisa: int,
    ) -> bool:
        """Return True if adding ``additional_gamma`` keeps gamma within cap.

        Cap: |total_gamma + additional_gamma| / (capital_INR / 1_00_000) ≤ max_gamma_per_lakh.

        Args:
            additional_gamma: Proposed gamma contribution.
            current_greeks: Current aggregated Greeks.
            capital_paisa: Total capital in paisa.

        Returns:
            True if within cap.
        """
        if capital_paisa <= 0:
            return True
        capital_inr = capital_paisa / 100.0
        lakhs = capital_inr / 100_000.0
        if lakhs <= 0:
            return True
        new_gamma = current_greeks.get("total_gamma", 0.0) + additional_gamma
        ratio = abs(new_gamma) / lakhs
        return ratio <= self.config.max_gamma_per_lakh

    def check_vega_cap(
        self,
        additional_vega: float,
        current_greeks: dict,
        capital_paisa: int,
    ) -> bool:
        """Return True if adding ``additional_vega`` keeps vega within cap.

        Cap: |total_vega + additional_vega| / (capital_INR / 1_00_000) ≤ max_vega_per_lakh_per_pct.

        Args:
            additional_vega: Proposed vega contribution (INR per 1% vol move).
            current_greeks: Current aggregated Greeks.
            capital_paisa: Total capital in paisa.

        Returns:
            True if within cap.
        """
        if capital_paisa <= 0:
            return True
        capital_inr = capital_paisa / 100.0
        lakhs = capital_inr / 100_000.0
        if lakhs <= 0:
            return True
        new_vega = current_greeks.get("total_vega", 0.0) + additional_vega
        ratio = abs(new_vega) / lakhs
        return ratio <= self.config.max_vega_per_lakh_per_pct

    def check_theta_budget(
        self,
        additional_theta: float,
        current_greeks: dict,
        capital_paisa: int,
    ) -> bool:
        """Return True if adding ``additional_theta`` stays within daily theta budget.

        Cap: |total_theta + additional_theta| / capital_INR ≤ max_theta_decay_pct / 100.
        Theta is typically negative for long options (daily decay), so we check
        the absolute value.

        Args:
            additional_theta: Proposed theta contribution (INR/day).
            current_greeks: Current aggregated Greeks.
            capital_paisa: Total capital in paisa.

        Returns:
            True if within budget.
        """
        if capital_paisa <= 0:
            return True
        capital_inr = capital_paisa / 100.0
        new_theta = current_greeks.get("total_theta", 0.0) + additional_theta
        ratio_pct = abs(new_theta) / capital_inr * 100.0
        return ratio_pct <= self.config.max_theta_decay_pct

    # ------------------------------------------------------------------
    # Combined gate
    # ------------------------------------------------------------------

    def can_add_options_position(
        self,
        proposed: dict,
        current_greeks: dict,
        capital_paisa: int,
    ) -> tuple[bool, str | None]:
        """Gate check before opening a new options position.

        Runs all four cap checks sequentially and returns on the first failure.

        Args:
            proposed: Dict with keys:
                ``instrument_key``, ``side`` (BUY/SELL), ``quantity``,
                ``expiry``, ``strike``, ``option_type``.
            current_greeks: Output of ``aggregate_portfolio`` for the current
                open book (excluding the proposed position).
            capital_paisa: Total capital in paisa.

        Returns:
            Tuple ``(allowed: bool, rejection_reason: str | None)``.
            ``rejection_reason`` is one of:
                ``"DELTA_CAP"``, ``"GAMMA_CAP"``, ``"VEGA_CAP"``, ``"THETA_BUDGET"``.
            ``None`` when allowed.
        """
        ikey = proposed.get("instrument_key", "")
        expiry = proposed.get("expiry", "")
        strike = float(proposed.get("strike", 0))
        otype = proposed.get("option_type", "CE")
        side = proposed.get("side", "BUY").upper()
        qty = int(proposed.get("quantity", 0))

        side_sign = 1 if side == "BUY" else -1

        greeks = self.fetch_position_greeks(ikey, expiry, strike, otype)
        if not greeks:
            greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

        add_delta = side_sign * qty * greeks["delta"]
        add_gamma = side_sign * qty * greeks["gamma"]
        add_vega = side_sign * qty * greeks["vega"]
        add_theta = side_sign * qty * greeks["theta"]

        if not self.check_delta_cap(add_delta, current_greeks, capital_paisa):
            return False, "DELTA_CAP"

        if not self.check_gamma_cap(add_gamma, current_greeks, capital_paisa):
            return False, "GAMMA_CAP"

        if not self.check_vega_cap(add_vega, current_greeks, capital_paisa):
            return False, "VEGA_CAP"

        if not self.check_theta_budget(add_theta, current_greeks, capital_paisa):
            return False, "THETA_BUDGET"

        return True, None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_today_snapshot(self, trade_date: Any) -> dict | None:
        """Retrieve the most recent snapshot row for ``trade_date``.

        Args:
            trade_date: A ``date`` or ISO-8601 str (``YYYY-MM-DD``).

        Returns:
            Dict with snapshot fields, or ``None`` if no row found.
        """
        if self._engine is None:
            return None

        try:
            from sqlalchemy import text as sa_text

            sql = sa_text(
                """
                SELECT ts, trade_date, total_delta, total_gamma, total_vega,
                       total_theta, total_rho, position_count,
                       underlying_breakdown, capital_paisa
                FROM portfolio_greeks_snapshot
                WHERE trade_date = :trade_date
                ORDER BY ts DESC
                LIMIT 1
                """
            )
            with self._engine.connect() as conn:
                row = conn.execute(sql, {"trade_date": str(trade_date)}).fetchone()
        except Exception as exc:
            log.error("get_today_snapshot DB error: %s", exc)
            return None

        if row is None:
            return None

        return {
            "ts": row[0],
            "trade_date": row[1],
            "total_delta": float(row[2]) if row[2] is not None else 0.0,
            "total_gamma": float(row[3]) if row[3] is not None else 0.0,
            "total_vega": float(row[4]) if row[4] is not None else 0.0,
            "total_theta": float(row[5]) if row[5] is not None else 0.0,
            "total_rho": float(row[6]) if row[6] is not None else None,
            "position_count": row[7],
            "underlying_breakdown": row[8] if row[8] is not None else {},
            "capital_paisa": row[9],
        }
