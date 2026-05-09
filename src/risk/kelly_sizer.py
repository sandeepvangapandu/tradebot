"""Kelly Criterion position sizing with rolling-window edge calculation.

Supersedes the legacy ``src/risk/position_sizer.py`` (KellyPositionSizer).
Key differences:
- Reads trade performance from the *database* (daily_pnl + positions tables)
  rather than maintaining an in-memory trade list.
- Persists daily Kelly allocations to ``kelly_allocations_daily``.
- Maintains a ``strategy_performance_rolling`` summary refreshed at EOD.
- Applies confluence-score scaling and strict per-strategy caps.
- Default: Half-Kelly for safety.

All monetary values are in **paisa** (1 INR = 100 paisa).
All times are in IST (Asia/Kolkata).

Wiring agent is responsible for calling:
  - ``evaluate_strategy()`` before each signal is sized.
  - ``update_rolling_perf()`` at ~16:00 IST daily (post-market).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class KellyConfig:
    """Runtime configuration for KellySizer.

    Attributes:
        rolling_window_days: Number of calendar days of trade history to use.
        min_trades_for_kelly: Below this count, use ``fallback_pct`` instead.
        fallback_pct: Capital fraction to risk when history is insufficient.
        half_kelly: Divide full Kelly fraction by 2 (safety default: True).
        min_pct: Hard floor on recommended capital fraction (0.1%).
        max_pct: Hard cap on recommended capital fraction (5%).
        confluence_multiplier: If True, scale by confluence_score / 100.
        edge_decay_factor: Per-day exponential weight decay on trades (unused
            in the current linear implementation — reserved for future use).
    """

    rolling_window_days: int = 30
    min_trades_for_kelly: int = 20
    fallback_pct: float = 0.005          # 0.5%
    half_kelly: bool = True
    min_pct: float = 0.001               # 0.1%
    max_pct: float = 0.05                # 5%
    confluence_multiplier: bool = True
    edge_decay_factor: float = 0.95


# ---------------------------------------------------------------------------
# Main sizer class
# ---------------------------------------------------------------------------


class KellySizer:
    """Rolling-window Kelly Criterion position sizer backed by a database.

    Args:
        db_engine: A SQLAlchemy sync engine pointing at the tradebot database.
            When ``None`` (unit tests), DB calls are skipped / mocked.
        config: Optional ``KellyConfig`` instance; defaults are safe.
    """

    def __init__(
        self,
        db_engine: Any = None,
        config: KellyConfig | None = None,
    ) -> None:
        self._engine = db_engine
        self._cfg = config or KellyConfig()

    # ------------------------------------------------------------------
    # Public API — read / compute
    # ------------------------------------------------------------------

    def compute_rolling_perf(self, strategy_name: str, trade_date: date) -> dict:
        """Aggregate trade statistics for a strategy over the rolling window.

        Reads from ``daily_pnl`` (and falls back to ``positions``) for the
        rolling window ending on *trade_date* (inclusive).

        Returns a dict with keys:
            trade_count, win_count, loss_count, win_rate,
            avg_win_paisa, avg_loss_paisa,
            expectancy, sharpe, max_drawdown_pct
        """
        window_start = trade_date - timedelta(days=self._cfg.rolling_window_days)

        pnl_rows: list[int] = []

        if self._engine is not None:
            pnl_rows = self._fetch_pnl_rows(strategy_name, window_start, trade_date)

        return self._aggregate_pnl_rows(pnl_rows)

    def compute_full_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Compute the full Kelly fraction.

        Formula: ``f* = (b*p - q) / b``
        where ``b = avg_win / avg_loss``, ``p = win_rate``, ``q = 1 - p``.

        Args:
            win_rate: Probability of a winning trade (0–1).
            avg_win: Average profit of winning trades (paisa, positive).
            avg_loss: Average loss of losing trades (paisa, positive absolute).

        Returns:
            Kelly fraction in ``[0, 1]``. Returns 0 if edge ≤ 0 or inputs
            are invalid.
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0

        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p

        numerator = b * p - q
        if numerator <= 0:
            return 0.0

        return max(0.0, min(1.0, numerator / b))

    def compute_recommended_pct(
        self,
        perf: dict,
        confluence_score: float | None = None,
    ) -> float:
        """Convert rolling performance stats into a final capital fraction.

        Pipeline:
        1. If trade_count < min_trades_for_kelly → return ``fallback_pct``.
        2. Compute full Kelly.
        3. If half_kelly → divide by 2.
        4. Scale by confluence_score / 100 (if enabled and provided).
        5. Clamp to [min_pct, max_pct].

        Args:
            perf: Dict returned by ``compute_rolling_perf``.
            confluence_score: Optional 0–100 score from the confluence engine.

        Returns:
            Recommended capital fraction (e.g. 0.02 = 2%).
        """
        trade_count: int = perf.get("trade_count", 0)

        if trade_count < self._cfg.min_trades_for_kelly:
            return self._cfg.fallback_pct

        win_rate: float = perf.get("win_rate", 0.0)
        avg_win: float = float(perf.get("avg_win_paisa") or 0)
        avg_loss: float = float(perf.get("avg_loss_paisa") or 0)

        full_kelly = self.compute_full_kelly(win_rate, avg_win, avg_loss)

        if full_kelly <= 0:
            return self._cfg.fallback_pct

        pct = full_kelly / 2.0 if self._cfg.half_kelly else full_kelly

        # Confluence scaling
        if (
            self._cfg.confluence_multiplier
            and confluence_score is not None
        ):
            scale = max(0.0, min(1.0, confluence_score / 100.0))
            pct *= scale

        # After confluence scaling, if result dropped below fallback use fallback
        if pct <= 0:
            return self._cfg.fallback_pct

        # Hard clamp
        return max(self._cfg.min_pct, min(self._cfg.max_pct, pct))

    def evaluate_strategy(
        self,
        strategy_name: str,
        trade_date: date,
        confluence_score: float | None = None,
    ) -> dict:
        """Full pipeline: rolling perf → Kelly → recommended pct → DB persist.

        Args:
            strategy_name: Strategy identifier (matches ``strategy_name`` in DB).
            trade_date: Date for which to compute allocation (usually today IST).
            confluence_score: Optional 0–100 score from the confluence engine.

        Returns:
            Record dict (same shape as ``kelly_allocations_daily`` row).
        """
        perf = self.compute_rolling_perf(strategy_name, trade_date)

        win_rate: float = perf.get("win_rate", 0.0)
        avg_win: float = float(perf.get("avg_win_paisa") or 0)
        avg_loss: float = float(perf.get("avg_loss_paisa") or 0)
        trade_count: int = perf.get("trade_count", 0)

        full_kelly = self.compute_full_kelly(win_rate, avg_win, avg_loss)
        half_kelly_pct = full_kelly / 2.0

        # Expectancy and edge
        expectancy: float = 0.0
        edge: float = 0.0
        if avg_loss > 0:
            q = 1.0 - win_rate
            expectancy = win_rate * avg_win - q * avg_loss
            edge = expectancy / avg_loss

        recommended_pct = self.compute_recommended_pct(perf, confluence_score)

        record: dict = {
            "trade_date": trade_date,
            "strategy_name": strategy_name,
            "win_rate": win_rate if trade_count >= self._cfg.min_trades_for_kelly else None,
            "avg_win_paisa": int(avg_win) if avg_win else None,
            "avg_loss_paisa": int(avg_loss) if avg_loss else None,
            "expectancy": expectancy if trade_count >= self._cfg.min_trades_for_kelly else None,
            "edge": edge if trade_count >= self._cfg.min_trades_for_kelly else None,
            "full_kelly_pct": full_kelly if trade_count >= self._cfg.min_trades_for_kelly else None,
            "half_kelly_pct": half_kelly_pct if trade_count >= self._cfg.min_trades_for_kelly else None,
            "recommended_pct": recommended_pct,
            "trade_count": trade_count,
        }

        if self._engine is not None:
            self._upsert_allocation(record)

        return record

    def size_position(
        self,
        strategy_name: str,
        trade_date: date,
        capital_paisa: int,
        sl_distance_paisa: int,
        confluence_score: float | None = None,
    ) -> dict:
        """Compute final position size in quantity units.

        Args:
            strategy_name: Strategy identifier.
            trade_date: Date for Kelly evaluation.
            capital_paisa: Total available capital in paisa.
            sl_distance_paisa: Stop-loss distance from entry in paisa (> 0).
            confluence_score: Optional 0–100 confluence score.

        Returns:
            Dict with keys:
                pct_used      – capital fraction actually used
                risk_paisa    – capital_paisa * pct_used (in paisa)
                position_qty  – risk_paisa // sl_distance_paisa (raw units)
                lot_count     – 0 (caller rounds to instrument lot_size)
        """
        record = self.evaluate_strategy(strategy_name, trade_date, confluence_score)
        pct_used: float = record["recommended_pct"]

        risk_paisa = int(capital_paisa * pct_used)

        if sl_distance_paisa <= 0:
            position_qty = 0
        else:
            position_qty = risk_paisa // sl_distance_paisa

        return {
            "pct_used": pct_used,
            "risk_paisa": risk_paisa,
            "position_qty": int(position_qty),
            "lot_count": 0,  # Caller rounds using instrument lot_size
        }

    def update_rolling_perf(self, strategy_name: str | None = None) -> int:
        """Recompute and persist ``strategy_performance_rolling``.

        Intended to be called at ~16:00 IST daily (post-market close).

        Args:
            strategy_name: If provided, update only this strategy.
                           If None, update all known strategies.

        Returns:
            Number of strategy rows updated.
        """
        if self._engine is None:
            return 0

        today = datetime.now(tz=IST).date()
        window_start = today - timedelta(days=self._cfg.rolling_window_days)

        strategy_names = self._fetch_strategy_names(strategy_name)

        count = 0
        for name in strategy_names:
            pnl_rows = self._fetch_pnl_rows(name, window_start, today)
            perf = self._aggregate_pnl_rows(pnl_rows)
            self._upsert_rolling_perf(name, perf)
            count += 1

        return count

    def get_today_allocation(
        self,
        strategy_name: str,
        trade_date: date,
    ) -> dict | None:
        """Fetch a previously persisted Kelly allocation record.

        Args:
            strategy_name: Strategy identifier.
            trade_date: Date of the allocation.

        Returns:
            Dict matching ``kelly_allocations_daily`` columns, or None.
        """
        if self._engine is None:
            return None

        return self._fetch_allocation(strategy_name, trade_date)

    # ------------------------------------------------------------------
    # Private helpers — DB I/O
    # ------------------------------------------------------------------

    def _fetch_pnl_rows(
        self,
        strategy_name: str,
        window_start: date,
        trade_date: date,
    ) -> list[int]:
        """Query daily_pnl for all trade PnL values in the rolling window.

        Returns a list of integers (paisa, positive=win, negative=loss).
        Falls back to an empty list on query failure.
        """
        from sqlalchemy import text

        sql = text(
            """
            SELECT realized_pnl_paisa
            FROM daily_pnl
            WHERE strategy_name = :sname
              AND trade_date BETWEEN :wstart AND :wend
              AND realized_pnl_paisa IS NOT NULL
            ORDER BY trade_date
            """
        )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sql,
                    {"sname": strategy_name, "wstart": window_start, "wend": trade_date},
                ).fetchall()
            return [int(r[0]) for r in rows]
        except Exception:
            return []

    def _fetch_strategy_names(self, strategy_name: str | None) -> list[str]:
        """Return list of strategy names to update.

        If ``strategy_name`` is provided, returns ``[strategy_name]``.
        Otherwise queries ``daily_pnl`` for all distinct names.
        """
        if strategy_name is not None:
            return [strategy_name]

        from sqlalchemy import text

        sql = text("SELECT DISTINCT strategy_name FROM daily_pnl")
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(sql).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def _upsert_allocation(self, record: dict) -> None:
        """Insert or update a row in ``kelly_allocations_daily``."""
        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO kelly_allocations_daily
              (trade_date, strategy_name, win_rate, avg_win_paisa, avg_loss_paisa,
               expectancy, edge, full_kelly_pct, half_kelly_pct, recommended_pct, trade_count)
            VALUES
              (:trade_date, :strategy_name, :win_rate, :avg_win_paisa, :avg_loss_paisa,
               :expectancy, :edge, :full_kelly_pct, :half_kelly_pct, :recommended_pct, :trade_count)
            ON CONFLICT (trade_date, strategy_name)
            DO UPDATE SET
              win_rate = EXCLUDED.win_rate,
              avg_win_paisa = EXCLUDED.avg_win_paisa,
              avg_loss_paisa = EXCLUDED.avg_loss_paisa,
              expectancy = EXCLUDED.expectancy,
              edge = EXCLUDED.edge,
              full_kelly_pct = EXCLUDED.full_kelly_pct,
              half_kelly_pct = EXCLUDED.half_kelly_pct,
              recommended_pct = EXCLUDED.recommended_pct,
              trade_count = EXCLUDED.trade_count
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(sql, record)
        except Exception:
            pass  # Non-fatal — sizing still works without persistence

    def _upsert_rolling_perf(self, strategy_name: str, perf: dict) -> None:
        """Insert or update a row in ``strategy_performance_rolling``."""
        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO strategy_performance_rolling
              (strategy_name, rolling_window_days, trade_count, win_count, loss_count,
               total_pnl_paisa, total_win_paisa, total_loss_paisa,
               sharpe_ratio, max_drawdown_pct, computed_at)
            VALUES
              (:strategy_name, :rolling_window_days, :trade_count, :win_count, :loss_count,
               :total_pnl_paisa, :total_win_paisa, :total_loss_paisa,
               :sharpe_ratio, :max_drawdown_pct, NOW())
            ON CONFLICT (strategy_name)
            DO UPDATE SET
              rolling_window_days = EXCLUDED.rolling_window_days,
              trade_count         = EXCLUDED.trade_count,
              win_count           = EXCLUDED.win_count,
              loss_count          = EXCLUDED.loss_count,
              total_pnl_paisa     = EXCLUDED.total_pnl_paisa,
              total_win_paisa     = EXCLUDED.total_win_paisa,
              total_loss_paisa    = EXCLUDED.total_loss_paisa,
              sharpe_ratio        = EXCLUDED.sharpe_ratio,
              max_drawdown_pct    = EXCLUDED.max_drawdown_pct,
              computed_at         = NOW()
            """
        )
        params = {
            "strategy_name": strategy_name,
            "rolling_window_days": self._cfg.rolling_window_days,
            "trade_count": perf.get("trade_count", 0),
            "win_count": perf.get("win_count", 0),
            "loss_count": perf.get("loss_count", 0),
            "total_pnl_paisa": perf.get("total_pnl_paisa", 0),
            "total_win_paisa": perf.get("total_win_paisa", 0),
            "total_loss_paisa": perf.get("total_loss_paisa", 0),
            "sharpe_ratio": perf.get("sharpe"),
            "max_drawdown_pct": perf.get("max_drawdown_pct"),
        }
        try:
            with self._engine.begin() as conn:
                conn.execute(sql, params)
        except Exception:
            pass

    def _fetch_allocation(self, strategy_name: str, trade_date: date) -> dict | None:
        """Fetch a single row from ``kelly_allocations_daily``."""
        from sqlalchemy import text

        sql = text(
            """
            SELECT trade_date, strategy_name, win_rate, avg_win_paisa, avg_loss_paisa,
                   expectancy, edge, full_kelly_pct, half_kelly_pct, recommended_pct, trade_count
            FROM kelly_allocations_daily
            WHERE strategy_name = :sname AND trade_date = :tdate
            """
        )
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    sql, {"sname": strategy_name, "tdate": trade_date}
                ).fetchone()
            if row is None:
                return None
            cols = [
                "trade_date", "strategy_name", "win_rate", "avg_win_paisa",
                "avg_loss_paisa", "expectancy", "edge", "full_kelly_pct",
                "half_kelly_pct", "recommended_pct", "trade_count",
            ]
            return dict(zip(cols, row))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Private helpers — pure maths
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_pnl_rows(pnl_rows: list[int]) -> dict:
        """Compute statistics from a flat list of trade PnL values (paisa).

        Args:
            pnl_rows: List of integers; positive = win, negative = loss.

        Returns:
            Dict with computed stats (safe defaults when list is empty).
        """
        if not pnl_rows:
            return {
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0.0,
                "avg_win_paisa": 0,
                "avg_loss_paisa": 0,
                "expectancy": 0.0,
                "sharpe": None,
                "max_drawdown_pct": None,
                "total_pnl_paisa": 0,
                "total_win_paisa": 0,
                "total_loss_paisa": 0,
            }

        wins = [v for v in pnl_rows if v > 0]
        losses = [abs(v) for v in pnl_rows if v < 0]

        trade_count = len(pnl_rows)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / trade_count if trade_count else 0.0

        avg_win = sum(wins) / win_count if wins else 0
        avg_loss = sum(losses) / loss_count if losses else 0

        q = 1.0 - win_rate
        expectancy = win_rate * avg_win - q * avg_loss

        total_pnl = sum(pnl_rows)
        total_win = sum(wins)
        total_loss = sum(losses)

        # Sharpe (simplified — daily PnL as returns, no risk-free rate)
        sharpe: float | None = None
        if len(pnl_rows) >= 2:
            try:
                mean_pnl = statistics.mean(pnl_rows)
                std_pnl = statistics.stdev(pnl_rows)
                if std_pnl > 0:
                    # Annualise for 252 trading days
                    sharpe = (mean_pnl / std_pnl) * math.sqrt(252)
            except Exception:
                sharpe = None

        # Max drawdown (peak-to-trough on cumulative PnL series)
        max_drawdown_pct: float | None = None
        if pnl_rows:
            cumulative = 0
            peak = 0
            max_dd = 0
            initial_capital = 1_000_000_000  # 1Cr notional for % calc
            for v in pnl_rows:
                cumulative += v
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd
            if peak > 0:
                max_drawdown_pct = (max_dd / (initial_capital + peak)) * 100

        return {
            "trade_count": trade_count,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "avg_win_paisa": int(avg_win),
            "avg_loss_paisa": int(avg_loss),
            "expectancy": expectancy,
            "sharpe": sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "total_pnl_paisa": total_pnl,
            "total_win_paisa": total_win,
            "total_loss_paisa": total_loss,
        }
