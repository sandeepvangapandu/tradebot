"""Portfolio-level risk: correlation, beta exposure, sector caps, VaR/CVaR.

Provides a master gate (``can_add_position``) that checks four independent
constraints before any new position is sized:

  1. SYMBOL_CAP  — symbol exposure < ``max_symbol_exposure_pct`` of capital
  2. SECTOR_CAP  — sector exposure < ``max_sector_exposure_pct`` of capital
  3. LEVERAGE_CAP — gross/net leverage within configured multiples
  4. CORRELATION_CAP — no more than ``correlation_max_grouped`` highly correlated
                       positions in the portfolio simultaneously

Data sources:
  - OHLCV bars from the ``bars`` table (instrument_key, timeframe='day', close).
  - Daily portfolio PnL from ``daily_pnl`` table for VaR computation.
  - Persists results to three Phase F.1 tables (see migration 20260508000050).

All monetary values are in **paisa** (1 INR = 100 paisa).
All times are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PortfolioRiskConfig:
    """Runtime configuration for PortfolioRisk.

    Attributes:
        correlation_lookback_days: Number of calendar days of bars used to
            compute rolling pairwise correlation (default 60).
        correlation_cap_threshold: Pairs with |corr| above this value are
            considered highly correlated (default 0.8).
        correlation_max_grouped: Maximum number of positions (including the
            proposed one) that may share high correlation with any existing
            symbol (default 3).
        max_sector_exposure_pct: Maximum capital deployed to any single
            sector as a percentage of total capital (default 40.0).
        max_symbol_exposure_pct: Maximum capital deployed to any single
            symbol as a percentage of total capital (default 20.0).
        max_gross_leverage: Maximum gross exposure multiple of capital
            (default 2.0 = 2× capital).
        max_net_long_leverage: Maximum net long exposure multiple of capital
            (default 1.5).
        var_confidence_levels: Confidence levels for VaR/CVaR computation
            (default [0.95, 0.99]).
        var_lookback_days: Number of daily returns used for historical VaR
            (default 252 — roughly 1 trading year).
    """

    correlation_lookback_days: int = 60
    correlation_cap_threshold: float = 0.8
    correlation_max_grouped: int = 3
    max_sector_exposure_pct: float = 40.0
    max_symbol_exposure_pct: float = 20.0
    max_gross_leverage: float = 2.0
    max_net_long_leverage: float = 1.5
    var_confidence_levels: list = field(default_factory=lambda: [0.95, 0.99])
    var_lookback_days: int = 252


# ---------------------------------------------------------------------------
# Helper: Pearson correlation
# ---------------------------------------------------------------------------


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    """Return Pearson correlation coefficient between two equal-length sequences.

    Returns 0.0 when variance is zero (constant series).

    Args:
        xs: First series.
        ys: Second series.

    Returns:
        Correlation coefficient in [-1.0, 1.0].
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (n - 1)
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / (n - 1))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / (n - 1))
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    return max(-1.0, min(1.0, cov / (std_x * std_y)))


def _log_returns(prices: list[float]) -> list[float]:
    """Compute log returns from a price series.

    Args:
        prices: Ordered price series (oldest first).

    Returns:
        List of log-return values (length = len(prices) - 1).
    """
    if len(prices) < 2:
        return []
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices)) if prices[i - 1] > 0]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PortfolioRisk:
    """Portfolio-level risk manager: correlation, exposure, VaR/CVaR.

    Args:
        db_engine: SQLAlchemy sync engine pointing at the tradebot database.
            When ``None`` (unit tests), all DB writes are silently skipped
            and bar data must be supplied via ``_bar_provider``.
        config: Optional ``PortfolioRiskConfig``; safe defaults are used.
        bar_provider: Optional callable ``(symbol, lookback_days, trade_date)
            -> list[float]`` that returns daily close prices (in paisa) for
            a given symbol.  Used to decouple unit tests from the database.
            When ``None``, prices are fetched from the ``bars`` table.
        pnl_provider: Optional callable ``(lookback_days, trade_date)
            -> list[int]`` that returns portfolio-level daily PnL values
            (paisa).  When ``None``, fetched from ``daily_pnl``.
    """

    def __init__(
        self,
        db_engine: Any = None,
        config: PortfolioRiskConfig | None = None,
        bar_provider: Any = None,
        pnl_provider: Any = None,
    ) -> None:
        self._engine = db_engine
        self._cfg = config or PortfolioRiskConfig()
        self._bar_provider = bar_provider
        self._pnl_provider = pnl_provider

    # ------------------------------------------------------------------
    # Internal data fetchers
    # ------------------------------------------------------------------

    def _fetch_closes(
        self, symbol: str, lookback_days: int, trade_date: date
    ) -> list[float]:
        """Return daily close prices (paisa) for *symbol* going back *lookback_days*.

        Uses ``_bar_provider`` when set, otherwise queries ``bars`` table.

        Args:
            symbol: Instrument key, e.g. ``"NSE_EQ|INE009A01021"``.
            lookback_days: Number of calendar days to look back.
            trade_date: Reference date (end of window).

        Returns:
            List of close prices in paisa, oldest-first.  May be empty.
        """
        if self._bar_provider is not None:
            result = self._bar_provider(symbol, lookback_days, trade_date)
            return [float(p) for p in result]

        if self._engine is None:
            return []

        from sqlalchemy import text

        cutoff = trade_date - timedelta(days=lookback_days)
        sql = text(
            """
            SELECT close
            FROM bars
            WHERE instrument_key = :sym
              AND timeframe = 'day'
              AND ts::date BETWEEN :cutoff AND :end_date
            ORDER BY ts ASC
            """
        )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sql, {"sym": symbol, "cutoff": cutoff, "end_date": trade_date}
                ).fetchall()
            return [float(r[0]) for r in rows]
        except Exception:
            return []

    def _fetch_portfolio_pnl(
        self, lookback_days: int, trade_date: date
    ) -> list[int]:
        """Return daily net portfolio PnL values (paisa) for VaR computation.

        Sums across all strategies per day.

        Args:
            lookback_days: Number of calendar days to look back.
            trade_date: Reference date (end of window).

        Returns:
            List of integers (paisa), one entry per trading day.  May be empty.
        """
        if self._pnl_provider is not None:
            return [int(v) for v in self._pnl_provider(lookback_days, trade_date)]

        if self._engine is None:
            return []

        from sqlalchemy import text

        cutoff = trade_date - timedelta(days=lookback_days)
        sql = text(
            """
            SELECT trade_date, SUM(realized_pnl_paisa)
            FROM daily_pnl
            WHERE trade_date BETWEEN :cutoff AND :end_date
              AND realized_pnl_paisa IS NOT NULL
            GROUP BY trade_date
            ORDER BY trade_date ASC
            """
        )
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sql, {"cutoff": cutoff, "end_date": trade_date}
                ).fetchall()
            return [int(r[1]) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    def compute_correlation_matrix(
        self,
        symbols: list[str],
        trade_date: date,
        lookback_days: int | None = None,
    ) -> "dict[tuple[str, str], float]":
        """Compute pairwise rolling correlation and persist to DB.

        For each ordered pair (a, b) with a <= b lexicographically, fetches
        daily close prices, computes log-returns, and calculates Pearson
        correlation.  Both (a, b) and (b, a) are written to
        ``correlation_matrix_daily`` for convenient lookup.

        Args:
            symbols: List of instrument keys to correlate.
            trade_date: Date for which to compute the matrix.
            lookback_days: Override for config ``correlation_lookback_days``.

        Returns:
            Dict mapping ``(symbol_a, symbol_b) -> correlation`` for all pairs
            including self-correlation (= 1.0) and reversed pairs.
        """
        n_days = lookback_days or self._cfg.correlation_lookback_days

        # Fetch log-returns for every symbol
        returns_map: dict[str, list[float]] = {}
        for sym in symbols:
            closes = self._fetch_closes(sym, n_days, trade_date)
            returns_map[sym] = _log_returns(closes)

        matrix: dict[tuple[str, str], float] = {}
        rows_to_insert: list[dict] = []

        for i, sym_a in enumerate(symbols):
            for j, sym_b in enumerate(symbols):
                if (sym_a, sym_b) in matrix:
                    matrix[(sym_b, sym_a)] = matrix[(sym_a, sym_b)]
                    continue
                if sym_a == sym_b:
                    corr = 1.0
                else:
                    ra = returns_map[sym_a]
                    rb = returns_map[sym_b]
                    n = min(len(ra), len(rb))
                    corr = _pearson_corr(ra[-n:], rb[-n:]) if n >= 2 else 0.0
                matrix[(sym_a, sym_b)] = corr
                matrix[(sym_b, sym_a)] = corr

                rows_to_insert.append(
                    {
                        "trade_date": trade_date,
                        "symbol_a": sym_a,
                        "symbol_b": sym_b,
                        "correlation": corr,
                        "lookback_days": n_days,
                    }
                )
                if sym_a != sym_b:
                    rows_to_insert.append(
                        {
                            "trade_date": trade_date,
                            "symbol_a": sym_b,
                            "symbol_b": sym_a,
                            "correlation": corr,
                            "lookback_days": n_days,
                        }
                    )

        self._persist_correlation_rows(rows_to_insert)
        return matrix

    def _persist_correlation_rows(self, rows: list[dict]) -> None:
        """Upsert correlation rows into ``correlation_matrix_daily``."""
        if not rows or self._engine is None:
            return

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO correlation_matrix_daily
                (trade_date, symbol_a, symbol_b, correlation, lookback_days)
            VALUES (:trade_date, :symbol_a, :symbol_b, :correlation, :lookback_days)
            ON CONFLICT (trade_date, symbol_a, symbol_b)
            DO UPDATE SET
                correlation = EXCLUDED.correlation,
                lookback_days = EXCLUDED.lookback_days
            """
        )
        try:
            with self._engine.begin() as conn:
                for row in rows:
                    conn.execute(sql, row)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Correlation getters
    # ------------------------------------------------------------------

    def get_correlation(
        self, symbol_a: str, symbol_b: str, trade_date: date
    ) -> float | None:
        """Return stored correlation between two symbols on a given date.

        Queries ``correlation_matrix_daily``.  Returns ``None`` if not found.

        Args:
            symbol_a: First instrument key.
            symbol_b: Second instrument key.
            trade_date: Date to query.

        Returns:
            Correlation float or None.
        """
        if self._engine is None:
            return None

        from sqlalchemy import text

        sql = text(
            """
            SELECT correlation
            FROM correlation_matrix_daily
            WHERE trade_date = :d AND symbol_a = :a AND symbol_b = :b
            LIMIT 1
            """
        )
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    sql, {"d": trade_date, "a": symbol_a, "b": symbol_b}
                ).fetchone()
            return float(row[0]) if row else None
        except Exception:
            return None

    def find_correlated_group(
        self,
        symbol: str,
        open_positions: list[dict],
        trade_date: date,
        threshold: float | None = None,
    ) -> list[str]:
        """Return symbols in the open portfolio highly correlated with *symbol*.

        Correlation is looked up from ``correlation_matrix_daily`` first; if
        unavailable the pair is skipped (treated as uncorrelated).

        Args:
            symbol: The candidate instrument key.
            open_positions: List of dicts, each with at least ``"symbol"`` key.
            trade_date: Date for correlation lookup.
            threshold: Override for ``correlation_cap_threshold``.

        Returns:
            List of instrument keys (from open_positions) whose absolute
            correlation with *symbol* exceeds the threshold.
        """
        thresh = threshold if threshold is not None else self._cfg.correlation_cap_threshold
        correlated: list[str] = []
        for pos in open_positions:
            pos_sym = pos.get("symbol", "")
            if pos_sym == symbol:
                continue
            corr = self.get_correlation(symbol, pos_sym, trade_date)
            if corr is None:
                continue
            if abs(corr) >= thresh:
                correlated.append(pos_sym)
        return correlated

    # ------------------------------------------------------------------
    # Beta
    # ------------------------------------------------------------------

    def compute_beta(
        self,
        symbol: str,
        benchmark_key: str = "NSE_INDEX|Nifty 50",
        lookback_days: int = 60,
        trade_date: date | None = None,
    ) -> float:
        """Compute beta of *symbol* relative to *benchmark_key*.

        Beta = cov(symbol_returns, benchmark_returns) / var(benchmark_returns).
        Returns 1.0 when variance is zero or insufficient data.

        Args:
            symbol: Instrument key.
            benchmark_key: Benchmark instrument key (default NIFTY 50).
            lookback_days: Number of calendar days of history to use.
            trade_date: Reference end date (defaults to today IST).

        Returns:
            Beta float. 1.0 if data unavailable.
        """
        td = trade_date or datetime.now(tz=IST).date()
        sym_closes = self._fetch_closes(symbol, lookback_days, td)
        bm_closes = self._fetch_closes(benchmark_key, lookback_days, td)

        sym_ret = _log_returns(sym_closes)
        bm_ret = _log_returns(bm_closes)

        n = min(len(sym_ret), len(bm_ret))
        if n < 2:
            return 1.0

        sr = sym_ret[-n:]
        br = bm_ret[-n:]
        mean_bm = sum(br) / n
        var_bm = sum((b - mean_bm) ** 2 for b in br) / (n - 1)
        if var_bm == 0.0:
            return 1.0

        mean_sr = sum(sr) / n
        cov = sum((s - mean_sr) * (b - mean_bm) for s, b in zip(sr, br)) / (n - 1)
        return cov / var_bm

    # ------------------------------------------------------------------
    # Exposure computation
    # ------------------------------------------------------------------

    def compute_exposure(
        self,
        open_positions: list[dict],
        capital_paisa: int,
        trade_date: date,
    ) -> dict:
        """Compute portfolio gross/net exposure, beta weights, and breakdowns.

        Each element of *open_positions* must contain:
          - ``"symbol"`` (str): instrument key.
          - ``"market_value_paisa"`` (int): absolute market value in paisa.
          - ``"signal_type"`` (str): ``"BUY"`` / ``"LONG"`` → long; anything
            else treated as short.
          - ``"sector"`` (str, optional): sector label; defaults to ``"UNKNOWN"``.
          - ``"beta"`` (float, optional): pre-computed beta; defaults to 1.0.

        Results are persisted to ``portfolio_exposure_snapshot``.

        Args:
            open_positions: List of position dicts as described above.
            capital_paisa: Total trading capital in paisa.
            trade_date: Date stamp for the snapshot.

        Returns:
            Dict with keys:
              - gross_paisa (int)
              - net_paisa (int)
              - beta_weighted_gross (float)
              - beta_weighted_net (float)
              - sector_breakdown (dict str → float pct_of_capital)
              - symbol_breakdown (dict str → float pct_of_capital)
        """
        gross = 0
        net = 0
        beta_gross = 0.0
        beta_net = 0.0
        sector_breakdown: dict[str, float] = {}
        symbol_breakdown: dict[str, float] = {}

        for pos in open_positions:
            mv = abs(int(pos.get("market_value_paisa", 0)))
            sig = str(pos.get("signal_type", "BUY")).upper()
            is_long = sig in ("BUY", "LONG")
            sym = pos.get("symbol", "UNKNOWN")
            sector = pos.get("sector", "UNKNOWN")
            beta = float(pos.get("beta", 1.0))

            gross += mv
            net += mv if is_long else -mv
            beta_gross += beta * mv
            beta_net += beta * mv if is_long else -(beta * mv)

            if capital_paisa > 0:
                pct = mv / capital_paisa * 100.0
                sector_breakdown[sector] = sector_breakdown.get(sector, 0.0) + pct
                symbol_breakdown[sym] = symbol_breakdown.get(sym, 0.0) + pct

        result = {
            "gross_paisa": gross,
            "net_paisa": net,
            "beta_weighted_gross": beta_gross,
            "beta_weighted_net": beta_net,
            "sector_breakdown": sector_breakdown,
            "symbol_breakdown": symbol_breakdown,
        }

        self._persist_exposure_snapshot(result, capital_paisa, trade_date)
        return result

    def _persist_exposure_snapshot(
        self, exposure: dict, capital_paisa: int, trade_date: date
    ) -> None:
        """Insert a row into ``portfolio_exposure_snapshot``."""
        if self._engine is None:
            return

        from sqlalchemy import text
        import json

        sql = text(
            """
            INSERT INTO portfolio_exposure_snapshot
                (trade_date, gross_exposure_paisa, net_exposure_paisa,
                 beta_weighted_gross, beta_weighted_net,
                 sector_breakdown, symbol_breakdown, capital_paisa)
            VALUES
                (:trade_date, :gross, :net, :bwg, :bwn,
                 :sector_json, :symbol_json, :capital)
            ON CONFLICT (trade_date, ts) DO NOTHING
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sql,
                    {
                        "trade_date": trade_date,
                        "gross": exposure["gross_paisa"],
                        "net": exposure["net_paisa"],
                        "bwg": exposure["beta_weighted_gross"],
                        "bwn": exposure["beta_weighted_net"],
                        "sector_json": json.dumps(exposure["sector_breakdown"]),
                        "symbol_json": json.dumps(exposure["symbol_breakdown"]),
                        "capital": capital_paisa,
                    },
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Cap checks
    # ------------------------------------------------------------------

    def check_sector_cap(
        self,
        sector: str,
        additional_paisa: int,
        exposure: dict,
        capital_paisa: int,
    ) -> bool:
        """Return True if adding *additional_paisa* to *sector* stays under the cap.

        Args:
            sector: Sector label.
            additional_paisa: Proposed additional notional (paisa).
            exposure: Current exposure dict from :meth:`compute_exposure`.
            capital_paisa: Total capital (paisa).

        Returns:
            True = allowed, False = would breach sector cap.
        """
        if capital_paisa <= 0:
            return True
        current_pct = exposure.get("sector_breakdown", {}).get(sector, 0.0)
        add_pct = additional_paisa / capital_paisa * 100.0
        return (current_pct + add_pct) <= self._cfg.max_sector_exposure_pct

    def check_symbol_cap(
        self,
        symbol: str,
        additional_paisa: int,
        exposure: dict,
        capital_paisa: int,
    ) -> bool:
        """Return True if adding *additional_paisa* to *symbol* stays under the cap.

        Args:
            symbol: Instrument key.
            additional_paisa: Proposed additional notional (paisa).
            exposure: Current exposure dict from :meth:`compute_exposure`.
            capital_paisa: Total capital (paisa).

        Returns:
            True = allowed, False = would breach symbol cap.
        """
        if capital_paisa <= 0:
            return True
        current_pct = exposure.get("symbol_breakdown", {}).get(symbol, 0.0)
        add_pct = additional_paisa / capital_paisa * 100.0
        return (current_pct + add_pct) <= self._cfg.max_symbol_exposure_pct

    def check_leverage_cap(
        self,
        additional_paisa: int,
        signal_type: str,
        exposure: dict,
    ) -> bool:
        """Return True if adding *additional_paisa* keeps leverage within bounds.

        Checks both gross leverage (max_gross_leverage × capital) and net long
        leverage (max_net_long_leverage × capital).  Capital is inferred from
        the exposure dict (``gross_paisa / current_gross_leverage``).

        Because capital is not directly stored in *exposure*, the caller must
        use a snapshot that was created with a valid ``capital_paisa``.  We
        derive capital from the snapshot if possible, otherwise we compare
        absolute paisa values against a naive limit.

        Note: This method operates on the *exposure* dict directly, which
        does NOT include capital.  Use the ``can_add_position`` master gate
        which passes capital explicitly.

        Args:
            additional_paisa: Proposed notional (paisa, always positive).
            signal_type: ``"BUY"`` / ``"LONG"`` → long; else short.
            exposure: Current exposure dict from :meth:`compute_exposure`.

        Returns:
            True = allowed.
        """
        # This method is intentionally kept simple — the master gate supplies
        # capital.  Without capital we cannot compute leverage ratios here.
        # Callers should prefer check_leverage_cap_with_capital.
        return True

    def check_leverage_cap_with_capital(
        self,
        additional_paisa: int,
        signal_type: str,
        exposure: dict,
        capital_paisa: int,
    ) -> bool:
        """Leverage cap check with capital explicitly provided.

        Args:
            additional_paisa: Proposed notional (paisa).
            signal_type: ``"BUY"`` / ``"LONG"`` → long; else short.
            exposure: Current exposure dict.
            capital_paisa: Total trading capital (paisa).

        Returns:
            True = allowed, False = would breach gross or net cap.
        """
        if capital_paisa <= 0:
            return True

        new_gross = exposure.get("gross_paisa", 0) + additional_paisa
        is_long = str(signal_type).upper() in ("BUY", "LONG")
        new_net = exposure.get("net_paisa", 0) + (
            additional_paisa if is_long else -additional_paisa
        )

        gross_ok = new_gross <= self._cfg.max_gross_leverage * capital_paisa
        net_ok = new_net <= self._cfg.max_net_long_leverage * capital_paisa
        return gross_ok and net_ok

    # ------------------------------------------------------------------
    # VaR / CVaR
    # ------------------------------------------------------------------

    def compute_var(
        self,
        trade_date: date,
        capital_paisa: int,
        method: str = "historical",
        lookback_days: int | None = None,
    ) -> dict:
        """Compute 1-day VaR and CVaR at 95% and 99% confidence.

        Historical method: empirical quantile of daily portfolio PnL distribution.
        Returns the *loss* at each quantile — a positive number means money at
        risk (losses are stored as negative PnL, so VaR = |quantile|).

        Results are persisted to ``var_daily``.

        Args:
            trade_date: Reference date.
            capital_paisa: Total capital (paisa) for the snapshot.
            method: ``"historical"`` (others reserved for future).
            lookback_days: Override for ``var_lookback_days``.

        Returns:
            Dict with keys:
              - var_95_paisa (int)
              - var_99_paisa (int)
              - cvar_95_paisa (int)
              - cvar_99_paisa (int)
        """
        n_days = lookback_days or self._cfg.var_lookback_days
        pnl_series = self._fetch_portfolio_pnl(n_days, trade_date)

        result = self._compute_var_from_pnl(pnl_series)
        self._persist_var(trade_date, capital_paisa, result, method)
        return result

    @staticmethod
    def _compute_var_from_pnl(pnl_series: list[int]) -> dict:
        """Compute VaR/CVaR from a list of daily PnL values.

        Args:
            pnl_series: Daily PnL in paisa (positive = profit, negative = loss).

        Returns:
            Dict with var_95_paisa, var_99_paisa, cvar_95_paisa, cvar_99_paisa.
            All values are non-negative (amount at risk).
        """
        zero_result = {
            "var_95_paisa": 0,
            "var_99_paisa": 0,
            "cvar_95_paisa": 0,
            "cvar_99_paisa": 0,
        }
        if len(pnl_series) < 5:
            return zero_result

        # Sort ascending → worst losses are at the front
        sorted_pnl = sorted(pnl_series)
        n = len(sorted_pnl)

        def _var(confidence: float) -> int:
            # VaR at confidence c: loss at quantile (1-c).
            # Index = position of the worst loss inside the (1-c) tail.
            # Use round() to avoid float-precision issues (0.01*200 = 1.999...).
            tail_size = max(1, int(round((1.0 - confidence) * n)))
            idx = tail_size - 1  # last element of the tail (least extreme loss in it)
            val = sorted_pnl[idx]
            return max(0, -val)  # flip sign: loss = positive

        def _cvar(confidence: float) -> int:
            # CVaR: mean of the (1-c) worst-loss tail. Same tail size as VaR.
            cutoff_idx = max(1, int(round((1.0 - confidence) * n)))
            tail = sorted_pnl[:cutoff_idx]
            if not tail:
                return 0
            mean_loss = -sum(tail) / len(tail)
            return max(0, int(mean_loss))

        return {
            "var_95_paisa": _var(0.95),
            "var_99_paisa": _var(0.99),
            "cvar_95_paisa": _cvar(0.95),
            "cvar_99_paisa": _cvar(0.99),
        }

    def _persist_var(
        self, trade_date: date, capital_paisa: int, var_result: dict, method: str
    ) -> None:
        """Upsert a row into ``var_daily``."""
        if self._engine is None:
            return

        from sqlalchemy import text

        sql = text(
            """
            INSERT INTO var_daily
                (trade_date, capital_paisa, var_95_paisa, var_99_paisa,
                 cvar_95_paisa, cvar_99_paisa, method)
            VALUES
                (:trade_date, :capital, :v95, :v99, :cv95, :cv99, :method)
            ON CONFLICT (trade_date)
            DO UPDATE SET
                capital_paisa  = EXCLUDED.capital_paisa,
                var_95_paisa   = EXCLUDED.var_95_paisa,
                var_99_paisa   = EXCLUDED.var_99_paisa,
                cvar_95_paisa  = EXCLUDED.cvar_95_paisa,
                cvar_99_paisa  = EXCLUDED.cvar_99_paisa,
                method         = EXCLUDED.method,
                computed_at    = NOW()
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    sql,
                    {
                        "trade_date": trade_date,
                        "capital": capital_paisa,
                        "v95": var_result["var_95_paisa"],
                        "v99": var_result["var_99_paisa"],
                        "cv95": var_result["cvar_95_paisa"],
                        "cv99": var_result["cvar_99_paisa"],
                        "method": method,
                    },
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Master gate
    # ------------------------------------------------------------------

    def can_add_position(
        self,
        symbol: str,
        sector: str,
        signal_type: str,
        proposed_paisa: int,
        open_positions: list[dict],
        capital_paisa: int,
        trade_date: date,
    ) -> tuple[bool, str | None]:
        """Master gate: decide if a new position can be added to the portfolio.

        Runs four checks in order; returns at the first failure:
          1. SYMBOL_CAP  — proposed exceeds per-symbol cap.
          2. SECTOR_CAP  — proposed exceeds per-sector cap.
          3. LEVERAGE_CAP — proposed would exceed gross or net leverage cap.
          4. CORRELATION_CAP — too many correlated positions already open.

        Args:
            symbol: Instrument key of the proposed position.
            sector: Sector of the proposed instrument.
            signal_type: ``"BUY"`` / ``"LONG"`` for long; else short.
            proposed_paisa: Notional value of the proposed position (paisa).
            open_positions: Current open portfolio (list of position dicts).
            capital_paisa: Total trading capital (paisa).
            trade_date: Reference date for correlation lookup.

        Returns:
            ``(True, None)`` when all checks pass.
            ``(False, reason)`` with reason in
            ``{"SYMBOL_CAP", "SECTOR_CAP", "LEVERAGE_CAP", "CORRELATION_CAP"}``.
        """
        exposure = self.compute_exposure(open_positions, capital_paisa, trade_date)

        # 1. Symbol cap
        if not self.check_symbol_cap(symbol, proposed_paisa, exposure, capital_paisa):
            return False, "SYMBOL_CAP"

        # 2. Sector cap
        if not self.check_sector_cap(sector, proposed_paisa, exposure, capital_paisa):
            return False, "SECTOR_CAP"

        # 3. Leverage cap
        if not self.check_leverage_cap_with_capital(
            proposed_paisa, signal_type, exposure, capital_paisa
        ):
            return False, "LEVERAGE_CAP"

        # 4. Correlation cap
        correlated = self.find_correlated_group(
            symbol, open_positions, trade_date
        )
        # correlated includes existing positions; count the group size (add +1 for the new one)
        if len(correlated) + 1 >= self._cfg.correlation_max_grouped:
            return False, "CORRELATION_CAP"

        return True, None
