"""Macro overlay: USDINR + commodities regime classification + cross-asset signals.

Subscribes to four macro instruments (USDINR currency futures, Crude/Gold/Silver
commodity futures on NSE_COM), computes daily return and z-score metrics, classifies
a regime trend (UP / DOWN / RANGE), and exposes cross-asset signal helpers that map
macro moves to individual equity impacts.

All price values are stored in paisa (int).

Stock-to-macro impact mapping (hard-coded for Nifty 50 Top-10):
  - TCS, INFY       → USDINR UP  → bullish  (IT exporters benefit from rupee weakness)
  - RELIANCE         → CRUDE UP   → bullish  (energy / refining margin benefit)
  - HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK, SBIN
                     → USDINR UP  → bearish  (currency depreciation = imported inflation
                                              = rate-hike risk = higher cost of funds)
  - HINDUNILVR, ITC  → CRUDE UP   → bearish  (input cost pressure on FMCG / paints)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static macro instrument catalogue
# ---------------------------------------------------------------------------

_MACRO_INSTRUMENTS: list[dict[str, str]] = [
    {
        "symbol": "USDINR",
        "instrument_key": "NCD_FO|7009",
        "category": "currency",
        "segment": "NCD_FO",
        "display_name": "USD/INR Front-Month Futures",
    },
    {
        "symbol": "CRUDE",
        "instrument_key": "NSE_COM|121620",
        "category": "commodity",
        "segment": "NSE_COM",
        "display_name": "Crude Oil Front-Month Futures",
    },
    {
        "symbol": "GOLD",
        "instrument_key": "NSE_COM|122886",
        "category": "commodity",
        "segment": "NSE_COM",
        "display_name": "Gold Front-Month Futures",
    },
    {
        "symbol": "SILVER",
        "instrument_key": "NSE_COM|121799",
        "category": "commodity",
        "segment": "NSE_COM",
        "display_name": "Silver Front-Month Futures",
    },
]

# ---------------------------------------------------------------------------
# Cross-asset impact mapping
# ---------------------------------------------------------------------------

# Maps stock_symbol → list of (macro_symbol, macro_direction, equity_impact, reason)
_CROSS_ASSET_MAP: dict[str, list[tuple[str, str, str, str]]] = {
    "TCS": [
        ("USDINR", "UP", "bullish",
         "USDINR uptrend: rupee weakness boosts IT export revenues (USD-denominated)"),
    ],
    "INFY": [
        ("USDINR", "UP", "bullish",
         "USDINR uptrend: rupee weakness boosts IT export revenues (USD-denominated)"),
    ],
    "RELIANCE": [
        ("CRUDE", "UP", "bullish",
         "Crude uptrend: refining margins widen, Reliance Energy segment benefits"),
    ],
    "HDFCBANK": [
        ("USDINR", "UP", "bearish",
         "USDINR uptrend: rupee depreciation → imported inflation → rate-hike risk → margin squeeze"),
    ],
    "ICICIBANK": [
        ("USDINR", "UP", "bearish",
         "USDINR uptrend: currency depreciation raises rate-hike risk for banks"),
    ],
    "AXISBANK": [
        ("USDINR", "UP", "bearish",
         "USDINR uptrend: currency depreciation raises rate-hike risk for banks"),
    ],
    "KOTAKBANK": [
        ("USDINR", "UP", "bearish",
         "USDINR uptrend: currency depreciation raises rate-hike risk for banks"),
    ],
    "SBIN": [
        ("USDINR", "UP", "bearish",
         "USDINR uptrend: currency depreciation raises rate-hike risk for PSU banks"),
    ],
    "HINDUNILVR": [
        ("CRUDE", "UP", "bearish",
         "Crude uptrend: crude-linked input costs rise, compressing FMCG margins"),
    ],
    "ITC": [
        ("CRUDE", "UP", "bearish",
         "Crude uptrend: packaging/logistics costs rise, headwind for FMCG/agri segments"),
    ],
}


# ---------------------------------------------------------------------------
# MacroOverlay
# ---------------------------------------------------------------------------

class MacroOverlay:
    """Compute and expose macro regime classification for USDINR, Crude, Gold, Silver.

    Args:
        instrument_manager: An :class:`~src.data.instruments.InstrumentManager` instance
            (used for validation/lookup; not required for pure computation).
        historical_provider: Callable ``(instrument_key: str, interval: str, days: int)``
            → ``pd.DataFrame`` with at minimum a ``close`` column (prices in **paisa**).
            The returned DataFrame must have a DatetimeIndex.
        db_engine: Optional SQLAlchemy engine for persistence.  When ``None``, the
            ``update_daily`` method skips persistence and only returns the computed df.
    """

    def __init__(
        self,
        instrument_manager: Any = None,
        historical_provider: Any = None,
        db_engine: Any = None,
    ) -> None:
        self._im = instrument_manager
        self._hp = historical_provider
        self._engine = db_engine

    # ------------------------------------------------------------------
    # Instrument catalogue
    # ------------------------------------------------------------------

    def get_active_instruments(self) -> list[dict]:
        """Return the list of active macro instruments.

        Returns:
            List of dicts with keys: symbol, instrument_key, category, segment,
            display_name.
        """
        return list(_MACRO_INSTRUMENTS)

    def get_macro_keys(self) -> list[str]:
        """Return instrument_key strings for all active macro instruments.

        Returns:
            List of Upstox instrument_key strings.
        """
        return [inst["instrument_key"] for inst in _MACRO_INSTRUMENTS]

    # ------------------------------------------------------------------
    # Computation helpers
    # ------------------------------------------------------------------

    def _fetch_closes(self, instrument_key: str, lookback_days: int) -> pd.Series:
        """Fetch close prices (paisa) for the given instrument.

        Args:
            instrument_key: Upstox instrument key.
            lookback_days: Number of trading days to fetch.

        Returns:
            pd.Series of close prices (paisa, int) indexed by timestamp.

        Raises:
            ValueError: If the historical provider returns empty data.
        """
        if self._hp is None:
            raise ValueError("historical_provider not set — cannot fetch prices")

        df = self._hp(instrument_key, "day", lookback_days)
        if df is None or df.empty:
            raise ValueError(
                f"No historical data returned for instrument_key={instrument_key!r}"
            )
        closes = df["close"].dropna()
        if closes.empty:
            raise ValueError(
                f"Close column is empty for instrument_key={instrument_key!r}"
            )
        return closes.astype(float)

    def compute_returns(self, instrument_key: str, lookback_days: int) -> float:
        """Compute percentage return over the last *lookback_days* trading days.

        The return is defined as ``(close[-1] / close[-lookback_days-1] - 1) * 100``.

        Args:
            instrument_key: Upstox instrument key.
            lookback_days: Look-back window in trading days (e.g. 5 for weekly).

        Returns:
            Percentage return as a float (e.g. 2.5 means +2.5 %).
        """
        required = lookback_days + 5  # fetch a buffer
        closes = self._fetch_closes(instrument_key, required)
        if len(closes) < lookback_days + 1:
            logger.warning(
                "Insufficient data for return calculation (need %d, got %d) key=%s",
                lookback_days + 1,
                len(closes),
                instrument_key,
            )
            return 0.0
        start_price = closes.iloc[-(lookback_days + 1)]
        end_price = closes.iloc[-1]
        if start_price == 0:
            return 0.0
        return float((end_price / start_price - 1.0) * 100.0)

    def compute_zscore(self, instrument_key: str, lookback: int = 20) -> float:
        """Compute z-score of the latest close relative to a rolling window.

        z-score = (close_today - mean_20d) / stdev_20d.

        Args:
            instrument_key: Upstox instrument key.
            lookback: Rolling window length in trading days (default 20).

        Returns:
            Z-score as a float.  Returns 0.0 when standard deviation is zero or
            data is insufficient.
        """
        required = lookback + 5
        closes = self._fetch_closes(instrument_key, required)
        if len(closes) < lookback:
            logger.warning(
                "Insufficient data for z-score (need %d, got %d) key=%s",
                lookback,
                len(closes),
                instrument_key,
            )
            return 0.0
        window = closes.iloc[-lookback:]
        mean = float(window.mean())
        std = float(window.std(ddof=1))
        if std == 0.0:
            return 0.0
        return float((closes.iloc[-1] - mean) / std)

    def classify_trend(
        self,
        return_5d: float,
        zscore: float,
        threshold: float = 0.5,
    ) -> str:
        """Classify macro trend based on 5-day return and z-score.

        Rules:
          - ``'UP'``    if return_5d > +1% AND zscore > +threshold
          - ``'DOWN'``  if return_5d < -1% AND zscore < -threshold
          - ``'RANGE'`` otherwise

        Args:
            return_5d: 5-day percentage return.
            zscore: 20-day z-score of the latest close.
            threshold: Z-score magnitude threshold (default 0.5).

        Returns:
            One of ``'UP'``, ``'DOWN'``, ``'RANGE'``.
        """
        if return_5d > 1.0 and zscore > threshold:
            return "UP"
        if return_5d < -1.0 and zscore < -threshold:
            return "DOWN"
        return "RANGE"

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------

    def update_daily(self, trade_date: date | datetime | None = None) -> pd.DataFrame:
        """Compute regime metrics for each macro instrument and persist.

        For each active instrument, fetches historical closes, computes 1d/5d/20d
        returns, 20d z-score, and trend classification.  Results are upserted into
        ``macro_regime_daily`` when a ``db_engine`` is provided.

        Args:
            trade_date: The trade date to record.  Defaults to today.

        Returns:
            DataFrame with columns:
            ``symbol``, ``trade_date``, ``close_paisa``, ``return_pct_1d``,
            ``return_pct_5d``, ``return_pct_20d``, ``trend``, ``zscore_20d``.
        """
        if trade_date is None:
            trade_date = date.today()
        if isinstance(trade_date, datetime):
            trade_date = trade_date.date()

        rows: list[dict] = []
        for inst in _MACRO_INSTRUMENTS:
            symbol = inst["symbol"]
            key = inst["instrument_key"]
            try:
                closes = self._fetch_closes(key, 30)
                close_today = int(closes.iloc[-1])

                ret_1d = self._pct_return(closes, 1)
                ret_5d = self._pct_return(closes, 5)
                ret_20d = self._pct_return(closes, 20)
                zscore = self._zscore_from_series(closes, 20)
                trend = self.classify_trend(ret_5d, zscore)

                rows.append({
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "close_paisa": close_today,
                    "return_pct_1d": round(ret_1d, 4),
                    "return_pct_5d": round(ret_5d, 4),
                    "return_pct_20d": round(ret_20d, 4),
                    "trend": trend,
                    "zscore_20d": round(zscore, 4),
                })
            except Exception as exc:
                logger.error("Failed to compute regime for %s: %s", symbol, exc)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        if self._engine is not None:
            self._persist(df)

        return df

    def _pct_return(self, closes: pd.Series, days: int) -> float:
        """Compute percentage return over *days* trading days."""
        if len(closes) < days + 1:
            return 0.0
        start = float(closes.iloc[-(days + 1)])
        end = float(closes.iloc[-1])
        if start == 0:
            return 0.0
        return (end / start - 1.0) * 100.0

    def _zscore_from_series(self, closes: pd.Series, lookback: int = 20) -> float:
        """Compute z-score of latest close within the rolling window."""
        if len(closes) < lookback:
            return 0.0
        window = closes.iloc[-lookback:]
        mean = float(window.mean())
        std = float(window.std(ddof=1))
        if std == 0.0:
            return 0.0
        return float((closes.iloc[-1] - mean) / std)

    def _persist(self, df: pd.DataFrame) -> None:
        """Upsert rows into macro_regime_daily table.

        Args:
            df: DataFrame from ``update_daily``.
        """
        try:
            from sqlalchemy import text

            upsert_sql = text("""
                INSERT INTO macro_regime_daily
                    (trade_date, symbol, close_paisa,
                     return_pct_1d, return_pct_5d, return_pct_20d,
                     trend, zscore_20d)
                VALUES
                    (:trade_date, :symbol, :close_paisa,
                     :return_pct_1d, :return_pct_5d, :return_pct_20d,
                     :trend, :zscore_20d)
                ON CONFLICT (trade_date, symbol) DO UPDATE SET
                    close_paisa    = EXCLUDED.close_paisa,
                    return_pct_1d  = EXCLUDED.return_pct_1d,
                    return_pct_5d  = EXCLUDED.return_pct_5d,
                    return_pct_20d = EXCLUDED.return_pct_20d,
                    trend          = EXCLUDED.trend,
                    zscore_20d     = EXCLUDED.zscore_20d
            """)
            with self._engine.begin() as conn:
                for row in df.itertuples(index=False):
                    conn.execute(
                        upsert_sql,
                        {
                            "trade_date": row.trade_date,
                            "symbol": row.symbol,
                            "close_paisa": int(row.close_paisa),
                            "return_pct_1d": float(row.return_pct_1d),
                            "return_pct_5d": float(row.return_pct_5d),
                            "return_pct_20d": float(row.return_pct_20d),
                            "trend": row.trend,
                            "zscore_20d": float(row.zscore_20d),
                        },
                    )
            logger.info(
                "macro_regime_daily: upserted %d rows for %s", len(df), df["trade_date"].iloc[0]
            )
        except Exception as exc:
            logger.error("Failed to persist macro regime rows: %s", exc)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_today_regime(self, trade_date: date | datetime | None = None) -> pd.DataFrame:
        """Fetch the macro regime for the given date from the DB.

        Args:
            trade_date: The date to query.  Defaults to today.

        Returns:
            DataFrame with columns: symbol, trend, zscore_20d, return_pct_5d, etc.
            Empty DataFrame when no data found or no engine configured.
        """
        if trade_date is None:
            trade_date = date.today()
        if isinstance(trade_date, datetime):
            trade_date = trade_date.date()

        if self._engine is None:
            logger.warning("No db_engine configured — returning empty regime")
            return pd.DataFrame()

        try:
            from sqlalchemy import text

            sql = text("""
                SELECT symbol, close_paisa, return_pct_1d, return_pct_5d,
                       return_pct_20d, trend, zscore_20d
                FROM macro_regime_daily
                WHERE trade_date = :trade_date
                ORDER BY symbol
            """)
            with self._engine.connect() as conn:
                result = conn.execute(sql, {"trade_date": trade_date})
                rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=result.keys())
        except Exception as exc:
            logger.error("Failed to fetch macro regime for %s: %s", trade_date, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Cross-asset signals
    # ------------------------------------------------------------------

    def cross_asset_signal_for_stock(
        self,
        stock_symbol: str,
        trade_date: date | datetime | None = None,
    ) -> dict:
        """Return cross-asset signal (bullish/bearish/neutral) for a stock.

        Looks up the stock in the hard-coded impact mapping, then fetches the
        current macro trend from the DB (or computes in-memory if no DB).

        Args:
            stock_symbol: NSE equity symbol, e.g. ``"TCS"``, ``"RELIANCE"``.
            trade_date: Date to evaluate.  Defaults to today.

        Returns:
            Dict with keys:
            - ``direction``: ``'bullish'``, ``'bearish'``, or ``'neutral'``
            - ``reasons``: list of explanation strings
        """
        if trade_date is None:
            trade_date = date.today()
        if isinstance(trade_date, datetime):
            trade_date = trade_date.date()

        symbol_upper = stock_symbol.upper()
        mapping = _CROSS_ASSET_MAP.get(symbol_upper, [])
        if not mapping:
            return {"direction": "neutral", "reasons": [f"No macro mapping for {symbol_upper}"]}

        # Fetch regime from DB if possible
        regime_df = self.get_today_regime(trade_date)
        regime_map: dict[str, str] = {}
        if not regime_df.empty:
            regime_map = dict(zip(regime_df["symbol"], regime_df["trend"]))

        bullish_reasons: list[str] = []
        bearish_reasons: list[str] = []

        for macro_sym, macro_direction, equity_impact, reason in mapping:
            current_trend = regime_map.get(macro_sym, "RANGE")
            if current_trend == macro_direction:
                if equity_impact == "bullish":
                    bullish_reasons.append(reason)
                elif equity_impact == "bearish":
                    bearish_reasons.append(reason)

        if bullish_reasons and not bearish_reasons:
            return {"direction": "bullish", "reasons": bullish_reasons}
        if bearish_reasons and not bullish_reasons:
            return {"direction": "bearish", "reasons": bearish_reasons}
        if bullish_reasons and bearish_reasons:
            return {
                "direction": "neutral",
                "reasons": bullish_reasons + bearish_reasons,
            }
        return {
            "direction": "neutral",
            "reasons": [
                f"No triggered macro signal for {symbol_upper} "
                f"(current trends: {regime_map})"
            ],
        }
