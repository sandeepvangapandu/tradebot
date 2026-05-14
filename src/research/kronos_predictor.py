"""Kronos foundation-model wrapper for trading bot.

Loads pre-trained Kronos model from HuggingFace, generates K-line forecasts,
and persists every prediction to Postgres for shadow-mode validation.

This is SHADOW MODE only — no orders routed through Kronos predictions.
Phase G.5 decision gate evaluates after 4-week shadow run.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from loguru import logger

from src.research.kronos_lib.predictor import KronosPredictor
from src.research.kronos_lib.model import Kronos
from src.research.kronos_lib.tokenizer import KronosTokenizer

IST = ZoneInfo("Asia/Kolkata")


class KronosForecaster:
    """Wraps the Kronos foundation model for financial K-line forecasting.

    Loads model weights lazily on first call to load() or predict(). All money
    values are stored in paisa (integer). Predictions are persisted to the
    ``kronos_forecasts`` Postgres table when a db_engine is provided.

    Args:
        model_size: One of 'mini', 'small', 'base'. Default 'small' (24.7M params).
        device: PyTorch device string. 'cpu' works for all model sizes.
        max_context: Maximum number of input bars fed to the model (capped at model limit).
        db_engine: SQLAlchemy engine for persisting forecasts. Optional — if None,
            predictions are returned but not stored.
    """

    def __init__(
        self,
        model_size: str = "small",                    # 'mini' | 'small' | 'base'
        device: str = "cpu",
        max_context: int = 512,
        db_engine=None,
    ):
        self.model_size = model_size
        self.device = device
        self.max_context = max_context
        self._engine = db_engine
        self._model = None
        self._tokenizer = None
        self._predictor = None
        self._model_version = f"NeoQuasar/Kronos-{model_size}"

    def load(self) -> None:
        """Lazy-load model + tokenizer from HuggingFace. Idempotent."""
        if self._predictor is not None:
            return
        logger.info("Loading Kronos-{} model...", self.model_size)
        t0 = time.time()
        tok_name = f"NeoQuasar/Kronos-Tokenizer-base"
        model_name = f"NeoQuasar/Kronos-{self.model_size}"
        self._tokenizer = KronosTokenizer.from_pretrained(tok_name)
        self._model = Kronos.from_pretrained(model_name)
        self._predictor = KronosPredictor(
            self._model, self._tokenizer,
            device=self.device,
            max_context=self.max_context,
        )
        logger.info("Kronos loaded in {:.1f}s", time.time() - t0)

    def predict(
        self,
        bars: pd.DataFrame,                           # columns: open/high/low/close/volume/amount; index = timestamp
        horizon: int = 30,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        """Generate forecast for next `horizon` bars.

        Returns DataFrame with same columns as input, indexed by future timestamps.

        Args:
            bars: Historical OHLCV bars. Must have columns open/high/low/close.
                  volume and amount are synthesized if missing.
            horizon: Number of future bars to forecast.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling threshold (0.9 recommended).
            sample_count: Number of parallel samples averaged for the final prediction.

        Returns:
            DataFrame with columns open/high/low/close/volume/amount, indexed by
            future timestamps.
        """
        self.load()
        # Validate input
        required_cols = {"open", "high", "low", "close", "volume", "amount"}
        missing = required_cols - set(bars.columns)
        if missing:
            # Synthesize 'amount' from close*volume if missing (NSE doesn't always provide)
            if "amount" in missing and "close" in bars.columns and "volume" in bars.columns:
                bars = bars.copy()
                bars["amount"] = bars["close"] * bars["volume"]
                missing.discard("amount")
        if missing:
            raise ValueError(f"Bars missing required columns: {missing}")

        if len(bars) > self.max_context:
            bars = bars.tail(self.max_context)

        # Construct future timestamp index
        if isinstance(bars.index, pd.DatetimeIndex):
            tf = bars.index[-1] - bars.index[-2] if len(bars) >= 2 else pd.Timedelta(minutes=1)
            future_ts = pd.date_range(start=bars.index[-1] + tf, periods=horizon, freq=tf)
        else:
            future_ts = pd.RangeIndex(len(bars), len(bars) + horizon)

        # Kronos predictor expects pd.Series with .dt accessor (not DatetimeIndex)
        x_ts = pd.Series(bars.index) if isinstance(bars.index, pd.DatetimeIndex) else None
        if isinstance(future_ts, pd.DatetimeIndex):
            y_ts = pd.Series(future_ts)
        else:
            y_ts = pd.Series(list(future_ts))

        forecast = self._predictor.predict(
            df=bars,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=horizon,
            T=temperature, top_p=top_p, sample_count=sample_count,
        )
        return forecast

    def predict_summary(
        self,
        instrument_key: str,
        bars: pd.DataFrame,
        timeframe: str = "5m",
        horizon: int = 12,
        persist: bool = True,
    ) -> dict:
        """Run prediction + compute summary metrics + optionally persist to DB.

        Returns dict with predicted_direction/change_pct/range_pct/etc.

        Args:
            instrument_key: Exchange instrument key (e.g. 'NSE_EQ|INE002A01018').
            bars: Historical OHLCV bars DataFrame.
            timeframe: Candle timeframe label (e.g. '5m', '15m', '1h').
            horizon: Bars to forecast.
            persist: If True and db_engine is set, insert into kronos_forecasts.

        Returns:
            dict with keys: ts, instrument_key, model_size, timeframe,
            context_bars, horizon_bars, current_close_paisa,
            predicted_close_paisa, predicted_high_paisa, predicted_low_paisa,
            predicted_direction, predicted_change_pct, predicted_range_pct,
            raw_forecast, inference_ms, model_version.
            Returns empty dict on model failure.
        """
        t0 = time.time()
        try:
            forecast = self.predict(bars, horizon=horizon)
        except Exception as exc:
            logger.warning("Kronos predict failed for {}: {}", instrument_key, exc)
            return {}
        inference_ms = int((time.time() - t0) * 1000)

        current_close = float(bars["close"].iloc[-1])
        pred_close = float(forecast["close"].iloc[-1])
        pred_high = float(forecast["high"].max())
        pred_low = float(forecast["low"].min())
        change_pct = (pred_close - current_close) / current_close * 100.0
        range_pct = (pred_high - pred_low) / current_close * 100.0

        if abs(change_pct) < 0.1:
            direction = "FLAT"
        elif change_pct > 0:
            direction = "UP"
        else:
            direction = "DOWN"

        summary = {
            "ts": datetime.now(tz=IST),
            "instrument_key": instrument_key,
            "model_size": self.model_size,
            "timeframe": timeframe,
            "context_bars": len(bars),
            "horizon_bars": horizon,
            "current_close_paisa": int(round(current_close)),
            "predicted_close_paisa": int(round(pred_close)),
            "predicted_high_paisa": int(round(pred_high)),
            "predicted_low_paisa": int(round(pred_low)),
            "predicted_direction": direction,
            "predicted_change_pct": change_pct,
            "predicted_range_pct": range_pct,
            "raw_forecast": forecast.to_dict(orient="records"),
            "inference_ms": inference_ms,
            "model_version": self._model_version,
        }

        if persist and self._engine is not None:
            self._persist(summary)

        return summary

    def _persist(self, summary: dict) -> None:
        """Insert into kronos_forecasts. Swallows DB errors to avoid crashing shadow loop."""
        from sqlalchemy import text
        sql = text("""
            INSERT INTO kronos_forecasts
              (ts, instrument_key, model_size, timeframe, context_bars, horizon_bars,
               current_close_paisa, predicted_close_paisa, predicted_high_paisa,
               predicted_low_paisa, predicted_direction, predicted_change_pct,
               predicted_range_pct, raw_forecast, inference_ms, model_version)
            VALUES
              (:ts, :ikey, :msize, :tf, :ctx, :hrz, :cur, :pc, :ph, :pl, :pd, :pcg, :prg,
               CAST(:raw AS jsonb), :ims, :mver)
        """)
        try:
            import json as _json
            with self._engine.begin() as conn:
                conn.execute(sql, {
                    "ts": summary["ts"],
                    "ikey": summary["instrument_key"],
                    "msize": summary["model_size"],
                    "tf": summary["timeframe"],
                    "ctx": summary["context_bars"],
                    "hrz": summary["horizon_bars"],
                    "cur": summary["current_close_paisa"],
                    "pc": summary["predicted_close_paisa"],
                    "ph": summary["predicted_high_paisa"],
                    "pl": summary["predicted_low_paisa"],
                    "pd": summary["predicted_direction"],
                    "pcg": summary["predicted_change_pct"],
                    "prg": summary["predicted_range_pct"],
                    "raw": _json.dumps(summary["raw_forecast"], default=str),
                    "ims": summary["inference_ms"],
                    "mver": summary["model_version"],
                })
        except Exception as exc:
            logger.warning("Kronos persist failed: {}", exc)

    def predicted_direction(self, instrument_key: str, bars: pd.DataFrame, horizon: int = 12) -> str:
        """Convenience accessor — returns 'UP'/'DOWN'/'FLAT' without persisting.

        Args:
            instrument_key: Exchange instrument key.
            bars: Historical OHLCV bars.
            horizon: Bars ahead to forecast.

        Returns:
            One of 'UP', 'DOWN', 'FLAT'.
        """
        s = self.predict_summary(instrument_key, bars, horizon=horizon, persist=False)
        return s.get("predicted_direction", "FLAT")
