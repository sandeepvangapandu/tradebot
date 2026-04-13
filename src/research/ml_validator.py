"""ML-based Signal Validator for trading signal confidence scoring.

Uses machine learning to validate trading signals and provide confidence scores.
The ML model learns from past trade outcomes to improve accuracy over time.

All monetary values are in PAISA (integer) - 1 Rupee = 100 paisa.
All timestamps are in IST (Asia/Kolkata).
"""

from __future__ import annotations

import json
import os
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

# Optional sklearn import - ML validator works in fallback mode without it
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GradientBoostingClassifier = None  # type: ignore
    RandomForestClassifier = None  # type: ignore
    StandardScaler = None  # type: ignore

from src.research.models import AnalysisComponent
from src.strategy.indicators import IndicatorEngine
from src.utils.degradation import graceful_degrade

IST = ZoneInfo("Asia/Kolkata")

# Feature names for the ML model
FEATURE_NAMES = [
    # Price action features
    "return_5min",
    "return_15min",
    "return_1h",
    "volatility_20",
    "momentum_10",
    # Technical indicators
    "rsi",
    "rsi_slope",
    "macd_histogram",
    "ema_slope_9",
    "ema_slope_21",
    "atr_percentile",
    # Market regime
    "adx",
    "trend_direction",
    "volatility_regime",
    # Volume profile
    "relative_volume",
    "vwap_distance_pct",
    # Time features
    "hour_of_day",
    "day_of_week",
    "minutes_since_open",
]

# Minimum samples required before training
MIN_SAMPLES_FOR_TRAINING = 100

# Default model path
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_TRAINING_LOG_PATH = Path("data/ml_training_log.jsonl")


class MLSignalValidator:
    """Machine Learning based signal validator.

    Uses a RandomForest or GradientBoosting classifier to predict the
    probability of a trade being profitable based on market conditions
    at the time of signal generation.

    Features are extracted from price action, technical indicators,
    market regime, volume profile, and time-based factors.

    Attributes:
        model: The sklearn classifier model
        scaler: StandardScaler for feature normalization
        model_path: Path to saved model file
        training_log_path: Path to training data log
        feature_names: List of feature names
        _lock: Thread lock for model predictions
        is_trained: Whether model has been trained
        model_version: Version string for model
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        training_log_path: Optional[str] = None,
        model_type: str = "random_forest",
    ) -> None:
        """Initialize the ML Signal Validator.

        Args:
            model_path: Path to load existing model from. If None and file
                doesn't exist, a new model will be created.
            training_log_path: Path to training log file (jsonl format).
            model_type: Type of model to use - "random_forest" or "gradient_boosting".
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_DIR / "signal_validator.pkl"
        self.training_log_path = Path(training_log_path) if training_log_path else DEFAULT_TRAINING_LOG_PATH
        self.model_type = model_type
        self.feature_names = FEATURE_NAMES
        self.is_trained = False
        self.model_version = "1.0.0"
        self._lock = threading.Lock()
        self._training_buffer: list[dict] = []
        self._sklearn_available = SKLEARN_AVAILABLE

        # Initialize model and scaler
        self.model: Optional[Any] = None
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
            logger.warning("sklearn not available - ML validator running in fallback mode")

        # Try to load existing model
        if self.model_path.exists():
            self._load_model()
        else:
            self._create_new_model()

        # Ensure training log directory exists
        self.training_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_new_model(self) -> None:
        """Create a new untrained model."""
        if not SKLEARN_AVAILABLE:
            self.model = None
            self.is_trained = False
            return

        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        self.is_trained = False
        logger.info(f"Created new {self.model_type} model")

    def _load_model(self) -> None:
        """Load model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                saved_data = pickle.load(f)

            self.model = saved_data["model"]
            self.scaler = saved_data["scaler"]
            self.model_version = saved_data.get("version", "1.0.0")
            self.is_trained = saved_data.get("is_trained", False)
            self.feature_names = saved_data.get("feature_names", FEATURE_NAMES)

            logger.info(f"Loaded ML model version {self.model_version} from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            self._create_new_model()

    def save_model(self, path: Optional[str] = None) -> None:
        """Save model to disk.

        Args:
            path: Path to save model. Uses self.model_path if None.
        """
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            saved_data = {
                "model": self.model,
                "scaler": self.scaler,
                "version": self.model_version,
                "is_trained": self.is_trained,
                "feature_names": self.feature_names,
                "saved_at": datetime.now(IST).isoformat(),
            }

            with open(save_path, "wb") as f:
                pickle.dump(saved_data, f)

            logger.info(f"Saved ML model version {self.model_version} to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {save_path}: {e}")
            raise

    def extract_features(
        self,
        data: pd.DataFrame,
        signal: dict[str, Any],
    ) -> np.ndarray:
        """Extract feature vector from market data and signal.

        Args:
            data: OHLCV DataFrame with at least 50 rows of data.
            signal: Signal dictionary with keys like:
                - type: "BUY_CE", "SELL_CE", etc.
                - entry_price: Entry price in paisa
                - timestamp: Signal timestamp

        Returns:
            Feature vector as numpy array of shape (n_features,).
        """
        if len(data) < 50:
            logger.warning(f"Insufficient data for feature extraction: {len(data)} rows")
            # Return zeros with correct shape
            return np.zeros(len(self.feature_names))

        engine = IndicatorEngine(data)
        features: dict[str, float] = {}

        # Price action features
        close = data["close"].values
        returns = pd.Series(close).pct_change()

        features["return_5min"] = returns.iloc[-5:].sum() if len(returns) >= 5 else 0.0
        features["return_15min"] = returns.iloc[-15:].sum() if len(returns) >= 15 else 0.0
        features["return_1h"] = returns.iloc[-60:].sum() if len(returns) >= 60 else features["return_15min"]

        # Volatility (20-period rolling std of returns)
        features["volatility_20"] = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0.0

        # Momentum (10-period rate of change)
        features["momentum_10"] = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0.0

        # Technical indicators
        rsi_series = engine.rsi(period=14)
        features["rsi"] = rsi_series.iloc[-1] if not rsi_series.empty else 50.0

        # RSI slope (change over last 5 periods)
        if len(rsi_series) >= 6:
            features["rsi_slope"] = rsi_series.iloc[-1] - rsi_series.iloc[-6]
        else:
            features["rsi_slope"] = 0.0

        # MACD histogram
        try:
            macd_result = engine.macd(fast=12, slow=26, signal=9)
            if macd_result is not None and "histogram" in macd_result:
                features["macd_histogram"] = macd_result["histogram"].iloc[-1]
            else:
                features["macd_histogram"] = 0.0
        except Exception:
            features["macd_histogram"] = 0.0

        # EMA slopes
        ema_9 = engine.ema(period=9)
        ema_21 = engine.ema(period=21)

        if len(ema_9) >= 6:
            features["ema_slope_9"] = (ema_9.iloc[-1] - ema_9.iloc[-6]) / ema_9.iloc[-6] if ema_9.iloc[-6] != 0 else 0.0
        else:
            features["ema_slope_9"] = 0.0

        if len(ema_21) >= 6:
            features["ema_slope_21"] = (ema_21.iloc[-1] - ema_21.iloc[-6]) / ema_21.iloc[-6] if ema_21.iloc[-6] != 0 else 0.0
        else:
            features["ema_slope_21"] = 0.0

        # ATR percentile
        try:
            atr_series = engine.atr(period=14)
            if len(atr_series) >= 20:
                atr_current = atr_series.iloc[-1]
                atr_20 = atr_series.iloc[-20:]
                features["atr_percentile"] = (atr_current - atr_20.min()) / (atr_20.max() - atr_20.min()) if atr_20.max() != atr_20.min() else 0.5
            else:
                features["atr_percentile"] = 0.5
        except Exception:
            features["atr_percentile"] = 0.5

        # Market regime features
        try:
            adx_series = engine.adx(period=14)
            features["adx"] = adx_series.iloc[-1] if not adx_series.empty else 25.0
        except Exception:
            features["adx"] = 25.0

        # Trend direction (-1 to 1)
        if len(ema_9) > 0 and len(ema_21) > 0:
            features["trend_direction"] = 1.0 if ema_9.iloc[-1] > ema_21.iloc[-1] else -1.0
        else:
            features["trend_direction"] = 0.0

        # Volatility regime (0=low, 1=normal, 2=high, 3=extreme)
        volatility = features["volatility_20"]
        if volatility < 0.1:
            features["volatility_regime"] = 0.0
        elif volatility < 0.2:
            features["volatility_regime"] = 1.0
        elif volatility < 0.4:
            features["volatility_regime"] = 2.0
        else:
            features["volatility_regime"] = 3.0

        # Volume profile
        volume = data["volume"].values
        if len(volume) >= 20:
            avg_volume_20 = np.mean(volume[-20:])
            features["relative_volume"] = volume[-1] / avg_volume_20 if avg_volume_20 > 0 else 1.0
        else:
            features["relative_volume"] = 1.0

        # VWAP distance
        try:
            vwap_series = engine.vwap()
            if not vwap_series.empty and len(close) > 0:
                features["vwap_distance_pct"] = (close[-1] - vwap_series.iloc[-1]) / vwap_series.iloc[-1] * 100 if vwap_series.iloc[-1] != 0 else 0.0
            else:
                features["vwap_distance_pct"] = 0.0
        except Exception:
            features["vwap_distance_pct"] = 0.0

        # Time features
        signal_time = signal.get("timestamp", datetime.now(IST))
        if isinstance(signal_time, str):
            signal_time = datetime.fromisoformat(signal_time.replace("Z", "+00:00"))

        features["hour_of_day"] = float(signal_time.hour)
        features["day_of_week"] = float(signal_time.weekday())

        # Minutes since market open (9:15 AM IST)
        market_open = signal_time.replace(hour=9, minute=15, second=0, microsecond=0)
        if signal_time < market_open:
            features["minutes_since_open"] = 0.0
        else:
            delta = signal_time - market_open
            features["minutes_since_open"] = float(delta.total_seconds() / 60)

        # Create feature vector in consistent order
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])

        return feature_vector

    @graceful_degrade(
        default_return={
            "prediction": "neutral",
            "confidence": 50.0,
            "expected_return": 1.0,
            "risk_score": 5,
            "key_features": [],
        },
        log_level="warning",
    )
    def predict(self, signal_data: dict[str, Any]) -> dict[str, Any]:
        """Make prediction for a signal.

        Args:
            signal_data: Dictionary containing:
                - features: Pre-computed feature vector (optional)
                - market_data: OHLCV DataFrame (if features not provided)
                - signal: Signal dictionary (if features not provided)

        Returns:
            Dictionary with prediction results:
                - prediction: "profitable", "unprofitable", or "neutral"
                - confidence: 0-100 confidence score
                - expected_return: Estimated R:R ratio
                - risk_score: 1-10 risk rating (lower is safer)
                - key_features: List of most important features
        """
        # Extract or use provided features
        if "features" in signal_data:
            features = signal_data["features"]
        elif "market_data" in signal_data and "signal" in signal_data:
            features = self.extract_features(signal_data["market_data"], signal_data["signal"])
        else:
            raise ValueError("Must provide either 'features' or both 'market_data' and 'signal'")

        # Handle cold-start (not enough training data)
        if not self.is_trained or self.model is None:
            logger.debug("Model not trained yet, returning neutral prediction")
            return {
                "prediction": "neutral",
                "confidence": 50.0,
                "expected_return": 1.0,
                "risk_score": 5,
                "key_features": [],
                "cold_start": True,
            }

        # Thread-safe prediction
        with self._lock:
            # Reshape for single prediction
            X = features.reshape(1, -1)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get prediction probabilities
            proba = self.model.predict_proba(X_scaled)[0]

            # Get feature importance for this prediction
            feature_importance = self.get_feature_importance()
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]

        # Interpret results (assuming class 1 is profitable)
        profitable_prob = proba[1] if len(proba) > 1 else proba[0]
        unprofitable_prob = proba[0] if len(proba) > 1 else 1 - proba[0]

        if profitable_prob > 0.6:
            prediction = "profitable"
            confidence = profitable_prob * 100
        elif unprofitable_prob > 0.6:
            prediction = "unprofitable"
            confidence = unprofitable_prob * 100
        else:
            prediction = "neutral"
            confidence = 50.0 + abs(profitable_prob - 0.5) * 100

        # Calculate expected return based on confidence
        expected_return = 1.0 + (confidence - 50) / 50 * 2  # 1.0 to 3.0 range

        # Risk score (1-10, lower is safer)
        # Higher confidence = lower risk
        risk_score = max(1, min(10, int(11 - confidence / 10)))

        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "expected_return": round(expected_return, 2),
            "risk_score": risk_score,
            "key_features": [f[0] for f in top_features],
            "probabilities": {
                "profitable": round(profitable_prob, 4),
                "unprofitable": round(unprofitable_prob, 4),
            },
        }

    def validate_signal(
        self,
        signal: dict[str, Any],
        market_data: pd.DataFrame,
    ) -> AnalysisComponent:
        """Validate a trading signal and return an AnalysisComponent.

        This is the main integration point with the trade scorecard system.

        Args:
            signal: Signal dictionary with type, entry_price, etc.
            market_data: OHLCV DataFrame for feature extraction.

        Returns:
            AnalysisComponent with ML-based score (0-100).
        """
        try:
            # Make prediction
            result = self.predict({
                "market_data": market_data,
                "signal": signal,
            })

            # Convert prediction to score (0-100)
            if result["prediction"] == "profitable":
                base_score = 70 + (result["confidence"] - 60) * 0.75  # 70-100 range
            elif result["prediction"] == "unprofitable":
                base_score = 30 - (result["confidence"] - 60) * 0.75  # 0-30 range
            else:
                base_score = 50  # Neutral

            score = max(0, min(100, base_score))

            # Build reasoning string
            reasoning_parts = [
                f"ML prediction: {result['prediction']}",
                f"confidence: {result['confidence']:.1f}%",
            ]
            if result.get("key_features"):
                reasoning_parts.append(f"key features: {', '.join(result['key_features'])}")
            if result.get("cold_start"):
                reasoning_parts.append("(cold start - model not trained)")

            return AnalysisComponent(
                name="ml_validator",
                score=round(score, 1),
                weight=0.10,  # Default weight, can be overridden
                confidence=result["confidence"],
                reasoning="; ".join(reasoning_parts),
                data=result,
            )

        except Exception as e:
            logger.error(f"ML validation failed: {e}")
            # Return neutral component on failure
            return AnalysisComponent(
                name="ml_validator",
                score=50.0,
                weight=0.10,
                confidence=0.0,
                reasoning=f"ML validation failed: {type(e).__name__}. Using neutral score.",
                data={"error": str(e)},
            )

    def train(self, trade_outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        """Train or retrain the model on trade outcomes.

        Args:
            trade_outcomes: List of dictionaries containing:
                - features: Feature vector (list or array)
                - profitable: Boolean indicating if trade was profitable
                - pnl_paisa: Actual P&L in paisa (optional)
                - signal_id: Signal identifier (optional)

        Returns:
            Dictionary with training results:
                - success: Whether training succeeded
                - samples_used: Number of samples used
                - accuracy: Training accuracy
                - message: Status message
        """
        if not SKLEARN_AVAILABLE:
            return {
                "success": False,
                "samples_used": 0,
                "accuracy": 0.0,
                "message": "sklearn not available - cannot train model",
            }

        if len(trade_outcomes) < MIN_SAMPLES_FOR_TRAINING:
            return {
                "success": False,
                "samples_used": len(trade_outcomes),
                "accuracy": 0.0,
                "message": f"Insufficient samples: {len(trade_outcomes)}/{MIN_SAMPLES_FOR_TRAINING}",
            }

        try:
            # Prepare training data
            X_list = []
            y_list = []

            for outcome in trade_outcomes:
                if "features" in outcome and "profitable" in outcome:
                    features = np.array(outcome["features"])
                    if len(features) == len(self.feature_names):
                        X_list.append(features)
                        y_list.append(1 if outcome["profitable"] else 0)

            if len(X_list) < MIN_SAMPLES_FOR_TRAINING:
                return {
                    "success": False,
                    "samples_used": len(X_list),
                    "accuracy": 0.0,
                    "message": f"Valid samples after filtering: {len(X_list)}/{MIN_SAMPLES_FOR_TRAINING}",
                }

            X = np.array(X_list)
            y = np.array(y_list)

            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            # Train model
            with self._lock:
                self.model.fit(X_scaled, y)
                self.is_trained = True

                # Calculate training accuracy
                train_accuracy = self.model.score(X_scaled, y)

            logger.info(f"Trained ML model on {len(X)} samples with accuracy {train_accuracy:.3f}")

            return {
                "success": True,
                "samples_used": len(X),
                "accuracy": round(train_accuracy, 4),
                "message": f"Model trained successfully on {len(X)} samples",
            }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {
                "success": False,
                "samples_used": 0,
                "accuracy": 0.0,
                "message": f"Training failed: {type(e).__name__}: {e}",
            }

    def log_trade_outcome(
        self,
        signal: dict[str, Any],
        features: np.ndarray,
        profitable: bool,
        pnl_paisa: int,
    ) -> None:
        """Log a trade outcome for future training.

        Args:
            signal: Original signal dictionary
            features: Feature vector used for prediction
            profitable: Whether trade was profitable
            pnl_paisa: Actual P&L in paisa
        """
        log_entry = {
            "timestamp": datetime.now(IST).isoformat(),
            "signal_id": signal.get("signal_id", "unknown"),
            "instrument_key": signal.get("instrument_key", "unknown"),
            "signal_type": signal.get("type", "unknown"),
            "features": features.tolist(),
            "profitable": profitable,
            "pnl_paisa": pnl_paisa,
            "model_version": self.model_version,
        }

        # Append to training log
        try:
            with open(self.training_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Add to buffer for potential immediate retraining
            self._training_buffer.append(log_entry)

            logger.debug(f"Logged trade outcome for signal {log_entry['signal_id']}: {'profitable' if profitable else 'loss'}")
        except Exception as e:
            logger.error(f"Failed to log trade outcome: {e}")

    def load_training_data(self) -> list[dict[str, Any]]:
        """Load all training data from the log file.

        Returns:
            List of training examples.
        """
        if not self.training_log_path.exists():
            return []

        training_data = []
        try:
            with open(self.training_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            training_data.append(entry)
                        except json.JSONDecodeError:
                            continue

            logger.info(f"Loaded {len(training_data)} training examples from {self.training_log_path}")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")

        return training_data

    def retrain_from_log(self) -> dict[str, Any]:
        """Retrain model using all data from training log.

        Returns:
            Training results dictionary.
        """
        training_data = self.load_training_data()
        return self.train(training_data)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores from the model.

        Returns:
            Dictionary mapping feature names to importance scores.
            Returns empty dict if model is not trained.
        """
        if not SKLEARN_AVAILABLE:
            return {}

        if not self.is_trained or self.model is None:
            return {}

        try:
            importances = self.model.feature_importances_
            return {
                name: float(importance)
                for name, importance in zip(self.feature_names, importances)
            }
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}

    def should_retrain(self, min_new_samples: int = 50) -> bool:
        """Check if model should be retrained.

        Args:
            min_new_samples: Minimum new samples needed to trigger retrain.

        Returns:
            True if retraining is recommended.
        """
        return len(self._training_buffer) >= min_new_samples

    def clear_training_buffer(self) -> None:
        """Clear the in-memory training buffer."""
        self._training_buffer.clear()

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model state.

        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "sklearn_available": SKLEARN_AVAILABLE,
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
            "training_buffer_size": len(self._training_buffer),
            "model_path": str(self.model_path),
            "training_log_path": str(self.training_log_path),
        }
