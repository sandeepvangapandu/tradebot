"""Application settings loaded from environment variables via Pydantic."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the trading bot.

    All monetary values are stored in paisa (1 Rupee = 100 paisa) to avoid
    floating-point precision issues. Convert to Rupees only for display.

    Attributes:
        upstox_username: Upstox login username.
        upstox_password: Upstox login password.
        upstox_pin_code: Upstox 6-digit PIN.
        upstox_totp_secret: Base32 TOTP secret for 2FA.
        upstox_client_id: OAuth2 client/app ID.
        upstox_client_secret: OAuth2 client secret.
        upstox_redirect_uri: OAuth2 redirect URI.
        trading_mode: 'paper' for simulation, 'live' for real orders.
        default_exchange: Default exchange segment (NSE, BSE).
        max_daily_loss: Maximum daily loss in paisa before trading halts.
        max_open_positions: Maximum concurrent open positions.
        max_position_size_pct: Max single-instrument exposure as % of capital.
        capital: Total trading capital in paisa.
        max_capital_deployment_pct: Max % of capital deployed at any time.
        consecutive_loss_pause: Pause after N consecutive losing trades.
        pause_minutes: Minutes to pause after consecutive losses.
        slippage_pct: Simulated slippage percentage for paper trading.
        telegram_bot_token: Telegram bot token for notifications.
        telegram_chat_id: Telegram chat ID for notifications.
        database_url: SQLAlchemy database connection string.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to the primary log file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Upstox credentials
    upstox_username: str = ""
    upstox_password: str = ""
    upstox_pin_code: str = ""
    upstox_totp_secret: str = ""
    upstox_client_id: str = ""
    upstox_client_secret: str = ""
    upstox_redirect_uri: str = "https://127.0.0.1/callback"

    # Trading
    trading_mode: Literal["paper", "live"] = "paper"
    default_exchange: str = "NSE"

    # Risk management (monetary values in paisa)
    # 10L capital = 100,000,000 paisa
    # max_daily_loss = 2% of 10L = ₹20,000 = 2,000,000 paisa
    max_daily_loss: int = 2_000_000
    max_open_positions: int = 3
    max_position_size_pct: int = 90      # straddle margin << notional index value
    capital: int = 100_000_000
    max_capital_deployment_pct: int = 95
    consecutive_loss_pause: int = 3
    pause_minutes: int = 30
    slippage_pct: float = 0.05

    # Notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Infrastructure
    database_url: str = "sqlite:///data/trading_bot.db"
    log_level: str = "INFO"
    log_file: str = "logs/trading_bot.log"

    # Research Module (AI Trade Research & Decision Engine)
    research_enabled: bool = True
    research_min_score: int = 45
    research_score_for_full_size: int = 55
    research_max_analysis_time_seconds: float = 2.0
    research_log_all: bool = True
    # Straddle-optimized weights: trend_alignment irrelevant for non-directional,
    # boost event_calendar (IV timing) and momentum (entry precision)
    research_weights: dict = Field(default_factory=lambda: {
        "trend_alignment": 0.05,
        "momentum": 0.20,
        "support_resistance": 0.15,
        "candle_context": 0.10,
        "market_regime": 0.05,
        "volume": 0.10,
        "options": 0.15,
        "pattern": 0.05,
        "correlation": 0.05,
        "events": 0.20,
        "ml_validator": 0.05,
        "strategy_confluence": 0.05,
    })
    research_regime_cache_minutes: int = 5
    research_sr_cache_minutes: int = 180
    research_iv_cache_minutes: int = 15
    research_correlation_cache_minutes: int = 30

    # Self-Learning System Configuration
    learning_enabled: bool = True
    learning_min_trades_for_analysis: int = 5
    learning_low_win_rate_threshold: float = 0.35
    learning_high_win_rate_threshold: float = 0.6
    learning_min_profit_factor: float = 1.0
    learning_target_profit_factor: float = 1.5
    learning_max_consecutive_losses_threshold: int = 4
    learning_position_size_reduction_factor: float = 0.5
    learning_position_size_increase_factor: float = 1.25
    learning_auto_adjust_size: bool = True
    learning_generate_daily_report: bool = True
    learning_export_lessons_path: str = "data/lessons.json"

    # AI Agent Pipeline Configuration
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024
    groq_rate_limit_rpm: int = 30  # Free tier: 30 requests/min

    # Agent pipeline settings
    agent_pipeline_enabled: bool = True
    regime_confidence_threshold: float = 0.3  # Skip pipeline if below
    signal_min_confidence: float = 0.5  # Reject signals below this
    news_sentiment_weight: float = 0.35
    volatility_sentiment_weight: float = 0.35
    breadth_sentiment_weight: float = 0.30
    news_cache_ttl_seconds: int = 300  # 5 minutes
    sentiment_cache_ttl_seconds: int = 600  # 10 minutes

    # Memory system settings
    memory_enabled: bool = True
    memory_decay_rate: float = 0.05  # 5% per week
    memory_decay_start_days: int = 30  # Start decay after 30 days
    memory_max_lessons_per_agent: int = 5  # Top N lessons injected
    memory_boost_factor: float = 1.1  # Boost useful lessons (capped at 2x)


from typing import Optional
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
