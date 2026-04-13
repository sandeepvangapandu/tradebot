"""Custom exception hierarchy for the trading bot."""


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""

    def __init__(self, message: str = "An unexpected trading bot error occurred.") -> None:
        self.message = message
        super().__init__(self.message)


class AuthenticationError(TradingBotError):
    """Raised when broker authentication fails."""

    def __init__(self, message: str = "Authentication failed.") -> None:
        super().__init__(message)


class TokenExpiredError(AuthenticationError):
    """Raised when the access token has expired and needs refresh."""

    def __init__(self, message: str = "Access token has expired.") -> None:
        super().__init__(message)


class OrderError(TradingBotError):
    """Raised when an order placement or modification fails."""

    def __init__(self, message: str = "Order operation failed.") -> None:
        super().__init__(message)


class InsufficientMarginError(OrderError):
    """Raised when there is not enough margin to place an order."""

    def __init__(self, message: str = "Insufficient margin for this order.") -> None:
        super().__init__(message)


class RiskLimitBreachedError(TradingBotError):
    """Raised when a risk management limit is breached."""

    def __init__(self, message: str = "Risk limit breached.") -> None:
        super().__init__(message)


class RateLimitError(TradingBotError):
    """Raised when the broker API rate limit is hit."""

    def __init__(self, message: str = "API rate limit exceeded.") -> None:
        super().__init__(message)


class WebSocketError(TradingBotError):
    """Raised on WebSocket connection or data errors."""

    def __init__(self, message: str = "WebSocket error occurred.") -> None:
        super().__init__(message)


class InstrumentError(TradingBotError):
    """Raised when an instrument lookup or validation fails."""

    def __init__(self, message: str = "Instrument error occurred.") -> None:
        super().__init__(message)
