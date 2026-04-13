"""Logging configuration using loguru."""

import sys

from loguru import logger as _logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/trading_bot.log",
) -> "loguru.Logger":  # type: ignore[name-defined]  # noqa: F821
    """Configure and return the application logger.

    Args:
        log_level: Minimum log level for the console sink.
        log_file: Path for the main rotating log file.

    Returns:
        Configured loguru logger instance.
    """
    # Remove the default stderr handler.
    _logger.remove()

    # Console handler — colored, INFO+ by default.
    _logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Main log file — rotation 10 MB, keep 30 days.
    _logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} — {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
    )

    # Error-only log file.
    _logger.add(
        "logs/errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} — {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
    )

    # Orders log file — only messages containing "ORDER".
    _logger.add(
        "logs/orders.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        filter=lambda record: "ORDER" in record["message"],
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
    )

    return _logger


# Module-level pre-configured logger instance.
logger = setup_logger()
