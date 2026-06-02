"""Logging configuration using loguru."""

import sys
from typing import Any

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


def add_db_sink(db_engine: Any, level: str = "INFO") -> None:
    """Attach a SQLite sink that writes every log record to the bot_logs table.

    Call this once from main.py after the SQLite engine is ready.
    The table is created automatically if it doesn't exist.

    Args:
        db_engine: SQLAlchemy engine pointing at data/trading_bot.db.
        level:     Minimum log level to persist (default INFO — avoids
                   flooding the DB with DEBUG noise).
    """
    from sqlalchemy import text

    try:
        with db_engine.begin() as conn:
            conn.execute(text(
                """CREATE TABLE IF NOT EXISTS bot_logs (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT    NOT NULL,
                    level    TEXT    NOT NULL,
                    module   TEXT,
                    function TEXT,
                    line     INTEGER,
                    message  TEXT    NOT NULL
                )"""
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_bot_logs_ts ON bot_logs(ts)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_bot_logs_level ON bot_logs(level)"
            ))
    except Exception as exc:
        _logger.warning("add_db_sink: could not create bot_logs table: {}", exc)
        return

    def _sqlite_sink(message: Any) -> None:
        record = message.record
        try:
            with db_engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO bot_logs (ts, level, module, function, line, message) "
                        "VALUES (:ts, :level, :module, :func, :line, :msg)"
                    ),
                    {
                        "ts":     record["time"].isoformat(),
                        "level":  record["level"].name,
                        "module": record["name"],
                        "func":   record["function"],
                        "line":   record["line"],
                        "msg":    record["message"],
                    },
                )
        except Exception:
            pass  # Never let DB errors kill the logging pipeline

    _logger.add(
        _sqlite_sink,
        level=level,
        format="{message}",
        enqueue=True,
    )


# Module-level pre-configured logger instance.
logger = setup_logger()
