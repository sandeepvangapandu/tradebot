"""Token caching and lifecycle management for Upstox access tokens."""

import json
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

from src.auth.auto_login import auto_login

IST = ZoneInfo("Asia/Kolkata")
TOKEN_EXPIRY_TIME = time(3, 30)  # 3:30 AM IST


class TokenManager:
    """Manages Upstox access token caching, validation, and refresh.

    Tokens are cached to disk as JSON and reused until they expire.
    Upstox tokens expire at 3:30 AM IST the following day (or same day
    if obtained before 3:30 AM).

    Args:
        cache_path: Path to the JSON file used for token caching.
    """

    def __init__(self, cache_path: str = "data/token_cache.json") -> None:
        self._cache_path = Path(cache_path)

    def _load_cache(self) -> dict | None:
        """Load the cached token data from disk.

        Returns:
            Parsed JSON dict with token info, or None if unavailable.
        """
        try:
            data = json.loads(self._cache_path.read_text())
            logger.debug("Loaded token cache from {}", self._cache_path)
            return data
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.debug("No valid token cache found: {}", exc)
            return None

    def _save_cache(self, token: str) -> None:
        """Save a token and its metadata to the cache file.

        Args:
            token: The access token string to cache.
        """
        now = datetime.now(tz=IST)
        expiry = self._calculate_expiry()
        data = {
            "access_token": token,
            "obtained_at": now.isoformat(),
            "expires_at": expiry.isoformat(),
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(data, indent=2))
        logger.info("Token cached, expires at {}", expiry.isoformat())

    def is_token_valid(self) -> bool:
        """Check whether a cached token exists and has not expired.

        Returns:
            True if a valid, non-expired token is cached.
        """
        cache = self._load_cache()
        if not cache or "access_token" not in cache or "expires_at" not in cache:
            return False

        try:
            expires_at = datetime.fromisoformat(cache["expires_at"])
            now = datetime.now(tz=IST)
            return now < expires_at
        except (ValueError, TypeError):
            return False

    def get_valid_token(self) -> str:
        """Return a valid access token, using cache or re-authenticating.

        Resolution order:
        1. Use cached token if still valid.
        2. Use UPSTOX_ACCESS_TOKEN env var if set (manual paste mode).
        3. Try TOTP auto-login (requires password).
        4. Fail with a helpful message pointing to manual_login.py.

        Returns:
            A valid Upstox access token.

        Raises:
            AuthenticationError: If no usable token can be obtained.
        """
        import os
        from src.utils.exceptions import AuthenticationError

        # 1. Cached token
        cache = self._load_cache()
        if cache and self.is_token_valid():
            logger.info("Using cached token (expires {})", cache["expires_at"])
            return cache["access_token"]

        # 2. Env var token (set by manual_login.py or pasted by user)
        env_token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
        if env_token:
            logger.info("Using UPSTOX_ACCESS_TOKEN from environment")
            self._save_cache(env_token)
            return env_token

        # 3. TOTP auto-login (only if password-based credentials present)
        if os.getenv("UPSTOX_PASSWORD") and os.getenv("UPSTOX_TOTP_SECRET"):
            logger.info("No cached/env token, attempting TOTP auto-login")
            return self.refresh_token()

        # 4. Helpful failure message
        raise AuthenticationError(
            "No valid Upstox token available.\n"
            "OTP-based accounts: run `python3 -m src.auth.manual_login` to log in.\n"
            "Password-based accounts: set UPSTOX_PASSWORD + UPSTOX_TOTP_SECRET in .env."
        )

    def refresh_token(self) -> str:
        """Force a fresh authentication regardless of cache state.

        On failure, fires an alert to Telegram (best-effort, swallows
        any notification error) so the user is notified out-of-band
        before discovering it via EOD report.

        Returns:
            A newly obtained Upstox access token.

        Raises:
            AuthenticationError: If auto-login fails.
        """
        logger.info("Refreshing token via auto-login")
        try:
            token = auto_login()
            self._save_cache(token)
            return token
        except Exception as exc:
            logger.error("Token refresh FAILED: {}", exc)
            self._alert_auth_failure(str(exc))
            raise

    @staticmethod
    def _alert_auth_failure(reason: str) -> None:
        """Best-effort Telegram alert for auth failures.

        Imports lazily to avoid a hard dependency on the notifications
        layer (TokenManager must work in CLI scripts that don't init
        Telegram).
        """
        try:
            import asyncio
            from src.notifications.telegram_bot import TelegramBot

            bot = TelegramBot()
            if not getattr(bot, "_bot_token", None):
                return  # No telegram configured
            coro = bot.send_error_alert(
                f"Upstox token refresh failed: {reason[:200]}.\n"
                f"Run `python3 -m src.auth.manual_login` to recover."
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(coro)
                else:
                    loop.run_until_complete(coro)
            except RuntimeError:
                asyncio.run(coro)
        except Exception as exc:
            logger.debug("Telegram auth alert suppressed: {}", exc)

    def _calculate_expiry(self) -> datetime:
        """Calculate the token expiry timestamp.

        Upstox tokens expire at 3:30 AM IST. If the current time is
        before 3:30 AM, expiry is today at 3:30 AM. Otherwise, expiry
        is tomorrow at 3:30 AM.

        Returns:
            Expiry datetime in IST.
        """
        now = datetime.now(tz=IST)
        expiry_today = now.replace(
            hour=TOKEN_EXPIRY_TIME.hour,
            minute=TOKEN_EXPIRY_TIME.minute,
            second=0,
            microsecond=0,
        )

        if now < expiry_today:
            return expiry_today

        # Next day 3:30 AM
        return expiry_today + timedelta(days=1)
