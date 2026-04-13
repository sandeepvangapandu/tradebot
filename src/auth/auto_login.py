"""Automated Upstox login using TOTP-based authentication."""

import os

from loguru import logger
from upstox_totp import UpstoxTOTP

from src.utils.exceptions import AuthenticationError


def auto_login() -> str:
    """Authenticate with Upstox using TOTP and return an access token.

    Reads credentials from environment variables and performs the full
    OAuth + TOTP flow via the upstox_totp package.

    Returns:
        The access token string for authenticated API calls.

    Raises:
        AuthenticationError: If any required env var is missing or login fails.
    """
    required_vars = [
        "UPSTOX_USERNAME",
        "UPSTOX_PASSWORD",
        "UPSTOX_PIN_CODE",
        "UPSTOX_TOTP_SECRET",
        "UPSTOX_CLIENT_ID",
        "UPSTOX_CLIENT_SECRET",
        "UPSTOX_REDIRECT_URI",
    ]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise AuthenticationError(f"Missing environment variables: {', '.join(missing)}")

    logger.info("Starting Upstox TOTP auto-login for user {}", os.environ["UPSTOX_USERNAME"])

    try:
        upx = UpstoxTOTP(
            username=os.environ["UPSTOX_USERNAME"],
            password=os.environ["UPSTOX_PASSWORD"],
            pin_code=os.environ["UPSTOX_PIN_CODE"],
            totp_secret=os.environ["UPSTOX_TOTP_SECRET"],
            client_id=os.environ["UPSTOX_CLIENT_ID"],
            client_secret=os.environ["UPSTOX_CLIENT_SECRET"],
            redirect_uri=os.environ["UPSTOX_REDIRECT_URI"],
        )

        response = upx.app_token.get_access_token()

        if response.success:
            logger.info("Upstox login successful")
            return response.data.access_token

        raise AuthenticationError(
            f"Upstox login failed: {getattr(response, 'message', 'Unknown error')}"
        )

    except AuthenticationError:
        raise
    except Exception as exc:
        logger.error("Upstox auto-login error: {}", exc)
        raise AuthenticationError(f"Auto-login failed: {exc}") from exc
