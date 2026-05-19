"""Automated Upstox login using TOTP-based authentication."""

import os
import time
import urllib.parse

import pyotp
import requests
from loguru import logger

from src.utils.exceptions import AuthenticationError


def auto_login() -> str:
    """Authenticate with Upstox using TOTP and return an access token.

    Reads credentials from environment variables and performs the full
    OAuth + TOTP flow via direct API calls.

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

    username = os.environ["UPSTOX_USERNAME"]
    password = os.environ["UPSTOX_PASSWORD"]
    pin_code = os.environ["UPSTOX_PIN_CODE"]
    totp_secret = os.environ["UPSTOX_TOTP_SECRET"]
    client_id = os.environ["UPSTOX_CLIENT_ID"]
    client_secret = os.environ["UPSTOX_CLIENT_SECRET"]
    redirect_uri = os.environ["UPSTOX_REDIRECT_URI"]

    logger.info("Starting Upstox TOTP auto-login for user {}", username)

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    try:
        # Step 1: Get auth code via login + TOTP
        totp = pyotp.TOTP(totp_secret)
        totp_code = totp.now()

        auth_url = "https://api.upstox.com/v2/login/authorization/dialog"
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
        }

        # Login with credentials
        login_resp = session.post(
            "https://api.upstox.com/v2/login/authorization/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "password",
                "username": username,
                "password": password,
                "totp": totp_code,
                "pin": pin_code,
            },
        )

        if login_resp.status_code == 200:
            data = login_resp.json()
            if "access_token" in data:
                logger.info("Upstox TOTP login successful")
                return data["access_token"]

        raise AuthenticationError(
            f"Upstox TOTP login failed: {login_resp.status_code} {login_resp.text[:200]}\n"
            "For OTP-based accounts run: python3 -m src.auth.manual_login"
        )

    except AuthenticationError:
        raise
    except Exception as exc:
        logger.error("Upstox auto-login error: {}", exc)
        raise AuthenticationError(f"Auto-login failed: {exc}") from exc
