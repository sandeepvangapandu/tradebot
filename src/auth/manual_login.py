"""Semi-automated Upstox OAuth login for OTP-only accounts.

Upstox now offers OTP-based login (no password). The official `upstox-totp`
automation requires a static password, so it doesn't work for these accounts.

Instead, this helper:
1. Opens the Upstox OAuth URL in the user's default browser.
2. The user logs in normally (mobile OTP → PIN → TOTP).
3. Upstox redirects to https://127.0.0.1/callback?code=AUTH_CODE.
4. The browser shows "site can't be reached" — that's expected.
5. The user copies the full URL from the address bar and pastes it here.
6. This script extracts the code, exchanges it for an access token,
   and caches it to data/token_cache.json for the rest of the trading day.

Run with:
    python3 -m src.auth.manual_login
"""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from datetime import datetime, time, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from zoneinfo import ZoneInfo

import requests
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")
TOKEN_EXPIRY_TIME = time(3, 30)  # Upstox tokens expire daily at 3:30 AM IST
CACHE_PATH = Path("data/token_cache.json")


def build_auth_url(client_id: str, redirect_uri: str) -> str:
    """Construct the Upstox OAuth authorization URL.

    Args:
        client_id: Upstox API key (UUID).
        redirect_uri: The redirect URI registered with the Upstox app.

    Returns:
        The full authorization URL the user must visit.
    """
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
    }
    return f"https://api.upstox.com/v2/login/authorization/dialog?{urlencode(params)}"


def extract_code_from_url(url: str) -> str:
    """Pull the OAuth `code` parameter out of a redirect URL.

    Args:
        url: The full URL the user copied from their browser address bar.

    Returns:
        The auth code string.

    Raises:
        ValueError: If the URL doesn't contain a code parameter.
    """
    parsed = urlparse(url.strip())
    qs = parse_qs(parsed.query)
    if "code" not in qs:
        raise ValueError(
            "No `code` parameter found in URL. Make sure you copied the full "
            "URL from the address bar AFTER logging in."
        )
    return qs["code"][0]


def exchange_code_for_token(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict:
    """Exchange an OAuth code for an access token.

    Args:
        code: The auth code from the redirect URL.
        client_id: Upstox API key.
        client_secret: Upstox API secret.
        redirect_uri: The redirect URI registered with the Upstox app.

    Returns:
        Token response dict with `access_token`, `user_id`, etc.

    Raises:
        RuntimeError: If the exchange fails.
    """
    response = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed [{response.status_code}]: {response.text}"
        )
    return response.json()


def calculate_expiry() -> datetime:
    """Return the next 3:30 AM IST after now (Upstox token expiry)."""
    now = datetime.now(tz=IST)
    today_expiry = now.replace(hour=3, minute=30, second=0, microsecond=0)
    if now >= today_expiry:
        return today_expiry + timedelta(days=1)
    return today_expiry


def cache_token(token: str, user_id: str = "") -> None:
    """Persist the access token to disk for reuse during the trading day.

    Args:
        token: The access token to cache.
        user_id: Optional user_id for logging.
    """
    expires_at = calculate_expiry()
    payload = {
        "access_token": token,
        "user_id": user_id,
        "obtained_at": datetime.now(tz=IST).isoformat(),
        "expires_at": expires_at.isoformat(),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Token cached to {} | expires {}", CACHE_PATH, expires_at.isoformat())


def update_env_token(token: str) -> None:
    """Update UPSTOX_ACCESS_TOKEN in the .env file (best-effort)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    lines = env_path.read_text().splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith("UPSTOX_ACCESS_TOKEN="):
            lines[i] = f"UPSTOX_ACCESS_TOKEN={token}"
            found = True
            break
    if not found:
        lines.append(f"UPSTOX_ACCESS_TOKEN={token}")
    env_path.write_text("\n".join(lines) + "\n")
    logger.info("Updated UPSTOX_ACCESS_TOKEN in .env")


def manual_login() -> str:
    """Run the full interactive login flow and return a fresh access token.

    Returns:
        The new access token string.
    """
    # Load credentials
    client_id = os.getenv("UPSTOX_CLIENT_ID", "").strip()
    client_secret = os.getenv("UPSTOX_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv(
        "UPSTOX_REDIRECT_URI", "https://127.0.0.1/callback"
    ).strip()

    if not client_id or not client_secret:
        # Try loading from .env directly
        from dotenv import load_dotenv
        load_dotenv()
        client_id = os.getenv("UPSTOX_CLIENT_ID", "").strip()
        client_secret = os.getenv("UPSTOX_CLIENT_SECRET", "").strip()
        redirect_uri = os.getenv(
            "UPSTOX_REDIRECT_URI", "https://127.0.0.1/callback"
        ).strip()

    if not client_id or not client_secret:
        print("ERROR: UPSTOX_CLIENT_ID or UPSTOX_CLIENT_SECRET missing in .env")
        sys.exit(1)

    auth_url = build_auth_url(client_id, redirect_uri)

    print()
    print("=" * 70)
    print("  UPSTOX MANUAL LOGIN")
    print("=" * 70)
    print()
    print("Step 1: Your browser will open the Upstox login page.")
    print("Step 2: Log in (mobile OTP, PIN, then TOTP).")
    print("Step 3: After login you'll be redirected to:")
    print(f"        {redirect_uri}?code=...")
    print("        The page will fail to load — that's expected.")
    print("Step 4: Copy the FULL URL from your browser address bar.")
    print("Step 5: Paste it below.")
    print()
    print(f"Login URL: {auth_url}")
    print()

    try:
        webbrowser.open(auth_url)
        print("Browser opened. Press Enter if it didn't open automatically.")
    except Exception:
        print("Could not open browser automatically. Visit the URL above manually.")

    print()
    redirected_url = input("Paste the redirected URL here: ").strip()

    code = extract_code_from_url(redirected_url)
    print(f"\nExtracted code: {code[:20]}...")
    print("Exchanging code for access token...")

    token_data = exchange_code_for_token(code, client_id, client_secret, redirect_uri)
    access_token = token_data["access_token"]
    user_id = token_data.get("user_id", "")

    cache_token(access_token, user_id)
    update_env_token(access_token)

    print()
    print("=" * 70)
    print("  ✅ LOGIN SUCCESSFUL")
    print("=" * 70)
    print(f"  User ID:      {user_id}")
    print(f"  User name:    {token_data.get('user_name', '')}")
    print(f"  Email:        {token_data.get('email', '')}")
    print(f"  Token cached: {CACHE_PATH}")
    print(f"  Valid until:  {calculate_expiry().strftime('%Y-%m-%d %H:%M IST')}")
    print("=" * 70)
    print()

    return access_token


if __name__ == "__main__":
    manual_login()
