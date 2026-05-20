#!/usr/bin/env python3
"""
Exchange an Upstox OAuth callback URL for an access token and save it.

Usage:
    python3 scripts/exchange_token.py "https://yourredirect.com/callback?code=XXXXXXXX"
"""

import json
import os
import sys
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TOKEN_CACHE = Path(__file__).parent.parent / "token_cache.json"


def exchange(callback_url: str) -> str:
    parsed = urllib.parse.urlparse(callback_url)
    params = urllib.parse.parse_qs(parsed.query)

    if "code" not in params:
        print(f"ERROR: No 'code' param found in URL: {callback_url}")
        sys.exit(1)

    code = params["code"][0]
    print(f"Auth code: {code[:10]}...")

    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        data={
            "code": code,
            "client_id": os.environ["UPSTOX_CLIENT_ID"],
            "client_secret": os.environ["UPSTOX_CLIENT_SECRET"],
            "redirect_uri": os.environ["UPSTOX_REDIRECT_URI"],
            "grant_type": "authorization_code",
        },
    )

    if resp.status_code != 200:
        print(f"ERROR: Token exchange failed: {resp.status_code} {resp.text[:300]}")
        sys.exit(1)

    data = resp.json()
    token = data.get("access_token")
    if not token:
        print(f"ERROR: No access_token in response: {data}")
        sys.exit(1)

    TOKEN_CACHE.write_text(json.dumps({"access_token": token}, indent=2))
    print(f"Token saved to {TOKEN_CACHE}")
    print("Done — bot can now start.")
    return token


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/exchange_token.py \"<full callback URL>\"")
        sys.exit(1)
    exchange(sys.argv[1])
