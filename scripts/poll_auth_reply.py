#!/usr/bin/env python3
"""
Poll Gmail inbox for a reply containing the Upstox callback URL.
Runs after send_auth_email.py, waits up to 90 minutes for user to reply,
then exchanges the code for a token automatically.

Usage (called from cron automatically):
    python3 scripts/poll_auth_reply.py

When you get the auth email:
  1. Click the login link, complete Upstox login
  2. Copy the full URL from your browser address bar (the redirect URL with ?code=...)
  3. Reply to the email with just that URL in the body
"""

import email
import imaplib
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).parent.parent / ".env")

TOKEN_CACHE = Path(__file__).parent.parent / "token_cache.json"
POLL_INTERVAL = 30          # seconds between inbox checks
RETRY_AFTER_SECONDS = 90 * 60  # resend auth email after 90 min no reply
HARD_STOP_SECONDS = 10 * 3600  # give up entirely after 10 hours


def _find_callback_url(text: str) -> str | None:
    """Extract a URL containing '?code=' from email body text."""
    # Look for redirect URI pattern with code param
    redirect_uri = os.environ.get("UPSTOX_REDIRECT_URI", "")
    patterns = [
        r'https?://[^\s<>"]+\?[^\s<>"]*code=[^\s<>"&]+[^\s<>"]*',
        r'https?://[^\s<>"]+code=[^\s<>"&]+',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            url = match.group(0).rstrip(".,;)")
            if "code=" in url:
                return url
    return None


def _exchange_code(callback_url: str) -> bool:
    parsed = urllib.parse.urlparse(callback_url)
    params = urllib.parse.parse_qs(parsed.query)
    if "code" not in params:
        logger.error("No 'code' param in URL: {}", callback_url)
        return False

    code = params["code"][0]
    logger.info("Exchanging auth code {}...", code[:8])

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
        logger.error("Token exchange failed: {} {}", resp.status_code, resp.text[:200])
        return False

    data = resp.json()
    token = data.get("access_token")
    if not token:
        logger.error("No access_token in response: {}", data)
        return False

    TOKEN_CACHE.write_text(json.dumps({"access_token": token}, indent=2))
    logger.info("Token saved to {}", TOKEN_CACHE)
    return True


def poll_inbox() -> bool:
    """Connect to Gmail IMAP, look for reply with callback URL. Return True if token saved."""
    host = os.environ.get("IMAP_HOST", "imap.gmail.com")
    user = os.environ["REPORT_EMAIL_FROM"]
    password = os.environ["REPORT_EMAIL_PASSWORD"]

    try:
        mail = imaplib.IMAP4_SSL(host)
        mail.login(user, password)
        mail.select("INBOX")

        # Search for unread messages from self (our replies come back as unread)
        _, msg_ids = mail.search(None, 'UNSEEN FROM "{}"'.format(user))
        ids = msg_ids[0].split()

        for mid in ids:
            _, msg_data = mail.fetch(mid, "(RFC822)")
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)
            subject = msg.get("Subject", "")

            # Only process replies to our auth emails
            if "TradingBot" not in subject and "Upstox" not in subject:
                continue

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode(errors="ignore")
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            url = _find_callback_url(body)
            if url:
                logger.info("Found callback URL in reply: {}...", url[:60])
                # Mark as read
                mail.store(mid, "+FLAGS", "\\Seen")
                mail.logout()
                return _exchange_code(url)

        mail.logout()
    except Exception as exc:
        logger.error("IMAP poll error: {}", exc)

    return False


def send_auth_email() -> None:
    """Send the Upstox auth URL email (imported inline to avoid circular dep)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "send_auth_email",
        Path(__file__).parent / "send_auth_email.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    url = mod.build_auth_url()
    mod.send_email(url)


def main() -> None:
    logger.info("Polling inbox for Upstox auth reply (hard stop {}h)...", HARD_STOP_SECONDS // 3600)
    total_elapsed = 0
    cycle_elapsed = 0

    while total_elapsed < HARD_STOP_SECONDS:
        if poll_inbox():
            logger.info("Token exchanged successfully. Bot can start.")
            sys.exit(0)

        time.sleep(POLL_INTERVAL)
        total_elapsed += POLL_INTERVAL
        cycle_elapsed += POLL_INTERVAL

        if cycle_elapsed >= RETRY_AFTER_SECONDS:
            logger.warning("No reply in 90 min — sending new auth email and resetting timer.")
            try:
                send_auth_email()
            except Exception as exc:
                logger.error("Failed to resend auth email: {}", exc)
            cycle_elapsed = 0
        else:
            logger.info("Waiting for reply... {}m this cycle, {}m total", cycle_elapsed // 60, total_elapsed // 60)

    logger.error("Hard stop after {}h — no auth reply received.", HARD_STOP_SECONDS // 3600)
    sys.exit(1)


if __name__ == "__main__":
    main()
