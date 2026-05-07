"""Live Upstox market data dashboard page.

Polls the Upstox `/v2/market-quote/ltp` and `/v2/market-quote/ohlc` REST
endpoints every few seconds to display real-time prices for the watchlist.

This page is read-only: it does NOT place orders. The bot itself runs as
a separate process (`python3 -m src.main`) and uses the WebSocket V3 feed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for standalone execution
_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from config.settings import get_settings
from src.auth.token_manager import TokenManager

IST = ZoneInfo("Asia/Kolkata")

# Default watchlist — indices and a few liquid F&O underlyings
DEFAULT_WATCHLIST = [
    "NSE_INDEX|Nifty Bank",
    "NSE_INDEX|Nifty 50",
    "NSE_INDEX|Nifty Fin Service",
    "NSE_EQ|INE002A01018",  # Reliance
    "NSE_EQ|INE467B01029",  # TCS
    "NSE_EQ|INE040A01034",  # HDFC Bank
    "NSE_EQ|INE090A01021",  # ICICI Bank
    "NSE_EQ|INE154A01025",  # ITC
]


def _fetch_ltp(token: str, instrument_keys: list[str]) -> dict:
    """Fetch LTP + last trade time from Upstox REST API.

    Args:
        token: Valid Upstox access token.
        instrument_keys: Upstox instrument keys.

    Returns:
        Dict mapping `<exch>:<symbol>` to quote payload.
    """
    if not instrument_keys:
        return {}
    encoded = ",".join(quote(k, safe="") for k in instrument_keys)
    url = f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={encoded}"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=8,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LTP API {resp.status_code}: {resp.text[:200]}")
    return resp.json().get("data", {})


def _fetch_ohlc(token: str, instrument_keys: list[str]) -> dict:
    """Fetch intraday OHLC for richer display (open/high/low/close/volume)."""
    if not instrument_keys:
        return {}
    encoded = ",".join(quote(k, safe="") for k in instrument_keys)
    url = (
        "https://api.upstox.com/v2/market-quote/ohlc?"
        f"instrument_key={encoded}&interval=1d"
    )
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=8,
    )
    if resp.status_code != 200:
        return {}
    return resp.json().get("data", {})


def _market_status() -> tuple[str, str]:
    """Return (label, color) for current Indian market session."""
    now = datetime.now(IST)
    hh, mm = now.hour, now.minute
    minutes = hh * 60 + mm
    weekday = now.weekday()  # 0=Mon .. 6=Sun
    if weekday >= 5:
        return "CLOSED (Weekend)", "off"
    if 9 * 60 <= minutes < 9 * 60 + 15:
        return "PRE-OPEN", "normal"
    if 9 * 60 + 15 <= minutes <= 15 * 60 + 30:
        return "OPEN", "normal"
    return "CLOSED", "off"


def render() -> None:
    """Render the live feed page."""
    st.title("📡 Live Upstox Feed")

    settings = get_settings()
    tm = TokenManager()

    # --- Token check ---
    cache = tm._load_cache() or {}
    token = cache.get("access_token", "")
    if not token or not tm.is_token_valid():
        st.error(
            "Upstox access token is missing or expired.\n\n"
            "Run this in your terminal to refresh:\n\n"
            "```bash\npython3 -m src.auth.manual_login\n```\n\n"
            "Then reload this page."
        )
        return

    # --- Header row ---
    status_label, status_kind = _market_status()
    now = datetime.now(IST)
    c1, c2, c3 = st.columns([2, 2, 2])
    if status_kind == "off":
        c1.error(f"Market: {status_label}")
    else:
        c1.success(f"Market: {status_label}")
    c2.metric("Mode", settings.trading_mode.upper())
    c3.caption(f"Last refresh: {now.strftime('%I:%M:%S %p IST')}")

    # --- Sidebar controls ---
    st.sidebar.markdown("### Live Feed Controls")
    refresh_sec = st.sidebar.slider("Refresh every (seconds)", 2, 30, 5)
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

    watchlist_default = "\n".join(DEFAULT_WATCHLIST)
    watchlist_text = st.sidebar.text_area(
        "Watchlist (one instrument_key per line)",
        value=st.session_state.get("watchlist_text", watchlist_default),
        height=200,
    )
    st.session_state["watchlist_text"] = watchlist_text
    instrument_keys = [
        line.strip() for line in watchlist_text.splitlines() if line.strip()
    ]

    # --- Fetch live data ---
    try:
        ltp_data = _fetch_ltp(token, instrument_keys)
        ohlc_data = _fetch_ohlc(token, instrument_keys)
    except Exception as exc:
        st.error(f"Failed to fetch live data: {exc}")
        return

    if not ltp_data:
        st.warning(
            "No data returned. Verify your watchlist instrument keys "
            "(e.g. `NSE_INDEX|Nifty Bank`, `NSE_EQ|INE002A01018`)."
        )
        return

    # --- Build display table ---
    rows = []
    for key in instrument_keys:
        # Upstox returns keys with ':' instead of '|' in the response
        resp_key = key.replace("|", ":")
        ltp_row = ltp_data.get(resp_key) or ltp_data.get(key) or {}
        ohlc_row = ohlc_data.get(resp_key) or ohlc_data.get(key) or {}

        ltp = ltp_row.get("last_price")
        ohlc = ohlc_row.get("ohlc", {}) or {}
        prev_close = ohlc.get("close")
        change = (ltp - prev_close) if (ltp is not None and prev_close) else None
        change_pct = (change / prev_close * 100) if (change is not None and prev_close) else None

        rows.append(
            {
                "Instrument": key.split("|", 1)[-1],
                "Segment": key.split("|", 1)[0],
                "LTP (Rs.)": ltp,
                "Open": ohlc.get("open"),
                "High": ohlc.get("high"),
                "Low": ohlc.get("low"),
                "Prev Close": prev_close,
                "Change": round(change, 2) if change is not None else None,
                "Change %": round(change_pct, 2) if change_pct is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    st.subheader("📊 Live Quotes")
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Change %": st.column_config.NumberColumn(format="%.2f %%"),
            "LTP (Rs.)": st.column_config.NumberColumn(format="%.2f"),
            "Change": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # --- Top movers ---
    valid = df.dropna(subset=["Change %"])
    if not valid.empty:
        st.subheader("🚀 Movers")
        a, b = st.columns(2)
        gainers = valid.nlargest(3, "Change %")[["Instrument", "LTP (Rs.)", "Change %"]]
        losers = valid.nsmallest(3, "Change %")[["Instrument", "LTP (Rs.)", "Change %"]]
        a.markdown("**Top Gainers**")
        a.dataframe(gainers, width='stretch', hide_index=True)
        b.markdown("**Top Losers**")
        b.dataframe(losers, width='stretch', hide_index=True)

    # --- Auto refresh ---
    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


# Auto-run when Streamlit executes this page
render()
