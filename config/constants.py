"""Static constants for Indian market trading."""

from datetime import time
from zoneinfo import ZoneInfo

# Timezone
IST: ZoneInfo = ZoneInfo("Asia/Kolkata")

# Market hours (IST)
PRE_OPEN_START: time = time(9, 0)
MARKET_OPEN: time = time(9, 15)
MARKET_CLOSE: time = time(15, 30)
INTRADAY_SQUARE_OFF: time = time(15, 15)

# Tick size in paisa (NSE equity = Rs 0.05)
TICK_SIZE_NSE: int = 5

# Fee structure (all percentages unless noted, monetary values in paisa)
# All percentage values are in decimal form (e.g., 0.0125 = 1.25%)
FEES: dict[str, float] = {
    # Brokerage - flat Rs 20 per order for intraday/F&O, free for delivery
    "brokerage_equity_delivery": 0.0,        # Free equity delivery
    "brokerage_equity_intraday": 2000,       # Rs 20 flat per order (paisa)
    "brokerage_fno": 2000,                   # Rs 20 flat per F&O order (paisa)

    # STT (Securities Transaction Tax) - as decimal
    "stt_equity_delivery_buy_pct": 0.0,      # No STT on equity delivery buy
    "stt_equity_delivery_sell_pct": 0.001,   # 0.1% on equity delivery sell
    "stt_equity_intraday_sell_pct": 0.00025, # 0.025% on intraday sell
    "stt_futures_buy_pct": 0.0,              # No STT on futures buy
    "stt_futures_sell_pct": 0.0002,          # 0.02% on futures sell (on contract value)
    "stt_options_buy_pct": 0.0,              # No STT on options buy
    "stt_options_sell_pct": 0.0125,          # 0.0125% on options sell (on premium)

    # Exchange transaction charges - as decimal (on turnover)
    "exchange_charge_nse_eq_pct": 0.00053,   # 0.053% NSE equity
    "exchange_charge_nse_fo_pct": 0.00053,   # 0.053% NSE F&O (on premium for options)
    "exchange_charge_bse_eq_pct": 0.00053,   # 0.053% BSE equity

    # GST - 18% on (brokerage + exchange charges)
    "gst_pct": 0.18,                         # 18% GST

    # SEBI charges - Rs 10 per crore = 0.0000001 as decimal
    "sebi_charge_rate_pct": 0.0000001,       # As decimal on turnover in rupees
    "sebi_charges_per_crore": 1000,          # Rs 10 per crore turnover (paisa)

    # Stamp duty (buy side only) - varies by state, using average rates
    "stamp_duty_equity_delivery_pct": 0.00015,  # 0.015% on delivery buy
    "stamp_duty_equity_intraday_pct": 0.00003,  # 0.003% on intraday buy
    "stamp_duty_fno_pct": 0.00002,              # 0.002% on F&O buy

    # DP charges (for delivery sell) - Rs 13.5 per scrip per day
    "dp_charges_paisa": 1350,                # Rs 13.50 in paisa

    # Slippage default
    "default_slippage_pct": 0.0005,          # 0.05% default slippage
}

# Upstox instrument master URLs
INSTRUMENT_URLS: dict[str, str] = {
    "NSE_EQ": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "BSE_EQ": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz",
    "NSE_FO": "https://assets.upstox.com/market-quote/instruments/exchange/NFO.json.gz",
    "MCX": "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz",
}

# Supported market segments
MARKET_SEGMENTS: list[str] = [
    "NSE_EQ",
    "BSE_EQ",
    "NSE_FO",
    "CDS",
    "MCX",
]
