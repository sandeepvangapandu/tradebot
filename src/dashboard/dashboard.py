"""Trading Bot — Overview Dashboard.

Single-pane terminal-style overview. Streamlit auto-discovers extra
pages in `pages/` (Trade History, Strategy Perf, Risk Monitor,
Backtester, Live Feed). This file IS the landing/Overview page.

Run with:
    bash run_dashboard.sh
or:
    streamlit run src/dashboard/dashboard.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Add project root before any project imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.settings import get_settings
from src.dashboard.data_service import DashboardDataService
from src.persistence.database import get_session, init_db

IST = ZoneInfo("Asia/Kolkata")
NY = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Page config + DB init
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TB · Overview",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

settings = get_settings()
init_db(settings.database_url)

# ---------------------------------------------------------------------------
# Theme — CSS injection
# ---------------------------------------------------------------------------

_CSS = """
<style>
  :root {
    --bg:        #0d1117;
    --panel:     #161b22;
    --panel-2:   #1c2230;
    --border:    #30363d;
    --text:      #c9d1d9;
    --text-dim:  #8b949e;
    --accent:    #58a6ff;
    --green:     #3fb950;
    --green-bg:  rgba(63, 185, 80, 0.10);
    --red:       #f85149;
    --red-bg:    rgba(248, 81, 73, 0.10);
    --amber:     #d29922;
  }

  html, body, [data-testid="stAppViewContainer"], .main {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace !important;
    font-feature-settings: 'tnum' on, 'lnum' on;
  }
  header[data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 1px solid var(--border); }
  section[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--border); }

  .ticker {
    display: flex; gap: 22px; align-items: center; flex-wrap: wrap;
    padding: 12px 18px; background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px; margin-bottom: 16px;
    font-size: 12px; letter-spacing: 0.4px; text-transform: uppercase;
  }
  .ticker .pill { padding: 4px 10px; border-radius: 3px; background: var(--panel-2); border: 1px solid var(--border); }
  .ticker .ok  { color: var(--green); border-color: rgba(63, 185, 80, 0.4); background: var(--green-bg); }
  .ticker .err { color: var(--red);   border-color: rgba(248, 81, 73, 0.4); background: var(--red-bg); }
  .ticker .warn{ color: var(--amber); border-color: rgba(210, 153, 34, 0.4); background: rgba(210, 153, 34, 0.10); }
  .ticker .lbl { color: var(--text-dim); margin-right: 4px; }

  h1, h2, h3, h4 { color: var(--text) !important; font-family: inherit !important; letter-spacing: 0.4px; }
  h1 { font-size: 20px !important; font-weight: 600 !important; }
  h2 { font-size: 13px !important; font-weight: 600 !important; text-transform: uppercase; color: var(--text-dim) !important; margin: 28px 0 6px 0 !important; letter-spacing: 1.2px; }
  hr { border-color: var(--border) !important; margin: 8px 0 !important; }

  [data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px 12px;
  }
  [data-testid="stMetricLabel"] {
    color: var(--text-dim) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    font-feature-settings: 'tnum' on;
  }
  [data-testid="stMetricDelta"] { font-size: 11px !important; color: var(--text-dim) !important; }
  [data-testid="stMetricDelta"] svg { display: none; }

  [data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }

  .js-plotly-plot { background: transparent !important; }

  #MainMenu, footer { visibility: hidden; }
  button[kind="primary"] { background: var(--accent) !important; border-color: var(--accent) !important; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly common
# ---------------------------------------------------------------------------

PLOT_BG = "#161b22"
GRID = "#30363d"
TEXT = "#c9d1d9"
GREEN = "#3fb950"
RED = "#f85149"
ACCENT = "#58a6ff"


def fmt_inr(rs: float, signed: bool = False) -> str:
    sign = ""
    if signed and rs != 0:
        sign = "+" if rs > 0 else "−"
        rs = abs(rs)
    elif rs < 0:
        sign = "−"
        rs = abs(rs)
    if abs(rs) >= 1_00_000:
        return f"{sign}₹{rs / 1_00_000:.2f}L"
    if abs(rs) >= 1_000:
        return f"{sign}₹{rs / 1_000:.2f}k"
    return f"{sign}₹{rs:,.2f}"


def age_str(ts) -> str:
    if ts is None:
        return "—"
    now = datetime.now(IST) if getattr(ts, "tzinfo", None) else datetime.now()
    mins = int((now - ts).total_seconds() / 60)
    if mins < 60:
        return f"{mins}m"
    return f"{mins // 60}h {mins % 60}m"


# ---------------------------------------------------------------------------
# Data pull
# ---------------------------------------------------------------------------

with get_session() as session:
    svc = DashboardDataService(session)
    today_pnl = svc.get_today_pnl()
    total_pnl = svc.get_total_pnl()
    today_fees = svc.get_today_fees()
    trades_today, wins_today = svc.get_today_trade_count()
    positions = svc.get_current_positions()
    equity_df = svc.get_equity_curve(days=30)
    recent_df = svc.get_recent_trades(limit=20)
    strat_perf = svc.get_strategy_performance()
    risk = svc.get_risk_metrics(
        max_positions=settings.max_open_positions,
        max_capital_pct=settings.max_capital_deployment_pct,
        daily_loss_limit=settings.max_daily_loss / 100,
        total_capital=settings.capital / 100,
    )
    token = svc.get_token_state()

# ---------------------------------------------------------------------------
# Header strip
# ---------------------------------------------------------------------------

now_ist = datetime.now(IST)
now_ny = datetime.now(NY)
mode_cls = "ok" if settings.trading_mode == "paper" else "warn"
hours_left = token.get("hours_left", 0.0)
tok_cls = "ok" if hours_left > 2 else ("warn" if hours_left > 0 else "err")
tok_text = f"{hours_left:.1f}h" if token.get("valid") else "EXPIRED"

st.markdown(
    f"""
<div class="ticker">
  <span><b>◆ TRADING BOT</b></span>
  <span class="pill {mode_cls}">MODE · {settings.trading_mode.upper()}</span>
  <span class="pill"><span class="lbl">IST</span>{now_ist.strftime("%a %d %b · %H:%M:%S")}</span>
  <span class="pill"><span class="lbl">NY</span>{now_ny.strftime("%H:%M")}</span>
  <span class="pill {tok_cls}">TOKEN · {tok_text}</span>
  <span class="pill"><span class="lbl">CAPITAL</span>{fmt_inr(settings.capital / 100)}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

win_rate = (wins_today / trades_today * 100) if trades_today else 0.0
unrealized_total = sum(p.unrealized_pnl for p in positions)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Today realized", fmt_inr(today_pnl, signed=True))
k2.metric("Today unrealized", fmt_inr(unrealized_total, signed=True))
k3.metric("Total realized", fmt_inr(total_pnl, signed=True))
k4.metric(
    "Trades today",
    f"{trades_today}",
    f"{wins_today}W / {trades_today - wins_today}L" if trades_today else "—",
)
k5.metric("Win rate", f"{win_rate:.0f}%" if trades_today else "—")
k6.metric("Open positions", f"{len(positions)} / {settings.max_open_positions}")

# ---------------------------------------------------------------------------
# Equity + daily P&L
# ---------------------------------------------------------------------------

st.markdown("## Equity & daily P&L · last 30 sessions")
if equity_df.empty:
    st.info("No closed trades in last 30 sessions.")
else:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.66, 0.34],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity_df["cumulative"],
            mode="lines",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(88, 166, 255, 0.10)",
            hovertemplate="%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    daily_colors = [GREEN if v >= 0 else RED for v in equity_df["realized_pnl"]]
    fig.add_trace(
        go.Bar(
            x=equity_df.index,
            y=equity_df["realized_pnl"],
            marker_color=daily_colors,
            hovertemplate="%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_layout(
        height=420,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
        margin=dict(l=50, r=20, t=10, b=40),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1c2230", font_color=TEXT, font_family="JetBrains Mono, monospace"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=TEXT, row=1, col=1)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=TEXT, row=2, col=1)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=TEXT, title="Cumulative ₹", row=1, col=1)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=TEXT, title="Daily ₹", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Risk gauges
# ---------------------------------------------------------------------------

st.markdown("## Risk gauges")


def _gauge(value_pct: float, title: str, ok_below: float, warn_below: float) -> go.Figure:
    color = GREEN
    if value_pct >= warn_below:
        color = RED
    elif value_pct >= ok_below:
        color = "#d29922"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"suffix": "%", "font": {"color": color, "size": 26}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT, "tickfont": {"color": TEXT, "size": 9}},
            "bar": {"color": color, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, ok_below], "color": "rgba(63, 185, 80, 0.15)"},
                {"range": [ok_below, warn_below], "color": "rgba(210, 153, 34, 0.15)"},
                {"range": [warn_below, 100], "color": "rgba(248, 81, 73, 0.15)"},
            ],
        },
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
        title=dict(text=title, font=dict(color=TEXT, size=11), x=0.5, xanchor="center", y=0.95),
    )
    return fig


g1, g2, g3 = st.columns(3)
g1.plotly_chart(
    _gauge(risk.capital_usage_pct, "Capital deployed", 60, 85),
    use_container_width=True, config={"displayModeBar": False},
)
g2.plotly_chart(
    _gauge(risk.position_usage_pct, "Position slots used", 60, 85),
    use_container_width=True, config={"displayModeBar": False},
)
g3.plotly_chart(
    _gauge(risk.daily_loss_usage_pct, "Daily loss budget used", 50, 80),
    use_container_width=True, config={"displayModeBar": False},
)

# ---------------------------------------------------------------------------
# Open positions
# ---------------------------------------------------------------------------

st.markdown("## Open positions")
if not positions:
    st.caption("No open positions.")
else:
    rows = []
    for p in positions:
        rows.append({
            "Strategy": p.strategy,
            "Side": p.side,
            "Instrument": p.instrument_key,
            "Qty": p.quantity,
            "Entry": f"₹{p.entry_price:,.2f}",
            "P&L": f"₹{p.unrealized_pnl:+,.2f}",
            "Age": age_str(p.opened_at),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Strategy performance
# ---------------------------------------------------------------------------

st.markdown("## Strategy performance · all time")
if not strat_perf:
    st.caption("No closed trades yet.")
else:
    strat_rows = []
    for sp in strat_perf:
        if sp.profit_factor:
            pf_str = f"{sp.profit_factor:.2f}"
        elif sp.total_pnl > 0:
            pf_str = "∞"
        else:
            pf_str = "0.00"
        strat_rows.append({
            "Strategy": sp.strategy_name,
            "Trades": sp.total_trades,
            "Win %": f"{sp.win_rate * 100:.1f}%",
            "W/L": f"{sp.win_count}/{sp.loss_count}",
            "Net P&L": f"₹{sp.total_pnl:+,.2f}",
            "Avg win": f"₹{sp.avg_win:,.2f}",
            "Avg loss": f"₹{sp.avg_loss:,.2f}",
            "PF": pf_str,
        })
    st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Recent trades
# ---------------------------------------------------------------------------

st.markdown("## Recent trades · last 20")
if recent_df.empty:
    st.caption("No trades yet.")
else:
    display = recent_df.copy()
    display["exit_time"] = pd.to_datetime(display["exit_time"]).dt.strftime("%d %b %H:%M")
    display["entry"] = display["entry"].map(lambda v: f"₹{v:,.2f}")
    display["exit"] = display["exit"].map(lambda v: f"₹{v:,.2f}")
    display["pnl"] = display["pnl"].map(lambda v: f"₹{v:+,.2f}")
    display["fees"] = display["fees"].map(lambda v: f"₹{v:,.2f}")
    display["duration_min"] = display["duration_min"].map(lambda v: f"{v:.1f}m")
    display.columns = ["Closed", "Strategy", "Instrument", "Side", "Qty", "Entry", "Exit", "P&L", "Fees", "Held"]
    st.dataframe(display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    f"DB · {settings.database_url}   |   "
    f"Today fees · ₹{today_fees:,.2f}   |   "
    f"Last refresh · {now_ist.strftime('%H:%M:%S IST')}"
)
