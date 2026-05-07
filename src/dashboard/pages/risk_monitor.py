"""Risk Monitor dashboard page."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import streamlit as st

from src.persistence.database import get_session, init_db
from config.settings import get_settings


def render():
    """Render the risk monitor page."""
    st.title("⚠️ Risk Monitor")

    try:
        from src.dashboard.data_service import DashboardDataService

        settings = get_settings()
        init_db(settings.database_url)

        capital_rupees = settings.capital / 100
        daily_loss_limit_rupees = settings.max_daily_loss / 100

        with get_session() as session:
            service = DashboardDataService(session)

            risk = service.get_risk_metrics(
                max_positions=settings.max_open_positions,
                max_capital_pct=settings.max_capital_deployment_pct,
                daily_loss_limit=daily_loss_limit_rupees,
                total_capital=capital_rupees,
            )
            positions = service.get_current_positions()

        # --- Risk Configuration ---
        st.subheader("📊 Risk Configuration")
        col1, col2, col3 = st.columns(3)
        col1.metric("Capital", f"Rs. {capital_rupees:,.2f}")
        col2.metric("Max Daily Loss", f"Rs. {daily_loss_limit_rupees:,.2f}")
        col3.metric("Max Open Positions", settings.max_open_positions)

        col4, col5, col6 = st.columns(3)
        col4.metric("Max Position Size", f"{settings.max_position_size_pct}%")
        col5.metric(
            "Max Capital Deployed", f"{settings.max_capital_deployment_pct}%"
        )
        col6.metric(
            "Circuit Breaker After", f"{settings.consecutive_loss_pause} losses"
        )

        st.markdown("---")

        # --- Risk Usage ---
        st.subheader("📈 Current Risk Usage")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Daily Loss Limit**")
            pct = min(risk.daily_loss_usage_pct / 100, 1.0)
            st.progress(pct if pct >= 0 else 0.0)
            if risk.daily_pnl < 0:
                st.caption(
                    f"Loss: Rs. {abs(risk.daily_pnl):,.2f} / "
                    f"Rs. {risk.daily_loss_limit:,.2f} "
                    f"({risk.daily_loss_usage_pct:.1f}%)"
                )
            else:
                st.caption(f"Profit today: Rs. {risk.daily_pnl:,.2f}")

        with col2:
            st.markdown("**Position Slots**")
            pos_pct = min(risk.position_usage_pct / 100, 1.0)
            st.progress(pos_pct if pos_pct >= 0 else 0.0)
            st.caption(
                f"{risk.open_positions} / {risk.max_positions} positions"
            )

        with col3:
            st.markdown("**Capital Deployed**")
            cap_pct = min(risk.capital_usage_pct / 100, 1.0)
            st.progress(cap_pct if cap_pct >= 0 else 0.0)
            st.caption(
                f"Rs. {risk.deployed_capital:,.2f} / "
                f"Rs. {risk.max_capital:,.2f}"
            )

        st.markdown("---")

        # --- Trading Mode ---
        st.subheader("System Status")
        if settings.trading_mode == "paper":
            st.success("PAPER TRADING MODE — No real orders are being placed")
        else:
            st.error("LIVE TRADING MODE — Real money at risk!")

        # --- Open Positions ---
        st.subheader("📊 Open Positions")
        if positions:
            pos_data = []
            for p in positions:
                pos_data.append(
                    {
                        "Strategy": p.strategy,
                        "Instrument": p.instrument_key,
                        "Side": p.side,
                        "Entry (Rs.)": p.entry_price,
                        "Qty": p.quantity,
                        "Unrealized P&L (Rs.)": p.unrealized_pnl,
                    }
                )
            st.dataframe(pd.DataFrame(pos_data), width='stretch')
        else:
            st.info("No open positions")

    except Exception as e:
        st.error(f"Error loading risk monitor: {e}")
        import traceback
        st.code(traceback.format_exc())


# Allow Streamlit multipage auto-discovery to run this page directly
render()
