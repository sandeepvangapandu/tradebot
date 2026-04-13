"""Backtester dashboard page — full-fidelity backtesting using live components.

Upload a CSV of minute-level OHLCV data, select instrument and date range,
and run all strategies through the same code path as live trading.
"""

import sys
import tempfile
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import streamlit as st


def render():
    """Render the backtester page."""
    st.title("🧪 Full-Fidelity Strategy Backtester")
    st.caption(
        "Upload minute-level OHLCV data and run all strategies through the same "
        "PaperBroker → RiskManager → OrderManager pipeline as live trading."
    )

    try:
        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------
        st.subheader("📁 Upload Data & Configure")

        data_file = st.file_uploader(
            "Upload CSV — columns: timestamp, open, high, low, close, volume",
            type=["csv"],
            help="1-minute OHLCV candles. Prices can be in Rupees or Paisa.",
        )

        # Discover strategies
        config_dir = Path("config/strategies")
        if config_dir.exists():
            strategy_files = sorted(config_dir.glob("*.json"))
            strategy_names = [f.stem for f in strategy_files]
        else:
            strategy_names = []
            strategy_files = []

        if not strategy_names:
            st.warning(
                "No strategy configs found in `config/strategies/`. "
                "Add JSON strategy files first."
            )
            return

        col_a, col_b = st.columns(2)
        with col_a:
            instrument_key = st.text_input(
                "Instrument Key",
                value="NSE_INDEX|Nifty Bank",
                help='e.g. "NSE_INDEX|Nifty Bank" or "NSE_INDEX|Nifty 50"',
            )
        with col_b:
            prices_unit = st.selectbox(
                "Price unit in CSV",
                ["Rupees", "Paisa"],
                help="Rupees will be converted to Paisa automatically",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            capital_rupees = st.number_input(
                "Capital (Rs.)", value=1000000, step=100000, min_value=10000
            )
        with col2:
            from_date = st.date_input("From date (optional)", value=None)
        with col3:
            to_date = st.date_input("To date (optional)", value=None)

        # ------------------------------------------------------------------
        # Options proxy mode — simulate ATM option trading from index data
        # ------------------------------------------------------------------
        with st.expander("⚙️ Options Proxy Mode (experimental — off by default)", expanded=False):
            st.caption(
                "EXPERIMENTAL: transforms raw index OHLCV into a synthetic ATM option "
                "premium series using a linear-delta model. This amplifies noise — a "
                "0.2% index move becomes a ~20% premium move — and can cause unrealistic "
                "SL hits. Leave OFF to trade the index directly (recommended). Use only "
                "when you have calibrated strategy SL/target percentages for premium scale."
            )
            options_proxy = st.checkbox(
                "Enable options proxy (experimental)",
                value=False,
                help="When enabled, each day's index is mapped to an ATM option premium "
                "using delta-based scaling. A 1% index move ≈ 100% premium move.",
            )
            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                option_premium_pct = st.number_input(
                    "ATM premium (% of underlying)",
                    value=0.5,
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1,
                    disabled=not options_proxy,
                    help="E.g., 0.5% → Rs 250 premium on a Rs 50,000 BankNifty.",
                )
            with opt_col2:
                option_delta = st.number_input(
                    "Option delta",
                    value=0.5,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.05,
                    disabled=not options_proxy,
                    help="Typical ATM option delta ≈ 0.5.",
                )

        st.markdown("---")

        # ------------------------------------------------------------------
        # Run backtest
        # ------------------------------------------------------------------
        if data_file is not None:
            if st.button("🚀 Run Full-Fidelity Backtest", type="primary", width="stretch"):
                from src.backtest.harness import BacktestHarness, HarnessResults

                # Save uploaded file to a temp location so harness can read it
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp.write(data_file.read())
                    tmp_path = tmp.name

                # Progress indicators — created before spinner so they update in real-time
                progress_bar = st.progress(0, text="Initialising components...")
                status_text = st.empty()

                with st.spinner("Backtest running..."):
                    status_text.info("⚙️ Loading data and wiring live components...")
                    harness = BacktestHarness(
                        data_file=tmp_path,
                        instrument_key=instrument_key,
                        strategy_dir=str(config_dir),
                        capital=int(capital_rupees * 100),  # rupees → paisa
                        start_date=from_date if from_date else None,
                        end_date=to_date if to_date else None,
                        prices_in_rupees=(prices_unit == "Rupees"),
                        options_proxy=options_proxy,
                        option_premium_pct=option_premium_pct,
                        option_delta=option_delta,
                    )

                    total_bars = len(harness._master_df)
                    status_text.info(f"▶️ Running {total_bars:,} bars through full live pipeline...")
                    progress_bar.progress(0, text=f"0 / {total_bars:,} bars")

                    def progress_callback(current_bar: int, total_bars: int) -> None:
                        if total_bars > 0:
                            pct = min(current_bar / total_bars, 1.0)
                            progress_bar.progress(
                                pct,
                                text=f"{current_bar:,} / {total_bars:,} bars  ({pct*100:.0f}%)",
                            )

                    results: HarnessResults = harness.run(progress_callback=progress_callback)

                progress_bar.progress(1.0, text="Done!")
                status_text.success(f"✅ Backtest complete — {results.bars_processed:,} bars, {len(results.trades)} trades")

                m = results.metrics

                # ------------------------------------------------------------------
                # Results display
                # ------------------------------------------------------------------
                st.markdown("---")
                st.header("📊 Backtest Results")

                # --- Summary metrics ---
                st.subheader("Summary")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Net P&L", f"Rs. {m.get('net_pnl_rupees', 0):,.2f}")
                m2.metric("Return", f"{m.get('return_pct', 0):+.2f}%")
                m3.metric("Win Rate", f"{m.get('win_rate', 0):.1f}%")
                m4.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
                m5.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}")
                m6.metric("Max Drawdown", f"Rs. {m.get('max_drawdown_rupees', 0):,.2f}")

                n1, n2, n3, n4 = st.columns(4)
                n1.metric("Total Trades", m.get("total_trades", 0))
                n2.metric("Total Fees", f"Rs. {m.get('total_fees_rupees', 0):,.2f}")
                n3.metric("Max DD %", f"{m.get('max_drawdown_pct', 0):.2f}%")
                n4.metric("Bars Processed", results.bars_processed)

                st.markdown("---")

                # --- Equity Curve ---
                st.subheader("📈 Equity Curve")
                if len(results.equity_curve) > 1:
                    eq_rupees = [e / 100 for e in results.equity_curve]
                    eq_df = pd.DataFrame({"Capital (Rs)": eq_rupees})
                    st.line_chart(eq_df)

                st.markdown("---")

                # --- Detailed Trade List ---
                st.subheader("📋 Trades")
                if results.trades:
                    def _fmt(t: dict) -> dict:
                        return {
                            "Strategy": t.get("strategy_id", ""),
                            "Side": t.get("side", ""),
                            "Entry Price (Rs)": round(t.get("entry_price", 0) / 100, 2),
                            "Exit Price (Rs)": round(t.get("exit_price", 0) / 100, 2),
                            "Qty": t.get("quantity", 0),
                            "Net P&L (Rs)": round(t.get("net_pnl", 0) / 100, 2),
                            "Entry Time": t.get("entry_time", ""),
                            "Exit Time": t.get("exit_time", ""),
                            "Exit Reason": t.get("exit_reason", ""),
                        }

                    trades_df = pd.DataFrame([_fmt(t) for t in results.trades])

                    def _color_pnl(val):
                        if isinstance(val, (int, float)):
                            return "color: #00cc66" if val > 0 else "color: #ff4444" if val < 0 else ""
                        return ""

                    styled = trades_df.style.map(_color_pnl, subset=["Net P&L (Rs)"])
                    st.dataframe(styled, width="stretch", hide_index=True)

                    csv_trades = trades_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Trades CSV",
                        data=csv_trades,
                        file_name="backtest_trades.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No trades were generated. Strategy conditions were not met, or data is too short.")

        else:
            # Instructions when no file is uploaded
            st.markdown(
                """
### How to use

1. **Download minute-level data** for your instrument (e.g., Nifty Bank from a data vendor)

2. **CSV format** — your file should have these columns:
   | timestamp | open | high | low | close | volume |
   |-----------|------|------|-----|-------|--------|
   | 2026-03-01 09:15:00 | 48450.50 | 48465.00 | 48445.00 | 48460.25 | 125000 |

   - **1-minute candles** required
   - Select **Rupees or Paisa** to match your data's price unit
   - Timestamp should include date and time

3. **Set the instrument key** — must match the `underlying.instrument_key` in your strategy JSONs

4. **Click Run** — the system runs the exact same pipeline as live trading:
   - `BarProvider` serves data bar-by-bar (no lookahead)
   - `StrategyEngine.evaluate_bar_sync()` evaluates conditions
   - `OrderManager.process_signal()` runs Kelly sizing → research → risk → PaperBroker
   - `PositionManager` tracks exits (SL, target, trailing stop, partial profit)
   - `LearningIntegration` adapts position sizing as trades close (in-memory only)
                """
            )

    except Exception as e:
        st.error(f"Error in backtester: {e}")
        import traceback
        st.code(traceback.format_exc())


# Allow Streamlit multipage auto-discovery to run this page directly
render()
