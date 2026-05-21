#!/usr/bin/env python3
"""Train ML models (RLExitAgent) from historical backtest data.

Runs the BacktestHarness on 6-month BankNifty 1-min data using straddle_proxy
mode to generate synthetic option trade trajectories, then trains the
RLExitAgent from those trajectories.

Usage:
    python3 scripts/train_ml_models.py [--episodes 500]

Output:
    models/rl_exit_agent.pkl  — trained Q-table checkpoint
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from loguru import logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BANKNIFTY_CSV   = "data/backtest/banknifty_1m_6mo.csv"
INSTRUMENT_KEY  = "NSE_INDEX|Nifty Bank"
CAPITAL_PAISA   = 100_000_00  # ₹10 lakh
CHECKPOINT_PATH = "models/rl_exit_agent.pkl"
STRATEGY_DIR    = "config/strategies"

# Synthetic straddle params matching harness defaults
OPTION_PREMIUM_PCT = 0.5   # 0.5 % of index = ~₹275 for BankNifty @55k
OPTION_DELTA       = 0.5


def _run_backtest() -> list[dict]:
    """Run full harness backtest and return list of closed trade dicts."""
    from src.backtest.harness import BacktestHarness

    logger.info("Initialising BacktestHarness on {} …", BANKNIFTY_CSV)
    harness = BacktestHarness(
        data_file=BANKNIFTY_CSV,
        instrument_key=INSTRUMENT_KEY,
        strategy_dir=STRATEGY_DIR,
        capital=CAPITAL_PAISA,
        straddle_proxy=True,
        option_premium_pct=OPTION_PREMIUM_PCT,
        option_delta=OPTION_DELTA,
        strategy_only="expiry_straddle_sell_banknifty",
    )

    total_bars = len(harness._master_df)

    def _progress(cur: int, tot: int) -> None:
        if cur % 5000 == 0:
            logger.info("  … {}/{} bars ({:.0f} %)", cur, tot, 100 * cur / tot)

    results = harness.run(progress_callback=_progress)

    logger.info(
        "Backtest complete | bars={} trades={} win_rate={:.1f}% total_pnl=₹{:.0f}",
        results.bars_processed,
        len(results.trades),
        results.metrics.get("win_rate", 0) * 100,
        results.metrics.get("total_pnl", 0) / 100,
    )
    return results.trades


def _synthesise_trajectory(trade: dict) -> list[tuple[dict, str, float, dict, bool]]:
    """Convert a single closed trade dict into a list of (s, a, r, s', done) tuples.

    Because the harness only returns aggregate trade data (not per-tick states),
    we synthesise a plausible N-step trajectory by interpolating the PnL curve
    from entry to exit in steps proportional to trade duration.

    Each step:
      - PnL grows linearly from 0 → final_pnl_pct
      - Momentum is estimated from PnL slope direction
      - trailing_active activates after PnL exceeds +30 %

    This is a heuristic approximation — exact when per-tick data is unavailable.
    The RL agent refines these priors during live trading.
    """
    entry_price: int = int(trade.get("entry_price", 0) or 0)
    exit_price:  int = int(trade.get("exit_price", 0) or 0)
    pnl_paisa:   int = int(trade.get("net_pnl", 0) or 0)
    duration_s:  int = int(trade.get("duration_seconds", 300) or 300)
    sl_price:    int = int(trade.get("stop_loss_price", 0) or 0)
    tgt_price:   int = int(trade.get("target_price", 0) or 0)

    if entry_price <= 0:
        return []

    final_pnl_pct = float(pnl_paisa / entry_price)
    final_pnl_pct = max(-2.0, min(2.0, final_pnl_pct))

    # Number of synthetic steps: 1 step per 5-minute bar, min 3
    n_steps = max(3, int(duration_s / 300))

    trajectory: list[tuple[dict, str, float, dict, bool]] = []

    sl_dist_entry  = abs(sl_price  - entry_price) / max(entry_price, 1) if sl_price  else 0.2
    tgt_dist_entry = abs(tgt_price - entry_price) / max(entry_price, 1) if tgt_price else 0.4

    for step in range(n_steps):
        frac      = step / n_steps          # 0.0 → ~1.0
        next_frac = (step + 1) / n_steps

        pnl_now  = final_pnl_pct * frac
        pnl_next = final_pnl_pct * next_frac

        pnl_change = pnl_next - pnl_now
        momentum   = float(math.copysign(min(abs(pnl_change) * 5, 1.0), pnl_change))
        trailing   = 1.0 if pnl_now > 0.3 else 0.0
        peak_gain  = max(0.0, pnl_now)

        # Distance to SL/target shrinks as PnL moves toward them
        sl_dist  = max(0.0, sl_dist_entry  - abs(pnl_now) * 0.5)
        tgt_dist = max(0.0, tgt_dist_entry - abs(pnl_now) * 0.5)

        done = (step == n_steps - 1)
        action = "EXIT" if done else "HOLD"

        state: dict[str, float] = {
            "pnl_pct":        float(pnl_now),
            "bars_held_norm": float(frac),
            "momentum":       momentum,
            "vol_regime":     1.0,
            "dist_to_sl":     sl_dist,
            "dist_to_target": tgt_dist,
            "trailing_active": trailing,
            "peak_gain_norm": peak_gain,
        }
        next_state: dict[str, float] = {k: 0.0 for k in state}

        if not done:
            next_state.update({
                "pnl_pct":        float(pnl_next),
                "bars_held_norm": float(next_frac),
                "momentum":       momentum,
                "vol_regime":     1.0,
                "dist_to_sl":     sl_dist,
                "dist_to_target": tgt_dist,
                "trailing_active": 1.0 if pnl_next > 0.3 else 0.0,
                "peak_gain_norm": max(0.0, pnl_next),
            })

        trajectory.append((state, action, pnl_change, next_state, done))

    return trajectory


def _train_rl_agent(trades: list[dict], checkpoint_path: str, n_passes: int) -> None:
    """Train RLExitAgent from backtest trade list and save checkpoint.

    Args:
        trades: List of closed trade dicts from HarnessResults.trades.
        checkpoint_path: Path to write the trained Q-table pickle.
        n_passes: How many times to iterate through the trade set (replay epochs).
    """
    from src.execution.rl_exit_agent import RLExitAgent

    agent = RLExitAgent(
        db_path="data/trading_bot.db",
        learning_rate=0.1,
        discount=0.95,
        epsilon=0.1,
    )

    logger.info("Training RLExitAgent | trades={} passes={}", len(trades), n_passes)

    for pass_idx in range(n_passes):
        total_reward = 0.0
        for trade in trades:
            trajectory = _synthesise_trajectory(trade)
            for state, action, reward, next_state, done in trajectory:
                agent.record_step(state, action, reward, next_state, done)
                total_reward += reward

        avg_r = total_reward / max(len(trades), 1)
        logger.info(
            "Pass {}/{} | episodes={} q_states={} avg_reward={:.4f}",
            pass_idx + 1, n_passes, agent.episode_count, agent.q_table_size, avg_r,
        )

    logger.info(
        "Training done | episodes={} q_states={} trained={}",
        agent.episode_count, agent.q_table_size, agent.is_trained,
    )
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)
    logger.info("Checkpoint saved → {}", checkpoint_path)


def _print_metrics(trades: list[dict]) -> None:
    """Print a quick performance table from backtest trades."""
    if not trades:
        logger.warning("No trades generated — strategy may not have fired on this data.")
        return

    wins  = [t for t in trades if (t.get("net_pnl") or 0) > 0]
    loss  = [t for t in trades if (t.get("net_pnl") or 0) <= 0]
    total_pnl = sum((t.get("net_pnl") or 0) for t in trades)
    gross_p = sum((t.get("net_pnl") or 0) for t in wins)
    gross_l = abs(sum((t.get("net_pnl") or 0) for t in loss))

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS — BankNifty Expiry Straddle (6 months)")
    print("=" * 60)
    print(f"  Total trades : {len(trades)}")
    print(f"  Win rate     : {100 * len(wins) / max(len(trades), 1):.1f}%")
    print(f"  Total P&L    : ₹{total_pnl / 100:,.0f}")
    print(f"  Avg P&L/trade: ₹{total_pnl / max(len(trades), 1) / 100:,.0f}")
    print(f"  Profit factor: {gross_p / max(gross_l, 1):.2f}")
    print(f"  Gross profit : ₹{gross_p / 100:,.0f}")
    print(f"  Gross loss   : ₹{gross_l / 100:,.0f}")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3,
                        help="Training passes over the backtest trade set (default: 3)")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip backtest and train RL agent from existing DB trades only")
    args = parser.parse_args()

    if args.skip_backtest:
        logger.info("Skip-backtest mode — warm-starting RL agent from trading_bot.db …")
        from src.execution.rl_exit_agent import RLExitAgent
        agent = RLExitAgent(db_path="data/trading_bot.db")
        replayed = agent.replay_from_db(n_episodes=200)
        logger.info("Replayed {} trades from DB", replayed)
        Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
        agent.save(CHECKPOINT_PATH)
        logger.info("Checkpoint saved → {}", CHECKPOINT_PATH)
        return 0

    # Full backtest + train path
    trades = _run_backtest()
    _print_metrics(trades)
    _train_rl_agent(trades, CHECKPOINT_PATH, n_passes=args.episodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
