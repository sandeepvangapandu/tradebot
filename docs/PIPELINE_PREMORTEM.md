# Paper/Live Trading Pipeline — Premortem & Bug Audit

_Read-only audit, 2026-06-13. No code changed. Findings only — fix later._

Scope: full signal→order→fill→position→exit→P&L→risk loop, paper **and** live.
Trace path: `_on_market_tick` → `StrategyEngine` → `SignalForwarder` → `OrderManager.process_signal`
→ broker `place_order` → `PositionManager.add_position` → `on_tick` exits → `_close_position` → `RiskManager`/`CircuitBreaker`.

Severity legend: **P0 critical** (wrong money / silent capital loss), **P1 high** (wrong risk gating or stats), **P2 medium** (degraded accuracy), **P3 low/cosmetic**.

---

## PART 1 — BUGS & CORRECTNESS ISSUES

### P0-1 — Exit P&L ignores actual fill price (slippage silently dropped)
**Where:** `position_manager.py:1451` (legacy partial), `:1535` (tier exit), `:1953` (final close).
All three compute realized P&L as `position.get_unrealized_pnl_for_quantity(price, qty)` using the **requested tick price**, never `response.average_price` returned by the broker.

- Paper broker fills at slippage/spread-adjusted price (`paper_broker.py:185-215`, `place_order` returns `average_price=fill_price`).
- **Entry** P&L baseline DOES use fill price (`order_manager.py:1199`).
- **Exit** P&L uses the clean tick price.

**Effect:** asymmetric. Every exit overstates profit by ~the exit slippage (≈0.05% of exit notional, or full half-spread when bid/ask present). Across hundreds of trades this systematically inflates paper P&L and win rate, and the same code runs live → live P&L records will disagree with broker contract notes.
**Fix direction:** use `response.average_price` (fallback to `price`) for the realized-P&L calc in all three exit paths.

---

### P0-2 — Positions closed via the 4-tier path skip trade logging, fees, circuit breaker, and learning
**Where:** `_execute_tier_exit` `position_manager.py:1561-1570`. When `partial_profit_manager.is_complete(...)` is true, the position is marked `is_closed=True` **inside the tier handler** and `_on_tick_single` then returns early (`:1263`) without ever calling `_close_position`.

Everything that only lives in `_close_position` is therefore skipped for tier-completed positions:
- `_trade_logger.log_trade(...)` → no row in the `trades` table (`:2046`)
- fee attribution `sum_fees_for_instrument` (`:2034`) → trade recorded with **zero fees anywhere**
- `circuit_breaker.record_win/record_loss` (`:2083-2085`) → consecutive-loss streak misses these outcomes
- `record_strategy_pnl` per-strategy daily cap (`:2070`)
- `on_position_close` learning callback (`:2088`)

Every directional position with a stop-loss gets 4 tiers initialized by default (`add_position` `:899-907`, 25/25/25/25, tier-4 trailing). The runner can be closed either by `_check_tier_4_trailing_stop`→`_close_position` (logs fine) **or** by `check_tiers` returning the tier-4 hit→`_execute_tier_exit`→`is_complete`→silent close. Which one fires is a race between two detectors each tick.

**Effect (live):** daily P&L summary, win-rate, and Kelly sizing all read the `trades` table — tier-closed trades are **invisible** to them. Per-strategy daily-loss quarantine and the consecutive-loss circuit breaker under-count. This is the most dangerous finding: real money moves, books don't record it, and the risk governor is partially blind.
**Fix direction:** route final tier close through `_close_position` (or factor the logging/fees/CB/learning block into a shared `_finalize_close` both paths call).

---

### P1-3 — `reset_daily()` is clobbered by the next P&L sync → daily loss limit is actually *cumulative*
**Where:** `risk_manager.reset_daily()` `:353-360` zeroes `realized_pnl`; but `_sync_risk_manager` `position_manager.py:2265` recomputes `get_total_realized_pnl()` = **sum over every closed position still in `self._positions`**, and `PositionManager` never purges closed positions (no `_positions.clear()` anywhere; square-off sets `is_closed=True` but keeps them).

So on a long-running multi-day process, the first position close after the 9:14 SOD reset overwrites `realized_pnl` back to lifetime cumulative P&L. `check_daily_loss_limit` (`:154`, `realized+unrealized <= -max_daily_loss`) then operates on cumulative, not daily.

**Effect:**
- Cumulative profit → daily loss limit effectively never trips (under-protective).
- Cumulative drawdown across days → kill switch can trip at SOD before a single trade (over-protective / stuck).

Only safe if the process is **fully restarted every day** (empty `_positions` at boot). Current shutdown/scheduler design implies a long-running process, so this is a live latent bug.
**Fix direction:** purge/exclude prior-day closed positions in `get_total_realized_pnl`, or have `reset_daily` reset a daily-anchored baseline the sync respects.

---

### P1-4 — Live partial fill then complete fill under-tracks position size
**Where:** `order_manager.on_order_fill` `:1272-1296`. On the first (partial) fill the position is created at the partial qty. `_handle_partial_fill` and `_handle_complete_fill` both call this callback (`order_tracker.py:289,272`). On the subsequent COMPLETE, `existing is not None` → the branch is skipped, so quantity is **never upgraded to the full filled amount**.

**Effect (live only — paper fills are atomic COMPLETE):** on illiquid options/F&O that fill in pieces, `PositionManager` manages fewer units than the broker actually holds. The residual broker quantity has no SL/target/trailing protection and no exit. Naked unmanaged exposure + understated P&L.
**Fix direction:** on fill update for an existing position, reconcile `remaining_quantity`/`original_quantity` to cumulative `filled_quantity`.

---

### P1-5 — Daily loss limit + Kelly use GROSS P&L (fees never subtracted from in-memory realized)
**Where:** `position.realized_pnl` / `total_realized_pnl` (`:286-295`) are gross; fees are only subtracted into the **DB record** `net_total_pnl` (`:2041`). `get_total_realized_pnl` (`:2217`) sums the gross figure, which feeds `RiskManager.update_pnl` → `check_daily_loss_limit`.

**Effect:** the live daily-loss governor sees losses smaller than reality by the cumulative fee drag, so it trips later than the configured limit. Compounds with P1-3.
**Fix direction:** subtract attributed fees from the realized figure used for risk sync, or track a net realized counter.

---

### P1-6 — Live positions are NOT flattened on shutdown
**Where:** `shutdown()` `main.py:2291` — `close_all_positions` is gated on `trading_mode == "paper"`. In live mode a crash/SIGTERM/redeploy leaves open MIS positions with the bot dead and no in-process trailing/SL running.

**Effect:** open live risk with no manager. Mitigated only by the 15:15 scheduler square-off *if the process is still alive at 15:15*. A mid-session restart between entry and 15:15 = unmanaged position.
**Fix direction:** decide policy explicitly. If broker-side GTT/SL-M orders back every position, this is acceptable; if not, flatten on shutdown in live too (or require bracket/cover orders).

---

### P2-7 — Duplicate / ambiguous tier-4 trailing-stop logic
**Where:** each tick computes both `check_tiers(...)` (which itself handles the tier-4 trailing case, `partial_profit.py:315`) and `_check_tier_4_trailing_stop(...)` (`position_manager.py:1245`). Two independent detectors for the same exit with potentially different thresholds; whichever wins determines whether the close is logged (see P0-2).
**Fix direction:** single source of truth for tier-4 trailing.

---

### P2-8 — Paper LIMIT orders assume 100% fill at the limit price
**Where:** `paper_broker._calculate_fill_price:203` returns `limit_price` unconditionally for LIMIT orders. Entry optimizer routes PULLBACK/BREAKOUT/VWAP entries as LIMIT (`order_manager.py:858`).
**Effect:** paper always gets the favorable limit fill; live LIMITs may not fill → different (worse) trade selection and fill rate live vs paper. Optimistic paper bias.
**Fix direction:** model limit fills against subsequent bar range, or treat unfilled limits as missed.

---

### P2-9 — Bid/ask cache has no staleness guard
**Where:** `_get_ltp` rejects ticks >60s old (`:150`), but `_calculate_fill_price` uses `_bid_cache`/`_ask_cache` directly with no age check (`:207-214`). A stale quote could fill even when LTP is correctly rejected. (LTP>0 is still required first via the MARKET guard at `place_order:324`, which limits exposure, but the bid/ask values themselves aren't freshness-checked.)
**Fix direction:** timestamp bid/ask and apply the same 60s guard.

---

## PART 2 — WHAT IS SOLID (why it can work)

- **Single execution code path** for paper and live behind `create_broker` — the core promise of the architecture holds; strategy/risk/position logic is broker-agnostic.
- **Stale-LTP fill rejection** (`paper_broker.py:131-156`) — directly closes the May-14 "₹95 fills a ₹815 option" class of disaster.
- **Duplicate-signal defense is layered**: position-exists check (`order_manager.py:591`), in-flight `_pending_instruments` lock (`:616`), and per-bar dedup in the engine (`engine.py:790-796`). Genuinely guards the double-entry/self-trip bug.
- **Combo atomicity guard** (`order_manager.py:1089-1130`) — unwinds filled legs if any straddle/condor leg fails, preventing naked option exposure. Good defensive design.
- **Fees are real and itemized** (brokerage + STT + GST + exchange + SEBI + stamp), and the DB trade record stores net-of-fee P&L — for the trades that *do* go through `_close_position`.
- **Layered risk**: pre-trade notional/position-count/deployment/circuit-limit checks, per-strategy daily-loss quarantine, global consecutive-loss circuit breaker with persisted state across restarts, hard kill switch on daily-loss breach. The *structure* is institutional-grade.
- **Self-deadlock in tier-4 trailing already fixed** (commit 7ab0529) and verified end-to-end.
- **Same-bar-exit guard** prevents entry/exit on the identical bar in backtest (`:1192`).
- **Agent pipeline can only shrink size, never grow it** (`order_manager.py:773-776`) — LLM can't bypass the risk check by inflating qty. Correct guardrail.

---

## PART 3 — PREMORTEM: "It's 3 months in and the account is down. What happened?"

**Most likely failure modes, ranked:**

1. **The books lied (P0-2).** A big chunk of trades closed through the tier system, never hit the `trades` table, so the dashboard/win-rate looked fine while fees and small tier losses quietly accumulated off-book. Risk governor under-counted losses and never throttled. → *Highest-probability silent killer.*
2. **Paper looked better than live ever could (P0-1, P2-8).** Backtest/paper overstated P&L via dropped exit slippage and guaranteed limit fills. Live underperformed the paper that justified go-live. The 3-month replay that greenlit Monday inherits both biases.
3. **Risk governor mis-fired (P1-3, P1-5).** Daily loss limit running on cumulative gross P&L either let a bad day run past the intended stop, or jammed the kill switch on at SOD and missed good days. Either way, realized risk ≠ configured risk.
4. **Naked live exposure (P1-4, P1-6).** A partial fill or a mid-session restart left an unmanaged position that gapped against the account with no SL in the loop.
5. **Strategy edge was thin to begin with.** Even with all bugs fixed: several strategies sit near PF≈1.0 (Supertrend equity), and short-straddle strategies carry fat tail risk that 3 months may not have sampled (no policy-day vol shock in-window). News-blackout + IV gates help but don't remove gap risk.

**Why it can still succeed:**
- The dangerous bugs are **accounting/gating bugs, not strategy-logic bugs** — the entry/exit *decisions* are sound; the *measurement and governance* around them leak. Fixable without touching alpha.
- Capital concentration and straddle SL were already de-risked this session.
- Defensive scaffolding (stale-price reject, combo unwind, dedup, kill switch, per-strategy quarantine) is unusually complete for a retail bot — the failure surface is narrower than typical.
- Paper-forward validation discipline is in place; once P0-1/P0-2 are fixed, paper P&L becomes a trustworthy live proxy.

**Minimum bar before real capital (recommended gate):**
fix **P0-2** (trade logging/fees/CB on tier close) and **P0-1** (exit fill price) first — without them you are flying blind on your own P&L. P1-3/P1-4/P1-6 before scaling size.

---

## Appendix — file:line index
- Tick entry: `main.py:2096`
- Signal forward + wave5 gate: `main.py:758`
- Order pipeline: `order_manager.py:548`
- Async live fill: `order_manager.py:1250`
- Exit P&L (3 paths): `position_manager.py:1451 / 1535 / 1953`
- Tier silent close: `position_manager.py:1561`
- Risk sync (cumulative): `position_manager.py:2255`
- Daily reset: `risk_manager.py:353`, `main.py:1929`
- Paper fill/slippage: `paper_broker.py:185`
- Square-off 15:15 / summary 15:30 / SOD 9:14: `main.py:1484-1511`
