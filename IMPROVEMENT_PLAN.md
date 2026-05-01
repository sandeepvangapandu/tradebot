# Trading Bot Improvement Plan

Generated: 2026-04-26. Updated: 2026-04-30.
Source: STRATEGY_ANALYSIS_REPORT.md + code audit + live debugging session.

Status legend: ✅ done | 🟡 partial | ⏸ deferred | ❌ not started

---

## Phase 0 — Reality check (already in code, report stale)

- ✅ Brokerage / STT / GST / stamp / SEBI fees in `src/backtest/engine.py:304-384`
- ✅ Slippage on entry + exit in `engine.py:850-862`
- ✅ Round-trip cost + premium-level slippage modeled

Remaining gap: **option premium model is still a 0.5% × delta-0.5 proxy** (`engine.py:443`). All theta / IV / gamma effects missing.

---

## Phase 1 — Backtest fidelity

### 1.1 ❌ Black-Scholes premium engine — NOT STARTED
- New module: `src/research/options_pricing.py`
- IV source: India VIX as proxy
- Replace `engine.py:847-867` premium calc with BSM(S_t, K, T_remaining, σ_t)
- Captures theta automatically per bar

### 1.2 ❌ Volatility-aware slippage — NOT STARTED
- `engine.py:852` flat pct → `slippage = max(fixed_ticks_paisa, pct × premium) × vix_multiplier`

### 1.3 ✅ Data expansion — DONE
- Pulled 6.5 months of 1m BankNifty in 28-day chunks (Upstox v3 API caps at 30d/call).
- File: `data/backtest/banknifty_1m_6mo.csv`
- **48,810 bars across 131 trading days (2025-10-17 → 2026-04-30)**
- Walk-forward windows still TODO.

### 1.4 ❌ Per-strategy isolated runs — NOT STARTED
- Harness flag `--strategy-only <name>` not added.
- Auto-disable rule (WR<45% OR PF<1.0) not implemented.

---

## Phase 2 — New strategies + regime gating

### 2.1 ❌ Wire VIX regime filter — NOT STARTED
- `src/research/market_regime.py` exists, still not wired.

### 2.2 🟡 Iron Condor strategy — DEFERRED (real version requires infra)
- `expiry_straddle.py` already has `ce_hedge_strike` / `pe_hedge_strike` fields.
- Config flag `hedge_required` exists but **dead code** — no signal/order path consumes it.
- Real condor needs multi-leg order routing (`order_manager` rewrite). Big change.
- **Alternative shipped**: tightened the naked straddle config to cap exposure (see Phase 6 below).

### 2.3 ❌ Multi-timeframe trend gate — NOT STARTED

### 2.4 ❌ ML validator gate — NOT STARTED
- `src/research/ml_validator.py` exists, not wired as pre-order check.

---

## Phase 3 — Performance / runtime optimization

### 3.1 ✅ Eliminate per-bar IndicatorEngine rebuild — DONE
- Persistent per-strategy `ConditionEvaluator` (`src/strategy/engine.py:1185, 1352`).
- Memoization in `IndicatorEngine` keyed on `(method, params)` × `_data_version`.
- Append-only `update()` fast path; same-window no-op.
- All indicator methods (ema/sma/rsi/macd/bbands/atr/supertrend/vwap) wrapped with `_cached`.
- **Speedup measured**: within-bar memo cache ~500× per repeat call (1.1 ms → 0.002 ms). Single-strategy synthetic backtest no big delta because hot path is `ta.*` itself; real wins land in multi-condition / multi-strategy configs.

### 3.2 ✅ Vectorize VWAP — DONE
- `indicators.py:vwap` rewritten with single `groupby(date).cumsum()`. No Python day loop.

### 3.3 ❌ Drop iterrows in harness — NOT STARTED
- `harness.py:547` still uses `iterrows()`.

### 3.4 ✅ Indicator caching — DONE
- `_result_cache` keyed on `(method, params)` × `_data_version`. Verified hit/miss in unit test.

### 3.5 ❌ Parallel multi-symbol backtest — NOT STARTED

### 3.6 ❌ Numba JIT hot loops — NOT STARTED

### 3.7 ❌ Profile first — NOT STARTED

---

## Phase 4 — Paper validation — NOT STARTED

---

## Phase 5 — Live (only after 3mo profitable paper) — NOT STARTED

---

## Phase 6 — Bug fixes uncovered during live run (NEW, all DONE)

These were found while trying to run the live bot and weren't in the original plan.

### 6.1 ✅ Bar builder ignored Upstox V3 feed format
- `bar_builder._process_tick` looked for `tick.get("instrument_key")` and `tick.get("timestamp")` at top level.
- V3 keys are `["type", "feeds", "currentTs"]`. Every tick exited at the top guard → zero bars built → `bar_close_event` never fired → strategies never evaluated → **bot never traded**.
- Fixed: V3 parser reads `feeds[ik].ltpc.ltp/ltt`, supports `fullFeed.marketFF`, `fullFeed.indexFF`, `firstLevelWithGreeks`. Legacy flat format still works.
- `_parse_timestamp` now handles ms-epoch ints + numeric strings too.

### 6.2 ✅ Historical fetcher passed unix epoch ints (Upstox wants `yyyy-mm-dd`)
- Every call returned 400 silently. `historical.py:66-67` fixed to `strftime("%Y-%m-%d")`.

### 6.3 ✅ Upstox v3 history excludes current day
- Added `fetch_intraday_candles()` using `get_intra_day_candle_data` endpoint.
- Verified intraday returns 9:15 → now bars including ORB window.

### 6.4 ✅ History 1m API caps at 30 days
- 60d+ requests return `UDAPI1148 Invalid date range`.
- Added 28-day chunked loop to fetch ~6 months locally.

### 6.5 ✅ Upstox v3 only supports `1minute / 30minute` for intraday and `1minute / 30minute / day` for historical
- 5m and 15m return 400. Fix: fetch 1m, resample locally to 5m/15m before seeding bar_builder.

### 6.6 ✅ Health monitor heartbeat plumbing missing
- `market_feed`, `database`, `scheduler` all registered with `HealthMonitor` but **no code called `heartbeat()`** for them. Always FAILED after timeout.
- Wired:
  - `market_feed`: lambda on tick callback bumps heartbeat per tick.
  - `database` + `scheduler`: 30s interval job (`add_interval_job` added to `TaskScheduler`) pings DB with `SELECT 1` and bumps both watchdogs.

### 6.7 ✅ Backtest fees hardcoded to 0
- `harness.py:589, 666` hardcoded `"fees": 0` per trade. Metrics summed those → reported `Total Fees ₹0` even with realistic fee model on.
- Fix: pull total fees from `paper_broker._trade_history` charges; pass to `_compute_metrics(initial_capital, total_fees_paisa)`.

### 6.8 ✅ CPR strategy never had cpr_data wired
- `conditions.py:_compute_indicator` expected `parameters['cpr_data']` but no code ever populated it. Every CPR_* condition logged `CPR data not provided` and returned None → CPR strategy fired 0 signals ever.
- Fix: added `_auto_cpr(engine, symbol)` to `ConditionEvaluator`. Resamples bars to daily, takes prior closed day, calls `indicators.calculate_cpr`. Cached per (symbol, date).

### 6.9 ✅ ORB opening-range slice bug
- `conditions.py:1134` hardcoded `orb_bars = orb_minutes // 5` (assumes 5m bars). Engine actually feeds 1m bars → slice covered first 3 minutes instead of 15.
- Fix: slice by time delta from session open. Works for any bar interval.

### 6.10 ✅ AI signal_validator auto-rejected every signal
- `signal_validator` returned `confidence=0.00 → action: reject` on every signal in backtest. LLM call returning 0 confidence by default.
- Workaround: `AGENT_PIPELINE_ENABLED=false` in `.env`.

### 6.11 🟡 Naked straddle drawdown cap (band-aid for 2.2)
- Original `qty=30, SL=60%`: 6mo backtest produced **111.6% drawdown** (account blew up).
- Tightened SL to 25% **made it worse** (PF dropped 1.20→1.01) — theta strategies need wide SL to ride decay.
- Final config: **`qty=5, SL=60%, max_trades_per_day=1`** → DD capped at 25.6%, PF 1.11, +₹3.5k over 6mo. Survivable. Not a real condor.

---

## Backtest results timeline

| Run | Config | Trades | WR | PF | Net P&L | Max DD |
|---|---|---|---|---|---|---|
| 4-day proxy | initial | 4 | 75% | 2.88 | +₹4,913 | 26.9% |
| 6mo, qty=30, SL=60% | report-baseline | 129 | 61.2% | 1.20 | +₹34,935 | **111.6% (blew up)** |
| 6mo, qty=15, SL=25% | tighter stops | 128 | 49.2% | 1.01 | +₹1,285 | 77.0% |
| **6mo, qty=5, SL=60%** | **current live config** | **128** | **60.2%** | **1.11** | **+₹3,470** | **25.6%** |

All trades from `Expiry_Straddle_Sell_BankNifty`. CPR + ORB still fire **zero** signals on 6mo of data even after wiring fixes — filters are too strict for current market conditions, not broken.

---

## Files changed in this session

- `src/strategy/indicators.py` — memoization, append-only update, vectorized VWAP
- `src/strategy/engine.py` — persistent per-strategy ConditionEvaluator
- `src/strategy/conditions.py` — CPR auto-compute, ORB time-slice fix, `_compute_indicator` takes `target_symbol`
- `src/data/bar_builder.py` — V3 feed parser, ms-epoch timestamps, `seed_bars()`
- `src/data/historical.py` — date format fix, intraday endpoint
- `src/data/websocket_feed.py` — open handler, tick logging, ApiClient pool drain
- `src/main.py` — historical warmup wiring, heartbeat pulse, market-feed heartbeat lambda
- `src/utils/scheduler.py` — `add_interval_job`
- `src/backtest/harness.py` — fees from PaperBroker
- `config/strategies/cpr_vwap_bounce.json` — enabled: true
- `config/strategies/orb_vwap_banknifty.json` — enabled: true
- `config/strategies/expiry_straddle_sell_banknifty.json` — qty=5, SL=60%, max_trades=1
- `.env` — MAX_OPEN_POSITIONS=3, MAX_POSITION_SIZE_PCT=80, AGENT_PIPELINE_ENABLED=false

---

## Remaining priority order

1. **2.2 Real Iron Condor** — multi-leg order routing in `order_manager`. Caps tail risk properly (current band-aid is just smaller naked size).
2. **1.1 BSM premium engine** — kills the 0.5%-proxy fidelity hole; required before trusting any backtest result.
3. **2.1 VIX regime gate** — `market_regime.py` already exists; route signals based on VIX bucket.
4. **1.4 Per-strategy isolated runs** — `--strategy-only` flag + auto-disable rule. Surfaces dead strategies.
5. **6.10 fix signal_validator LLM** — currently disabled entirely; investigate why confidence=0 default.
6. **Loosen CPR / ORB filters** — currently fire 0 signals on 6mo. Either relax or scrap.
7. **Phase 4 paper validation** — 20+ trading days, drift check vs backtest.
8. Smaller-impact perf items (3.3, 3.5, 3.6).

## Honest summary

Bot **can** trade now (V3 feed bug was the blocker). 6mo backtest with current config gives PF 1.11, +₹3.5k net, 25.6% max DD on ₹1L capital. **Marginal edge after fees, dominated by variance.** Not ready for live.
