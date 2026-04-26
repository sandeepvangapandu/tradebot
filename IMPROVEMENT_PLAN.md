# Trading Bot Improvement Plan

Generated: 2026-04-26. Source: STRATEGY_ANALYSIS_REPORT.md + current code audit.

---

## Phase 0 — Reality check (already done in code, report stale)

- [x] Brokerage / STT / GST / stamp / SEBI fees in `src/backtest/engine.py:304-384`
- [x] Slippage on entry + exit in `engine.py:850-862`
- [x] Round-trip cost + premium-level slippage modeled

Remaining gap: fees realistic, but **option premium model is still a 0.5% × delta-0.5 proxy** (`engine.py:443`). All theta / IV / gamma effects missing.

---

## Phase 1 — Backtest fidelity (1–2 weeks)

### 1.1 Black-Scholes premium engine
- New module: `src/research/options_pricing.py`
  - `bsm_price(S, K, T, sigma, r, kind)`
  - `greeks(S, K, T, sigma, r)` → delta, theta, vega, gamma
- IV source: India VIX as proxy (or NSE bhavcopy IV per strike)
- Replace `engine.py:847-867` premium calc with BSM(S_t, K, T_remaining, σ_t)
- Captures theta automatically per bar — kills the "options decay" objection in report sec 3

### 1.2 Volatility-aware slippage
- `engine.py:852` flat pct → `slippage = max(fixed_ticks_paisa, pct × premium) × vix_multiplier`
- VIX>20 → 1.5× spread; VIX<12 → 1.0×

### 1.3 Data expansion
- 6 months 1m BankNifty + Nifty + India VIX
- Cache parquet under `data/historical/`
- Walk-forward windows: 3mo train / 1mo test, roll monthly

### 1.4 Per-strategy isolated runs
- Harness flag `--strategy-only <name>` (single-strategy mode, bypass confluence)
- Auto-disable any strategy with WR<45% or PF<1.0
- Output per-strategy report card under `data/backtest_results/`

---

## Phase 2 — New strategies + regime gating (2–3 weeks)

### 2.1 Wire VIX regime filter
- `src/research/market_regime.py` exists, not wired
- Add `RegimeGate` in `src/strategy/engine.py` evaluation pipeline:
  - VIX<12 → only mean-reversion strategies allowed
  - VIX 12–18 → trend strategies allowed
  - VIX>18 → only theta-positive (condor / straddle-sell) allowed
- Config knob in `config/pro_trader_config.json`

### 2.2 Iron Condor strategy
- New: `src/strategy/iron_condor.py`
  - Sell ATM±1 CE/PE, buy ATM±3 CE/PE wings
  - Entry 10:30 IST, exit at 50% max profit or 15:00
  - Defined risk = wing width − credit
- Reuse `expiry_straddle.py` infra for multi-leg orders

### 2.3 Multi-timeframe trend gate
- Universal pre-filter in `evaluate_bar_sync` at `engine.py:1258`
- Daily EMA20 vs EMA50, 15m EMA20 alignment, 5m pullback to EMA21
- Reject signals against higher-TF trend

### 2.4 ML validator gate
- `src/research/ml_validator.py` exists — wire as final pre-order check
- Block signal if predicted edge < 0.55 confidence

---

## Phase 3 — Performance / runtime optimization

Found in audit:

### 3.1 Eliminate per-bar IndicatorEngine rebuild (CRITICAL)
- `src/strategy/engine.py:975, 1078, 1125` instantiate `IndicatorEngine(df)` **per bar**
- Comment at `:1349` says "fresh evaluator per bar avoids stale cache" — fix the cache, not bypass it
- Cost: O(N) bars × O(N) recompute = **O(N²) backtest time**
- Fix:
  - Single persistent `IndicatorEngine` per symbol
  - Incremental update via `engine.update(new_bar)` (already exists at `:74`)
  - Indicators use rolling-window deques, not full-df recompute
- Expect 10–50× backtest speedup

### 3.2 Vectorize VWAP
- `indicators.py:299` Python `for date, group in groupby` over days
- Replace with single `groupby(date).cumsum()` vectorized

### 3.3 Drop iterrows in harness
- `src/backtest/harness.py:547` `for i, (bar_time, bar) in enumerate(self._master_df.iterrows())`
- iterrows is ~50× slower than `itertuples()` or `.values` ndarray loop
- For pure backtest, prefer numpy column arrays

### 3.4 Indicator caching
- `IndicatorEngine._cache` exists `:48` but unused effectively
- Memoize on `(indicator_name, params, last_bar_ts)` — hit cache when only newest bar added

### 3.5 Parallel multi-symbol backtest
- `concurrent.futures.ProcessPoolExecutor` over symbols
- One process per symbol, merge trade list at end

### 3.6 Numba JIT hot loops
- ATR, Supertrend, RSI inner loops → `@njit`
- 5–20× per indicator on long series

### 3.7 Profile first
- `python -m cProfile -o bt.prof src/backtest/runner.py …`
- `snakeviz bt.prof` — confirm 3.1 is the hotspot before optimizing elsewhere

---

## Phase 4 — Paper validation (4+ weeks)

- Enable top-3 by Phase 1.4 ranking
- 20 trading days minimum
- Compare paper vs backtest expectancy — drift > 20% means model bad
- Telegram daily P&L digest

---

## Phase 5 — Live (only after 3mo profitable paper)

- ₹25k–50k capital
- 1 strategy live first
- Scale only after 30+ trades positive expectancy

---

## Priority order

1. **3.1 indicator rebuild fix** — unblocks fast iteration on everything else
2. **1.1 BSM** — core fidelity
3. **1.4 isolated strategy runs** — finds the 1–2 strategies with edge
4. **2.1 VIX regime gate** — kills wrong-regime signals
5. **2.2 Iron Condor** — adds theta-positive edge
6. Rest in order
