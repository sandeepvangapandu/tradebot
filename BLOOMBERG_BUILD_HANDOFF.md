# Bloomberg-Level Data + Execution Build — Handoff Document

**Last updated:** 2026-05-10 (wiring agent complete)
**Branch:** `feature/bloomberg-data-execution` (pushed to GitHub `sandeepvangapandu/tradebot`)
**Audience:** Next Claude Code session picking up this work — read this entire document before doing anything.

---

## TL;DR (read this first)

The user is building an autonomous Indian-markets trading bot (Upstox broker, NSE/BSE/NFO/MCX/CDS) that should reach "Bloomberg-level" data depth before going live. Current scope: paper trading only (Phase 1 of project plan). Top-10 NSE equities + NIFTY/BANKNIFTY indices for trading universe.

**8 of 13 build phases are complete.** All phase work consists of independent modules + Postgres migrations + tests. No `main.py` integration has happened yet — that is the next-session "wiring" phase. The bot still runs from `main` branch unchanged. New work lives entirely on `feature/bloomberg-data-execution`.

**Test suite:** 1817 passing, 12 skipped (env-dependent). 5 previously failing tests in conditions_news.py are no longer failing (conditions_news.py patching fixed prior to this session).

**Critical next step:** Wave 5 (cross-phase intelligence — confluence engine, rejection filter, regime router, Kelly sizer) → Phase F (portfolio risk) → Phase E (execution upgrades) → wiring agent (touches `main.py` to integrate everything) → paper-forward testing → optional backtester rebuild.

---

## Project Context

### What the user is building

Autonomous trading bot for Indian markets:
- **Broker:** Upstox (API v2 + V3 WebSocket). User account `EY5091` enabled for all segments — NSE, NFO, BSE, BFO, NCD_FO (currency), NSE_COM (commodities), MCX (legacy).
- **Markets:** Indian stock + derivatives. NSE primary.
- **Mode:** Paper trading only for Phase 1. Live deferred until paper-forward validation passes.
- **Capital:** ₹5 lakh paper (decided 2026-05-08).
- **Universe:** Top 10 NSE equities + NIFTY/BANKNIFTY indices.
  - Top 10 (locked, ISIN-verified vs live instrument master): RELIANCE, HDFCBANK, ICICIBANK, TCS, INFY, HINDUNILVR, ITC, AXISBANK, KOTAKBANK, SBIN.
- **Strategies (already loaded by bot, all options-type):** Expiry_Straddle_Sell_BankNifty, ORB_VWAP_BankNifty, CPR_VWAP_Bounce_BankNifty.

### What "Bloomberg-level" means in this context

Real institutional-grade data pipeline + execution intelligence — NOT the literal Bloomberg Terminal. Phased breakdown:

| Phase | Scope |
|---|---|
| Phase 0 | Postgres + Supabase storage foundation |
| Phase D | Universe scanner (Top-10 wiring, daily liquidity/vol/RS rank) |
| Phase A | Microstructure (L2 depth, CVD, options chain OI/IV/PCR, volume profile) |
| Phase B | Cross-instrument context (sectors, VIX regime, macro overlay USDINR/Crude/Gold, FII/DII flows, bond yield) |
| Phase C | Events (corporate actions, earnings calendar, news RSS+sentiment, block deals + insider trading) |
| Wave 5 | Cross-phase intelligence: confluence engine, rejection filter, regime router, Kelly sizer |
| Phase F | Portfolio risk (correlation matrix, beta-weighted exposure, Greek aggregation, VaR/CVaR, sector caps) |
| Phase E | Execution upgrades (smart order router, multi-leg basket, pre-trade margin check, reconciliation) |
| Wiring | Single agent touches `main.py` to integrate every phase module into runtime |
| Forward test | 4 weeks paper trading on live market data |
| Backtester | Decide whether to rebuild for tick+L2 replay based on forward results |

### User's locked-in decisions

| Topic | Decision |
|---|---|
| Universe | Top 10 (above) |
| News source | Free RSS scraping (Moneycontrol, ET, Reuters, BusinessLine). Paid APIs deferred until live. |
| Storage backend | Supabase (free tier) + local Postgres dev mirror (Homebrew Postgres 16) |
| Postgres tier | Free; tiered hot/cold retention to stay under 500MB DB |
| Backtester strategy | Paper-forward test first (4 weeks); rebuild for tick+L2 only if microstructure signals prove out |
| Hardware | Mac, 32GB RAM, 2TB SSD; bot RAM budget ~6-8GB |
| Sleep policy | `caffeinate` keeps Mac awake for Indian market hours (09:15-15:30 IST = 23:45-06:00 EDT) |
| Build approach | Full A→F sequential (no partial phases). Aggressive parallel sub-agent dispatch within each wave. |
| Phase order | Postgres → D → A → B → C → Wave 5 cross-phase → F → E → wiring → forward test |

### Hardware + ops

- **Mac:** 32GB RAM, 2TB SSD, latest macOS.
- **Local Postgres 16** running via Homebrew. Service: `brew services start postgresql@16`. Connection: `postgresql://sandeepvangapandu@localhost:5432/tradebot`.
- **Supabase project:** `https://laaqbjmsbjwgqyskechs.supabase.co`. GitHub integration (Mode 1 — migrations-only) watches branch `feature/bloomberg-data-execution`. Pushes to that branch auto-deploy `supabase/migrations/*.sql`.
- **Bot deployment:** Mac-only. No cloud yet. Move-to-cloud planned post-live-go-decision.

### Credentials and config

- Stored in `/Users/sandeepvangapandu/Downloads/Trading/.env` (gitignored, NOT readable from agent context).
- Keys present:
  - `SUPABASE_URL` (40 chars)
  - `SUPABASE_PUBLISHABLE_KEY` (46 chars — public-facing)
  - `SUPABASE_SERVICE_ROLE_KEY` (219 chars — secret JWT)
  - `SUPABASE_DB_URL` (85 chars — direct connection)
  - `LOCAL_DB_URL=postgresql://sandeepvangapandu@localhost:5432/tradebot`
  - `ENV=dev`
- Storage layer (`src/storage/db.py`) routes via `ENV` env var: `dev` → `LOCAL_DB_URL`, anything else → `SUPABASE_DB_URL`.
- Upstox token cached at `data/token_cache.json`. Re-auth via `src/auth/auto_login.py`.
- **DO NOT** ask the user to paste secrets in chat. Always tell them to update `.env` and read from there.

---

## Phase Status

| Phase | Status | Migrations | Modules | Tests | Commits |
|---|---|---|---|---|---|
| **0 Foundation** | ✅ Complete | `20260508000000_phase0_foundation.sql` | `src/storage/{db,models,migrate,archiver}.py` | 20 storage tests | `2a3dc11` |
| **D Universe Scanner** | ✅ Complete | `20260508000001_phase_d_universe.sql` | `src/research/universe_scanner.py` + `run_universe_scan.py` | 25 tests | `d091eda`, `80f0f9d`, `16544a2`, `b60d601` |
| **A.1 L2 Depth** | ✅ Complete | `20260508000010_phase_a1_depth_metrics.sql` | `src/data/depth_feed.py` | 31 tests | `7147212`, `a06cd6e`, `f8b0956` |
| **A.2 Tick Metrics + CVD** | ✅ Complete | `20260508000011_phase_a2_tick_metrics.sql` | `src/data/tick_metrics.py` + `src/strategy/conditions_microstructure.py` | 43 tests | `85dd641`, `e47eb5c`, `3c114f2`, `6e50675` |
| **A.3 Options Chain OI/IV/PCR** | ✅ Complete | `20260508000012_phase_a3_options_chain.sql` | `src/data/options_chain_feed.py` + `src/strategy/conditions_options.py` | 33 tests | `ae2ebef`, `6e3b69d`, `44b9ef6` |
| **A.4 Volume Profile** | ✅ Complete | `20260508000013_phase_a4_volume_profile.sql` | `src/indicators/volume_profile.py` + `src/strategy/conditions_volume_profile.py` | 53 tests | `e8f73d5`, `015cb32`, `edfd8c4` |
| **B.1 Sector Rotation** | ✅ Complete | `20260508000020_phase_b1_sector_rotation.sql` | `src/research/sector_rotation.py` + `src/strategy/conditions_sector.py` | 32 tests | `35da6b5`, `4a6075f`, `658fa5f` |
| **B.2 VIX Regime** | ✅ Complete | `20260508000021_phase_b2_vix_regime.sql` | `src/research/vix_regime.py` + `src/strategy/conditions_vix.py` | 53 tests | `3fd1241` |
| **B.3 Macro Overlay** | ✅ Complete | `20260508000022_phase_b3_macro_overlay.sql` | `src/research/macro_overlay.py` + `src/strategy/conditions_macro.py` | 57 tests | `57e7cb8`, `f693608`, `627fa92`, `a178cb4` |
| **B.4 FII/DII Flows + Bond Yield** | ✅ Complete | `20260508000023_phase_b4_flows_yields.sql` | `src/data/flow_scraper.py` + `src/research/flow_regime.py` + `src/strategy/conditions_flows.py` | 72 tests | `1aa3f3e`, `a66c31b`, `1f7521f`, `e53b146`, `7ea4670` |
| **C.1 Corporate Actions** | ✅ Complete | `20260508000030_phase_c1_corporate_actions.sql` | `src/data/corporate_actions_scraper.py` + `src/research/corporate_calendar.py` + `src/strategy/conditions_corporate.py` | (in C-bundle) | `2d5976d`, `736293c`, `cc3e2a4` |
| **C.2 Earnings Calendar** | ✅ Complete | `20260508000031_phase_c2_earnings_calendar.sql` | `src/data/earnings_scraper.py` + `src/research/earnings_calendar.py` + `src/strategy/conditions_earnings.py` | 55 tests | `611270f` |
| **C.3 News RSS + Sentiment** | ✅ Complete (5 known test fails) | `20260508000032_phase_c3_news_sentiment.sql` | `src/data/news_rss_scraper.py` + `src/research/sentiment_classifier.py` + `src/research/news_query.py` + `src/strategy/conditions_news.py` | (in C-bundle) | `cc3e2a4` |
| **C.4 Block + Insider** | ✅ Complete | `20260508000033_phase_c4_block_insider.sql` | `src/data/block_deals_scraper.py` + `src/data/insider_scraper.py` + `src/research/insider_signals.py` + `src/strategy/conditions_insider.py` | (in C-bundle) | `cc3e2a4` |
| **Wave 5 Cross-Phase** | ✅ Complete (partial — see CHANGE_LOG) | `20260508000040-43` | `src/strategy/{confluence_engine,rejection_filter,regime_router}.py`, `src/risk/kelly_sizer.py` | (in prior commits) | `28ae2bf` |
| **F Portfolio Risk** | ✅ Complete | `20260508000050-51` | `src/risk/{portfolio_risk,greek_aggregator}.py` | (in prior commits) | `28ae2bf` |
| **E Execution Upgrades** | ✅ Complete | `20260508000060-63` | `src/execution/{smart_router,reconciler,slippage_monitor,order_validator}.py` | (in prior commits) | `28ae2bf` |
| **Wiring** | ✅ Complete (see CHANGE_LOG) | - | `src/main.py`, `src/strategy/engine.py` | 8 new (test_main_wiring.py) | `3e15e01` |
| **Forward test** | ⏳ Pending | - | - | - | - |
| **Backtester rebuild** | ⏳ Optional | - | - | - | - |

Plus auxiliary work completed:
- **Mock fixtures + scenario harness** (`tests/fixtures/`): 19 files, 125 smoke tests. Includes 10 named scenarios (bull_open_breakout, earnings_blackout, liquidity_collapse, correlated_overload, vix_spike_regime, partial_fill_recovery, connection_drop, eod_squareoff, news_negative_reject, sector_rotation_winner). Commits `e9759f7`, `4ff03fe`, `7ca0638`, `9bc08ff`.
- **Legacy test fixes** (~96 tests): test_upstox_live (43), test_strategy_engine (8 numpy bool), test_trade_analyzer (10), test_market_regime (5 fixed/skipped), test_graceful_degradation (5), test_trade_scorecard (4), test_signal_validator (3), test_indicators (3), test_position_sizer (2 + source bug fix), test_pattern_recognition (2), test_partial_profit (2), test_integration (2), test_cpr_vwap_strategy (2), test_technical_analyzer (1), test_pipeline (1 + source bug fix), test_order_manager (1 + source bug fix), test_report_generator (8 — 4 skipped requiring kaleido).
- **Source bugs fixed during legacy test work** — see "Source Bugs Discovered" section below.

---

## Architecture Map

### Data flow (intended after wiring)

```
EOD scheduler (cron) ──► [Phase B.4 flow_scraper, Phase C.1 corp_actions, Phase C.2 earnings, Phase C.4 blocks/insider]
                          ──► Postgres EOD tables

Pre-market 08:30 IST ──► [Phase D universe_scanner.rank_all]
                         ──► writes universe_daily_snapshot, liquidity/volatility/RS ranks

WebSocket tick (live) ──► [bar_builder] ──► OHLCV bars
                       ├─► [Phase A.1 depth_feed] ──► depth_snapshots + depth_metrics_minute
                       ├─► [Phase A.2 tick_metrics] ──► tick_metrics_minute + in-memory CVD
                       └─► [Phase A.4 volume_profile, intraday computation]

REST poll every 30s ──► [Phase A.3 options_chain_feed] ──► options_chain_snapshots + strike_oi_history

Hourly RSS poll ──► [Phase C.3 news_rss_scraper] ──► news_articles
                  └─► [sentiment_classifier] ──► news_sentiment_symbol

Strategy engine bar-close evaluation ──► [conditions_*.py modules read all of the above]
                                       ──► raw signal
                                       ──► [Wave 5 confluence_engine] ──► confluence score 0-100
                                       ──► [Wave 5 rejection_filter] ──► reject if spread too wide / blackout / news / correlation overload
                                       ──► [Wave 5 regime_router] ──► gate by VIX regime, sector rotation
                                       ──► [Wave 5 kelly_sizer + Phase F portfolio_risk] ──► sized order
                                       ──► [Phase E smart_router] ──► OrderManager ──► PaperBroker
                                       ──► fills written to fills + positions tables
                                       ──► [Phase E reconciler + slippage_monitor] ──► drift detection

Daily 23:59 ──► [Phase 0 archiver] ──► moves ticks/depth older than 7d to Parquet on local SSD
```

### File inventory (new this build)

**Source modules (new, all on `feature/bloomberg-data-execution`):**

```
src/storage/
├── __init__.py
├── db.py                            # async + sync engines, env-routed (LOCAL/SUPABASE)
├── models.py                        # SQLAlchemy 2.x ORM mappings
├── migrate.py                       # CLI runner for supabase/migrations/*.sql
└── archiver.py                      # cold-storage archiver (skeleton — full impl pending)

src/data/
├── depth_feed.py                    # Phase A.1 — L2 order book parser + minute aggregator
├── tick_metrics.py                  # Phase A.2 — aggressor classification + CVD
├── options_chain_feed.py            # Phase A.3 — REST poller, OI/IV/PCR/max-pain
├── flow_scraper.py                  # Phase B.4 — NSE/MC FII/DII + bond yield scraper
├── corporate_actions_scraper.py     # Phase C.1 — NSE corp action API
├── earnings_scraper.py              # Phase C.2 — Moneycontrol + NSE results
├── news_rss_scraper.py              # Phase C.3 — RSS 2.0/Atom parser
├── block_deals_scraper.py           # Phase C.4 — NSE block + bulk deals
└── insider_scraper.py               # Phase C.4 — NSE SAST/PIT disclosures

src/research/
├── universe_scanner.py              # Phase D — daily Top-10 ranking
├── run_universe_scan.py             # Phase D — CLI runner for scheduler
├── sector_rotation.py               # Phase B.1 — 9 sector indices + RS rank
├── vix_regime.py                    # Phase B.2 — LOW/NORMAL/HIGH/SPIKE classifier
├── macro_overlay.py                 # Phase B.3 — USDINR/Crude/Gold/Silver regime
├── flow_regime.py                   # Phase B.4 — FII/DII streak + combined signal
├── corporate_calendar.py            # Phase C.1 — query + blackout filter
├── earnings_calendar.py             # Phase C.2 — query + T-2/T+1 blackout
├── sentiment_classifier.py          # Phase C.3 — rule-based + lexicon, Top-10 alias matching
├── news_query.py                    # Phase C.3 — recent articles + aggregate sentiment
└── insider_signals.py               # Phase C.4 — block flow + promoter activity

src/strategy/
├── conditions_microstructure.py     # Phase A.2 — CVD divergence, aggressor flow
├── conditions_options.py            # Phase A.3 — PCR, IV percentile, max-OI
├── conditions_volume_profile.py     # Phase A.4 — POC/VAH/VAL spot tests
├── conditions_sector.py             # Phase B.1 — sector top/bottom quartile
├── conditions_vix.py                # Phase B.2 — VIX regime/percentile/spike
├── conditions_macro.py              # Phase B.3 — macro trend + cross-asset signals
├── conditions_flows.py              # Phase B.4 — FII buying, flow regime, yield
├── conditions_corporate.py          # Phase C.1 — corporate blackout
├── conditions_earnings.py           # Phase C.2 — earnings blackout, surprise
├── conditions_news.py               # Phase C.3 — sentiment, neg/pos news, volume spike
└── conditions_insider.py            # Phase C.4 — block buy/sell, promoter buying

src/indicators/
├── __init__.py
└── volume_profile.py                # Phase A.4 — POC/VAH/VAL bin algorithm

src/strategy/instrument_resolver.py  # Created in earlier session — options chain resolver
```

**Tests (new, mirror module structure):**

```
tests/storage/                   # 20 tests (Phase 0)
tests/research/                  # ~150+ tests covering all phases
tests/data/                      # ~150+ tests (NOTE: tests/data is gitignored, used `git add -f`)
tests/strategy/                  # ~150+ tests
tests/indicators/                # 22 tests (Phase A.4)
tests/fixtures/                  # 19 mock fixture files + 10 scenarios
tests/test_fixtures_smoke.py     # 125 fixture smoke tests
```

**Migrations (idempotent SQL):**

```
supabase/migrations/
├── 20260508000000_phase0_foundation.sql           # bot_config, schema_version, instruments, bars, ticks, depth_snapshots, orders, fills, positions, daily_pnl, signals, bot_runs
├── 20260508000001_phase_d_universe.sql            # universe_constituents, liquidity/volatility/rs_rank_daily, universe_daily_snapshot
├── 20260508000010_phase_a1_depth_metrics.sql      # depth_metrics_minute
├── 20260508000011_phase_a2_tick_metrics.sql       # tick_metrics_minute
├── 20260508000012_phase_a3_options_chain.sql      # options_chain_snapshots, options_strike_oi_history
├── 20260508000013_phase_a4_volume_profile.sql     # volume_profile_daily
├── 20260508000020_phase_b1_sector_rotation.sql    # sector_indices, sector_rank_daily (9 sectors seeded)
├── 20260508000021_phase_b2_vix_regime.sql         # vix_regime_daily, vix_regime_intraday
├── 20260508000022_phase_b3_macro_overlay.sql      # macro_instruments (USDINR/Crude/Gold/Silver, 4 seeded), macro_regime_daily
├── 20260508000023_phase_b4_flows_yields.sql       # fii_dii_flows_daily, bond_yield_daily, flow_regime_daily
├── 20260508000030_phase_c1_corporate_actions.sql  # corporate_actions
├── 20260508000031_phase_c2_earnings_calendar.sql  # earnings_calendar
├── 20260508000032_phase_c3_news_sentiment.sql     # news_sources (6 seeded), news_articles, news_sentiment_symbol
└── 20260508000033_phase_c4_block_insider.sql      # block_deals, bulk_deals, insider_trades
```

All migrations applied locally (`schema_version` table has 14 entries). Pushed to Supabase via auto-deploy on branch push.

---

## Test Status

**Final count:** 1473 passing, 5 failing, 12 skipped.

### 5 Known Failures (all in `tests/strategy/test_conditions_news.py`)

```
TestHasNegativeNewsRecent::test_true_when_negative_dominant
TestHasPositiveNewsRecent::test_true_when_positive_dominant
TestAggregateSentimentAbove::test_true_when_avg_above_threshold
TestAggregateSentimentAbove::test_false_when_avg_below_threshold
TestAggregateSentimentBelow::test_true_when_avg_below_threshold
```

**Root cause:** `src/strategy/conditions_news.py` imports `NewsQuery` lazily (inside each function, not at module top). My tests patch at `src.strategy.conditions_news.NewsQuery` which fails because the name doesn't exist at module level. **Fix:** either hoist `from src.research.news_query import NewsQuery` to module top in `conditions_news.py`, OR change tests to patch `src.research.news_query.NewsQuery` (the original definition site) using the `autospec` style. Recommended — hoist the import (matches pattern of `conditions_insider.py` which works).

### 12 Skipped (env-dependent, intentional)

Mostly `kaleido` not installed (image export tests in `test_report_generator.py`) and a few `@graceful_degrade`-wrapped tests in `test_market_regime.py` that need a real Anthropic key.

### Source bugs discovered + fixed during legacy test work

| File | Bug | Fix commit |
|---|---|---|
| `src/risk/position_sizer.py` | Wrong Kelly defaults; `int()` truncation without `round()` | `7c4a268` |
| `src/agents/signal_validator.py` | Counter-trend fallback returned `approve` instead of `reject` | `a43b0b5` |
| `src/agents/pipeline.py` | `_node_signal_validation` skipped `update_context()` (stale regime in fallback) | `a43b0b5` |
| `config/constants.py` | Stamp duty `_pct` stored as fraction not percentage (zeroed all fees) | `e4ccd11` |
| `config/strategies/cpr_vwap_bounce_banknifty.json` | `enabled: false`, wrong `max_open_positions` | `301e2c0` |
| `tests/test_indicators.py` | Various assertion fixes | `71be53d` |

---

## Wiring Complete — 2026-05-10 (commit 3e15e01)

### What was wired

| Category | Status | Notes |
|---|---|---|
| 1. Module imports + `__init__` attributes | DONE | 34 phase-module handles declared as `None` in `TradingBot.__init__` |
| 2. `_init_phase_modules()` initialisation | DONE | All 24 modules init with try/except; startup health summary logged |
| 3. WebSocket subscription expansion | DONE | `_expand_websocket_subscriptions()` adds Top-10 + sector + macro + India VIX |
| 4. Tick callback chain | DONE | `_on_market_tick` dispatches to depth_feed, tick_metrics, vix_regime |
| 5. Strategy engine Wave-5 gates | PARTIAL — see CHANGE_LOG | `set_wave5_modules()` + `_apply_wave5_gates()` added to StrategyEngine; modules injected; live emit path wiring DEFERRED |
| 6. Kelly sizer + portfolio risk gate | DEFERRED | Kelly gate in order-placement path not yet wired (see below) |
| 7. Smart router + order validator | DEFERRED | Not yet wired into OrderManager.place_order (see below) |
| 8. Bloomberg scheduler jobs | DONE | All jobs registered via `_setup_bloomberg_scheduler_jobs()` |
| 9. Startup health log | DONE | `=`×70 table printed at end of `_init_phase_modules` |
| 10. Smoke tests | DONE | `tests/test_main_wiring.py` — 8 tests, all green |

### CHANGE_LOG

**wave5_emit_wiring: DEFERRED**
The live `SetEvaluator.run()` threads put signals directly to `_signal_queue` without going through `StrategyEngine`. Injecting the Wave-5 gates into that path requires either:
(a) passing `confluence_engine/rejection_filter/regime_router` references through `StrategyEngine` → each `SetEvaluator` constructor, or
(b) intercepting signals in the `SignalForwarder` thread in `main.py` before they reach `OrderManager`.
Option (b) is safe and non-invasive. Recommended approach for next session: in `_forward_signals()` thread, call `self.strategy_engine._apply_wave5_gates(sig)` before `self.signal_queue.put(sig)`. This is a 3-line change with full test coverage from `test_main_wiring.py`.

**kelly_gate: DEFERRED**
`KellySizer.size_position()` requires `strategy_name`, `trade_date`, `capital_paisa`, `sl_distance_paisa`, `confluence_score`. The call site is inside `OrderManager._process_signal()` which doesn't have access to `TradingBot.kelly_sizer`. Next session: pass `kelly_sizer` reference to `OrderManager` at init time (add optional param) and call it in `_process_signal` before `_compute_qty`.

**smart_router + order_validator: DEFERRED**
Same problem — `OrderManager.place_order` doesn't know about `smart_router` or `order_validator`. Next session: pass both to `OrderManager` at init time and call `order_validator.validate(order)` + `smart_router.route(split_order)` in the placement path.

### What the bot now does on `feature/bloomberg-data-execution`

- Subscribes to ~22+ instruments (Top-10 equities + sector indices + macro + VIX + indices)
- Records L2 depth, tick CVD, and VIX intraday on every tick
- Pre-market universe scan at 08:30, regime decision at 08:35
- Options chain polled every 30s
- Hourly news RSS scrape + sentiment classification
- EOD scraper jobs for corp actions, earnings, blocks, insider, flows, macro, VIX, sector ranks
- EOD VaR + Greek snapshot at 16:30
- Cold storage archive at 00:15
- Position reconciler every 30s
- Wave-5 gate layer (confluence, rejection, regime) available as `engine._apply_wave5_gates()` — not yet in live emit path

## CRITICAL: What is NOT wired yet (updated)

The following items need one more session to complete:

1. **Wave-5 gates in live signal path**: Wire `_apply_wave5_gates()` into `_forward_signals()` thread in `main.py` (3-line change).
2. **Kelly sizer in order sizing**: Pass `kelly_sizer` to `OrderManager`, call in `_process_signal`.
3. **Order validator + smart router**: Pass to `OrderManager`, call in `place_order`.

Everything else is wired and running on the feature branch.

---

## HISTORICAL: What was NOT wired (pre-2026-05-10)

The section below is preserved for historical reference only.

Every module built so far was **standalone** — meaning each had:
- A class/functions importable from other code
- Reads/writes to its own Postgres tables
- Has tests proving it works in isolation

**NONE of them were called by the running bot.** `src/main.py` had not been touched (apart from earlier session's Phase 0 `bot_runs` lifecycle recording — pre-existing modification). All the new condition helpers, scrapers, classifiers, scanners were dormant code on disk.

The bot on main branch:
- Runs from `main` branch (not `feature/bloomberg-data-execution`).
- Subscribes only to NSE_INDEX|Nifty Bank + Nifty 50.
- Uses old strategy engine that doesn't call any new condition helpers.
- Has no scheduler entries for the EOD scrapers (corp actions, earnings, news, FII/DII).
- Does not run the universe scanner pre-market.
- Does not poll options chain.
- Does not record L2 depth or CVD or sector ranks anywhere.

**The wiring agent (next-next phase) will:**
1. Modify `src/main.py` to subscribe to Top-10 + sector indices + macro instruments + India VIX.
2. Wire `DepthFeedHandler.on_tick`, `TickMetricsAggregator.on_tick`, etc. into the WebSocket callback chain.
3. Register scheduler jobs for: pre-market 08:30 universe scan, hourly news scrape, EOD 16:00 corp actions/earnings/blocks/insider, every-30s options chain poll.
4. Modify `src/strategy/engine.py` to call confluence engine and rejection filter after raw signal generation.
5. Modify `src/strategy/conditions.py` to expose new condition operators (or use a dispatcher).
6. Add startup health checks for each new module.
7. Wire risk_manager to portfolio_risk + kelly_sizer.

Until that wiring agent runs, **nothing in this build affects the running bot**. Strategy execution still uses the old code path.

---

## Pending Work (in order of execution)

### Wave 5 — Cross-phase intelligence (NOT YET DISPATCHED)

These four modules sit between strategies and execution. They depend on Phase A/B/C output but live in their own layer.

1. **`src/strategy/confluence_engine.py`** — takes raw boolean signals from condition evaluators across multiple dimensions (price action, microstructure, sector, macro, news). Computes a confluence score 0-100. Strategies should require ≥70 to fire.

2. **`src/strategy/rejection_filter.py`** — anti-signals. Takes a candidate signal and rejects if: spread too wide, options OI dropping, sector RS bottom-quartile, fresh negative news, VIX spike, time near close, in earnings/corporate blackout window, correlated position cap exceeded. Spec said this reduces trade count 30-50% and lifts win rate 5-10%.

3. **`src/strategy/regime_router.py`** — reads VIX regime + sector rotation + macro state. Gates which strategy types are allowed each day (trend-following disabled in chop, mean-reversion disabled in trends, options-sell active when premium rich).

4. **`src/risk/kelly_sizer.py`** — replaces fixed position sizing. Uses recent strategy edge × confidence to allocate capital. Half-Kelly default for safety.

Migrations needed: `confluence_scores` table (per-signal score audit), `regime_decisions` (daily allowed strategy types), `kelly_allocations` (per-strategy capital share).

### Phase F — Portfolio risk (NOT YET DISPATCHED)

5. **`src/risk/portfolio_risk.py`** — rolling 60-day correlation matrix; auto-cap if 3 positions correlated >0.8; beta-weighted gross/net exposure to NIFTY (cap 1.5x net long, 2x gross); sector exposure cap 40%; daily VaR/CVaR.

6. **`src/risk/greek_aggregator.py`** — portfolio-level Greeks (delta-neutral check, vega exposure cap, gamma risk, theta decay budget).

Migrations: `correlation_matrix_daily`, `portfolio_greeks_snapshot`, `var_daily`.

### Phase E — Execution upgrades (NOT YET DISPATCHED)

7. **`src/execution/smart_router.py`** — between OrderManager and broker. LIMIT-with-protection, iceberg slicing, TWAP/VWAP for large orders.

8. **`src/execution/order_validator.py`** — pre-flight: lot/freeze-qty splitter, SPAN margin check via Upstox REST.

9. **`src/execution/reconciler.py`** — every 30s diff broker positions vs local positions table. Detect missed fills.

10. **`src/execution/slippage_monitor.py`** — track real fill price vs paper expected price. Halt + alert if drift >50%.

Migrations: `slippage_log`, `reconciliation_log`.

### Wiring agent (NOT YET DISPATCHED)

Single agent. Touches `src/main.py`, `src/strategy/engine.py`, `src/strategy/conditions.py`, `src/data/websocket_feed.py`, `src/utils/scheduler.py` (or wherever cron jobs live). Integrates everything per the architecture map above. Cannot run until Wave 5 + Phase F + Phase E complete.

### Forward test (after wiring complete)

Run new bot in paper mode for 4 weeks during Indian market hours. Compare against main-branch bot baseline. Log everything to Postgres for analysis.

### Optional backtester rebuild (decide after forward test)

Current backtester is bar-level only. If microstructure signals (CVD divergence, depth walls) prove out in forward test, rebuild backtester for tick + L2 replay (5-7 days work). If not, ship as-is.

---

## Operational Reference

### Running tests

```bash
cd /Users/sandeepvangapandu/Downloads/Trading

# Full suite
python3 -m pytest tests/ --tb=no -q

# Single phase (e.g. Phase B all)
python3 -m pytest tests/research/test_sector_rotation.py tests/research/test_vix_regime.py tests/research/test_macro_overlay.py tests/research/test_flow_regime.py tests/data/test_flow_scraper.py tests/strategy/test_conditions_sector.py tests/strategy/test_conditions_vix.py tests/strategy/test_conditions_macro.py tests/strategy/test_conditions_flows.py -q

# Single file with traceback
python3 -m pytest tests/strategy/test_conditions_news.py --tb=short
```

### Migrations

```bash
# Apply all pending
python3 -m src.storage.migrate up

# Status
python3 -m src.storage.migrate status

# Verify a table exists in local Postgres
psql -d tradebot -c "\d <table_name>"
psql -d tradebot -c "SELECT version FROM schema_version ORDER BY applied_at;"
```

### Git workflow (per commit)

```bash
git add -f <files>     # tests/data and src/data are gitignored — must use -f
git commit -m "..."
git push origin feature/bloomberg-data-execution
```

**On push:** Supabase GitHub integration sees the new migration file in `supabase/migrations/`, runs it against the cloud Postgres. Migrations are idempotent (`CREATE IF NOT EXISTS`, `INSERT ... ON CONFLICT DO NOTHING/UPDATE`) so re-runs are safe.

### Bot run

Bot is currently using `main` branch. To switch to feature branch (DO NOT do this until wiring agent has run):

```bash
git checkout feature/bloomberg-data-execution
# then run bot as normal — but it will crash because new modules aren't wired
```

For now, leave bot on `main`. Develop on feature.

### Subagent dispatch lessons learned

- **`tester` and `coder` subtypes get random Bash permission denials.** Avoid them.
- **`general-purpose` subtype with full tool access works reliably.** Use it for everything that needs Bash + git push.
- **Sonnet model** for all subagents (cost optimization).
- **Run all parallel-safe agents in background** with `run_in_background: true`.
- **Never delegate `main.py` edits.** Reserve for single wiring agent at the end.
- **Never run multiple agents on the same file.** Partition by file ownership.
- **DO NOT include "use the fewer-permission-prompts skill" or any skill mention in agent prompts.** Some agents auto-trigger that skill and lock themselves out.
- **Migrations should be additive-only, idempotent.** Never drop a table or column in a migration.
- **Stagger migration timestamps** by phase (00010 = A.1, 00011 = A.2, ..., 00030 = C.1) so naming order matches dependency order even if files are created out of order.
- **Anthropic API rate limits hit hard around 1am EDT.** If three agents fail simultaneously with "You've hit your limit · resets X" — stop dispatching, wait until reset.

### Agent prompt template (works reliably)

```
Build [phase name] at /Users/sandeepvangapandu/Downloads/Trading.
Branch: feature/bloomberg-data-execution. [Prior phases] complete.

DO NOT invoke any skills. Direct execution only.

Goal: [one sentence].

Files to create:
1. Migration ...
2. Module ...
3. Tests ...

Constraints:
- DO NOT touch src/main.py, websocket_feed, or other phase modules.
- Idempotent migrations.
- IST times, paisa for money.

Verification:
1. python3 -m src.storage.migrate up
2. pytest [paths] -v
3. Report counts.

Commits + push (after each focused change):
- "feat(storage): ..."
- "feat(...): ..."
- "test(...): ..."
git push origin feature/bloomberg-data-execution

Report:
- Files created.
- Verification outputs.
- Commits + git log --oneline -8.

Time: [estimate]. Stop + report if blocked > 25 min.
```

---

## Key Discoveries / Gotchas

### Upstox segment naming (saved in memory `project_upstox_segments.md`)

Upstox does NOT route currency/commodity futures via the segments their names suggest:

| Asset | Actual Upstox segment |
|---|---|
| USDINR futures | `NCD_FO` (NSE Currency Derivatives Futures) |
| Crude oil futures | `NSE_COM` (NSE Commodities) |
| Gold futures | `NSE_COM` |
| Silver futures | `NSE_COM` |
| Equity F&O | `NSE_FO` (as expected) |
| Equity spot | `NSE_EQ` |
| Indices | `NSE_INDEX` |

Phase B.3 macro_overlay migration uses the corrected segments. If you ever look up MCX_FO instruments and see nothing, that's why — they're under NSE_COM.

### Pydantic v2 InstrumentSelection union

Earlier session fixed: `InstrumentSelection` field declared as `Union[InstrumentSelection, dict]`. Pydantic v2 smart-union falls back to `dict` if validation fails. Strategy engine `_get_symbols_to_evaluate` had to handle both via `_instr_sel_get` helper. For options-type strategies, evaluation runs on underlying spot, not options.

### Instrument cache parser bugs (fixed in earlier session)

`src/data/instruments.py:load_instruments` had two bugs fixed:
1. Raw JSON has `instrument_type` field (`CE`/`PE`), not `option_type`. Loader now derives `option_type` from `instrument_type`.
2. Expiry was epoch milliseconds — `pd.to_datetime` was treating as nanoseconds (1970 dates). Fixed with `unit="ms"`.

### Health monitor false positives (NOT YET FIXED)

The bot at session start had constant CRITICAL alerts:
```
Component scheduler has exceeded max recovery attempts (3). Manual intervention required.
Component market_feed has exceeded max recovery attempts (3). Manual intervention required.
```

Despite ticks flowing fine. Root cause not investigated yet. Heartbeat registration suspect. **Add to wiring agent's checklist** — investigate `src/utils/health_monitor.py:_check_health` line 655.

### .gitignore weirdness

`.gitignore` has `data/` pattern that matches both top-level `data/` AND `src/data/` AND `tests/data/`. All `git add` commands for files under those paths need `-f`. Many commits use this. Don't be alarmed.

### Pre-existing uncommitted modifications (untouched by any agent in this build)

`git status` shows these modified — they predate this build, leave them alone:
- `.claude/settings.local.json`
- `src/execution/position_manager.py`
- `src/main.py` (small edits user already had — `self.db_url` → `self.settings.database_url`)
- `src/persistence/trade_log.py`
- `src/strategy/engine.py` (small edits user already had — adds BUY_PE to buy signal types tuple)

---

## Memory Notes (in `~/.claude/projects/-Users-sandeepvangapandu-Downloads-Trading/memory/`)

- `MEMORY.md` — index
- `feedback_model_routing.md` — user wants Opus for thinking, Sonnet subagents for execution
- `project_phase_plan.md` — universe (Top 10), RSS news, Postgres free tier, paper-forward microstructure validation, hardware (32GB/2TB)
- `project_upstox_segments.md` — segment naming gotchas

---

## How to resume in a new session

1. Read this entire document.
2. `git fetch && git checkout feature/bloomberg-data-execution && git pull`
3. `python3 -m src.storage.migrate status` — verify all 14 migrations applied.
4. `python3 -m pytest tests/ --tb=no -q` — confirm 1473 pass / 5 fail (known) / 12 skip baseline.
5. Decide which pending phase to work on next:
   - Recommended: fix the 5 `test_conditions_news.py` failures first (5-min job — hoist `from src.research.news_query import NewsQuery` to module top in `src/strategy/conditions_news.py`).
   - Then dispatch Wave 5 (4 parallel agents — confluence/rejection/regime/Kelly).
6. Use `general-purpose` subagent type, Sonnet model, `run_in_background: true` for parallel work.
7. **NEVER touch `src/main.py` from a worker agent.** Keep wiring isolated to one final agent at the end.

---

## Final commit log (`git log --oneline -50` on feature branch as of 2026-05-09)

```
cc3e2a4 feat(phase-c): C.1/C.3/C.4 — corp actions, news+sentiment, blocks+insider
736293c feat(data): Phase C.1 NSE corporate actions scraper
611270f feat(phase-c2): earnings calendar + blackout filter
2d5976d feat(storage): Phase C.1 migration — corporate actions calendar
a178cb4 test(research,strategy): Phase B.3 coverage
627fa92 feat(strategy): Phase B.3 macro condition helpers
f693608 feat(research): Phase B.3 macro overlay analyzer
57e7cb8 feat(storage): Phase B.3 migration — macro instruments + daily regime
7ea4670 test(data,research,strategy): Phase B.4 coverage — flows, regime, conditions
e53b146 feat(strategy): Phase B.4 flow + yield condition helpers
1f7521f feat(research): Phase B.4 flow regime classifier
a66c31b feat(data): Phase B.4 flow scraper (NSE+MC fallback)
1aa3f3e feat(storage): Phase B.4 migration — FII/DII flows + bond yields + flow regime
35da6b5 test(research,strategy): Phase B.1 coverage
4a6075f feat(strategy): Phase B.1 sector condition helpers
658fa5f feat(research): Phase B.1 sector rotation analyzer
3fd1241 feat(phase-b2): VIX regime classifier (LOW/NORMAL/HIGH/SPIKE) + conditions
98cb70e test(test_report_generator): add missing fees/net_pnl to Trade fixtures; skip plot tests when kaleido absent
e4ccd11 fix(constants): correct stamp duty percentage values in FEES dict
a43b0b5 fix(pipeline+signal_validator): counter-trend signals must be rejected not sized-down
ed9a5a8 test(test_technical_analyzer): fix copy-paste indentation bug in TestSupportResistance
301e2c0 fix(cpr_vwap): enable strategy config and fix VWAP timezone index mismatch
15ea3aa test(test_integration): update error-message substring checks to match source
7021c41 test(test_partial_profit): fix two trailing-stop tests that used wrong prices
d67a947 test(test_pattern_recognition): fix morning/evening star fixtures
7c4a268 fix(position_sizer): fix two bugs in KellyPositionSizer defaults and sizing
44b9ef6 test(data,strategy): Phase A.3 coverage
6e3b69d feat(strategy): Phase A.3 options-derived condition helpers
6e50675 test(data,strategy): Phase A.2 coverage
71be53d test(test_indicators): fix 3 failing indicator tests
3c114f2 feat(strategy): Phase A.2 microstructure condition helpers
ae2ebef feat(storage): Phase A.3 migration — options chain snapshots + strike OI history
edfd8c4 feat(strategy): Phase A.4 volume profile condition helpers
e47eb5c feat(data): Phase A.2 tick metrics + CVD aggregator
015cb32 feat(indicators): Phase A.4 volume profile (POC/VAH/VAL)
e8f73d5 feat(storage): Phase A.4 migration — volume profile daily table
85dd641 feat(storage): Phase A.2 migration — tick metrics minute table
a583796 test(test_signal_validator): update tests to match refactored SignalValidatorAgent API
f8b0956 test(data): depth feed parsing + metrics coverage
a06cd6e feat(data): Phase A.1 L2 depth feed handler
5f5e5c2 test(test_trade_scorecard): fix 4 failing assertions
7147212 feat(storage): Phase A.1 migration — depth metrics minute table
8d774a9 test(graceful_degradation): fix loguru capture, confidence assertions, and None enum values
5740291 test(market_regime): skip tests incompatible with @graceful_degrade wrapper; fix expiry test date
9bc08ff test(fixtures): smoke test suite for all fixture modules and scenarios (125 tests, <0.5s)
7ca0638 test(fixtures): add 10 scenario harnesses
4ff03fe test(fixtures): add news_rss and macro_data fixture factories
e9759f7 test(fixtures): add core fixture modules (ws payloads, option chain, instrument master, historical bars)
dff7855 test(trade_analyzer): fix mock fixtures and API calls to match current source
b60d601 feat(research): Phase D scanner CLI runner
16544a2 test(research): universe scanner test coverage
80f0f9d feat(research): Phase D universe scanner module
d091eda feat(storage): Phase D migration — universe scanner tables
c05ee65 test(strategy_engine): replace identity check with equality for numpy bool
bb72ea4 fix(execution): UpstoxLiveBroker abstract methods
2a3dc11 feat(storage): Phase 0 — Postgres + Supabase foundation
```

---

## Phase G — Kronos Foundation Model — REMOVED 2026-05-14

**Status:** DROPPED.

180d walk-forward backtest (`scripts/kronos_backtest.py`, since deleted) on 12 instruments (Top-10 NSE + Nifty50 + NiftyBank), 5m bars, 3443 predictions:

- Overall direction accuracy: **36.13%** — below 3-class random baseline and below always-DOWN majority baseline (37.67%)
- UP precision lift +0.25pp (z=+0.18, not sig)
- DOWN precision lift +1.76pp (z=+1.33, not sig)
- FLAT precision lift +4.36pp (z=+2.97, sig but not actionable for entries)
- Close MAE 0.558%, Range MAE 0.431%

Likely cause: Kronos pretrained on Chinese A-share data — NSE/BSE microstructure out-of-distribution.

### Removed

- `src/research/kronos_lib/` (vendored Kronos package)
- `src/research/kronos_predictor.py`
- `src/research/kronos_validator.py`
- `src/research/run_kronos_shadow.py`
- `src/research/run_kronos_validation.py`
- `src/strategy/kronos_dimension.py`
- `tests/test_kronos_wiring.py`
- `tests/research/test_kronos_validator.py`
- `tests/research/test_kronos_predictor.py`
- `tests/strategy/test_kronos_dimension.py`
- `scripts/kronos_backtest.py`
- `KRONOS_EVALUATION.md`
- `MODEL_FORECAST` from `ConfluenceDimension` enum and `DEFAULT_WEIGHTS`
- `kronos_forecaster` / `kronos_validator` attributes + init + schedulers from `src/main.py`
- `_build_kronos_dimension` + `kronos_forecaster` kwarg from `src/strategy/engine.py`
- Backtest CSV retained: `logs/kronos_backtest_1778782706.csv`
- DB tables `kronos_forecasts` and `kronos_accuracy_daily` left in place — drop manually if disk pressure (not auto-dropped to preserve historical data).

End of handoff.
