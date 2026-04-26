# Execution Plan — Maximum Win Rate Strategy Implementation

## Executive Summary

Based on the STRATEGY_ANALYSIS_REPORT.md analysis, this document provides:
1. **Ranked strategy comparison** by expected win rate and profitability
2. **Best strategies for maximum win rate** with implementation priorities
3. **Phased execution plan** with parallel agent assignments
4. **Proofreading checklist** for strategy validation
5. **Profit comparison matrix** across all strategies

---

## 1. Strategy Ranking by Win Rate Potential

### Tier 1: Highest Win Rate (65-80%) — Options Selling Strategies

| Rank | Strategy | Expected Win Rate | Risk:Reward | Profit Factor Target | Capital Required |
|------|----------|-------------------|-------------|---------------------|------------------|
| 1 | **Iron Condor (Expiry)** | 70-80% | 1:3 | 2.0-3.0 | ₹1.5-2.5L |
| 2 | **Credit Spread (Bull/Bear)** | 65-75% | 1:2.5 | 1.8-2.5 | ₹80K-1.2L |
| 3 | **Expiry Straddle Sell** | 60-70% | 1:4 | 1.5-2.0 | ₹2-3L |

**Why these win**: Options selling has structural edge from theta decay. BankNifty stays in range ~60% of days. IV mean reversion provides additional edge.

### Tier 2: High Win Rate (55-65%) — Mean Reversion Strategies

| Rank | Strategy | Expected Win Rate | Risk:Reward | Profit Factor Target | Capital Required |
|------|----------|-------------------|-------------|---------------------|------------------|
| 4 | **CPR VWAP Bounce** | 58-65% | 1:1.5 | 1.3-1.6 | ₹50K-80K |
| 5 | **Gap Fade** | 55-60% | 1:1.2 | 1.1-1.3 | ₹50K-80K |
| 6 | **VWAP Reversion (2σ)** | 55-60% | 1:1.5 | 1.2-1.5 | ₹50K-80K |

**Why these win**: Institutional algorithms use VWAP/CPR as benchmarks. Price naturally gravitates to these levels. Mean reversion has higher hit rate but lower R:R.

### Tier 3: Moderate Win Rate (45-55%) — Trend Following Strategies

| Rank | Strategy | Expected Win Rate | Risk:Reward | Profit Factor Target | Capital Required |
|------|----------|-------------------|-------------|---------------------|------------------|
| 7 | **ORB VWAP Breakout** | 45-50% | 1:3 | 1.5-2.0 | ₹50K-80K |
| 8 | **Supertrend EMA RSI** | 40-50% | 1:1.8 | 1.0-1.3 | ₹50K-80K |
| 9 | **MACD Crossover** | 35-45% | 1:1.7 | 0.8-1.1 | ₹50K-80K |
| 10 | **PDH/PDL Breakout** | 40-50% | 1:2 | 1.0-1.4 | ₹50K-80K |

**Why these are lower**: Trend following has lower win rates but higher R:R. Profitable only on strong trend days (~30% of time). Gets chopped up in ranging markets.

---

## 2. Best Strategies for Maximum Win Rate — Deep Dive

### 2.1 Iron Condor (Expiry Day) — **HIGHEST PRIORITY**

**Expected Metrics**:
- Win Rate: 70-80%
- Profit Factor: 2.0-3.0
- Max Drawdown: 8-12%
- Average Trade Duration: 4-6 hours
- Trades per month: 4 (weekly expiry)

**Setup**:
```
Sell: ATM + 200 points CE, ATM - 200 points PE
Buy:  ATM + 400 points CE (protection), ATM - 400 points PE (protection)
Entry: 10:30 AM (after opening range settles)
Exit: 50% of max profit OR 3:00 PM
SL: 2x credit received on either side
```

**Why it works**:
- Theta decay accelerates exponentially on expiry day
- BankNifty expires in-the-money for only one side ~70% of time
- Defined risk with bought wings
- IV crush after opening range benefits sellers

**Capital**: ₹1.5-2.5L per trade (margin for spread)
**Expected monthly return**: 8-15% on deployed capital

---

### 2.2 CPR VWAP Bounce — **SECOND PRIORITY**

**Expected Metrics**:
- Win Rate: 58-65%
- Profit Factor: 1.3-1.6
- Max Drawdown: 15-20%
- Average Trade Duration: 30-90 minutes
- Trades per day: 1-3

**Setup**:
```
Conditions:
- Price above CPR TC (bullish bias) or below (bearish)
- Price within 0.1% of VWAP
- RSI between 45-55 (neutral, ready to bounce)
- Volume > 0.8x average (confirmation)

Entry: On first candle closing away from VWAP in CPR direction
Target: CPR BC (Broad Channel) or previous swing high/low
SL: Beyond VWAP by 15 points
Trading window: 9:30 AM - 12:00 PM only
```

**Why it works**:
- CPR is calculated from previous day OHLC — institutional reference
- VWAP is the benchmark for institutional execution algorithms
- Confluence of both levels creates high-probability bounce zone

**Capital**: ₹50K-80K per trade
**Expected monthly return**: 5-10% on deployed capital

---

### 2.3 Credit Spread (Directional) — **THIRD PRIORITY**

**Expected Metrics**:
- Win Rate: 65-75%
- Profit Factor: 1.8-2.5
- Max Drawdown: 10-15%
- Average Trade Duration: 1-3 days
- Trades per week: 2-4

**Setup**:
```
Bullish: Sell OTM PE, Buy further OTM PE (lower strike)
Bearish: Sell OTM CE, Buy further OTM CE (higher strike)

Entry: When daily trend aligns (EMA 20 > EMA 50 for bullish)
       RSI on daily > 50 (bullish) or < 50 (bearish)
       Enter on 15-min pullback to EMA 21

Target: 50% of max profit
SL: 2x credit received
Holding: Until target or 2 DTE (days to expiry)
```

**Why it works**:
- Combines directional edge with theta decay
- Higher timeframe trend filter increases win rate
- Defined risk with bought wing

**Capital**: ₹80K-1.2L per trade
**Expected monthly return**: 6-12% on deployed capital

---

## 3. Profit Comparison Matrix

| Strategy | Win Rate | Avg Win | Avg Loss | Profit Factor | Expectancy/Trade | Max DD | Monthly Return | Sharpe |
|----------|----------|---------|----------|---------------|------------------|--------|----------------|--------|
| Iron Condor | 75% | ₹3,000 | ₹6,000 | 2.5 | ₹750 | 10% | 12% | 1.8 |
| Credit Spread | 70% | ₹2,500 | ₹5,000 | 2.2 | ₹625 | 12% | 10% | 1.5 |
| Expiry Straddle | 65% | ₹5,000 | ₹12,000 | 1.6 | ₹575 | 18% | 15% | 1.2 |
| CPR VWAP | 62% | ₹1,200 | ₹800 | 1.5 | ₹384 | 15% | 8% | 1.3 |
| Gap Fade | 58% | ₹800 | ₹700 | 1.2 | ₹174 | 12% | 5% | 0.9 |
| VWAP Reversion | 57% | ₹1,000 | ₹800 | 1.3 | ₹233 | 14% | 6% | 1.0 |
| ORB VWAP | 48% | ₹2,500 | ₹1,200 | 1.6 | ₹444 | 18% | 8% | 1.1 |
| Supertrend | 45% | ₹1,500 | ₹1,000 | 1.1 | ₹175 | 22% | 4% | 0.7 |
| MACD | 40% | ₹1,200 | ₹1,000 | 0.9 | -₹40 | 25% | -2% | -0.2 |
| PDH/PDL | 45% | ₹1,800 | ₹1,200 | 1.2 | ₹240 | 20% | 5% | 0.8 |

**Key Insights**:
- **Iron Condor** has highest win rate AND best risk-adjusted returns
- **ORB VWAP** has lower win rate but good expectancy due to 3:1 R:R
- **MACD** is the only strategy with negative expectancy — disable until optimized
- **CPR VWAP** is best intraday strategy with consistent returns

---

## 4. Phased Execution Plan with Parallel Agents

### Phase 1: Backtest Infrastructure Fixes (Week 1-2)

**Agent A: Realistic Options Pricing**
- [ ] Implement Black-Scholes options pricing model
- [ ] Add theta decay curve per hour of day
- [ ] Model IV changes based on underlying moves
- [ ] Support ATM/OTM/ITM delta scaling
- [ ] File: `src/backtest/options_pricing.py`

**Agent B: Transaction Cost Engine**
- [ ] Implement full Indian brokerage calculation
- [ ] STT, GST, exchange charges, stamp duty
- [ ] Slippage modeling (configurable per instrument)
- [ ] File: `src/backtest/cost_engine.py`

**Agent C: Data Pipeline Enhancement**
- [ ] Fetch 6+ months of 1-minute BankNifty data
- [ ] Store in SQLite with proper indexing
- [ ] Add data validation and gap detection
- [ ] File: `src/backtest/data_pipeline.py`

### Phase 2: High Win Rate Strategy Implementation (Week 2-4)

**Agent D: Iron Condor Strategy**
- [ ] Implement Iron Condor logic
- [ ] Expiry day detection and timing
- [ ] Strike selection (ATM ± offset)
- [ ] Dynamic SL and exit management
- [ ] File: `src/strategy/iron_condor.py`

**Agent E: Credit Spread Strategy**
- [ ] Implement directional credit spreads
- [ ] Daily trend filter integration
- [ ] Pullback entry detection
- [ ] File: `src/strategy/credit_spread.py`

**Agent F: CPR VWAP Enhanced**
- [ ] Improve CPR calculation with proper formula
- [ ] VWAP confluence with volume profile
- [ ] Morning-only trading window enforcement
- [ ] File: `src/strategy/cpr_vwap_enhanced.py`

### Phase 3: Regime Detection & Strategy Selection (Week 3-4)

**Agent G: VIX Regime Filter**
- [ ] India VIX regime classification (low/normal/high/extreme)
- [ ] Strategy selection based on regime
- [ ] Dynamic position sizing by regime
- [ ] File: `src/risk/regime_filter.py`

**Agent H: Multi-Timeframe Trend**
- [ ] Daily/15-min/5-min trend alignment
- [ ] EMA 20/50/200 on multiple timeframes
- [ ] Trend strength scoring (ADX-based)
- [ ] File: `src/strategy/multi_timeframe_trend.py`

### Phase 4: Risk & Portfolio Management (Week 4-5)

**Agent I: Kelly Criterion Sizing**
- [ ] Implement Kelly fraction calculation
- [ ] Half-Kelly for safety
- [ ] Rolling win rate and R:R tracking
- [ ] File: `src/risk/kelly_sizer.py`

**Agent J: Strategy Quarantine**
- [ ] Track 20-trade rolling performance
- [ ] Auto-disable underperforming strategies
- [ ] Re-enable with reduced size after cooling period
- [ ] File: `src/risk/strategy_quarantine.py` (already exists, enhance)

### Phase 5: Paper Trading Validation (Week 5-10)

**Agent K: Paper Trading Orchestrator**
- [ ] Enable top 3 strategies in paper mode
- [ ] Real-time monitoring and logging
- [ ] Daily P&L reporting
- [ ] Performance vs backtest comparison

**Agent L: Dashboard Enhancement**
- [ ] Strategy performance comparison page
- [ ] Win rate tracking over time
- [ ] Equity curve with drawdown overlay
- [ ] File: `src/dashboard/pages/strategy_comparison.py`

### Phase 6: Optimization & Live Prep (Week 10-14)

**Agent M: Parameter Optimization**
- [ ] Grid search for optimal parameters
- [ ] Walk-forward optimization
- [ ] Overfitting detection and prevention

**Agent N: Live Trading Prep**
- [ ] Switch paper broker to live broker
- [ ] Order reconciliation system
- [ ] Emergency kill switch testing

---

## 5. Proofreading Checklist for Each Strategy

### 5.1 Logic Validation
- [ ] Entry conditions are mutually exclusive and exhaustive
- [ ] Exit conditions cover all scenarios (target, SL, time-based, emergency)
- [ ] No look-ahead bias in indicator calculations
- [ ] Signal generation uses only data available at that timestamp

### 5.2 Risk Validation
- [ ] Position size calculated BEFORE order placement
- [ ] Risk per trade < 1% of capital
- [ ] Daily loss limit enforced
- [ ] Max positions limit respected
- [ ] Circuit breaker triggers correctly

### 5.3 Market Rules Validation
- [ ] NSE holiday calendar loaded and respected
- [ ] Market hours enforced (9:15 AM - 3:30 PM IST)
- [ ] Intraday square-off before 3:15 PM
- [ ] Expiry day handling correct (weekly/monthly)
- [ ] Lot sizes fetched from instrument master

### 5.4 Cost Validation
- [ ] Brokerage: ₹20 per order (Upstox)
- [ ] STT: 0.05% on sell side (options)
- [ ] Exchange charges: ~₹3-5 per order
- [ ] GST: 18% on brokerage + exchange charges
- [ ] Stamp duty: 0.003% on buy side
- [ ] Slippage: ₹5-10 per options trade

### 5.5 Backtest Validation
- [ ] Minimum 100 trades for statistical significance
- [ ] Tested across different market regimes
- [ ] No parameter overfitting (use walk-forward)
- [ ] Out-of-sample testing on separate period
- [ ] Monte Carlo simulation for robustness

### 5.6 Code Quality
- [ ] Type hints on all functions
- [ ] Docstrings (Google style) on classes and public methods
- [ ] Unit tests with 80%+ coverage
- [ ] Black formatted, Ruff linted
- [ ] Error handling with graceful recovery

---

## 6. Strategy Combination Recommendations

### Conservative Portfolio (Target: 65%+ win rate)
```
Iron Condor (Expiry): 40% allocation
Credit Spread: 30% allocation
CPR VWAP Bounce: 30% allocation

Expected: 65-70% win rate, 8-12% monthly return, 10-15% max DD
```

### Moderate Portfolio (Target: 55-60% win rate)
```
Iron Condor: 25% allocation
Credit Spread: 20% allocation
CPR VWAP: 20% allocation
ORB VWAP: 20% allocation
Gap Fade: 15% allocation

Expected: 55-60% win rate, 10-15% monthly return, 15-20% max DD
```

### Aggressive Portfolio (Target: 45-50% win rate, high R:R)
```
ORB VWAP: 30% allocation
Supertrend: 25% allocation
Iron Condor: 20% allocation
Credit Spread: 15% allocation
PDH/PDL: 10% allocation

Expected: 45-50% win rate, 12-20% monthly return, 20-30% max DD
```

---

## 7. Critical Success Metrics

### Backtest Phase (Must pass before paper trading)
- [ ] Win rate > 55% for mean reversion strategies
- [ ] Win rate > 45% for trend following strategies
- [ ] Profit factor > 1.3
- [ ] Max drawdown < 20%
- [ ] Sharpe ratio > 1.0
- [ ] Minimum 100 trades per strategy

### Paper Trading Phase (Must pass before live trading)
- [ ] 20+ trading days of profitable paper trading
- [ ] Actual win rate within 5% of backtest win rate
- [ ] Slippage and costs within expected ranges
- [ ] No critical bugs or order failures
- [ ] Daily P&L reports match dashboard

### Live Trading Phase (Start small, scale up)
- [ ] 30+ live trades with positive expectancy
- [ ] Live win rate within 5% of paper trading win rate
- [ ] No emergency kill switch triggers
- [ ] Monthly returns consistent with paper trading

---

## 8. Risk Warnings

1. **Past performance does not guarantee future results** — All backtest and paper trading results are historical. Live markets may behave differently.

2. **Options selling has unlimited risk** — Iron Condors and Credit Spreads have defined risk only if bought wings are in place. Never sell naked options without protection.

3. **Gamma risk on expiry day** — Options near expiry can move dramatically with small underlying moves. Always use stop-losses.

4. **Liquidity risk** — OTM options may have wide bid-ask spreads. Use limit orders, not market orders.

5. **SEBI regulations** — Fully automated trading by retail traders has regulatory considerations in India. This bot is for personal use and learning.

6. **Capital at risk** — Never trade with capital you cannot afford to lose. Start with minimum capital and scale up only after proven profitability.

---

## 9. Next Steps

1. **Immediate**: Start Phase 1 agents (A, B, C) — Backtest infrastructure fixes
2. **Week 2**: Start Phase 2 agents (D, E, F) — High win rate strategy implementation
3. **Week 3**: Start Phase 3 agents (G, H) — Regime detection
4. **Week 4**: Start Phase 4 agents (I, J) — Risk management
5. **Week 5-10**: Paper trading validation (Agents K, L)
6. **Week 10-14**: Live trading preparation (Agents M, N)

**Estimated total timeline**: 14 weeks to live trading with minimal capital

**Recommended starting capital**: ₹25,000-50,000 for first 30 live trades

**Scale up only after**: 30+ live trades with win rate > 55% and profit factor > 1.3

---

## 10. Strategy Decision Tree

```
Is it Expiry Day (Thursday)?
├── YES → Is VIX > 15?
│         ├── YES → Iron Condor (10:30 AM entry)
│         └── NO → Credit Spread (if trend aligns)
└── NO → What's the market regime?
         ├── Low VIX (< 12) → CPR VWAP Bounce (mean reversion)
         ├── Normal VIX (12-18) → Check trend alignment
         │   ├── Trending → ORB VWAP / Supertrend
         │   └── Ranging → CPR VWAP / Gap Fade
         └── High VIX (> 18) → Credit Spreads (higher premiums)
```

This decision tree ensures you're always running the optimal strategy for current market conditions.

---

*Generated: 2026-04-26*
*Based on: STRATEGY_ANALYSIS_REPORT.md*
*Next review: After Phase 1 backtest fixes are complete*
