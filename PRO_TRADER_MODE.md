# PRO TRADER MODE - Maximum Profit Configuration

## Philosophy

**Trade like a professional who makes a living from the markets.**

- Take EVERY high-quality signal that meets criteria
- No arbitrary daily trade limits
- Size positions based on opportunity quality, not fear
- Let winners run, compound aggressively
- Accept that losses are part of the game - manage them, don't avoid them

---

## Key Differences from Conservative Mode

| Aspect | Conservative Mode | PRO TRADER MODE |
|--------|------------------|-----------------|
| **Trade Limit** | Max 3-5/day | **Unlimited** - take every opportunity |
| **Position Size** | Small (1-2 lots) | **Aggressive** (Kelly Criterion, up to 40%) |
| **Kelly Setting** | Half-Kelly (50%) | **Full Kelly (100%)** for max growth |
| **Daily Loss Limit** | ₹3,000 (3%) | **₹5,000 (5%)** - wider for more opportunities |
| **Max Positions** | 3-5 | **10** - scale when opportunities abound |
| **Capital Deployment** | 80% max | **95% max** - almost fully deployed |
| **Circuit Breaker** | 3 losses | **5 losses** - give edge time to work |

---

## Risk Management (Still Protected)

PRO MODE doesn't mean reckless. We still have protections:

### 1. Circuit Breaker (Catastrophic Protection)
- **Trigger**: Daily loss ≥ 8% of capital
- **Action**: Trading halted for the day
- **Why**: Prevents account blow-up from black swan events

### 2. Daily Loss Limit
- **Limit**: 5% of capital (₹5,000 on ₹1L)
- **Action**: Stop trading if hit
- **Why**: Protects capital while allowing aggressive trading

### 3. Position Size Limits
- **Max per position**: 30% of capital
- **Max concurrent**: 10 positions
- **Why**: Diversification prevents single-trade blow-up

### 4. Kelly Criterion Sizing
```
Position Size = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win × Capital

Example:
- Win Rate: 55%
- Avg Win: ₹2,000
- Avg Loss: ₹1,000
- Kelly = (0.55×2000 - 0.45×1000)/2000 = 32.5%
- On ₹1L capital = ₹32,500 per trade
```

---

## When We Trade

### Session 1: Opening Range (9:20-11:30 AM)
**Best opportunities of the day**

| Strategy | Condition | Why Trade |
|----------|-----------|-----------|
| **ORB + VWAP** | Opening breakout with VWAP | Highest conviction moves |
| **Gap Fade** | 0.2-0.5% gaps | Mean reversion in first hour |
| **PDH/PDL Break** | Clean break of prev day levels | Institutional participation |
| **MACD Crossover** | ADX > 25 | Trend day confirmation |

**Position Size**: 1.5x normal (high conviction)

### Session 2: Afternoon Trend (1:30-3:15 PM)
**Continuation moves**

| Strategy | Condition | Why Trade |
|----------|-----------|-----------|
| **Supertrend + EMA** | EMA aligned, ADX > 20 | Afternoon trend following |
| **RSI Divergence** | Near S/R, multi-TF | Reversal at key levels |
| **CPR Bounce** | Price in value zone | Mean reversion plays |

**Position Size**: 1.0x normal (standard)

### Avoid: Lunch Hour (11:30 AM - 1:30 PM)
- Low volume, choppy price action
- False breakouts common
- Opportunity cost is low - better to wait

---

## Position Sizing Rules

### Base Size (Kelly Criterion)
```python
base_size = kelly_fraction × capital / entry_price
```

### Multipliers
| Condition | Multiplier | Example |
|-----------|------------|---------|
| High Conviction (Score 90+) | 2.0x | ₹50K position |
| Strong Trend (ADX > 30) | 1.5x | ₹37.5K position |
| After Win Streak (3+) | 1.3x | ₹32.5K position |
| Normal Setup | 1.0x | ₹25K position |
| After Loss | 0.8x | ₹20K position (cool down) |
| High Volatility | 0.7x | ₹17.5K position |

### Maximum Position Size
```
Hard Limit: min(30% of capital, 10 lots per trade)
```

---

## Compounding Strategy

### Winners: Add to Position
When a trade is winning (+30% unrealized):
1. Add 50% more to winning position
2. Move stop to breakeven on original position
3. Trail new portion with wider stop
4. Maximum 2 additions per winning trade

**Example**:
- Initial: Buy 2 lots at ₹100
- At ₹130 (+30%): Add 1 lot (50% of 2)
- New position: 3 lots, stop at ₹100 (breakeven)
- Target: Scale out at 2:1, 3:1, 4:1 R:R

### Losers: Cut Quickly
- Initial stop: Strategy defined (typically 20-30%)
- No averaging down
- If stopped, look for re-entry if setup reforms
- Maximum 2 re-entries per original signal

---

## Expected Performance

### Daily Targets
| Metric | Target | Realistic Range |
|--------|--------|-----------------|
| **Trades per day** | 5-15 | 3-20 (market dependent) |
| **Win rate** | 55% | 50-60% |
| **Avg R:R** | 1.8:1 | 1.5:2.1 |
| **Expected daily return** | 1-2% | 0.5-3% |
| **Monthly return target** | 25-40% | 15-50% |

### Risk Metrics
| Metric | Limit | Typical |
|--------|-------|---------|
| **Max daily loss** | -5% | Rarely hit with edge |
| **Max drawdown** | -15% | -10 to -12% |
| **Consecutive loss days** | 3 | Usually 1-2 |
| **Recovery time** | 2-3 days | After drawdowns |

---

## Psychological Rules

### DO:
✅ Take every setup that meets criteria
✅ Size aggressively when you have edge
✅ Add to winners, cut losers
✅ Trade through drawdowns (trust the edge)
✅ Review daily - learn, adjust, improve

### DON'T:
❌ Skip setups because you "feel" unsure
❌ Reduce size out of fear after losses
❌ Chase trades outside your rules
❌ Overthink - execute the system
❌ Change strategy during drawdowns

---

## Example Trading Day

### Scenario: Normal Trend Day

**9:20 AM** - Market opens with gap up
- Check Gap Fade: Gap is 0.8% (too large, skip)
- Wait for ORB formation

**9:35 AM** - ORB forms, price breaks above
- Signal: ORB Long + VWAP bullish
- Score: 88/100 (high conviction)
- Position: 3 lots (1.5x multiplier)
- Entry: ₹50,000 premium
- Stop: ₹37,500 (25% SL)
- Target: ₹87,500 (75% gain, 3:1 R:R)

**10:15 AM** - Position at +35% unrealized
- Add 1.5 lots (50% of 3)
- Move stop to breakeven on original 3 lots
- Trail new 1.5 lots with 20% stop

**11:00 AM** - Hit first target (Tier 1 at 1:1)
- Exit 25% of position (1.1 lots)
- Realized: +₹12,500
- Holding: 3.4 lots remaining

**1:45 PM** - Afternoon Supertrend signal
- Signal: Supertrend Short + EMA aligned
- Score: 75/100 (normal)
- Position: 2 lots (1.0x)
- Different strike (hedge morning long)

**2:30 PM** - First position hits Tier 2 (2:1)
- Exit another 25%
- Realized total: +₹37,500 from first trade

**3:00 PM** - Square off all positions
- Final P&L: +₹52,000
- Day return: +5.2%
- Trade count: 2 trades, 5 partial exits

---

## Capital Growth Projection

### Starting Capital: ₹10 Lakhs

| Month | Capital | Monthly Return | Cumulative |
|-------|---------|----------------|------------|
| 1 | ₹10,00,000 | +20% | ₹12,00,000 |
| 2 | ₹12,00,000 | +25% | ₹15,00,000 |
| 3 | ₹15,00,000 | +30% | ₹19,50,000 |
| 6 | ₹19,50,000 | +35% avg | ₹66,00,000 |
| 12 | ₹66,00,000 | +35% avg | ₹2,20,00,000 |

**Note**: This assumes consistent edge and no major drawdowns. Actual results will vary.

---

## Configuration

All PRO TRADER settings are in:
```
config/pro_trader_config.json
```

To activate PRO MODE:
```bash
export TRADING_MODE=pro
python -m src.main --config config/pro_trader_config.json
```

---

## Final Words

**"The market rewards those who show up every day with an edge and the discipline to execute."**

- Don't fear losses - fear missing opportunities
- Size matters - bet big when you have edge
- Consistency beats perfection
- Let the math work: 55% win rate × 1.8 R:R = +45% expectancy per trade
- Trade like your livelihood depends on it - because it does

**Now go make money.** 💰
