import re
import pandas as pd

# Read entire log
with open("logs/trading_bot.log", encoding="utf-8") as f:
    content = f.read()
    lines = content.split("\n")

# Find last backtest start
last_start = 0
for i, line in enumerate(lines):
    if "RiskManager initialised" in line:
        last_start = i

recent = lines[last_start:]

entries = {}
for line in recent:
    if "Position added:" in line:
        # Use a simplified approach
        parts = line.split("Position added: ")[1] if "Position added: " in line else ""
        if parts:
            tokens = parts.split(" | ")
            pos_id = tokens[0].strip()
            strategy = tokens[1].strip() if len(tokens) > 1 else "unknown"
            
            # Extract side/qty/instrument
            side_match = re.search(r"(BUY|SELL) (\d+)", parts)
            price_match = re.search(r"@ ([\d\.]+)", parts)
            sl_match = re.search(r"SL: ([\d\.]+)", parts)
            tgt_match = re.search(r"Target: ([\d\.]+)", parts)
            
            if side_match and price_match and sl_match and tgt_match:
                ep = float(price_match.group(1))
                sl = float(sl_match.group(1))
                tgt = float(tgt_match.group(1))
                entries[pos_id] = {
                    "strategy": strategy,
                    "side": side_match.group(1),
                    "qty": int(side_match.group(2)),
                    "entry_price": ep,
                    "sl": sl,
                    "target": tgt,
                    "sl_pct": abs(ep - sl) / ep * 100 if ep > 0 else 0,
                    "target_pct": abs(tgt - ep) / ep * 100 if ep > 0 else 0
                }

closes = {}
for line in recent:
    if "Position closed:" in line:
        parts = line.split("Position closed: ")[1] if "Position closed: " in line else ""
        if parts:
            tokens = parts.split(" | ")
            pos_id = tokens[0].strip()
            
            reason_match = re.search(r"Reason: (\S+)", parts)
            pnl_match = re.search(r"Total P&L: ([\-\d\.]+)", parts)
            
            if reason_match and pnl_match:
                closes[pos_id] = {
                    "reason": reason_match.group(1),
                    "total_pnl": float(pnl_match.group(1))
                }

# Merge
rows = []
for pid, e in entries.items():
    if pid in closes:
        rows.append({**e, **closes[pid], "pos_id": pid})

df = pd.DataFrame(rows)
print(f"Entries: {len(entries)}, Closes: {len(closes)}, Merged: {len(rows)}")

if not df.empty:
    print(f"\n=== STRATEGY BREAKDOWN ({len(df)} trades) ===")
    for strat in df["strategy"].unique():
        sdf = df[df["strategy"] == strat]
        wins = (sdf["total_pnl"] > 0).sum()
        print(f"\n{strat}: {len(sdf)} trades | Win Rate: {wins/len(sdf)*100:.1f}% | Total PnL: {sdf['total_pnl'].sum():.2f}")
        print(f"  Avg SL%: {sdf['sl_pct'].mean():.1f}% | Avg Target%: {sdf['target_pct'].mean():.1f}%")
        print(f"  Wins: {wins} (avg +{sdf[sdf['total_pnl']>0]['total_pnl'].mean():.2f})" if wins > 0 else "  Wins: 0")
        losses = (sdf["total_pnl"] <= 0).sum()
        print(f"  Losses: {losses} (avg {sdf[sdf['total_pnl']<=0]['total_pnl'].mean():.2f})" if losses > 0 else "  Losses: 0")

    print(f"\n=== EXIT REASON BREAKDOWN ===")
    for reason in df["reason"].unique():
        rdf = df[df["reason"] == reason]
        print(f"{reason}: {len(rdf)} trades | Total PnL: {rdf['total_pnl'].sum():.2f} | Avg: {rdf['total_pnl'].mean():.2f}")

    print(f"\n=== AGGREGATE ===")
    print(f"Total Trades: {len(df)}")
    print(f"Total PnL: {df['total_pnl'].sum():.2f}")
    print(f"Win Rate: {(df['total_pnl']>0).sum()/len(df)*100:.1f}%")
    print(f"Winners: {(df['total_pnl']>0).sum()}, Losers: {(df['total_pnl']<=0).sum()}")
    if (df['total_pnl']>0).sum() > 0:
        print(f"Avg Win: {df[df['total_pnl']>0]['total_pnl'].mean():.2f}")
    if (df['total_pnl']<=0).sum() > 0:
        print(f"Avg Loss: {df[df['total_pnl']<=0]['total_pnl'].mean():.2f}")
    
    total_wins = df[df['total_pnl']>0]['total_pnl'].sum()
    total_losses = abs(df[df['total_pnl']<=0]['total_pnl'].sum())
    if total_losses > 0:
        print(f"Profit Factor: {total_wins/total_losses:.2f}")
