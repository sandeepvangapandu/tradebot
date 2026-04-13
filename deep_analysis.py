import re
import pandas as pd

lines = open("logs/trading_bot.log").readlines()

trades = []
for line in lines:
    if "Position closed:" in line and "Final P&L:" in line:
        # Extract strategy from position ID and surrounding context
        m = re.search(
            r"Position closed: (\S+) \| Reason: (\S+) \|.*?Exit: ([\d\.]+).*?Blended: ([\d\.]+).*?Final P&L: ([\-\d\.]+).*?Total P&L: ([\-\d\.]+).*?Partial exits: (\d+)",
            line
        )
        if m:
            pos_id = m.group(1)
            reason = m.group(2)
            exit_price = float(m.group(3))
            blended = float(m.group(4))
            final_pnl = float(m.group(5))
            total_pnl = float(m.group(6))
            partial_exits = int(m.group(7))
            trades.append({
                "pos_id": pos_id,
                "reason": reason,
                "exit_price": exit_price,
                "blended": blended,
                "final_pnl": final_pnl,
                "total_pnl": total_pnl,
                "partial_exits": partial_exits
            })

# Also get entry info
entries = {}
for line in lines:
    if "Position added:" in line:
        m = re.search(
            r"Position added: (\S+) \| (\S+) \| (BUY|SELL) (\d+) (\S+) @ ([\d\.]+) \| SL: ([\d\.]+) \| Target: ([\d\.]+)",
            line
        )
        if m:
            entries[m.group(1)] = {
                "strategy": m.group(2),
                "side": m.group(3),
                "qty": int(m.group(4)),
                "entry_price": float(m.group(6)),
                "sl": float(m.group(7)),
                "target": float(m.group(8))
            }

# Merge
for t in trades:
    if t["pos_id"] in entries:
        t.update(entries[t["pos_id"]])

df = pd.DataFrame(trades)
print("=== TRADE BREAKDOWN ===")
print(f"Total Trades: {len(df)}")
print(f"Total P&L: {df['total_pnl'].sum():.2f}")
print()

print("=== BY EXIT REASON ===")
reason_grp = df.groupby("reason").agg(
    count=("total_pnl", "count"),
    total_pnl=("total_pnl", "sum"),
    avg_pnl=("total_pnl", "mean"),
    wins=("total_pnl", lambda x: (x > 0).sum())
)
print(reason_grp.to_string())
print()

print("=== BY STRATEGY ===")
if "strategy" in df.columns:
    strat_grp = df.groupby("strategy").agg(
        count=("total_pnl", "count"),
        total_pnl=("total_pnl", "sum"),
        avg_pnl=("total_pnl", "mean"),
        wins=("total_pnl", lambda x: (x > 0).sum()),
        win_rate=("total_pnl", lambda x: f"{(x > 0).sum() / len(x) * 100:.1f}%")
    )
    print(strat_grp.to_string())
    print()

print("=== ALL TRADES ===")
cols = [c for c in ["strategy", "side", "entry_price", "sl", "target", "exit_price", "total_pnl", "reason", "partial_exits"] if c in df.columns]
print(df[cols].to_string())
