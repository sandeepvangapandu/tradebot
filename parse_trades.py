import re

logs = open("logs/trading_bot.log").readlines()
trades = []

for line in logs:
    if "Position closed:" in line and "Final P&L:" in line:
        # Example format: ... Reason: STOP_LOSS | Exit: 115.89 | Blended: ... | Final P&L: -916.35 | ...
        match = re.search(r"Reason: (.*?) \|.*Final P&L: ([\-\d\.]+) \|", line)
        if match:
            reason = match.group(1).strip()
            pnl = float(match.group(2).strip())
            trades.append({"reason": reason, "pnl": pnl})

df = pd.DataFrame(trades)
if not df.empty:
    print(df.groupby('reason').agg(['count', 'sum', 'mean']))
    print(f"\nTotal trades: {len(trades)}")
    print(f"Total PnL: {sum([t['pnl'] for t in trades])}")
else:
    print("No trades found")
