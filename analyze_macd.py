import sqlite3
import pandas as pd
conn = sqlite3.connect('data/trading_bot.db')
query = """
SELECT strategy, side, entry_price, exit_price, quantity, realized_pnl, entry_time, exit_time
FROM trades
WHERE strategy LIKE '%macd%' OR strategy LIKE '%MACD%'
"""
df = pd.read_sql(query, conn)

print(f"Total MACD Trades: {len(df)}")
print(f"Total MACD PNL: {df['realized_pnl'].sum() / 100} INR")
print(df.to_string())
