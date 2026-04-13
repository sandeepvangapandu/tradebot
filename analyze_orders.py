import sqlite3
import pandas as pd
conn = sqlite3.connect('data/trading_bot.db')
query = """
SELECT strategy, instrument_key, transaction_type, status, price, avg_fill_price, quantity, placed_at 
FROM orders
"""
df = pd.read_sql(query, conn)

print(f"Total Orders: {len(df)}")
print(df.to_string())
