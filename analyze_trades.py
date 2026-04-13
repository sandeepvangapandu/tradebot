import sqlite3
import pandas as pd

def get_trades_summary():
    conn = sqlite3.connect('data/trading_bot.db')
    
    query = """
    SELECT strategy, side, entry_price, exit_price, quantity, realized_pnl, entry_time, exit_time
    FROM trades
    """
    df = pd.read_sql(query, conn)
    
    print(f"Total Trades: {len(df)}")
    print(f"Total PNL: {df['realized_pnl'].sum() / 100} INR")
    print(df.to_string())

if __name__ == '__main__':
    get_trades_summary()
