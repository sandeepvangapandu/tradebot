from tvDatafeed import TvDatafeed, Interval
import pandas as pd

tv = TvDatafeed()
# 5000 bars ≈ 13 trading days
df1 = tv.get_hist('BANKNIFTY', 'NSE', Interval.in_1_minute, n_bars=25000)
df2 = tv.get_hist('BANKNIFTY', 'NSE', Interval.in_1_minute, n_bars=30000)
# df2 will contain older bars; concat and deduplicate
df = pd.concat([df2, df1]).drop_duplicates().sort_index()
df.to_csv("nifty50_1min_30days.csv")