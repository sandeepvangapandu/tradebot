import pandas as pd
from src.strategy.indicators import IndicatorEngine

df = pd.read_csv('nifty50_1min_30days 2.csv')
df.rename(columns={'datetime': 'timestamp'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)

engine = IndicatorEngine(df)

dte = (2 - df.index.dayofweek) % 7
dte_series = pd.Series(dte, index=df.index)

atr = engine.atr(14)
atr_ma = atr.rolling(50).mean()
synth_iv = 20 * (atr / atr_ma.replace(0, float("nan")))
synth_iv = synth_iv.fillna(20.0)

adx = engine.adx(14)

out = pd.DataFrame({'dte': dte_series, 'iv': synth_iv, 'adx': adx})
wednesdays = out[out['dte'] == 0]
print("Wednesdays data summary:")
print(wednesdays.describe())
print(wednesdays.tail(10))
