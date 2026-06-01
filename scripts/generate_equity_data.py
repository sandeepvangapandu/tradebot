#!/usr/bin/env python3
"""Generate synthetic 1-min equity CSV files for new instruments.

Uses an existing equity file as a base, scales prices to the target range,
and applies sector-matched beta/noise to produce realistic synthetic data
for backtesting. Output format matches existing equity CSVs (paisa, IST).

Usage:
    python3 scripts/generate_equity_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("data/backtest/equity")

# (symbol, base_csv, target_open_paisa, beta, noise_scale, seed)
# beta: intraday move scaling vs base
# noise_scale: log-return noise std per bar
TARGETS = [
    # WIPRO: IT sector, lower beta than INFY (~0.75x), price ~₹270
    ("WIPRO",       "infy_1m_6mo.csv",      27_000, 0.75, 0.0012,  42),
    # AXISBANK: Banking sector, higher beta than ICICIBANK (~1.10x), price ~₹1,100
    ("AXISBANK",    "icicibank_1m_6mo.csv", 110_000, 1.10, 0.0014,  77),
    # MARUTI: Auto sector, mid-beta (~0.90x vs TCS), price ~₹12,500
    ("MARUTI",      "tcs_1m_6mo.csv",     1_250_000, 0.90, 0.0010,  99),
    # TATAMOTORS: High-beta auto (~1.20x vs SBIN), price ~₹700 — strong trending behavior
    ("TATAMOTORS",  "sbin_1m_6mo.csv",      70_000, 1.20, 0.0016, 113),
    # HCLTECH: IT but higher beta than WIPRO (~1.10x vs INFY), price ~₹1,800 — cleaner trends
    ("HCLTECH",     "infy_1m_6mo.csv",     180_000, 1.10, 0.0013, 131),
    # BHARTIARTL: Telecom, medium-beta, sustained multi-month trends, price ~₹1,600
    ("BHARTIARTL",  "icicibank_1m_6mo.csv", 160_000, 0.95, 0.0011, 157),
    # ADANIENT: High-beta conglomerate, large ATR, frequent breakouts, price ~₹2,600
    ("ADANIENT",    "tcs_1m_6mo.csv",      260_000, 1.30, 0.0018, 179),
    # HINDALCO: High-beta metals/aluminium, strong cyclical trends, price ~₹680
    ("HINDALCO",    "tcs_1m_6mo.csv",       68_000, 1.25, 0.0016, 201),
    # JSWSTEEL: High-beta steel, large intraday ATR, frequent breakouts, price ~₹990
    ("JSWSTEEL",    "tcs_1m_6mo.csv",       99_000, 1.30, 0.0017, 223),
]


def generate(symbol: str, base_file: str, target_open: int,
             beta: float, noise_scale: float, seed: int) -> None:
    src = OUT_DIR / base_file
    if not src.exists():
        print(f"  SKIP {symbol}: base file {src} not found")
        return

    df = pd.read_csv(src)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        print(f"  SKIP {symbol}: missing columns in {src}")
        return

    rng = np.random.default_rng(seed)

    # Scale factor: map base open → target open
    base_open = df["open"].iloc[0]
    scale = target_open / base_open

    out = df.copy()
    for col in ("open", "high", "low", "close"):
        # Scale + apply beta-weighted log-return noise per bar
        prices = df[col].values.astype(float) * scale
        n = len(prices)
        noise = rng.normal(0, noise_scale * abs(beta - 1.0 + 0.01), size=n)
        drift = np.exp(np.cumsum(noise) - 0.5 * noise_scale**2 * np.arange(n))
        out[col] = np.round(prices * drift).astype(int)

    # Enforce OHLC sanity: high >= max(open,close), low <= min(open,close)
    out["high"] = np.maximum(out["high"], np.maximum(out["open"], out["close"]))
    out["low"]  = np.minimum(out["low"],  np.minimum(out["open"], out["close"]))

    # Volume: scale inversely with price (higher price stocks have lower share volume)
    price_ratio = base_open / target_open
    out["volume"] = np.round(df["volume"].values * price_ratio * (0.8 + rng.random(len(df)) * 0.4)).astype(int)

    dst = OUT_DIR / f"{symbol.lower()}_1m_6mo.csv"
    out.to_csv(dst, index=False)
    rows = len(out)
    avg_close = out["close"].mean() / 100
    print(f"  {symbol:<12} {rows:>6} bars  avg_close ₹{avg_close:,.0f}  → {dst}")


def main() -> None:
    print(f"\nGenerating synthetic equity data in {OUT_DIR}/\n")
    for args in TARGETS:
        generate(*args)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
