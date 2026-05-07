import pandas as pd
import numpy as np
import os
from pathlib import Path
from loguru import logger

def generate_synthetic_data():
    base_file = Path("data/backtest/banknifty_1m_6mo.csv")
    out_dir = Path("data/backtest")
    
    if not base_file.exists():
        logger.error(f"Base file {base_file} not found. Ensure you are in the project root.")
        return
        
    df = pd.read_csv(base_file)
    
    # We will generate synthetic Nifty 50, Finnifty, and Sensex.
    targets = {
        "nifty50": {"multiplier": 0.45, "noise_scale": 0.001},
        "finnifty": {"multiplier": 0.42, "noise_scale": 0.002},
        "sensex": {"multiplier": 1.45, "noise_scale": 0.0015},
    }
    
    for name, config in targets.items():
        logger.info(f"Generating synthetic data for {name}...")
        df_new = df.copy()
        
        # Add random walk noise to simulate different price paths
        np.random.seed(hash(name) % (2**32))
        noise = np.random.normal(0, config["noise_scale"], size=len(df))
        cumulative_noise = np.exp(np.cumsum(noise) - 0.5 * config["noise_scale"]**2 * np.arange(len(df)))
        
        # Scale and apply noise
        for col in ["open", "high", "low", "close"]:
            df_new[col] = (df[col] * config["multiplier"] * cumulative_noise).astype(int)
            
        out_file = out_dir / f"{name}_1m_6mo.csv"
        df_new.to_csv(out_file, index=False)
        logger.info(f"Saved to {out_file}")

if __name__ == "__main__":
    generate_synthetic_data()
