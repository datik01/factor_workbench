import sys
import pandas as pd
from factor_miner import discover_alpha_factors
from constituents.universe_builder import load_universe
import tools

print("Loading universe...")
tickers = load_universe("SP500")[:30]
print(f"Loaded {len(tickers)} tickers.")

print("Fetching data...")
df = tools.fetch_universe_data(tickers, 2024, 2025, force_refresh=False)
print(f"Data shape: {df.shape}")

print("Running Miner...")
def progress(pct, msg):
    print(f"[{pct}%] {msg}")

res = discover_alpha_factors(df, generations=2, pop_size=30, progress_callback=progress)
print("Result:")
print(res)
