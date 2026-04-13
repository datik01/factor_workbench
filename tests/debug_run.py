import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import run_cross_sectional_backtest
from app import THEMES, ALL_TICKERS, CONSTITUENT_TIMELINE
import json
import random

sampled = random.sample(ALL_TICKERS, 100)
res_str = run_cross_sectional_backtest(
    tickers=sampled,
    theme="momentum_1m",
    start_year=2024,
    end_year=2026,
    invert_factor=False,
    rebalance_freq="M",
    initial_aum=1000000,
    constituent_timeline=CONSTITUENT_TIMELINE
)
print("ERROR IS:", json.loads(res_str).get("error", "SUCCESS"))
