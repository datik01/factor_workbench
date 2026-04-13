import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constituents.universe_builder import get_latest_constituents
from tools import run_cross_sectional_backtest
import json

tickers = get_latest_constituents("SP500")[:100]
res = run_cross_sectional_backtest(
    tickers=tickers,
    themes=[],
    custom_formula="div(add(Open, Low), abs(Low))",
    start_year=2024,
    end_year=2025,
    portfolio_size=20
)
parsed = json.loads(res)
if "error" in parsed:
    print("ERROR IN RESULT:", parsed["error"])
else:
    print("Metrics:", parsed["metrics"])
    print("Plots length:", len(parsed.get("plots", {})))
