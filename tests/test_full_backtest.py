import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constituents.universe_builder import get_latest_constituents
from tools import run_cross_sectional_backtest
import json

tickers = get_latest_constituents("SP500")[:50]
try:
    res = run_cross_sectional_backtest(
        tickers=tickers,
        themes=["momentum_1m"],
        custom_formula="sqrt(div(add(Open, Low), abs_f(Low)))",
        start_year=2024,
        end_year=2025,
        portfolio_size=20 # Small
    )
    #print(res)
    parsed = json.loads(res)
    if "error" in parsed:
        print("ERROR IN RESULT:", parsed["error"])
    else:
        print("Plots keys:", parsed["plots"].keys())
        print("Metrics:", parsed["metrics"])
except Exception as e:
    import traceback
    traceback.print_exc()
