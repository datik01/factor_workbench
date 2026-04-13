import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constituents.universe_builder import get_latest_constituents
from tools import run_cross_sectional_backtest

tickers = get_latest_constituents("SP500")[:20]
try:
    res = run_cross_sectional_backtest(
        tickers=tickers,
        themes=[],
        custom_formula="sqrt(div(add(Open, Low), abs_f(Low)))",
        start_year=2024,
        end_year=2025
    )
    print("Success. Length res:", len(res))
    import json
    parsed = json.loads(res)
    print("Metrics:", parsed.get("metrics"))
except Exception as e:
    print("ERROR:", e)
