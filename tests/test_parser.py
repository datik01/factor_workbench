import unittest
import pandas as pd
import numpy as np
import json
from tools import _compute_factor_scores, execute_gplearn_formula

class TestParser(unittest.TestCase):
    def test_parser(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", "2024-01-10")
        tickers = ["AAPL", "MSFT"]
        
        records = []
        for d in dates:
            for t in tickers:
                records.append({
                    "date": d,
                    "ticker": t,
                    "open": np.random.rand() * 100,
                    "high": np.random.rand() * 100 + 5,
                    "low": np.random.rand() * 100 - 5,
                    "close": np.random.rand() * 100,
                    "volume": np.random.randint(1000, 1000000)
                })
        df = pd.DataFrame(records)
        
        # Add basic dummy
        df["daily_return"] = df.groupby("ticker")["close"].pct_change()
        
        formula = "sqrt(div(add(Open, Low), abs(Low)))"
        res = _compute_factor_scores(df, themes=[], custom_formula=formula)
        
        print("Success evaluating custom formula.")
        print(res.head())
        self.assertTrue("factor_score" in res.columns)

if __name__ == '__main__':
    unittest.main()
