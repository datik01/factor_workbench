import unittest
import pandas as pd
import numpy as np
from factor_miner import discover_alpha_factors

class TestFactorMiner(unittest.TestCase):
    def test_factor_miner_loop(self):
        # Create a synthetic DataFrame mimicking tools.fetch_universe_data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", "2024-02-01")
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
        
        # Test the loop
        results = discover_alpha_factors(
            df, 
            generations=2, 
            pop_size=20, 
            progress_callback=lambda p, m: print(f"[{p}%] {m}")
        )
        
        print("Final GP Results:", results)
        self.assertTrue(len(results) > 0, "Miner failed to return any formulas.")
        for res in results:
            self.assertIn("formula", res)
            self.assertIn("ic_fitness", res)

if __name__ == '__main__':
    unittest.main()
