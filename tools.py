"""
tools.py
Factor Workbench: Institutional-Grade Cross-Sectional Portfolio Engine
Danny Atik - SYSEN 5381

Designed for full-universe R2K factor analysis:
  - Concurrent data fetching via ThreadPoolExecutor
  - Local parquet cache to avoid redundant API calls
  - Cross-sectional quintile portfolio construction
  - IC analysis, regression alpha/beta, drawdown analytics
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from massive import RESTClient
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import requests
from pnl_calendar import generate_pnl_calendar_html
import orderflow

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))
load_dotenv(os.path.join(_script_dir, "..", "..", ".env"))

API_KEY = os.getenv("MASSIVE_API_KEY")
CACHE_DIR = os.path.join(_script_dir, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_WORKERS = 15  # concurrent API threads

def send_pushover_alert(title, msg, priority=0, url=None, url_title=None):
    """Sends a push notification via Pushover."""
    pushover_user = "unhnekvna84gojok2dnegsopvkxprk"
    pushover_token = "ag3g78r79s9hnsq7eis1z53an1nbfo"
    try:
        data = {
            "token": pushover_token,
            "user": pushover_user,
            "message": msg,
            "title": title,
            "priority": priority,
        }
        if url:
            data["url"] = url
        if url_title:
            data["url_title"] = url_title
            
        requests.post("https://api.pushover.net/1/messages.json", data=data, timeout=5)
    except Exception as e:
        print(f"Pushover failed: {e}")

# ═══════════════════════════════════════════════════════════════
# Data Layer: Single-Ticker Fetch
# ═══════════════════════════════════════════════════════════════

def _fetch_single_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV for one ticker. Returns empty DataFrame on failure."""
    if not API_KEY:
        return pd.DataFrame()
    try:
        client = RESTClient(api_key=API_KEY)
        resp = client.list_aggs(
            ticker=ticker.upper(), multiplier=1, timespan="day",
            from_=start_date, to=end_date,
            sort="asc", limit=50000, raw=True,
        )
        data = json.loads(resp.data.decode("utf-8"))
        bars = data.get("results", [])
        if not bars:
            return pd.DataFrame()

        records = []
        for bar in bars:
            t = bar.get("t") or bar.get("timestamp")
            date_str = datetime.fromtimestamp(t / 1000).strftime("%Y-%m-%d") if t else ""
            records.append({
                "date": date_str,
                "ticker": ticker.upper(),
                "open": bar.get("o") or bar.get("open"),
                "high": bar.get("h") or bar.get("high"),
                "low": bar.get("l") or bar.get("low"),
                "close": bar.get("c") or bar.get("close"),
                "volume": bar.get("v") or bar.get("volume", 0),
                "vwap": bar.get("vw"),
                "trades": bar.get("n"),
            })
        df = pd.DataFrame(records)
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        
        # --- PIT FUNDAMENTAL HYBRIDIZATION ---
        try:
            fin_url = f"https://api.polygon.io/vX/reference/financials?ticker={ticker.upper()}&timeframe=quarterly&limit=100&sort=filing_date&apiKey={API_KEY}"
            f_resp = requests.get(fin_url, timeout=12)
            if f_resp.status_code == 200:
                f_data = f_resp.json().get("results", [])
                
                f_records = []
                for rep in f_data:
                    f_date = rep.get("filing_date")
                    if not f_date: continue
                    fin = rep.get("financials", {})
                    inc = fin.get("income_statement", {})
                    bal = fin.get("balance_sheet", {})
                    cf = fin.get("cash_flow_statement", {})

                    record = {"date": f_date}
                    fields_to_extract = {
                        "eps": (inc, "basic_earnings_per_share"),
                        "revenues": (inc, "revenues"),
                        "gross_profit": (inc, "gross_profit"),
                        "cost_of_revenue": (inc, "cost_of_revenue"),
                        "operating_income": (inc, "operating_income_loss"),
                        "net_income": (inc, "net_income_loss"),
                        "interest_expense": (inc, "interest_expense_operating"),
                        "research_and_development": (inc, "research_and_development"),
                        "shares": (inc, "weighted_average_number_of_shares_outstanding_basic"),
                        
                        "equity": (bal, "equity"),
                        "assets": (bal, "assets"),
                        "liabilities": (bal, "liabilities"),
                        "current_assets": (bal, "current_assets"),
                        "current_liabilities": (bal, "current_liabilities"),
                        "inventory": (bal, "inventory"),
                        
                        "net_cash_flow": (cf, "net_cash_flow"),
                        "operating_cash_flow": (cf, "net_cash_flow_from_operating_activities"),
                        "dividends_paid": (cf, "net_cash_flow_from_financing_activities_dividend_payments"),
                    }
                    
                    for key, (statement, poly_key) in fields_to_extract.items():
                        record[key] = statement.get(poly_key, {}).get("value", np.nan)
                        
                    # Structural fallback for shares outstanding missing from Income Statement using Balance Sheet
                    if pd.isna(record.get("shares")):
                        record["shares"] = bal.get("common_stock_shares_outstanding", {}).get("value", np.nan)
                        
                    f_records.append(record)
                if f_records:
                    f_df = pd.DataFrame(f_records)
                    f_df["date"] = pd.to_datetime(f_df["date"])
                    f_df = f_df.sort_values("date").drop_duplicates("date")
                    
                    # Asof merge accurately simulating forward-fill timeline natively without future bias
                    df = pd.merge_asof(df.sort_values("date"), f_df, on="date", direction="backward")
                    
                    # Fill explicit nans for math boundaries securely
                    df.ffill(inplace=True)
                    df.fillna(0, inplace=True)
                    
                    # Build Composite Factors
                    df["market_cap"] = df["shares"] * df["close"]
                    df["pe_ratio"] = np.where((df["eps"] > 0) & (df["close"] > 0), df["close"] / df["eps"], 0.0)
                    df["pb_ratio"] = np.where((df["equity"] > 0) & (df["close"] > 0), df["close"] / (df["equity"] / 1e6), 0.0) # Scaling proxy
                    df["ps_ratio"] = np.where((df["revenues"] > 0) & (df["close"] > 0), df["close"] / (df["revenues"] / 1e6), 0.0)
                    
        except Exception:
            pass
            
        # Ensure default fundamental columns exist if fetch purely failed
        expected_cols = [
            "pe_ratio", "pb_ratio", "ps_ratio", "eps", "revenues", "gross_profit", "cost_of_revenue",
            "operating_income", "net_income", "interest_expense", "research_and_development", "shares", "market_cap",
            "equity", "assets", "liabilities", "current_assets", 
            "current_liabilities", "inventory", "net_cash_flow", "operating_cash_flow", "dividends_paid"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Data Layer: Concurrent Universe Fetch + Cache
# ═══════════════════════════════════════════════════════════════

def _cache_path(n_tickers: int, start_year: int, end_year: int) -> str:
    today = datetime.now().strftime("%Y%m%d")
    return os.path.join(CACHE_DIR, f"universe_{n_tickers}_{start_year}_{end_year}_{today}.parquet")


def fetch_universe_data(
    tickers: list,
    start_year: int = 2020,
    end_year: int = 2025,
    progress_callback=None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Batch-fetch daily OHLCV for the full universe using concurrent API calls.
    Results are cached locally as parquet for same-day re-runs.

    Parameters
    ----------
    tickers : list of str
        Full universe (e.g. 2000 R2K constituents)
    lookback_years : int
    progress_callback : callable(current, total, ticker, status)
    force_refresh : bool
        If True, bypass cache and re-fetch everything

    Returns
    -------
    pd.DataFrame
        Panel data indexed by date with ticker column
    """
    cache_file_target = _cache_path(len(tickers), start_year, end_year)

    # Check cache first (Decoupled from explicit Date bound to allow cross-day persistence of structural data)
    if not force_refresh:
        import glob
        import re
        
        all_caches = glob.glob(os.path.join(CACHE_DIR, "universe_*.parquet"))
        valid_caches = []
        
        # Scan filesystem for ANY cache file encapsulating the requested temporal vectors!
        for cache_candidate in all_caches:
            basename = os.path.basename(cache_candidate)
            match = re.search(r'universe_\d+_(\d{4})_(\d{4})_', basename)
            if match:
                c_start, c_end = int(match.group(1)), int(match.group(2))
                # Valid iff the cached timeline formally traps our execution bounds
                if c_start <= start_year and c_end >= end_year:
                    valid_caches.append(cache_candidate)
                    
        if valid_caches:
            # Check the largest valid cache files first to maximize matrix intersection probability
            latest_cache = sorted(valid_caches, key=os.path.getsize, reverse=True)[0]
            if progress_callback:
                progress_callback(0, 0, "", "Loading from persistent cross-sectional cache...")
            df = pd.read_parquet(latest_cache)
            
            # Physically intersect the massive cache dynamically against the target requested array
            cached_tickers = set(df["ticker"].unique())
            target_tickers = set(tickers)
            intersect = cached_tickers.intersection(target_tickers)
            
            # Safety threshold has been completely removed to prevent synchronous API rebuilds at 09:25 AM
            if "eps" in df.columns:
                if "eps" not in df.columns or (df["pe_ratio"] == 0).all():
                    pass # Trigger rebuild naturally
                else:
                    from datetime import datetime
                    
                    # Standardize boundary arrays to explicit temporal string vectors cleanly!
                    cached_max_date = str(df["date"].max())[:10]
                    # Check if market has closed today (after 5 PM EST) to safely fetch today's aggregates
                    now = pd.Timestamp.now(tz="US/Eastern")
                    if now.hour >= 17:
                        target_date = (now.tz_localize(None).normalize() - pd.tseries.offsets.BDay(0)).strftime("%Y-%m-%d")
                    else:
                        target_date = (now.tz_localize(None).normalize() - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
                    
                    cutoff_end_limit = f"{end_year}-12-31"
                    target_date = min(target_date, cutoff_end_limit)
                    
                    # Delta Merge Subsystem (Fetch missing daily boundaries silently)
                    if cached_max_date < target_date:
                        days_diff = (pd.to_datetime(target_date) - pd.to_datetime(cached_max_date)).days
                        if days_diff >= 1:
                            if progress_callback:
                                progress_callback(0, 0, "", f"Cache decayed! Delta-Fetching {days_diff} missing days from {cached_max_date}...")
                            
                            delta_frames = []
                            d_failed = 0
                            d_completed = 0
                            d_total = len(tickers)
                            
                            def _d_worker(t):
                                return t, _fetch_single_ticker(t, cached_max_date, target_date)
                                
                            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                d_futures = {executor.submit(_d_worker, t): t for t in tickers} # Only ping API for active subset 
                                for future in as_completed(d_futures):
                                    t_non = d_futures[future]
                                    try:
                                        t, d_df = future.result()
                                        if not d_df.empty: 
                                            delta_frames.append(d_df)
                                        else:
                                            d_failed += 1
                                    except Exception:
                                        d_failed += 1
                                        
                                    d_completed += 1
                                    if progress_callback and d_completed % 25 == 0:
                                        progress_callback(d_completed, d_total, t_non, f"📡 Delta-Sync {d_completed}/{d_total} ({d_failed} gaps)")
                            
                            if delta_frames:
                                delta_df = pd.concat(delta_frames, ignore_index=True)
                                df = pd.concat([df, delta_df], ignore_index=True)
                                df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
                                df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
                                df.to_parquet(latest_cache, index=False) # Safely overwrite Master Cache
                    
                    # Dynamically subset the master dataframe securely preventing massive GPU RAM consumption
                    df = df[df["ticker"].isin(intersect)].copy()
                    
                    # Physically sever the bounds to only export the targeted Matrix slice mechanically padding -1 year for technical lag hooks!
                    cutoff_start = f"{start_year - 1}-01-01"
                    df = df[(df["date"] >= cutoff_start) & (df["date"] <= cutoff_end_limit)].copy()
                    
                    if progress_callback:
                        progress_callback(len(intersect), len(intersect), "", f"Sub-mapped {len(intersect)} targets explicitly from Master Cache ({start_year}-{end_year}).")
                    return df
            else:
                if progress_callback:
                    progress_callback(0, 0, "", f"Cache intersection gap ({len(intersect)}/{len(tickers)}). Rebuilding specific cache...")

    if not API_KEY:
        raise ValueError("MASSIVE_API_KEY is not set in .env")

    # Pad start_date by 1 year (365 days) to ensure 252-day momentum can be calculated for day 1 of start_year
    start_date = f"{start_year - 1}-01-01"
    end_date = f"{end_year}-12-31"

    all_frames = []
    completed = 0
    total = len(tickers)
    failed = 0

    def _worker(ticker):
        return ticker, _fetch_single_ticker(ticker, start_date, end_date)

    if progress_callback:
        progress_callback(0, total, "", f"Fetching {total} tickers ({MAX_WORKERS} threads)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_worker, t): t for t in tickers}

        for future in as_completed(futures):
            nonlocal_ticker = futures[future]
            try:
                ticker, df = future.result()
                if not df.empty:
                    all_frames.append(df)
                else:
                    failed += 1
            except Exception:
                failed += 1

            completed += 1
            if progress_callback and completed % 25 == 0:
                progress_callback(
                    completed, total, nonlocal_ticker,
                    f"📡 {completed}/{total} records verified ({failed} missing/delisted)"
                )

    if not all_frames:
        raise ValueError("No data fetched for any ticker in the universe.")

    universe = pd.concat(all_frames, ignore_index=True)
    universe = universe.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Cache to parquet matching the generated exact wildcard bound structurally
    universe.to_parquet(cache_file_target, index=False)

    if progress_callback:
        n = universe["ticker"].nunique()
        progress_callback(n, n, "", f"✅ {n}/{total} cross-sections loaded ({failed} missing/delisted)")

    return universe


# ═══════════════════════════════════════════════════════════════
# Factor Computation (Cross-Sectional)
# ═══════════════════════════════════════════════════════════════

def execute_gplearn_formula(df: pd.DataFrame, formula_str: str) -> np.ndarray:
    """
    Safely translates internal gplearn Abstract Syntax Trees into raw pandas/numpy executions natively.
    """
    def add(a, b): return a + b
    def sub(a, b): return a - b
    def mul(a, b): return a * b
    def div(a, b):
        b_safe = np.where(np.abs(b) < 1e-6, 1.0, b)
        return np.where(np.abs(b) < 1e-6, 1.0, a / b_safe)
    def abs_f(a): return np.abs(a)
    def sqrt(a): return np.sqrt(np.abs(a))
    def log(a): return np.log(np.abs(a) + 1e-5)
    def rank(a): return pd.Series(a).rank(pct=True).values

    # Temporal boundary maps ensuring bleeding doesn't occur across tickers
    t_mask_5 = (df['ticker'] != df['ticker'].shift(5)).values
    t_mask_10 = (df['ticker'] != df['ticker'].shift(9)).values
    t_mask_20 = (df['ticker'] != df['ticker'].shift(19)).values
    
    def _arr(x):
        if isinstance(x, (float, int)): return np.full(len(df), float(x))
        return np.asarray(x)

    def delay_5(a):
        a = _arr(a)
        r = np.roll(a, 5)
        r[:5] = a[:5]; r[t_mask_5] = a[t_mask_5]
        return r
        
    t_mask_sma_5 = (df['ticker'] != df['ticker'].shift(4)).values
    def sma_5(a):
        a = _arr(a)
        r = pd.Series(a).rolling(5).mean().bfill().values.copy()
        r[t_mask_sma_5] = a[t_mask_sma_5]
        return r
    def sma_10(a):
        a = _arr(a)
        r = pd.Series(a).rolling(10).mean().bfill().values.copy()
        r[t_mask_10] = a[t_mask_10]
        return r
    t_mask_60 = (df['ticker'] != df['ticker'].shift(59)).values
    
    def sma_20(a):
        a = _arr(a)
        r = pd.Series(a).rolling(20).mean().bfill().values.copy()
        r[t_mask_20] = a[t_mask_20]
        return r
    def sma_60(a):
        a = _arr(a)
        r = pd.Series(a).rolling(60).mean().bfill().values.copy()
        r[t_mask_60] = a[t_mask_60]
        return r
    def ts_max_20(a):
        a = _arr(a)
        r = pd.Series(a).rolling(20).max().bfill().values.copy()
        r[t_mask_20] = a[t_mask_20]
        return r
    def ts_min_20(a):
        a = _arr(a)
        r = pd.Series(a).rolling(20).min().bfill().values.copy()
        r[t_mask_20] = a[t_mask_20]
        return r

    t_mask_14 = (df['ticker'] != df['ticker'].shift(13)).values
    t_mask_26 = (df['ticker'] != df['ticker'].shift(25)).values
    
    def vol_10(a):
        a = _arr(a)
        r = pd.Series(a).rolling(10).std().bfill().values.copy()
        r[t_mask_10] = 0.0 
        return r
    def vol_20(a):
        a = _arr(a)
        r = pd.Series(a).rolling(20).std().bfill().values.copy()
        r[t_mask_20] = 0.0 
        return r
    def vol_60(a):
        a = _arr(a)
        r = pd.Series(a).rolling(60).std().bfill().values.copy()
        r[t_mask_60] = 0.0 
        return r

    def rsi_14(a):
        a = _arr(a)
        delta = pd.Series(a).diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(14).mean().bfill()
        avg_loss = loss.rolling(14).mean().bfill()
        rs = avg_gain / avg_loss.replace(0, 1e-5)
        rsi = 100 - (100 / (1 + rs))
        r = rsi.values.copy()
        r[t_mask_14] = 50.0  # Reset to neutral cross-ticker boundary
        return r

    def macd_line(a):
        a = _arr(a)
        sema = pd.Series(a).ewm(span=12, adjust=False).mean()
        lema = pd.Series(a).ewm(span=26, adjust=False).mean()
        r = (sema - lema).values.copy()
        r[t_mask_26] = 0.0 # Reset cross-ticker boundary
        return r

    df = df.copy()
    # Natively protect against internal array masking evaluation collapse explicitly
    df.fillna(0.0, inplace=True)

    # Pre-parse memory bindings matching factor_miner's target states
    env = {
        "add": add, "sub": sub, "mul": mul, "div": div,
        "abs": abs_f, "sqrt": sqrt, "log": log, "rank": rank,
        "delay_5": delay_5, "sma_5": sma_5, "sma_10": sma_10, "sma_20": sma_20, "sma_60": sma_60,
        "ts_max_20": ts_max_20, "ts_min_20": ts_min_20,
        "vol_10": vol_10, "vol_20": vol_20, "vol_60": vol_60, "rsi_14": rsi_14, "macd_line": macd_line,
    }
    
    # Dynamically bind ALL numeric columns securely eradicating mapping errors effortlessly!
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col.lower() not in env:
                env[col.lower()] = df[col].values
            if col.upper() not in env:
                env[col.upper()] = df[col].values
            
    # Secure string enforcement completely eradicating case-sensitivity crashes
    f_str = formula_str.lower()
    
    # Fully isolates eval mapping against sandbox dictionary
    return eval(f_str, {"__builtins__": {}}, env)

def _compute_factor_scores(universe: pd.DataFrame, themes: list, custom_formula: str = None, progress_callback=None, universe_filter: str = None, execution_timing: str = "Current Day Close") -> pd.DataFrame:
    """
    For each ticker, compute a daily factor score across multiple composites.
    Cross-sectionally ranks each specific factor, and computes the Rank-Sum equal-weight.
    """
    df = universe.copy()
    
    df = df.sort_values(["ticker", "date"])
    
    df["_is_valid_universe"] = True
    if universe_filter and universe_filter.strip():
        if progress_callback: progress_callback(10, 100, "", f"Applying Universe Filter: {universe_filter}")
        
    filter_and_formula = str(universe_filter) + " " + str(custom_formula)
    
    # Dynamically inject requested technical boundary columns for eval logic
    if "high_3m" in filter_and_formula:
        # 63 trading days = 3 months
        df["high_3m"] = df.groupby("ticker")["high"].shift(1).groupby(df["ticker"]).rolling(63, min_periods=21).max().reset_index(level=0, drop=True)
    if "high_1m" in filter_and_formula:
        # 21 trading days = 1 month
        df["high_1m"] = df.groupby("ticker")["high"].shift(1).groupby(df["ticker"]).rolling(21, min_periods=10).max().reset_index(level=0, drop=True)
    if "perf_1w" in filter_and_formula:
        # 5 trading days = 1 week
        df["perf_1w"] = df.groupby("ticker")["close"].pct_change(5)
    if "perf_1d" in filter_and_formula:
        # 1 trading day = daily performance
        df["perf_1d"] = df.groupby("ticker")["close"].pct_change(1)
    if "avg_30d_volume" in filter_and_formula or "rel_vol_1d" in filter_and_formula:
        # 21 trading days ~ 1 calendar month
        df["avg_30d_volume"] = df.groupby("ticker")["volume"].shift(1).groupby(df["ticker"]).rolling(21, min_periods=10).mean().reset_index(level=0, drop=True)
        if "rel_vol_1d" in filter_and_formula:
            df["rel_vol_1d"] = df["volume"] / df["avg_30d_volume"]
            
    if "sma_20" in filter_and_formula:
        df["_sma_20_val"] = df.groupby("ticker")["close"].shift(1).groupby(df["ticker"]).rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
        if "dist_sma_20" in filter_and_formula:
            df["dist_sma_20"] = df["close"] / df["_sma_20_val"]
            
    if "atr_pct_14" in filter_and_formula:
        df["prev_close"] = df.groupby("ticker")["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["prev_close"]).abs()
        tr3 = (df["low"] - df["prev_close"]).abs()
        df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
        df["atr_14"] = df.groupby("ticker")["tr"].rolling(14, min_periods=7).mean().reset_index(level=0, drop=True)
        df["atr_pct_14"] = df["atr_14"] / df["close"]
        
    if "rsi_14" in filter_and_formula:
        delta = df.groupby("ticker")["close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.groupby(df["ticker"]).ewm(com=13, adjust=False).mean().reset_index(level=0, drop=True)
        ema_down = down.groupby(df["ticker"]).ewm(com=13, adjust=False).mean().reset_index(level=0, drop=True)
        rs = ema_up / ema_down
        df["rsi_14"] = 100 - (100 / (1 + rs))
            
    # Support user's custom GP formula in the universe filter dynamically
    if "sma_10(volume / ts_max_20(trades))" in universe_filter:
        df["custom_gp_filter"] = execute_gplearn_formula(df, "sma_10(volume / ts_max_20(trades))")
        universe_filter = universe_filter.replace("sma_10(volume / ts_max_20(trades))", "custom_gp_filter")
        
    try:
        df["_is_valid_universe"] = df.eval(universe_filter)
    except Exception as e:
        raise ValueError(f"Invalid Universe Filter syntax: {e}")

    # Per-ticker rolling calculations
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    df["returns"] = df["daily_return"]
    
    # Boundary logic separating physical execution arrays structurally avoiding synthetic latency
    if execution_timing == "Intraday (Open to Close)":
        next_close = df.groupby("ticker")["close"].shift(-1)
        next_open = df.groupby("ticker")["open"].shift(-1)
        df["fwd_return"] = ((next_close - next_open) / next_open).clip(lower=-0.5, upper=0.5)
    else:
        df["fwd_return"] = df.groupby("ticker")["daily_return"].shift(-1).clip(lower=-0.5, upper=0.5)
    
    # Fast native cross-sectional percentile scaling to bypass pandas groupby.rank
    def _fast_cross_rank(arr):
        temp = pd.DataFrame({"d": df["date"].values, "v": arr, "i": np.arange(len(df))})
        temp = temp.sort_values(["d", "v"])
        temp["cnt"] = temp.groupby("d").cumcount() + 1.0
        sz = temp.groupby("d")["i"].transform("size")
        temp["pct"] = temp["cnt"] / sz
        return temp.sort_values("i")["pct"].values

    # Sandbox Escape: Prioritize GP algorithm overrides
    if custom_formula and custom_formula.strip():
        if progress_callback:
            progress_callback(50, 100, "", f"Injecting abstract GP Formula internally: {custom_formula}")
        df["factor_score"] = execute_gplearn_formula(df, custom_formula)
        df.loc[~df["_is_valid_universe"], "factor_score"] = np.nan # Enforce universe filter chronologically
        
        # Absolutely NO jitter injection. It permanently corrupts cross-temporal execution determinism!
        df["factor_rank"] = _fast_cross_rank(df["factor_score"].values)
        return df

    rank_cols = []
    
    for i, theme in enumerate(themes):
        if progress_callback:
            progress_callback(i, len(themes), "", f"Ranking multi-factor: {theme}...")
            
        theme_lower = theme.lower()
        col_name = f"fs_{theme_lower}"

        if "momentum_1m" in theme_lower:
            df[col_name] = df.groupby("ticker")["close"].pct_change(21)
        elif "momentum_3m" in theme_lower:
            df[col_name] = df.groupby("ticker")["close"].pct_change(63)
        elif "momentum_6m" in theme_lower:
            df[col_name] = df.groupby("ticker")["close"].pct_change(126)
        elif "momentum_12m" in theme_lower:
            df[col_name] = df.groupby("ticker")["close"].pct_change(252)
        elif "momentum" in theme_lower:
            df[col_name] = df.groupby("ticker")["close"].pct_change(20)
        elif "reversion" in theme_lower:
            df[col_name] = -df.groupby("ticker")["close"].pct_change(5)
        elif "volatility" in theme_lower:
            r_vol = pd.Series(df["daily_return"].values).rolling(20).std().values
            r_vol[(df["ticker"] != df["ticker"].shift(19)).values] = np.nan
            df[col_name] = -r_vol
        elif "volume" in theme_lower:
            vol = df["volume"].values
            sma_vol = pd.Series(vol).rolling(20).mean().values
            sma_vol[(df["ticker"] != df["ticker"].shift(19)).values] = np.nan
            df[col_name] = vol / sma_vol
        elif "size" in theme_lower:
            df[col_name] = -(df["close"] * df["volume"])
        else:
            df[col_name] = df.groupby("ticker")["close"].pct_change(10)
            
        df[f"rank_{col_name}"] = _fast_cross_rank(df[col_name].values)
        rank_cols.append(f"rank_{col_name}")

    if progress_callback:
        progress_callback(len(themes), len(themes), "", "Generating composite rankings...")

    df["factor_score"] = df[rank_cols].mean(axis=1)
    df.loc[~df["_is_valid_universe"], "factor_score"] = np.nan # Enforce universe filter chronologically

    # Cross-sectional rank of the composite mean within each day (percentile 0..1)
    df["factor_rank"] = _fast_cross_rank(df["factor_score"].values)

    return df


# ═══════════════════════════════════════════════════════════════
# Portfolio Construction & Backtest
# ═══════════════════════════════════════════════════════════════

def _pit_filter(scored_df: pd.DataFrame, timeline: dict, progress_callback=None) -> pd.DataFrame:
    """
    Point-in-Time constituent filter.
    For each trading day, keep only tickers that were actual R2K members
    at the most recent quarterly rebalance before that day.
    """
    if not timeline or scored_df.empty:
        return scored_df

    quarter_dates = pd.Series(sorted(pd.to_datetime(list(timeline.keys()))))
    quarter_tickers = {pd.to_datetime(k): set(v) for k, v in timeline.items()}

    # Pre-compute the valid tickers for every unique date utilizing fast array broadcasting
    unique_dates = scored_df['date'].unique()
    latest_date_in_df = scored_df['date'].max()
    valid_map = {}
    
    if progress_callback:
        progress_callback(10, 100, "", "Pre-computing quarterly boundary mappings...")
        
    for d in unique_dates:
        if d == latest_date_in_df:
            # Exempt absolute execution payload from historical lag truncation to maintain identical live execution parity!
            valid_map[d] = set(scored_df[scored_df['date'] == d]['ticker'])
        else:
            valid_qs = quarter_dates[quarter_dates <= d]
            if valid_qs.empty:
                q_key = quarter_dates.iloc[0]
            else:
                q_key = valid_qs.iloc[-1]
            valid_map[d] = quarter_tickers[q_key]

    if progress_callback:
        progress_callback(50, 100, "", "Executing vectorized PIT filter mapping...")
    
    # Natively extract C arrays for 100x speedup over pd.apply(axis=1)
    dates_array = scored_df['date'].values
    tickers_array = scored_df['ticker'].values
    
    mask = [t in valid_map[d] for d, t in zip(dates_array, tickers_array)]

    filtered = scored_df[mask].copy()
    
    if progress_callback:
        progress_callback(100, 100, "", f"Completed PIT filter.")
        
    return filtered


def run_cross_sectional_backtest(
    tickers: list,
    themes: list,
    custom_formula: str = None,
    portfolio_size: float = 100,
    sizing_strategy: str = "Dynamic (1/Active)",
    liquidity_cap_type: str = "Max ADV %",
    liquidity_cap_value: float = 5.0,
    portfolio_sizing_type: str = "Absolute Count",
    max_position_weight: float = None,
    strategy_type: str = "Long/Short",
    execution_timing: str = "Current Day Close",
    start_year: int = 2020,
    end_year: int = 2025,
    invert_factor: bool = False,
    rebalance_freq: str = "M",
    initial_aum: float = 1000000,
    vol_target: float = 0.0,
    atr_sl_mult: float = 0.0,
    atr_tp_mult: float = 0.0,
    progress_callback=None,
    constituent_timeline: dict = None,
    benchmark_ticker: str = "IWM",
    quantiles: int = 5,
    enable_calendar: bool = True,
    universe_filter: str = None,
    max_factor_score: float = None,
    slippage_bps: float = 0.0,
    sl_slippage_bps: float = 0.0,
) -> str:
    """
    Full cross-sectional factor backtest:
      1. Fetch daily data for entire universe (concurrent + cached)
      2. Compute factor scores per ticker per day
      3. Apply point-in-time constituent filter (if timeline provided)
      4. Each day: Rank full universe explicitly, allocating exactly Portfolio Size / 2 to Long and Short legs.
      5. Compute portfolio analytics

    Parameters
    ----------
    constituent_timeline : dict, optional
        {quarter_date: [ticker_list]} for survivorship-bias-free filtering.
        If provided, each day's cross-section is restricted to tickers that
        were actual R2K members at that time.

    Returns JSON with metrics and Plotly chart JSON.
    """
    try:
        universe = fetch_universe_data(
            tickers, start_year=start_year, end_year=end_year,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(0, 100, "", "Computing multi-factor composite scores...")

        scored = _compute_factor_scores(universe, themes, custom_formula=custom_formula, progress_callback=progress_callback, universe_filter=universe_filter, execution_timing=execution_timing)
        
        # Mathematically strip invalid algorithmic states (Inf/NaN) generated by unconstrained symbolic nodes before sorting!
        scored["factor_score"] = scored["factor_score"].replace([np.inf, -np.inf], np.nan)
        
        if max_factor_score is not None:
            scored.loc[scored["factor_score"] > max_factor_score, "factor_score"] = np.nan
            
        # We NO LONGER drop rows with missing `fwd_return` because the strictly absolute most recent day inherently has no forward boundary resolved yet (it's what we execute right now!)
        # scored = scored.dropna(subset=["fwd_return"]).copy()
        
        if invert_factor:
            scored["factor_score"] *= -1
            scored["factor_rank"] = 1.0 - scored["factor_rank"]
            
        if atr_sl_mult > 0 or atr_tp_mult > 0:
            if progress_callback: progress_callback(0, 100, "", "Computing Intraday ATR volatility limits...")
            scored["prev_close"] = scored.groupby("ticker")["close"].shift(1)
            tr1 = scored["high"] - scored["low"]
            tr2 = (scored["high"] - scored["prev_close"]).abs()
            tr3 = (scored["low"] - scored["prev_close"]).abs()
            scored["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))
            scored["atr_14"] = scored.groupby("ticker")["tr"].rolling(14, min_periods=7).mean().reset_index(level=0, drop=True)
            scored["atr_pct"] = scored["atr_14"] / scored["close"]

        # Remove padded year dates that were only for factor computation, aligning to exact analysis grid
        scored = scored[(scored["date"] >= f"{start_year}-01-01") & (scored["date"] <= f"{end_year}-12-31")]

        # Apply point-in-time filtering if timeline is available
        if constituent_timeline:
            if progress_callback:
                progress_callback(0, 100, "", "Applying point-in-time constituent filter...")
            pre_count = scored["ticker"].nunique()
            scored = _pit_filter(scored, constituent_timeline, progress_callback=progress_callback)
            post_count = scored["ticker"].nunique()
            if progress_callback:
                progress_callback(100, 100, "", f"PIT filter: {pre_count} → {post_count} tickers (survivorship bias-free)")

        n_unique = scored["ticker"].nunique()
        
        if portfolio_sizing_type == "Percentage":
            actual_size = int(n_unique * (portfolio_size / 100.0))
            portfolio_size_bound = max(2, (actual_size // 2) * 2)
        else:
            portfolio_size_bound = int(portfolio_size)
            
        if n_unique < portfolio_size_bound:
            portfolio_size_bound = max(2, (n_unique // 2) * 2)

        if progress_callback:
            progress_callback(0, 100, "", "Executing vectorized backtest constraints...")

        # ── Portfolio construction ───────────────────────────
        if strategy_type in ["Long/Short", "Short/Short"]:
            leg_size = max(1, portfolio_size_bound // 2)
        else:
            leg_size = max(1, portfolio_size_bound) # Don't divide the portfolio size if strictly Long-Only or Short-Only

        # Mathematically absolute determinism: Never use stochastic numpy arrays that decouple structurally over varied temporal matrix bounds!
        # Always tie-break mathematically tied factor outputs using ascending alphabetical Ticker hashes!
        scored = scored.sort_values(["date", "factor_score", "ticker"], ascending=[True, False, True])
        
        scored["position"] = 0.0
        scored["long_rank"] = np.nan
        scored["short_rank"] = np.nan
        
        valid_mask = ~scored["factor_score"].isna()
        valid_scored = scored[valid_mask]
        
        # High score -> rank 1, ..., rank N (only among valid items)
        long_rank = valid_scored.groupby("date").cumcount() + 1
        
        # Low score -> rank 1, ..., rank N (inverted from the descending sort)
        valid_group_sizes = valid_scored.groupby("date")["ticker"].transform("size")
        short_rank = valid_group_sizes - long_rank + 1

        scored.loc[valid_mask, "long_rank"] = long_rank.values
        scored.loc[valid_mask, "short_rank"] = short_rank.values

        if strategy_type in ["Long/Short", "Long Only"]:
            scored.loc[(scored["long_rank"] <= leg_size) & valid_mask, "position"] = 1.0
        if strategy_type == "Short/Short":
            scored.loc[(scored["long_rank"] <= leg_size) & valid_mask, "position"] = -1.0
            
        if strategy_type in ["Long/Short", "Short Only", "Short/Short"]:
            scored.loc[(scored["short_rank"] <= leg_size) & valid_mask, "position"] = -1.0
            
        # VERY IMPORTANT: Return matrix back to chronological Ticker sequential structures for valid temporal FFills below!
        scored = scored.sort_values(["ticker", "date"])

        if rebalance_freq != "D":
            if rebalance_freq == "W":
                period_dt = scored['date'].dt.to_period("W")
            elif rebalance_freq == "M":
                period_dt = scored['date'].dt.to_period("M")
            elif rebalance_freq == "Q":
                period_dt = scored['date'].dt.to_period("Q")
            elif rebalance_freq == "Y":
                period_dt = scored['date'].dt.to_period("Y")
            else:
                period_dt = scored['date'].dt.to_period("M")
                
            last_days = scored.groupby(period_dt)["date"].max()
            is_rebalance = scored["date"].isin(last_days)
            
            # Lock position sizes dynamically based on boundary triggers
            scored["position"] = scored["position"].where(is_rebalance)
            scored["position"] = scored.groupby("ticker")["position"].ffill().fillna(0.0)
            

                            
        total_sl_hits = 0
        total_tp_hits = 0
        
        # --- UNIVERSAL INTRA-DAY STOP LOSS / TAKE PROFIT RISK ENGINE ---
        if atr_sl_mult > 0 or atr_tp_mult > 0:
            if progress_callback:
                progress_callback(50, 100, "", "Structurally binding Intraday Risk Management caps...")
                
            # Map T+1 Execution boundaries organically against physical simulation definitions
            next_open = scored.groupby("ticker")["open"].shift(-1)
            next_high = scored.groupby("ticker")["high"].shift(-1)
            next_low = scored.groupby("ticker")["low"].shift(-1)
            
            if execution_timing == "Intraday (Open to Close)" or execution_timing == "Next Day Open":
                # Execution evaluates naturally anchored to T+1 opening cross
                exec_entry = next_open
            else:
                exec_entry = scored["close"]
                
            # Intraday Maximum Adverse/Favorable Excursion
            max_adv_long = (next_low / exec_entry) - 1.0
            max_adv_short = (next_high / exec_entry) - 1.0
            
            max_fav_long = (next_high / exec_entry) - 1.0
            max_fav_short = (next_low / exec_entry) - 1.0
            
            sl_limit = atr_sl_mult * scored["atr_pct"] if atr_sl_mult > 0 else pd.Series(999.0, index=scored.index)
            tp_limit = atr_tp_mult * scored["atr_pct"] if atr_tp_mult > 0 else pd.Series(999.0, index=scored.index)
            
            # Flag positions piercing boundary states natively
            long_sl_hit = (scored["position"] > 0) & (max_adv_long <= -sl_limit)
            short_sl_hit = (scored["position"] < 0) & (max_adv_short >= sl_limit)
            long_tp_hit = (scored["position"] > 0) & (max_fav_long >= tp_limit)
            short_tp_hit = (scored["position"] < 0) & (max_fav_short <= -tp_limit)
            
            # --- NEW HIGH-FIDELITY RESOLUTION LOGIC ---
            # If both SL and TP are hit on the same day, we dynamically query minute-level data
            # to determine chronologically which was hit first.
            long_conflict = long_sl_hit & long_tp_hit
            short_conflict = short_sl_hit & short_tp_hit
            
            conflict_mask = long_conflict | short_conflict
            
            print(f"DEBUG: long_conflict={long_conflict.sum()}, short_conflict={short_conflict.sum()}")
            
            if conflict_mask.any():
                if progress_callback:
                    progress_callback(50, 100, "", f"Resolving {conflict_mask.sum()} intraday SL/TP conflicts via Minute API...")
                
                try:
                    conflict_client = RESTClient(api_key=API_KEY)
                    # Iterate through the exact rows that have conflicts
                    
                    ticker_dates = df.groupby('ticker')['date'].apply(lambda x: np.sort(x.values)).to_dict()
                    
                    for idx, row in scored[conflict_mask].iterrows():
                        ticker = row["ticker"]
                        
                        dates_for_ticker = ticker_dates.get(ticker, [])
                        future_dates = [d for d in dates_for_ticker if d > pd.to_datetime(row["date"])]
                        
                        if len(future_dates) == 0:
                            continue
                            
                        exec_date = pd.to_datetime(future_dates[0])
                        date_str = exec_date.strftime("%Y-%m-%d")
                        
                        is_long = row["position"] > 0
                        sl_thresh = -row["sl_limit"] if is_long else row["sl_limit"]
                        tp_thresh = row["tp_limit"] if is_long else -row["tp_limit"]
                        exec_entry_val = exec_entry.loc[idx]
                        
                        # Fetch 1-minute data for this specific day
                        try:
                            resp = conflict_client.list_aggs(
                                ticker=ticker.upper(), multiplier=1, timespan="minute",
                                from_=date_str, to=date_str,
                                sort="asc", limit=1000
                            )
                            # Convert response objects to a list to check if data exists
                            min_bars = list(resp)
                            
                            sl_first = True # Conservative default
                            
                            for bar in min_bars:
                                bar_high_ret = (bar.high / exec_entry_val) - 1.0
                                bar_low_ret = (bar.low / exec_entry_val) - 1.0
                                
                                # Check if SL or TP is hit in this minute bar
                                sl_triggered = (bar_low_ret <= sl_thresh) if is_long else (bar_high_ret >= sl_thresh)
                                tp_triggered = (bar_high_ret >= tp_thresh) if is_long else (bar_low_ret <= tp_thresh)
                                
                                if sl_triggered and not tp_triggered:
                                    sl_first = True
                                    break
                                elif tp_triggered and not sl_triggered:
                                    sl_first = False
                                    break
                                elif sl_triggered and tp_triggered:
                                    # If both hit in the exact same minute bar, default conservative (SL first)
                                    sl_first = True
                                    break
                                    
                            print(f"DEBUG RESOLVED: ticker={ticker}, date={date_str}, sl_first={sl_first}, bars={len(min_bars)}")
                            if sl_first:
                                if is_long:
                                    long_tp_hit.loc[idx] = False
                                else:
                                    short_tp_hit.loc[idx] = False
                            else:
                                if is_long:
                                    long_sl_hit.loc[idx] = False
                                else:
                                    short_sl_hit.loc[idx] = False
                                    
                        except Exception as e:
                            print(f"DEBUG EXCEPTION: ticker={ticker}, error={e}")
                            # If API fails, default to conservative (SL hit first)
                            if is_long:
                                long_tp_hit.loc[idx] = False
                            else:
                                short_tp_hit.loc[idx] = False
                except Exception as e:
                    # Catch RESTClient init errors
                    long_tp_hit = long_tp_hit & ~long_conflict
                    short_tp_hit = short_tp_hit & ~short_conflict
            # -----------------------------------------
            
            # Assign returns according to the resolved execution boundaries
            if atr_sl_mult > 0:
                # Add columns if they don't exist in the series temporarily
                
                # Apply isolated SL penalty on top of the turnover slippage
                # Since standard slippage triggers automatically when position drops to 0, 
                # we just deduct the DIFFERENCE so the total equals exactly sl_slippage_bps.
                # If sl_slippage_bps is smaller than standard slippage, no extra penalty is applied.
                sl_penalty = max(0.0, (sl_slippage_bps - slippage_bps) / 10000.0)
                
                scored.loc[long_sl_hit, "fwd_return"] = -(sl_limit.loc[long_sl_hit]) - sl_penalty
                scored.loc[short_sl_hit, "fwd_return"] = sl_limit.loc[short_sl_hit] + sl_penalty
            if atr_tp_mult > 0:
                scored.loc[long_tp_hit, "fwd_return"] = tp_limit.loc[long_tp_hit]
                scored.loc[short_tp_hit, "fwd_return"] = -(tp_limit.loc[short_tp_hit])
                
            # If tracking multi-day sequential holdings, execute physical halt so position drops flat on T+2
            if rebalance_freq != "D":
                scored["trade_id"] = is_rebalance.astype(int).groupby(scored["ticker"]).cumsum()
                exit_triggered = long_sl_hit | short_sl_hit | long_tp_hit | short_tp_hit
                already_exited = exit_triggered.groupby([scored["ticker"], scored["trade_id"]]).cumsum() > 0
                was_exited_yesterday = already_exited.groupby(scored["ticker"]).shift(1).fillna(False)
                scored["position"] = scored["position"].where(~was_exited_yesterday, 0.0)

            if atr_sl_mult > 0:
                total_sl_hits = int((long_sl_hit | short_sl_hit).sum())
            if atr_tp_mult > 0:
                total_tp_hits = int((long_tp_hit | short_tp_hit).sum())


        if progress_callback:
            progress_callback(50, 100, "", "Calculating historical portfolio compound returns...")

        # Extract master timeline ensuring cash-heavy defensive periods don't mathematically delete calendar dates!
        master_dates = pd.Series(0.0, index=pd.to_datetime(np.sort(scored["date"].unique())))
        
        # Define trade groups across contiguous positional strings purely for trade-level analytics processing independent of calendar loops
        scored["trade_group"] = (scored["position"] != scored.groupby("ticker")["position"].shift(1)).groupby(scored["ticker"]).cumsum()

        # Assign mathematically robust position weights based on sizing strategy
        scored["weight"] = 0.0
        active_mask = scored["position"] != 0
        
        if sizing_strategy == "Fixed Fractional (1/N)":
            scored.loc[active_mask, "weight"] = 1.0 / portfolio_size_bound
        elif sizing_strategy == "Inverse Volatility (ATR)":
            inv_atr = 1.0 / scored.loc[active_mask, "atr_pct"].replace(0, np.nan)
            inv_atr_sum = inv_atr.groupby(scored.loc[active_mask, "date"]).transform("sum")
            scored.loc[active_mask, "weight"] = inv_atr / inv_atr_sum
        else: # "Dynamic (1/Active)"
            active_counts = scored.loc[active_mask].groupby("date")["ticker"].transform("count")
            scored.loc[active_mask, "weight"] = 1.0 / active_counts
            
        if max_position_weight is not None and max_position_weight < 1.0:
            scored.loc[active_mask, "weight"] = scored.loc[active_mask, "weight"].clip(upper=max_position_weight)

        # Daily portfolio return = sum of weighted positioned returns
        portfolio = scored[active_mask].copy()
        if not portfolio.empty:
            # --- Realistic Liquidity Cap via Iterative Equity Compounding ---
            if liquidity_cap_type != "None":
                portfolio = portfolio.sort_values("date")
                final_weights = []
                current_equity = initial_aum
                
                for date, group in portfolio.groupby("date", sort=False):
                    intended_dollars = current_equity * group["weight"]
                    
                    if liquidity_cap_type == "Max ADV %":
                        adv_dollars = group["avg_30d_volume"] * group["close"]
                        max_dollars = adv_dollars * (liquidity_cap_value / 100.0)
                    else: # Static $ Ceiling
                        max_dollars = float(liquidity_cap_value)
                        
                    capped_dollars = np.minimum(intended_dollars, max_dollars)
                    actual_weights = capped_dollars / max(current_equity, 1.0)
                    
                    final_weights.extend(actual_weights.tolist())
                    
                    # Update equity for the next day's sizing using gross returns
                    daily_ret = np.sum(group["position"] * group["fwd_return"] * actual_weights)
                    current_equity = current_equity * (1 + daily_ret)
                
                portfolio["weight"] = final_weights
                scored.loc[active_mask, "weight"] = portfolio["weight"]

            portfolio["port_contrib"] = portfolio["position"] * portfolio["fwd_return"]
            portfolio["weighted_contrib"] = portfolio["port_contrib"] * portfolio["weight"]
            daily_port_ret = portfolio.groupby("date")["weighted_contrib"].sum()
            
            if slippage_bps > 0:
                slippage_pct = slippage_bps / 10000.0
                scored["signed_weight"] = scored["position"] * scored["weight"]
                turnover = (scored["signed_weight"] - scored.groupby("ticker")["signed_weight"].shift(1).fillna(0.0)).abs()
                daily_slippage_sum = (turnover * slippage_pct).groupby(scored["date"]).sum()
                daily_port_ret = daily_port_ret - daily_slippage_sum

            daily_port_ret = daily_port_ret.reindex(master_dates.index).fillna(0.0)
        else:
            daily_port_ret = master_dates.copy()
        
        # Volatility Targeting (Dynamic Risk Parity Leverage)
        if vol_target > 0.0:
            est_vol = daily_port_ret.rolling(window=100, min_periods=20).std() * np.sqrt(252)
            est_vol = est_vol.replace(0, np.nan).fillna(vol_target) 
            k_leverage = vol_target / est_vol
            k_leverage = k_leverage.clip(upper=3.0) 
            daily_port_ret = daily_port_ret * k_leverage
            
        daily_port_ret.name = "port_return"

        # Benchmark: Actual ETF limits
        min_date = scored["date"].min().strftime("%Y-%m-%d")
        max_date = scored["date"].max().strftime("%Y-%m-%d")
        proxy_df = _fetch_single_ticker(benchmark_ticker, min_date, max_date)
        
        if not proxy_df.empty:
            proxy_df = proxy_df.set_index("date").sort_index()
            daily_bench_ret = proxy_df["close"].pct_change().dropna()
            daily_bench_ret = daily_bench_ret.reindex(master_dates.index).fillna(0)
        else:
            daily_bench_ret = scored.groupby("date")["fwd_return"].mean().reindex(master_dates.index).fillna(0)
        daily_bench_ret.name = "bench_return"

        # Long-only leg (Extracted from the pre-filtered active portfolio slice instead of 2.5M rows)
        long_only = portfolio[portfolio["position"] == 1.0].groupby("date")["fwd_return"].mean().reindex(master_dates.index).fillna(0.0)

        # Short-only leg
        short_only = portfolio[portfolio["position"] == -1.0].groupby("date")["fwd_return"].mean().reindex(master_dates.index).fillna(0.0)
        
        if vol_target > 0.0:
            long_only = long_only * k_leverage
            short_only = short_only * k_leverage

        long_only.name = "long_return"
        short_only.name = "short_return"

        # Explicitly lock all dataframes tightly into the Master Calendar ensuring no Pandas dict-merge shape collapses!
        combined = pd.DataFrame(index=master_dates.index)
        combined["port_return"] = daily_port_ret
        combined["bench_return"] = daily_bench_ret
        combined["long_return"] = long_only if not long_only.empty else 0.0
        combined["short_return"] = short_only if not short_only.empty else 0.0
        combined = combined.fillna(0.0)

        if progress_callback:
            progress_callback(80, 100, "", "Aggregating performance metrics & plotting...")

        if len(combined) < 50:
            return json.dumps({"error": "Too few trading days after filtering.", "success": False})

        # ── Metrics ─────────────────────────────────────────
        cum_port = (1 + combined["port_return"]).cumprod() * initial_aum
        cum_bench = (1 + combined["bench_return"]).cumprod() * initial_aum
        cum_long = (1 + combined["long_return"]).cumprod() * initial_aum
        cum_short = (1 + combined["short_return"]).cumprod() * initial_aum

        total_port = (cum_port.iloc[-1] / initial_aum) - 1
        total_bench = (cum_bench.iloc[-1] / initial_aum) - 1
        n_days = len(combined)

        ann_port = (1 + total_port) ** (252 / n_days) - 1
        ann_bench = (1 + total_bench) ** (252 / n_days) - 1
        ann_vol = combined["port_return"].std() * np.sqrt(252)
        ann_bench_vol = combined["bench_return"].std() * np.sqrt(252)
        
        sharpe = ann_port / ann_vol if ann_vol > 0 else 0
        bench_sharpe = ann_bench / ann_bench_vol if ann_bench_vol > 0 else 0
        alpha = ann_port - ann_bench
        
        # Portfolio Beta to Benchmark
        cov_matrix = np.cov(combined["port_return"], combined["bench_return"])
        port_beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
        
        # Absolute Total Return in USD
        total_ret_usd = cum_port.iloc[-1] - initial_aum

        # Max drawdown
        rolling_max = cum_port.cummax()
        drawdown = cum_port / rolling_max - 1
        max_dd = drawdown.min()
        
        rolling_max_bench = cum_bench.cummax()
        bench_drawdown = cum_bench / rolling_max_bench - 1
        bench_max_dd = bench_drawdown.min()

        # Information Coefficient (Vectorized Pearson on native percentiles == Spearman)
        # Bypassing the catastrophic python pandas lambda apply loop (15-30s) by using C-level cov/corr bincount matrices (2ms)
        # We also mathematically bypass pd.astype('category') string hashing which introduces immense global scan overhead
        unique_dates, w_m = np.unique(scored["date"], return_inverse=True)
        x_m = scored["factor_rank"].values
        y_m = scored["fwd_return"].values
        
        counts = np.bincount(w_m)
        counts_safe = np.where(counts == 0, 1, counts)
        
        x_mean = np.bincount(w_m, weights=x_m) / counts_safe
        y_mean = np.bincount(w_m, weights=y_m) / counts_safe
        
        x_demeaned = x_m - x_mean[w_m]
        y_demeaned = y_m - y_mean[w_m]
        
        cov_xy = np.bincount(w_m, weights=x_demeaned * y_demeaned) / counts_safe
        var_x = np.bincount(w_m, weights=x_demeaned**2) / counts_safe
        var_y = np.bincount(w_m, weights=y_demeaned**2) / counts_safe
        
        std_xy = np.sqrt(var_x * var_y)
        std_xy_safe = np.where(std_xy < 1e-8, 1, std_xy)
        
        daily_ic = cov_xy / std_xy_safe
        ic_by_day = pd.Series(daily_ic, index=unique_dates)
        
        mean_ic = ic_by_day.mean()
        ic_ir = mean_ic / ic_by_day.std() if ic_by_day.std() > 0 else 0

        # Turnover estimate (daily rank changes)
        # Vectorized Numpy shift over the boundary
        boundary_mask = scored["ticker"] == scored["ticker"].shift(1)
        scored["prev_pos"] = np.where(boundary_mask, scored["position"].shift(1), 0.0)
        
        scored["trade_abs"] = (scored["position"] - scored["prev_pos"]).abs()
        
        # Mathematically replace slow groupby("date").sum() with native bincount over our pre-calced w_m array!
        daily_trades_arr = np.bincount(w_m, weights=scored["trade_abs"].values, minlength=len(unique_dates))
        daily_trades = pd.Series(daily_trades_arr, index=unique_dates)
        
        # Optimize global lambda scan into localized C array count
        portfolio["abs_pos"] = portfolio["position"].abs()
        port_w_m = np.searchsorted(unique_dates, portfolio["date"].values)
        daily_gross_arr = np.bincount(port_w_m, weights=portfolio["abs_pos"].values, minlength=len(unique_dates))
        daily_gross = pd.Series(daily_gross_arr, index=unique_dates)
        
        # Mean fraction of the portfolio rotated on any given day
        daily_turnover_fraction = (daily_trades / 2.0) / daily_gross.replace(0, np.nan)
        
        # Isolate days where trades physically occurred to decouple from manual configuration frequencies
        active_turnover = daily_turnover_fraction[daily_turnover_fraction > 1e-4]
        avg_turnover = active_turnover.mean() if not active_turnover.empty else 0.0

        # Regression
        valid_scored = scored.dropna(subset=["fwd_return"])
        sample = valid_scored.sample(n=min(10000, len(valid_scored)), random_state=42) if not valid_scored.empty else valid_scored
        
        # Prevent linregress crashing if formula produces identical/0-variance array
        if sample["factor_score"].nunique() > 1 and not sample.empty:
            slope, intercept, r_val, p_val, std_err = stats.linregress(
                sample["factor_score"], sample["fwd_return"]
            )
        else:
            slope, intercept, r_val, p_val, std_err = 0.0, 0.0, 0.0, 1.0, 0.0

        # ── Plots ───────────────────────────────────────────

        # 1. Equity Curve (L/S, Long-only, Short-only, Benchmark)
        fig_equity = go.Figure()

        fig_equity.add_trace(go.Scatter(
            x=cum_bench.index, y=cum_bench.values,
            mode="lines", name=f"Index Benchmark ({benchmark_ticker})",
            line=dict(color="#f39c12", width=2, dash="dashdot"),
        ))

        fig_equity.add_trace(go.Scatter(
            x=cum_port.index, y=cum_port.values,
            mode="lines", name=f"{strategy_type} Factor Portfolio",
            line=dict(color="#00d4aa", width=3),
        ))
        
        fig_equity.add_trace(go.Scatter(
            x=cum_long.index, y=cum_long.values,
            mode="lines", name="Long Leg (Top Quantile)",
            line=dict(color="#54a0ff", width=1.5, dash="dot"),
            visible="legendonly",
        ))
        fig_equity.add_trace(go.Scatter(
            x=cum_short.index, y=cum_short.values,
            mode="lines", name="Short Leg (Bottom Quantile)",
            line=dict(color="#ff6b6b", width=1.5, dash="dot"),
            visible="legendonly",
        ))
        
        formatted_themes = " + ".join(themes).replace("_", " ").title()
        
        fig_equity.update_layout(
            title=f"Cumulative Returns — {formatted_themes} (Click Legend Keys to Toggle Traces)",
            xaxis_title="Date", yaxis_title="Portfolio Value ($)",
            yaxis=dict(tickformat="$,.0f"),
            template="plotly_dark", height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)"),
        )

        # 2. Quantile Returns Bar
        # Instead of calling pd.qcut (which internally runs multi-second massive sorts and rank(method="first") ties over 2.5M matrices),
        # convert the native C cross-sectional percentages mathematically directly into categorical groupings.
        scored["quintile_num"] = np.ceil(scored["factor_rank"] * quantiles).clip(1, quantiles).astype(int)
        
        q_map = {i: f"Q{i}" for i in range(1, quantiles + 1)}
        q_map[1] = "Q1 (Low)"
        q_map[quantiles] = f"Q{quantiles} (High)"
        
        # Replace 0.5s pandas grouping with 2ms numpy mean constraint vectorization
        # Mask out NaNs and Infs so latest day fwd_return doesn't poison the bin sums
        # Force strict numeric arrays
        safe_fwd = pd.to_numeric(scored["fwd_return"], errors="coerce").values
        safe_rank = pd.to_numeric(scored["factor_rank"], errors="coerce").values
        
        valid_mask = np.isfinite(safe_fwd) & np.isfinite(safe_rank)
        
        # Calculate quantiles purely on the valid slice
        valid_ranks = safe_rank[valid_mask]
        q_num_valid = np.ceil(valid_ranks * quantiles).clip(1, quantiles).astype(int)
        fwd_valid = safe_fwd[valid_mask]
        
        q_counts = np.bincount(q_num_valid, minlength=quantiles+1)
        q_sums = np.bincount(q_num_valid, weights=fwd_valid, minlength=quantiles+1)
        q_means = q_sums / np.maximum(q_counts, 1)
        
        # q_num spans 1 to `quantiles` (idx 0 is empty)
        q_returns = pd.Series(q_means[1:], index=range(1, quantiles + 1)) * 252
        q_returns.index = q_returns.index.map(q_map)
        
        # Build dynamic color gradient natively
        from plotly.colors import sample_colorscale
        bar_colors = sample_colorscale("Turbo", [i / (quantiles - 1) for i in range(quantiles)]) if quantiles > 2 else ["#ff6b6b", "#00d4aa"]
        
        fig_qbar = go.Figure(data=[go.Bar(
            x=q_returns.index.astype(str), y=q_returns.values,
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in q_returns.values],
            textposition="outside",
        )])
        fig_qbar.update_layout(
            title="Annualized Return by Factor Quantile (Cross-Sectional)",
            xaxis_title="Factor Quantile", yaxis_title="Annualized Return",
            template="plotly_dark", height=380,
        )

        # 3. Rolling IC
        rolling_ic = ic_by_day.rolling(20).mean()
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Bar(
            x=ic_by_day.index, y=ic_by_day.values,
            name="Daily IC", marker_color="rgba(100,150,255,0.25)",
        ))
        fig_ic.add_trace(go.Scatter(
            x=rolling_ic.index, y=rolling_ic.values,
            mode="lines", name="20d Rolling IC",
            line=dict(color="#ffd700", width=2),
        ))
        fig_ic.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
        fig_ic.update_layout(
            title="Information Coefficient (Spearman Rank Correlation)",
            xaxis_title="Date", yaxis_title="IC",
            template="plotly_dark", height=350,
        )

        # 4. Drawdown chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", mode="lines", name="Drawdown",
            line=dict(color="#ff6b6b", width=1),
            fillcolor="rgba(255,107,107,0.3)",
        ))
        fig_dd.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date", yaxis_title="Drawdown",
            template="plotly_dark", height=300,
        )

        # 5. Yearly/Monthly/Daily PNL Bar Chart
        if "year" not in combined.columns:
            combined["year"] = combined.index.year
        if "month" not in combined.columns:
            combined["month"] = combined.index.to_period('M').astype(str)
            
        yearly_pnl = combined.groupby("year")["port_return"].apply(lambda x: (1 + x).prod() - 1)
        monthly_pnl = combined.groupby("month")["port_return"].apply(lambda x: (1 + x).prod() - 1)
        daily_ret = combined["port_return"]
        
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Bar(
            x=yearly_pnl.index.astype(str), y=yearly_pnl.values,
            marker_color=["#00d4aa" if v > 0 else "#ff6b6b" for v in yearly_pnl.values],
            text=[f"{v*100:+.1f}%" for v in yearly_pnl.values],
            textposition="auto",
        ))
        fig_yearly.add_hline(y=0, line_color="white", opacity=0.3)
        fig_yearly.update_layout(
            title="Yearly Portfolio Net Return",
            xaxis_title="Year", yaxis_title="Return",
            template="plotly_dark", height=320,
            yaxis=dict(tickformat=".1%"),
        )
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly_pnl.index.astype(str), y=monthly_pnl.values,
            marker_color=["#00d4aa" if v > 0 else "#ff6b6b" for v in monthly_pnl.values],
        ))
        fig_monthly.add_hline(y=0, line_color="white", opacity=0.3)
        fig_monthly.update_layout(
            title="Monthly Portfolio Net Return",
            xaxis_title="Month", yaxis_title="Return",
            template="plotly_dark", height=320,
            yaxis=dict(tickformat=".1%"),
        )
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily_ret.index.astype(str), y=daily_ret.values,
            marker_color=["#00d4aa" if v > 0 else "#ff6b6b" for v in daily_ret.values],
        ))
        fig_daily.add_hline(y=0, line_color="white", opacity=0.3)
        fig_daily.update_layout(
            title="Daily Portfolio Net Return",
            xaxis_title="Day", yaxis_title="Return",
            template="plotly_dark", height=320,
            yaxis=dict(tickformat=".1%"),
        )
        # Extract Live Triggers natively enforcing Cross-Sectional volume to prevent single-stock array trailing bounds
        date_counts = scored.groupby("date").size()
        robust_dates = date_counts[date_counts > 50].index
        latest_date = robust_dates.max() if len(robust_dates) > 0 else scored["date"].max()
        
        latest_cross_section = scored[scored["date"] == latest_date]
        current_longs = latest_cross_section[latest_cross_section["position"] > 0]["ticker"].tolist()
        
        raw_shorts = latest_cross_section[latest_cross_section["position"] < 0]["ticker"].tolist()
        
        # Inject HTB Check for UI
        try:
            from tradestation_api import TradeStationClient
            ts = TradeStationClient(is_live=False)
            current_shorts = []
            for ticker in raw_shorts:
                if ts.is_hard_to_borrow(ticker):
                    current_shorts.append(f"{ticker} (HTB)")
                else:
                    current_shorts.append(ticker)
        except Exception as e:
            current_shorts = raw_shorts

        strat_ret = daily_port_ret
        strat_ret.index = pd.to_datetime(strat_ret.index)

        calendar_html_out = ""
        if enable_calendar:
            if progress_callback:
                progress_callback(95, 100, "", "Generating HTML P&L Calendar logic...")
            # Pre-filter for active positions to minimize grouped iteration payload
            longs_df, shorts_df = {}, {}
            if not portfolio.empty:
                portfolio["cum_trade_ret"] = (1 + portfolio["port_contrib"].fillna(0.0)).groupby([portfolio["ticker"], portfolio["trade_group"]]).cumprod() - 1
                _active_str_dates = portfolio["date"].astype(str).values
                _active_tickers = portfolio["ticker"].values
                _active_pos = portfolio["position"].values
                _active_cum_ret = portfolio["cum_trade_ret"].values
            else:
                _active_str_dates = []
                _active_tickers = []
                _active_pos = []
                _active_cum_ret = []
            
            for d, t, p, r in zip(_active_str_dates, _active_tickers, _active_pos, _active_cum_ret):
                fmt_t = f"{t} ({r*100:+.1f}%)"
                if p > 0: longs_df.setdefault(d, []).append(fmt_t)
                elif p < 0: shorts_df.setdefault(d, []).append(fmt_t)
            
            # Map everything exactly to YYYY-MM-DD string keys to ensure hashing matches regardless of np.datetime vs pd.Timestamp vs str
            longs_str = {pd.to_datetime(k).strftime('%Y-%m-%d'): v for k, v in longs_df.items()}
            shorts_str = {pd.to_datetime(k).strftime('%Y-%m-%d'): v for k, v in shorts_df.items()}
            
            daily_holdings = {}
            for d in portfolio['date'].unique():
                dt_key = pd.to_datetime(d)
                str_key = dt_key.strftime('%Y-%m-%d')
                daily_holdings[dt_key] = {
                    "longs": longs_str.get(str_key, []),
                    "shorts": shorts_str.get(str_key, [])
                }
            
            # Pass the precise algorithmic trade dates natively bypassing artificial API gaps masquerading as turnover!
            true_trade_dates = [d.strftime('%Y-%m-%d') for d in active_turnover.index]
            calendar_html_out = generate_pnl_calendar_html(strat_ret, daily_holdings, true_trade_dates)

        latest_date_str = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, 'strftime') else str(latest_date)[:10]

        # Trade Analytics (Positional & Daily)
        if not portfolio.empty:
            # 1. Positional Metrics (Individual trades)
            pos_pnl = portfolio["weighted_contrib"]
            pos_win_rate = (pos_pnl > 0).mean()
            pos_avg_win = pos_pnl[pos_pnl > 0].mean() if (pos_pnl > 0).any() else 0.0
            pos_avg_loss = pos_pnl[pos_pnl < 0].mean() if (pos_pnl < 0).any() else 0.0
            pos_max_win = pos_pnl.max()
            pos_max_loss = pos_pnl.min()
            pos_expected_value = (pos_win_rate * pos_avg_win) + ((1 - pos_win_rate) * pos_avg_loss)
            pos_profit_factor = abs(pos_pnl[pos_pnl > 0].sum() / pos_pnl[pos_pnl < 0].sum()) if (pos_pnl < 0).any() and pos_pnl[pos_pnl < 0].sum() != 0 else 99.9
            
            # 2. Daily Portfolio Metrics (Aggregated by day)
            daily_pnl = portfolio.groupby("date")["weighted_contrib"].sum()
            daily_win_rate = (daily_pnl > 0).mean() if not daily_pnl.empty else 0.0
            daily_avg_win = daily_pnl[daily_pnl > 0].mean() if (daily_pnl > 0).any() else 0.0
            daily_avg_loss = daily_pnl[daily_pnl < 0].mean() if (daily_pnl < 0).any() else 0.0
            daily_max_win = daily_pnl.max() if not daily_pnl.empty else 0.0
            daily_max_loss = daily_pnl.min() if not daily_pnl.empty else 0.0
            daily_expected_value = (daily_win_rate * daily_avg_win) + ((1 - daily_win_rate) * daily_avg_loss)
            daily_profit_factor = abs(daily_pnl[daily_pnl > 0].sum() / daily_pnl[daily_pnl < 0].sum()) if (daily_pnl < 0).any() and daily_pnl[daily_pnl < 0].sum() != 0 else 99.9
            
            # 3. Streaks (Daily)
            is_win = daily_pnl > 0
            is_loss = daily_pnl < 0
            win_blocks = (~is_win).cumsum()
            win_streaks = is_win.groupby(win_blocks).sum()
            win_streaks = win_streaks[win_streaks > 0]
            loss_blocks = (~is_loss).cumsum()
            loss_streaks = is_loss.groupby(loss_blocks).sum()
            loss_streaks = loss_streaks[loss_streaks > 0]
            
            max_win_streak = int(win_streaks.max()) if not win_streaks.empty else 0
            avg_win_streak = float(win_streaks.mean()) if not win_streaks.empty else 0.0
            max_loss_streak = int(loss_streaks.max()) if not loss_streaks.empty else 0
            avg_loss_streak = float(loss_streaks.mean()) if not loss_streaks.empty else 0.0
            
        else:
            pos_win_rate = pos_avg_win = pos_avg_loss = pos_max_win = pos_max_loss = pos_expected_value = pos_profit_factor = 0.0
            daily_win_rate = daily_avg_win = daily_avg_loss = daily_max_win = daily_max_loss = daily_expected_value = daily_profit_factor = 0.0
            max_win_streak = avg_win_streak = max_loss_streak = avg_loss_streak = 0

        # Institutional Metrics
        downside_returns = daily_port_ret[daily_port_ret < 0]
        down_vol = np.sqrt(252) * downside_returns.std() if not downside_returns.empty else 1e-6
        sortino_ratio = ann_port / down_vol if down_vol != 0 else 0.0
        calmar_ratio = ann_port / abs(max_dd) if max_dd != 0 else 0.0

        metrics = {
            "latest_date": latest_date_str,
            "current_longs": current_longs,
            "current_shorts": current_shorts,
            "calendar_html": calendar_html_out,
            "sharpe_ratio": round(sharpe, 3),
            "ann_alpha": round(alpha, 4),
            "ann_port_return": round(ann_port, 4),
            "ann_bench_return": round(ann_bench, 4),
            "max_drawdown": round(max_dd, 4),
            "mean_ic": round(mean_ic, 4),
            "ic_ir": round(ic_ir, 3),
            "regression_beta": round(slope, 6),
            "p_value": round(p_val, 4),
            "r_squared": round(r_val ** 2, 4),
            "n_tickers": n_unique,
            "n_trading_days": n_days,
            "total_port_return": round(total_port, 4),
            "total_bench_return": round(total_bench, 4),
            "bench_sharpe": round(bench_sharpe, 4),
            "bench_max_dd": round(bench_max_dd, 4),
            "universe_size": int(n_unique),
            "ann_vol": round(ann_vol, 4),
            "avg_turnover": round(float(avg_turnover), 3),
            "port_beta": round(port_beta, 3),
            "total_ret_usd": round(total_ret_usd, 2),
            "sortino_ratio": round(float(sortino_ratio), 3),
            "calmar_ratio": round(float(calmar_ratio), 3),
            "pos_win_rate": round(float(pos_win_rate), 4),
            "pos_avg_win": round(float(pos_avg_win), 4),
            "pos_avg_loss": round(float(pos_avg_loss), 4),
            "pos_max_win": round(float(pos_max_win), 4),
            "pos_max_loss": round(float(pos_max_loss), 4),
            "pos_expected_value": round(float(pos_expected_value), 4),
            "pos_profit_factor": round(float(pos_profit_factor), 3),
            "daily_win_rate": round(float(daily_win_rate), 4),
            "daily_avg_win": round(float(daily_avg_win), 4),
            "daily_avg_loss": round(float(daily_avg_loss), 4),
            "daily_max_win": round(float(daily_max_win), 4),
            "daily_max_loss": round(float(daily_max_loss), 4),
            "daily_expected_value": round(float(daily_expected_value), 4),
            "daily_profit_factor": round(float(daily_profit_factor), 3),
            "max_win_streak": int(max_win_streak),
            "avg_win_streak": round(float(avg_win_streak), 2),
            "max_loss_streak": int(max_loss_streak),
            "avg_loss_streak": round(float(avg_loss_streak), 2),
            "quintile_returns": {str(k): round(v, 4) for k, v in q_returns.items()},
            "yearly_returns": {str(k): round(v, 4) for k, v in yearly_pnl.items()},
            "sl_hits": total_sl_hits,
            "tp_hits": total_tp_hits,
        }
        
        if not portfolio.empty:
            is_sl = long_sl_hit.loc[portfolio.index] | short_sl_hit.loc[portfolio.index]
            is_tp = long_tp_hit.loc[portfolio.index] | short_tp_hit.loc[portfolio.index]
            is_eod = ~(is_sl | is_tp)
            eod_pnl = portfolio.loc[is_eod, "weighted_contrib"]
            metrics["eod_pos_hits"] = int((eod_pnl > 0).sum())
            metrics["eod_neg_hits"] = int((eod_pnl <= 0).sum())
        else:
            metrics["eod_pos_hits"] = 0
            metrics["eod_neg_hits"] = 0

        return json.dumps({
            "success": True,
            "metrics": metrics,
            "plots": {
                "equity_json": fig_equity.to_json(),
                "yearly_json": fig_yearly.to_json(),
                "monthly_json": fig_monthly.to_json(),
                "daily_json": fig_daily.to_json(),
                "quintile_json": fig_qbar.to_json(),
                "ic_json": fig_ic.to_json(),
                "drawdown_json": fig_dd.to_json(),
            },
        })

    except Exception as e:
        import traceback
        return json.dumps({"error": traceback.format_exc(), "success": False})


# ═══════════════════════════════════════════════════════════════
# Tool Metadata (for LLM function calling)
# ═══════════════════════════════════════════════════════════════

tools_metadata = [
    {
        "type": "function",
        "function": {
            "name": "run_cross_sectional_backtest",
            "description": (
                "Runs a cross-sectional factor backtest across the full Russell 2000 "
                "universe. Ranks stocks daily by factor score, goes long top quintile / "
                "short bottom quintile, and measures portfolio performance."
            ),
            "parameters": {
                "type": "object",
                "required": ["theme"],
                "properties": {
                    "theme": {
                        "type": "string",
                        "description": "Factor theme: 'momentum', 'mean reversion', 'volatility', 'volume', or 'size'.",
                    },
                },
            },
        },
    }
]
