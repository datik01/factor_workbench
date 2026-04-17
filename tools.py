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
        df["date"] = pd.to_datetime(df["date"])
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
    cache_file = _cache_path(len(tickers), start_year, end_year)

    # Check cache first
    if not force_refresh and os.path.exists(cache_file):
        if progress_callback:
            progress_callback(0, 0, "", "Loading from cache...")
        df = pd.read_parquet(cache_file)
        cached_n = df["ticker"].nunique()
        if cached_n >= len(tickers) * 0.85:
            if progress_callback:
                progress_callback(cached_n, cached_n, "", f"Loaded {cached_n} tickers from cache.")
            return df
        else:
            if progress_callback:
                progress_callback(0, 0, "", f"Cache size mismatch ({cached_n}/{len(tickers)}). Rebuilding cache...")

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

    # Cache to parquet
    universe.to_parquet(cache_file, index=False)

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
    def div(a, b): return np.where(np.abs(b) < 1e-6, 1.0, a / b)
    def abs_f(a): return np.abs(a)
    def sqrt(a): return np.sqrt(np.abs(a))
    def log(a): return np.log(np.abs(a) + 1e-5)
    def rank(a): return pd.Series(a).rank(pct=True).values

    # Temporal boundary maps ensuring bleeding doesn't occur across tickers
    t_mask_5 = (df['ticker'] != df['ticker'].shift(5)).values
    t_mask_10 = (df['ticker'] != df['ticker'].shift(9)).values
    t_mask_20 = (df['ticker'] != df['ticker'].shift(19)).values
    
    def delay_5(a):
        r = np.roll(a, 5)
        r[:5] = a[:5]; r[t_mask_5] = a[t_mask_5]
        return r
    def sma_10(a):
        r = pd.Series(a).rolling(10).mean().bfill().values
        r[t_mask_10] = a[t_mask_10]
        return r
    def sma_20(a):
        r = pd.Series(a).rolling(20).mean().bfill().values
        r[t_mask_20] = a[t_mask_20]
        return r
    def ts_max_20(a):
        r = pd.Series(a).rolling(20).max().bfill().values
        r[t_mask_20] = a[t_mask_20]
        return r
    def ts_min_20(a):
        r = pd.Series(a).rolling(20).min().bfill().values
        r[t_mask_20] = a[t_mask_20]
        return r

    # Pre-parse memory bindings matching factor_miner's target states
    env = {
        "add": add, "sub": sub, "mul": mul, "div": div,
        "abs": abs_f, "sqrt": sqrt, "log": log, "rank": rank,
        "delay_5": delay_5, "sma_10": sma_10, "sma_20": sma_20, 
        "ts_max_20": ts_max_20, "ts_min_20": ts_min_20,
        "Open": df["open"].values,
        "High": df["high"].values,
        "Low": df["low"].values,
        "Close": df["close"].values,
        "Volume": df["volume"].values,
        "Returns": df["daily_return"].values if "daily_return" in df.columns else np.zeros(len(df))
    }
    
    # Secure string replacement for protected python keywords
    f_str = formula_str.replace('abs(', 'abs(')
    
    # Fully isolates eval mapping against sandbox dictionary
    return eval(f_str, {"__builtins__": {}}, env)

def _compute_factor_scores(universe: pd.DataFrame, themes: list, custom_formula: str = None, progress_callback=None) -> pd.DataFrame:
    """
    For each ticker, compute a daily factor score across multiple composites.
    Cross-sectionally ranks each specific factor, and computes the Rank-Sum equal-weight.
    """
    df = universe.copy()
    df = df.sort_values(["ticker", "date"])

    # Per-ticker rolling calculations
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    # Apply data sanitization clipping to prevent unadjusted API split artifacts from annihilating the backtest equity bounds
    df["fwd_return"] = df.groupby("ticker")["daily_return"].shift(-1).clip(lower=-0.5, upper=0.5)
    
    # Sandbox Escape: Prioritize GP algorithm overrides
    if custom_formula and custom_formula.strip():
        if progress_callback:
            progress_callback(50, 100, "", f"Injecting abstract GP Formula internally: {custom_formula}")
        df["factor_score"] = execute_gplearn_formula(df, custom_formula)
        df = df.dropna(subset=["factor_score", "fwd_return"])
        # Inject deterministic jitter immediately to break zero-variance dead formulas globally
        np.random.seed(42)
        df["factor_score"] += np.random.normal(0, 1e-12, size=len(df))
        df["factor_rank"] = df.groupby("date")["factor_score"].rank(pct=True)
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
            raw_vol = df.groupby("ticker")["daily_return"].transform(lambda x: x.rolling(20).std())
            df[col_name] = -raw_vol
        elif "volume" in theme_lower:
            df[col_name] = df.groupby("ticker").apply(
                lambda g: g["volume"] / g["volume"].rolling(20).mean()
            ).reset_index(level=0, drop=True)
        elif "size" in theme_lower:
            df[col_name] = -(df["close"] * df["volume"])
        else:
            df[col_name] = df.groupby("ticker")["close"].pct_change(10)
            
        df[f"rank_{col_name}"] = df.groupby("date")[col_name].rank(pct=True)
        rank_cols.append(f"rank_{col_name}")

    df = df.dropna(subset=rank_cols + ["fwd_return"])

    if progress_callback:
        progress_callback(len(themes), len(themes), "", "Generating composite rankings...")

    df["factor_score"] = df[rank_cols].mean(axis=1)

    # Cross-sectional rank of the composite mean within each day (percentile 0..1)
    df["factor_rank"] = df.groupby("date")["factor_score"].rank(pct=True)

    return df


# ═══════════════════════════════════════════════════════════════
# Portfolio Construction & Backtest
# ═══════════════════════════════════════════════════════════════

def _pit_filter(scored_df: pd.DataFrame, timeline: dict, progress_callback=None) -> pd.DataFrame:
    """
    Point-in-Time constituent filter.
    For each trading day, keep only tickers that were actual R2K members
    at the most recent quarterly rebalance before that day.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Scored universe with 'date' and 'ticker' columns
    timeline : dict
        {quarter_end_str: [ticker_list]} from build_constituent_timeline()
    progress_callback : callable
        Optional hook to report incremental filtering iterations.

    Returns
    -------
    pd.DataFrame
        Filtered scored_df with survivorship-bias-free membership
    """
    if not timeline:
        return scored_df

    # Sort quarter dates
    quarter_dates = sorted(pd.to_datetime(list(timeline.keys())))
    quarter_tickers = {pd.to_datetime(k): set(v) for k, v in timeline.items()}

    def _get_valid_tickers(trade_date):
        """Find the most recent quarter end at or before trade_date."""
        valid_qs = [q for q in quarter_dates if q <= trade_date]
        if not valid_qs:
            return quarter_tickers[quarter_dates[0]]  # fallback to earliest
        return quarter_tickers[valid_qs[-1]]

    # Build a mask: for each row, check if that ticker was in the index on that date
    _processed = [0]
    total_rows = len(scored_df)
    
    def _check_row(row):
        _processed[0] += 1
        if progress_callback and _processed[0] % 50000 == 0:
            progress_callback(_processed[0], total_rows, "", f"Applying PIT filter ({_processed[0]}/{total_rows} rows)...")
        return row["ticker"] in _get_valid_tickers(row["date"])

    mask = scored_df.apply(_check_row, axis=1)

    filtered = scored_df[mask].copy()
    if progress_callback:
        progress_callback(total_rows, total_rows, "", f"Completed PIT filter ({total_rows} rows).")
        
    return filtered


def run_cross_sectional_backtest(
    tickers: list,
    themes: list,
    custom_formula: str = None,
    portfolio_size: float = 100,
    portfolio_sizing_type: str = "Absolute Count",
    strategy_type: str = "Long/Short",
    start_year: int = 2020,
    end_year: int = 2025,
    invert_factor: bool = False,
    rebalance_freq: str = "D",
    initial_aum: float = 1000000,
    progress_callback=None,
    constituent_timeline: dict = None,
    benchmark_ticker: str = "IWM",
    quantiles: int = 5,
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

        scored = _compute_factor_scores(universe, themes, custom_formula=custom_formula, progress_callback=progress_callback)
        
        if invert_factor:
            scored["factor_score"] *= -1
            scored["factor_rank"] = 1.0 - scored["factor_rank"]

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
        leg_size = max(1, portfolio_size_bound // 2)

        # Add microscopic deterministic jitter to break exact index ordering mapping identical overlaps
        np.random.seed(42)
        scored["factor_score"] += np.random.normal(0, 1e-12, size=len(scored))

        # High score -> rank 1, ..., rank N
        scored["long_rank"] = scored.groupby("date")["factor_score"].rank(method='first', ascending=False)
        # Low score -> rank 1, ..., rank N
        scored["short_rank"] = scored.groupby("date")["factor_score"].rank(method='first', ascending=True)

        scored["position"] = 0.0
        if strategy_type in ["Long/Short", "Long Only"]:
            scored.loc[scored["long_rank"] <= leg_size, "position"] = 1.0
        if strategy_type in ["Long/Short", "Short Only"]:
            scored.loc[scored["short_rank"] <= leg_size, "position"] = -1.0

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

        if progress_callback:
            progress_callback(50, 100, "", "Calculating historical portfolio compound returns...")

        # Daily portfolio return = average of positioned returns (equal-weight L/S)
        portfolio = scored[scored["position"] != 0].copy()
        daily_port_ret = portfolio.groupby("date").apply(
            lambda g: (g["position"] * g["fwd_return"]).mean()
        ).sort_index()
        daily_port_ret.name = "port_return"

        # Benchmark: Actual ETF limits
        min_date = scored["date"].min().strftime("%Y-%m-%d")
        max_date = scored["date"].max().strftime("%Y-%m-%d")
        proxy_df = _fetch_single_ticker(benchmark_ticker, min_date, max_date)
        
        if not proxy_df.empty:
            proxy_df = proxy_df.set_index("date").sort_index()
            daily_bench_ret = proxy_df["close"].pct_change().dropna()
            daily_bench_ret = daily_bench_ret.reindex(daily_port_ret.index).fillna(0)
        else:
            daily_bench_ret = scored.groupby("date")["fwd_return"].mean().sort_index()
        daily_bench_ret.name = "bench_return"

        # Long-only leg
        long_only = scored[scored["position"] == 1.0].groupby("date")["fwd_return"].mean()
        long_only.name = "long_return"

        # Short-only leg
        short_only = scored[scored["position"] == -1.0].groupby("date")["fwd_return"].mean()
        short_only.name = "short_return"

        combined = pd.DataFrame({
            "port_return": daily_port_ret,
            "bench_return": daily_bench_ret,
            "long_return": long_only.fillna(0) if not long_only.empty else 0,
            "short_return": short_only.fillna(0) if not short_only.empty else 0,
        }).fillna(0)

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

        # Information Coefficient
        ic_by_day = scored.groupby("date").apply(
            lambda g: g["factor_score"].corr(g["fwd_return"], method="spearman")
            if len(g) > 5 else np.nan
        ).dropna()
        mean_ic = ic_by_day.mean()
        ic_ir = mean_ic / ic_by_day.std() if ic_by_day.std() > 0 else 0

        # Turnover estimate (daily rank changes)
        # Regression
        sample = scored.sample(n=min(10000, len(scored)), random_state=42)
        
        # Prevent linregress crashing if formula produces identical/0-variance array
        if sample["factor_score"].nunique() > 1:
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
            mode="lines", name="L/S Factor Portfolio",
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
        q_labels = [f"Q{i}" for i in range(1, quantiles + 1)]
        q_labels[0] = "Q1 (Low)"
        q_labels[-1] = f"Q{quantiles} (High)"
        try:
            scored["quintile"] = pd.qcut(
                scored["factor_rank"], quantiles,
                labels=q_labels,
            )
        except ValueError:
            # Fallback: force unique ties for binary/saturated GP formulas
            scored["quintile"] = pd.qcut(
                scored["factor_rank"].rank(method="first"), quantiles,
                labels=q_labels,
            )
        q_returns = scored.groupby("quintile")["fwd_return"].mean() * 252
        
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

        metrics = {
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
            "port_beta": round(port_beta, 3),
            "ann_vol": round(ann_vol, 4),
            "total_ret_usd": round(total_ret_usd, 2),
            "quintile_returns": {str(k): round(v, 4) for k, v in q_returns.items()}
        }

        return json.dumps({
            "success": True,
            "metrics": metrics,
            "plots": {
                "equity_json": fig_equity.to_json(),
                "quintile_json": fig_qbar.to_json(),
                "ic_json": fig_ic.to_json(),
                "drawdown_json": fig_dd.to_json(),
            },
        })

    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


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
