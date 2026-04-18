import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from scipy.stats import pearsonr
import warnings
from scipy.stats import ConstantInputWarning, NearConstantInputWarning

warnings.filterwarnings('ignore', category=ConstantInputWarning)
warnings.filterwarnings('ignore', category=NearConstantInputWarning)
warnings.filterwarnings('ignore', message='.*constant.*')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*semlock.*')
warnings.filterwarnings('ignore', category=UserWarning)
# ═══════════════════════════════════════════════════════════════
# Fitness Metrics & Primitives (Genetic Programming POC)
# ═══════════════════════════════════════════════════════════════

ENFORCE_MONOTONICITY = False

def _check_monotonicity_penalty(y_m, pred_m, w_m):
    """
    C-level vectorized Quintile binning to ensure [Q1_ret, Q2_ret, Q3_ret, Q4_ret, Q5_ret] is monotonically increasing.
    Returns 1.0 if highly monotonic (Spearman > 0.8), else 0.0 (death penalty).
    """
    if not ENFORCE_MONOTONICITY:
        return 1.0
        
    try:
        # Cross-sectionally normalize pred_m using w_m (Date Groups) natively in C
        counts = np.bincount(w_m)
        counts_safe = np.where(counts == 0, 1, counts)
        
        pred_means = np.bincount(w_m, weights=pred_m) / counts_safe
        pred_sq_means = np.bincount(w_m, weights=pred_m**2) / counts_safe
        pred_vars = pred_sq_means - pred_means**2
        pred_stds = np.sqrt(np.maximum(pred_vars, 1e-8))
        
        # Daily Cross-Sectional Z-Score!
        pred_z = (pred_m - pred_means[w_m]) / pred_stds[w_m]
        
        # Reject mathematically stagnant matrices (constants/0-variance)
        if np.var(pred_z) < 1e-6: return 0.0
        
        # 1-pass extraction of 4 decile bounds
        boundaries = np.quantile(pred_z, [0.2, 0.4, 0.6, 0.8])
        # Reject over-saturated trees that collapse bins
        if len(np.unique(boundaries)) < 4: return 0.0
            
        bins = np.digitize(pred_z, boundaries)
        counts = np.bincount(bins)
        if np.any(counts == 0): return 0.0
        
        means = np.bincount(bins, weights=y_m) / counts
        
        from scipy.stats import spearmanr
        corr, _ = spearmanr(means, [0, 1, 2, 3, 4])
        
        if np.isnan(corr) or corr < 0.8 or means[0] >= means[-1]:
            return 0.0
            
        return 1.0
    except Exception:
        return 0.0

def _ic_metric(y, y_pred, w):
    """
    Evaluates fitness based on Cross-Sectional Information Coefficient (IC).
    We mathematically isolate Cross-Sectional Alpha by de-meaning variables globally per date,
    entirely bypassing slow Pandas grouped `.rank()` or `.apply()` overheads.
    w is an integer array mapping identically to df['date'].
    """
    mask = ~np.isnan(y_pred) & ~np.isnan(y) & ~np.isinf(y_pred)
    if not np.any(mask) or len(y[mask]) < 2:
        return 0.0
        
    y_m = y[mask]
    pred_m = y_pred[mask]
    w_m = w[mask].astype(int)
    
    # 80 Microsecond Cross-Sectional De-meaning via C-level Numpy bin counting!
    counts = np.bincount(w_m)
    # Avoid division by zero
    counts_safe = np.where(counts == 0, 1, counts)
    
    # De-mean predictions by grouping
    pred_means = np.bincount(w_m, weights=pred_m) / counts_safe
    pred_demeaned = pred_m - pred_means[w_m]
    
    # De-mean forward returns by grouping
    y_means = np.bincount(w_m, weights=y_m) / counts_safe
    y_demeaned = y_m - y_means[w_m]
    
    # Global Pearson on purely cross-sectionally bounded arrays inherently extracts Cross-Sectional Alpha Correlation!
    r, _ = pearsonr(y_demeaned, pred_demeaned)
    
    if np.isnan(r):
        return 0.0
        
    penalty = _check_monotonicity_penalty(y_m, pred_m, w_m)
    return abs(r) * penalty

# Map metric to gplearn
ic_fitness = make_fitness(function=_ic_metric, greater_is_better=True)

def _sharpe_metric(y, y_pred, w):
    mask = ~np.isnan(y_pred) & ~np.isnan(y) & ~np.isinf(y_pred)
    if not np.any(mask) or len(y[mask]) < 2: return 0.0
    
    y_m, pred_m, w_m = y[mask], y_pred[mask], w[mask].astype(int)
    counts = np.bincount(w_m)
    counts_safe = np.where(counts == 0, 1, counts)
    
    pred_means = np.bincount(w_m, weights=pred_m) / counts_safe
    w_weights = pred_m - pred_means[w_m]
    
    abs_w = np.abs(w_weights)
    abs_sums = np.bincount(w_m, weights=abs_w)
    abs_sums_safe = np.where(abs_sums == 0, 1, abs_sums)
    w_weights_norm = w_weights / abs_sums_safe[w_m]
    
    daily_pnl = np.bincount(w_m, weights=w_weights_norm * y_m)[counts > 0]
    std = np.std(daily_pnl)
    if std < 1e-6: return 0.0
    
    penalty = _check_monotonicity_penalty(y_m, pred_m, w_m)
    return (np.mean(daily_pnl) / std) * penalty

def _pnl_dd_metric(y, y_pred, w):
    mask = ~np.isnan(y_pred) & ~np.isnan(y) & ~np.isinf(y_pred)
    if not np.any(mask) or len(y[mask]) < 2: return 0.0
    
    y_m, pred_m, w_m = y[mask], y_pred[mask], w[mask].astype(int)
    counts = np.bincount(w_m)
    counts_safe = np.where(counts == 0, 1, counts)
    
    pred_means = np.bincount(w_m, weights=pred_m) / counts_safe
    w_weights = pred_m - pred_means[w_m]
    
    abs_w = np.abs(w_weights)
    abs_sums = np.bincount(w_m, weights=abs_w)
    abs_sums_safe = np.where(abs_sums == 0, 1, abs_sums)
    w_weights_norm = w_weights / abs_sums_safe[w_m]
    
    valid_pnl = np.bincount(w_m, weights=w_weights_norm * y_m)[counts > 0]
    cum_pnl = np.cumsum(valid_pnl)
    if len(cum_pnl) == 0: return 0.0
    
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = np.max(drawdowns)
    total_pnl = cum_pnl[-1]
    
    if total_pnl <= 0: return total_pnl
    
    penalty = _check_monotonicity_penalty(y_m, pred_m, w_m)
    return (total_pnl / (max_dd + 1e-4)) * penalty

sharpe_fitness = make_fitness(function=_sharpe_metric, greater_is_better=True)
pnl_dd_fitness = make_fitness(function=_pnl_dd_metric, greater_is_better=True)

def _cs_rank(x):
    """
    Custom Primitive: Returns a simple normalized rank.
    (In a true quant engine, this is cross-sectional grouped by date).
    """
    arr = np.nan_to_num(x)
    return pd.Series(arr).rank(pct=True).values

cs_rank_func = make_function(function=_cs_rank, name='rank', arity=1)

# ═══════════════════════════════════════════════════════════════
# Time-Series Primitives (Using Fast 1D Numpy Shifts with Global Categorical Boundary Masking)
# ═══════════════════════════════════════════════════════════════

GLOBAL_MASK_5 = None
GLOBAL_MASK_10 = None
GLOBAL_MASK_14 = None
GLOBAL_MASK_20 = None
GLOBAL_MASK_26 = None

def _arr(x, mask_len):
    if isinstance(x, (float, int)): return np.full(mask_len, float(x))
    return np.nan_to_num(x)

def _ts_delay_5(x):
    if GLOBAL_MASK_5 is None: return np.zeros_like(x)
    x_val = _arr(x, len(GLOBAL_MASK_5))
    res = np.roll(x_val, 5)
    res[GLOBAL_MASK_5] = np.nan
    return np.nan_to_num(res)

def _ts_sma_10(x):
    if GLOBAL_MASK_10 is None: return np.zeros_like(x)
    x_val = _arr(x, len(GLOBAL_MASK_10))
    res = pd.Series(x_val).rolling(10).mean().values
    res[GLOBAL_MASK_10] = np.nan
    return np.nan_to_num(res)

def _ts_sma_20(x):
    if GLOBAL_MASK_20 is None: return np.zeros_like(x)
    x_val = _arr(x, len(GLOBAL_MASK_20))
    res = pd.Series(x_val).rolling(20).mean().values
    res[GLOBAL_MASK_20] = np.nan
    return np.nan_to_num(res)

def _ts_max_20(x):
    if GLOBAL_MASK_20 is None: return np.zeros_like(x)
    x_val = _arr(x, len(GLOBAL_MASK_20))
    res = pd.Series(x_val).rolling(20).max().values
    res[GLOBAL_MASK_20] = np.nan
    return np.nan_to_num(res)

def _ts_min_20(x):
    if GLOBAL_MASK_20 is None: return np.zeros_like(x)
    x_val = _arr(x, len(GLOBAL_MASK_20))
    res = pd.Series(x_val).rolling(20).min().values
    res[GLOBAL_MASK_20] = np.nan
    return np.nan_to_num(res)

def _ts_rsi_14(x):
    if GLOBAL_MASK_14 is None: return np.zeros_like(x) + 50.0
    x_val = _arr(x, len(GLOBAL_MASK_14))
    delta = pd.Series(x_val).diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(14).mean().bfill()
    avg_loss = loss.rolling(14).mean().bfill()
    rs = avg_gain / avg_loss.replace(0, 1e-5)
    rsi = 100 - (100 / (1 + rs))
    res = rsi.values
    res[GLOBAL_MASK_14] = 50.0
    return np.nan_to_num(res)

def _ts_macd_line(x):
    if GLOBAL_MASK_26 is None: return np.zeros_like(x)
    x_val = _arr(x, len(GLOBAL_MASK_26))
    sema = pd.Series(x_val).ewm(span=12, adjust=False).mean()
    lema = pd.Series(x_val).ewm(span=26, adjust=False).mean()
    res = (sema - lema).values
    res[GLOBAL_MASK_26] = 0.0
    return np.nan_to_num(res)

def _ts_vol_20(x):
    if GLOBAL_MASK_20 is None: return np.zeros_like(x)
    x_val = _arr(x, len(GLOBAL_MASK_20))
    res = pd.Series(x_val).rolling(20).std().bfill().values
    res[GLOBAL_MASK_20] = 0.0
    return np.nan_to_num(res)

delay_5 = make_function(function=_ts_delay_5, name='delay_5', arity=1)
sma_10 = make_function(function=_ts_sma_10, name='sma_10', arity=1)
sma_20 = make_function(function=_ts_sma_20, name='sma_20', arity=1)
ts_max_20 = make_function(function=_ts_max_20, name='ts_max_20', arity=1)
ts_min_20 = make_function(function=_ts_min_20, name='ts_min_20', arity=1)
rsi_14 = make_function(function=_ts_rsi_14, name='rsi_14', arity=1)
macd_line = make_function(function=_ts_macd_line, name='macd_line', arity=1)
vol_20 = make_function(function=_ts_vol_20, name='vol_20', arity=1)

# ═══════════════════════════════════════════════════════════════
# Factor Miner Execution Matrix
# ═══════════════════════════════════════════════════════════════

def discover_alpha_factors(
    df: pd.DataFrame, 
    generations: int = 4, 
    pop_size: int = 200, 
    horizon: int = 1,
    fitness_metric: str = "ic",
    syntax_set: str = "all",
    enforce_monotonicity: bool = False,
    progress_callback=None
):
    """
    Executes a GPU-capable/Threaded Symbolic Regression to discover synergistic alpha formulas natively targeting a bounded predictive horizon.
    """
    global ENFORCE_MONOTONICITY
    ENFORCE_MONOTONICITY = enforce_monotonicity
    
    if progress_callback: 
        progress_callback(10, "Aligning Target Tensors...")
    
    df = df.copy()
    df.sort_values(by=["ticker", "date"], inplace=True)
    
    # Calculate truth bounds dynamically mapped to the bounded Horizon
    if "fwd_return" not in df.columns:
        df["fwd_return"] = df.groupby("ticker")["close"].shift(-horizon) / df["close"] - 1
        
    if "vwap" in df.columns:
        df["vwap"] = df["vwap"].fillna(df["close"])
    if "trades" in df.columns:
        df["trades"] = df["trades"].fillna(1)
        
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "fwd_return"])
    
    features = ["open", "high", "low", "close", "volume", "returns"]
    if "vwap" in df.columns: features.append("vwap")
    if "trades" in df.columns: features.append("trades")
    
    if "returns" not in df.columns:
        df["returns"] = df.groupby("ticker")["close"].pct_change()
    df.dropna(subset=features + ["fwd_return"], inplace=True)
        
    X = df[features].values
    y = df["fwd_return"].values
    
    # ── Temporal Boundary Masks ──
    global GLOBAL_MASK_5, GLOBAL_MASK_10, GLOBAL_MASK_14, GLOBAL_MASK_20, GLOBAL_MASK_26
    GLOBAL_MASK_5 = (df['ticker'] != df['ticker'].shift(5)).values
    GLOBAL_MASK_10 = (df['ticker'] != df['ticker'].shift(9)).values  # 10-day rolling needs 9 prior days blanked
    GLOBAL_MASK_14 = (df['ticker'] != df['ticker'].shift(13)).values # 14-day rolling 
    GLOBAL_MASK_20 = (df['ticker'] != df['ticker'].shift(19)).values # 20-day rolling needs 19 prior days blanked
    GLOBAL_MASK_26 = (df['ticker'] != df['ticker'].shift(25)).values # 26-day rolling
    
    # Encode dates into integer IDs explicitly for passing through `sample_weight` mapping natively
    w = df["date"].astype('category').cat.codes.values

    if progress_callback: 
        progress_callback(30, "Initializing Genetic Engine (gplearn)...")

    # The mathematical bounds the engine is allowed to combine
    if syntax_set == "linear":
        function_set = ['add', 'sub', 'mul', 'div', 'abs', 'log', 'sqrt']
    elif syntax_set == "cross_sectional":
        function_set = ['add', 'sub', 'mul', 'div', 'abs', cs_rank_func]
    elif syntax_set == "technical":
        function_set = ['add', 'sub', 'mul', 'div', delay_5, sma_10, sma_20, ts_max_20, ts_min_20, rsi_14, macd_line, vol_20]
    else:
        function_set = ['add', 'sub', 'mul', 'div', 'abs', 'log', 'sqrt', cs_rank_func, delay_5, sma_10, sma_20, ts_max_20, ts_min_20, rsi_14, macd_line, vol_20]
    
    if fitness_metric == "ic": target_metric = ic_fitness
    elif fitness_metric == "sharpe": target_metric = sharpe_fitness
    elif fitness_metric == "pnl_dd": target_metric = pnl_dd_fitness
    else: target_metric = "mean absolute error"
    
    # Optional logic: Stop running evolution if the structural error hits optimal bounds early
    stop_crit = 0.2 if fitness_metric == "ic" else 0.05
    if fitness_metric in ["sharpe", "pnl_dd"]:
        stop_crit = 3.0 # i.e. 3.0 Sharpe or 3.0 Calmar
    
    est = SymbolicRegressor(
        population_size=pop_size,
        generations=generations,
        tournament_size=20,
        function_set=function_set,
        metric=target_metric,
        stopping_criteria=stop_crit,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        verbose=0,
        n_jobs=1,
        random_state=42
    )
    
    if progress_callback: 
        progress_callback(50, f"Evaluating {generations * pop_size} Synthetic ASTs natively...")
        
    # Launch Evolutionary Tree Search mapping 'w' through natively as grouped temporal boundaries
    est.fit(X, y, sample_weight=w)
    
    if progress_callback: 
        progress_callback(90, "Extracting Top Alpha Formulations...")
    
    # Scrape the final generation tree for the most optimal structural formulas
    final_programs = est._programs[-1]
    best_programs = sorted(
        [p for p in final_programs if p is not None], 
        key=lambda x: x.fitness_, 
        reverse=True
    )
    
    results = []
    seen = set()
    feature_names = ["Open", "High", "Low", "Close", "Volume", "Returns"]
    if "vwap" in df.columns: feature_names.append("VWAP")
    if "trades" in df.columns: feature_names.append("Trades")
    
    for p in best_programs:
        formula_str = str(p)
        # Skip overly simplistic rules
        if len(formula_str) > 15 and formula_str not in seen:
            seen.add(formula_str)
            
            # Map X[N] back to human-readable strings
            for i, name in enumerate(feature_names):
                formula_str = formula_str.replace(f"X{i}", name)
                
            results.append({
                "formula": formula_str,
                "fitness_score": round(p.fitness_, 4)
            })
            
        if len(results) >= 5:
            break
            
    if progress_callback: 
        progress_callback(100, "Done")
            
    return results
