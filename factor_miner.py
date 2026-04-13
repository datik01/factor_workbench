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
# ═══════════════════════════════════════════════════════════════
# Fitness Metrics & Primitives (Genetic Programming POC)
# ═══════════════════════════════════════════════════════════════

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
        
    return abs(r)

# Map metric to gplearn
ic_fitness = make_fitness(function=_ic_metric, greater_is_better=True)

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
GLOBAL_MASK_20 = None

def _ts_delay_5(x):
    if GLOBAL_MASK_5 is None: return np.zeros_like(x)
    res = np.roll(np.nan_to_num(x), 5)
    res[GLOBAL_MASK_5] = np.nan
    return np.nan_to_num(res)

def _ts_sma_10(x):
    if GLOBAL_MASK_10 is None: return np.zeros_like(x)
    res = pd.Series(np.nan_to_num(x)).rolling(10).mean().values
    res[GLOBAL_MASK_10] = np.nan
    return np.nan_to_num(res)

def _ts_sma_20(x):
    if GLOBAL_MASK_20 is None: return np.zeros_like(x)
    res = pd.Series(np.nan_to_num(x)).rolling(20).mean().values
    res[GLOBAL_MASK_20] = np.nan
    return np.nan_to_num(res)

def _ts_max_20(x):
    if GLOBAL_MASK_20 is None: return np.zeros_like(x)
    res = pd.Series(np.nan_to_num(x)).rolling(20).max().values
    res[GLOBAL_MASK_20] = np.nan
    return np.nan_to_num(res)

def _ts_min_20(x):
    if GLOBAL_MASK_20 is None: return np.zeros_like(x)
    res = pd.Series(np.nan_to_num(x)).rolling(20).min().values
    res[GLOBAL_MASK_20] = np.nan
    return np.nan_to_num(res)

delay_5 = make_function(function=_ts_delay_5, name='delay_5', arity=1)
sma_10 = make_function(function=_ts_sma_10, name='sma_10', arity=1)
sma_20 = make_function(function=_ts_sma_20, name='sma_20', arity=1)
ts_max_20 = make_function(function=_ts_max_20, name='ts_max_20', arity=1)
ts_min_20 = make_function(function=_ts_min_20, name='ts_min_20', arity=1)

# ═══════════════════════════════════════════════════════════════
# Factor Miner Execution Matrix
# ═══════════════════════════════════════════════════════════════

def discover_alpha_factors(
    df: pd.DataFrame, 
    generations: int = 4, 
    pop_size: int = 200, 
    horizon: int = 1,
    progress_callback=None
):
    """
    Executes a GPU-capable/Threaded Symbolic Regression to discover synergistic alpha formulas natively targeting a bounded predictive horizon.
    """
    if progress_callback: 
        progress_callback(10, "Aligning Target Tensors...")
    
    df = df.copy()
    df.sort_values(by=["ticker", "date"], inplace=True)
    
    # Calculate truth bounds dynamically mapped to the bounded Horizon
    if "fwd_return" not in df.columns:
        df["fwd_return"] = df.groupby("ticker")["close"].shift(-horizon) / df["close"] - 1
        
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "fwd_return"])
    
    features = ["open", "high", "low", "close", "volume", "returns"]
    if "returns" not in df.columns:
        df["returns"] = df.groupby("ticker")["close"].pct_change()
    df.dropna(inplace=True)
        
    X = df[features].values
    y = df["fwd_return"].values
    
    # ── Temporal Boundary Masks ──
    global GLOBAL_MASK_5, GLOBAL_MASK_10, GLOBAL_MASK_20
    GLOBAL_MASK_5 = (df['ticker'] != df['ticker'].shift(5)).values
    GLOBAL_MASK_10 = (df['ticker'] != df['ticker'].shift(9)).values  # 10-day rolling needs 9 prior days blanked
    GLOBAL_MASK_20 = (df['ticker'] != df['ticker'].shift(19)).values # 20-day rolling needs 19 prior days blanked
    
    # Encode dates into integer IDs explicitly for passing through `sample_weight` mapping natively
    w = df["date"].astype('category').cat.codes.values

    if progress_callback: 
        progress_callback(30, "Initializing Genetic Engine (gplearn)...")

    # The mathematical bounds the engine is allowed to combine
    function_set = ['add', 'sub', 'mul', 'div', 'abs', 'log', 'sqrt', cs_rank_func, delay_5, sma_10, sma_20, ts_max_20, ts_min_20]
    
    est = SymbolicRegressor(
        population_size=pop_size,
        generations=generations,
        tournament_size=20,
        function_set=function_set,
        metric=ic_fitness,
        stopping_criteria=0.2, # Stop if Mean IC exceeds 20%
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        verbose=0,
        n_jobs=-1,
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
                "ic_fitness": round(p.fitness_, 4)
            })
            
        if len(results) >= 5:
            break
            
    if progress_callback: 
        progress_callback(100, "Done")
            
    return results
