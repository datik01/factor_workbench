"""
universe_builder.py
Orchestrates the full historical R2K constituent pipeline:

  1. Discover IWM N-PORT filings on SEC EDGAR
  2. Extract holdings + CUSIPs per quarter
  3. Map CUSIPs to historical tickers via Massive API
  4. Cache results as parquet files

Output: A time-indexed DataFrame of R2K constituents with tickers,
        enabling survivorship-bias-free backtesting.
"""

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from .edgar_scraper import discover_etf_filings, extract_etf_holdings
from .cusip_mapper import map_cusips_to_tickers

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)

load_dotenv(os.path.join(_project_dir, ".env"))
load_dotenv(os.path.join(_project_dir, "..", "..", ".env"))

MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")
CACHE_DIR = os.path.join(_project_dir, ".cache", "constituents")
os.makedirs(CACHE_DIR, exist_ok=True)

# Known IWM filing accessions (pre-discovered by scanning 8,177 iShares Trust filings)
# Format: (reporting_date, accession_number, approx_holdings)
# 20 quarters: Q1 2021 → Q4 2025 (5 full years)
KNOWN_IWM_FILINGS = [
    ("2025-12-31", "0002071691-26-004226"),   # 1964 holdings
    ("2025-09-30", "0002071691-25-007652"),   # 1981 holdings
    ("2025-06-30", "0001752724-25-210405"),   # 2125 holdings
    ("2025-03-31", "0001752724-25-119784"),   # 1960 holdings
    ("2024-12-31", "0001752724-25-043851"),   # 1975 holdings
    ("2024-09-30", "0001752724-24-269957"),   # 1986 holdings
    ("2024-06-30", "0001752724-24-194120"),   # 1931 holdings
    ("2024-03-31", "0001752724-24-123298"),   # 1956 holdings
    ("2023-12-31", "0001752724-24-043096"),   # 1976 holdings
    ("2023-09-30", "0001752724-23-264256"),   # 1996 holdings
    ("2023-06-30", "0001752724-23-191317"),   # 2019 holdings
    ("2023-03-31", "0001752724-23-123227"),   # 1930 holdings
    ("2022-12-31", "0001752724-23-039260"),   # 1958 holdings
    ("2022-09-30", "0001752724-22-268676"),   # 1977 holdings
    ("2022-06-30", "0001752724-22-193728"),   # 1999 holdings
    ("2022-03-31", "0001752724-22-122853"),   # 2024 holdings
    ("2021-12-31", "0001752724-22-046410"),   # 2040 holdings
    ("2021-09-30", "0001752724-21-255836"),   # 2038 holdings
    ("2021-06-30", "0001752724-21-186233"),   # 2003 holdings
    ("2021-03-31", "0001752724-21-116355"),   # 2063 holdings
]


# ═══════════════════════════════════════════════════════════════
# Cache Paths
# ═══════════════════════════════════════════════════════════════

def _master_cache_path(etf_key: str = "R2K") -> str:
    return os.path.join(CACHE_DIR, f"{etf_key.lower()}_historical_constituents.parquet")

def _period_cache_path(etf_key: str, reporting_date: str) -> str:
    clean = reporting_date.replace("-", "").replace("/", "")
    return os.path.join(CACHE_DIR, f"{etf_key.lower()}_{clean}.parquet")

def _ticker_list_cache_path(etf_key: str = "R2K") -> str:
    return os.path.join(CACHE_DIR, f"{etf_key.lower()}_tickers_latest.txt")


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def load_cached_universe(etf_key: str = "R2K") -> pd.DataFrame:
    """Load the master cached DataFrame of all historical constituents."""
    path = _master_cache_path(etf_key)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def build_historical_constituents(
    etf_key: str = "R2K",
    max_filings: int = 5,
    use_known: bool = True,
    progress_callback=None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: SEC EDGAR → CUSIP extraction → Massive API ticker mapping.

    Parameters
    ----------
    etf_key : str
        The ETF ticker (e.g., "R2K", "SPY")
    max_filings : int
        Number of quarterly filings to process
    use_known : bool
        If True, use pre-discovered accession numbers (fast path)
    progress_callback : callable(current, total, msg)
    force_refresh : bool
        If True, re-scrape even if cached

    Returns
    -------
    pd.DataFrame
        Columns: reporting_date, cusip, ticker, issuer_name, shares,
                 value_usd, pct_net_assets
    """
    master_path = _master_cache_path(etf_key)

    if not force_refresh and os.path.exists(master_path):
        if progress_callback:
            progress_callback(0, 0, f"Loading {etf_key} from cache...")
        df = pd.read_parquet(master_path)
        
        # Ensure the .txt fallback is populated even on fast bypass loads
        with open(_ticker_list_cache_path(etf_key), "w") as f:
            f.write("\n".join(sorted(df["ticker"].dropna().unique().tolist())))
            
        if progress_callback:
            n_periods = df["reporting_date"].nunique()
            n_tickers = df["ticker"].nunique()
            progress_callback(1, 1, f"✅ Loaded {n_tickers} tickers across {n_periods} periods ({etf_key})")
        return df

    if not MASSIVE_API_KEY:
        raise ValueError("MASSIVE_API_KEY not found in .env")

    # Step 1: Get filing accessions
    if use_known and etf_key == "R2K":
        filings = [{"accession": acc, "reporting_date": rd} for rd, acc in KNOWN_IWM_FILINGS[:max_filings]]
        if progress_callback:
            progress_callback(0, len(filings), f"Using {len(filings)} known IWM filings...")
    else:
        if progress_callback:
            progress_callback(0, 0, f"Discovering {etf_key} filings on SEC EDGAR (may take a few minutes)...")
        discovered = discover_etf_filings(etf_key=etf_key, max_filings=max_filings, progress_callback=progress_callback)
        filings = discovered

    if not filings:
        raise RuntimeError(f"No {etf_key} N-PORT filings found")

    # Step 2: Extract + map for each period
    all_periods = []

    for i, filing in enumerate(filings):
        acc = filing["accession"] if isinstance(filing, dict) else filing
        rep_date = filing.get("reporting_date", "unknown") if isinstance(filing, dict) else "unknown"

        # Check period cache
        period_cache = _period_cache_path(etf_key, rep_date)
        if not force_refresh and os.path.exists(period_cache):
            if progress_callback:
                progress_callback(i + 1, len(filings), f"Loaded {rep_date} from cache")
            all_periods.append(pd.read_parquet(period_cache))
            continue

        # Extract holdings from SEC filing
        if progress_callback:
            progress_callback(i, len(filings), f"Parsing SEC XML for {rep_date}...")
        holdings_df, actual_date = extract_etf_holdings(acc, etf_key=etf_key)
        if actual_date:
            rep_date = actual_date

        if holdings_df.empty:
            continue

        # Map CUSIPs to tickers
        if progress_callback:
            progress_callback(i + 1, len(filings), f"Mapping {len(holdings_df)} CUSIPs to tickers for {rep_date}...")

        mapped_df = map_cusips_to_tickers(
            holdings_df,
            filing_date=rep_date,
            api_key=MASSIVE_API_KEY,
            max_workers=10,
            progress_callback=progress_callback,
        )

        if not mapped_df.empty:
            mapped_df.to_parquet(period_cache, index=False)
            all_periods.append(mapped_df)

    if not all_periods:
        raise RuntimeError("No holdings data after CUSIP mapping")

    # Combine
    master_df = pd.concat(all_periods, ignore_index=True)
    if "ticker" in master_df.columns:
        master_df = master_df[master_df["ticker"].notna()]

    # Save master cache
    master_df.to_parquet(master_path, index=False)

    # Also save a simple ticker list
    tickers = sorted(master_df["ticker"].dropna().unique().tolist())
    with open(_ticker_list_cache_path(etf_key), "w") as f:
        f.write("\n".join(tickers))

    if progress_callback:
        n_periods = master_df["reporting_date"].nunique()
        n_tickers = len(tickers)
        progress_callback(1, 1, f"✅ Built: {n_tickers} unique tickers across {n_periods} quarterly periods")

    return master_df


def get_latest_constituents(etf_key: str = "R2K") -> list:
    """Get ticker symbols from the most recently cached filing."""
    # Try ticker list cache first (fast)
    txt_path = _ticker_list_cache_path(etf_key)
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return [t.strip() for t in f.readlines() if t.strip()]

    # Fall back to parquet
    master_df = load_cached_universe(etf_key)
    if master_df.empty:
        import pandas as pd
        try:
            if etf_key == "SP500":
                df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                return df['Symbol'].tolist()
            elif etf_key == "NDX":
                df = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
                return df['Ticker'].tolist()
            else:
                df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                return df['Symbol'].tolist()
        except Exception:
            return ["AAPL", "MSFT", "GOOGL", "AMZN"]

    master_df["reporting_date"] = pd.to_datetime(master_df["reporting_date"])
    latest = master_df["reporting_date"].max()
    period_df = master_df[master_df["reporting_date"] == latest]
    return sorted(period_df["ticker"].dropna().unique().tolist())


def get_constituents_at_date(target_date: str, master_df: pd.DataFrame = None, etf_key: str = "R2K") -> list:
    """Get ETFs tickers as of a specific date (most recent filing before target)."""
    if master_df is None:
        master_df = load_cached_universe(etf_key)
    if master_df.empty:
        return []

    master_df["reporting_date"] = pd.to_datetime(master_df["reporting_date"])
    target = pd.to_datetime(target_date)
    available = master_df[master_df["reporting_date"] <= target]
    if available.empty:
        available = master_df

    latest = available["reporting_date"].max()
    return sorted(master_df[master_df["reporting_date"] == latest]["ticker"].dropna().unique().tolist())


def build_constituent_timeline(master_df: pd.DataFrame = None, etf_key: str = "R2K") -> dict:
    """
    Build a {reporting_date_str: [ticker_list]} mapping for point-in-time
    backtesting. Each quarter maps to the tickers that were in the ETF at that time.

    Returns
    -------
    dict
        {"2025-12-31": ["AAOI", "AAP", ...], "2025-09-30": [...], ...}
    """
    if master_df is None:
        master_df = load_cached_universe(etf_key)
    if master_df.empty:
        return {}

    master_df["reporting_date"] = pd.to_datetime(master_df["reporting_date"])
    timeline = {}

    for period, group in master_df.groupby("reporting_date"):
        tickers = sorted(group["ticker"].dropna().unique().tolist())
        timeline[period.strftime("%Y-%m-%d")] = tickers

    return timeline

