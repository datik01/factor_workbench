"""
app.py
Factor Workbench — Institutional-Grade Shiny for Python Dashboard
Danny Atik - SYSEN 5381

Cross-sectional factor portfolio construction across the full Russell 2000
universe with multi-agent AI orchestration via Gemma 4 (Ollama).
"""

from shiny import App, ui, render, reactive
import pandas as pd
import random
import os
from datetime import datetime
import subprocess

from tradestation_api import TradeStationClient
import tools
from factor_miner import discover_alpha_factors
from constituents.universe_builder import get_latest_constituents

# --- HOTFIX: Plotly/Packaging Strict Version Parser Bug ---
# Plotly defines its widget version as "0.9.*" which breaks packaging>=22
import packaging.version
_orig_Version = packaging.version.Version
def _patched_Version(v):
    if isinstance(v, str) and v.endswith(".*"):
        v = v.replace(".*", ".0")
    return _orig_Version(v)
packaging.version.Version = _patched_Version
# --------------------------------------------------------

# ═══════════════════════════════════════════════════════════════
# Load Russell 2000 Ticker Universe (SEC EDGAR-derived, bias-free)
# ═══════════════════════════════════════════════════════════════

_script_dir = os.path.dirname(os.path.abspath(__file__))

# Core Engine successfully dynamically loads constituents arrays.

try:
    from constituents.universe_builder import build_constituent_timeline
    CONSTITUENT_TIMELINE = build_constituent_timeline()
except Exception:
    CONSTITUENT_TIMELINE = None


# ═══════════════════════════════════════════════════════════════
# Factor Themes
# ═══════════════════════════════════════════════════════════════

THEMES = {
    "Momentum (1-Month)": "momentum_1m",
    "Momentum (3-Month)": "momentum_3m",
    "Momentum (6-Month)": "momentum_6m",
    "Momentum (12-Month)": "momentum_12m",
    "Mean Reversion (5-Day)": "reversion",
    "Low Volatility": "volatility",
    "Abnormal Volume": "volume",
    "Size (Market Cap Proxy)": "size",
    "PE Ratio (Fundamental)": "pe_ratio",
    "PB Ratio (Fundamental)": "pb_ratio",
    "PS Ratio (Fundamental)": "ps_ratio",
    "EPS (Fundamental)": "eps",
    "Revenues (Fundamental)": "revenues",
    "Gross Profit (FS)": "gross_profit",
    "Operating Income (FS)": "operating_income",
    "Net Income (FS)": "net_income",
    "R&D Spend (FS)": "research_and_development",
    "Equity (FS)": "equity",
    "Total Assets (FS)": "assets",
    "Total Liabilities (FS)": "liabilities",
    "Current Assets (FS)": "current_assets",
    "Current Liabilities (FS)": "current_liabilities",
    "Inventory (FS)": "inventory",
    "Net Cash Flow (FS)": "net_cash_flow",
    "Operating Cash Flow (FS)": "operating_cash_flow",
    "Cost of Revenue (FS)": "cost_of_revenue",
    "Interest Expense (FS)": "interest_expense",
    "Dividends Paid (FS)": "dividends_paid",
    "Market Cap (Derived)": "market_cap",
    "Shares Outstanding (Derived)": "shares",
}

# ═══════════════════════════════════════════════════════════════
# Premium CSS
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0c10;
    --bg-secondary: #12151e;
    --bg-card: #161a26;
    --bg-elevated: #1c2033;
    --border: #252a3a;
    --text-primary: #e8eaf0;
    --text-secondary: #8b90a0;
    --accent-teal: #00d4aa;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --gradient-primary: linear-gradient(135deg, #00d4aa 0%, #3b82f6 100%);
    --gradient-purple: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
}

* { box-sizing: border-box; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

/* ── Sidebar ── */
.sidebar {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    padding: 20px !important;
}

/* ── Cards ── */
.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
    backdrop-filter: blur(12px);
}

/* ── Modals ── */
.modal-content {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
}
.modal-header, .modal-footer {
    border-color: var(--border) !important;
}
.modal-title {
    font-weight: 600;
    color: var(--text-primary) !important;
}

/* ── Navigation Tabs ── */
.nav-pills .nav-link {
    color: var(--text-secondary) !important;
    font-weight: 500;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    transition: all 0.25s ease;
}
.nav-pills .nav-link:hover {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}
.nav-pills .nav-link.active {
    background: var(--gradient-primary) !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3);
}

/* ── Value Boxes ── */
.bslib-value-box {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    overflow: hidden;
}
.log-message {
    font-size: 0.95rem;
    color: #ffffff !important;
    white-space: pre-wrap;
    opacity: 0.9;
}
.bslib-value-box .value-box-title {
    font-size: 0.85rem !important;
    white-space: nowrap;
    opacity: 0.9;
}
.bslib-value-box .value-box-value {
    font-size: 1.6rem !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bslib-value-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}

/* ── Primary Button ── */
.btn-run {
    background: var(--gradient-primary) !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    font-size: 15px !important;
    letter-spacing: 0.3px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(0, 212, 170, 0.25);
    width: 100%;
}
.btn-run:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0, 212, 170, 0.4) !important;
}


/* ── Header ── */
.app-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.app-subtitle {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 400;
    margin: 2px 0 0 0;
}

/* ── Status ── */
.status-text {
    color: var(--text-secondary);
    font-style: italic;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 8px 12px;
    background: var(--bg-elevated);
    border-radius: 8px;
    border: 1px solid var(--border);
}

/* ── Metric Labels ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}

/* ── Slider Fixes ── */
.irs--shiny .irs-bar { background: var(--accent-teal); border-color: var(--accent-teal); }
.irs--shiny .irs-handle { border-color: var(--accent-teal); }
.irs--shiny .irs-single { background: var(--accent-teal); }

/* ── Label Styling ── */
label.control-label {
    color: var(--text-secondary) !important;
    font-weight: 500;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 6px;
}

.form-check-label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* ── Select Input ── */
.selectize-input, .form-control, .form-select {
    background-color: var(--bg-elevated) !important;
    border: 2px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
.selectize-dropdown {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
.selectize-input .item {
    background: rgba(0, 212, 170, 0.2) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0, 212, 170, 0.3) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}
.selectize-dropdown .option.active {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Config Section ── */
.config-section {
    background: var(--bg-elevated);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid var(--border);
    margin-bottom: 16px;
}
.config-section h6 {
    color: var(--accent-teal);
    font-weight: 700;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
}
.universe-badge {
    display: inline-block;
    background: var(--gradient-purple);
    color: white;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}
"""


def tip(label: str, text: str):
    return ui.tooltip(ui.span(label, " ", ui.tags.span("❔", style="font-size: 0.85em; opacity: 0.8; cursor: help;")), text)

# ═══════════════════════════════════════════════════════════════
# UI Layout
# ═══════════════════════════════════════════════════════════════

app_ui = ui.page_fluid(
    ui.tags.style(CUSTOM_CSS),

    # Header
    ui.div(
        ui.h2("⚡ Factor Workbench", class_="app-title"),
        ui.p("Cross-Sectional Portfolio Construction Engine", class_="app-subtitle"),
        style="padding: 20px 24px 10px 24px;",
    ),

    ui.layout_sidebar(
        ui.sidebar(
            # Universe Selection
            ui.div(
                ui.h6(tip("Universe", "The bounded asset ETF proxy index to map and filter historical constituents against.")),
                ui.input_select("universe_selection", "", choices={"R2K": "Russell 2000 Index", "SP500": "S&P 500 Index", "NDX": "Nasdaq 100 Index"}, selected="R2K"),
                ui.input_text("universe_filter", tip("Universe Filter Expression", "Optional logical string to evaluate dynamically on the historical pool (e.g. operating_cash_flow > 0)"), value="close >= 2 and avg_30d_volume >= 3000000 and perf_1d < -0.09 and rel_vol_1d > 1 and atr_pct_14 < 0.15 and close < vwap", placeholder="e.g. operating_cash_flow > 0"),
                class_="config-section",
            ),

            # Factor Config
            ui.div(
                ui.h6("Factor Configuration"),
                ui.input_selectize("themes", tip("Base Analytics", "Pre-configured factors to evaluate cross-sectionally."), choices=list(THEMES.keys()), multiple=True, selected=["momentum_1m"]),
                ui.input_text("custom_formula", tip("🧪 Custom GP Alpha Formula", "Inject an exact mathematical PyGP syntax tree to bypass standard themes."), value="mul(div(close, sma_10(close)), div(volume, avg_30d_volume))", placeholder="e.g. sma_20(rsi_14(Close)) (Overrides themes)"),
                ui.input_select("mined_formula_dropdown", tip("🧬 Selected Mined Alpha", "Pull winning formulas from the Alpha Miner output."), choices={"None": "None"}, selected="None"),
                ui.input_switch("invert_factor", tip("Invert Factor (Low to High)", "Flips the strategy to buy the lowest scoring stocks instead of the highest."), value=True),
                ui.input_switch("enable_calendar", tip("Generate P&L Calendar", "Disable to massively speed up backtest finalization by bypassing HTML construction."), value=True),
                class_="config-section",
            ),

            # Portfolio Config
            ui.div(
                ui.h6("Portfolio Configuration"),
                ui.input_select("strategy_type", tip("Strategy Type", "Dictates capitalization allocation (Long/Short neutral, or directional Long/Short only)."), choices=["Long/Short", "Long Only", "Short Only", "Short/Short"], selected="Short Only"),
                ui.input_select("quantile_split", tip("Analysis Quantiles", "Splits the ranked universe into N fractional buckets to evaluate top vs bottom tier spreads."), choices={"3": "Tertiles (3)", "4": "Quartiles (4)", "5": "Quintiles (5)", "10": "Deciles (10)"}, selected="10"),
                ui.input_select("portfolio_sizing_type", tip("Portfolio Sizing Logic", "Allocate capital by fixed asset bounds or by dynamic universe percentages."), choices=["Absolute Count", "Percentage"], selected="Absolute Count"),
                ui.input_numeric("portfolio_size", tip("Portfolio Size / Percent limit", "Enter absolute count of assets (e.g., 100) or total universe percentage (e.g., 20)"), value=25),
                ui.input_select("sizing_strategy", tip("Position Sizing Strategy", "Mathematical mechanism for allocating capital among targets."), choices=["Dynamic (1/Active)", "Fixed Fractional (1/N)", "Inverse Volatility (ATR)"], selected="Inverse Volatility (ATR)"),
                ui.input_select("liquidity_cap_type", tip("Liquidity Cap Type", "How to cap the maximum position size."), choices=["None", "Max ADV %", "Static $ Ceiling"], selected="Max ADV %"),
                ui.input_numeric("liquidity_cap_value", tip("Cap Value (% or $)", "Value corresponding to the cap type (e.g. 5 for 5% ADV or 50000 for $50k)."), value=5.0, min=0.1),
                ui.input_numeric("max_position_weight", tip("Max Position Weight %", "Absolute maximum portfolio percentage allocated to a single asset. (e.g., 15 for 15%). 100 disables cap."), value=100.0, min=1.0, max=100.0, step=1.0),
                ui.input_select("execution_timing", tip("Execution Timing", "Trade exactly at EOD Close (Standard, holds overnight) vs Intraday Open to Close (Zero overnight gap risk)."), choices=["Current Day Close", "Intraday (Open to Close)"], selected="Intraday (Open to Close)"),
                ui.input_numeric("initial_aum", tip("Initial AUM ($)", "Starting simulation capital dictating absolute dollar returns."), value=30000),
                ui.input_slider("year_range", tip("Analysis Period", "Historical year boundaries for testing."), min=2006, max=datetime.now().year, value=(2019, 2026), sep=""),
                ui.input_select("rebalance_freq", tip("Rebalance Frequency", "How often the algorithm recalculates ranks and shifts portfolio capital."), 
                                choices={"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}, 
                                selected="D"),
                ui.input_select("vol_target", tip("Volatility Targeting (Risk Parity)", "Dynamically scale margin leverage to hit a fixed annualized risk constraint (1.0x to 3.0x max)."), choices={"0": "Unleveraged (1.0x)", "0.10": "10% Target (Conservative)", "0.15": "15% Target", "0.20": "20% Target (Aggressive)", "0.25": "25% Target"}, selected="0"),
                ui.input_numeric("atr_sl_mult", tip("ATR Stop Loss", "Hard intraday exit overriding holding logic if negative movement drops below N * Average True Range (0 to disable)."), value=2.5, min=0.0, step=0.1),
                ui.input_numeric("atr_tp_mult", tip("ATR Take Profit", "Absolute profit taking limit bounding excessive positive movement instantly locking returns (0 to disable)."), value=1.25, min=0.0, step=0.1),
                ui.hr(),
                ui.input_numeric("slippage_bps", tip("Slippage Penalty (BPS)", "Frictional cost per trade (10 bps = 0.1%). Applied structurally to both Entry and Exit metrics."), value=0, min=0, step=5),
                ui.input_numeric("sl_slippage_bps", tip("SL Slippage Penalty (BPS)", "Specific frictional cost applied ONLY when a Stop-Loss is hit, overriding the baseline slippage."), value=50, min=0, step=5),
                ui.hr(),
                ui.input_switch("metric_scope", tip("Show Positional Metrics", "Toggle between individual Positional metrics vs Daily Aggregated metrics for win rates and expected value."), value=True),
                class_="config-section",
            ),

        ui.input_action_button("run_btn", "🚀 Run Portfolio Analysis", class_="btn-run"),
        ui.output_ui("stop_btn_ui"),

        ui.div(style="height: 16px;"),
            ui.output_ui("status_text"),

            width=300,
        ),

        ui.navset_card_pill(
            ui.nav_panel("📊 Dashboard",
                ui.output_ui("value_boxes"),
                ui.output_ui("plots_ui"),
            ),
            ui.nav_panel("📅 P&L Calendar",
                ui.output_ui("calendar_ui"),
            ),
            ui.nav_panel("🧬 AI Alpha Miner",
                ui.h4("Automated Formulaic Factor Discovery", style="color: white; mt-3"),
                ui.p("Uses PyGP (gplearn Symbolic Regression) to natively evolve and discover automated mathematical alpha synergies.", style="color: white; opacity: 0.9;"),
                ui.tags.ul(
                    ui.tags.li("Symbolic Regression: Generates thousands of randomized mathematical syntax trees exploring theoretical vectors.", style="color: #c7c7c7; margin-bottom: 5px;"),
                    ui.tags.li("Cross-Sectional Culling: Replaces weaker formulas iteratively using genetic mutation, crossover, and fitness tournament selection.", style="color: #c7c7c7; margin-bottom: 5px;"),
                    ui.tags.li("Parsimony Pressure: Mathematically penalizes formulas that become overly nested to prevent curve-fitting and hallucinatory extraction.", style="color: #c7c7c7; margin-bottom: 5px;"),
                    class_="mb-4"
                ),
                ui.layout_columns(
                    ui.input_select("miner_universe", tip("Universe Target", "The asset pool the genetic engine uses to train its formulas."), ["R2K", "SP500", "NDX"], selected="SP500"),
                    ui.input_select("miner_horizon", tip("Optimization Horizon", "The forward-looking return window the AI attempts to predict."), choices={"1": "Daily (1-Day)", "5": "Weekly (5-Day)", "21": "Monthly (21-Day)", "63": "Quarterly (63-Day)", "252": "Yearly (252-Day)"}, selected="1"),
                    ui.input_select("miner_fitness", tip("Genetic Fitness Objective", "The mathematical risk or accuracy metric the AI maximizes during evolution."), choices={"ic": "Information Coefficient (Rank)", "mae": "Mean Absolute Error (Magnitude)", "sharpe": "Sharpe Ratio (Return/Risk)", "pnl_dd": "Calmar Ratio (PNL / Max Drawdown)"}, selected="ic"),
                        ui.div(
                            ui.input_selectize(
                                "miner_funcs", 
                                tip("Theoretical Component Set", "Restricts the AI to only use specific mathematical functions and data structures."), 
                                choices={
                                    "Arithmetic Operations": {"grp_arithmetic": "All Arithmetic Operations", "add": "Addition (+)", "sub": "Subtraction (-)", "mul": "Multiplication (*)", "div": "Division (/)", "abs": "Absolute Value", "log": "Logarithm", "sqrt": "Square Root"},
                                    "Time-Series Technicals": {"grp_technicals": "All Time-Series Technicals", "delay_5": "5-Day Lag/Delay", "sma_10": "10-Day SMA", "sma_20": "20-Day SMA", "sma_60": "60-Day SMA", "ts_max_20": "20-Day Max", "ts_min_20": "20-Day Min", "rsi_14": "14-Day RSI", "macd_line": "MACD", "vol_10": "10-Day Volatility", "vol_20": "20-Day Volatility", "vol_60": "60-Day Volatility"},
                                    "Cross-Sectional Scoring": {"grp_cross_sectional": "All Cross-Sectional Scoring", "cs_rank_func": "Cross-Sectional Rank"},
                                    "Pricing & Volume": {"grp_pricing": "All Pricing & Volume", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume", "vwap": "VWAP", "trades": "Trades"},
                                    "Fundamental Valuation": {"grp_valuation": "All Fundamental Valuation", "pe_ratio": "PE Ratio", "pb_ratio": "PB Ratio", "ps_ratio": "PS Ratio", "market_cap": "Market Cap"},
                                    "Income Statement": {"grp_income": "All Income Statement", "eps": "EPS", "revenues": "Revenues", "gross_profit": "Gross Profit", "cost_of_revenue": "Cost of Revenue", "operating_income": "Operating Income", "net_income": "Net Income", "interest_expense": "Interest Expense", "research_and_development": "R&D Spend", "shares": "Shares Out"},
                                    "Balance Sheet": {"grp_balance": "All Balance Sheet", "equity": "Total Equity", "assets": "Total Assets", "liabilities": "Total Liabilities", "current_assets": "Current Assets", "current_liabilities": "Current Liab", "inventory": "Inventory"},
                                    "Cash Flow Statement": {"grp_cash": "All Cash Flow Statement", "net_cash_flow": "Net Cash Flow", "operating_cash_flow": "Operating Cash Flow", "dividends_paid": "Dividends Paid"}
                                },
                                selected=["add", "sub", "mul", "div", "close", "pe_ratio", "market_cap"], 
                                multiple=True
                            ),
                            ui.div(
                                ui.input_action_button("btn_select_all_funcs", "Select All", class_="btn-primary btn-sm"),
                                ui.input_action_button("btn_clear_all_funcs", "Clear All", class_="btn-outline-danger btn-sm"),
                                class_="d-flex gap-2 mt-2 align-items-center"
                            )
                        ),
                    ui.input_select("miner_strategy_type", tip("Strategy Directionality", "Strictly isolate Alpha execution boundaries to purely Long or Short vectors."), choices={"ls": "Symmetrical Long/Short", "long": "Long Only", "short": "Short Only", "ss": "Short/Short"}, selected="ls"),
                    ui.input_select("miner_quantile", tip("Evaluation Tail Sizing (Quantiles)", "Zero out ALL middle-range distribution structures dynamically and calculate PNL purely on the 10/20% tails!"), choices={"0": "Global Weighting", "5": "Quintiles (Top/Bottom 20%)", "10": "Deciles (10%)", "20": "Vigintiles (5%)"}, selected="10"),
                    ui.input_numeric("miner_generations", tip("Generational Evolution", "How many times the AI breeds, mutates, and culls the formulas."), value=3, min=1, max=50),
                    ui.input_numeric("miner_pop", tip("Population Map Size", "The number of formulas generated and tested per generation."), value=100, min=10, max=10000),
                    ui.input_switch("miner_monotonicity", tip("Enforce Monotonic Quantiles", "Strictly kills factors where Q1->Q5 returns are not linearly scaling (e.g. U-shaped curves)."), value=True),
                    ui.input_slider("miner_year_range", tip("Timeline Horizon", "Historical window to fetch metrics over."), min=2006, max=2026, value=[2018, 2026], sep=""),
                    ui.input_slider("miner_oos", tip("Out-Of-Sample (OOS) %", "Reserve the newest N% of timeline purely for Validation testing to prevent extreme curve fitting biases."), min=0, max=50, value=20, step=5),
                    col_widths={"sm": (4, 4, 4, 4, 4, 4, 4, 4, 4, 12)},
                    class_="mb-5",
                    fill=False,
                    fillable=False
                ),
                ui.output_ui("miner_action_btn"),
                ui.output_ui("miner_results_ui")
            ),
            ui.nav_panel("📈 Live Execution",
                ui.tags.style("label, .card-header, p { color: #e2e8f0 !important; }"),
                ui.h4("TradeStation Execution Engine", style="color: white; mt-3"),
                ui.p("Connects natively to your TradeStation API token and physically executes the backtested algorithm."),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Broker Config (TradeStation)"),
                        ui.input_switch("ts_is_live", "Live Environment (DANGER!)", value=False),
                        ui.input_action_button("ts_sync_btn", "🔐 Sync TradeStation Tokens", class_="btn-secondary btn-sm mb-2"),
                        ui.output_ui("ts_account_selector"),
                        ui.output_ui("ts_equity_block"),
                        ui.hr(),
                        ui.h5("📊 Intraday PnL Snapshot", style="color: #e2e8f0;"),
                        ui.output_ui("ts_intraday_pnl"),
                        class_="mb-3"
                    ),
                    ui.card(
                        ui.card_header("Execution Payload"),
                        ui.layout_columns(
                            ui.input_select("live_universe", "Universe Target", choices={"R2K": "Russell 2000", "SP500": "S&P 500", "NDX": "Nasdaq 100"}, selected="R2K"),
                            ui.input_select("live_strategy_type", "Strategy Directionality", choices=["Long Only", "Short Only", "Short/Short"], selected="Short Only"),
                            ui.input_numeric("live_portfolio_size", "Portfolio Ticker Size", value=5, min=1, max=500),
                            ui.input_select("live_sizing_strategy", "Position Sizing Strategy", choices=["Dynamic (1/Active)", "Fixed Fractional (1/N)", "Inverse Volatility (ATR)"], selected="Inverse Volatility (ATR)"),
                            ui.input_select("live_liquidity_cap_type", "Liquidity Cap Type", choices=["None", "Max ADV %", "Static $ Ceiling"], selected="Max ADV %"),
                            ui.input_numeric("live_liquidity_cap_value", "Cap Value (% or $)", value=5.0, min=0.1),
                            ui.input_numeric("live_atr_sl_mult", "Risk Multiple (ATR Stop)", value=2.5, min=0.1, max=10.0, step=0.1),
                            ui.input_numeric("live_atr_tp_mult", "Take Profit Multiple (ATR)", value=1.25, min=0.0, max=10.0, step=0.1),
                            col_widths={"sm": (3, 3, 3, 3)},
                            fill=False,
                            fillable=False,
                            class_="mb-2"
                        ),
                        ui.input_switch("live_invert_factor", "Invert Factor (Low to High - Buy Weakest vs Strongest)", value=True),
                        ui.input_text("live_universe_filter", "Ticker Space Filter (Leave blank for all)", value="close >= 2 and close <= 50 and avg_30d_volume >= 500000 and perf_1d < -0.09 and rel_vol_1d > 1 and atr_pct_14 < 0.15"),
                        ui.hr(),
                        ui.input_checkbox("live_use_custom_formula", "Override Alpha Engine with Custom Formula", value=True),
                        ui.input_text("live_custom_formula", "Manual Alpha Formula", value="mul(div(close, sma_10(close)), div(volume, avg_30d_volume))"),
                        # Capital is now purely 100% implicitly synced to account selector
                        ui.p("Total Capital to Allocate: ", ui.strong("AUTO-SYNCED TO SELECTOR EQUITY", style="color: #5eff5e; font-size: 0.9em;")),
                        ui.hr(),
                        ui.p("Target Alpha Formula:"),
                        ui.tags.code(ui.output_text("ts_formula_disp"), style="word-wrap: break-word; color: #5eff5e; background: #222; padding: 4px; display: block; border-radius: 4px;"),
                        ui.p("The algorithm fires asynchronously in the background. Do not close Python terminal.", class_="text-warning mt-2"),
                        ui.input_action_button("ts_arm_btn", "⚡ ARM SYSTEM (Executes 09:30 AM)", class_="btn-danger"),
                        ui.tags.pre(ui.output_text("ts_status_disp"), style="color: #6edff2; font-family: monospace; margin-top: 10px;"),
                        class_="mb-3"
                    ),
                     col_widths={"xs": (12, 12), "sm": (5, 7)},
                    fill=False,
                    fillable=False
                ),
                ui.card(
                    ui.card_header("Background Task Logs", style="color: white;"),
                    ui.tags.pre(ui.output_text("ts_live_logs"), style="color: #10b981; font-family: monospace; white-space: pre-wrap; overflow-y: auto; max-height: 350px;"),
                    style="background: #121212;"
                )
            ),
        ),
    ),
)



# ═══════════════════════════════════════════════════════════════
# Server
# ═══════════════════════════════════════════════════════════════

import threading

def server(input, output, session):
    workflow_result = reactive.Value(None)
    status_msg = reactive.Value("Ready — SEC Engine Synchronized.")
    
    is_running = reactive.Value(False)
    cancel_flag = False

    # Miner State
    miner_results_val = reactive.Value(None)
    miner_status_val = reactive.Value("Ready to mine!")
    miner_running = reactive.Value(False)
    miner_progress_state = {"done": False, "res": None, "msg": "", "error": ""}

    @render.ui
    def miner_action_btn():
        if miner_running.get():
            return ui.div(
                ui.h4("🧬 Genetic Alpha Mining in Progress...", class_="text-info"),
                ui.p(miner_status_val.get(), style="color: #00d4aa;"),
                class_="p-4 text-center mt-5"
            )
            
        if miner_progress_state.get("error"):
            return ui.div(
                ui.h4("⚠️ Engine Initialization Error", class_="text-danger"),
                ui.p(miner_status_val.get(), style="color: #ff4a4a;"),
                ui.input_action_button("btn_run_miner", "Retry Miner", class_="btn-run w-100 mt-4"),
                class_="p-4 text-center mt-5"
            )
            
        elements = [ui.input_action_button("btn_run_miner", "Launch Factor Miner (Genetic Search)", class_="btn-run w-100 mb-4")]
        
        if miner_results_val.get() is None:
            elements.append(
                ui.div(
                    ui.h5("🔬 Understanding Biological Signal Discovery", style="color: #00d4aa; margin-bottom: 15px;"),
                    ui.p(
                        "The Alpha Miner is an institutional-grade Genetic Programming (GP) Engine. It natively extracts " ,
                        ui.span("pure directional predictive correlation", style="color: #4dabf7; font-weight: bold;"),
                        " from millions of generated mathematical trees across 20 years of SEC data.",
                        style="color: #c7c7c7;"
                    ),
                    ui.p(
                        "Unlike the True Portfolio Backtester—which physically forces fixed capital through strict mark-to-market geometric portfolio constraints—the GP engine evaluates ",
                        ui.span("raw mathematical fitness", style="font-style: italic;"),
                        " by rapidly stacking pseudo-arithmetic arrays. This prevents the execution friction from trapping the biological engine in local minimums, allowing it to natively evaluate 10,000+ formulas in seconds!",
                        style="color: #c7c7c7;"
                    ),
                    ui.p(
                        "Configure your array constraints and press Launch to initialize convergence.",
                        style="color: #797979; font-style: italic; margin-top: 15px;"
                    ),
                    class_="p-4 rounded text-start",
                    style="background-color: #1a1e23; border: 1px solid #2d333b; border-left: 4px solid #00d4aa;"
                )
            )
            
        return ui.div(*elements)

    def make_sparkline_svg(data, oos_percent=20, color="#00d4aa", oos_color="#4dabf7"):
        if not data or len(data) < 2: return ""
        min_v, max_v = min(data), max(data)
        rng = max_v - min_v if max_v != min_v else 1
        
        split_idx = int(len(data) * (1 - oos_percent / 100))
        if oos_percent <= 0: split_idx = len(data)
        
        pts_is = []
        pts_oos = []
        width, height = 200, 50
        
        for i, v in enumerate(data):
            x = (i / (len(data)-1)) * width
            y = height - ((v - min_v) / rng) * height
            coord = f"{x},{y}"
            if i <= split_idx:
                pts_is.append(coord)
            if i >= split_idx:
                pts_oos.append(coord)
                
        str_is = " ".join(pts_is)
        str_oos = " ".join(pts_oos)
        
        return f'''
        <svg width="{width}" height="{height}" style="background:rgba(0,0,0,0.15); border: 1px solid rgba(0,212,170,0.2); border-radius: 6px; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.borderColor='#00d4aa'; this.style.boxShadow='0 0 8px rgba(0,212,170,0.4)';" onmouseout="this.style.borderColor='rgba(0,212,170,0.2)'; this.style.boxShadow='none';">
            <defs>
                <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="1.5" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>
            <polyline points="{str_is}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke" />
            <polyline points="{str_oos}" fill="none" stroke="{oos_color}" stroke-width="2.5" vector-effect="non-scaling-stroke" filter="url(#glow)" />
        </svg>
        '''

    @render.ui
    def miner_results_ui():
        results_payload = miner_results_val.get()
        if results_payload and isinstance(results_payload, dict):
            
            def build_cards(res_list, prefix):
                cards = []
                for i, r in enumerate(res_list):
                    oos_badge = ""
                    if 'oos_score' in r:
                        oos_badge = f" | OOS VALIDATION: {r['oos_score']:.4f}"
                    
                    spark_html = ""
                    m_oos = input.miner_oos()
                    m_oos_val = int(m_oos) if m_oos else 20
                    if 'eq_curve' in r and len(r['eq_curve']) > 0:
                        svg_str = make_sparkline_svg(r['eq_curve'], oos_percent=m_oos_val)
                        spark_html = f'''<div style="cursor:pointer; margin-left: 15px;" onclick="Shiny.setInputValue('sparkline_clicked', '{prefix}_{i}', {{priority: 'event'}})" title="Click to View Plotly HD Curve">
                            {svg_str}
                        </div>'''
                    
                    cards.append(
                        ui.div(
                            ui.div(
                                ui.h5(f"Rank {i+1} | IS SCORE: {r.get('fitness_score', 0):.4f}{oos_badge}", style="font-weight: 700; font-size: 0.85rem; color: #00d4aa; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;"),
                                ui.tags.code(r.get('formula', ''), style="font-size: 1.15rem; color: #e5e5e5; font-weight: 500;"),
                                style="flex: 1;"
                            ),
                            ui.HTML(spark_html),
                            style="display: flex; justify-content: space-between; align-items: center; padding: 18px 20px; border-radius: 14px; border: 1px solid var(--border); margin-bottom: 14px; background: rgba(0, 212, 170, 0.08); border-left: 4px solid var(--accent-teal);"
                        )
                    )
                return ui.div(*cards)
            
            return ui.div(
                ui.hr(), 
                ui.h4("🏆 Top Discovered Alpha Formulas", class_="mt-3 mb-3", style="color: white;"),
                ui.navset_card_tab(
                    ui.nav_panel("Top 10 In-Sample", build_cards(results_payload.get("top_is", []), "is")),
                    ui.nav_panel("Top 10 Out-Of-Sample", build_cards(results_payload.get("top_oos", []), "oos")),
                    ui.nav_panel("Top 10 Robust Combined", build_cards(results_payload.get("top_combined", []), "combined"))
                )
            )
        return ui.div()

    @reactive.Effect
    @reactive.event(input.btn_select_all_funcs)
    def _select_all_funcs():
        all_opts = [
            "grp_arithmetic", "grp_technicals", "grp_cross_sectional", 
            "grp_pricing", "grp_valuation", "grp_income", 
            "grp_balance", "grp_cash"
        ]
        ui.update_selectize("miner_funcs", selected=all_opts)

    @reactive.Effect
    @reactive.event(input.btn_clear_all_funcs)
    def _clear_all_funcs():
        ui.update_selectize("miner_funcs", selected=[])

    @reactive.Effect
    @reactive.event(input.miner_funcs)
    def _sanitize_funcs():
        current = list(input.miner_funcs()) if input.miner_funcs() else []
        if not current: return
        
        group_map = {
            "grp_arithmetic": ["add", "sub", "mul", "div", "abs", "log", "sqrt"],
            "grp_technicals": ["delay_5", "sma_10", "sma_20", "sma_60", "ts_max_20", "ts_min_20", "rsi_14", "macd_line", "vol_10", "vol_20", "vol_60"],
            "grp_cross_sectional": ["cs_rank_func"],
            "grp_pricing": ["open", "high", "low", "close", "volume", "vwap", "trades"],
            "grp_valuation": ["pe_ratio", "pb_ratio", "ps_ratio", "market_cap"],
            "grp_income": ["eps", "revenues", "gross_profit", "cost_of_revenue", "operating_income", "net_income", "interest_expense", "research_and_development", "shares"],
            "grp_balance": ["equity", "assets", "liabilities", "current_assets", "current_liabilities", "inventory"],
            "grp_cash": ["net_cash_flow", "operating_cash_flow", "dividends_paid"]
        }
        
        purged = False
        new_sel = current.copy()
        
        for master, children in group_map.items():
            if master in new_sel:
                for child in children:
                    if child in new_sel:
                        new_sel.remove(child)
                        purged = True
                        
        if purged:
            ui.update_selectize("miner_funcs", selected=new_sel)



    @reactive.Effect
    def _poll_miner_thread():
        if not miner_running.get():
            return
        reactive.invalidate_later(0.2)
        miner_status_val.set(miner_progress_state["msg"])
        if miner_progress_state["done"]:
            miner_running.set(False)
            if miner_progress_state["error"]:
                miner_status_val.set(f"Error: {miner_progress_state['error']}")
            else:
                miner_results_val.set(miner_progress_state["res"])
                miner_status_val.set("Complete.")

    @reactive.Effect
    @reactive.event(input.sparkline_clicked)
    def handle_sparkline_click():
        clk_id = input.sparkline_clicked()
        if not clk_id or "_" not in str(clk_id): return
        
        tab_prefix, str_idx = clk_id.split("_")
        idx = int(str_idx)
        
        results_payload = miner_results_val.get()
        if not results_payload or not isinstance(results_payload, dict): return
        
        results = results_payload.get(f"top_{tab_prefix}")
        
        if results and 0 <= idx < len(results):
            r = results[idx]
            eq_curve = r.get("eq_curve", [])
            if not eq_curve: return
            
            m_oos = input.miner_oos()
            m_oos_val = int(m_oos) if m_oos else 20
            split_idx = int(len(eq_curve) * (1 - m_oos_val / 100))
            if m_oos_val <= 0: split_idx = len(eq_curve)
            
            is_y = eq_curve[:split_idx]
            is_x = list(range(split_idx))
            
            oos_y = eq_curve[split_idx-1:]
            oos_x = list(range(split_idx-1, len(eq_curve)))
            
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=is_x, y=is_y, mode='lines', name='In-Sample Bound', line=dict(color='#00d4aa', width=2), fill='tozeroy', fillcolor='rgba(0, 212, 170, 0.1)'))
            
            if m_oos_val > 0 and len(oos_y) > 0:
                fig.add_trace(go.Scatter(x=oos_x, y=oos_y, mode='lines', name='OOS Validation Bounds', line=dict(color='#4dabf7', width=3), fill='tozeroy', fillcolor='rgba(77, 171, 247, 0.25)'))
                
            fig.update_layout(
                title=f"<span style='color:white; font-size: 20px;'>Factor {idx+1} Generational Equity Projection</span><br><span style='font-size:14px; color:#888;'>Ast Vector: {r['formula']}</span>",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified",
                margin=dict(l=40, r=40, t=80, b=40),
                height=450
            )
            html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            ui.modal_show(ui.modal(
                ui.HTML(html_str),
                title=f"Factor PNL Distribution Preview (Rank {idx+1})",
                size="xl",
                easy_close=True,
                footer=ui.modal_button("Close")
            ))

    @reactive.Effect
    @reactive.event(input.btn_run_miner)
    def run_miner():
        if miner_running.get() or is_running():
            return
        miner_running.set(True)
        miner_results_val.set(None)
        miner_progress_state["done"] = False
        miner_progress_state["error"] = ""
        miner_progress_state["msg"] = "Initializing Miner Thread..."
        
        def miner_cb(pct, msg):
            miner_progress_state["msg"] = msg

        m_universe = input.miner_universe()
        m_gens = input.miner_generations()
        m_pop = input.miner_pop()
        m_horizon = int(input.miner_horizon())
        m_fitness = input.miner_fitness()
        m_funcs = input.miner_funcs()
        m_monotonicity = input.miner_monotonicity()
        start_y, end_y = input.miner_year_range()
        m_oos = int(input.miner_oos())
        m_strategy = input.miner_strategy_type()
        m_quantile = int(input.miner_quantile())

        def _bg_miner():
            try:
                miner_cb(5, f"Fetching Baseline DataFrame ({m_universe}... 20+ Years)...")
                tickers = get_latest_constituents(m_universe)[:80] # Proxy subset
                df = tools.fetch_universe_data(tickers, start_y, end_y, force_refresh=False)
                
                miner_cb(20, f"Executing Genetic Evolution (Pop: {m_pop}, Gens: {m_gens}, Horizon: {m_horizon}d)...")
                results = discover_alpha_factors(
                    df, 
                    generations=m_gens, 
                    pop_size=m_pop, 
                    horizon=m_horizon, 
                    fitness_metric=m_fitness, 
                    syntax_set=m_funcs, 
                    enforce_monotonicity=m_monotonicity,
                    oos_percent=m_oos,
                    strategy_dir=m_strategy,
                    eval_quantile=m_quantile,
                    progress_callback=miner_cb
                )
                
                miner_progress_state["res"] = results
            except Exception as e:
                import traceback
                traceback.print_exc()
                miner_progress_state["error"] = str(e)
            finally:
                miner_progress_state["done"] = True

        threading.Thread(target=_bg_miner, daemon=True).start()
    progress_state = {"pct": 0, "msg": "", "done": True, "res": None, "error": None}
    reactive_progress = reactive.Value({"pct": 0, "msg": ""})

    @output
    @render.ui
    def stop_btn_ui():
        return ui.HTML("")  # Removed from sidebar entirely

    @output
    @render.ui
    def modal_progress():
        if not is_running():
            return ui.HTML("")
        
        state = reactive_progress.get()
        pct = state["pct"]
        msg = state["msg"]
        return ui.HTML(f'''
        <div style="margin-bottom: 8px; font-weight: 500; text-align: center;">{msg}</div>
        <div style="text-align: right; font-size: 0.85rem; color: #00d4aa; font-weight: 600; margin-bottom: 4px;">{pct:.0f}%</div>
        <div class="progress" style="height: 24px; border-radius: 6px; background-color: #1a1e23; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);">
          <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: {pct}%; background-color: #00d4aa; color: #ffffff;" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
        ''')

    @reactive.Effect
    @reactive.event(input.stop_btn)
    def handle_stop():
        nonlocal cancel_flag
        cancel_flag = True
        status_msg.set("⚠️ Stopping background engine...")
        ui.modal_remove()

    @reactive.Effect
    @reactive.event(input.cal_cell_click)
    def handle_cal_cell_click():
        click_data = input.cal_cell_click()
        if not click_data: return
        parts = click_data.split('|')
        if len(parts) >= 3:
            date_str = parts[0]
            l_arr = [x for x in parts[1].split(',') if x.strip()]
            s_arr = [x for x in parts[2].split(',') if x.strip()]
            
            content = ui.div(
                ui.h6(f"Longs ({len(l_arr)})", style="color: #00d4aa;"),
                ui.p(", ".join(l_arr) if l_arr else "None", style="word-break: break-all; opacity: 0.9; line-height: 1.6;"),
                ui.hr(style="border-color: #2a2e39;"),
                ui.h6(f"Shorts ({len(s_arr)})", style="color: #ff6b6b;"),
                ui.p(", ".join(s_arr) if s_arr else "None", style="word-break: break-all; opacity: 0.9; line-height: 1.6;")
            )
            ui.modal_show(ui.modal(
                content,
                title=f"Traded Stocks - {date_str}",
                size="l",
                easy_close=True,
                footer=ui.modal_button("Close")
            ))

    @reactive.Effect
    def _poll_bg_thread():
        if not is_running():
            return
            
        reactive.invalidate_later(0.15)  # Poll roughly 6 times per second
        
        # Asymptotic logarithmic interpolation: never visually locks up by smoothly reducing speed as it approaches the 19.9% bounds limit over infinite time scales
        if progress_state["msg"] == "Initializing Backtest Engine..." and progress_state["pct"] < 19.9:
            progress_state["pct"] += (19.9 - progress_state["pct"]) * 0.015
            
        # Sync reactive values to trigger frontend redraw natively!
        curr_state = {"pct": progress_state["pct"], "msg": progress_state["msg"]}
        reactive_progress.set(curr_state)
        
        if progress_state["msg"]:
            status_msg.set(progress_state["msg"])
                
        # Check completion
        if progress_state["done"]:
            is_running.set(False)
            ui.modal_remove()
                
            res_payload = progress_state.get("res") or {}
            
            if progress_state["error"]:
                workflow_result.set({"error": progress_state["error"]})
                status_msg.set(f"❌ {progress_state['error']}")
            elif "error" in res_payload:
                workflow_result.set({"error": res_payload["error"]})
                status_msg.set(f"❌ {res_payload['error']}")
            elif progress_state["res"]:
                workflow_result.set(progress_state["res"])
                n = progress_state["res"].get("metrics", {}).get("n_tickers", "?")
                status_msg.set(f"✅ Analysis complete — {n} stocks processed.")

    @reactive.Effect
    @reactive.event(miner_results_val)
    def update_mined_dropdown():
        res_payload = miner_results_val.get()
        if res_payload and isinstance(res_payload, dict):
            choices = {"None": "None"}
            
            top_combined = res_payload.get("top_combined", [])
            for i, item in enumerate(top_combined):
                f_str = item.get("formula", "")
                choices[f_str] = f"Combined Rank {i+1} : {f_str[:40]}... (IS: {item.get('fitness_score',0):.3f})"
                
            ui.update_select("mined_formula_dropdown", choices=choices)

    @reactive.Effect
    @reactive.event(input.run_btn)
    def run_analysis():
        nonlocal cancel_flag

        if is_running():
            return

        workflow_result.set(None)
        theme_keys = list(input.themes())
        custom_f = input.custom_formula().strip()
        mined_f = input.mined_formula_dropdown()
        
        if mined_f and mined_f != "None":
            custom_formula_opt = mined_f
        elif custom_f:
            custom_formula_opt = custom_f
        else:
            custom_formula_opt = None

        if not theme_keys and not custom_formula_opt:
            status_msg.set("⚠️ Please select at least one factor or provide a custom formula.")
            return
        invert_factor = input.invert_factor()
        univ_filter = input.universe_filter()
        start_year, end_year = input.year_range()
        initial_aum = input.initial_aum()
        rebalance_freq = input.rebalance_freq()
        portfolio_size = float(input.portfolio_size() or 0)
        sizing_strategy = str(input.sizing_strategy())
        liquidity_cap_type = str(input.liquidity_cap_type())
        liquidity_cap_value = float(input.liquidity_cap_value() or 100.0)
        portfolio_sizing_type = input.portfolio_sizing_type()
        max_position_weight_val = float(input.max_position_weight() or 100.0) / 100.0
        strategy_type = input.strategy_type()
        exec_timing = input.execution_timing()
        quantile_split = int(input.quantile_split() or 10)
        enable_cal_val = input.enable_calendar()
        vol_target_val = float(input.vol_target() or 0)
        atr_sl_val = float(input.atr_sl_mult() or 0.0)
        atr_tp_val = float(input.atr_tp_mult() or 0.0)
        slippage_bps_val = float(input.slippage_bps() or 0.0)
        sl_slippage_bps_val = float(input.sl_slippage_bps() or 0.0)

        if custom_formula_opt:
            status_msg.set(f"Initializing Automated Custom Alpha Formula composite...")
        else:
            formatted_str = " + ".join(theme_keys).replace("_", " ").title()
            status_msg.set(f"Initializing {formatted_str} composite backtest...")

        active_universe = input.universe_selection()
        txt_path = os.path.join(_script_dir, ".cache", "constituents", f"{active_universe.lower()}_tickers_latest.txt")
        
        dynamic_tickers = []
        if os.path.exists(txt_path):
            with open(txt_path) as _f:
                dynamic_tickers = [t.strip() for t in _f.readlines() if t.strip()]
        
        cache_needs_rebuild = False
        if not dynamic_tickers:
            cache_needs_rebuild = True
        
        try:
            from constituents.universe_builder import build_constituent_timeline
            timeline = build_constituent_timeline(etf_key=active_universe)
        except Exception:
            timeline = None

        # Reset Background state
        cancel_flag = False
        progress_state["pct"] = 0
        progress_state["msg"] = "Initializing Backtest Engine..."
        progress_state["done"] = False
        progress_state["res"] = None
        progress_state["error"] = None
        
        is_running.set(True)
        
        m = ui.modal(
            ui.output_ui("modal_progress"),
            title="Executing Strategy Toolkit",
            easy_close=False,
            footer=ui.input_action_button("stop_btn", "⏹ Stop Engine", class_="btn-danger")
        )
        ui.modal_show(m)

        def _bg_worker():
            def ui_progress(current, total, ticker, msg=""):
                if cancel_flag:
                    raise InterruptedError("Backtest forcefully cancelled by user.")
                
                # Global mapping bounds mapping the system systematically monotonically:
                global_pct = progress_state["pct"]
                
                if "Cache is empty" in msg or "Discovering" in msg or "SEC XML" in msg or "Mapping" in msg:
                    local_pct = (current / total * 100) if total > 0 else 50
                    global_pct = (local_pct * 0.19) # Give it 0-19% for SEC rebuilding
                elif any(x in msg for x in ["Initializing", "Fetching", "Loaded", "Cache"]):
                    local_pct = (current / total * 100) if total > 0 else 100
                    global_pct = 20 + (local_pct * 0.05)
                elif "multi-factor" in msg or "composite rankings" in msg or "Ranking factor" in msg:
                    local_pct = (current / total * 100) if total > 0 else 100
                    global_pct = 25 + (local_pct * 0.10)
                elif "point-in-time" in msg or "PIT filter" in msg:
                    local_pct = (current / total * 100) if total > 0 else 100
                    global_pct = 35 + (local_pct * 0.10)
                elif "Backtesting day" in msg or "Executing vector" in msg or "Calculating historical" in msg or "Aggregating performance" in msg:
                    local_pct = (current / total * 100) if total > 0 else 0
                    global_pct = 45 + (local_pct * 0.55)
                else:
                    global_pct = progress_state["pct"]
                
                # Prevent backwards sliding during asynchronous event injections
                if global_pct > progress_state["pct"]:
                    progress_state["pct"] = min(global_pct, 100)
                progress_state["msg"] = msg if msg else f"📡 {current}/{total}: {ticker}"

            proxy_map = {"R2K": "IWM", "SP500": "IVV", "NDX": "QQQ"}
            benchmark_ticker = proxy_map.get(active_universe, "IWM")

            try:
                nonlocal dynamic_tickers, timeline
                if cache_needs_rebuild:
                    ui_progress(0, 0, "", "⚠️ Cache is empty! Forcing a full rebuild from SEC EDGAR + Massive APIs. This will take ~5-10 minutes...")
                    from constituents.universe_builder import build_historical_constituents, get_latest_constituents, build_constituent_timeline
                    
                    master_df = build_historical_constituents(
                        etf_key=active_universe,
                        max_filings=5,
                        use_known=(active_universe == "R2K"),
                        progress_callback=ui_progress,
                        force_refresh=True
                    )
                    
                    timeline = build_constituent_timeline(master_df, etf_key=active_universe)
                    dynamic_tickers = get_latest_constituents(active_universe)
                    
                    if not dynamic_tickers:
                        raise ValueError(f"Failed to rebuild SEC data for {active_universe}")

                import json
                
                # INTERCEPT ARGUMENTS
                intercepted_args = {
                    "tickers": len(dynamic_tickers),
                    "themes": theme_keys,
                    "custom_formula": custom_formula_opt,
                    "portfolio_size": portfolio_size,
                    "sizing_strategy": sizing_strategy,
                    "liquidity_cap_type": liquidity_cap_type,
                    "liquidity_cap_value": liquidity_cap_value,
                    "portfolio_sizing_type": portfolio_sizing_type,
                    "max_position_weight": max_position_weight_val,
                    "strategy_type": strategy_type,
                    "execution_timing": exec_timing,
                    "start_year": start_year,
                    "end_year": end_year,
                    "invert_factor": invert_factor,
                    "rebalance_freq": rebalance_freq,
                    "initial_aum": initial_aum,
                    "benchmark_ticker": benchmark_ticker,
                    "universe_filter": univ_filter,
                    "quantiles": quantile_split,
                    "enable_calendar": enable_cal_val,
                    "vol_target": vol_target_val,
                    "atr_sl_mult": atr_sl_val,
                    "atr_tp_mult": atr_tp_val,
                    "slippage_bps": slippage_bps_val,
                    "sl_slippage_bps": sl_slippage_bps_val,
                }
                with open("intercepted_args.json", "w") as f:
                    json.dump(intercepted_args, f, indent=4)
                    
                res_str = tools.run_cross_sectional_backtest(
                    tickers=dynamic_tickers,
                    themes=theme_keys,
                    custom_formula=custom_formula_opt,
                    portfolio_size=portfolio_size,
                    sizing_strategy=sizing_strategy,
                    liquidity_cap_type=liquidity_cap_type,
                    liquidity_cap_value=liquidity_cap_value,
                    portfolio_sizing_type=portfolio_sizing_type,
                    max_position_weight=max_position_weight_val,
                    strategy_type=strategy_type,
                    execution_timing=exec_timing,
                    start_year=start_year,
                    end_year=end_year,
                    invert_factor=invert_factor,
                    rebalance_freq=rebalance_freq,
                    initial_aum=initial_aum,
                    progress_callback=ui_progress,
                    constituent_timeline=timeline,
                    benchmark_ticker=benchmark_ticker,
                    universe_filter=univ_filter,
                    quantiles=quantile_split,
                    enable_calendar=enable_cal_val,
                    vol_target=vol_target_val,
                    atr_sl_mult=atr_sl_val,
                    atr_tp_mult=atr_tp_val,
                    slippage_bps=slippage_bps_val,
                    sl_slippage_bps=sl_slippage_bps_val,
                )
                progress_state["res"] = json.loads(res_str)
            except Exception as e:
                progress_state["error"] = str(e)
            finally:
                progress_state["done"] = True

        threading.Thread(target=_bg_worker, daemon=True).start()

    @output
    @render.ui
    def status_text():
        return ui.div(ui.p(status_msg()), class_="status-text")

    @output
    @render.text
    def metric_universe_size():
        res = workflow_result.get()
        if res and "metrics" in res:
            return f"{res['metrics'].get('n_tickers', '?')}"
        
        active_universe = input.universe_selection()
        txt_path = os.path.join(_script_dir, ".cache", "constituents", f"{active_universe.lower()}_tickers_latest.txt")
        if os.path.exists(txt_path):
            with open(txt_path) as _f:
                return str(len([t for t in _f.readlines() if t.strip()]))
        return "Not Cached"

    @output
    @render.ui
    def calendar_ui():
        res = workflow_result.get()
        if res is None:
            return ui.HTML('<div style="color: #f59e0b; padding: 20px;">Run a backtest to populate calendar.</div>')
        if "error" in res:
            return ui.HTML(f'<div style="color: #ef4444; padding: 20px;">⚠️ {res["error"]}</div>')

        html_str = res.get("metrics", {}).get("calendar_html", "")
        if not html_str:
            return ui.HTML('<div style="color: #f59e0b; padding: 20px;">No Calendar Data Available.</div>')
            
        return ui.HTML(html_str)

    @output
    @render.ui
    def value_boxes():
        res = workflow_result()
        if res is None:
            return ui.div(
                ui.HTML("""
                <div style="text-align: center; padding: 60px 20px; color: #555;">
                    <div style="font-size: 3rem; margin-bottom: 12px;">📈</div>
                    <div style="font-size: 1.1rem; font-weight: 500;">Select a factor and click Run</div>
                    <div style="font-size: 0.85rem; margin-top: 6px; color: #444;">
                        The engine will fetch, score, rank, and construct a long/short portfolio
                    </div>
                </div>
                """),
            )

        if "error" in res:
            return ui.div(
                ui.HTML(f'<div style="color: #ef4444; padding: 20px; font-weight: 500;">⚠️ {res["error"]}</div>'),
            )

        m = res.get("metrics", {})
        if not m:
            return ui.HTML('<div style="color: #f59e0b;">No metrics returned. See Agent Logs.</div>')

        def _color(val, good_threshold, bad_threshold, higher_is_better=True):
            if not isinstance(val, (int, float)):
                return "#2d3436"
            if higher_is_better:
                return "#00d4aa" if val >= good_threshold else ("#f59e0b" if val >= bad_threshold else "#ef4444")
            else:
                return "#00d4aa" if val <= good_threshold else ("#f59e0b" if val <= bad_threshold else "#ef4444")

        def _fmt(val):
            if isinstance(val, (int, float)):
                return f"{val:.3f}"
            return str(val)

        def _fmt_pct(val):
            if isinstance(val, (int, float)):
                return f"{val*100:.1f}%"
            return str(val)
            
        def _fmt_doll(val):
            if isinstance(val, (int, float)):
                return f"${val:,.0f}"
            return str(val)

        return ui.div(
            ui.h6("Strategy Performance Metrics", style="color: #ffffff; margin-bottom: 12px; margin-top: 5px; font-weight: 600; font-size: 1.05rem;"),
            ui.layout_columns(
                ui.value_box(tip("Net Profit ($)", "Cumulative absolute dollar simulation growth over the tested period."), _fmt_doll(m.get('total_ret_usd', 'N/A')),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Strategy Ann. Ret", "The geometric average yearly return."), _fmt_pct(m.get('ann_port_return', 'N/A')),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Strategy Ann Vol", "Annualized standard deviation outlining generalized expected risk."), _fmt_pct(m.get('ann_vol', 'N/A')),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Strategy Sharpe", "Risk-adjusted return (Annualized Return divided by Volatility)."), _fmt(m.get('sharpe_ratio', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('sharpe_ratio'), 0.5, 0), fg="white")),
                ui.value_box(tip("Strategy Sortino", "Downside risk-adjusted return (Annualized Return divided by Downside Deviation)."), _fmt(m.get('sortino_ratio', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('sortino_ratio'), 0.5, 0), fg="white")),
                ui.value_box(tip("Strategy Calmar", "Drawdown risk-adjusted return (Annualized Return divided by Max Drawdown)."), _fmt(m.get('calmar_ratio', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('calmar_ratio'), 1.0, 0), fg="white")),
                ui.value_box(tip("Strategy Max DD", "Maximum peak-to-trough percentage capital destruction."), _fmt_pct(m.get('max_drawdown', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('max_drawdown'), -0.15, -0.25, False), fg="white")),
                gap="12px"
            ),
            
            ui.h6(f"Trade Analytics ({'Positional' if input.metric_scope() else 'Daily'} Scope)", style="color: #ffffff; margin-bottom: 12px; margin-top: 20px; font-weight: 600; font-size: 1.05rem;"),
            ui.layout_columns(
                ui.value_box(tip("Win Rate", "Percentage of trades/days where the portfolio generated positive P&L."), _fmt_pct(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}win_rate", "N/A")),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Avg Win", "Average profit generated on winning trades/days."), _fmt_pct(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}avg_win", "N/A")),
                             theme=ui.value_box_theme(bg=_color(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}avg_win", 0), 0.05, 0), fg="white")),
                ui.value_box(tip("Avg Loss", "Average loss generated on losing trades/days."), _fmt_pct(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}avg_loss", "N/A")),
                             theme=ui.value_box_theme(bg=_color(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}avg_loss", 0), 0, -0.05), fg="white")),
                ui.value_box(tip("Max Win", "Largest single profit generated."), _fmt_pct(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}max_win", "N/A")),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Max Loss", "Largest single loss generated."), _fmt_pct(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}max_loss", "N/A")),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Expected Value", "Statistical Expected Value (EV) per trade/day based on Win Rate and Avg Win/Loss."), _fmt_pct(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}expected_value", "N/A")),
                             theme=ui.value_box_theme(bg=_color(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}expected_value", 0), 0.01, -0.01), fg="white")),
                ui.value_box(tip("Profit Factor", "Gross Profit divided by Gross Loss. Value > 1 indicates profitability."), _fmt(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}profit_factor", "N/A")),
                             theme=ui.value_box_theme(bg=_color(m.get(f"{'pos_' if input.metric_scope() else 'daily_'}profit_factor", 0), 1.5, 1.0), fg="white")),
                ui.value_box(tip("Stop Loss Hits", "Total number of times a position pierced the ATR Stop Loss boundary."), f"{m.get('sl_hits', 'N/A')}",
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Take Profit Hits", "Total number of times a position hit the Take Profit boundary intraday."), f"{m.get('tp_hits', 'N/A')}",
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("EOD Closes (Positive PNL)", "Total number of positions closed naturally at MOC in the green."), f"{m.get('eod_pos_hits', 'N/A')}",
                             theme=ui.value_box_theme(bg=_color(1, 0, 0), fg="white")),
                ui.value_box(tip("EOD Closes (Negative PNL)", "Total number of positions closed naturally at MOC in the red."), f"{m.get('eod_neg_hits', 'N/A')}",
                             theme=ui.value_box_theme(bg=_color(-1, 0, 0), fg="white")),
                ui.value_box(tip("Max Win Streak", "Highest number of consecutive winning days."), f"{m.get('max_win_streak', 'N/A')}",
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Avg Win Streak", "Average number of consecutive winning days."), f"{m.get('avg_win_streak', 'N/A')}",
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Max Loss Streak", "Highest number of consecutive losing days."), f"{m.get('max_loss_streak', 'N/A')}",
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Avg Loss Streak", "Average number of consecutive losing days."), f"{m.get('avg_loss_streak', 'N/A')}",
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                gap="12px"
            ),
            
            ui.h6(f"Index Benchmark Metrics ({input.universe_selection()})", style="color: #ffffff; margin-bottom: 12px; margin-top: 20px; font-weight: 600; font-size: 1.05rem;"),
            ui.layout_columns(
                ui.value_box(tip("Index Total Ret", "Cumulative compound capitalization growth over the benchmark's tested period."), _fmt_pct(m.get('total_bench_return', 'N/A')), 
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Index Ann. Ret", "The geometric average yearly benchmark return."), _fmt_pct(m.get('ann_bench_return', 'N/A')), 
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Index Sharpe", "Risk-adjusted return (Annualized Return divided by Volatility)."), _fmt(m.get('bench_sharpe', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('bench_sharpe', 0), 0.5, 0), fg="white")),
                ui.value_box(tip("Index Max DD", "Maximum peak-to-trough percentage capital destruction."), _fmt_pct(m.get('bench_max_dd', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('bench_max_dd', 0), -0.15, -0.25, False), fg="white")),
                gap="12px"
            ),
            
            ui.h6("Factor & Execution Analytics", style="color: #ffffff; margin-bottom: 12px; margin-top: 20px; font-weight: 600; font-size: 1.05rem;"),
            ui.layout_columns(
                ui.value_box(tip("Ann. Alpha", "Annualized excess return generated above the underlying benchmark index."), _fmt_pct(m.get('ann_alpha', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('ann_alpha'), 0.01, -0.01), fg="white")),
                ui.value_box(tip("Portfolio Beta", "Systematic relative volatility mapping structural correlation to the index."), _fmt(m.get('port_beta', 'N/A')),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Mean IC", "Information Coefficient. The average rank correlation between predictions and actual forward returns."), _fmt(m.get('mean_ic', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('mean_ic'), 0.02, 0), fg="white")),
                ui.value_box(tip("IC IR", "Information Ratio of the IC. Determines the consistency of the predictive edge."), _fmt(m.get('ic_ir', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('ic_ir'), 0.3, 0), fg="white")),
                ui.value_box(tip("Avg. Turnover", "Average fraction of active capital rotated strictly per-rebalancing event. \nFormula: (Σ|Δ Position| / 2) / Σ|Gross Exposure|"), _fmt_pct(m.get('avg_turnover', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('avg_turnover'), 0.30, 0.80, False), fg="white")),
                ui.value_box(tip("Universe Size", "Total number of active assets analyzed in the final rebalance."), f"{m.get('n_tickers', '?')}",
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                gap="12px"
            ),
            
            ui.h6(f"Live Target Execution (Latest Date: {m.get('latest_date', 'N/A')})", style="color: #ffffff; margin-bottom: 12px; margin-top: 20px; font-weight: 600; font-size: 1.05rem;"),
            ui.layout_columns(
                ui.div(ui.value_box("Buy (Long Leg)", "", ui.tags.div(", ".join(m.get('current_longs', [])) if m.get('current_longs') else "None", style="font-size: 0.95rem; word-break: break-word; color: #ffffff; font-weight: 600;"), theme=ui.value_box_theme(bg="#1a1e28", fg="white")), style="border: 2px solid #00d4aa; border-radius: 10px; overflow: hidden;"),
                ui.div(ui.value_box("Sell (Short Leg)", "", ui.tags.div(", ".join(m.get('current_shorts', [])) if m.get('current_shorts') else "None", style="font-size: 0.95rem; word-break: break-word; color: #ffffff; font-weight: 600;"), theme=ui.value_box_theme(bg="#1a1e28", fg="white")), style="border: 2px solid #ff4a4a; border-radius: 10px; overflow: hidden;"),
                gap="12px"
            )
        )

    import plotly.io as pio
    import base64

    @output
    @render.ui
    def plots_ui():
        res = workflow_result()
        if not res or "error" in res or not res.get("plots"):
            return ui.HTML("")

        def _make_iframe(key, height="320px"):
            chart_json = res["plots"].get(key)
            if not chart_json: return ""
            fig = pio.from_json(chart_json)
            raw_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            raw_html = raw_html.replace("<head>", "<head><style>body { margin: 0; background-color: #111111 !important; }</style>")
            b64 = base64.b64encode(raw_html.encode("utf-8")).decode("utf-8")
            return f'<iframe src="data:text/html;base64,{b64}" style="width: 100%; height: {height}; border: none; overflow: hidden; border-radius: 8px;" scrolling="no"></iframe>'

        parts = []
        
        eq = _make_iframe("equity_json", "470px")
        if eq: parts.append(ui.HTML(eq))
        
        yr = _make_iframe("yearly_json", "320px")
        mo = _make_iframe("monthly_json", "320px")
        dy = _make_iframe("daily_json", "320px")
        
        if yr or mo or dy:
            tabs = ui.navset_pill(
                ui.nav_panel("Yearly", ui.HTML(yr)),
                ui.nav_panel("Monthly", ui.HTML(mo)),
                ui.nav_panel("Daily", ui.HTML(dy)),
            )
            parts.append(tabs)

        for key in ["quintile_json", "ic_json", "drawdown_json"]:
            h = "380px" if key in ["quintile_json", "ic_json"] else "320px"
            html = _make_iframe(key, h)
            if html: parts.append(ui.HTML(html))

        return ui.div(*parts, style="margin-top: 16px; display: flex; flex-direction: column; gap: 24px;")


    
    # ═══════════════════════════════════════════════════════════════
    # Live TradeStation Server Logic
    # ═══════════════════════════════════════════════════════════════
    
    @render.text
    def ts_formula_disp():
        if input.live_use_custom_formula():
            return input.live_custom_formula()
        return "N/A"
        
    ts_account_choices = reactive.value({})
    ts_accounts_data = reactive.value([])
    
    @reactive.effect
    @reactive.event(input.ts_sync_btn, input.ts_is_live)
    def sync_tradestation():
        try:
            ts = TradeStationClient(is_live=input.ts_is_live())
            accs = ts.get_accounts()
            # mapping
            acc_opts = {a["AccountID"]: f"{a['AccountID']} ({a['AccountType']})" for a in accs}
            ts_account_choices.set(acc_opts)
            ts_accounts_data.set(accs)
        except Exception as e:
            err_str = str(e)
            ts_account_choices.set({"error": f"API Error: {err_str}"})
            
            # Auto-open browser login if the refresh token is missing or unauthorized
            if "refresh" in err_str.lower() or "unauthorized" in err_str.lower():
                import webbrowser
                webbrowser.open("http://localhost:8080/login")
            
    @render.ui
    def ts_account_selector():
        opts = ts_account_choices.get()
        if not opts:
            return ui.p("Click Sync TradeStation Tokens", class_="text-muted")
        
        # Default to the SIM account if it exists
        def_sel = None
        for k in opts.keys():
            if "SIM3068044M" in k:
                def_sel = k
                break
                
        return ui.input_select("ts_account", "Account ID", choices=opts, selected=def_sel)
        
    @render.ui
    def ts_equity_block():
        raw_act = input.ts_account()
        if not raw_act or raw_act == "error":
            return ui.p("")
            
        # UI dropdown passes the label in this text context. Strip it strictly to AccountID:
        act = raw_act.split(" ")[0]
        
        try:
            ts = TradeStationClient(is_live=input.ts_is_live())
            equity = ts.get_equity(act)
            
            try:
                p_size = float(str(input.live_port_size()).strip() or "100000") if hasattr(input, "live_port_size") else 100000.0
            except Exception:
                p_size = 100000.0
                
            return ui.div(
                ui.h5("Real-time Equity"),
                ui.h3(f"${equity:,.2f}", style="color: #6edff2;"),
                ui.p(f"Allocation per Ticker: ${(equity * 0.95)/p_size:,.2f}", class_="text-muted")
            )
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            return ui.p(f"Fetch failed: {e} | Trace: {err}", class_="text-danger", style="font-size: 0.8rem; white-space: pre-wrap;")

    @render.ui
    def ts_intraday_pnl():
        reactive.invalidate_later(30)  # Auto-refresh every 30 seconds
        
        try:
            raw_act = input.ts_account()
            if not raw_act or raw_act == "error":
                return ui.p("No account selected.", class_="text-muted")
            
            act = raw_act.split(" ")[0]
            
            # Read current targets from the log file
            import re
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_execution_status.log")
            if not os.path.exists(log_path):
                return ui.p("No execution log found. Arm the system first.", class_="text-muted")
            
            with open(log_path, "r") as f:
                log_lines = f.readlines()
            
            # Extract the most recent target tickers from the log
            target_tickers = []
            for line in reversed(log_lines):
                match = re.search(r'SELL SHORT: (\w+)', line)
                if match:
                    target_tickers.append(match.group(1))
                if 'TARGET EXECUTION PAYLOAD' in line:
                    break
            
            if not target_tickers:
                # Also try BUY targets
                for line in reversed(log_lines):
                    match = re.search(r'BUY: (\w+)', line)
                    if match:
                        target_tickers.append(match.group(1))
                    if 'TARGET EXECUTION PAYLOAD' in line:
                        break
            
            target_tickers = list(reversed(target_tickers))
            
            if not target_tickers:
                return ui.p("No targets found in log. Arm the system first.", class_="text-muted")
            
            # Get config from inputs
            try:
                p_size = int(input.live_portfolio_size() or 25)
            except Exception:
                p_size = 25
            risk_mult = float(input.live_atr_sl_mult() or 2.5)
            tp_mult = float(input.live_atr_tp_mult() or 1.25)
            direction = str(input.live_strategy_type())
            
            ts_client = TradeStationClient(is_live=input.ts_is_live())
            equity = ts_client.get_equity(act)
            alloc = (equity * 0.95) / p_size
            
            from tools import _fetch_single_ticker
            import numpy as np
            from zoneinfo import ZoneInfo
            import datetime as dt
            ET = ZoneInfo("America/New_York")
            now_et = dt.datetime.now(tz=ET)
            today_str = now_et.strftime("%Y-%m-%d")
            today_prefix = now_et.strftime("%Y-%m-%dT")
            
            # Pre-calculate ATR for each ticker from daily data
            atr_map = {}
            for ticker in target_tickers:
                try:
                    df = _fetch_single_ticker(ticker, "2026-04-01", now_et.strftime("%Y-%m-%d"))
                    if df.empty:
                        continue
                    df = df.sort_values("date").reset_index(drop=True)
                    prev_c = df["close"].shift(1)
                    tr_vals = pd.concat([df["high"] - df["low"], (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
                    atr_14 = tr_vals.rolling(14).mean()
                    atr_map[ticker] = atr_14.iloc[-1] / df["close"].iloc[-1]
                except Exception:
                    atr_map[ticker] = 0.05
            
            rows = []
            total_pnl = 0.0
            
            for ticker in target_tickers:
                try:
                    bars = ts_client.get_intraday_bars(ticker, interval=1, days_back=1)
                    if not bars:
                        rows.append(ui.tags.tr(
                            ui.tags.td(ticker), ui.tags.td("No data", colspan="6")
                        ))
                        continue
                    
                    t_open = None
                    t_high = 0
                    t_low = 99999
                    current_price = float(bars[-1].get("Close", 0))
                    
                    for b in bars:
                        time_str = b.get('TimeStamp', '')
                        if today_prefix not in time_str:
                            continue
                        hour_min = time_str.split("T")[1][:5]
                        if t_open is None and hour_min >= "13:30":
                            t_open = float(b.get("Open"))
                        if t_open is not None and hour_min >= "13:30":
                            h = float(b.get("High"))
                            l = float(b.get("Low"))
                            if h > t_high: t_high = h
                            if l < t_low: t_low = l
                    
                    if t_open is None:
                        rows.append(ui.tags.tr(
                            ui.tags.td(ticker, style="font-weight:bold;"), ui.tags.td("Awaiting 9:30 AM open", colspan="6", style="color:#fbbf24;")
                        ))
                        continue
                    
                    atr_pct = atr_map.get(ticker, 0.05)
                    qty = int(alloc / t_open)
                    is_short = direction in ["Short Only", "Short/Short"]
                    
                    if is_short:
                        stop_price = t_open * (1.0 + atr_pct * risk_mult)
                        target_price = t_open * (1.0 - atr_pct * tp_mult)
                        sl_hit = t_high >= stop_price
                        tp_hit = t_low <= target_price
                    else:
                        stop_price = t_open * (1.0 - atr_pct * risk_mult)
                        target_price = t_open * (1.0 + atr_pct * tp_mult)
                        sl_hit = t_low <= stop_price
                        tp_hit = t_high >= target_price
                    
                    if sl_hit:
                        exit_price = stop_price
                        status = "🛑 STOPPED"
                        status_color = "#ef4444;"
                    elif tp_hit:
                        exit_price = target_price
                        status = "✅ TP HIT"
                        status_color = "#10b981;"
                    else:
                        exit_price = current_price
                        status = "📊 OPEN"
                        status_color = "#6edff2;"
                    
                    if is_short:
                        pnl = qty * (t_open - exit_price)
                    else:
                        pnl = qty * (exit_price - t_open)
                    
                    total_pnl += pnl
                    pnl_color = "#10b981" if pnl >= 0 else "#ef4444"
                    
                    rows.append(ui.tags.tr(
                        ui.tags.td(ticker, style="font-weight:bold; color:#e2e8f0;"),
                        ui.tags.td(f"${t_open:.2f}", style="color:#94a3b8;"),
                        ui.tags.td(f"${stop_price:.2f}", style="color:#ef4444;"),
                        ui.tags.td(f"${target_price:.2f}", style="color:#10b981;"),
                        ui.tags.td(f"${exit_price:.2f}", style="color:#e2e8f0;"),
                        ui.tags.td(f"${pnl:+.2f}", style=f"color:{pnl_color}; font-weight:bold;"),
                        ui.tags.td(status, style=f"color:{status_color}")
                    ))
                except Exception as e:
                    rows.append(ui.tags.tr(
                        ui.tags.td(ticker), ui.tags.td(str(e)[:40], colspan="6", style="color:#ef4444;")
                    ))
            
            pnl_color = "#10b981" if total_pnl >= 0 else "#ef4444"
            pnl_pct = total_pnl / equity * 100 if equity > 0 else 0
            
            table = ui.tags.table(
                ui.tags.thead(ui.tags.tr(
                    ui.tags.th("Ticker"), ui.tags.th("Entry"), ui.tags.th("SL"),
                    ui.tags.th("TP"), ui.tags.th("Current"), ui.tags.th("PnL"), ui.tags.th("Status")
                )),
                ui.tags.tbody(*rows),
                ui.tags.tfoot(ui.tags.tr(
                    ui.tags.td("", colspan="5", style="text-align:right; font-weight:bold; color:#e2e8f0;"),
                    ui.tags.td(f"${total_pnl:+.2f}", style=f"font-weight:bold; font-size:1.1em; color:{pnl_color};"),
                    ui.tags.td(f"({pnl_pct:+.2f}%)", style=f"color:{pnl_color};")
                )),
                style="width:100%; font-size:0.85rem; border-collapse:collapse;",
                class_="table table-dark table-sm table-striped"
            )
            
            return ui.div(
                table,
                ui.p(f"Last updated: {now_et.strftime('%I:%M:%S %p ET')} | Auto-refreshes every 30s", 
                     style="color:#64748b; font-size:0.75rem; margin-top:6px;")
            )
        except Exception as e:
            return ui.p(f"PnL fetch error: {e}", class_="text-danger", style="font-size:0.8rem;")

    ts_execution_status = reactive.value("")

    @reactive.effect
    @reactive.event(input.ts_arm_btn)
    def arm_live_system():
        raw_act = input.ts_account()
        if not raw_act or raw_act == "error":
            ts_execution_status.set("Account not configured!")
            return
            
        act = raw_act.split(" ")[0]
            
        formula = input.live_custom_formula() if input.live_use_custom_formula() else "Returns"
        filt = input.live_universe_filter()
        
        try:
            port_size = str(int(input.live_portfolio_size() or 5))
        except Exception:
            port_size = "3"
            
        risk_mult = input.live_atr_sl_mult()
        is_live = input.ts_is_live()
        
        # Spawn the Background Subprocess safely passing parameters!
        try:
            import sys
            import os
            import subprocess
            
            # Kill any existing instances of live_executor.py to prevent concurrent duplicate runs
            try:
                subprocess.run(["pkill", "-f", "live_executor.py"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except Exception:
                pass
                
            # Reconstruct the exact path using sys.executable to ensure virtualenv consistency
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_executor.py")
            cmd = [
                sys.executable, script_path,
                "--account", str(act),
                "--formula", str(formula),
                "--filter", str(filt) if filt else "close > 0",
                "--portfolio_size", str(port_size),
                "--sizing_strategy", f'"{str(input.live_sizing_strategy())}"',
                "--liquidity_cap_type", f'"{str(input.live_liquidity_cap_type())}"',
                "--liquidity_cap_value", str(input.live_liquidity_cap_value() or 100.0),
                "--risk_mult", str(risk_mult),
                "--tp_mult", str(input.live_atr_tp_mult()),
                "--is_live", str(is_live),
                "--universe", str(input.live_universe()),
                "--invert", str(input.live_invert_factor()),
                "--direction", str(input.live_strategy_type()),
                "--themes", ",".join(input.themes()) if input.themes() else "",
                "--start_year", str(input.year_range()[0])
            ]
            
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ts_execution_status.set(f"✅ ARMED! Background Bot is waiting for 09:30 AM.")
        except Exception as e:
            ts_execution_status.set(f"❌ Spawning Error: {e}")
            
    @render.text
    def ts_status_disp():
        return ts_execution_status.get()
        
    def poll_logs():
        try:
            with open("live_execution_status.log", "r") as f:
                lines = f.readlines()
                return "".join(lines[-20:]) # Last 20 lines
        except:
            return "(No Live Logs Found Yet)"
            
    # Polling effect
    @render.text
    def ts_live_logs():
        # Triggers UI refresh purely to read text
        reactive.invalidate_later(2) 
        return poll_logs()

app = App(app_ui, server)

