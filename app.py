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

# Primary source: SEC EDGAR N-PORT extracted ticker list
_ticker_cache = os.path.join(_script_dir, ".cache", "constituents", "r2k_tickers_latest.txt")
if os.path.exists(_ticker_cache):
    with open(_ticker_cache) as _f:
        ALL_TICKERS = [t.strip() for t in _f.readlines() if t.strip()]
else:
    # Fallback: approximate CSV
    try:
        tickers_df = pd.read_csv(os.path.join(_script_dir, "r2k_approx.csv"), header=None)
        ALL_TICKERS = [t for t in tickers_df[0].dropna().tolist() if isinstance(t, str) and t.isalpha() and len(t) <= 5]
    except Exception:
        ALL_TICKERS = []

UNIVERSE_SIZE = len(ALL_TICKERS)
_TICKER_SOURCE = "SEC EDGAR N-PORT" if os.path.exists(_ticker_cache) else "Approximate CSV"

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

/* ── Agent Cards ── */
.agent-card {
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid var(--border);
    margin-bottom: 14px;
    transition: transform 0.2s ease;
    position: relative;
    overflow: hidden;
}
.agent-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
}
.agent-card:hover { transform: translateX(4px); }
.agent-hypothesis { background: rgba(59, 130, 246, 0.08); }
.agent-hypothesis::before { background: var(--accent-blue); }
.agent-quant { background: rgba(0, 212, 170, 0.08); }
.agent-quant::before { background: var(--accent-teal); }
.agent-risk { background: rgba(245, 158, 11, 0.08); }
.agent-risk::before { background: var(--accent-amber); }
.agent-label {
    font-weight: 700;
    font-size: 0.85rem;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    opacity: 0.95;
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
.selectize-input {
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
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
                class_="config-section",
            ),

            # Factor Config
            ui.div(
                ui.h6("Factor Configuration"),
                ui.input_selectize("themes", tip("Base Analytics", "Pre-configured factors to evaluate cross-sectionally."), choices=list(THEMES.keys()), multiple=True, selected=["momentum_1m"]),
                ui.input_text("custom_formula", tip("🧪 Custom GP Alpha Formula", "Inject an exact mathematical PyGP syntax tree to bypass standard themes."), placeholder="e.g. sqrt(div(Open, Low)) (Overrides themes)"),
                ui.input_select("mined_formula_dropdown", tip("🧬 Selected Mined Alpha", "Pull winning formulas from the Alpha Miner output."), choices={"None": "None"}, selected="None"),
                ui.input_switch("invert_factor", tip("Invert Factor (Low to High)", "Flips the strategy to buy the lowest scoring stocks instead of the highest."), value=False),
                class_="config-section",
            ),

            # Portfolio Config
            ui.div(
                ui.h6("Portfolio Configuration"),
                ui.input_select("strategy_type", tip("Strategy Type", "Dictates capitalization allocation (Long/Short neutral, or directional Long/Short only)."), choices=["Long/Short", "Long Only", "Short Only"]),
                ui.input_select("quantile_split", tip("Analysis Quantiles", "Splits the ranked universe into N fractional buckets to evaluate top vs bottom tier spreads."), choices={"3": "Tertiles (3)", "4": "Quartiles (4)", "5": "Quintiles (5)", "10": "Deciles (10)"}, selected="5"),
                ui.input_select("portfolio_sizing_type", tip("Portfolio Sizing Logic", "Allocate capital by fixed asset bounds or by dynamic universe percentages."), choices=["Absolute Count", "Percentage"], selected="Absolute Count"),
                ui.input_numeric("portfolio_size", tip("Portfolio Size / Percent limit", "Enter absolute count of assets (e.g., 100) or total universe percentage (e.g., 20)"), value=100),
                ui.input_numeric("initial_aum", tip("Initial AUM ($)", "Starting simulation capital dictating absolute dollar returns."), value=1000000),
                ui.input_slider("year_range", tip("Analysis Period", "Historical year boundaries for testing."), min=2021, max=datetime.now().year, value=(2021, datetime.now().year), sep=""),
                ui.input_select("rebalance_freq", tip("Rebalance Frequency", "How often the algorithm recalculates ranks and shifts portfolio capital."), 
                                choices={"D": "Daily", "M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}, 
                                selected="D"),
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
            ui.nav_panel("🤖 Agent Logs",
                ui.output_ui("agent_logs"),
            ),
            ui.nav_panel("🧬 AI Alpha Miner",
                ui.output_ui("miner_ui"),
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
    status_msg = reactive.Value(f"Ready — {UNIVERSE_SIZE} tickers loaded.")
    
    is_running = reactive.Value(False)
    cancel_flag = False

    # Miner State
    miner_results_val = reactive.Value(None)
    miner_status_val = reactive.Value("Ready to mine!")
    miner_running = reactive.Value(False)
    miner_progress_state = {"done": False, "res": None, "msg": "", "error": ""}

    @render.ui
    def miner_ui():
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
        
        cards = [
            ui.h4("Automated Formulaic Factor Discovery", style="color: white;"),
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
                ui.input_select("miner_funcs", tip("Theoretical Syntax Set", "Restricts the AI to only use specific mathematical functions."), choices={"all": "Unrestricted Matrix (All)", "linear": "Linear Arithmetic (+, -, *, /)", "cross_sectional": "Cross-Sectional Scoring (Rank)", "technical": "Time-Series Technicals (SMA, Delay)"}, selected="all"),
                ui.input_numeric("miner_generations", tip("Generational Evolution limits", "How many times the AI breeds, mutates, and culls the formulas."), value=3, min=1, max=10),
                ui.input_numeric("miner_pop", tip("Population Tree Map Size", "The number of formulas generated and tested per generation."), value=100, min=10, max=500),
                col_widths={"sm": (4, 4, 4, 4, 4, 4)}
            ),

            ui.input_action_button("btn_run_miner", "Launch Factor Miner (Genetic Search)", class_="btn-run w-100 mb-4")
        ]
        
        results = miner_results_val.get()
        if results:
            cards.append(ui.hr())
            cards.append(ui.h4("🏆 Top Discovered Alphabetic Formulas", class_="mt-3 mb-3", style="color: white;"))
            for i, r in enumerate(results):
                cards.append(
                    ui.div(
                        ui.h5(f"Rank {i+1} | Evolution Metric Score: {r['fitness_score']:.4f}", class_="agent-label text-success"),
                        ui.tags.code(r['formula'], style="font-size: 1.15rem; color: #e5e5e5; font-weight: 500;"),
                        class_="agent-card agent-quant mt-2"
                    )
                )
        return ui.div(*cards)

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
        start_y, end_y = input.year_range()

        def _bg_miner():
            try:
                import tools
                from constituents.universe_builder import get_latest_constituents
                from factor_miner import discover_alpha_factors
                
                miner_cb(5, f"Fetching Baseline DataFrame ({m_universe} Array Subset)...")
                tickers = get_latest_constituents(m_universe)[:80] # Proxy subset
                df = tools.fetch_universe_data(tickers, start_y, end_y, force_refresh=False)
                
                miner_cb(20, f"Executing Genetic Evolution (Pop: {m_pop}, Gens: {m_gens}, Horizon: {m_horizon}d)...")
                results = discover_alpha_factors(df, generations=m_gens, pop_size=m_pop, horizon=m_horizon, fitness_metric=m_fitness, syntax_set=m_funcs, progress_callback=miner_cb)
                
                miner_progress_state["res"] = results
            except Exception as e:
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
        res = miner_results_val.get()
        if res:
            choices = {"None": "None"}
            for i, item in enumerate(res):
                choices[item["formula"]] = f"Factor {i+1} : {item['formula'][:40]}... (Score: {item['fitness_score']:.3f})"
            ui.update_select("mined_formula_dropdown", choices=choices)

    @reactive.Effect
    @reactive.event(input.run_btn)
    def run_analysis():
        nonlocal cancel_flag
        from agents import run_agentic_workflow

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
        start_year, end_year = input.year_range()
        initial_aum = input.initial_aum()
        rebalance_freq = input.rebalance_freq()
        portfolio_size = float(input.portfolio_size())
        portfolio_sizing_type = input.portfolio_sizing_type()
        strategy_type = input.strategy_type()
        quantile_split = int(input.quantile_split())

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
                    ui_progress(0, 0, "", "⚠️ Cache is empty! Forcing a full rebuild from SEC EDGAR. This will take ~5-10 minutes...")
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

                res = run_agentic_workflow(
                    tickers=dynamic_tickers,
                    themes=theme_keys,
                    custom_formula=custom_formula_opt,
                    portfolio_size=portfolio_size,
                    portfolio_sizing_type=portfolio_sizing_type,
                    strategy_type=strategy_type,
                    start_year=start_year,
                    end_year=end_year,
                    invert_factor=invert_factor,
                    rebalance_freq=rebalance_freq,
                    initial_aum=initial_aum,
                    progress_callback=ui_progress,
                    constituent_timeline=timeline,
                    benchmark_ticker=benchmark_ticker,
                    quantiles=quantile_split,
                )
                progress_state["res"] = res
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
                ui.value_box(tip("Total Ret ($)", "Cumulative absolute dollar simulation growth over the tested period."), _fmt_doll(m.get('total_ret_usd', 'N/A')),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Strategy Ann. Ret", "The geometric average yearly return."), _fmt_pct(m.get('ann_port_return', 'N/A')),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Strategy Ann Vol", "Annualized standard deviation outlining generalized expected risk."), _fmt_pct(m.get('ann_vol', 'N/A')),
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
                ui.value_box(tip("Strategy Sharpe", "Risk-adjusted return (Annualized Return divided by Volatility)."), _fmt(m.get('sharpe_ratio', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('sharpe_ratio'), 0.5, 0), fg="white")),
                ui.value_box(tip("Strategy Max DD", "Maximum peak-to-trough percentage capital destruction."), _fmt_pct(m.get('max_drawdown', 'N/A')),
                             theme=ui.value_box_theme(bg=_color(m.get('max_drawdown'), -0.15, -0.25, False), fg="white")),
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
                ui.value_box(tip("Universe Size", "Total number of active assets analyzed in the final rebalance."), f"{m.get('n_tickers', '?')}",
                             theme=ui.value_box_theme(bg="#2d3436", fg="white")),
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

        html_parts = []
        for key in ["equity_json", "quintile_json", "ic_json", "drawdown_json"]:
            chart_json = res["plots"].get(key)
            if chart_json:
                fig = pio.from_json(chart_json)
                
                # Render to raw HTML string
                raw_html = fig.to_html(full_html=True, include_plotlyjs="cdn")
                
                # Make the iframe background transparent dark
                raw_html = raw_html.replace("<head>", "<head><style>body { margin: 0; background-color: #111111 !important; }</style>")
                
                # Encode b64 to bypass script-stripping in Shiny's ui.HTML
                b64 = base64.b64encode(raw_html.encode("utf-8")).decode("utf-8")
                height = "470px" if key == "equity_json" else "380px"
                
                iframe = f'<iframe src="data:text/html;base64,{b64}" style="width: 100%; height: {height}; border: none; overflow: hidden; border-radius: 8px;" scrolling="no"></iframe>'
                html_parts.append(iframe)

        return ui.HTML(f"<div style='margin-top: 16px; display: flex; flex-direction: column; gap: 24px;'>{''.join(html_parts)}</div>")

    @output
    @render.ui
    def agent_logs():
        res = workflow_result()
        if not res or "error" in res:
            return ui.HTML(
                '<div style="text-align: center; padding: 50px; color: #555;">'
                '🤖 Agent analysis logs will appear here after running.</div>'
            )

        logs = res.get("logs", [])
        cards = []
        for log in logs:
            agent = log.get("agent", "Unknown")
            msg = log.get("message", "").replace("\n", "<br>")

            css = "agent-hypothesis" if "Hypothesis" in agent else (
                "agent-quant" if "Quant" in agent else (
                    "agent-risk" if "Risk" in agent else ""))
            icon = "🧪" if "Hypothesis" in agent else (
                "📊" if "Quant" in agent else (
                    "🛡️" if "Risk" in agent else "⚠️"))

            cards.append(f"""
            <div class="agent-card {css}">
                <div class="agent-label">{icon} {agent}</div>
                <div style="font-size: 0.9rem; line-height: 1.6; color: #c8cad0;">{msg}</div>
            </div>
            """)

        return ui.HTML(f"<div style='margin-top: 14px;'>{''.join(cards)}</div>")


app = App(app_ui, server)
