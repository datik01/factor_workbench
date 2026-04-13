"""
agents.py
Factor Workbench: Multi-Agent Orchestration Module
Danny Atik - SYSEN 5381

Three-agent pipeline using Ollama (local LLM) for orchestration
and direct tool calling for the Quant Analyst stage.

Agents:
  1. Hypothesis Generator — theorizes about factor viability
  2. Quant Analyst — calls run_cross_sectional_backtest() tool, summarizes results
  3. Risk Manager — evaluates metrics, issues PASS/FAIL verdict
"""

import json
import requests
import os
from dotenv import load_dotenv
from tools import run_cross_sectional_backtest

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Ollama Configuration
# ═══════════════════════════════════════════════════════════════

PORT = 11434
OLLAMA_HOST = f"http://localhost:{PORT}"
CHAT_URL = f"{OLLAMA_HOST}/api/chat"

# Preferred model
PREFERRED_MODELS = ["gemma4", "gemma4:latest"]


def _get_best_model() -> str:
    """Check which models are available locally and return the best one."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        available = [m["name"] for m in resp.json().get("models", [])]
        for model in PREFERRED_MODELS:
            if model in available:
                return model
        return available[0] if available else "gemma3:12b"
    except Exception:
        return "gemma3:12b"


MODEL = _get_best_model()


def ollama_chat(system_prompt: str, user_message: str, model: str = MODEL) -> str:
    """
    Send a chat completion request to Ollama.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    try:
        response = requests.post(CHAT_URL, json=body, timeout=180)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "No response generated.")
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Please start it with: ollama serve"
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Agent System Prompts
# ═══════════════════════════════════════════════════════════════

HYPOTHESIS_PROMPT = (
    "You are an expert Quantitative Finance Hypothesis Generator. "
    "Given a factor theme and a stock universe, provide a concise theoretical hypothesis "
    "(3-4 sentences) on why this factor might or might not generate alpha in a "
    "long/short portfolio construction framework. Reference academic factor literature "
    "(Fama-French, Carhart, etc.) where relevant. Be specific about the economic intuition."
)

QUANT_SUMMARY_PROMPT = (
    "You are a Quantitative Analyst reviewing cross-sectional backtest results. "
    "Summarize the key findings in 3-4 sentences. Focus on:\n"
    "1) Information Coefficient (IC) and IC Information Ratio — do they indicate predictive power?\n"
    "2) Long/short portfolio Sharpe ratio and alpha vs equal-weight benchmark\n"
    "3) Quantile spread — is the return monotonic from lowest to highest quantile?\n"
    "4) Statistical significance (regression p-value, R²)\n"
    "Use only the data provided. Do NOT invent numbers."
)

RISK_MANAGER_PROMPT = (
    "You are a strict Risk Manager evaluating a quantitative long/short factor strategy. "
    "Given the backtest metrics, provide a 3-4 sentence risk assessment.\n"
    "Evaluate: maximum drawdown risk, Sharpe ratio quality, IC stability, and sample size.\n"
    "Conclude with 'RECOMMENDATION: DEPLOY' or 'RECOMMENDATION: DO NOT DEPLOY'.\n"
    "DEPLOY criteria: |Sharpe| > 0.3 AND |mean IC| > 0.02 AND max drawdown > -0.25 "
    "AND n_trading_days > 200."
)


# ═══════════════════════════════════════════════════════════════
# Multi-Agent Workflow
# ═══════════════════════════════════════════════════════════════

def run_agentic_workflow(
    tickers: list,
    themes: list,
    custom_formula: str = None,
    portfolio_size: int = 100,
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
) -> dict:
    """
    Executes the three-agent pipeline with cross-sectional portfolio construction.
    """
    logs = []

    # ─────────────────────────────────────────────────────────
    # AGENT 1: Hypothesis Generator
    # ─────────────────────────────────────────────────────────
    formatted_themes = ", ".join(themes)
    hypothesis = ollama_chat(
        system_prompt=HYPOTHESIS_PROMPT,
        user_message=(
            f"Generate a hypothesis for the composite '{formatted_themes}' factor strategy. "
            f"The backtest spans {start_year} to {end_year} and rebalances at frequency '{rebalance_freq}'. "
            f"Directional inversion is {'ON (Contrarian tilt)' if invert_factor else 'OFF (Standard tilt)'}. "
            f"The strategy ranks the entire Russell 2000 universe and targets a '{strategy_type}' portfolio holding exactly {portfolio_size} stocks."
        ),
    )
    logs.append({"agent": "Hypothesis Generator", "message": hypothesis})

    # ─────────────────────────────────────────────────────────
    # AGENT 2: Quant Analyst (Tool Execution + LLM Summary)
    # ─────────────────────────────────────────────────────────

    # Step 2a: Execute the backtest tool
    tool_result_str = run_cross_sectional_backtest(
        tickers=tickers,
        themes=themes,
        custom_formula=custom_formula,
        portfolio_size=portfolio_size,
        strategy_type=strategy_type,
        start_year=start_year,
        end_year=end_year,
        invert_factor=invert_factor,
        rebalance_freq=rebalance_freq,
        initial_aum=initial_aum,
        progress_callback=progress_callback,
        constituent_timeline=constituent_timeline,
        benchmark_ticker=benchmark_ticker,
        quantiles=quantiles,
    )
    tool_data = json.loads(tool_result_str)

    plots_json = {}
    metrics = {}

    if tool_data.get("success"):
        plots_json = tool_data.get("plots", {})
        metrics = tool_data.get("metrics", {})

        # Step 2b: LLM summarizes backtest results
        metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        quant_summary = ollama_chat(
            system_prompt=QUANT_SUMMARY_PROMPT,
            user_message=(
                f"Hypothesis: {hypothesis}\n\n"
                f"Cross-sectional backtest results for '{formatted_themes}' factor composite "
                f"on {metrics.get('n_tickers', '?')} stocks over "
                f"{metrics.get('n_trading_days', '?')} trading days:\n"
                f"{metrics_text}"
            ),
        )
        logs.append({"agent": "Quant Analyst", "message": quant_summary})
    else:
        error_msg = f"Backtest failed: {tool_data.get('error', 'Unknown error')}"
        logs.append({"agent": "Quant Analyst", "message": error_msg})
        return {"logs": logs, "plots": plots_json, "metrics": metrics, "error": error_msg}

    # ─────────────────────────────────────────────────────────
    # AGENT 3: Risk Manager
    # ─────────────────────────────────────────────────────────
    risk_review = ollama_chat(
        system_prompt=RISK_MANAGER_PROMPT,
        user_message=(
            f"Quant Summary: {quant_summary}\n\n"
            f"Raw Metrics:\n{json.dumps(metrics, indent=2)}"
        ),
    )
    logs.append({"agent": "Risk Manager", "message": risk_review})

    return {
        "logs": logs,
        "plots": plots_json,
        "metrics": metrics,
    }
