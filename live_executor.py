import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import argparse
import traceback
from zoneinfo import ZoneInfo
import concurrent.futures

ET = ZoneInfo("America/New_York")

def now_et():
    return datetime.datetime.now(tz=ET)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from tradestation_api import TradeStationClient
from tools import _compute_factor_scores, fetch_universe_data, send_pushover_alert

def log_msg(msg):
    with open("live_execution_status.log", "a") as f:
        f.write(f"[{now_et().strftime('%Y-%m-%d %H:%M:%S ET')}] {msg}\n")
    print(msg)

PORTFOLIO_SIZE = 3
RISK_ATR_MULT = 3.0
TARGET_ACCOUNT = None
IS_LIVE = False

def fetch_live_universe(universe_str, explicit_start_year):
    log_msg(f"[*] Loading Unified {universe_str} Matrix Constituents...")
    constituents_path = os.path.join(BASE_DIR, ".cache", "constituents", f"{universe_str.lower()}_tickers_latest.txt")
    if not os.path.exists(constituents_path):
        raise FileNotFoundError(f"Missing Unified R2K cache (Path: {constituents_path})")
        
    with open(constituents_path, "r") as f:
        universe_tickers = [t.strip() for t in f.readlines() if t.strip()]
        
    log_msg(f"[*] Booting massive unified dataset for {len(universe_tickers)} tickers...")
    current_year = now_et().year
    df = fetch_universe_data(
        tickers=universe_tickers,
        start_year=explicit_start_year, 
        end_year=current_year,
        progress_callback=None,
        force_refresh=False
    )
    df["Returns"] = df.get("returns", df.get("Returns", 0.0))
    df["Volume"] = df.get("volume", df.get("Volume", 0.0))
    df["VWAP"] = df.get("vwap", df.get("VWAP", 0.0))
    return df

def execute_eod_liquidation(args):
    log_msg("\n⏰ 15:49:30 PM! INITIATING END-OF-DAY FLAT LIQUIDATION (MOC) ⏰")
    ts = TradeStationClient(is_live=args.is_live)
    act_id = args.account
    base_url = "https://sim-api.tradestation.com/v3" if str(act_id).upper().startswith("SIM") else ts.base_url

    ts.tokens = ts._load_tokens()

    try:
        log_msg("[*] Sweeping resting OSO Stop-Loss orders...")
        ord_resp = ts._make_request("GET", f"{base_url}/brokerage/accounts/{act_id}/orders")
        if ord_resp.status_code == 200:
            working_orders = [o for o in ord_resp.json().get("Orders", []) if o.get("Status") in ["OPN", "DON", "ACK", "QUE", "UCN", "OSN"]]
            for o in working_orders:
                sym = o.get("Legs", [{}])[0].get("Symbol")
                oid = o.get("OrderID")
                ts._make_request("DELETE", f"{base_url}/orderexecution/orders/{oid}")
                log_msg(f"[*] Canceled Working Bracket {oid} for {sym}")
    except Exception as e:
        log_msg(f"[-] Order Sweep Exception: {e}")

    time.sleep(2)
    
    try:
        log_msg("[*] Fetching physically held inventory block...")
        pos_resp = ts._make_request("GET", f"{base_url}/brokerage/accounts/{act_id}/positions")
        if pos_resp.status_code == 200:
            positions = pos_resp.json().get("Positions", [])
            for p in positions:
                sym = p.get("Symbol")
                qty = p.get("Quantity")
                
                if int(float(qty)) != 0:
                    is_short = int(float(qty)) < 0
                    log_msg(f"[*] ROUTING EOD LIQUIDATION OF {qty} SHARES FOR {sym} (Market-On-Close)")
                    order_payload = {
                        "AccountID": act_id,
                        "Symbol": sym,
                        "Quantity": str(abs(int(float(qty)))),
                        "OrderType": "Market",
                        "TradeAction": "BuyToCover" if is_short else "Sell",
                        "TimeInForce": {"Duration": "CLO"},
                        "Route": "Intelligent"
                    }
                    res = ts._make_request("POST", f"{base_url}/orderexecution/orders", json=order_payload)
                    if res.status_code in [200, 201]:
                        log_msg(f"[+] FLAT: {sym} -> {res.json()}")
                    else:
                        log_msg(f"[-] Execution Reject on {sym}: {res.text}")
    except Exception as e:
        log_msg(f"[-] Position Clear Exception: {e}")

    log_msg("\n✅ TRADING EXHAUSTED FOR DAY. PORTFOLIO FLAT. ✅")
    send_pushover_alert("Portfolio Flattened", "All resting brackets canceled and inventory liquidated for the day. (MOC Executed)", priority=0)

def execute_daily_entry(args, state_cache):
    log_msg("\n⏰ 09:27:00 AM! INITIATING MOO ENTRY ROUTING ⏰")
    ts = TradeStationClient(is_live=args.is_live)
    act_id = args.account
    base_url = "https://sim-api.tradestation.com/v3" if str(act_id).upper().startswith("SIM") else ts.base_url
    
    ts.tokens = ts._load_tokens()
    equity = ts.get_equity(act_id)
    if equity <= 0:
        log_msg("[-] Account Equity is ZERO. Aborting entry.")
        return False
        
    log_msg(f"[+] Connected to Account {act_id} | Equity: ${equity:,.2f}")
    
    universe_df = fetch_live_universe(args.universe, args.start_year)
    
    # [MEMORY OPTIMIZATION]: Prevent Swap-Thrashing on 1GB Droplets!
    # The max strategy lookback is 30 days. Trimming the df from 1-year (800k rows) 
    # to 60-days (~200k rows) prevents the groupby().rolling() from exhausting RAM.
    import pandas as pd
    universe_df['date'] = pd.to_datetime(universe_df['date'])
    max_d = universe_df['date'].max()
    universe_df = universe_df[universe_df['date'] >= (max_d - pd.Timedelta(days=60))].copy()
    universe_df = universe_df.sort_values(['ticker', 'date']).reset_index(drop=True)

    theme_arr = [x for x in args.themes.split(",") if x.strip()] if args.themes else []
    
    scored = _compute_factor_scores(
        universe=universe_df,
        themes=theme_arr,
        custom_formula=args.formula,
        progress_callback=None,
        universe_filter=args.filter,
        execution_timing="Intraday (Open to Close)"
    )
    
    latest_date = scored["date"].max()
    log_msg(f"[+] Evaluated Matrix as of EOD: {latest_date.strftime('%Y-%m-%d')}")
    
    todays_targets = scored[scored["date"] == latest_date].copy()
    
    if getattr(args, 'direction', '') == "Short Only":
        todays_targets = todays_targets.sort_values(["factor_score", "ticker"], ascending=[args.invert, True])
    else:
        todays_targets = todays_targets.sort_values(["factor_score", "ticker"], ascending=[not args.invert, True])
        
    todays_targets = todays_targets.dropna(subset=["factor_score"])
    top_targets = todays_targets.head(args.portfolio_size).copy()
    
    if top_targets.empty:
        log_msg("[-] ZERO actionable targets found that met the Universe Filter.")
        return False
        


    for idx, row in top_targets.iterrows():
        ticker = row["ticker"]
        ticker_data = universe_df[universe_df["ticker"] == ticker].sort_values("date").tail(15)
        atr_pct = 0.05
        if len(ticker_data) >= 7:
            ticker_data["prev_close"] = ticker_data["close"].shift(1)
            tr1 = ticker_data["high"] - ticker_data["low"]
            tr2 = (ticker_data["high"] - ticker_data["prev_close"]).abs()
            tr3 = (ticker_data["low"] - ticker_data["prev_close"]).abs()
            ticker_data["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = ticker_data["tr"].rolling(14, min_periods=7).mean().iloc[-1]
            close_price = ticker_data["close"].iloc[-1]
            atr_pct = atr / close_price if close_price > 0 else 0.05
        top_targets.at[idx, "atr_pct"] = atr_pct

    log_msg("="*50)
    log_msg("🔥 TARGET EXECUTION PAYLOAD: QUEUED 🔥")
    action_str = "SELL SHORT" if getattr(args, 'direction', '') == "Short Only" else "BUY LONG"
    for _, row in top_targets.iterrows():
        log_msg(f"  --> {action_str}: {row['ticker']} | Score: {row['factor_score']:.2f}")
    log_msg("="*50)
    
    active_capital = equity * 1.0
    if args.sizing_strategy == "Fixed Fractional (1/N)":
        top_targets["allocation"] = active_capital / args.portfolio_size
    elif args.sizing_strategy == "Inverse Volatility (ATR)":
        inv_atr = 1.0 / top_targets["atr_pct"].replace(0, 0.0001)
        top_targets["allocation"] = active_capital * (inv_atr / inv_atr.sum())
    else:
        top_targets["allocation"] = active_capital / max(1, len(top_targets))

    if args.liquidity_cap_type != "None":
        if args.liquidity_cap_type == "Max ADV %":
            dollar_adv = top_targets["avg_30d_volume"] * top_targets["close"]
            max_dollars = dollar_adv * (args.liquidity_cap_value / 100.0)
            top_targets["allocation"] = top_targets[["allocation"]].join(max_dollars.rename("max_dollars")).min(axis=1)
        else:
            top_targets["allocation"] = top_targets["allocation"].clip(upper=args.liquidity_cap_value)

    targets_to_monitor = {}
    is_short = getattr(args, 'direction', '') == "Short Only"
    
    for _, target in top_targets.iterrows():
        try:
            ticker = target["ticker"]
            bars = ts.get_historical_daily_bars(ticker, days_back=1)
            if not bars: continue
            
            yesterday_close = float(bars[-1]["Close"])
            qty = int(target["allocation"] / yesterday_close)
            if qty <= 0: continue
            
            log_msg(f"[*] ROUTING {qty} SHARES OF {ticker} @ MARKET-ON-OPEN")
            
            duration = "DAY" if getattr(args, 'force_intraday', False) else "OPG"
            order_payload = {
                "AccountID": act_id,
                "Symbol": ticker,
                "Quantity": str(qty),
                "OrderType": "Market",
                "TradeAction": "SellShort" if is_short else "Buy",
                "TimeInForce": {"Duration": duration},
                "Route": "Intelligent"
            }
            res = ts._make_request("POST", f"{base_url}/orderexecution/orders", json=order_payload)
            if res.status_code in [200, 201]:
                log_msg(f"[+] MOO QUEUED: {ticker} -> {res.json()}")
                targets_to_monitor[ticker] = {
                    "quantity": qty,
                    "atr_pct": target.get("atr_pct", 0.05),
                    "sl_mult": args.sl_mult,
                    "tp_mult": getattr(args, 'tp_mult', 0.0),
                    "direction": "Short Only" if is_short else "Long Only"
                }
            else:
                log_msg(f"[-] MOO REJECT: {ticker} -> {res.text}")
                send_pushover_alert("Order Rejected", f"{ticker} order was rejected: {res.text}", priority=1)
        except Exception as e:
             log_msg(f"[-] FATAL MOO ERROR {ticker}: {str(e)}")
             send_pushover_alert("Fatal Order Routing Error", f"{ticker} routing failed: {str(e)}", priority=1)

    if targets_to_monitor:
        state_cache["pending_risk"] = targets_to_monitor
        queued_msg = ", ".join([f"{v['quantity']} {k}" for k, v in targets_to_monitor.items()])
        send_pushover_alert("Targets Successfully Queued", f"Routed MOO orders for: {queued_msg}", priority=0)
        return True
    return False

def execute_risk_brackets(args, state_cache):
    log_msg("\n⏰ 09:30:00 AM! INITIATING 0.25s RAPID-POLLING OCO RISK LOOP ⏰")
    ts = TradeStationClient(is_live=args.is_live)
    act_id = args.account
    base_url = "https://sim-api.tradestation.com/v3" if str(act_id).upper().startswith("SIM") else ts.base_url
    
    targets_to_monitor = state_cache.get("pending_risk", {})
    if not targets_to_monitor: return
    
    loop_start_time = time.time()
    ts.tokens = ts._load_tokens()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.portfolio_size) as executor:
        while time.time() - loop_start_time < 30.0 and len(targets_to_monitor) > 0:
            try:
                positions = ts.get_positions(act_id)
                filled_tickers = [p.get("Symbol") for p in positions if int(float(p.get("Quantity", 0))) != 0]
                
                futures = []
                for ticker in list(targets_to_monitor.keys()):
                    if ticker in filled_tickers:
                        pos_data = next((p for p in positions if p.get("Symbol") == ticker), None)
                        if pos_data:
                            exact_open = float(pos_data.get("AveragePrice", 0.0))
                            if exact_open <= 0.0: continue 
                                
                            meta = targets_to_monitor[ticker]
                            atr_limit = meta["atr_pct"] * meta["sl_mult"]
                            
                            if meta["direction"] == "Short Only":
                                stop_price = exact_open * (1.0 + atr_limit)
                                target_price = exact_open * (1.0 - (meta["atr_pct"] * meta["tp_mult"])) if meta["tp_mult"] > 0 else None
                                exit_action = "BuyToCover"
                            else:
                                stop_price = exact_open * (1.0 - atr_limit)
                                target_price = exact_open * (1.0 + (meta["atr_pct"] * meta["tp_mult"])) if meta["tp_mult"] > 0 else None
                                exit_action = "Sell"
                                
                            log_msg(f"[+] ⚡ FILL DETECTED: {ticker} @ ${exact_open:.2f}. ROUTING OCO (STOP: {stop_price:.2f} | LIMIT: {target_price})")
                            
                            oco_group_name = f"Exit_{ticker}_{now_et().strftime('%m%d%H%M%S')}"
                            futures.append(
                                executor.submit(
                                    ts.place_oco_exit_orders,
                                    act_id, ticker, meta["quantity"], stop_price, target_price, exit_action, oco_group_name
                                )
                            )
                            del targets_to_monitor[ticker]
                            
                if futures:
                    concurrent.futures.wait(futures)
            except Exception as e:
                log_msg(f"[-] Polling Error: {e}")
            time.sleep(0.25)
        
    if targets_to_monitor:
        log_msg(f"[-] Risk Loop completed. {len(targets_to_monitor)} targets failed to fill. Skipped brackets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Executor Bot")
    parser.add_argument("--account", type=str, required=True)
    parser.add_argument("--formula", type=str, required=True)
    parser.add_argument("--filter", type=str, required=True)
    parser.add_argument("--portfolio_size", type=int, default=3)
    parser.add_argument("--sizing_strategy", type=str, default="Fixed Fractional (1/N)")
    parser.add_argument("--liquidity_cap_type", type=str, default="Max ADV %")
    parser.add_argument("--liquidity_cap_value", type=float, default=100.0)
    parser.add_argument("--start_year", type=int, default=now_et().year)
    parser.add_argument("--sl_mult", type=float, default=0.2)
    parser.add_argument("--tp_mult", type=float, default=0.0)
    parser.add_argument("--is_live", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--universe", type=str, default="R2K")
    parser.add_argument("--invert", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--direction", type=str, default="Long Only")
    parser.add_argument("--themes", type=str, default="")
    parser.add_argument("--force_intraday", action="store_true")
    
    args = parser.parse_args()
    
    log_msg("="*50)
    log_msg("🚀 FACTOR WORKBENCH: STATE MACHINE DAEMON 🚀")
    log_msg(f"Targeting Account: {args.account}")
    log_msg("="*50)
    
    state_cache = {"pending_risk": {}}
    last_entry_date = None
    last_risk_date = None
    last_liq_date = None

    while True:
        try:
            now = now_et()
            today = now.date()
            is_trading_day = now.weekday() <= 4
            
            if getattr(args, 'is_live', False):
                t_entry = now.replace(hour=9, minute=25, second=30, microsecond=0)
            else:
                t_entry = now.replace(hour=9, minute=27, second=0, microsecond=0)
            t_risk = now.replace(hour=9, minute=30, second=0, microsecond=0)
            t_liq = now.replace(hour=15, minute=49, second=30, microsecond=0)
            
            if is_trading_day:
                # Force intraday mode snaps execution immediately if launched mid-day
                if args.force_intraday and last_entry_date != today and now > t_entry and now < t_liq:
                    log_msg("[*] FORCE INTRADAY - Running immediate Entry/Risk Pipeline...")
                    has_targets = execute_daily_entry(args, state_cache)
                    if has_targets:
                        execute_risk_brackets(args, state_cache)
                    last_entry_date = today
                    last_risk_date = today
            
                # 1. MOO Entry Trigger (09:27)
                if now >= t_entry and now < t_risk and last_entry_date != today:
                    execute_daily_entry(args, state_cache)
                    last_entry_date = today
                    
                # 2. OCO Risk Bracket Trigger (09:30)
                if now >= t_risk and now < t_liq and last_risk_date != today:
                    execute_risk_brackets(args, state_cache)
                    last_risk_date = today
                    
                # 3. MOC EOD Liquidation Trigger (15:49:30)
                if now >= t_liq and last_liq_date != today:
                    execute_eod_liquidation(args)
                    last_liq_date = today
                    
            time.sleep(2)
            
        except Exception as e:
            log_msg(f"FATAL DAEMON EXCEPTION: {traceback.format_exc()}")
            send_pushover_alert("FATAL DAEMON CRASH", f"The state machine crashed: {str(e)}\\n\\n{traceback.format_exc()[-200:]}", priority=1)
            time.sleep(60)
