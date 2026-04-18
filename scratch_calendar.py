import pandas as pd
import numpy as np
import datetime
import calendar

def generate_pnl_calendar_html(strat_ret: pd.Series, daily_holdings: dict = None) -> str:
    # strat_ret index is datetime, values are floats (returns)
    df = strat_ret.to_frame(name="ret")
    df['date'] = df.index
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.weekday
    
    html = """
    <style>
        .cal-wrapper { padding: 10px; color: #d1d4dc; font-family: sans-serif; }
        .calendar_table { border-collapse: collapse; background-color: #1e222d; font-size: 12px; margin: 0 auto 30px auto; width: 100%; max-width: 1200px; }
        .calendar_table th, .calendar_table td { border: 1px solid #434651; text-align: center; }
        .calendar_table th.month { font-size: 1.4em; padding: 10px; text-align: left; background-color: #2a2e39;}
        .calendar_table td { width: 12.5%; height: 80px; vertical-align: top; text-align: right; padding: 5px;}
        .calendar_table th.weekly-pnl { width: 12.5%; background-color: #2a2e39; border-left: 2px solid #5d606b; }
        .calendar_table td.weekly-pnl-cell { font-weight: bold; font-size: 14px; text-align: center; vertical-align: middle; background-color: #1a1e28; border-left: 2px solid #5d606b; }
        .calendar_table td.noday { background-color: #131722; border-color: #131722; }
        .daynum { color: #b2b5be; font-size: 11px; float: left; margin-right: 5px;}
        .pnl { font-size: 14px; font-weight: bold; float: right; padding: 2px 4px; border-radius: 3px; }
        .win { color: #26a69a; background-color: rgba(38,166,154,0.1); }
        .loss { color: #ef5350; background-color: rgba(239,83,80,0.1); }
        .notrade .pnl { color: #5d606b; }
        .trade_day { cursor: pointer; transition: background-color 0.2s; }
        .trade_day:hover { background-color: #2a2e39; }
        .td-content { display: flex; flex-direction: column; height: 100%; }
        .td-header { display: flex; justify-content: space-between; align-items: baseline; }
        
        .modal-overlay { display: none; position: fixed; z-index: 10000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.7); justify-content: center; align-items: center; }
        .modal-box { background-color: #161a26; border: 1px solid #252a3a; border-radius: 12px; padding: 20px; min-width: 300px; max-width: 500px; max-height: 80vh; overflow-y: auto; color: white; text-align: left; }
        .modal-close { float: right; cursor: pointer; font-size: 20px; font-weight: bold; color: #8b90a0; }
        .modal-close:hover { color: white; }
    </style>
    
    <div id="holdingsModal" class="modal-overlay" onclick="if(event.target === this) this.style.display='none'">
        <div class="modal-box">
            <span class="modal-close" onclick="document.getElementById('holdingsModal').style.display='none'">&times;</span>
            <h4 id="modalTitle" style="margin-top:0">Holdings</h4>
            <div id="modalContent" style="margin-top:15px; font-size: 14px; line-height: 1.6;"></div>
        </div>
    </div>
    
    <script>
    function showHoldings(dateStr, longsStr, shortsStr) {
        document.getElementById('modalTitle').innerText = "Traded Stocks - " + dateStr;
        let l_arr = longsStr ? longsStr.split(',') : [];
        let s_arr = shortsStr ? shortsStr.split(',') : [];
        let content = "<strong>Longs (" + l_arr.length + ")</strong><br><span style='color:#00d4aa;'>" +
                      (l_arr.length > 0 ? l_arr.join(', ') : 'None') + "</span>" + 
                      "<br><br><strong>Shorts (" + s_arr.length + ")</strong><br><span style='color:#ff6b6b;'>" +
                      (s_arr.length > 0 ? s_arr.join(', ') : 'None') + "</span>";
        document.getElementById('modalContent').innerHTML = content;
        document.getElementById('holdingsModal').style.display = 'flex';
    }
    </script>
    <div class="cal-wrapper">
    """
    
    # Iterate over years
    for yr in sorted(df['year'].unique()):
        # Iterate over months
        for mo in sorted(df[df['year'] == yr]['month'].unique()):
            month_name = calendar.month_name[mo]
            sub_df = df[(df['year'] == yr) & (df['month'] == mo)]
            month_ret = (1 + sub_df['ret']).prod() - 1
            month_win_class = 'win' if month_ret > 0 else 'loss'
            
            html += f"<table class='calendar_table'>"
            html += f"<tr><th colspan='8' class='month'><div>{month_name} {yr} <span class='pnl {month_win_class}' style='float:right;'>Total: {month_ret*100:.2f}%</span></div></th></tr>"
            html += "<tr><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th><th>Sun</th><th class='weekly-pnl'>Weekly PnL</th></tr>"
            
            cal = calendar.monthcalendar(yr, mo)
            for week in cal:
                html += "<tr>"
                
                week_rets = [] # to calculate weekly pnl
                
                for day in week:
                    if day == 0:
                        html += "<td class='noday'></td>"
                    else:
                        date_str = f"{yr}-{mo:02d}-{day:02d}"
                        date_idx = pd.to_datetime(date_str)
                        if date_idx in sub_df.index:
                            r = sub_df.loc[date_idx, 'ret']
                            week_rets.append(r)
                            r_class = 'win' if r > 0 else ('loss' if r < 0 else 'notrade')
                            r_str = f"{r*100:+.2f}%"
                            
                            l_joined = ""
                            s_joined = ""
                            if daily_holdings and date_idx in daily_holdings:
                                h = daily_holdings[date_idx]
                                l_joined = ",".join(h.get('longs', []))
                                s_joined = ",".join(h.get('shorts', []))
                                
                            html += f"<td class='trade_day' onclick=\"showHoldings('{date_str}', '{l_joined}', '{s_joined}')\">"
                            html += f"<div class='td-content'>"
                            html += f"<div class='td-header'><span class='daynum'>{day}</span><span class='pnl {r_class}'>{r_str}</span></div>"
                            html += f"</div></td>"
                        else:
                            html += f"<td class='noday'><span class='daynum'>{day}</span></td>"
                            
                # Added Weekly PnL calculation cell
                if week_rets:
                    week_ret = (1 + pd.Series(week_rets)).prod() - 1
                    w_class = 'win' if week_ret > 0 else 'loss'
                    w_str = f"{week_ret*100:+.2f}%"
                    html += f"<td class='weekly-pnl-cell'><span class='pnl {w_class}'>{w_str}</span></td>"
                else:
                    html += "<td class='weekly-pnl-cell noday'></td>"
                    
                html += "</tr>"
            html += "</table>"
            
    html += "</div>"
    return html

if __name__ == "__main__":
    dates = pd.date_range(start="2024-01-01", end="2024-02-15", freq="D")
    rets = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    daily_holdings = {
        pd.to_datetime("2024-01-05"): {"longs": ["AAPL", "MSFT"], "shorts": ["TSLA"]}
    }
    out = generate_pnl_calendar_html(rets, daily_holdings)
    with open("test_cal.html", "w") as f:
        f.write(out)
    print("Done!")
