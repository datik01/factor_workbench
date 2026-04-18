import pandas as pd
import numpy as np
import datetime
import calendar

def generate_pnl_calendar_html(strat_ret: pd.Series, daily_holdings: dict = None) -> str:
    str_holdings = {}
    if daily_holdings:
        str_holdings = {k.strftime('%Y-%m-%d') if hasattr(k, 'strftime') else str(k)[:10]: v for k, v in daily_holdings.items()}
        
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
    </style>
    <div class="cal-wrapper">
    """
    
    # Iterate over years
    for yr in sorted(df['year'].unique()):
        year_df = df[df['year'] == yr]
        year_ret = (1 + year_df['ret']).prod() - 1
        year_win_class = 'win' if year_ret > 0 else 'loss'
        
        html += f"<div style='font-size: 2em; font-weight: bold; margin: 30px 0 10px 0; border-bottom: 2px solid #434651; padding-bottom: 5px; color: #ffffff;'>{yr}: Yearly PNL <span class='pnl {year_win_class}' style='font-size: 0.8em; margin-left: 10px;'>{year_ret*100:+.2f}%</span></div>"
        
        # Iterate over months
        for mo in sorted(df[df['year'] == yr]['month'].unique()):
            month_name = calendar.month_name[mo]
            sub_df = df[(df['year'] == yr) & (df['month'] == mo)]
            month_ret = (1 + sub_df['ret']).prod() - 1
            month_win_class = 'win' if month_ret > 0 else 'loss'
            
            html += f"<table class='calendar_table'>"
            html += f"<tr><th colspan='8' class='month'><div>{month_name} {yr} <span class='pnl {month_win_class}' style='float:right;'>Monthly PNL: {month_ret*100:+.2f}%</span></div></th></tr>"
            html += "<tr><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th><th>Sun</th><th class='weekly-pnl'>Weekly PnL</th></tr>"
            
            ret_dict = sub_df.set_index('day')['ret'].to_dict()
            
            cal = calendar.monthcalendar(yr, mo)
            for week in cal:
                html += "<tr>"
                
                week_rets = [] # to calculate weekly pnl
                
                for day in week:
                    if day == 0:
                        html += "<td class='noday'></td>"
                    else:
                        if day in ret_dict:
                            r = ret_dict[day]
                            week_rets.append(r)
                            r_class = 'win' if r > 0 else ('loss' if r < 0 else 'notrade')
                            r_str = f"{r*100:+.2f}%"
                            
                            date_str = f"{yr}-{mo:02d}-{day:02d}"
                            l_joined = ""
                            s_joined = ""
                            if str_holdings and date_str in str_holdings:
                                h = str_holdings[date_str]
                                l_joined = ",".join(h.get('longs', []))
                                s_joined = ",".join(h.get('shorts', []))
                                
                            html += f"<td class='trade_day' onclick=\"Shiny.setInputValue('cal_cell_click', '{date_str}|{l_joined}|{s_joined}', {{priority: 'event'}})\">"
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
