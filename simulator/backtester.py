"""Event-driven intraday backtester.

Execution: target computed on bar t executes at bar t+1's open (no lookahead).
Costs: commission + slippage as a fraction of notional at every flip.
"""
import numpy as np
import pandas as pd


def run(df, strategy, initial=100000, commission=0.0005, slippage=0.0005):
    """Execute `strategy` on intraday OHLCV `df` and return a result dict."""
    rated = strategy.compute(df)
    target = rated["target"].fillna(0).astype(int)

    # Position held during bar t = target decided on bar t-1
    pos = target.shift(1).fillna(0).astype(int)

    # Per-bar close return of the asset
    ret = df["Close"].pct_change().fillna(0)

    # Gross strategy return (earn only when positioned)
    gross = pos * ret

    # Costs: every position flip charges commission+slippage
    turnover = pos.diff().abs().fillna(pos)
    costs = turnover * (commission + slippage)
    net = gross - costs

    # Equity curves
    equity = (1 + net).cumprod() * initial
    bh = (1 + ret).cumprod() * initial

    # Trade log: each flip becomes a trade record
    trades = _trade_log(df, pos, rated)

    # Equity per bar (for charting)
    equity_series = equity
    equity_series.name = "equity"

    metrics = _metrics(net, equity, bh, pos, trades, initial)

    return {
        "rated": rated,
        "pos": pos,
        "net": net,
        "equity": equity_series,
        "bh_equity": bh,
        "trades": trades,
        "metrics": metrics,
    }


def _trade_log(df, pos, rated):
    flips = pos.diff().ne(0)
    idx = np.where(flips.values)[0]
    rows = []
    entry_price = None
    entry_time = None
    for i in idx:
        p = int(pos.iloc[i])
        px = float(df["Close"].iloc[i])
        ts = df.index[i]
        if p == 1:
            entry_price = px
            entry_time = ts
            rows.append({
                "time": ts, "side": "BUY", "price": px,
                "entry_price": None, "pnl": None, "pnl_pct": None,
                "bars_held": None,
            })
        elif p == 0 and entry_price is not None:
            pnl = px - entry_price
            pnl_pct = (px / entry_price - 1) * 100
            bars = int((ts - entry_time) / pd.Timedelta(minutes=5))
            rows.append({
                "time": ts, "side": "SELL", "price": px,
                "entry_price": entry_price, "pnl": pnl, "pnl_pct": pnl_pct,
                "bars_held": bars,
            })
            entry_price = None

    # If still long at the end, close at last price (for P&L accounting)
    if entry_price is not None:
        px = float(df["Close"].iloc[-1])
        ts = df.index[-1]
        pnl = px - entry_price
        pnl_pct = (px / entry_price - 1) * 100
        bars = int((ts - entry_time) / pd.Timedelta(minutes=5))
        rows.append({
            "time": ts, "side": "SELL*", "price": px,
            "entry_price": entry_price, "pnl": pnl, "pnl_pct": pnl_pct,
            "bars_held": bars,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["time","side","price","entry_price","pnl","pnl_pct","bars_held"]
    )


def _metrics(net, equity, bh, pos, trades, initial):
    n = len(net)
    years = n / (252 * 75)  # 75 bars/day at 5m
    bars_per_year = 252 * 75

    total_ret = (equity.iloc[-1] / initial - 1) * 100
    bh_ret = (bh.iloc[-1] / initial - 1) * 100
    cagr = ((equity.iloc[-1] / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    vol = net.std() * np.sqrt(bars_per_year) * 100
    sharpe = (net.mean() * bars_per_year) / (net.std() * np.sqrt(bars_per_year)) if net.std() > 0 else 0

    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    mdd = dd.min()

    exposure = pos.mean() * 100
    num_flips = int(pos.diff().ne(0).sum())

    # Win rate from completed round-trips (SELL rows)
    sells = trades[trades["side"].str.startswith("SELL")] if len(trades) else pd.DataFrame()
    wins = int((sells["pnl"] > 0).sum()) if len(sells) else 0
    trips = len(sells)
    win_rate = (wins / trips * 100) if trips else 0

    avg_win = sells[sells["pnl"] > 0]["pnl_pct"].mean() if wins else 0
    avg_loss = sells[sells["pnl"] <= 0]["pnl_pct"].mean() if trips - wins else 0
    profit_factor = (
        sells[sells["pnl"] > 0]["pnl"].sum() / abs(sells[sells["pnl"] <= 0]["pnl"].sum())
        if trips - wins and sells[sells["pnl"] <= 0]["pnl"].sum() != 0 else np.nan
    )

    # Best / worst single trade
    best = sells["pnl_pct"].max() if trips else 0
    worst = sells["pnl_pct"].min() if trips else 0

    return {
        "total_ret_pct": round(total_ret, 2),
        "bh_ret_pct": round(bh_ret, 2),
        "cagr_pct": round(cagr, 2),
        "vol_pct": round(vol, 2),
        "sharpe": round(sharpe, 2),
        "mdd_pct": round(mdd, 2),
        "calmar": round(cagr / abs(mdd), 2) if mdd else 0,
        "exposure_pct": round(exposure, 1),
        "num_flips": num_flips,
        "round_trips": trips,
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(float(avg_win), 2) if not np.isnan(avg_win) else 0,
        "avg_loss_pct": round(float(avg_loss), 2) if not np.isnan(avg_loss) else 0,
        "profit_factor": round(float(profit_factor), 2) if not np.isnan(profit_factor) else 0,
        "best_pct": round(float(best), 2) if not np.isnan(best) else 0,
        "worst_pct": round(float(worst), 2) if not np.isnan(worst) else 0,
        "n_bars": n,
        "years": round(years, 2),
    }
