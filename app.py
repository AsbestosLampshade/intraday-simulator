"""Flask visual simulator — intraday backtesting with interactive charts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulator.synthetic import generate_intraday, expand_daily_to_intraday
from simulator.strategies import ALL as STRATEGY_CLASSES, BY_ID
from simulator.backtester import run as backtest_run

app = Flask(__name__)

CSV_DIR = Path.home() / "data" / "StockData" / "CSV"
MAX_BARS = 6000  # safety cap for chart payload


def _load_real_intraday(symbol, bars=4500):
    """Load a real stock's daily CSV and expand the most recent window to 5m."""
    # symbol comes as e.g. "TITAN" -> file TITAN_NS.csv
    candidates = [f"{symbol}_NS.csv", f"{symbol}.csv", f"{symbol}_NS.csv".replace(".", "_")]
    path = None
    for c in candidates:
        p = CSV_DIR / c
        if p.exists():
            path = p
            break
    if path is None:
        # try any file matching stem
        for p in CSV_DIR.glob("*.csv"):
            if p.stem.replace("_NS", "") == symbol.replace(".NS", "").replace("_NS", ""):
                path = p
                break
    if path is None or not path.exists():
        return None, f"Stock {symbol} not found"

    daily = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    # Take the last N daily bars that will expand to ~bars intraday bars
    days_needed = max(5, int(np.ceil(bars / 75)))
    daily = daily.tail(days_needed)
    if len(daily) < 5:
        return None, f"Not enough daily data for {symbol}"
    intraday = expand_daily_to_intraday(daily, seed=42)
    # Trim to requested bars (most recent)
    if len(intraday) > bars:
        intraday = intraday.tail(bars)
    return intraday, None


def _get_intraday(source, symbol, bars, seed):
    if source == "synthetic":
        days = max(5, int(np.ceil(bars / 75)))
        df = generate_intraday(days=days, seed=seed)
        if len(df) > bars:
            df = df.tail(bars)
        return df, None
    else:
        return _load_real_intraday(symbol, bars=bars)


def _df_to_records(df):
    """Convert OHLCV df to JSON-serializable list of dicts."""
    df = df.copy()
    # Ensure Datetime index is stringified as ISO
    df.index = pd.to_datetime(df.index)
    records = []
    for ts, row in df.iterrows():
        records.append({
            "t": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "o": round(float(row["Open"]), 2),
            "h": round(float(row["High"]), 2),
            "l": round(float(row["Low"]), 2),
            "c": round(float(row["Close"]), 2),
            "v": int(row["Volume"]),
        })
    return records


def _series_to_records(series, name="value"):
    """Equity series -> list of {t, v}."""
    out = []
    for ts, v in series.items():
        out.append({"t": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S"), "v": round(float(v), 2)})
    return out


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/strategies")
def api_strategies():
    out = []
    for cls in STRATEGY_CLASSES:
        inst = cls()
        out.append({
            "id": cls.id,
            "name": cls.name,
            "params": getattr(cls, "params", {}),
            "description": getattr(cls, "description", ""),
            "logic": inst.logic_text() if hasattr(inst, "logic_text") else "",
            "color": getattr(cls, "color", "#888"),
        })
    return jsonify(out)


@app.route("/api/stocks")
def api_stocks():
    stocks = []
    for p in sorted(CSV_DIR.glob("*.csv")):
        if p.name == "manifest.csv":
            continue
        stem = p.stem.replace("_NS", "")
        try:
            df = pd.read_csv(p, usecols=["Date"])
            stocks.append({"id": stem, "label": stem, "bars": len(df)})
        except Exception:
            stocks.append({"id": stem, "label": stem, "bars": 0})
    # Synthetic as first option
    stocks.insert(0, {"id": "__SYNTHETIC__", "label": "★ Synthetic (60d • 5-min • uneven trends)", "bars": 4500})
    return jsonify(stocks)


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    strategy_id = body.get("strategy", "sma")
    source = body.get("source", "synthetic")  # synthetic | real
    symbol = body.get("symbol", "__SYNTHETIC__")
    bars = min(int(body.get("bars", 1500)), MAX_BARS)
    seed = int(body.get("seed", 42))
    commission = float(body.get("commission", 0.0005))
    slippage = float(body.get("slippage", 0.0005))
    # strategy params override
    param_overrides = body.get("params", {})

    # Resolve source
    if symbol == "__SYNTHETIC__":
        source = "synthetic"

    df, err = _get_intraday(source, symbol, bars=bars, seed=seed)
    if err:
        return jsonify({"error": err}), 400
    if df is None or len(df) < 100:
        return jsonify({"error": "Not enough data"}), 400

    # Instantiate strategy with overrides
    cls = BY_ID.get(strategy_id)
    if cls is None:
        return jsonify({"error": f"Unknown strategy {strategy_id}"}), 400

    # Build instance with merged params
    defaults = dict(getattr(cls, "params", {}))
    merged = {**defaults, **{k: v for k, v in param_overrides.items() if k in defaults}}
    try:
        strat = cls(**merged) if merged else cls()
    except TypeError:
        strat = cls()

    result = backtest_run(df, strat, commission=commission, slippage=slippage)

    # Build chart payloads
    ohlc = _df_to_records(df)
    equity = _series_to_records(result["equity"])
    bh_equity = _series_to_records(result["bh_equity"])

    # Indicator overlays (from rated df) — only include numeric indicator columns
    rated = result["rated"]
    # Identify indicator columns (not OHLCV/signal/target)
    skip = {"Open", "High", "Low", "Close", "Volume", "signal", "target"}
    indicators = {}
    for col in rated.columns:
        if col in skip:
            continue
        s = rated[col].dropna()
        if len(s) == 0:
            continue
        # Sample: send as list of {t, v} aligned to df index
        vals = []
        for ts, v in rated[col].items():
            if pd.notna(v):
                vals.append({"t": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S"), "v": round(float(v), 2)})
        if len(vals) > 20:
            indicators[col] = vals

    # Trades
    trades_df = result["trades"]
    trades = []
    if len(trades_df):
        for _, r in trades_df.iterrows():
            trades.append({
                "t": pd.Timestamp(r["time"]).strftime("%Y-%m-%dT%H:%M:%S"),
                "side": r["side"],
                "price": round(float(r["price"]), 2),
                "entry": round(float(r["entry_price"]), 2) if pd.notna(r["entry_price"]) else None,
                "pnl": round(float(r["pnl"]), 2) if pd.notna(r["pnl"]) else None,
                "pnl_pct": round(float(r["pnl_pct"]), 2) if pd.notna(r["pnl_pct"]) else None,
                "bars_held": int(r["bars_held"]) if pd.notna(r["bars_held"]) else None,
            })

    # Position shading: segments where pos==1
    pos = result["pos"]
    # Compress into segments
    segments = []
    in_pos = False
    seg_start = None
    for ts, v in pos.items():
        v = int(v)
        if v == 1 and not in_pos:
            seg_start = pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")
            in_pos = True
        elif v == 0 and in_pos:
            seg_end = pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")
            segments.append({"from": seg_start, "to": seg_end})
            in_pos = False
    if in_pos:
        segments.append({"from": seg_start, "to": pd.Timestamp(pos.index[-1]).strftime("%Y-%m-%dT%H:%M:%S")})

    return jsonify({
        "ohlc": ohlc,
        "equity": equity,
        "bh_equity": bh_equity,
        "indicators": indicators,
        "trades": trades,
        "segments": segments,
        "metrics": result["metrics"],
        "strategy": {
            "id": strategy_id,
            "name": strat.name,
            "logic": strat.logic_text() if hasattr(strat, "logic_text") else "",
            "color": getattr(strat, "color", "#888"),
            "params": merged,
        },
        "meta": {
            "bars": len(df),
            "from": df.index[0].strftime("%Y-%m-%dT%H:%M:%S"),
            "to": df.index[-1].strftime("%Y-%m-%dT%H:%M:%S"),
            "source": source,
            "symbol": symbol,
        }
    })


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    print(f"  Intraday Simulator  →  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
