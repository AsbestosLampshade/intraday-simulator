# Intraday Simulator

Interactive 5-minute intraday backtesting simulator with visual buy/sell tracing, volume, and P&L. Compare 9 conventional trading strategies on real Indian stock data or synthetic intraday paths.

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Flask](https://img.shields.io/badge/flask-3.x-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

- **Candlestick chart** — 5-min OHLC with indicator overlays, ▲ BUY / ▼ SELL markers, and green shading when long
- **Volume bars** — per-bar volume (green up / red down)
- **Equity curve** — strategy vs dotted Buy & Hold
- **Trade log** — entry, exit, P&L in ₹ and %, bars held
- **KPIs** — total return, CAGR, Sharpe, max drawdown, win rate, profit factor, best/worst trade
- **Strategy logic panel** — plain-English explanation of what each strategy does
- **Compare all 9** — one click runs every strategy on the same data and ranks them
- **Two data sources** — 35 real Indian stocks (expanded to 5m via Brownian bridge) or fully synthetic intraday with U-shaped volatility and fat tails

## Strategies

| Strategy | Type | Idea |
|---|---|---|
| Buy & Hold | Benchmark | Buy once, never sell |
| SMA Crossover (20/50) | Trend | Fast SMA > Slow SMA → long |
| EMA Crossover (12/26) | Trend | Exponentially-weighted crossover — faster |
| MACD (12/26/9) | Trend | MACD line > Signal line → long |
| RSI Reversal (14, 30/70) | Mean-reversion | Buy oversold, sell overbought |
| Bollinger Reversion (20, 2σ) | Mean-reversion | Buy below lower band, exit above upper |
| Momentum (60 bars) | Trend | Trailing return > 0 → long |
| Donchian Breakout (55) | Trend | Breakout above N-bar high → long |
| Z-Score Reversion (20, 1.5) | Mean-reversion | Z < −1.5 → long, Z > 0 → flat |

> Execution: signals computed on bar *t* execute at bar *t+1* open (no look-ahead). Commission + slippage charged on every position flip.

## Quick Start

```bash
pip install -r requirements.txt
python app.py --port 8766
# open http://127.0.0.1:8766
```

### Controls

- **Stock / source** — any of the 35 CSVs in `~/data/StockData/CSV/` or ★ Synthetic
- **Bars** — 750 / 1500 / 3000 / 4500 (10–60 trading days)
- **Seed** — changes the synthetic path (Shuffle = random seed)
- **Strategy params** — edit fast/slow/period/std live in the sidebar

### Data

Place daily OHLCV CSVs (columns: `Date,Open,High,Low,Close,Volume`) in `data/StockData/CSV/` or point `CSV_DIR` in `app.py` to your location. Synthetic mode needs no external data.

## Project Structure

```
intraday-sim/
├── app.py                  # Flask server + API
├── requirements.txt
├── simulator/
│   ├── synthetic.py        # Intraday generation + daily→5m expansion
│   ├── strategies.py       # 9 strategy definitions
│   └── backtester.py       # Event-driven backtester
└── templates/
    └── index.html          # Frontend (Plotly.js)
```

## License

MIT
