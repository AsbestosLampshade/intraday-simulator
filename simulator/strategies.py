"""Intraday strategy library.

Each strategy is a class exposing:
  - `name`, `params`, `description`  (for the UI)
  - `logic_text()`  -> markdown-ish explanation shown in the simulator
  - `compute(df)`   -> returns a DataFrame with signal/target/indicator columns
    merged onto df. The simulator uses the `target` column to drive
    execution (executed at next bar open).

Indicators are computed bar-by-bar using only past data (no lookahead).
"""
import numpy as np
import pandas as pd


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def bbands(close, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    return mid, mid + k*sd, mid - k*sd

def atr(df, n=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def zscore(close, n=20):
    m = sma(close, n)
    sd = close.rolling(n).std()
    return (close - m) / sd.replace(0, np.nan)


class BuyAndHold:
    name = "Buy & Hold"
    id = "buyhold"
    params = {}
    color = "#888888"

    description = "Buy at the first bar and never sell. The passive benchmark."

    def logic_text(self):
        return (
            "Buy one unit at the open of the first bar and hold to the end. "
            "No signals, no indicator. Return equals the asset's own drift."
        )

    def compute(self, df):
        out = df.copy()
        out["target"] = 1
        out["signal"] = 1
        return out


class SMACrossover:
    name = "SMA Crossover"
    id = "sma"
    params = {"fast": 20, "slow": 50}
    color = "#e67e22"

    def __init__(self, fast=20, slow=50):
        self.fast, self.slow = fast, slow

    description = "Go long when a fast SMA crosses above a slow SMA."

    def logic_text(self):
        return (
            f"Fast SMA({self.fast}) vs Slow SMA({self.slow}). "
            "Signal flips long when the short-term average overtakes the "
            "long-term average (momentum turning up), flat when it falls below. "
            "Classic trend-following: buys strength, avoids drawdowns by cutting "
            "exposure when trend reverses."
        )

    def compute(self, df):
        out = df.copy()
        out[f"SMA{self.fast}"] = sma(df["Close"], self.fast)
        out[f"SMA{self.slow}"] = sma(df["Close"], self.slow)
        out["signal"] = (out[f"SMA{self.fast}"] > out[f"SMA{self.slow}"]).astype(int)
        out["target"] = out["signal"]
        return out


class EMACrossover:
    name = "EMA Crossover"
    id = "ema"
    params = {"fast": 12, "slow": 26}
    color = "#3498db"

    def __init__(self, fast=12, slow=26):
        self.fast, self.slow = fast, slow

    description = "EMA-based crossover — reacts faster than SMA."

    def logic_text(self):
        return (
            f"Same idea as SMA crossover but with exponential weighting (α=2/(n+1)). "
            f"EMA({self.fast}) vs EMA({self.slow}). Recent prices matter more, so "
            "crossovers trigger sooner. Earlier entries/exits, but more whipsaw in "
            "choppy conditions."
        )

    def compute(self, df):
        out = df.copy()
        out[f"EMA{self.fast}"] = ema(df["Close"], self.fast)
        out[f"EMA{self.slow}"] = ema(df["Close"], self.slow)
        out["signal"] = (out[f"EMA{self.fast}"] > out[f"EMA{self.slow}"]).astype(int)
        out["target"] = out["signal"]
        return out


class MACDStrategy:
    name = "MACD"
    id = "macd"
    params = {"fast": 12, "slow": 26, "signal": 9}
    color = "#9b59b6"

    def __init__(self, fast=12, slow=26, signal=9):
        self.fast, self.slow, self.signal = fast, slow, signal

    description = "MACD line crossing its signal line."

    def logic_text(self):
        return (
            f"MACD = EMA({self.fast}) − EMA({self.slow}); Signal = EMA(MACD, {self.signal}). "
            "When MACD crosses above Signal, momentum is accelerating → long; "
            "below → flat. Measures the *rate of change* of momentum rather than "
            "price level alone."
        )

    def compute(self, df):
        out = df.copy()
        out["MACD"] = ema(df["Close"], self.fast) - ema(df["Close"], self.slow)
        out["MACD_signal"] = ema(out["MACD"], self.signal)
        out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
        out["signal"] = (out["MACD"] > out["MACD_signal"]).astype(int)
        out["target"] = out["signal"]
        return out


class RSIReversal:
    name = "RSI Reversal"
    id = "rsi"
    params = {"period": 14, "buy": 30, "sell": 70}
    color = "#2ecc71"

    def __init__(self, period=14, buy=30, sell=70):
        self.period, self.buy, self.sell = period, buy, sell

    description = "Mean-reversion: buy oversold, sell overbought."

    def logic_text(self):
        return (
            f"RSI({self.period}) with thresholds {self.buy}/{self.sell}. "
            "RSI < buy → oversold → enter long (bet on bounce). "
            "RSI > sell → overbought → exit. Opposite philosophy to trend-"
            "following: buys weakness. Works in ranges, fails in trends/"
            "regime changes (catches falling knives)."
        )

    def compute(self, df):
        out = df.copy()
        out["RSI"] = rsi(df["Close"], self.period)
        pos = pd.Series(0, index=df.index, dtype=int)
        in_pos = False
        for i in range(len(df)):
            v = out["RSI"].iloc[i]
            if pd.isna(v):
                continue
            if not in_pos and v < self.buy:
                in_pos = True
            elif in_pos and v > self.sell:
                in_pos = False
            pos.iloc[i] = int(in_pos)
        out["signal"] = pos
        out["target"] = pos
        return out


class BollingerReversion:
    name = "Bollinger Reversion"
    id = "bb"
    params = {"period": 20, "std": 2.0}
    color = "#e74c3c"

    def __init__(self, period=20, std=2.0):
        self.period, self.std = period, std

    description = "Buy when price breaks below the lower Bollinger band."

    def logic_text(self):
        return (
            f"Bollinger({self.period}, {self.std}σ). Middle = SMA, bands = ±{self.std}×rolling σ. "
            "Price < lower band → statistically extreme → long (expect reversion "
            "to the mean). Exit when price hits the upper band. Bands widen in "
            "volatile regimes, which makes this strategy *widen its entry* just "
            "when it should be most cautious."
        )

    def compute(self, df):
        out = df.copy()
        mid, upper, lower = bbands(df["Close"], self.period, self.std)
        out["BB_mid"] = mid
        out["BB_up"] = upper
        out["BB_low"] = lower
        pos = pd.Series(0, index=df.index, dtype=int)
        in_pos = False
        for i in range(len(df)):
            c = df["Close"].iloc[i]
            lo = lower.iloc[i]
            hi = upper.iloc[i]
            if pd.isna(lo) or pd.isna(hi):
                continue
            if not in_pos and c < lo:
                in_pos = True
            elif in_pos and c > hi:
                in_pos = False
            pos.iloc[i] = int(in_pos)
        out["signal"] = pos
        out["target"] = pos
        return out


class MomentumStrategy:
    name = "Momentum"
    id = "mom"
    params = {"lookback": 60}
    color = "#1abc9c"

    def __init__(self, lookback=60):
        # 60 bars ~= 1 trading day at 5m (intraday: 60×5m = 5h). Keep daily-like length
        self.lookback = lookback

    description = "Hold if trailing return is positive."

    def logic_text(self):
        return (
            f"Trailing {self.lookback}-bar return (roughly one trading day at 5-min bars). "
            "If the stock is up over the lookback → long; if down → flat. The "
            "momentum anomaly is one of the most robust in finance: winners "
            "tend to keep winning. Slow to enter the first leg of a trend, "
            "slow to exit a reversal — but the most defensive trend strategy."
        )

    def compute(self, df):
        out = df.copy()
        out["mom"] = df["Close"].pct_change(self.lookback)
        out["signal"] = (out["mom"] > 0).fillna(0).astype(int)
        out["target"] = out["signal"]
        return out


class DonchianBreakout:
    name = "Donchian Breakout"
    id = "donchian"
    params = {"lookback": 55}
    color = "#f39c12"

    def __init__(self, lookback=55):
        self.lookback = lookback

    description = "Breakout: long above N-bar high, flat below N-bar low."

    def logic_text(self):
        return (
            f"Donchian({self.lookback}). Upper = max(High, last {self.lookback} bars), "
            f"Lower = min(Low, last {self.lookback} bars). Close > Upper → breakout up → long; "
            "Close < Lower → breakdown → flat. Captures explosive moves. The most "
            "defensive trend strategy: it was the best at limiting bear losses."
        )

    def compute(self, df):
        out = df.copy()
        upper = df["High"].rolling(self.lookback).max().shift(1)
        lower = df["Low"].rolling(self.lookback).min().shift(1)
        out["DON_up"] = upper
        out["DON_low"] = lower
        pos = pd.Series(0, index=df.index, dtype=int)
        in_pos = False
        for i in range(len(df)):
            c = df["Close"].iloc[i]
            u = upper.iloc[i]
            lo = lower.iloc[i]
            if not in_pos and pd.notna(u) and c > u:
                in_pos = True
            elif in_pos and pd.notna(lo) and c < lo:
                in_pos = False
            pos.iloc[i] = int(in_pos)
        out["signal"] = pos
        out["target"] = pos
        return out


class ZScoreReversion:
    name = "Z-Score Reversion"
    id = "zscore"
    params = {"period": 20, "entry": 1.5, "exit": 0.0}
    color = "#34495e"

    def __init__(self, period=20, entry=1.5, exit=0.0):
        self.period, self.entry, self.exit = period, entry, exit

    description = "Statistical mean-reversion on z-score."

    def logic_text(self):
        return (
            f"Z = (Close − SMA{self.period}) / rolling σ. Z < −{self.entry} → extreme "
            f"low → long; Z > {self.exit} → back to mean → flat. A normalized, "
            "volatility-adjusted signal. Elegant statistically, but it assumes "
            "daily prices are stationary. They aren't — so it double-counts "
            "losers during trends and was the worst performer in the 10-year test."
        )

    def compute(self, df):
        out = df.copy()
        out["Z"] = zscore(df["Close"], self.period)
        pos = pd.Series(0, index=df.index, dtype=int)
        in_pos = False
        for i in range(len(df)):
            z = out["Z"].iloc[i]
            if pd.isna(z):
                continue
            if not in_pos and z < -self.entry:
                in_pos = True
            elif in_pos and z > self.exit:
                in_pos = False
            pos.iloc[i] = int(in_pos)
        out["signal"] = pos
        out["target"] = pos
        return out


ALL = [
    BuyAndHold,
    SMACrossover,
    EMACrossover,
    MACDStrategy,
    RSIReversal,
    BollingerReversion,
    MomentumStrategy,
    DonchianBreakout,
    ZScoreReversion,
]

BY_ID = {cls.id: cls for cls in ALL}
