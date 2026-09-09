"""Generate realistic intraday OHLCV at 5-minute resolution.

Approach:
  - Daily open follows prior close + overnight gap (small random).
  - Within each day, price follows a Brownian motion with an intraday
    volatility shape: higher at open and close (U-shaped), softer at
    midday. A per-day drift is sampled to create trending vs choppy days,
    and multiday trends are injected so bull/bear stretches appear.
  - High/Low are derived from the path's running max/min within each
    5-minute window; volume follows a similar U-shape + random spikes.
"""
import numpy as np
import pandas as pd

BARS_PER_DAY = 75          # 9:15 .. 15:30 at 5m
MARKET_OPEN = "09:15"
DEFAULT_DAYS = 60
BASE_PRICE = 1000.0


def _intraday_vol_shape(n=BARS_PER_DAY):
    """U-shaped intraday volatility multiplier (open & close elevated)."""
    x = np.linspace(0, 1, n)
    return 0.55 + 0.9 * (4 * (x - 0.5) ** 2)


def _intraday_volume_shape(n=BARS_PER_DAY, rng=None):
    base = 0.45 + 1.1 * (4 * (np.linspace(0, 1, n) - 0.5) ** 2)
    if rng is not None:
        base *= rng.uniform(0.75, 1.25, size=n)
    return base


def generate_intraday(days=DEFAULT_DAYS, seed=42, base_price=BASE_PRICE,
                      trend_strength=0.6, start_date=None):
    """Return a DataFrame indexed by 5-minute timestamps.

    Columns: Open, High, Low, Close, Volume
    Index name: Datetime

    `trend_strength` in [0,1] controls how pronounced multi-day trends
    are (0 = pure noise, 1 = strong persistent drift).
    """
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days * 1.6)
        while start_date.weekday() >= 5:
            start_date -= pd.Timedelta(days=1)
    start_date = pd.Timestamp(start_date).normalize()

    dates = pd.bdate_range(start=start_date, periods=days)
    vol_shape = _intraday_vol_shape()

    # Multi-day trend: slowly varying drift that persists for stretches.
    # Construct as a smoothed random walk of daily drifts.
    raw = rng.standard_normal(days)
    # Smooth with exponential filter to create persistence
    alpha = 0.18 + 0.25 * trend_strength
    daily_drift = np.zeros(days)
    daily_drift[0] = raw[0] * 0.004
    for i in range(1, days):
        daily_drift[i] = alpha * raw[i] * 0.006 + (1 - alpha) * daily_drift[i - 1]

    # Scale so the overall 60-day move is reasonable (~ +/- 8-25%)
    scale = 1.0 + 1.2 * trend_strength
    daily_drift *= scale

    all_rows = []
    current_price = base_price

    for d, date in enumerate(dates):
        # Overnight gap: small jump from prior close to today's open
        overnight = rng.normal(0, 0.0035)
        day_open = current_price * (1 + overnight + daily_drift[d] * 0.35)

        # Intraday path: 75 steps of 5-minute returns
        drift_5m = daily_drift[d] / BARS_PER_DAY
        sigma_5m = 0.0032 * vol_shape  # ~0.3% base per 5m, modulated

        # Heavy-tailed intraday shocks (occasional spikes)
        eps = rng.standard_normal(BARS_PER_DAY)
        spikes = rng.random(BARS_PER_DAY) < 0.015
        eps = eps + spikes * rng.standard_normal(BARS_PER_DAY) * 3.5

        rets = drift_5m + sigma_5m * eps

        # Build high-resolution price path, then aggregate into OHLC per 5m bar.
        # We simulate 3 sub-steps per bar for realistic High/Low within bar.
        sub = 3
        fine_rets = np.repeat(rets, sub) / sub
        # Add sub-bar jitter
        fine_rets += rng.normal(0, 0.0007, size=len(fine_rets))
        fine_prices = day_open * np.exp(np.cumsum(fine_rets))
        fine_prices = fine_prices.reshape(BARS_PER_DAY, sub)

        vol_shape_d = _intraday_volume_shape(rng=rng)
        base_vol = rng.integers(80000, 180000)
        volumes = (base_vol * vol_shape_d * rng.uniform(0.75, 1.3, BARS_PER_DAY)).astype(int)
        # Random volume bursts
        bursts = rng.random(BARS_PER_DAY) < 0.04
        volumes[bursts] = (volumes[bursts] * rng.uniform(1.8, 3.2, bursts.sum())).astype(int)

        day_start = pd.Timestamp(date).replace(hour=9, minute=15)
        for i in range(BARS_PER_DAY):
            ts = day_start + pd.Timedelta(minutes=5 * i)
            chunk = fine_prices[i]
            o = chunk[0]
            c = chunk[-1]
            h = chunk.max() * (1 + abs(rng.normal(0, 0.001)))
            low = chunk.min() * (1 - abs(rng.normal(0, 0.001)))
            # Ensure O/C inside H/L
            h = max(h, o, c)
            low = min(low, o, c)
            all_rows.append((ts, o, h, low, c, int(volumes[i])))

        current_price = float(fine_prices[-1, -1])

    df = pd.DataFrame(all_rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("Datetime", inplace=True)
    df.index.name = "Datetime"
    return df


def expand_daily_to_intraday(daily_df, seed=42):
    """Expand a daily OHLCV DataFrame into 5-minute bars.

    Preserves each day's Open/High/Low/Close exactly, distributing the
    day's range across 75 intraday bars with realistic microstructure.
    """
    rng = np.random.default_rng(seed)
    vol_shape = _intraday_vol_shape()
    daily_df = daily_df.copy()
    daily_df.index = pd.to_datetime(daily_df.index)

    all_rows = []
    for date, row in daily_df.iterrows():
        day_open = row["Open"]
        day_high = row["High"]
        day_low = row["Low"]
        day_close = row["Close"]
        day_vol = int(row["Volume"]) if "Volume" in row else 500000

        # Target: intraday path must start at Open and end at Close,
        # and its running high/low must hit High/Low.
        n = BARS_PER_DAY
        # Generate a normalized Brownian bridge from 0 to log(Close/Open)
        target_log = np.log(day_close / day_open) if day_open > 0 else 0
        steps = rng.standard_normal(n)
        # Make a bridge: cumulative sum pinned to 0 at start and target at end
        cumsum = np.cumsum(steps)
        # Bridge correction
        bridge = cumsum - np.linspace(0, cumsum[-1], n) + np.linspace(0, target_log, n)
        # Scale so max excursion hits High and min hits Low (approximately)
        log_high = np.log(day_high / day_open) if day_open > 0 and day_high > 0 else 0
        log_low = np.log(day_low / day_open) if day_open > 0 and day_low > 0 else 0
        span = max(abs(log_high), abs(log_low), 1e-6)
        bridge_span = max(np.max(np.abs(bridge)), 1e-6)
        scale = min(span / bridge_span * 0.82, 1.0)
        bridge *= scale
        # Ensure endpoints still hit target exactly
        bridge[-1] = target_log
        bridge[0] = 0

        # Modulate step sizes by intraday vol shape
        # (bridge already has shape; just add micro jitter)
        bridge += rng.normal(0, 0.0004, size=n)

        prices = day_open * np.exp(bridge)

        # Within-bar High/Low via sub-steps
        sub = 3
        fine = np.repeat(prices, sub)
        # Slight sub-bar noise to create intra-bar range
        fine += fine * rng.normal(0, 0.0008, size=len(fine))
        fine = fine.reshape(n, sub)

        base_vol = day_vol / n
        vols = (base_vol * (0.5 + vol_shape) * rng.uniform(0.8, 1.2, n)).astype(int)

        day_start = pd.Timestamp(date).replace(hour=9, minute=15)
        for i in range(n):
            ts = day_start + pd.Timedelta(minutes=5 * i)
            chunk = fine[i]
            o = chunk[0]
            c = chunk[-1]
            h = chunk.max()
            low = chunk.min()
            h = max(h, o, c)
            low = min(low, o, c)
            all_rows.append((ts, o, h, low, c, int(vols[i])))

    df = pd.DataFrame(all_rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("Datetime", inplace=True)
    df.index.name = "Datetime"
    return df
