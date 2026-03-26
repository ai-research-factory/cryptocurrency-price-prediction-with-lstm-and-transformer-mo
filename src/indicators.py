"""Technical indicators: momentum and volatility.

Cycle 6: Added frequency-adaptive indicator periods for hourly data.
"""

import numpy as np
import pandas as pd


# Default indicator periods (designed for daily data)
DEFAULT_PERIODS = {
    "rsi": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "roc": 10,
    "stoch": 14,
    "atr": 14,
    "bb": 20,
    "hist_vol": 20,
}

# Hourly periods: scale by ~24 to represent one daily cycle in hourly bars
# Use round numbers that are practical for hourly trading
HOURLY_PERIODS = {
    "rsi": 24,
    "macd_fast": 24,
    "macd_slow": 48,
    "macd_signal": 12,
    "roc": 24,
    "stoch": 24,
    "atr": 24,
    "bb": 48,
    "hist_vol": 48,
}


def get_indicator_periods(interval: str = "1d") -> dict:
    """Return indicator periods appropriate for the data frequency.

    Cycle 6: Adjusts indicator lookback windows for hourly data to align
    with daily cycles, addressing OQ #2 (indicator periods for different frequencies).
    """
    if interval in ("1h", "2h"):
        return HOURLY_PERIODS
    elif interval in ("4h",):
        # 4h: scale by ~6 from daily
        return {
            "rsi": 18,
            "macd_fast": 18,
            "macd_slow": 36,
            "macd_signal": 9,
            "roc": 18,
            "stoch": 18,
            "atr": 18,
            "bb": 30,
            "hist_vol": 30,
        }
    return DEFAULT_PERIODS


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (momentum)."""
    return close.pct_change(periods=period) * 100


def stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Stochastic %K."""
    lowest = low.rolling(period).min()
    highest = high.rolling(period).max()
    return 100 * (close - lowest) / (highest - lowest)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range (volatility)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands: upper, middle, lower, and bandwidth."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    return upper, middle, lower, bandwidth


def historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualized historical volatility."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252)


def compute_all_indicators(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """Compute all technical indicators and return feature DataFrame.

    Cycle 6: Added interval parameter for frequency-adaptive indicator periods.
    """
    periods = get_indicator_periods(interval)
    features = pd.DataFrame(index=df.index)

    # Momentum indicators
    features["rsi"] = rsi(df["close"], period=periods["rsi"])
    macd_line, macd_signal_line, macd_hist = macd(
        df["close"], fast=periods["macd_fast"],
        slow=periods["macd_slow"], signal=periods["macd_signal"],
    )
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal_line
    features["macd_hist"] = macd_hist
    features["roc_10"] = rate_of_change(df["close"], periods["roc"])
    features["stoch_k"] = stochastic_oscillator(
        df["high"], df["low"], df["close"], period=periods["stoch"],
    )

    # Volatility indicators
    features["atr"] = atr(df["high"], df["low"], df["close"], period=periods["atr"])
    bb_upper, bb_mid, bb_lower, bb_bw = bollinger_bands(
        df["close"], period=periods["bb"],
    )
    features["bb_bandwidth"] = bb_bw
    features["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    features["hist_vol"] = historical_volatility(df["close"], period=periods["hist_vol"])

    # Price-based
    features["log_return"] = np.log(df["close"] / df["close"].shift(1))
    features["volume_change"] = df["volume"].pct_change()

    return features
