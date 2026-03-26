"""Technical indicators: momentum and volatility."""

import numpy as np
import pandas as pd


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


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators and return feature DataFrame."""
    features = pd.DataFrame(index=df.index)

    # Momentum indicators
    features["rsi"] = rsi(df["close"])
    macd_line, macd_signal, macd_hist = macd(df["close"])
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_hist
    features["roc_10"] = rate_of_change(df["close"], 10)
    features["stoch_k"] = stochastic_oscillator(df["high"], df["low"], df["close"])

    # Volatility indicators
    features["atr"] = atr(df["high"], df["low"], df["close"])
    bb_upper, bb_mid, bb_lower, bb_bw = bollinger_bands(df["close"])
    features["bb_bandwidth"] = bb_bw
    features["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    features["hist_vol"] = historical_volatility(df["close"])

    # Price-based
    features["log_return"] = np.log(df["close"] / df["close"].shift(1))
    features["volume_change"] = df["volume"].pct_change()

    return features
