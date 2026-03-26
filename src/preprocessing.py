"""Data preprocessing pipeline for real market data.

Handles feature normalization, stationarity transforms, and data quality
checks specific to the indicator-based feature matrix.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_price_indicators(features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Normalize price-scale indicators relative to price level.

    ATR and MACD are in absolute price units, which makes them non-stationary.
    Dividing by close price makes them comparable across time and tickers.
    """
    normalized = features.copy()

    close = ohlcv["close"]

    # ATR as percentage of price
    if "atr" in normalized.columns:
        normalized["atr"] = normalized["atr"] / close

    # MACD as percentage of price
    if "macd" in normalized.columns:
        normalized["macd"] = normalized["macd"] / close
    if "macd_signal" in normalized.columns:
        normalized["macd_signal"] = normalized["macd_signal"] / close
    if "macd_hist" in normalized.columns:
        normalized["macd_hist"] = normalized["macd_hist"] / close

    return normalized


def clip_extreme_values(features: pd.DataFrame, n_std: float = 5.0) -> pd.DataFrame:
    """Clip feature values beyond n_std standard deviations from rolling median.

    Uses rolling statistics to avoid look-ahead bias.
    """
    clipped = features.copy()
    for col in clipped.columns:
        series = clipped[col]
        rolling_median = series.rolling(100, min_periods=20).median()
        rolling_std = series.rolling(100, min_periods=20).std()
        lower = rolling_median - n_std * rolling_std
        upper = rolling_median + n_std * rolling_std
        clipped[col] = series.clip(lower=lower, upper=upper)
    return clipped


def handle_inf_values(features: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values with NaN, then forward-fill."""
    cleaned = features.replace([np.inf, -np.inf], np.nan)
    n_inf = features.isin([np.inf, -np.inf]).sum().sum()
    if n_inf > 0:
        logger.info(f"Replaced {n_inf} infinite values")
    cleaned = cleaned.ffill(limit=3)
    return cleaned


def preprocess_features(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    clip_std: float = 5.0,
) -> pd.DataFrame:
    """Full preprocessing pipeline for feature matrix.

    1. Normalize price-scale indicators for stationarity
    2. Handle inf/NaN values
    3. Clip extreme values
    """
    processed = normalize_price_indicators(features, ohlcv)
    processed = handle_inf_values(processed)
    processed = clip_extreme_values(processed, n_std=clip_std)
    return processed


def compute_feature_stats(features: pd.DataFrame) -> dict:
    """Compute and return feature-level statistics for reporting."""
    stats = {}
    for col in features.columns:
        series = features[col].dropna()
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "pct_nan": float(features[col].isna().mean() * 100),
            "skew": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }
    return stats
