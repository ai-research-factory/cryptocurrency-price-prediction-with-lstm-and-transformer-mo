"""Data loading, validation, and preprocessing from ARF Data API."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def fetch_ohlcv(
    ticker: str,
    interval: str = "1d",
    period: str = "5y",
    cache_dir: str = "data",
) -> pd.DataFrame:
    """Fetch OHLCV data from ARF Data API with local caching."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    safe_ticker = ticker.replace("/", "_").replace("^", "")
    cache_file = cache_path / f"{safe_ticker}_{interval}_{period}.csv"

    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        url = f"{API_BASE}?ticker={ticker}&interval={interval}&period={period}"
        logger.info(f"Fetching data from API: {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        cache_file.write_text(resp.text)
        df = pd.read_csv(cache_file)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Ensure standard column names
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            col_map[col] = lower
    if col_map:
        df = df.rename(columns=col_map)

    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df[REQUIRED_COLS].dropna()
    logger.info(f"Loaded {len(df)} rows for {ticker}")
    return df


def fetch_multiple_tickers(
    tickers: list[str],
    interval: str = "1d",
    period: str = "5y",
    cache_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers."""
    results = {}
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, interval, period, cache_dir)
            results[ticker] = df
            logger.info(f"Fetched {ticker}: {len(df)} rows, "
                        f"{df.index.min().date()} to {df.index.max().date()}")
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
    return results


def compute_returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Compute log returns."""
    return np.log(df[col] / df[col].shift(1))


def validate_ohlcv(df: pd.DataFrame) -> dict:
    """Validate OHLCV data quality and return a report."""
    report = {
        "n_rows": len(df),
        "date_range": (str(df.index.min()), str(df.index.max())),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_indices": int(df.index.duplicated().sum()),
        "zero_volume_rows": int((df["volume"] == 0).sum()),
        "negative_prices": int((df[["open", "high", "low", "close"]] < 0).any(axis=1).sum()),
        "ohlc_violations": 0,
        "issues": [],
    }

    # Check high >= low
    violations = df["high"] < df["low"]
    n_violations = int(violations.sum())
    report["ohlc_violations"] = n_violations
    if n_violations > 0:
        report["issues"].append(f"{n_violations} rows where high < low")

    if report["duplicate_indices"] > 0:
        report["issues"].append(f"{report['duplicate_indices']} duplicate timestamps")

    if report["negative_prices"] > 0:
        report["issues"].append(f"{report['negative_prices']} rows with negative prices")

    if report["zero_volume_rows"] > 0:
        pct = report["zero_volume_rows"] / len(df) * 100
        report["issues"].append(f"{report['zero_volume_rows']} zero-volume rows ({pct:.1f}%)")

    # Check for large gaps in daily data
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        large_gaps = time_diffs[time_diffs > median_diff * 5]
        if len(large_gaps) > 0:
            report["large_gaps"] = len(large_gaps)
            report["issues"].append(f"{len(large_gaps)} large time gaps detected")

    report["is_clean"] = len(report["issues"]) == 0
    return report


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV data: remove duplicates, handle missing values, fix anomalies."""
    original_len = len(df)
    cleaned = df.copy()

    # Remove duplicate indices (keep last)
    if cleaned.index.duplicated().any():
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        logger.info(f"Removed {original_len - len(cleaned)} duplicate rows")

    # Remove rows with negative prices
    price_cols = ["open", "high", "low", "close"]
    neg_mask = (cleaned[price_cols] < 0).any(axis=1)
    if neg_mask.any():
        cleaned = cleaned[~neg_mask]
        logger.info(f"Removed {neg_mask.sum()} rows with negative prices")

    # Fix high < low by swapping
    swap_mask = cleaned["high"] < cleaned["low"]
    if swap_mask.any():
        cleaned.loc[swap_mask, ["high", "low"]] = (
            cleaned.loc[swap_mask, ["low", "high"]].values
        )
        logger.info(f"Fixed {swap_mask.sum()} rows where high < low")

    # Forward-fill small gaps in price data (up to 3 periods)
    if cleaned.isna().any().any():
        cleaned = cleaned.ffill(limit=3)
        remaining_na = cleaned.isna().sum().sum()
        if remaining_na > 0:
            cleaned = cleaned.dropna()
            logger.info(f"Dropped {remaining_na} remaining NaN values after forward fill")

    # Remove extreme outliers in returns (>10 std from rolling mean)
    log_ret = np.log(cleaned["close"] / cleaned["close"].shift(1))
    rolling_mean = log_ret.rolling(50, min_periods=10).mean()
    rolling_std = log_ret.rolling(50, min_periods=10).std()
    z_score = (log_ret - rolling_mean) / rolling_std.clip(lower=1e-8)
    outlier_mask = z_score.abs() > 10
    if outlier_mask.any():
        n_outliers = outlier_mask.sum()
        cleaned = cleaned[~outlier_mask]
        logger.info(f"Removed {n_outliers} extreme return outliers (|z| > 10)")

    logger.info(f"Cleaned data: {original_len} -> {len(cleaned)} rows")
    return cleaned


def compute_data_summary(df: pd.DataFrame, ticker: str = "") -> dict:
    """Compute summary statistics for a dataset."""
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    return {
        "ticker": ticker,
        "n_rows": len(df),
        "date_range": [str(df.index.min()), str(df.index.max())],
        "price_range": [float(df["close"].min()), float(df["close"].max())],
        "mean_daily_return": float(log_ret.mean()),
        "std_daily_return": float(log_ret.std()),
        "annualized_volatility": float(log_ret.std() * np.sqrt(252)),
        "skewness": float(log_ret.skew()),
        "kurtosis": float(log_ret.kurtosis()),
        "mean_volume": float(df["volume"].mean()),
    }
