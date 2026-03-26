"""Data loading and preprocessing from ARF Data API."""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"


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

    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df[required].dropna()
    logger.info(f"Loaded {len(df)} rows for {ticker}")
    return df


def compute_returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Compute log returns."""
    return np.log(df[col] / df[col].shift(1))
