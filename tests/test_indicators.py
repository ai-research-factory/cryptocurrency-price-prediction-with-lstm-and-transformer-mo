"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    rsi,
    macd,
    rate_of_change,
    stochastic_oscillator,
    atr,
    bollinger_bands,
    historical_volatility,
    compute_all_indicators,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n).astype(float)

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_rsi_range(sample_ohlcv):
    result = rsi(sample_ohlcv["close"])
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd_returns_three(sample_ohlcv):
    line, signal, hist = macd(sample_ohlcv["close"])
    assert len(line) == len(sample_ohlcv)
    assert len(signal) == len(sample_ohlcv)
    # histogram = line - signal
    valid_idx = line.dropna().index.intersection(signal.dropna().index)
    np.testing.assert_allclose(
        hist.loc[valid_idx].values,
        (line.loc[valid_idx] - signal.loc[valid_idx]).values,
        atol=1e-10,
    )


def test_roc(sample_ohlcv):
    result = rate_of_change(sample_ohlcv["close"], 10)
    assert len(result) == len(sample_ohlcv)
    assert result.iloc[10:].notna().all()


def test_stochastic_range(sample_ohlcv):
    result = stochastic_oscillator(
        sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
    )
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_atr_positive(sample_ohlcv):
    result = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
    valid = result.dropna()
    assert (valid > 0).all()


def test_bollinger_bands(sample_ohlcv):
    upper, middle, lower, bw = bollinger_bands(sample_ohlcv["close"])
    valid_idx = upper.dropna().index
    assert (upper.loc[valid_idx] >= middle.loc[valid_idx]).all()
    assert (middle.loc[valid_idx] >= lower.loc[valid_idx]).all()
    assert (bw.loc[valid_idx] > 0).all()


def test_hist_vol_positive(sample_ohlcv):
    result = historical_volatility(sample_ohlcv["close"])
    valid = result.dropna()
    assert (valid >= 0).all()


def test_compute_all_indicators(sample_ohlcv):
    features = compute_all_indicators(sample_ohlcv)
    expected_cols = [
        "rsi", "macd", "macd_signal", "macd_hist", "roc_10",
        "stoch_k", "atr", "bb_bandwidth", "bb_pct", "hist_vol",
        "log_return", "volume_change",
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"
    assert len(features) == len(sample_ohlcv)
