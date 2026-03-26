"""Tests for data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import compute_all_indicators
from src.preprocessing import (
    normalize_price_indicators,
    clip_extreme_values,
    handle_inf_values,
    preprocess_features,
    compute_feature_stats,
)


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 200
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


@pytest.fixture
def sample_features(sample_ohlcv):
    return compute_all_indicators(sample_ohlcv)


def test_normalize_price_indicators(sample_features, sample_ohlcv):
    normalized = normalize_price_indicators(sample_features, sample_ohlcv)
    # ATR should now be relative to price (much smaller values)
    valid_atr = normalized["atr"].dropna()
    assert valid_atr.abs().max() < 1.0  # percentage scale, should be small


def test_normalize_preserves_bounded_indicators(sample_features, sample_ohlcv):
    normalized = normalize_price_indicators(sample_features, sample_ohlcv)
    # RSI should be unchanged (already 0-100 scale)
    pd.testing.assert_series_equal(
        normalized["rsi"], sample_features["rsi"]
    )


def test_handle_inf_values():
    df = pd.DataFrame({
        "a": [1.0, np.inf, 3.0, -np.inf, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    cleaned = handle_inf_values(df)
    assert not np.isinf(cleaned.values).any()
    # inf should be forward-filled
    assert cleaned["a"].iloc[1] == 1.0  # forward fill from 1.0


def test_clip_extreme_values(sample_features):
    # Add an extreme value
    modified = sample_features.copy()
    modified.iloc[150, 0] = modified.iloc[150, 0] + 1000
    clipped = clip_extreme_values(modified, n_std=5.0)
    # The extreme value should be clipped
    assert clipped.iloc[150, 0] < modified.iloc[150, 0]


def test_preprocess_features_full_pipeline(sample_features, sample_ohlcv):
    processed = preprocess_features(sample_features, sample_ohlcv)
    assert processed.shape == sample_features.shape
    assert not np.isinf(processed.values).any()
    # All columns should still be present
    assert list(processed.columns) == list(sample_features.columns)


def test_compute_feature_stats(sample_features):
    stats = compute_feature_stats(sample_features)
    assert set(stats.keys()) == set(sample_features.columns)
    for col, col_stats in stats.items():
        assert "mean" in col_stats
        assert "std" in col_stats
        assert "min" in col_stats
        assert "max" in col_stats
        assert "skew" in col_stats
