"""Tests for data loading, validation, cleaning, and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data import validate_ohlcv, clean_ohlcv, compute_data_summary, compute_returns


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
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


def test_validate_clean_data(sample_ohlcv):
    report = validate_ohlcv(sample_ohlcv)
    assert report["n_rows"] == 200
    assert report["missing_values"] == 0
    assert report["is_clean"]


def test_validate_detects_duplicates(sample_ohlcv):
    # Add duplicate rows
    dup = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:3]])
    report = validate_ohlcv(dup)
    assert report["duplicate_indices"] == 3
    assert not report["is_clean"]


def test_validate_detects_negative_prices(sample_ohlcv):
    bad = sample_ohlcv.copy()
    bad.iloc[5, bad.columns.get_loc("close")] = -1.0
    report = validate_ohlcv(bad)
    assert report["negative_prices"] >= 1
    assert not report["is_clean"]


def test_validate_detects_ohlc_violations(sample_ohlcv):
    bad = sample_ohlcv.copy()
    bad.iloc[10, bad.columns.get_loc("high")] = bad.iloc[10]["low"] - 1
    report = validate_ohlcv(bad)
    assert report["ohlc_violations"] >= 1


def test_clean_removes_duplicates(sample_ohlcv):
    dup = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:3]])
    cleaned = clean_ohlcv(dup)
    assert not cleaned.index.duplicated().any()


def test_clean_removes_negative_prices(sample_ohlcv):
    bad = sample_ohlcv.copy()
    bad.iloc[5, bad.columns.get_loc("close")] = -1.0
    cleaned = clean_ohlcv(bad)
    assert (cleaned["close"] >= 0).all()


def test_clean_fixes_high_low_swap(sample_ohlcv):
    bad = sample_ohlcv.copy()
    idx = bad.index[10]
    orig_high = bad.loc[idx, "high"]
    orig_low = bad.loc[idx, "low"]
    bad.loc[idx, "high"] = orig_low - 0.5
    bad.loc[idx, "low"] = orig_high + 0.5
    cleaned = clean_ohlcv(bad)
    assert (cleaned["high"] >= cleaned["low"]).all()


def test_compute_data_summary(sample_ohlcv):
    summary = compute_data_summary(sample_ohlcv, "TEST")
    assert summary["ticker"] == "TEST"
    assert summary["n_rows"] == 200
    assert "mean_daily_return" in summary
    assert "annualized_volatility" in summary
    assert "skewness" in summary
    assert "kurtosis" in summary


def test_compute_returns(sample_ohlcv):
    ret = compute_returns(sample_ohlcv)
    assert len(ret) == len(sample_ohlcv)
    assert ret.iloc[0] != ret.iloc[0]  # first is NaN
    valid = ret.dropna()
    assert len(valid) == len(sample_ohlcv) - 1
