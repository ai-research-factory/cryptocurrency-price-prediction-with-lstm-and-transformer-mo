"""Tests for evaluation metrics (Cycle 3 enhanced)."""

import numpy as np
import pytest

from src.evaluation import (
    compute_trading_metrics,
    compute_naive_baseline,
    compute_significance_vs_baseline,
    cost_sensitivity_analysis,
    get_annualization_factor,
    _compute_adaptive_window_sizes,
)


def test_trading_metrics_structure():
    preds = np.array([0.01, -0.01, 0.02, 0.01, -0.005])
    actuals = np.array([0.005, -0.003, 0.01, -0.002, 0.001])
    metrics = compute_trading_metrics(preds, actuals)

    required_keys = [
        "total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
        "calmar_ratio", "win_rate", "n_trades", "n_periods",
        "total_cost", "mean_period_return", "std_period_return",
    ]
    for key in required_keys:
        assert key in metrics


def test_all_positive_predictions():
    preds = np.ones(100) * 0.01
    actuals = np.ones(100) * 0.001
    metrics = compute_trading_metrics(preds, actuals, cost_bps=0)
    assert metrics["total_return"] > 0


def test_cost_reduces_return():
    preds = np.ones(50) * 0.01
    actuals = np.ones(50) * 0.001
    no_cost = compute_trading_metrics(preds, actuals, cost_bps=0)
    with_cost = compute_trading_metrics(preds, actuals, cost_bps=10)
    assert with_cost["total_return"] <= no_cost["total_return"]


def test_naive_baseline():
    actuals = np.ones(100) * 0.001
    bl = compute_naive_baseline(actuals, cost_bps=0)
    assert bl["total_return"] > 0
    assert "sharpe_ratio" in bl
    assert "sortino_ratio" in bl
    assert "calmar_ratio" in bl


def test_annualization_factors():
    assert get_annualization_factor("1d") == 252
    assert get_annualization_factor("1h") == 252 * 24
    assert get_annualization_factor("unknown") == 252  # default


def test_hourly_annualization():
    """Hourly data should use a larger annualization factor."""
    rng = np.random.RandomState(123)
    preds = np.ones(200) * 0.001
    actuals = rng.randn(200) * 0.001 + 0.0005  # noisy positive returns
    m_daily = compute_trading_metrics(preds, actuals, cost_bps=0, interval="1d")
    m_hourly = compute_trading_metrics(preds, actuals, cost_bps=0, interval="1h")
    # Same data, different annualization → hourly Sharpe should be larger
    assert abs(m_hourly["sharpe_ratio"]) > abs(m_daily["sharpe_ratio"])


def test_sortino_ratio():
    preds = np.ones(100) * 0.01
    actuals = np.random.RandomState(42).randn(100) * 0.01
    metrics = compute_trading_metrics(preds, actuals, cost_bps=0)
    assert "sortino_ratio" in metrics
    # Sortino should differ from Sharpe when returns are asymmetric
    assert metrics["sortino_ratio"] != metrics["sharpe_ratio"]


def test_total_cost_tracked():
    preds = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
    actuals = np.zeros(5)
    metrics = compute_trading_metrics(preds, actuals, cost_bps=10)
    assert metrics["total_cost"] > 0
    assert metrics["n_trades"] > 0


def test_significance_test():
    strategy = np.ones(100) * 0.01
    baseline = np.zeros(100)
    sig = compute_significance_vs_baseline(strategy, baseline)
    assert "t_stat" in sig
    assert "p_value" in sig
    assert sig["significant_5pct"] is True  # clearly different

    # Identical returns → not significant
    same = np.ones(100) * 0.01
    sig2 = compute_significance_vs_baseline(same, same)
    assert sig2["significant_5pct"] is False


def test_significance_insufficient_data():
    sig = compute_significance_vs_baseline(np.array([1.0]), np.array([0.0]))
    assert sig["p_value"] == 1.0


def test_cost_sensitivity_analysis():
    preds = np.ones(100) * 0.01
    actuals = np.ones(100) * 0.001
    levels = [0.0, 5.0, 10.0, 20.0]
    results = cost_sensitivity_analysis(preds, actuals, levels)
    assert len(results) == 4
    # Higher cost → lower return
    returns = [r["total_return"] for r in results]
    for i in range(len(returns) - 1):
        assert returns[i] >= returns[i + 1]


def test_adaptive_window_sizes():
    # Should guarantee at least 3 windows
    train, test, step = _compute_adaptive_window_sizes(n_samples=400, seq_len=30)
    assert train >= 200
    assert test >= 30
    # Verify at least 3 windows fit
    n_windows = 0
    start = 0
    while start + train + test + 30 <= 400:
        n_windows += 1
        start += step
    assert n_windows >= 3


def test_adaptive_window_small_dataset():
    train, test, step = _compute_adaptive_window_sizes(n_samples=300, seq_len=30)
    assert train >= 200
    assert test >= 30
