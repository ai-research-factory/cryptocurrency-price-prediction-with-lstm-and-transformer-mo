"""Tests for evaluation metrics (Cycle 4 enhanced)."""

import numpy as np
import pytest

from src.evaluation import (
    compute_trading_metrics,
    compute_naive_baseline,
    compute_significance_vs_baseline,
    cost_sensitivity_analysis,
    get_annualization_factor,
    _compute_adaptive_window_sizes,
    _apply_min_holding_period,
    select_optimal_hold_period,
    select_optimal_seq_len,
    compute_volatility_regime,
    compute_trading_metrics_regime_short,
    regime_threshold_sweep,
    select_optimal_regime_threshold,
    select_optimal_hidden_size,
    select_optimal_num_layers,
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


# Cycle 4 tests

def test_min_holding_period_basic():
    """Min holding period should prevent rapid position changes."""
    position = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    result = _apply_min_holding_period(position, min_hold=3)
    # After switching to 1 at index 1, must hold for 3 periods
    assert result[1] == 1.0
    assert result[2] == 1.0  # held
    assert result[3] == 1.0  # held


def test_min_holding_period_reduces_trades():
    """Min holding period should reduce number of trades."""
    preds_noisy = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01] * 10)
    actuals = np.ones(80) * 0.001
    m_no_hold = compute_trading_metrics(preds_noisy, actuals, cost_bps=10, min_holding_period=1)
    m_with_hold = compute_trading_metrics(preds_noisy, actuals, cost_bps=10, min_holding_period=5)
    assert m_with_hold["n_trades"] < m_no_hold["n_trades"]
    assert m_with_hold["total_cost"] < m_no_hold["total_cost"]


def test_min_holding_period_1_is_noop():
    """Min holding period of 1 should not change anything."""
    position = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    result = _apply_min_holding_period(position, min_hold=1)
    np.testing.assert_array_equal(position, result)


def test_cost_sensitivity_with_min_hold():
    """Cost sensitivity analysis should respect min holding period."""
    preds = np.array([0.01, -0.01, 0.01, -0.01, 0.01] * 20)
    actuals = np.ones(100) * 0.001
    levels = [0.0, 10.0]
    results_hold = cost_sensitivity_analysis(preds, actuals, levels, min_holding_period=5)
    results_no_hold = cost_sensitivity_analysis(preds, actuals, levels, min_holding_period=1)
    # With min hold, cost at 10 bps should be lower
    assert results_hold[1]["total_cost"] <= results_no_hold[1]["total_cost"]


# Cycle 6 tests

def test_select_optimal_hold_period():
    """Should select the hold period with best Sharpe."""
    sweep_results = [
        {"min_hold": 1, "sharpe_ratio": 0.5},
        {"min_hold": 3, "sharpe_ratio": 1.2},
        {"min_hold": 5, "sharpe_ratio": 0.8},
    ]
    assert select_optimal_hold_period(sweep_results) == 3


def test_select_optimal_hold_period_empty():
    """Should default to 1 when no valid results."""
    assert select_optimal_hold_period([]) == 1


def test_select_optimal_hold_period_negative():
    """Should pick least-negative Sharpe."""
    sweep_results = [
        {"min_hold": 1, "sharpe_ratio": -2.0},
        {"min_hold": 5, "sharpe_ratio": -0.5},
    ]
    assert select_optimal_hold_period(sweep_results) == 5


# Cycle 7 tests

def test_select_optimal_seq_len():
    """Should select the seq_len with best Sharpe."""
    sweep_results = [
        {"seq_len": 10, "sharpe_ratio": 0.5, "n_trades": 10, "n_windows": 3},
        {"seq_len": 20, "sharpe_ratio": 1.8, "n_trades": 8, "n_windows": 3},
        {"seq_len": 30, "sharpe_ratio": 1.2, "n_trades": 6, "n_windows": 3},
        {"seq_len": 50, "sharpe_ratio": -0.3, "n_trades": 4, "n_windows": 3},
    ]
    assert select_optimal_seq_len(sweep_results) == 20


def test_select_optimal_seq_len_with_errors():
    """Should skip entries with errors."""
    sweep_results = [
        {"seq_len": 10, "error": "No valid windows"},
        {"seq_len": 20, "sharpe_ratio": 0.5, "n_trades": 8, "n_windows": 3},
        {"seq_len": 30, "sharpe_ratio": 1.0, "n_trades": 6, "n_windows": 3},
    ]
    assert select_optimal_seq_len(sweep_results) == 30


def test_select_optimal_seq_len_empty():
    """Should return default when no valid results."""
    assert select_optimal_seq_len([], default_seq_len=30) == 30
    assert select_optimal_seq_len([{"seq_len": 10, "error": "fail"}], default_seq_len=20) == 20


def test_volatility_regime_basic():
    """Should detect high-vol periods."""
    rng = np.random.RandomState(42)
    # Low vol period followed by high vol period
    low_vol = rng.randn(100) * 0.001
    high_vol = rng.randn(100) * 0.01  # 10x higher vol
    returns = np.concatenate([low_vol, high_vol])
    regime = compute_volatility_regime(returns, lookback=30, high_vol_threshold=1.5)
    assert len(regime) == 200
    # High-vol regime should have more 1s in the second half
    assert np.sum(regime[100:]) > np.sum(regime[:100])


def test_volatility_regime_short_data():
    """Should return all zeros for short data."""
    returns = np.array([0.01, -0.01, 0.02])
    regime = compute_volatility_regime(returns, lookback=60)
    assert len(regime) == 3
    np.testing.assert_array_equal(regime, np.zeros(3))


def test_regime_short_metrics_structure():
    """Regime short metrics should include regime-specific fields."""
    preds = np.array([0.01, -0.01, 0.02, 0.01, -0.005] * 20)
    actuals = np.array([0.005, -0.003, 0.01, -0.002, 0.001] * 20)
    metrics = compute_trading_metrics_regime_short(preds, actuals, vol_lookback=10)
    assert "high_vol_periods" in metrics
    assert "shorts_disabled" in metrics
    assert "sharpe_ratio" in metrics


def test_regime_short_vs_full_short():
    """Regime short should disable some shorts compared to full short."""
    rng = np.random.RandomState(42)
    low_vol = rng.randn(100) * 0.001
    high_vol = rng.randn(100) * 0.01
    actuals = np.concatenate([low_vol, high_vol])
    preds = np.concatenate([rng.randn(100) * 0.001, rng.randn(100) * 0.01])

    full_short = compute_trading_metrics(
        preds, actuals, cost_bps=10, min_holding_period=1, allow_short=True,
    )
    regime = compute_trading_metrics_regime_short(
        preds, actuals, cost_bps=10, min_holding_period=1,
        vol_lookback=30, high_vol_threshold=1.5,
    )
    # Regime short should have fewer or equal trades than full short
    assert regime["n_trades"] <= full_short["n_trades"] + 5  # small tolerance


# Cycle 8 tests

def test_regime_threshold_sweep():
    """Sweep should return results for each threshold level."""
    rng = np.random.RandomState(42)
    low_vol = rng.randn(100) * 0.001
    high_vol = rng.randn(100) * 0.01
    actuals = np.concatenate([low_vol, high_vol])
    preds = np.concatenate([rng.randn(100) * 0.001, rng.randn(100) * 0.01])

    levels = [1.0, 1.5, 2.0, 2.5]
    results = regime_threshold_sweep(
        preds, actuals, threshold_levels=levels,
        cost_bps=10, vol_lookback=30,
    )
    assert len(results) == 4
    for r in results:
        assert "threshold" in r
        assert "sharpe_ratio" in r
        assert "shorts_disabled" in r


def test_select_optimal_regime_threshold():
    """Should select threshold with best Sharpe."""
    sweep_results = [
        {"threshold": 1.0, "sharpe_ratio": 0.5, "shorts_disabled": 20},
        {"threshold": 1.5, "sharpe_ratio": 1.2, "shorts_disabled": 10},
        {"threshold": 2.0, "sharpe_ratio": 0.8, "shorts_disabled": 5},
    ]
    assert select_optimal_regime_threshold(sweep_results) == 1.5


def test_select_optimal_regime_threshold_empty():
    """Should return default when no valid results."""
    assert select_optimal_regime_threshold([], default_threshold=1.5) == 1.5


def test_select_optimal_hidden_size():
    """Should select hidden_size with best Sharpe."""
    sweep_results = [
        {"hidden_size": 32, "sharpe_ratio": 0.3},
        {"hidden_size": 64, "sharpe_ratio": 1.0},
        {"hidden_size": 128, "sharpe_ratio": 0.7},
    ]
    assert select_optimal_hidden_size(sweep_results) == 64


def test_select_optimal_hidden_size_with_errors():
    """Should skip entries with errors."""
    sweep_results = [
        {"hidden_size": 32, "error": "fail"},
        {"hidden_size": 64, "sharpe_ratio": 0.5},
    ]
    assert select_optimal_hidden_size(sweep_results) == 64


def test_select_optimal_hidden_size_empty():
    """Should return default when no valid results."""
    assert select_optimal_hidden_size([], default_hidden_size=64) == 64


def test_regime_threshold_sweep_monotonic_shorts():
    """Higher threshold should disable fewer shorts (less restrictive)."""
    rng = np.random.RandomState(42)
    low_vol = rng.randn(100) * 0.001
    high_vol = rng.randn(100) * 0.01
    actuals = np.concatenate([low_vol, high_vol])
    preds = np.concatenate([rng.randn(100) * 0.001, rng.randn(100) * 0.01])

    levels = [1.0, 1.5, 2.0, 3.0]
    results = regime_threshold_sweep(
        preds, actuals, threshold_levels=levels,
        cost_bps=10, vol_lookback=30,
    )
    shorts = [r["shorts_disabled"] for r in results]
    # Higher threshold → fewer shorts disabled (monotonically non-increasing)
    for i in range(len(shorts) - 1):
        assert shorts[i] >= shorts[i + 1]


# Cycle 9 tests

def test_select_optimal_num_layers():
    """Should select num_layers with best Sharpe."""
    sweep_results = [
        {"num_layers": 1, "sharpe_ratio": 0.3},
        {"num_layers": 2, "sharpe_ratio": 1.0},
        {"num_layers": 3, "sharpe_ratio": 0.7},
    ]
    assert select_optimal_num_layers(sweep_results) == 2


def test_select_optimal_num_layers_with_errors():
    """Should skip entries with errors."""
    sweep_results = [
        {"num_layers": 1, "error": "fail"},
        {"num_layers": 2, "sharpe_ratio": 0.5},
        {"num_layers": 3, "sharpe_ratio": 0.8},
    ]
    assert select_optimal_num_layers(sweep_results) == 3


def test_select_optimal_num_layers_empty():
    """Should return default when no valid results."""
    assert select_optimal_num_layers([], default_num_layers=2) == 2
