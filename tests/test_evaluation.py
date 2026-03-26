"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation import compute_trading_metrics, compute_naive_baseline


def test_trading_metrics_structure():
    preds = np.array([0.01, -0.01, 0.02, 0.01, -0.005])
    actuals = np.array([0.005, -0.003, 0.01, -0.002, 0.001])
    metrics = compute_trading_metrics(preds, actuals)

    required_keys = [
        "total_return", "sharpe_ratio", "max_drawdown",
        "win_rate", "n_trades", "n_days",
    ]
    for key in required_keys:
        assert key in metrics


def test_all_positive_predictions():
    """If always long and market goes up, should have positive return."""
    preds = np.ones(100) * 0.01  # always predict positive
    actuals = np.ones(100) * 0.001  # market goes up slightly
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
