"""Walk-forward validation and metrics computation."""

import logging

import numpy as np
import pandas as pd

from .models import build_model
from .training import prepare_data, train_model, predict

logger = logging.getLogger(__name__)


def compute_trading_metrics(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    cost_bps: float = 10.0,
) -> dict:
    """Compute trading metrics from predicted vs actual returns.

    Strategy: go long when predicted return > 0, else flat.
    """
    cost = cost_bps / 10_000  # convert basis points to decimal

    # Position: 1 if predicted return > 0, else 0
    position = (predictions > 0).astype(float)
    # Detect trades (position changes)
    trades = np.abs(np.diff(position, prepend=0))

    # Strategy returns = position * actual_return - trade_cost
    gross_returns = position * actual_returns
    net_returns = gross_returns - trades * cost

    # Metrics
    total_return = np.expm1(np.sum(net_returns))
    n_days = len(net_returns)
    annual_factor = 252 / max(n_days, 1)

    mean_daily = np.mean(net_returns)
    std_daily = np.std(net_returns)
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 1e-10 else 0.0

    # Max drawdown
    cum_returns = np.cumsum(net_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = running_max - cum_returns
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0

    # Win rate
    winning_days = np.sum(net_returns > 0)
    active_days = np.sum(position > 0)
    win_rate = winning_days / max(active_days, 1)

    # Trade count
    n_trades = int(np.sum(trades > 0))

    return {
        "total_return": float(total_return),
        "annualized_return": float(total_return * annual_factor),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "n_trades": n_trades,
        "n_days": n_days,
        "mean_daily_return": float(mean_daily),
        "std_daily_return": float(std_daily),
    }


def compute_naive_baseline(actual_returns: np.ndarray, cost_bps: float = 10.0) -> dict:
    """Buy-and-hold baseline metrics."""
    cost = cost_bps / 10_000
    # Buy and hold: always long, only 1 trade at start
    net_returns = actual_returns.copy()
    net_returns[0] -= cost  # entry cost

    total_return = np.expm1(np.sum(net_returns))
    n_days = len(net_returns)
    annual_factor = 252 / max(n_days, 1)
    mean_daily = np.mean(net_returns)
    std_daily = np.std(net_returns)
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 1e-10 else 0.0

    cum_returns = np.cumsum(net_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = running_max - cum_returns
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(total_return * annual_factor),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_days": n_days,
    }


def walk_forward_validation(
    features: np.ndarray,
    targets: np.ndarray,
    model_type: str,
    model_kwargs: dict,
    seq_len: int = 30,
    train_size: int = 500,
    test_size: int = 60,
    step_size: int = 60,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    cost_bps: float = 10.0,
    device: str = "cpu",
) -> dict:
    """Run walk-forward validation.

    Slides a window: train on [start..start+train_size],
    test on [start+train_size..start+train_size+test_size],
    then step forward.
    """
    n = len(features)
    all_preds = []
    all_actuals = []
    window_metrics = []
    window_idx = 0

    start = 0
    while start + train_size + test_size + seq_len <= n:
        window_end = start + train_size + test_size
        window_features = features[start:window_end]
        window_targets = targets[start:window_end]

        train_ds, test_ds, _, pred_offset = prepare_data(
            window_features, window_targets, train_size, seq_len
        )

        if len(test_ds) == 0:
            start += step_size
            continue

        model = build_model(model_type, features.shape[1], **model_kwargs)
        train_model(model, train_ds, epochs=epochs, batch_size=batch_size, lr=lr, device=device)

        preds = predict(model, test_ds, device=device)
        actuals = window_targets[train_size + seq_len : train_size + seq_len + len(preds)]

        # Window-level metrics
        w_metrics = compute_trading_metrics(preds, actuals, cost_bps)
        w_metrics["window"] = window_idx
        window_metrics.append(w_metrics)

        all_preds.extend(preds.tolist())
        all_actuals.extend(actuals.tolist())

        logger.info(
            f"Window {window_idx}: Sharpe={w_metrics['sharpe_ratio']:.3f}, "
            f"Return={w_metrics['total_return']:.4f}"
        )

        start += step_size
        window_idx += 1

    if not all_preds:
        return {"error": "No valid windows"}

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Aggregate metrics
    agg_metrics = compute_trading_metrics(all_preds, all_actuals, cost_bps)
    baseline_metrics = compute_naive_baseline(all_actuals, cost_bps)

    # Stability: fraction of windows with positive Sharpe
    positive_windows = sum(1 for w in window_metrics if w["sharpe_ratio"] > 0)
    stability = positive_windows / max(len(window_metrics), 1)

    return {
        "model_type": model_type,
        "aggregate": agg_metrics,
        "baseline_buy_hold": baseline_metrics,
        "stability_ratio": float(stability),
        "n_windows": len(window_metrics),
        "positive_windows": positive_windows,
        "window_details": window_metrics,
    }
