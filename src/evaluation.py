"""Walk-forward validation and metrics computation.

Cycle 3: Enhanced with purged walk-forward, interval-aware annualization,
Sortino/Calmar ratios, cost sensitivity analysis, and significance testing.

Cycle 4: Added minimum holding period, feature importance analysis,
and ensemble prediction support.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .models import build_model
from .training import prepare_data, train_model, predict

logger = logging.getLogger(__name__)

# Annualization factors: number of trading periods per year
ANNUALIZATION_FACTORS = {
    "1d": 252,
    "1h": 252 * 24,     # 6048
    "4h": 252 * 6,      # 1512
    "15m": 252 * 24 * 4, # 24192
    "1m": 252 * 24 * 60, # 362880
}


def get_annualization_factor(interval: str = "1d") -> int:
    """Return annualization factor for a given data interval."""
    return ANNUALIZATION_FACTORS.get(interval, 252)


def _apply_min_holding_period(position: np.ndarray, min_hold: int) -> np.ndarray:
    """Enforce minimum holding period on position changes.

    Once a position change occurs, the new position is held for at least
    min_hold periods before another change is allowed. This reduces
    unnecessary trade churn from noisy predictions.
    """
    if min_hold <= 1:
        return position
    result = position.copy()
    hold_counter = 0
    for i in range(1, len(result)):
        if hold_counter > 0:
            result[i] = result[i - 1]
            hold_counter -= 1
        elif result[i] != result[i - 1]:
            hold_counter = min_hold - 1
    return result


def compute_trading_metrics(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    cost_bps: float = 10.0,
    interval: str = "1d",
    min_holding_period: int = 1,
) -> dict:
    """Compute trading metrics from predicted vs actual returns.

    Strategy: go long when predicted return > 0, else flat.
    Cycle 4: Added min_holding_period to reduce trade frequency.
    """
    ann_factor = get_annualization_factor(interval)
    cost = cost_bps / 10_000

    # Position: 1 if predicted return > 0, else 0
    position = (predictions > 0).astype(float)
    position = _apply_min_holding_period(position, min_holding_period)
    trades = np.abs(np.diff(position, prepend=0))

    # Strategy returns
    gross_returns = position * actual_returns
    net_returns = gross_returns - trades * cost

    # Core metrics
    total_return = np.expm1(np.sum(net_returns))
    n_periods = len(net_returns)
    annual_factor = ann_factor / max(n_periods, 1)

    mean_ret = np.mean(net_returns)
    std_ret = np.std(net_returns)
    sharpe = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 1e-10 else 0.0

    # Sortino ratio (downside deviation only)
    downside = net_returns[net_returns < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-10
    sortino = (mean_ret / downside_std * np.sqrt(ann_factor)) if downside_std > 1e-10 else 0.0

    # Max drawdown
    cum_returns = np.cumsum(net_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = running_max - cum_returns
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Calmar ratio (annualized return / max drawdown)
    ann_return = float(total_return * annual_factor)
    calmar = ann_return / max_dd if max_dd > 1e-10 else 0.0

    # Win rate
    winning_periods = np.sum(net_returns > 0)
    active_periods = np.sum(position > 0)
    win_rate = winning_periods / max(active_periods, 1)

    n_trades = int(np.sum(trades > 0))
    total_cost = float(np.sum(trades * cost))

    return {
        "total_return": float(total_return),
        "annualized_return": ann_return,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_dd,
        "calmar_ratio": float(calmar),
        "win_rate": float(win_rate),
        "n_trades": n_trades,
        "total_cost": total_cost,
        "n_periods": n_periods,
        "mean_period_return": float(mean_ret),
        "std_period_return": float(std_ret),
    }


def compute_naive_baseline(
    actual_returns: np.ndarray,
    cost_bps: float = 10.0,
    interval: str = "1d",
) -> dict:
    """Buy-and-hold baseline metrics."""
    ann_factor = get_annualization_factor(interval)
    cost = cost_bps / 10_000
    net_returns = actual_returns.copy()
    net_returns[0] -= cost  # entry cost

    total_return = np.expm1(np.sum(net_returns))
    n_periods = len(net_returns)
    annual_factor = ann_factor / max(n_periods, 1)
    mean_ret = np.mean(net_returns)
    std_ret = np.std(net_returns)
    sharpe = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 1e-10 else 0.0

    downside = net_returns[net_returns < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-10
    sortino = (mean_ret / downside_std * np.sqrt(ann_factor)) if downside_std > 1e-10 else 0.0

    cum_returns = np.cumsum(net_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = running_max - cum_returns
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    ann_return = float(total_return * annual_factor)
    calmar = ann_return / max_dd if max_dd > 1e-10 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": ann_return,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_dd,
        "calmar_ratio": float(calmar),
        "n_periods": n_periods,
    }


def compute_significance_vs_baseline(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict:
    """Test whether strategy excess returns are significantly different from zero.

    Uses a paired t-test on per-period returns.
    """
    excess = strategy_returns - baseline_returns
    if len(excess) < 3:
        return {"t_stat": 0.0, "p_value": 1.0, "significant_5pct": False}

    t_stat, p_value = scipy_stats.ttest_1samp(excess, 0.0)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_5pct": bool(p_value < 0.05),
    }


def cost_sensitivity_analysis(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    cost_levels_bps: list[float],
    interval: str = "1d",
    min_holding_period: int = 1,
) -> list[dict]:
    """Evaluate strategy at multiple transaction cost levels."""
    results = []
    for cost_bps in cost_levels_bps:
        metrics = compute_trading_metrics(
            predictions, actual_returns, cost_bps, interval, min_holding_period
        )
        results.append({
            "cost_bps": cost_bps,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "total_return": metrics["total_return"],
            "total_cost": metrics["total_cost"],
            "sortino_ratio": metrics["sortino_ratio"],
        })
    return results


def compute_feature_importance(
    features: np.ndarray,
    targets: np.ndarray,
    model_type: str,
    model_kwargs: dict,
    feature_names: list[str],
    seq_len: int = 30,
    train_size: int = 500,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    cost_bps: float = 10.0,
    interval: str = "1d",
    device: str = "cpu",
    n_repeats: int = 3,
) -> dict:
    """Compute permutation feature importance.

    Trains a model once, then measures Sharpe degradation when each feature
    is shuffled. Higher degradation = more important feature.
    """
    n = len(features)
    if n < train_size + seq_len + 30:
        return {"error": "Not enough data for feature importance"}

    # Use last portion as test
    test_size = min(n - train_size - seq_len, 200)

    train_features = features[:train_size]
    train_targets = targets[:train_size]
    test_features = features[train_size:train_size + test_size]
    test_targets = targets[train_size:train_size + test_size]

    window_features = np.vstack([train_features, test_features])
    window_targets = np.concatenate([train_targets, test_targets])

    train_ds, test_ds, scaler_stats, _ = prepare_data(
        window_features, window_targets, train_size, seq_len
    )

    if len(test_ds) == 0:
        return {"error": "Empty test set"}

    model = build_model(model_type, features.shape[1], **model_kwargs)
    train_model(model, train_ds, epochs=epochs, batch_size=batch_size, lr=lr, device=device)

    # Baseline predictions
    base_preds = predict(model, test_ds, device=device)
    actuals = window_targets[train_size + seq_len: train_size + seq_len + len(base_preds)]
    base_metrics = compute_trading_metrics(base_preds, actuals, cost_bps, interval)
    base_sharpe = base_metrics["sharpe_ratio"]

    # Permute each feature and measure degradation
    importances = {}
    for feat_idx, feat_name in enumerate(feature_names):
        sharpe_drops = []
        for _ in range(n_repeats):
            permuted_features = window_features.copy()
            # Shuffle only the test portion of this feature
            rng = np.random.RandomState()
            perm_idx = rng.permutation(test_size)
            permuted_features[train_size:train_size + test_size, feat_idx] = \
                permuted_features[train_size + perm_idx, feat_idx]

            # Re-scale with same stats
            test_scaled = (permuted_features[train_size:] - scaler_stats["mean"]) / scaler_stats["std"]
            from .training import TimeSeriesDataset
            perm_ds = TimeSeriesDataset(
                test_scaled, window_targets[train_size:], seq_len
            )
            perm_preds = predict(model, perm_ds, device=device)
            perm_actuals = window_targets[train_size + seq_len: train_size + seq_len + len(perm_preds)]
            perm_metrics = compute_trading_metrics(perm_preds, perm_actuals, cost_bps, interval)
            sharpe_drops.append(base_sharpe - perm_metrics["sharpe_ratio"])

        importances[feat_name] = {
            "mean_sharpe_drop": float(np.mean(sharpe_drops)),
            "std_sharpe_drop": float(np.std(sharpe_drops)),
        }

    # Sort by importance (highest drop = most important)
    sorted_features = sorted(importances.keys(), key=lambda f: importances[f]["mean_sharpe_drop"], reverse=True)

    return {
        "base_sharpe": float(base_sharpe),
        "feature_importances": importances,
        "ranking": sorted_features,
    }


def _compute_adaptive_window_sizes(
    n_samples: int,
    seq_len: int,
    min_train_size: int = 200,
    min_test_size: int = 30,
    min_windows: int = 3,
    train_ratio: float = 0.7,
) -> tuple[int, int, int]:
    """Compute window sizes that guarantee at least min_windows walk-forward windows.

    Returns (train_size, test_size, step_size).
    """
    usable = n_samples - seq_len
    if usable < min_train_size + min_test_size:
        return min_train_size, min_test_size, min_test_size

    # Try to fit min_windows with non-overlapping test blocks
    # n >= train_size + test_size * min_windows + seq_len
    max_test_budget = usable - min_train_size
    test_size = max(min_test_size, max_test_budget // (min_windows + 2))
    train_size = max(min_train_size, int(test_size * train_ratio / (1 - train_ratio)))

    # Verify we get enough windows; shrink train if needed
    while train_size + test_size * min_windows + seq_len > n_samples and train_size > min_train_size:
        train_size = max(min_train_size, train_size - 50)

    step_size = test_size
    return train_size, test_size, step_size


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
    purge_gap: int = 0,
    interval: str = "1d",
    adaptive_window: bool = False,
    cost_sensitivity_levels: Optional[list[float]] = None,
    min_holding_period: int = 1,
) -> dict:
    """Run purged walk-forward validation.

    Slides a window with an optional purge gap between train and test
    to prevent information leakage from overlapping sequences.
    """
    n = len(features)

    # Adaptive window sizing to ensure sufficient windows
    if adaptive_window:
        train_size, test_size, step_size = _compute_adaptive_window_sizes(
            n, seq_len, min_train_size=200, min_test_size=30, min_windows=3,
        )
        logger.info(
            f"Adaptive window: train={train_size}, test={test_size}, step={step_size}"
        )

    all_preds = []
    all_actuals = []
    window_metrics = []
    window_idx = 0

    start = 0
    while start + train_size + purge_gap + test_size + seq_len <= n:
        train_end = start + train_size
        test_start = train_end + purge_gap
        test_end = test_start + test_size

        # Extract train window (without purged region)
        train_features = features[start:train_end]
        train_targets = targets[start:train_end]

        # Extract test window (after purge gap)
        test_features = features[test_start:test_end]
        test_targets = targets[test_start:test_end]

        # Combine for prepare_data, which expects contiguous array
        # We pass train data and test data separately via the train_end split
        window_features = np.vstack([train_features, test_features])
        window_targets = np.concatenate([train_targets, test_targets])

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
        w_metrics = compute_trading_metrics(preds, actuals, cost_bps, interval, min_holding_period)
        w_metrics["window"] = window_idx
        window_metrics.append(w_metrics)

        all_preds.extend(preds.tolist())
        all_actuals.extend(actuals.tolist())

        logger.info(
            f"Window {window_idx}: Sharpe={w_metrics['sharpe_ratio']:.3f}, "
            f"Sortino={w_metrics['sortino_ratio']:.3f}, "
            f"Return={w_metrics['total_return']:.4f}"
        )

        start += step_size
        window_idx += 1

    if not all_preds:
        return {"error": "No valid windows"}

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Aggregate metrics
    agg_metrics = compute_trading_metrics(all_preds, all_actuals, cost_bps, interval, min_holding_period)
    baseline_metrics = compute_naive_baseline(all_actuals, cost_bps, interval)

    # Strategy vs baseline significance test
    position = (all_preds > 0).astype(float)
    position = _apply_min_holding_period(position, min_holding_period)
    cost = cost_bps / 10_000
    trades = np.abs(np.diff(position, prepend=0))
    strategy_returns = position * all_actuals - trades * cost
    significance = compute_significance_vs_baseline(strategy_returns, all_actuals)

    # Stability: fraction of windows with positive Sharpe
    positive_windows = sum(1 for w in window_metrics if w["sharpe_ratio"] > 0)
    stability = positive_windows / max(len(window_metrics), 1)

    # Per-window Sharpe statistics
    window_sharpes = [w["sharpe_ratio"] for w in window_metrics]
    sharpe_stats = {
        "mean": float(np.mean(window_sharpes)),
        "std": float(np.std(window_sharpes)),
        "min": float(np.min(window_sharpes)),
        "max": float(np.max(window_sharpes)),
    }

    # Cost sensitivity analysis
    cost_sensitivity = None
    if cost_sensitivity_levels:
        cost_sensitivity = cost_sensitivity_analysis(
            all_preds, all_actuals, cost_sensitivity_levels, interval, min_holding_period
        )

    result = {
        "model_type": model_type,
        "aggregate": agg_metrics,
        "baseline_buy_hold": baseline_metrics,
        "significance_vs_baseline": significance,
        "stability_ratio": float(stability),
        "n_windows": len(window_metrics),
        "positive_windows": positive_windows,
        "window_sharpe_stats": sharpe_stats,
        "window_details": window_metrics,
        "all_predictions": all_preds.tolist(),
        "all_actuals": all_actuals.tolist(),
        "walk_forward_config": {
            "train_size": train_size,
            "test_size": test_size,
            "step_size": step_size,
            "purge_gap": purge_gap,
            "adaptive_window": adaptive_window,
            "interval": interval,
            "annualization_factor": get_annualization_factor(interval),
            "min_holding_period": min_holding_period,
        },
    }
    if cost_sensitivity is not None:
        result["cost_sensitivity"] = cost_sensitivity

    return result
