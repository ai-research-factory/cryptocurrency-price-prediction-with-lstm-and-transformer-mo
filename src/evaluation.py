"""Walk-forward validation and metrics computation.

Cycle 3: Enhanced with purged walk-forward, interval-aware annualization,
Sortino/Calmar ratios, cost sensitivity analysis, and significance testing.

Cycle 4: Added minimum holding period, feature importance analysis,
and ensemble prediction support.

Cycle 5: Added long/short strategy, classification mode, min holding period
sweep, and inverse-variance ensemble weighting.

Cycle 6: Added sequence length sensitivity sweep, adaptive per-ticker hold period.

Cycle 7: Added per-model-ticker adaptive seq_len selection,
volatility-regime-based short toggling.

Cycle 8: Added regime threshold calibration sweep, hidden size search,
early stopping support.
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
    allow_short: bool = False,
    classification: bool = False,
) -> dict:
    """Compute trading metrics from predicted vs actual returns.

    Strategy: go long when predicted return > 0, else flat (or short if allow_short).
    For classification mode, predictions are probabilities: >0.5 = long, <0.5 = short/flat.
    Cycle 5: Added long/short strategy and classification mode support.
    """
    ann_factor = get_annualization_factor(interval)
    cost = cost_bps / 10_000

    if classification:
        # Predictions are probabilities [0, 1]; threshold at 0.5
        if allow_short:
            position = np.where(predictions > 0.5, 1.0, -1.0)
        else:
            position = (predictions > 0.5).astype(float)
    else:
        # Predictions are return forecasts
        if allow_short:
            position = np.sign(predictions)  # +1, 0, or -1
        else:
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
    allow_short: bool = False,
    classification: bool = False,
) -> list[dict]:
    """Evaluate strategy at multiple transaction cost levels."""
    results = []
    for cost_bps in cost_levels_bps:
        metrics = compute_trading_metrics(
            predictions, actual_returns, cost_bps, interval, min_holding_period,
            allow_short=allow_short, classification=classification,
        )
        results.append({
            "cost_bps": cost_bps,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "total_return": metrics["total_return"],
            "total_cost": metrics["total_cost"],
            "sortino_ratio": metrics["sortino_ratio"],
        })
    return results


def min_holding_period_sweep(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    hold_levels: list[int],
    cost_bps: float = 10.0,
    interval: str = "1d",
    allow_short: bool = False,
    classification: bool = False,
) -> list[dict]:
    """Sweep minimum holding periods to find optimal trade frequency.

    Cycle 5: Evaluates strategy at different min_hold values.
    """
    results = []
    for hold in hold_levels:
        metrics = compute_trading_metrics(
            predictions, actual_returns, cost_bps, interval, hold,
            allow_short=allow_short, classification=classification,
        )
        results.append({
            "min_hold": hold,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "total_return": metrics["total_return"],
            "n_trades": metrics["n_trades"],
            "total_cost": metrics["total_cost"],
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
    allow_short: bool = False,
    classification: bool = False,
    min_hold_sweep_levels: Optional[list[int]] = None,
    warmup_epochs: int = 0,
    early_stopping_patience: int = 0,
) -> dict:
    """Run purged walk-forward validation.

    Slides a window with an optional purge gap between train and test
    to prevent information leakage from overlapping sequences.

    Cycle 5: Added allow_short, classification, and min_hold_sweep_levels.
    Cycle 8: Added early_stopping_patience for training.
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
        window_features = np.vstack([train_features, test_features])
        window_targets = np.concatenate([train_targets, test_targets])

        train_ds, test_ds, _, pred_offset = prepare_data(
            window_features, window_targets, train_size, seq_len,
            classification=classification,
        )

        if len(test_ds) == 0:
            start += step_size
            continue

        model = build_model(model_type, features.shape[1], **model_kwargs)
        train_model(
            model, train_ds, epochs=epochs, batch_size=batch_size, lr=lr,
            device=device, classification=classification,
            warmup_epochs=warmup_epochs,
            early_stopping_patience=early_stopping_patience,
        )

        preds = predict(model, test_ds, device=device)
        actuals = window_targets[train_size + seq_len : train_size + seq_len + len(preds)]

        # Window-level metrics
        w_metrics = compute_trading_metrics(
            preds, actuals, cost_bps, interval, min_holding_period,
            allow_short=allow_short, classification=classification,
        )
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
    agg_metrics = compute_trading_metrics(
        all_preds, all_actuals, cost_bps, interval, min_holding_period,
        allow_short=allow_short, classification=classification,
    )
    baseline_metrics = compute_naive_baseline(all_actuals, cost_bps, interval)

    # Strategy vs baseline significance test
    if classification:
        if allow_short:
            position = np.where(all_preds > 0.5, 1.0, -1.0)
        else:
            position = (all_preds > 0.5).astype(float)
    else:
        if allow_short:
            position = np.sign(all_preds)
        else:
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
            all_preds, all_actuals, cost_sensitivity_levels, interval,
            min_holding_period, allow_short=allow_short, classification=classification,
        )

    # Cycle 5: Min holding period sweep
    hold_sweep = None
    if min_hold_sweep_levels:
        hold_sweep = min_holding_period_sweep(
            all_preds, all_actuals, min_hold_sweep_levels, cost_bps, interval,
            allow_short=allow_short, classification=classification,
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
            "allow_short": allow_short,
            "classification": classification,
        },
    }
    if cost_sensitivity is not None:
        result["cost_sensitivity"] = cost_sensitivity
    if hold_sweep is not None:
        result["min_hold_sweep"] = hold_sweep

    return result


def seq_len_sensitivity_sweep(
    features: np.ndarray,
    targets: np.ndarray,
    model_type: str,
    model_kwargs: dict,
    seq_len_levels: list[int],
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
    min_holding_period: int = 1,
    allow_short: bool = False,
    warmup_epochs: int = 0,
    early_stopping_patience: int = 0,
) -> list[dict]:
    """Sweep sequence lengths to find optimal lookback window.

    Cycle 6: Evaluates model performance at different seq_len values.
    Runs a single walk-forward pass per seq_len for efficiency.
    """
    results = []
    for seq_len in seq_len_levels:
        logger.info(f"  seq_len sweep: testing seq_len={seq_len}")
        wf_result = walk_forward_validation(
            features=features,
            targets=targets,
            model_type=model_type,
            model_kwargs=model_kwargs,
            seq_len=seq_len,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            cost_bps=cost_bps,
            device=device,
            purge_gap=purge_gap,
            interval=interval,
            adaptive_window=adaptive_window,
            min_holding_period=min_holding_period,
            allow_short=allow_short,
            warmup_epochs=warmup_epochs,
            early_stopping_patience=early_stopping_patience,
        )
        if "error" in wf_result:
            results.append({
                "seq_len": seq_len,
                "error": wf_result["error"],
            })
        else:
            agg = wf_result["aggregate"]
            results.append({
                "seq_len": seq_len,
                "sharpe_ratio": agg["sharpe_ratio"],
                "sortino_ratio": agg["sortino_ratio"],
                "total_return": agg["total_return"],
                "n_trades": agg["n_trades"],
                "n_windows": wf_result["n_windows"],
            })
    return results


def select_optimal_hold_period(
    hold_sweep_results: list[dict],
) -> int:
    """Select the optimal minimum holding period from sweep results.

    Cycle 6: Picks the hold period with the best Sharpe ratio.
    Returns the optimal min_hold value.
    """
    valid = [r for r in hold_sweep_results if "sharpe_ratio" in r]
    if not valid:
        return 1
    best = max(valid, key=lambda r: r["sharpe_ratio"])
    return best["min_hold"]


def select_optimal_seq_len(
    seq_len_sweep_results: list[dict],
    default_seq_len: int = 30,
) -> int:
    """Select the optimal sequence length from sweep results.

    Cycle 7: Picks the seq_len with the best Sharpe ratio from sweep.
    Falls back to default if no valid results.
    """
    valid = [r for r in seq_len_sweep_results if "sharpe_ratio" in r and "error" not in r]
    if not valid:
        return default_seq_len
    best = max(valid, key=lambda r: r["sharpe_ratio"])
    return best["seq_len"]


def compute_volatility_regime(
    actual_returns: np.ndarray,
    lookback: int = 60,
    high_vol_threshold: float = 1.5,
) -> np.ndarray:
    """Classify each period as low-vol (0) or high-vol (1) regime.

    Cycle 7: Uses rolling volatility relative to expanding historical mean.
    High-vol regime = rolling vol > high_vol_threshold * expanding mean vol.
    Returns array of 0/1 regime labels (same length as input).
    """
    n = len(actual_returns)
    regime = np.zeros(n, dtype=float)
    if n < lookback + 1:
        return regime

    rolling_vol = pd.Series(actual_returns).rolling(lookback).std().values

    # Expanding mean of rolling vol (use only past data, no lookahead)
    expanding_mean_vol = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        if not np.isnan(rolling_vol[i]):
            vol_sum += rolling_vol[i]
            vol_count += 1
        expanding_mean_vol[i] = vol_sum / max(vol_count, 1)

    for i in range(lookback, n):
        if expanding_mean_vol[i] > 1e-10 and rolling_vol[i] > high_vol_threshold * expanding_mean_vol[i]:
            regime[i] = 1.0

    return regime


def compute_trading_metrics_regime_short(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    cost_bps: float = 10.0,
    interval: str = "1d",
    min_holding_period: int = 1,
    classification: bool = False,
    vol_lookback: int = 60,
    high_vol_threshold: float = 1.5,
) -> dict:
    """Compute trading metrics with regime-based short toggling.

    Cycle 7: Only allows short positions in low-volatility regimes.
    In high-volatility regimes, shorts are disabled (position = max(position, 0)).
    """
    ann_factor = get_annualization_factor(interval)
    cost = cost_bps / 10_000

    # Generate base position (long/short)
    if classification:
        position = np.where(predictions > 0.5, 1.0, -1.0)
    else:
        position = np.sign(predictions)

    # Compute volatility regime
    regime = compute_volatility_regime(actual_returns, vol_lookback, high_vol_threshold)

    # In high-vol regime, disable shorts (clamp to 0)
    position = np.where((regime == 1.0) & (position < 0), 0.0, position)

    position = _apply_min_holding_period(position, min_holding_period)
    trades = np.abs(np.diff(position, prepend=0))

    gross_returns = position * actual_returns
    net_returns = gross_returns - trades * cost

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

    winning_periods = np.sum(net_returns > 0)
    active_periods = np.sum(np.abs(position) > 0)
    win_rate = winning_periods / max(active_periods, 1)

    n_trades = int(np.sum(trades > 0))
    total_cost = float(np.sum(trades * cost))

    high_vol_periods = int(np.sum(regime == 1.0))
    shorts_disabled = int(np.sum((regime == 1.0) & (np.sign(predictions) < 0)))

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
        "high_vol_periods": high_vol_periods,
        "shorts_disabled": shorts_disabled,
    }


def regime_threshold_sweep(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    threshold_levels: list[float],
    cost_bps: float = 10.0,
    interval: str = "1d",
    min_holding_period: int = 1,
    classification: bool = False,
    vol_lookback: int = 60,
) -> list[dict]:
    """Sweep volatility regime thresholds to find optimal per-ticker calibration.

    Cycle 8: Evaluates regime-short strategy at different high_vol_threshold values.
    """
    results = []
    for threshold in threshold_levels:
        metrics = compute_trading_metrics_regime_short(
            predictions, actual_returns, cost_bps, interval,
            min_holding_period, classification=classification,
            vol_lookback=vol_lookback, high_vol_threshold=threshold,
        )
        results.append({
            "threshold": threshold,
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "total_return": metrics["total_return"],
            "n_trades": metrics["n_trades"],
            "high_vol_periods": metrics["high_vol_periods"],
            "shorts_disabled": metrics["shorts_disabled"],
        })
    return results


def select_optimal_regime_threshold(
    sweep_results: list[dict],
    default_threshold: float = 1.5,
) -> float:
    """Select the regime threshold that maximizes Sharpe ratio.

    Cycle 8: If no improvement over no-regime (threshold=inf), returns default.
    """
    valid = [r for r in sweep_results if "sharpe_ratio" in r]
    if not valid:
        return default_threshold
    best = max(valid, key=lambda r: r["sharpe_ratio"])
    return best["threshold"]


def hidden_size_sweep(
    features: np.ndarray,
    targets: np.ndarray,
    model_type: str,
    model_kwargs: dict,
    hidden_size_levels: list[int],
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
    min_holding_period: int = 1,
    allow_short: bool = False,
    warmup_epochs: int = 0,
    early_stopping_patience: int = 0,
) -> list[dict]:
    """Sweep hidden sizes to find optimal architecture per model type.

    Cycle 8: Evaluates model performance at different hidden_size values.
    """
    results = []
    for hs in hidden_size_levels:
        logger.info(f"  hidden_size sweep: testing hidden_size={hs} for {model_type}")
        mkwargs = dict(model_kwargs)
        # Set the appropriate size parameter
        if model_type in ("lstm", "gru"):
            mkwargs["hidden_size"] = hs
        elif model_type == "transformer":
            # For transformer, d_model must be divisible by nhead
            nhead = mkwargs.get("nhead", 4)
            d_model = max(hs, nhead)
            d_model = d_model - (d_model % nhead)  # ensure divisible
            mkwargs["d_model"] = d_model
            mkwargs["dim_feedforward"] = d_model * 2

        wf_result = walk_forward_validation(
            features=features,
            targets=targets,
            model_type=model_type,
            model_kwargs=mkwargs,
            seq_len=seq_len,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            cost_bps=cost_bps,
            device=device,
            purge_gap=purge_gap,
            interval=interval,
            adaptive_window=adaptive_window,
            min_holding_period=min_holding_period,
            allow_short=allow_short,
            warmup_epochs=warmup_epochs,
            early_stopping_patience=early_stopping_patience,
        )
        if "error" in wf_result:
            results.append({
                "hidden_size": hs,
                "error": wf_result["error"],
            })
        else:
            agg = wf_result["aggregate"]
            results.append({
                "hidden_size": hs,
                "sharpe_ratio": agg["sharpe_ratio"],
                "sortino_ratio": agg["sortino_ratio"],
                "total_return": agg["total_return"],
                "n_trades": agg["n_trades"],
                "n_windows": wf_result["n_windows"],
            })
    return results


def select_optimal_hidden_size(
    sweep_results: list[dict],
    default_hidden_size: int = 64,
) -> int:
    """Select the hidden size with best Sharpe ratio."""
    valid = [r for r in sweep_results if "sharpe_ratio" in r and "error" not in r]
    if not valid:
        return default_hidden_size
    best = max(valid, key=lambda r: r["sharpe_ratio"])
    return best["hidden_size"]
