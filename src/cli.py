"""CLI entry point for running experiments.

Cycle 4: Added ensemble model, feature importance, min holding period, ETH/USDT.
Cycle 5: Added classification mode, long/short strategy, min hold sweep,
inverse-variance ensemble weighting.
Cycle 6: Added GRU model, sequence length sweep, adaptive per-ticker hold period,
frequency-adaptive indicator periods, 3-model ensemble.
Cycle 7: Added per-model-ticker adaptive seq_len, Transformer warm-up scheduling,
additional tickers (SPY, MSFT), volatility-regime short toggling.
Cycle 8: Added early stopping, ensemble fix, per-ticker regime threshold calibration,
per-model hidden size search.
Cycle 9: Added selective ensemble (drop underperforming models), per-model
num_layers search, fixed early stopping + warmup interaction.
Cycle 10: Added multi-seed averaging, joint hyperparameter search,
adaptive mode selection (classification vs regression per model/ticker).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import fetch_ohlcv, compute_returns, validate_ohlcv, clean_ohlcv, compute_data_summary
from .indicators import compute_all_indicators
from .preprocessing import preprocess_features, compute_feature_stats
from .evaluation import (
    walk_forward_validation, compute_feature_importance,
    compute_trading_metrics, compute_naive_baseline,
    compute_significance_vs_baseline, cost_sensitivity_analysis,
    _apply_min_holding_period, min_holding_period_sweep,
    seq_len_sensitivity_sweep, select_optimal_hold_period,
    select_optimal_seq_len, compute_trading_metrics_regime_short,
    regime_threshold_sweep, select_optimal_regime_threshold,
    hidden_size_sweep, select_optimal_hidden_size,
    num_layers_sweep, select_optimal_num_layers,
    walk_forward_validation_multiseed,
    joint_hyperparam_search, select_optimal_joint_params,
    mode_selection_sweep, select_optimal_mode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_interval(ticker: str, data_cfg: dict) -> str:
    """Get the interval for a ticker, considering overrides."""
    overrides = data_cfg.get("ticker_overrides", {}).get(ticker, {})
    return overrides.get("interval", data_cfg.get("interval", "1d"))


def prepare_ticker_data(ticker: str, data_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fetch, validate, clean, and prepare features for a single ticker.

    Cycle 6: Passes interval to compute_all_indicators for frequency-adaptive periods.
    Returns (features_df, ohlcv_df, data_report).
    """
    overrides = data_cfg.get("ticker_overrides", {}).get(ticker, {})
    interval = overrides.get("interval", data_cfg.get("interval", "1d"))
    period = overrides.get("period", data_cfg.get("period", "5y"))

    df = fetch_ohlcv(ticker=ticker, interval=interval, period=period)

    validation = validate_ohlcv(df)
    logger.info(f"Validation for {ticker}: {len(df)} rows, "
                f"issues: {validation['issues'] or 'none'}")

    df = clean_ohlcv(df)
    summary = compute_data_summary(df, ticker)
    features_df = compute_all_indicators(df, interval=interval)
    features_df = preprocess_features(features_df, df)

    data_report = {
        "validation": validation,
        "summary": summary,
        "n_rows_after_clean": len(df),
        "indicator_interval": interval,
    }

    return features_df, df, data_report


def _build_ensemble_result(
    model_results: dict,
    features: np.ndarray,
    targets: np.ndarray,
    eval_cfg: dict,
    interval: str,
    cost_sensitivity_levels: list[float] | None,
    min_holding_period: int,
    allow_short: bool = False,
    classification: bool = False,
    ensemble_method: str = "equal",
) -> dict | None:
    """Build ensemble result by combining predictions from all models.

    Cycle 5: Added inverse-variance weighting and long/short support.
    Cycle 8: Improved alignment for adaptive seq_len by matching tail ends.
    ensemble_method: "equal" for simple average, "inverse_variance" for
    weighting by inverse of per-model return variance.
    """
    # Collect all_preds and all_actuals from each model
    model_all_preds = {}
    model_all_actuals = {}
    for mtype, result in model_results.items():
        if "error" in result:
            continue
        if "all_predictions" not in result or "all_actuals" not in result:
            continue
        preds = np.array(result["all_predictions"])
        actuals = np.array(result["all_actuals"])
        if len(preds) == 0 or len(actuals) == 0:
            continue
        model_all_preds[mtype] = preds
        model_all_actuals[mtype] = actuals

    if len(model_all_preds) < 2:
        return None

    # Align predictions to common length (may differ with adaptive seq_len)
    # Use tail alignment: all models' last predictions correspond to the same time period
    pred_lengths = [len(p) for p in model_all_preds.values()]
    min_len = min(pred_lengths)
    if min_len == 0:
        return None
    if len(set(pred_lengths)) != 1:
        logger.info(f"Prediction lengths differ ({dict(zip(model_all_preds.keys(), pred_lengths))}), "
                     f"tail-aligning to common length {min_len}")
    model_all_preds = {m: p[-min_len:] for m, p in model_all_preds.items()}

    # Use actuals from the model with the shortest predictions (already tail-aligned)
    shortest_model = min(model_all_actuals.keys(), key=lambda m: len(model_all_actuals[m]))
    all_actuals = np.array(model_all_actuals[shortest_model])[-min_len:]
    cost_bps = eval_cfg.get("cost_bps", 10.0)

    # Compute weights
    if ensemble_method == "inverse_variance":
        # Weight each model by inverse of its return variance (lower variance = higher weight)
        variances = {}
        for mtype, preds in model_all_preds.items():
            metrics = compute_trading_metrics(
                preds, all_actuals, cost_bps, interval, min_holding_period,
                allow_short=allow_short, classification=classification,
            )
            var = metrics["std_period_return"] ** 2
            variances[mtype] = max(var, 1e-10)

        total_inv_var = sum(1.0 / v for v in variances.values())
        weights = {m: (1.0 / v) / total_inv_var for m, v in variances.items()}
        logger.info(f"Inverse-variance weights: {weights}")

        ensemble_preds = sum(
            weights[m] * model_all_preds[m] for m in model_all_preds
        )
    else:
        # Equal weighting
        pred_arrays = list(model_all_preds.values())
        ensemble_preds = np.mean(pred_arrays, axis=0)
        weights = {m: 1.0 / len(model_all_preds) for m in model_all_preds}

    agg_metrics = compute_trading_metrics(
        ensemble_preds, all_actuals, cost_bps, interval, min_holding_period,
        allow_short=allow_short, classification=classification,
    )
    baseline_metrics = compute_naive_baseline(all_actuals, cost_bps, interval)

    # Significance test
    if classification:
        if allow_short:
            position = np.where(ensemble_preds > 0.5, 1.0, -1.0)
        else:
            position = (ensemble_preds > 0.5).astype(float)
    else:
        if allow_short:
            position = np.sign(ensemble_preds)
        else:
            position = (ensemble_preds > 0).astype(float)
    position = _apply_min_holding_period(position, min_holding_period)
    cost = cost_bps / 10_000
    trades = np.abs(np.diff(position, prepend=0))
    strategy_returns = position * all_actuals - trades * cost
    significance = compute_significance_vs_baseline(strategy_returns, all_actuals)

    result = {
        "model_type": f"ensemble_{ensemble_method}",
        "ensemble_method": ensemble_method,
        "ensemble_weights": {m: float(w) for m, w in weights.items()},
        "aggregate": agg_metrics,
        "baseline_buy_hold": baseline_metrics,
        "significance_vs_baseline": significance,
        "component_models": list(model_all_preds.keys()),
    }

    if cost_sensitivity_levels:
        result["cost_sensitivity"] = cost_sensitivity_analysis(
            ensemble_preds, all_actuals, cost_sensitivity_levels, interval,
            min_holding_period, allow_short=allow_short, classification=classification,
        )

    return result


def run_experiment(config: dict) -> dict:
    """Run full experiment pipeline with Cycle 10 enhancements."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    tickers = data_cfg.get("tickers", [data_cfg.get("ticker", "AAPL")])
    if isinstance(tickers, str):
        tickers = [tickers]

    # Cycle 3: cost sensitivity levels
    cost_sensitivity_levels = eval_cfg.get("cost_sensitivity_bps", None)
    purge_gap = eval_cfg.get("purge_gap", 0)
    adaptive_window = eval_cfg.get("adaptive_window", False)
    # Cycle 4: minimum holding period
    min_holding_period = eval_cfg.get("min_holding_period", 1)
    run_feature_importance = eval_cfg.get("feature_importance", False)
    run_ensemble = eval_cfg.get("ensemble", True)
    # Cycle 5: new options
    allow_short = eval_cfg.get("allow_short", False)
    classification = eval_cfg.get("classification", False)
    ensemble_method = eval_cfg.get("ensemble_method", "equal")
    min_hold_sweep_levels = eval_cfg.get("min_hold_sweep", None)
    # Cycle 6: new options
    seq_len_sweep_levels = eval_cfg.get("seq_len_sweep", None)
    adaptive_hold = eval_cfg.get("adaptive_hold", False)
    # Cycle 7: new options
    adaptive_seq_len = eval_cfg.get("adaptive_seq_len", False)
    warmup_epochs = eval_cfg.get("warmup_epochs", 0)
    regime_short = eval_cfg.get("regime_short", False)
    vol_lookback = eval_cfg.get("vol_lookback", 60)
    high_vol_threshold = eval_cfg.get("high_vol_threshold", 1.5)
    # Cycle 8: new options
    early_stopping_patience = eval_cfg.get("early_stopping_patience", 0)
    regime_threshold_levels = eval_cfg.get("regime_threshold_sweep", None)
    hidden_size_levels = eval_cfg.get("hidden_size_sweep", None)
    adaptive_hidden_size = eval_cfg.get("adaptive_hidden_size", False)
    # Cycle 9: new options
    num_layers_levels = eval_cfg.get("num_layers_sweep", None)
    adaptive_num_layers = eval_cfg.get("adaptive_num_layers", False)
    selective_ensemble = eval_cfg.get("selective_ensemble", False)
    selective_ensemble_threshold = eval_cfg.get("selective_ensemble_threshold", 0.0)
    # Cycle 10: new options
    n_seeds = eval_cfg.get("n_seeds", 1)
    joint_search = eval_cfg.get("joint_search", False)
    joint_search_samples = eval_cfg.get("joint_search_samples", 12)
    adaptive_mode = eval_cfg.get("adaptive_mode", False)

    all_ticker_results = {}

    for ticker in tickers:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Processing ticker: {ticker}")
        logger.info(f"{'#'*60}")

        features_df, ohlcv_df, data_report = prepare_ticker_data(ticker, data_cfg)

        # Resolve interval for this ticker (for annualization)
        interval = _resolve_interval(ticker, data_cfg)

        # Target: next-period log return
        target = compute_returns(ohlcv_df).shift(-1)
        target.name = "target"

        combined = pd.concat([features_df, target], axis=1).dropna()
        feature_cols = [c for c in combined.columns if c != "target"]

        features = combined[feature_cols].values
        targets = combined["target"].values

        feature_stats = compute_feature_stats(combined[feature_cols])

        logger.info(f"Dataset: {len(combined)} samples, {len(feature_cols)} features")
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Interval: {interval}, Purge gap: {purge_gap}, "
                     f"Adaptive window: {adaptive_window}, "
                     f"Min holding period: {min_holding_period}, "
                     f"Allow short: {allow_short}, "
                     f"Classification: {classification}")

        model_results = {}

        # Cycle 6: Adaptive per-ticker hold period
        # First pass: if adaptive_hold, run hold sweep with default seq_len to find optimal
        ticker_hold = min_holding_period
        if adaptive_hold and min_hold_sweep_levels:
            logger.info(f"\n{'='*60}")
            logger.info(f"Adaptive hold: running hold sweep for {ticker} to find optimal")
            logger.info(f"{'='*60}")
            # Use first model type for the sweep (fast)
            first_mtype = model_cfg["types"][0]
            mkwargs_sweep = model_cfg.get(first_mtype, {})
            sweep_result = walk_forward_validation(
                features=features,
                targets=targets,
                model_type=first_mtype,
                model_kwargs=mkwargs_sweep,
                seq_len=train_cfg.get("seq_len", 30),
                train_size=eval_cfg.get("train_size", 500),
                test_size=eval_cfg.get("test_size", 60),
                step_size=eval_cfg.get("step_size", 60),
                epochs=train_cfg.get("epochs", 50),
                batch_size=train_cfg.get("batch_size", 32),
                lr=train_cfg.get("lr", 1e-3),
                cost_bps=eval_cfg.get("cost_bps", 10.0),
                purge_gap=purge_gap,
                interval=interval,
                adaptive_window=adaptive_window,
                min_holding_period=min_holding_period,
                allow_short=allow_short,
                classification=False,
                min_hold_sweep_levels=min_hold_sweep_levels,
            )
            if "min_hold_sweep" in sweep_result:
                ticker_hold = select_optimal_hold_period(sweep_result["min_hold_sweep"])
                logger.info(f"Adaptive hold: selected min_hold={ticker_hold} for {ticker}")
            else:
                logger.info(f"Adaptive hold: sweep failed, using default min_hold={ticker_hold}")

        # Cycle 10: Joint hyperparameter search (replaces sequential sweeps)
        model_hidden_sizes = {}
        model_num_layers = {}
        model_seq_lens = {}
        joint_search_results = {}

        if joint_search and hidden_size_levels and num_layers_levels and seq_len_sweep_levels:
            for mtype in model_cfg["types"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Joint hyperparam search for {mtype.upper()} on {ticker}")
                logger.info(f"{'='*60}")
                mkwargs = dict(model_cfg.get(mtype, {}))
                mtype_warmup = warmup_epochs if mtype == "transformer" else 0
                js_results = joint_hyperparam_search(
                    features=features,
                    targets=targets,
                    model_type=mtype,
                    model_kwargs=mkwargs,
                    hidden_size_levels=hidden_size_levels,
                    num_layers_levels=num_layers_levels,
                    seq_len_levels=seq_len_sweep_levels,
                    n_samples=joint_search_samples,
                    train_size=eval_cfg.get("train_size", 500),
                    test_size=eval_cfg.get("test_size", 60),
                    step_size=eval_cfg.get("step_size", 60),
                    epochs=train_cfg.get("epochs", 50),
                    batch_size=train_cfg.get("batch_size", 32),
                    lr=train_cfg.get("lr", 1e-3),
                    cost_bps=eval_cfg.get("cost_bps", 10.0),
                    purge_gap=purge_gap,
                    interval=interval,
                    adaptive_window=adaptive_window,
                    min_holding_period=ticker_hold,
                    allow_short=allow_short,
                    warmup_epochs=mtype_warmup,
                    early_stopping_patience=early_stopping_patience,
                )
                defaults = {
                    "hidden_size": model_cfg.get(mtype, {}).get("hidden_size",
                                   model_cfg.get(mtype, {}).get("d_model", 64)),
                    "num_layers": model_cfg.get(mtype, {}).get("num_layers", 2),
                    "seq_len": train_cfg.get("seq_len", 30),
                }
                best = select_optimal_joint_params(js_results, defaults)
                model_hidden_sizes[mtype] = best["hidden_size"]
                model_num_layers[mtype] = best["num_layers"]
                model_seq_lens[mtype] = best["seq_len"]
                joint_search_results[mtype] = js_results
                logger.info(f"Joint search: selected hs={best['hidden_size']}, "
                            f"nl={best['num_layers']}, sl={best['seq_len']} "
                            f"for {mtype.upper()} on {ticker}")
        else:
            # Fallback: sequential sweeps (Cycles 8-9 behavior)
            if adaptive_hidden_size and hidden_size_levels:
                for mtype in model_cfg["types"]:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Adaptive hidden_size: sweeping for {mtype.upper()} on {ticker}")
                    logger.info(f"{'='*60}")
                    mkwargs = model_cfg.get(mtype, {})
                    mtype_warmup = warmup_epochs if mtype == "transformer" else 0
                    hs_sweep = hidden_size_sweep(
                        features=features,
                        targets=targets,
                        model_type=mtype,
                        model_kwargs=mkwargs,
                        hidden_size_levels=hidden_size_levels,
                        seq_len=train_cfg.get("seq_len", 30),
                        train_size=eval_cfg.get("train_size", 500),
                        test_size=eval_cfg.get("test_size", 60),
                        step_size=eval_cfg.get("step_size", 60),
                        epochs=train_cfg.get("epochs", 50),
                        batch_size=train_cfg.get("batch_size", 32),
                        lr=train_cfg.get("lr", 1e-3),
                        cost_bps=eval_cfg.get("cost_bps", 10.0),
                        purge_gap=purge_gap,
                        interval=interval,
                        adaptive_window=adaptive_window,
                        min_holding_period=ticker_hold,
                        allow_short=allow_short,
                        warmup_epochs=mtype_warmup,
                        early_stopping_patience=early_stopping_patience,
                    )
                    optimal_hs = select_optimal_hidden_size(hs_sweep, 64)
                    model_hidden_sizes[mtype] = optimal_hs
                    model_hidden_sizes[f"{mtype}_sweep"] = hs_sweep
                    logger.info(f"Adaptive hidden_size: selected hidden_size={optimal_hs} for {mtype.upper()} on {ticker}")

            if adaptive_num_layers and num_layers_levels:
                for mtype in model_cfg["types"]:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Adaptive num_layers: sweeping for {mtype.upper()} on {ticker}")
                    logger.info(f"{'='*60}")
                    mkwargs = dict(model_cfg.get(mtype, {}))
                    if mtype in model_hidden_sizes:
                        if mtype in ("lstm", "gru"):
                            mkwargs["hidden_size"] = model_hidden_sizes[mtype]
                        elif mtype == "transformer":
                            nhead = mkwargs.get("nhead", 4)
                            d_model = max(model_hidden_sizes[mtype], nhead)
                            d_model = d_model - (d_model % nhead)
                            mkwargs["d_model"] = d_model
                            mkwargs["dim_feedforward"] = d_model * 2
                    mtype_warmup = warmup_epochs if mtype == "transformer" else 0
                    nl_sweep = num_layers_sweep(
                        features=features,
                        targets=targets,
                        model_type=mtype,
                        model_kwargs=mkwargs,
                        num_layers_levels=num_layers_levels,
                        seq_len=train_cfg.get("seq_len", 30),
                        train_size=eval_cfg.get("train_size", 500),
                        test_size=eval_cfg.get("test_size", 60),
                        step_size=eval_cfg.get("step_size", 60),
                        epochs=train_cfg.get("epochs", 50),
                        batch_size=train_cfg.get("batch_size", 32),
                        lr=train_cfg.get("lr", 1e-3),
                        cost_bps=eval_cfg.get("cost_bps", 10.0),
                        purge_gap=purge_gap,
                        interval=interval,
                        adaptive_window=adaptive_window,
                        min_holding_period=ticker_hold,
                        allow_short=allow_short,
                        warmup_epochs=mtype_warmup,
                        early_stopping_patience=early_stopping_patience,
                    )
                    optimal_nl = select_optimal_num_layers(nl_sweep, 2)
                    model_num_layers[mtype] = optimal_nl
                    model_num_layers[f"{mtype}_sweep"] = nl_sweep
                    logger.info(f"Adaptive num_layers: selected num_layers={optimal_nl} for {mtype.upper()} on {ticker}")

            if adaptive_seq_len and seq_len_sweep_levels:
                for mtype in model_cfg["types"]:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Adaptive seq_len: sweeping for {mtype.upper()} on {ticker}")
                    logger.info(f"{'='*60}")
                    mkwargs = dict(model_cfg.get(mtype, {}))
                    if mtype in model_hidden_sizes:
                        if mtype in ("lstm", "gru"):
                            mkwargs["hidden_size"] = model_hidden_sizes[mtype]
                        elif mtype == "transformer":
                            nhead = mkwargs.get("nhead", 4)
                            d_model = max(model_hidden_sizes[mtype], nhead)
                            d_model = d_model - (d_model % nhead)
                            mkwargs["d_model"] = d_model
                            mkwargs["dim_feedforward"] = d_model * 2
                    if mtype in model_num_layers:
                        mkwargs["num_layers"] = model_num_layers[mtype]
                    mtype_warmup = warmup_epochs if mtype == "transformer" else 0
                    sl_sweep = seq_len_sensitivity_sweep(
                        features=features,
                        targets=targets,
                        model_type=mtype,
                        model_kwargs=mkwargs,
                        seq_len_levels=seq_len_sweep_levels,
                        train_size=eval_cfg.get("train_size", 500),
                        test_size=eval_cfg.get("test_size", 60),
                        step_size=eval_cfg.get("step_size", 60),
                        epochs=train_cfg.get("epochs", 50),
                        batch_size=train_cfg.get("batch_size", 32),
                        lr=train_cfg.get("lr", 1e-3),
                        cost_bps=eval_cfg.get("cost_bps", 10.0),
                        purge_gap=purge_gap,
                        interval=interval,
                        adaptive_window=adaptive_window,
                        min_holding_period=ticker_hold,
                        allow_short=allow_short,
                        warmup_epochs=mtype_warmup,
                        early_stopping_patience=early_stopping_patience,
                    )
                    optimal_sl = select_optimal_seq_len(sl_sweep, train_cfg.get("seq_len", 30))
                    model_seq_lens[mtype] = optimal_sl
                    logger.info(f"Adaptive seq_len: selected seq_len={optimal_sl} for {mtype.upper()} on {ticker}")
                    model_seq_lens[f"{mtype}_sweep"] = sl_sweep

        # Cycle 10: Per-model adaptive mode selection
        model_modes = {}
        if adaptive_mode:
            for mtype in model_cfg["types"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Adaptive mode: sweeping regression vs classification for {mtype.upper()} on {ticker}")
                logger.info(f"{'='*60}")
                mkwargs = dict(model_cfg.get(mtype, {}))
                if mtype in model_hidden_sizes:
                    if mtype in ("lstm", "gru"):
                        mkwargs["hidden_size"] = model_hidden_sizes[mtype]
                    elif mtype == "transformer":
                        nhead = mkwargs.get("nhead", 4)
                        d_model = max(model_hidden_sizes[mtype], nhead)
                        d_model = d_model - (d_model % nhead)
                        mkwargs["d_model"] = d_model
                        mkwargs["dim_feedforward"] = d_model * 2
                if mtype in model_num_layers:
                    mkwargs["num_layers"] = model_num_layers[mtype]
                mtype_seq_len = model_seq_lens.get(mtype, train_cfg.get("seq_len", 30))
                mtype_warmup = warmup_epochs if mtype == "transformer" else 0
                ms_results = mode_selection_sweep(
                    features=features,
                    targets=targets,
                    model_type=mtype,
                    model_kwargs=mkwargs,
                    seq_len=mtype_seq_len,
                    train_size=eval_cfg.get("train_size", 500),
                    test_size=eval_cfg.get("test_size", 60),
                    step_size=eval_cfg.get("step_size", 60),
                    epochs=train_cfg.get("epochs", 50),
                    batch_size=train_cfg.get("batch_size", 32),
                    lr=train_cfg.get("lr", 1e-3),
                    cost_bps=eval_cfg.get("cost_bps", 10.0),
                    purge_gap=purge_gap,
                    interval=interval,
                    adaptive_window=adaptive_window,
                    min_holding_period=ticker_hold,
                    allow_short=allow_short,
                    warmup_epochs=mtype_warmup,
                    early_stopping_patience=early_stopping_patience,
                )
                use_cls = select_optimal_mode(ms_results)
                model_modes[mtype] = use_cls
                model_modes[f"{mtype}_sweep"] = ms_results
                mode_label = "classification" if use_cls else "regression"
                logger.info(f"Adaptive mode: selected {mode_label} for {mtype.upper()} on {ticker}")

        # Run models (regression or adaptive mode) with optional multi-seed
        for mtype in model_cfg["types"]:
            # Cycle 10: Determine mode from adaptive mode selection
            mtype_cls = model_modes.get(mtype, False) if adaptive_mode else False
            mode_label = "classification" if mtype_cls else "regression"
            logger.info(f"\n{'='*60}")
            logger.info(f"Running walk-forward for {mtype.upper()} ({mode_label}) on {ticker}")
            logger.info(f"{'='*60}")

            mkwargs = dict(model_cfg.get(mtype, {}))
            if mtype_cls:
                mkwargs["classification"] = True
            # Apply adaptive hidden size if selected
            if mtype in model_hidden_sizes:
                if mtype in ("lstm", "gru"):
                    mkwargs["hidden_size"] = model_hidden_sizes[mtype]
                elif mtype == "transformer":
                    nhead = mkwargs.get("nhead", 4)
                    d_model = max(model_hidden_sizes[mtype], nhead)
                    d_model = d_model - (d_model % nhead)
                    mkwargs["d_model"] = d_model
                    mkwargs["dim_feedforward"] = d_model * 2
                logger.info(f"  Using adaptive hidden_size={model_hidden_sizes[mtype]} for {mtype.upper()}")
            # Apply adaptive num_layers if selected
            if mtype in model_num_layers:
                mkwargs["num_layers"] = model_num_layers[mtype]
                logger.info(f"  Using adaptive num_layers={model_num_layers[mtype]} for {mtype.upper()}")
            mtype_seq_len = model_seq_lens.get(mtype, train_cfg.get("seq_len", 30))
            mtype_warmup = warmup_epochs if mtype == "transformer" else 0

            if mtype in model_seq_lens:
                logger.info(f"  Using adaptive seq_len={mtype_seq_len} for {mtype.upper()}")

            # Cycle 10: Use multi-seed averaging if n_seeds > 1
            wf_func = walk_forward_validation_multiseed if n_seeds > 1 else walk_forward_validation
            wf_kwargs = dict(
                features=features,
                targets=targets,
                model_type=mtype,
                model_kwargs=mkwargs,
                seq_len=mtype_seq_len,
                train_size=eval_cfg.get("train_size", 500),
                test_size=eval_cfg.get("test_size", 60),
                step_size=eval_cfg.get("step_size", 60),
                epochs=train_cfg.get("epochs", 50),
                batch_size=train_cfg.get("batch_size", 32),
                lr=train_cfg.get("lr", 1e-3),
                cost_bps=eval_cfg.get("cost_bps", 10.0),
                purge_gap=purge_gap,
                interval=interval,
                adaptive_window=adaptive_window,
                cost_sensitivity_levels=cost_sensitivity_levels,
                min_holding_period=ticker_hold,
                allow_short=allow_short,
                classification=mtype_cls,
                min_hold_sweep_levels=min_hold_sweep_levels,
                warmup_epochs=mtype_warmup,
                early_stopping_patience=early_stopping_patience,
            )
            if n_seeds > 1:
                wf_kwargs["n_seeds"] = n_seeds
                logger.info(f"  Using multi-seed averaging with {n_seeds} seeds")

            result = wf_func(**wf_kwargs)
            model_results[mtype] = result

            # Store joint search results if available
            if mtype in joint_search_results:
                result["joint_search"] = joint_search_results[mtype]
            # Store mode sweep results
            if f"{mtype}_sweep" in model_modes:
                result["mode_sweep"] = model_modes[f"{mtype}_sweep"]
            if adaptive_mode:
                result["adaptive_mode"] = mode_label

            # Store seq_len sweep results (from adaptive phase or run fresh)
            if f"{mtype}_sweep" in model_seq_lens:
                result["seq_len_sweep"] = model_seq_lens[f"{mtype}_sweep"]
                result["adaptive_seq_len"] = mtype_seq_len
            elif seq_len_sweep_levels and not adaptive_seq_len and not joint_search:
                logger.info(f"  Running seq_len sweep for {mtype.upper()} on {ticker}")
                sl_sweep = seq_len_sensitivity_sweep(
                    features=features,
                    targets=targets,
                    model_type=mtype,
                    model_kwargs=mkwargs,
                    seq_len_levels=seq_len_sweep_levels,
                    train_size=eval_cfg.get("train_size", 500),
                    test_size=eval_cfg.get("test_size", 60),
                    step_size=eval_cfg.get("step_size", 60),
                    epochs=train_cfg.get("epochs", 50),
                    batch_size=train_cfg.get("batch_size", 32),
                    lr=train_cfg.get("lr", 1e-3),
                    cost_bps=eval_cfg.get("cost_bps", 10.0),
                    purge_gap=purge_gap,
                    interval=interval,
                    adaptive_window=adaptive_window,
                    min_holding_period=ticker_hold,
                    allow_short=allow_short,
                    warmup_epochs=mtype_warmup,
                )
                result["seq_len_sweep"] = sl_sweep

            # Cycle 7: Regime-based short toggling comparison
            if regime_short and "all_predictions" in result and "all_actuals" in result:
                preds_arr = np.array(result["all_predictions"])
                actuals_arr = np.array(result["all_actuals"])
                regime_metrics = compute_trading_metrics_regime_short(
                    preds_arr, actuals_arr,
                    cost_bps=eval_cfg.get("cost_bps", 10.0),
                    interval=interval,
                    min_holding_period=ticker_hold,
                    classification=False,
                    vol_lookback=vol_lookback,
                    high_vol_threshold=high_vol_threshold,
                )
                result["regime_short"] = regime_metrics

                # Cycle 8: Per-ticker regime threshold calibration
                if regime_threshold_levels:
                    rt_sweep = regime_threshold_sweep(
                        preds_arr, actuals_arr,
                        threshold_levels=regime_threshold_levels,
                        cost_bps=eval_cfg.get("cost_bps", 10.0),
                        interval=interval,
                        min_holding_period=ticker_hold,
                        classification=False,
                        vol_lookback=vol_lookback,
                    )
                    result["regime_threshold_sweep"] = rt_sweep
                    optimal_threshold = select_optimal_regime_threshold(rt_sweep, high_vol_threshold)
                    result["optimal_regime_threshold"] = optimal_threshold

        # Cycle 5: Run classification mode for comparison
        # Cycle 10: Skip if adaptive_mode is enabled (mode already selected above)
        if classification and not adaptive_mode:
            for mtype in model_cfg["types"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running walk-forward for {mtype.upper()} (classification) on {ticker}")
                logger.info(f"{'='*60}")

                mkwargs = dict(model_cfg.get(mtype, {}))
                mkwargs["classification"] = True
                # Cycle 8: Apply adaptive hidden size
                if mtype in model_hidden_sizes:
                    if mtype in ("lstm", "gru"):
                        mkwargs["hidden_size"] = model_hidden_sizes[mtype]
                    elif mtype == "transformer":
                        nhead = mkwargs.get("nhead", 4)
                        d_model = max(model_hidden_sizes[mtype], nhead)
                        d_model = d_model - (d_model % nhead)
                        mkwargs["d_model"] = d_model
                        mkwargs["dim_feedforward"] = d_model * 2
                # Cycle 9: Apply adaptive num_layers
                if mtype in model_num_layers:
                    mkwargs["num_layers"] = model_num_layers[mtype]
                mtype_seq_len = model_seq_lens.get(mtype, train_cfg.get("seq_len", 30))
                mtype_warmup = warmup_epochs if mtype == "transformer" else 0

                result = walk_forward_validation(
                    features=features,
                    targets=targets,
                    model_type=mtype,
                    model_kwargs=mkwargs,
                    seq_len=mtype_seq_len,
                    train_size=eval_cfg.get("train_size", 500),
                    test_size=eval_cfg.get("test_size", 60),
                    step_size=eval_cfg.get("step_size", 60),
                    epochs=train_cfg.get("epochs", 50),
                    batch_size=train_cfg.get("batch_size", 32),
                    lr=train_cfg.get("lr", 1e-3),
                    cost_bps=eval_cfg.get("cost_bps", 10.0),
                    purge_gap=purge_gap,
                    interval=interval,
                    adaptive_window=adaptive_window,
                    cost_sensitivity_levels=cost_sensitivity_levels,
                    min_holding_period=ticker_hold,
                    allow_short=allow_short,
                    classification=True,
                    min_hold_sweep_levels=min_hold_sweep_levels,
                    warmup_epochs=mtype_warmup,
                    early_stopping_patience=early_stopping_patience,
                )
                model_results[f"{mtype}_cls"] = result

        # Cycle 5/6: Ensemble models (both equal and inverse-variance)
        # Cycle 9: Added selective ensemble that drops underperforming models
        if run_ensemble and len(model_cfg["types"]) >= 2:
            for method in ["equal", "inverse_variance"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Building ENSEMBLE ({method}) for {ticker}")
                logger.info(f"{'='*60}")
                # Only ensemble regression models
                reg_results = {m: model_results[m] for m in model_cfg["types"] if m in model_results}
                ensemble_result = _build_ensemble_result(
                    reg_results, features, targets, eval_cfg, interval,
                    cost_sensitivity_levels, ticker_hold,
                    allow_short=allow_short, classification=False,
                    ensemble_method=method,
                )
                if ensemble_result is not None:
                    model_results[f"ensemble_{method}"] = ensemble_result
                    logger.info(f"Ensemble ({method}) Sharpe: "
                                f"{ensemble_result['aggregate']['sharpe_ratio']:.4f}")
                else:
                    logger.warning(f"Could not build {method} ensemble")

            # Cycle 9: Selective ensemble - drop models with Sharpe below threshold
            if selective_ensemble:
                for method in ["equal", "inverse_variance"]:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Building SELECTIVE ENSEMBLE ({method}) for {ticker} "
                                f"(threshold={selective_ensemble_threshold})")
                    logger.info(f"{'='*60}")
                    # Filter to models with Sharpe above threshold
                    selected_results = {}
                    for m in model_cfg["types"]:
                        if m not in model_results or "error" in model_results[m]:
                            continue
                        model_sharpe = model_results[m]["aggregate"]["sharpe_ratio"]
                        if model_sharpe >= selective_ensemble_threshold:
                            selected_results[m] = model_results[m]
                            logger.info(f"  Including {m.upper()} (Sharpe={model_sharpe:.4f})")
                        else:
                            logger.info(f"  Excluding {m.upper()} (Sharpe={model_sharpe:.4f} < {selective_ensemble_threshold})")

                    if len(selected_results) >= 2:
                        sel_ensemble = _build_ensemble_result(
                            selected_results, features, targets, eval_cfg, interval,
                            cost_sensitivity_levels, ticker_hold,
                            allow_short=allow_short, classification=False,
                            ensemble_method=method,
                        )
                        if sel_ensemble is not None:
                            sel_ensemble["selective"] = True
                            sel_ensemble["excluded_models"] = [
                                m for m in model_cfg["types"]
                                if m not in selected_results and m in model_results
                            ]
                            model_results[f"selective_ensemble_{method}"] = sel_ensemble
                            logger.info(f"Selective ensemble ({method}) Sharpe: "
                                        f"{sel_ensemble['aggregate']['sharpe_ratio']:.4f}")
                    elif len(selected_results) == 1:
                        logger.info(f"Selective ensemble ({method}): only 1 model above threshold, "
                                    f"skipping (use individual model)")
                    else:
                        logger.info(f"Selective ensemble ({method}): no models above threshold")

        # Cycle 4: Feature importance analysis (skip in Cycle 5 to save time)
        feat_importance = {}
        if run_feature_importance:
            for mtype in model_cfg["types"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Computing feature importance for {mtype.upper()} on {ticker}")
                logger.info(f"{'='*60}")
                mkwargs = model_cfg.get(mtype, {})
                fi = compute_feature_importance(
                    features=features,
                    targets=targets,
                    model_type=mtype,
                    model_kwargs=mkwargs,
                    feature_names=feature_cols,
                    seq_len=train_cfg.get("seq_len", 30),
                    train_size=eval_cfg.get("train_size", 500),
                    epochs=train_cfg.get("epochs", 50),
                    batch_size=train_cfg.get("batch_size", 32),
                    lr=train_cfg.get("lr", 1e-3),
                    cost_bps=eval_cfg.get("cost_bps", 10.0),
                    interval=interval,
                )
                feat_importance[mtype] = fi
                if "ranking" in fi:
                    logger.info(f"Feature ranking ({mtype}): {fi['ranking']}")

        # Cycle 7: Collect per-model adaptive seq_len selections
        adaptive_seq_lens = {m: sl for m, sl in model_seq_lens.items()
                             if not m.endswith("_sweep")} if model_seq_lens else None

        # Cycle 8: Collect per-model adaptive hidden size selections
        adaptive_hidden_sizes = {m: hs for m, hs in model_hidden_sizes.items()
                                  if not m.endswith("_sweep")} if model_hidden_sizes else None

        # Cycle 9: Collect per-model adaptive num_layers selections
        adaptive_num_layers_map = {m: nl for m, nl in model_num_layers.items()
                                    if not m.endswith("_sweep")} if model_num_layers else None

        # Cycle 10: Collect adaptive mode selections
        adaptive_modes_map = {m: ("classification" if v else "regression")
                              for m, v in model_modes.items()
                              if not str(m).endswith("_sweep")} if model_modes else None

        all_ticker_results[ticker] = {
            "n_samples": len(combined),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "data_report": data_report,
            "feature_stats": feature_stats,
            "interval": interval,
            "adaptive_hold_period": ticker_hold if adaptive_hold else None,
            "adaptive_seq_lens": adaptive_seq_lens,
            "adaptive_hidden_sizes": adaptive_hidden_sizes,
            "adaptive_num_layers": adaptive_num_layers_map,
            "adaptive_modes": adaptive_modes_map,
            "joint_search": bool(joint_search_results),
            "n_seeds": n_seeds if n_seeds > 1 else None,
            "results": model_results,
            "feature_importance": feat_importance if feat_importance else None,
        }

    return {
        "config": config,
        "tickers": tickers,
        "ticker_results": all_ticker_results,
    }


def save_results(experiment: dict, output_dir: str = "reports/cycle_10"):
    """Save experiment results for all tickers."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "tickers": experiment["tickers"],
        "per_ticker": {},
    }

    for ticker, ticker_data in experiment["ticker_results"].items():
        ticker_metrics = {
            "n_samples": ticker_data["n_samples"],
            "n_features": ticker_data["n_features"],
            "interval": ticker_data.get("interval", "1d"),
            "data_summary": ticker_data["data_report"]["summary"],
            "models": {},
        }
        for model_name, result in ticker_data["results"].items():
            if "error" in result:
                ticker_metrics["models"][model_name] = {"error": result["error"]}
                continue
            model_metrics = {
                "aggregate": result["aggregate"],
                "baseline_buy_hold": result.get("baseline_buy_hold", {}),
                "significance_vs_baseline": result.get("significance_vs_baseline", {}),
            }
            # Fields present in walk-forward results but not ensemble
            if "stability_ratio" in result:
                model_metrics["stability_ratio"] = result["stability_ratio"]
                model_metrics["n_windows"] = result["n_windows"]
                model_metrics["positive_windows"] = result["positive_windows"]
                model_metrics["window_sharpe_stats"] = result.get("window_sharpe_stats", {})
                model_metrics["walk_forward_config"] = result.get("walk_forward_config", {})
            if "component_models" in result:
                model_metrics["component_models"] = result["component_models"]
            if "ensemble_method" in result:
                model_metrics["ensemble_method"] = result["ensemble_method"]
                model_metrics["ensemble_weights"] = result.get("ensemble_weights", {})
            if "cost_sensitivity" in result:
                model_metrics["cost_sensitivity"] = result["cost_sensitivity"]
            if "min_hold_sweep" in result:
                model_metrics["min_hold_sweep"] = result["min_hold_sweep"]
            if "seq_len_sweep" in result:
                model_metrics["seq_len_sweep"] = result["seq_len_sweep"]
            if "adaptive_seq_len" in result:
                model_metrics["adaptive_seq_len"] = result["adaptive_seq_len"]
            if "regime_short" in result:
                model_metrics["regime_short"] = result["regime_short"]
            if "regime_threshold_sweep" in result:
                model_metrics["regime_threshold_sweep"] = result["regime_threshold_sweep"]
            if "optimal_regime_threshold" in result:
                model_metrics["optimal_regime_threshold"] = result["optimal_regime_threshold"]
            if "selective" in result:
                model_metrics["selective"] = result["selective"]
                model_metrics["excluded_models"] = result.get("excluded_models", [])
            # Cycle 10 fields
            if "n_seeds" in result:
                model_metrics["n_seeds"] = result["n_seeds"]
            if "joint_search" in result:
                model_metrics["joint_search"] = result["joint_search"]
            if "mode_sweep" in result:
                model_metrics["mode_sweep"] = result["mode_sweep"]
            if "adaptive_mode" in result:
                model_metrics["adaptive_mode"] = result["adaptive_mode"]
            ticker_metrics["models"][model_name] = model_metrics

        # Cycle 4: Feature importance
        if ticker_data.get("feature_importance"):
            ticker_metrics["feature_importance"] = ticker_data["feature_importance"]

        # Cycle 6: Adaptive hold period
        if ticker_data.get("adaptive_hold_period") is not None:
            ticker_metrics["adaptive_hold_period"] = ticker_data["adaptive_hold_period"]

        # Cycle 7: Adaptive seq_len per model
        if ticker_data.get("adaptive_seq_lens"):
            ticker_metrics["adaptive_seq_lens"] = ticker_data["adaptive_seq_lens"]

        # Cycle 8: Adaptive hidden sizes per model
        if ticker_data.get("adaptive_hidden_sizes"):
            ticker_metrics["adaptive_hidden_sizes"] = ticker_data["adaptive_hidden_sizes"]

        # Cycle 9: Adaptive num_layers per model
        if ticker_data.get("adaptive_num_layers"):
            ticker_metrics["adaptive_num_layers"] = ticker_data["adaptive_num_layers"]

        # Cycle 10: Adaptive modes, joint search, multi-seed
        if ticker_data.get("adaptive_modes"):
            ticker_metrics["adaptive_modes"] = ticker_data["adaptive_modes"]
        if ticker_data.get("joint_search"):
            ticker_metrics["joint_search"] = True
        if ticker_data.get("n_seeds"):
            ticker_metrics["n_seeds"] = ticker_data["n_seeds"]

        metrics["per_ticker"][ticker] = ticker_metrics

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out / 'metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run price prediction experiment")
    parser.add_argument("command", choices=["run-experiment"])
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = run_experiment(config)

    output_dir = args.output_dir or "reports/cycle_10"
    save_results(experiment, output_dir=output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (Cycle 10)")
    print("=" * 70)
    for ticker, ticker_data in experiment["ticker_results"].items():
        print(f"\n{'─'*70}")
        adaptive_hold_str = ""
        if ticker_data.get("adaptive_hold_period") is not None:
            adaptive_hold_str = f", adaptive_hold={ticker_data['adaptive_hold_period']}"
        adaptive_sl_str = ""
        if ticker_data.get("adaptive_seq_lens"):
            sl_parts = [f"{m}={sl}" for m, sl in ticker_data["adaptive_seq_lens"].items()]
            adaptive_sl_str = f", adaptive_seq_len=[{', '.join(sl_parts)}]"
        adaptive_hs_str = ""
        if ticker_data.get("adaptive_hidden_sizes"):
            hs_parts = [f"{m}={hs}" for m, hs in ticker_data["adaptive_hidden_sizes"].items()]
            adaptive_hs_str = f", adaptive_hidden=[{', '.join(hs_parts)}]"
        adaptive_nl_str = ""
        if ticker_data.get("adaptive_num_layers"):
            nl_parts = [f"{m}={nl}" for m, nl in ticker_data["adaptive_num_layers"].items()]
            adaptive_nl_str = f", adaptive_layers=[{', '.join(nl_parts)}]"
        adaptive_mode_str = ""
        if ticker_data.get("adaptive_modes"):
            mode_parts = [f"{m}={md}" for m, md in ticker_data["adaptive_modes"].items()]
            adaptive_mode_str = f", adaptive_mode=[{', '.join(mode_parts)}]"
        seeds_str = ""
        if ticker_data.get("n_seeds"):
            seeds_str = f", n_seeds={ticker_data['n_seeds']}"
        joint_str = ""
        if ticker_data.get("joint_search"):
            joint_str = ", joint_search=true"
        print(f"Ticker: {ticker} ({ticker_data['n_samples']} samples, "
              f"{ticker_data['n_features']} features, interval={ticker_data.get('interval', '1d')}"
              f"{adaptive_hold_str}{adaptive_sl_str}{adaptive_hs_str}{adaptive_nl_str}"
              f"{adaptive_mode_str}{seeds_str}{joint_str})")
        print(f"{'─'*70}")
        for model_name, result in ticker_data["results"].items():
            if "error" in result:
                print(f"\n  {model_name.upper()}: {result['error']}")
                continue
            agg = result["aggregate"]
            bl = result.get("baseline_buy_hold", {})
            sig = result.get("significance_vs_baseline", {})
            wf = result.get("walk_forward_config", {})
            n_win = result.get('n_windows', '-')
            if wf:
                cls_label = " [CLS]" if wf.get("classification") else ""
                short_label = " [L/S]" if wf.get("allow_short") else ""
                print(f"\n  {model_name.upper()}{cls_label}{short_label} (windows={n_win}, "
                      f"train={wf.get('train_size','?')}, test={wf.get('test_size','?')}, "
                      f"purge={wf.get('purge_gap',0)}, min_hold={wf.get('min_holding_period',1)}):")
            else:
                components = result.get("component_models", [])
                method = result.get("ensemble_method", "equal")
                weights = result.get("ensemble_weights", {})
                weight_str = ", ".join(f"{m}={w:.2f}" for m, w in weights.items())
                print(f"\n  {model_name.upper()} ({method}, weights: {weight_str}):")
            print(f"    Sharpe (net):     {agg['sharpe_ratio']:.4f}  (baseline: {bl.get('sharpe_ratio', 0):.4f})")
            print(f"    Sortino (net):    {agg['sortino_ratio']:.4f}  (baseline: {bl.get('sortino_ratio', 0):.4f})")
            print(f"    Total Return:     {agg['total_return']:.4f}  (baseline: {bl.get('total_return', 0):.4f})")
            print(f"    Max Drawdown:     {agg['max_drawdown']:.4f}  (baseline: {bl.get('max_drawdown', 0):.4f})")
            print(f"    Calmar:           {agg['calmar_ratio']:.4f}  (baseline: {bl.get('calmar_ratio', 0):.4f})")
            print(f"    Win Rate:         {agg['win_rate']:.4f}")
            print(f"    Total Cost:       {agg['total_cost']:.6f}")
            if "stability_ratio" in result:
                print(f"    Stability:        {result['stability_ratio']:.4f}")
            if sig:
                print(f"    Significance:     p={sig.get('p_value', 'N/A'):.4f} "
                      f"({'significant' if sig.get('significant_5pct') else 'not significant'} at 5%)")
            if "min_hold_sweep" in result:
                print(f"    Min hold sweep:")
                for hs in result["min_hold_sweep"]:
                    print(f"      hold={hs['min_hold']:2d} → Sharpe={hs['sharpe_ratio']:.4f}, "
                          f"trades={hs['n_trades']}")
            if "seq_len_sweep" in result:
                print(f"    Seq len sweep:")
                for sl in result["seq_len_sweep"]:
                    if "error" in sl:
                        print(f"      seq_len={sl['seq_len']:3d} → {sl['error']}")
                    else:
                        print(f"      seq_len={sl['seq_len']:3d} → Sharpe={sl['sharpe_ratio']:.4f}, "
                              f"trades={sl['n_trades']}")
            if "adaptive_seq_len" in result:
                print(f"    Adaptive seq_len: {result['adaptive_seq_len']}")
            if "regime_short" in result:
                rs = result["regime_short"]
                print(f"    Regime short:     Sharpe={rs['sharpe_ratio']:.4f}, "
                      f"Return={rs['total_return']:.4f}, "
                      f"high_vol_periods={rs['high_vol_periods']}, "
                      f"shorts_disabled={rs['shorts_disabled']}")
            if "regime_threshold_sweep" in result:
                print(f"    Regime threshold sweep:")
                for rt in result["regime_threshold_sweep"]:
                    print(f"      threshold={rt['threshold']:.2f} → Sharpe={rt['sharpe_ratio']:.4f}, "
                          f"shorts_disabled={rt['shorts_disabled']}")
            if "optimal_regime_threshold" in result:
                print(f"    Optimal regime threshold: {result['optimal_regime_threshold']:.2f}")
            if "cost_sensitivity" in result:
                print(f"    Cost sensitivity:")
                for cs in result["cost_sensitivity"]:
                    print(f"      {cs['cost_bps']:5.1f} bps → Sharpe={cs['sharpe_ratio']:.4f}, "
                          f"Return={cs['total_return']:.4f}")

        # Print feature importance if available
        if ticker_data.get("feature_importance"):
            print(f"\n  Feature Importance:")
            for mtype, fi in ticker_data["feature_importance"].items():
                if "ranking" in fi:
                    print(f"    {mtype.upper()} ranking: {', '.join(fi['ranking'][:5])} (top 5)")


if __name__ == "__main__":
    main()
