"""CLI entry point for running experiments.

Cycle 4: Added ensemble model, feature importance, min holding period, ETH/USDT.
Cycle 5: Added classification mode, long/short strategy, min hold sweep,
inverse-variance ensemble weighting.
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
    features_df = compute_all_indicators(df)
    features_df = preprocess_features(features_df, df)

    data_report = {
        "validation": validation,
        "summary": summary,
        "n_rows_after_clean": len(df),
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
    ensemble_method: "equal" for simple average, "inverse_variance" for
    weighting by inverse of per-model return variance.
    """
    # Collect all_preds and all_actuals from each model's window_details
    model_all_preds = {}
    for mtype, result in model_results.items():
        if "error" in result or "all_predictions" not in result:
            continue
        model_all_preds[mtype] = np.array(result["all_predictions"])

    if len(model_all_preds) < 2:
        return None

    # All models should have same length predictions
    lengths = [len(p) for p in model_all_preds.values()]
    if len(set(lengths)) != 1:
        logger.warning("Model prediction lengths differ, cannot ensemble")
        return None

    all_actuals = np.array(model_results[list(model_all_preds.keys())[0]]["all_actuals"])
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
    """Run full experiment pipeline with Cycle 5 enhancements."""
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

        # Run regression mode (baseline from previous cycles)
        for mtype in model_cfg["types"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running walk-forward validation for {mtype.upper()} (regression) on {ticker}")
            logger.info(f"{'='*60}")

            mkwargs = model_cfg.get(mtype, {})

            result = walk_forward_validation(
                features=features,
                targets=targets,
                model_type=mtype,
                model_kwargs=mkwargs,
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
                cost_sensitivity_levels=cost_sensitivity_levels,
                min_holding_period=min_holding_period,
                allow_short=allow_short,
                classification=False,
                min_hold_sweep_levels=min_hold_sweep_levels,
            )
            model_results[mtype] = result

        # Cycle 5: Run classification mode for comparison
        if classification:
            for mtype in model_cfg["types"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running walk-forward for {mtype.upper()} (classification) on {ticker}")
                logger.info(f"{'='*60}")

                mkwargs = dict(model_cfg.get(mtype, {}))
                mkwargs["classification"] = True

                result = walk_forward_validation(
                    features=features,
                    targets=targets,
                    model_type=mtype,
                    model_kwargs=mkwargs,
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
                    cost_sensitivity_levels=cost_sensitivity_levels,
                    min_holding_period=min_holding_period,
                    allow_short=allow_short,
                    classification=True,
                    min_hold_sweep_levels=min_hold_sweep_levels,
                )
                model_results[f"{mtype}_cls"] = result

        # Cycle 5: Ensemble models (both equal and inverse-variance)
        if run_ensemble and len(model_cfg["types"]) >= 2:
            for method in ["equal", "inverse_variance"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Building ENSEMBLE ({method}) for {ticker}")
                logger.info(f"{'='*60}")
                # Only ensemble regression models
                reg_results = {m: model_results[m] for m in model_cfg["types"] if m in model_results}
                ensemble_result = _build_ensemble_result(
                    reg_results, features, targets, eval_cfg, interval,
                    cost_sensitivity_levels, min_holding_period,
                    allow_short=allow_short, classification=False,
                    ensemble_method=method,
                )
                if ensemble_result is not None:
                    model_results[f"ensemble_{method}"] = ensemble_result
                    logger.info(f"Ensemble ({method}) Sharpe: "
                                f"{ensemble_result['aggregate']['sharpe_ratio']:.4f}")
                else:
                    logger.warning(f"Could not build {method} ensemble")

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

        all_ticker_results[ticker] = {
            "n_samples": len(combined),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "data_report": data_report,
            "feature_stats": feature_stats,
            "interval": interval,
            "results": model_results,
            "feature_importance": feat_importance if feat_importance else None,
        }

    return {
        "config": config,
        "tickers": tickers,
        "ticker_results": all_ticker_results,
    }


def save_results(experiment: dict, output_dir: str = "reports/cycle_5"):
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
            ticker_metrics["models"][model_name] = model_metrics

        # Cycle 4: Feature importance
        if ticker_data.get("feature_importance"):
            ticker_metrics["feature_importance"] = ticker_data["feature_importance"]

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

    output_dir = args.output_dir or "reports/cycle_5"
    save_results(experiment, output_dir=output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (Cycle 5)")
    print("=" * 70)
    for ticker, ticker_data in experiment["ticker_results"].items():
        print(f"\n{'─'*70}")
        print(f"Ticker: {ticker} ({ticker_data['n_samples']} samples, "
              f"{ticker_data['n_features']} features, interval={ticker_data.get('interval', '1d')})")
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
