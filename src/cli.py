"""CLI entry point for running experiments."""

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
from .evaluation import walk_forward_validation

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


def run_experiment(config: dict) -> dict:
    """Run full experiment pipeline with Cycle 3 enhancements."""
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
                     f"Adaptive window: {adaptive_window}")

        model_results = {}
        for mtype in model_cfg["types"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running walk-forward validation for {mtype.upper()} on {ticker}")
            logger.info(f"{'='*60}")

            mkwargs = model_cfg.get(mtype, {})
            mkwargs["input_size"] = len(feature_cols)

            result = walk_forward_validation(
                features=features,
                targets=targets,
                model_type=mtype,
                model_kwargs={k: v for k, v in mkwargs.items() if k != "input_size"},
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
            )
            model_results[mtype] = result

        all_ticker_results[ticker] = {
            "n_samples": len(combined),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "data_report": data_report,
            "feature_stats": feature_stats,
            "interval": interval,
            "results": model_results,
        }

    return {
        "config": config,
        "tickers": tickers,
        "ticker_results": all_ticker_results,
    }


def save_results(experiment: dict, output_dir: str = "reports/cycle_3"):
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
                "baseline_buy_hold": result["baseline_buy_hold"],
                "significance_vs_baseline": result.get("significance_vs_baseline", {}),
                "stability_ratio": result["stability_ratio"],
                "n_windows": result["n_windows"],
                "positive_windows": result["positive_windows"],
                "window_sharpe_stats": result.get("window_sharpe_stats", {}),
                "walk_forward_config": result.get("walk_forward_config", {}),
            }
            if "cost_sensitivity" in result:
                model_metrics["cost_sensitivity"] = result["cost_sensitivity"]
            ticker_metrics["models"][model_name] = model_metrics
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

    output_dir = args.output_dir or "reports/cycle_3"
    save_results(experiment, output_dir=output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY (Cycle 3)")
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
            bl = result["baseline_buy_hold"]
            sig = result.get("significance_vs_baseline", {})
            wf = result.get("walk_forward_config", {})
            print(f"\n  {model_name.upper()} (windows={result['n_windows']}, "
                  f"train={wf.get('train_size','?')}, test={wf.get('test_size','?')}, "
                  f"purge={wf.get('purge_gap',0)}):")
            print(f"    Sharpe (net):     {agg['sharpe_ratio']:.4f}  (baseline: {bl['sharpe_ratio']:.4f})")
            print(f"    Sortino (net):    {agg['sortino_ratio']:.4f}  (baseline: {bl['sortino_ratio']:.4f})")
            print(f"    Total Return:     {agg['total_return']:.4f}  (baseline: {bl['total_return']:.4f})")
            print(f"    Max Drawdown:     {agg['max_drawdown']:.4f}  (baseline: {bl['max_drawdown']:.4f})")
            print(f"    Calmar:           {agg['calmar_ratio']:.4f}  (baseline: {bl['calmar_ratio']:.4f})")
            print(f"    Win Rate:         {agg['win_rate']:.4f}")
            print(f"    Total Cost:       {agg['total_cost']:.6f}")
            print(f"    Stability:        {result['stability_ratio']:.4f}")
            if sig:
                print(f"    Significance:     p={sig.get('p_value', 'N/A'):.4f} "
                      f"({'significant' if sig.get('significant_5pct') else 'not significant'} at 5%)")
            if "cost_sensitivity" in result:
                print(f"    Cost sensitivity:")
                for cs in result["cost_sensitivity"]:
                    print(f"      {cs['cost_bps']:5.1f} bps → Sharpe={cs['sharpe_ratio']:.4f}, "
                          f"Return={cs['total_return']:.4f}")


if __name__ == "__main__":
    main()
