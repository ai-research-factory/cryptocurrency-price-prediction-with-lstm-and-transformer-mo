"""CLI entry point for running experiments."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import fetch_ohlcv, compute_returns
from .indicators import compute_all_indicators
from .evaluation import walk_forward_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(config: dict) -> dict:
    """Run full experiment pipeline."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    # 1. Fetch data
    logger.info(f"Fetching data for {data_cfg['ticker']}...")
    df = fetch_ohlcv(
        ticker=data_cfg["ticker"],
        interval=data_cfg.get("interval", "1d"),
        period=data_cfg.get("period", "5y"),
    )

    # 2. Compute indicators
    logger.info("Computing technical indicators...")
    features_df = compute_all_indicators(df)

    # 3. Target: next-day log return
    target = compute_returns(df).shift(-1)  # predict next day's return
    target.name = "target"

    # 4. Align and drop NaN
    combined = pd.concat([features_df, target], axis=1).dropna()
    feature_cols = [c for c in combined.columns if c != "target"]

    features = combined[feature_cols].values
    targets = combined["target"].values

    logger.info(f"Dataset: {len(combined)} samples, {len(feature_cols)} features")
    logger.info(f"Features: {feature_cols}")

    # 5. Run walk-forward for each model type
    results = {}
    for mtype in model_cfg["types"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running walk-forward validation for {mtype.upper()}")
        logger.info(f"{'='*60}")

        mkwargs = model_cfg.get(mtype, {})
        mkwargs["input_size"] = len(feature_cols)  # will be overridden by build_model

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
        )
        results[mtype] = result

    return {
        "config": config,
        "ticker": data_cfg["ticker"],
        "n_samples": len(combined),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "results": results,
    }


def save_results(experiment: dict, output_dir: str = "reports/cycle_1"):
    """Save experiment results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics = {
        "ticker": experiment["ticker"],
        "n_samples": experiment["n_samples"],
        "n_features": experiment["n_features"],
        "models": {},
    }
    for model_name, result in experiment["results"].items():
        if "error" in result:
            metrics["models"][model_name] = {"error": result["error"]}
            continue
        metrics["models"][model_name] = {
            "aggregate": result["aggregate"],
            "baseline_buy_hold": result["baseline_buy_hold"],
            "stability_ratio": result["stability_ratio"],
            "n_windows": result["n_windows"],
            "positive_windows": result["positive_windows"],
        }

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out / 'metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description="Run price prediction experiment")
    parser.add_argument("command", choices=["run-experiment"])
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = run_experiment(config)
    save_results(experiment)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for model_name, result in experiment["results"].items():
        if "error" in result:
            print(f"\n{model_name.upper()}: {result['error']}")
            continue
        agg = result["aggregate"]
        bl = result["baseline_buy_hold"]
        print(f"\n{model_name.upper()}:")
        print(f"  Sharpe Ratio (net):  {agg['sharpe_ratio']:.4f}")
        print(f"  Total Return (net):  {agg['total_return']:.4f}")
        print(f"  Max Drawdown:        {agg['max_drawdown']:.4f}")
        print(f"  Win Rate:            {agg['win_rate']:.4f}")
        print(f"  Stability:           {result['stability_ratio']:.4f}")
        print(f"  Baseline Sharpe:     {bl['sharpe_ratio']:.4f}")
        print(f"  Baseline Return:     {bl['total_return']:.4f}")


if __name__ == "__main__":
    main()
