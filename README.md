# Cryptocurrency Price Prediction with LSTM and Transformer Models

Predicts asset returns using LSTM and Transformer models with momentum and volatility technical indicators, evaluated via purged walk-forward validation with transaction cost sensitivity analysis.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run experiment (multi-ticker)
python3 -m src.cli run-experiment --config configs/default.yaml

# Custom output directory
python3 -m src.cli run-experiment --config configs/default.yaml --output-dir reports/cycle_3

# Run tests
python3 -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to configure:
- **Data:** Tickers list, interval, period (fetched from ARF Data API), per-ticker overrides
- **Models:** LSTM and/or Transformer with architecture parameters
- **Training:** Sequence length, epochs, batch size, learning rate
- **Evaluation:** Walk-forward window sizes, purge gap, adaptive windowing, cost sensitivity levels

### Per-Ticker Overrides

```yaml
data:
  tickers: ["BTC/USDT", "AAPL"]
  interval: "1d"
  period: "5y"
  ticker_overrides:
    "BTC/USDT":
      interval: "1h"
      period: "1y"
```

### Cycle 3 Evaluation Options

```yaml
evaluation:
  purge_gap: 5                                    # Gap between train/test to prevent leakage
  adaptive_window: true                           # Auto-size windows for sufficient samples
  cost_sensitivity_bps: [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # Cost sweep
```

## Project Structure

```
src/
  data.py            — Data fetching, validation, cleaning, and caching
  preprocessing.py   — Feature normalization, outlier clipping, stationarity transforms
  indicators.py      — Technical indicators (RSI, MACD, ATR, Bollinger, etc.)
  models.py          — LSTM and Transformer architectures
  training.py        — Dataset preparation, training loop, prediction
  evaluation.py      — Purged walk-forward validation, cost sensitivity, risk metrics
  cli.py             — CLI experiment runner (multi-ticker)
configs/
  default.yaml       — Default experiment configuration
tests/               — Unit tests for all modules
reports/cycle_1/     — Cycle 1 experiment results
reports/cycle_2/     — Cycle 2 experiment results
reports/cycle_3/     — Cycle 3 experiment results
```

## Features

- **Multi-ticker support:** Run experiments across multiple assets (crypto + equities)
- **Data quality pipeline:** Validation, cleaning, outlier removal, gap handling
- **Feature preprocessing:** Price-indicator normalization for stationarity, extreme value clipping
- **12 technical indicators:** RSI, MACD (line/signal/histogram), ROC, Stochastic %K, ATR, Bollinger Bandwidth, Bollinger %B, Historical Volatility, Log Return, Volume Change
- **Purged walk-forward validation:** Configurable gap between train/test sets; adaptive window sizing
- **Interval-aware annualization:** Correct Sharpe/Sortino calculation for daily, hourly, and other frequencies
- **Cost sensitivity analysis:** Evaluate strategies across multiple transaction cost levels
- **Risk metrics:** Sharpe, Sortino, Calmar ratios — all net of transaction costs
- **Statistical significance:** Paired t-test vs buy-and-hold baseline
- **Baseline comparison:** Buy-and-hold benchmark included

## Cycle 3 Results

### BTC/USDT (Hourly, 4 windows, purge=5)

| Model | Sharpe | Sortino | Return | Max DD | Stability | p-value |
|-------|--------|---------|--------|--------|-----------|---------|
| LSTM | -0.68 | -0.48 | -0.7% | 4.8% | 50% | 0.73 |
| Transformer | -0.88 | -0.76 | -1.1% | 5.2% | 50% | 0.65 |
| Buy-Hold | 0.81 | 0.80 | +1.3% | 8.2% | — | — |

### AAPL (Daily, 3 windows, purge=5)

| Model | Sharpe | Sortino | Return | Max DD | Stability | p-value |
|-------|--------|---------|--------|--------|-----------|---------|
| LSTM | 0.08 | 0.07 | +4.0% | 34.9% | 67% | 0.33 |
| Transformer | -0.09 | -0.07 | -4.0% | 40.6% | 33% | 0.24 |
| Buy-Hold | 0.35 | 0.33 | +21.0% | 40.6% | — | — |

No model achieves statistical significance vs buy-and-hold (p < 0.05). Cost sensitivity shows strategies are viable only at low cost levels (< 7 bps BTC, < 17 bps AAPL). See `reports/cycle_3/` for details.
