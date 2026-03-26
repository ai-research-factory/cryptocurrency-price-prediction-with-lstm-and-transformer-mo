# Cryptocurrency Price Prediction with LSTM and Transformer Models

Predicts asset returns using LSTM and Transformer models with momentum and volatility technical indicators, evaluated via purged walk-forward validation with transaction cost sensitivity analysis.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run experiment (multi-ticker with ensemble)
python3 -m src.cli run-experiment --config configs/default.yaml

# Custom output directory
python3 -m src.cli run-experiment --config configs/default.yaml --output-dir reports/cycle_4

# Run tests
python3 -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to configure:
- **Data:** Tickers list, interval, period (fetched from ARF Data API), per-ticker overrides
- **Models:** LSTM and/or Transformer with architecture parameters
- **Training:** Sequence length, epochs, batch size, learning rate
- **Evaluation:** Walk-forward window sizes, purge gap, adaptive windowing, cost sensitivity levels, minimum holding period, feature importance, ensemble

### Per-Ticker Overrides

```yaml
data:
  tickers: ["BTC/USDT", "ETH/USDT", "AAPL"]
  interval: "1d"
  period: "5y"
  ticker_overrides:
    "BTC/USDT":
      interval: "1h"
      period: "1y"
    "ETH/USDT":
      interval: "1h"
      period: "1y"
```

### Cycle 4 Evaluation Options

```yaml
evaluation:
  purge_gap: 5                                    # Gap between train/test to prevent leakage
  adaptive_window: true                           # Auto-size windows for sufficient samples
  cost_sensitivity_bps: [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # Cost sweep
  min_holding_period: 3                           # Minimum periods before position change
  feature_importance: true                        # Permutation feature importance analysis
  ensemble: true                                  # LSTM+Transformer ensemble averaging
```

## Project Structure

```
src/
  data.py            — Data fetching, validation, cleaning, and caching
  preprocessing.py   — Feature normalization, outlier clipping, stationarity transforms
  indicators.py      — Technical indicators (RSI, MACD, ATR, Bollinger, etc.)
  models.py          — LSTM and Transformer architectures
  training.py        — Dataset preparation, training loop with LR scheduling, prediction
  evaluation.py      — Purged walk-forward, cost sensitivity, risk metrics, feature importance
  cli.py             — CLI experiment runner (multi-ticker, ensemble)
configs/
  default.yaml       — Default experiment configuration
tests/               — Unit tests for all modules (55 tests)
reports/cycle_1/     — Cycle 1 experiment results
reports/cycle_2/     — Cycle 2 experiment results
reports/cycle_3/     — Cycle 3 experiment results
reports/cycle_4/     — Cycle 4 experiment results
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
- **Minimum holding period:** Reduces trade churn from noisy predictions (Cycle 4)
- **Ensemble model:** Averages LSTM + Transformer predictions for diversification (Cycle 4)
- **Feature importance:** Permutation-based feature importance analysis (Cycle 4)
- **Learning rate scheduling:** ReduceLROnPlateau for better convergence (Cycle 4)

## Cycle 4 Results

### BTC/USDT (Hourly, 4 windows, min_hold=3)

| Model | Sharpe | Sortino | Return | Max DD | Trades | p-value |
|-------|--------|---------|--------|--------|--------|---------|
| LSTM | -1.46 | -1.16 | -1.8% | 7.8% | 23 | 0.54 |
| Transformer | -2.18 | -1.72 | -2.4% | 5.2% | 19 | 0.52 |
| Ensemble | -1.32 | -1.11 | -1.6% | 8.6% | 21 | 0.58 |
| Buy-Hold | 0.81 | 0.80 | +1.3% | 8.2% | — | — |

### ETH/USDT (Hourly, 4 windows, min_hold=3)

| Model | Sharpe | Sortino | Return | Max DD | Trades | p-value |
|-------|--------|---------|--------|--------|--------|---------|
| LSTM | -6.51 | -4.49 | -8.7% | 11.2% | 19 | 0.19 |
| Transformer | -0.79 | -0.44 | -0.9% | 5.8% | 9 | 0.89 |
| Ensemble | -1.04 | -0.60 | -1.1% | 5.1% | 11 | 0.87 |
| Buy-Hold | 0.06 | 0.06 | +0.1% | 11.3% | — | — |

### AAPL (Daily, 3 windows, min_hold=3)

| Model | Sharpe | Sortino | Return | Max DD | Trades | p-value |
|-------|--------|---------|--------|--------|--------|---------|
| LSTM | -0.06 | -0.05 | -2.8% | 36.6% | 40 | 0.25 |
| Transformer | 0.21 | 0.20 | +11.9% | 40.6% | 17 | 0.35 |
| Ensemble | 0.06 | 0.05 | +3.1% | 40.6% | 34 | 0.25 |
| Buy-Hold | 0.35 | 0.33 | +21.0% | 40.6% | — | — |

Key finding: Minimum holding period reduced Transformer trades by 60-67%, improving AAPL Transformer from Sharpe -0.09 (Cycle 3) to +0.21 (Cycle 4). No model achieves statistical significance vs buy-and-hold (p < 0.05). See `reports/cycle_4/` for details.
