# Cryptocurrency Price Prediction with LSTM and Transformer Models

Predicts asset returns using LSTM and Transformer models with momentum and volatility technical indicators, evaluated via purged walk-forward validation with transaction cost sensitivity analysis.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run experiment (multi-ticker with ensemble, classification, long/short)
python3 -m src.cli run-experiment --config configs/default.yaml

# Custom output directory
python3 -m src.cli run-experiment --config configs/default.yaml --output-dir reports/cycle_5

# Run tests
python3 -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to configure:
- **Data:** Tickers list, interval, period (fetched from ARF Data API), per-ticker overrides
- **Models:** LSTM and/or Transformer with architecture parameters
- **Training:** Sequence length, epochs, batch size, learning rate
- **Evaluation:** Walk-forward window sizes, purge gap, adaptive windowing, cost sensitivity levels, minimum holding period, feature importance, ensemble, classification mode, long/short strategy

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

### Cycle 5 Evaluation Options

```yaml
evaluation:
  purge_gap: 5                                    # Gap between train/test to prevent leakage
  adaptive_window: true                           # Auto-size windows for sufficient samples
  cost_sensitivity_bps: [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # Cost sweep
  min_holding_period: 3                           # Minimum periods before position change
  feature_importance: false                       # Permutation feature importance analysis
  ensemble: true                                  # LSTM+Transformer ensemble
  classification: true                            # Direction classification mode (BCE loss)
  allow_short: true                               # Long/short strategy
  ensemble_method: "inverse_variance"             # "equal" or "inverse_variance"
  min_hold_sweep: [1, 2, 3, 5, 10]               # Min hold period sweep levels
```

## Project Structure

```
src/
  data.py            -- Data fetching, validation, cleaning, and caching
  preprocessing.py   -- Feature normalization, outlier clipping, stationarity transforms
  indicators.py      -- Technical indicators (RSI, MACD, ATR, Bollinger, etc.)
  models.py          -- LSTM and Transformer architectures (regression + classification)
  training.py        -- Dataset preparation, training loop with LR scheduling, prediction
  evaluation.py      -- Purged walk-forward, cost sensitivity, risk metrics, hold sweep
  cli.py             -- CLI experiment runner (multi-ticker, ensemble, classification)
configs/
  default.yaml       -- Default experiment configuration
tests/               -- Unit tests for all modules (51 tests)
reports/cycle_1/     -- Cycle 1 experiment results
reports/cycle_2/     -- Cycle 2 experiment results
reports/cycle_3/     -- Cycle 3 experiment results
reports/cycle_4/     -- Cycle 4 experiment results
reports/cycle_5/     -- Cycle 5 experiment results
```

## Features

- **Multi-ticker support:** Run experiments across multiple assets (crypto + equities)
- **Data quality pipeline:** Validation, cleaning, outlier removal, gap handling
- **Feature preprocessing:** Price-indicator normalization for stationarity, extreme value clipping
- **12 technical indicators:** RSI, MACD (line/signal/histogram), ROC, Stochastic %K, ATR, Bollinger Bandwidth, Bollinger %B, Historical Volatility, Log Return, Volume Change
- **Purged walk-forward validation:** Configurable gap between train/test sets; adaptive window sizing
- **Interval-aware annualization:** Correct Sharpe/Sortino calculation for daily, hourly, and other frequencies
- **Cost sensitivity analysis:** Evaluate strategies across multiple transaction cost levels
- **Risk metrics:** Sharpe, Sortino, Calmar ratios -- all net of transaction costs
- **Statistical significance:** Paired t-test vs buy-and-hold baseline
- **Baseline comparison:** Buy-and-hold benchmark included
- **Minimum holding period:** Reduces trade churn from noisy predictions (Cycle 4)
- **Ensemble model:** Averages LSTM + Transformer predictions (Cycle 4)
- **Feature importance:** Permutation-based feature importance analysis (Cycle 4)
- **Learning rate scheduling:** ReduceLROnPlateau for better convergence (Cycle 4)
- **Direction classification:** BCE loss with sigmoid output for direction prediction (Cycle 5)
- **Long/short strategy:** Allows short positions for bearish signals (Cycle 5)
- **Min holding period sweep:** Finds optimal trade frequency per ticker/model (Cycle 5)
- **Inverse-variance ensemble:** Weights models by inverse return variance (Cycle 5)

## Cycle 5 Results

### BTC/USDT (Hourly, 4 windows, min_hold=3, long/short)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -0.40 | -0.6% | 23 | |
| Transformer (regression) | **3.92** | **+6.5%** | 2 | Low trade count |
| LSTM (classification) | -0.10 | -0.2% | 32 | |
| Transformer (classification) | -5.81 | -8.9% | 30 | |
| Ensemble (equal) | 3.85 | +6.4% | 4 | |
| Buy-Hold | 0.81 | +1.3% | -- | |

### ETH/USDT (Hourly, 4 windows, min_hold=3, long/short)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -5.37 | -10.2% | 19 | |
| Transformer (regression) | -1.67 | -3.3% | 28 | |
| LSTM (classification) | -8.63 | -15.8% | 18 | |
| Transformer (classification) | -5.23 | -9.9% | 24 | |
| Ensemble (equal) | -9.95 | -18.0% | 26 | |
| Buy-Hold | 0.06 | +0.1% | -- | |

### AAPL (Daily, 3 windows, min_hold=3, long/short)

| Model | Sharpe | Return | Max DD | Trades | Notes |
|-------|--------|--------|--------|--------|-------|
| LSTM (regression) | 0.09 | +4.8% | 38.4% | 50 | |
| Transformer (regression) | 0.17 | +9.5% | 37.1% | 33 | |
| LSTM (classification) | -1.02 | -42.8% | 68.4% | 61 | |
| Transformer (classification) | -0.12 | -6.5% | 34.3% | 71 | |
| **Ensemble (equal)** | **0.39** | **+23.7%** | **26.9%** | 48 | **Beats buy-hold** |
| Buy-Hold | 0.35 | +21.0% | 40.6% | -- | |

Key findings:
- **AAPL ensemble beats buy-and-hold** for the first time: Sharpe 0.39 vs 0.35 with lower drawdown (26.9% vs 40.6%)
- **Classification is worse than regression** in 5 of 6 model/ticker combinations
- **Long/short helps equities** (AAPL) but **hurts crypto** (ETH)
- **Min hold sweep** reveals per-ticker optima: BTC LSTM optimal at hold=5 (Sharpe 1.71 vs -0.40 at hold=3)
- No model achieves statistical significance vs buy-and-hold (p < 0.05)

See `reports/cycle_5/` for full metrics and details.
