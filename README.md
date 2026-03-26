# Cryptocurrency Price Prediction with LSTM and Transformer Models

Predicts asset returns using LSTM, GRU, and Transformer models with momentum and volatility technical indicators, evaluated via purged walk-forward validation with transaction cost sensitivity analysis.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run experiment (multi-ticker with 3-model ensemble, classification, long/short)
python3 -m src.cli run-experiment --config configs/default.yaml

# Custom output directory
python3 -m src.cli run-experiment --config configs/default.yaml --output-dir reports/cycle_6

# Run tests
python3 -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to configure:
- **Data:** Tickers list, interval, period (fetched from ARF Data API), per-ticker overrides
- **Models:** LSTM, GRU, and/or Transformer with architecture parameters
- **Training:** Sequence length, epochs, batch size, learning rate
- **Evaluation:** Walk-forward window sizes, purge gap, adaptive windowing, cost sensitivity levels, minimum holding period, feature importance, ensemble, classification mode, long/short strategy, seq_len sweep, adaptive hold

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

### Cycle 6 Evaluation Options

```yaml
evaluation:
  purge_gap: 5                                    # Gap between train/test to prevent leakage
  adaptive_window: true                           # Auto-size windows for sufficient samples
  cost_sensitivity_bps: [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # Cost sweep
  min_holding_period: 3                           # Default minimum periods before position change
  feature_importance: false                       # Permutation feature importance analysis
  ensemble: true                                  # Multi-model ensemble
  classification: true                            # Direction classification mode (BCE loss)
  allow_short: true                               # Long/short strategy
  ensemble_method: "inverse_variance"             # "equal" or "inverse_variance"
  min_hold_sweep: [1, 2, 3, 5, 10]               # Min hold period sweep levels
  seq_len_sweep: [10, 20, 30, 50]                 # Sequence length sensitivity sweep (Cycle 6)
  adaptive_hold: true                             # Auto-select best hold per ticker (Cycle 6)
```

## Project Structure

```
src/
  data.py            -- Data fetching, validation, cleaning, and caching
  preprocessing.py   -- Feature normalization, outlier clipping, stationarity transforms
  indicators.py      -- Technical indicators with frequency-adaptive periods (Cycle 6)
  models.py          -- LSTM, GRU, and Transformer architectures (Cycle 6: +GRU)
  training.py        -- Dataset preparation, training loop with LR scheduling, prediction
  evaluation.py      -- Purged walk-forward, cost sensitivity, risk metrics, seq_len sweep
  cli.py             -- CLI experiment runner (multi-ticker, 3-model ensemble, adaptive hold)
configs/
  default.yaml       -- Default experiment configuration
tests/               -- Unit tests for all modules (60 tests)
reports/cycle_1/     -- Cycle 1 experiment results
reports/cycle_2/     -- Cycle 2 experiment results
reports/cycle_3/     -- Cycle 3 experiment results
reports/cycle_4/     -- Cycle 4 experiment results
reports/cycle_5/     -- Cycle 5 experiment results
reports/cycle_6/     -- Cycle 6 experiment results
```

## Features

- **Multi-ticker support:** Run experiments across multiple assets (crypto + equities)
- **Data quality pipeline:** Validation, cleaning, outlier removal, gap handling
- **Feature preprocessing:** Price-indicator normalization for stationarity, extreme value clipping
- **12 technical indicators:** RSI, MACD (line/signal/histogram), ROC, Stochastic %K, ATR, Bollinger Bandwidth, Bollinger %B, Historical Volatility, Log Return, Volume Change
- **Frequency-adaptive indicator periods:** Hourly data uses longer lookback windows aligned with daily cycles (Cycle 6)
- **Purged walk-forward validation:** Configurable gap between train/test sets; adaptive window sizing
- **Interval-aware annualization:** Correct Sharpe/Sortino calculation for daily, hourly, and other frequencies
- **Cost sensitivity analysis:** Evaluate strategies across multiple transaction cost levels
- **Risk metrics:** Sharpe, Sortino, Calmar ratios -- all net of transaction costs
- **Statistical significance:** Paired t-test vs buy-and-hold baseline
- **Baseline comparison:** Buy-and-hold benchmark included
- **Minimum holding period:** Reduces trade churn from noisy predictions (Cycle 4)
- **Ensemble model:** LSTM + GRU + Transformer 3-model ensemble (Cycle 6)
- **Feature importance:** Permutation-based feature importance analysis (Cycle 4)
- **Learning rate scheduling:** ReduceLROnPlateau for better convergence (Cycle 4)
- **Direction classification:** BCE loss with sigmoid output for direction prediction (Cycle 5)
- **Long/short strategy:** Allows short positions for bearish signals (Cycle 5)
- **Min holding period sweep:** Finds optimal trade frequency per ticker/model (Cycle 5)
- **Inverse-variance ensemble:** Weights models by inverse return variance (Cycle 5)
- **GRU model:** Third model type for genuine ensemble diversity (Cycle 6)
- **Sequence length sweep:** Tests model sensitivity to lookback window size (Cycle 6)
- **Adaptive per-ticker hold:** Auto-selects optimal hold period per ticker (Cycle 6)

## Cycle 6 Results

### BTC/USDT (Hourly, 4 windows, adaptive_hold=1, freq-adaptive indicators)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -0.08 | -0.1% | 23 | |
| **GRU (regression)** | **2.65** | **+4.2%** | 5 | **New model** |
| **Transformer (regression)** | **5.30** | **+8.1%** | 2 | Best single model |
| **Ensemble (equal, 3-model)** | **4.85** | **+7.3%** | 1 | Beats buy-hold |
| Buy-Hold | 1.24 | +1.8% | -- | |

### ETH/USDT (Hourly, 4 windows, adaptive_hold=10, freq-adaptive indicators)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -0.31 | -0.5% | 10 | Best regression |
| GRU (regression) | -5.49 | -8.5% | 5 | |
| Transformer (regression) | -1.12 | -1.7% | 5 | |
| Ensemble (equal) | -0.44 | -0.7% | 5 | |
| Buy-Hold | -0.13 | -0.2% | -- | Negative period |

### AAPL (Daily, 3 windows, adaptive_hold=5, standard indicators)

| Model | Sharpe | Return | Max DD | Trades | Notes |
|-------|--------|--------|--------|--------|-------|
| LSTM (regression) | -0.27 | -11.1% | 39.3% | 40 | |
| GRU (regression) | 0.01 | +0.5% | 32.3% | 43 | Near break-even |
| Transformer (regression) | -1.13 | -35.1% | 55.3% | 38 | |
| Ensemble (equal) | -0.97 | -41.2% | 71.1% | -- | |
| Buy-Hold | 0.35 | +21.0% | 40.6% | -- | |

### Key findings:
- **GRU provides genuine ensemble diversity**: Different prediction profile from LSTM despite similar architecture
- **Seq_len is highly sensitive**: BTC prefers 30, ETH LSTM prefers 50 (Sharpe 7.31 vs -6.94), AAPL Transformer prefers 20 (Sharpe 1.72)
- **Adaptive hold improves per-ticker**: BTC=1, ETH=10, AAPL=5 selected automatically
- **BTC ensemble improved**: 3-model ensemble Sharpe 4.85 (up from 3.85 in Cycle 5)
- **AAPL Cycle 5 result was fragile**: Ensemble regressed from 0.39 to -0.97 with parameter changes
- **No statistical significance**: Consistent with prior cycles (p > 0.05 everywhere meaningful)

See `reports/cycle_6/` for full metrics and details.
