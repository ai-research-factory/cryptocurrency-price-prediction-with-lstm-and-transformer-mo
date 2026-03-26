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
python3 -m src.cli run-experiment --config configs/default.yaml --output-dir reports/cycle_8

# Run tests
python3 -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to configure:
- **Data:** Tickers list, interval, period (fetched from ARF Data API), per-ticker overrides
- **Models:** LSTM, GRU, and/or Transformer with architecture parameters
- **Training:** Sequence length, epochs, batch size, learning rate
- **Evaluation:** Walk-forward window sizes, purge gap, adaptive windowing, cost sensitivity levels, minimum holding period, feature importance, ensemble, classification mode, long/short strategy, seq_len sweep, adaptive hold, adaptive seq_len, warm-up, regime short, early stopping, hidden size search, regime threshold calibration

### Per-Ticker Overrides

```yaml
data:
  tickers: ["BTC/USDT", "ETH/USDT", "AAPL", "SPY", "MSFT"]
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

### Cycle 8 Evaluation Options

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
  adaptive_seq_len: true                          # Auto-select best seq_len per model/ticker (Cycle 7)
  warmup_epochs: 5                                # Transformer LR warm-up epochs (Cycle 7)
  regime_short: true                              # Volatility-regime-based short toggling (Cycle 7)
  vol_lookback: 60                                # Rolling vol lookback for regime detection (Cycle 7)
  high_vol_threshold: 1.5                         # High-vol threshold multiplier (Cycle 7)
  early_stopping_patience: 7                      # Early stopping patience in epochs (Cycle 8)
  regime_threshold_sweep: [1.0, 1.25, 1.5, 2.0, 2.5]  # Per-ticker regime calibration (Cycle 8)
  hidden_size_sweep: [32, 64, 128]                # Per-model hidden size search (Cycle 8)
  adaptive_hidden_size: true                      # Auto-select best hidden size per model/ticker (Cycle 8)
```

## Project Structure

```
src/
  data.py            -- Data fetching, validation, cleaning, and caching
  preprocessing.py   -- Feature normalization, outlier clipping, stationarity transforms
  indicators.py      -- Technical indicators with frequency-adaptive periods (Cycle 6)
  models.py          -- LSTM, GRU, and Transformer architectures (Cycle 6: +GRU)
  training.py        -- Dataset preparation, training with LR scheduling + warm-up + early stopping (Cycle 8)
  evaluation.py      -- Purged walk-forward, cost sensitivity, risk metrics, regime short, threshold/hidden sweep (Cycle 8)
  cli.py             -- CLI experiment runner (5-ticker, adaptive architecture, regime calibration)
configs/
  default.yaml       -- Default experiment configuration
tests/               -- Unit tests for all modules (79 tests)
reports/cycle_1/     -- Cycle 1 experiment results
reports/cycle_2/     -- Cycle 2 experiment results
reports/cycle_3/     -- Cycle 3 experiment results
reports/cycle_4/     -- Cycle 4 experiment results
reports/cycle_5/     -- Cycle 5 experiment results
reports/cycle_6/     -- Cycle 6 experiment results
reports/cycle_7/     -- Cycle 7 experiment results
reports/cycle_8/     -- Cycle 8 experiment results
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
- **Ensemble model:** LSTM + GRU + Transformer 3-model ensemble (Cycle 6, fixed in Cycle 8)
- **Feature importance:** Permutation-based feature importance analysis (Cycle 4)
- **Learning rate scheduling:** ReduceLROnPlateau for better convergence (Cycle 4)
- **Direction classification:** BCE loss with sigmoid output for direction prediction (Cycle 5)
- **Long/short strategy:** Allows short positions for bearish signals (Cycle 5)
- **Min holding period sweep:** Finds optimal trade frequency per ticker/model (Cycle 5)
- **Inverse-variance ensemble:** Weights models by inverse return variance (Cycle 5)
- **GRU model:** Third model type for genuine ensemble diversity (Cycle 6)
- **Sequence length sweep:** Tests model sensitivity to lookback window size (Cycle 6)
- **Adaptive per-ticker hold:** Auto-selects optimal hold period per ticker (Cycle 6)
- **Adaptive per-model seq_len:** Auto-selects optimal lookback per model/ticker (Cycle 7)
- **Transformer warm-up:** Linear LR warm-up for better Transformer convergence (Cycle 7)
- **Volatility-regime short toggling:** Disables shorts in high-volatility regimes (Cycle 7)
- **Early stopping:** Reduces overfitting with validation-based early stopping (Cycle 8)
- **Per-model hidden size search:** Auto-selects optimal architecture per model/ticker (Cycle 8)
- **Regime threshold calibration:** Per-ticker optimal regime threshold via sweep (Cycle 8)

## Cycle 8 Results

### BTC/USDT (Hourly, adaptive_hold=5, adaptive hidden/seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | 1.49 | +1.2% | 8 | hidden=32, seq=50 |
| **LSTM (classification)** | **2.04** | **+1.7%** | 2 | Beats baseline |
| **Ensemble (equal)** | **7.10** | **+6.1%** | 11 | **Best overall** |
| Buy-Hold | -0.76 | -0.6% | -- | BTC in downtrend |

### ETH/USDT (Hourly, adaptive_hold=5, adaptive hidden/seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| **Transformer (regression)** | **1.54** | **+2.9%** | 10 | hidden=32, seq=30, beats baseline |
| Ensemble (equal) | 0.11 | +0.2% | 6 | |
| Buy-Hold | -0.13 | -- | -- | ETH in downtrend |

### AAPL (Daily, adaptive_hold=3, adaptive hidden/seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| Transformer (regression) | -0.04 | -2.2% | 53 | hidden=32, seq=20 |
| Transformer (classification) | -0.01 | -0.8% | 23 | Near flat |
| Buy-Hold | 0.35-0.58 | varied | -- | |

### SPY (Daily, adaptive_hold=10, adaptive hidden/seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| **Transformer (regression)** | **1.70** | **+63.9%** | 29 | **hidden=128, seq=50, beats baseline** |
| **Transformer (regime short)** | **2.40** | improved | -- | threshold=2.0 |
| LSTM (classification) | 0.31 | +11.9% | 23 | |
| Buy-Hold | 0.98 | +33.3% | -- | |

### MSFT (Daily, adaptive_hold=10, adaptive hidden/seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| **GRU (regression)** | **0.67** | **+35.9%** | 26 | hidden=128, seq=30 |
| **GRU (regime, threshold=1.0)** | **1.14** | improved | -- | 65 shorts disabled |
| Transformer (regime short) | 0.24 | +11.0% | -- | threshold=1.25 |
| Buy-Hold | 0.99-1.05 | +57-65% | -- | |

### Key findings:
- **Ensemble fix restores key capability**: BTC ensemble (Sharpe 7.10) is the best project result
- **SPY Transformer beats buy-and-hold** (Sharpe 1.70, regime 2.40 vs baseline 0.98) -- first significant equity win
- **Per-model hidden size search finds diverse optima**: no universal best size; smaller models often preferred
- **Early stopping reduces overfitting**: combined with hidden size search for better regularization
- **Per-ticker regime calibration improves results**: MSFT GRU improves from 0.67 to 1.14

See `reports/cycle_8/` for full metrics and details.
