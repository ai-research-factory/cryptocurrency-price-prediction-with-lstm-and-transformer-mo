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
python3 -m src.cli run-experiment --config configs/default.yaml --output-dir reports/cycle_9

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

### Cycle 9 Evaluation Options

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
  num_layers_sweep: [1, 2, 3]                     # Per-model num_layers search (Cycle 9)
  adaptive_num_layers: true                       # Auto-select best depth per model/ticker (Cycle 9)
  selective_ensemble: true                        # Drop models with negative Sharpe from ensemble (Cycle 9)
  selective_ensemble_threshold: 0.0               # Minimum Sharpe to include in selective ensemble (Cycle 9)
```

## Project Structure

```
src/
  data.py            -- Data fetching, validation, cleaning, and caching
  preprocessing.py   -- Feature normalization, outlier clipping, stationarity transforms
  indicators.py      -- Technical indicators with frequency-adaptive periods (Cycle 6)
  models.py          -- LSTM, GRU, and Transformer architectures (Cycle 6: +GRU)
  training.py        -- Dataset preparation, training with LR scheduling + warm-up + early stopping (Cycle 9: warmup-aware)
  evaluation.py      -- Purged walk-forward, cost sensitivity, risk metrics, regime short, threshold/hidden/layers sweep (Cycle 9)
  cli.py             -- CLI experiment runner (5-ticker, adaptive architecture, selective ensemble)
configs/
  default.yaml       -- Default experiment configuration
tests/               -- Unit tests for all modules (83 tests)
reports/cycle_1/     -- Cycle 1 experiment results
reports/cycle_2/     -- Cycle 2 experiment results
reports/cycle_3/     -- Cycle 3 experiment results
reports/cycle_4/     -- Cycle 4 experiment results
reports/cycle_5/     -- Cycle 5 experiment results
reports/cycle_6/     -- Cycle 6 experiment results
reports/cycle_7/     -- Cycle 7 experiment results
reports/cycle_8/     -- Cycle 8 experiment results
reports/cycle_9/     -- Cycle 9 experiment results
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
- **Selective ensemble:** Drops underperforming models (Sharpe < threshold) from ensemble (Cycle 9)
- **Per-model num_layers search:** Auto-selects optimal depth [1, 2, 3] per model/ticker (Cycle 9)
- **Warmup-aware early stopping:** Early stopping skips warmup phase to let LR stabilize (Cycle 9)

## Cycle 9 Results

### AAPL (Daily, adaptive_hold=5, adaptive HS/NL/SL)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| **Transformer (regression)** | **0.66** | **+43.6%** | 38 | HS=32, NL=1, SL=30, stability=1.0 |
| Transformer (regime, thr=1.0) | 0.96 | improved | -- | 135 shorts disabled |
| Buy-Hold | 0.35 | +21.0% | -- | |

### ETH/USDT (Hourly, adaptive_hold=3, adaptive HS/NL/SL)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| **GRU (regression)** | **3.66** | **+8.2%** | 11 | HS=32, NL=3, SL=20, stability=0.75 |
| Buy-Hold | 0.46 | +1.0% | -- | |

### MSFT (Daily, adaptive_hold=2, adaptive HS/NL/SL)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| **Selective Ensemble** | **1.05** | **+64.9%** | -- | All 3 models positive |
| **GRU (regime, thr=1.5)** | **1.08** | **+73.2%** | -- | 10 shorts disabled |
| Transformer (regression) | 0.86 | +51.2% | 77 | HS=32, NL=2, SL=20 |
| Buy-Hold | 1.01-1.05 | +65-70% | -- | |

### SPY (Daily, adaptive_hold=3, adaptive HS/NL/SL)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM CLS | 0.76 | +29.7% | 9 | Classification beats regression |
| Buy-Hold | 1.16 | +48.3% | -- | |

### Key Findings (Cycle 9):
- **num_layers preferences are model-specific**: GRU prefers 3 layers across all tickers; Transformer prefers 1-2
- **Selective ensemble works when all models are positive**: MSFT selective ensemble (1.05) matches full ensemble since all models contribute positively
- **Results vary across data periods**: BTC went from Cycle 8 ensemble 7.10 to all-negative, confirming market-regime dependence
- **AAPL Transformer achieves perfect stability** (3/3 positive windows) with shallow architecture (NL=1)
- **Early stopping + warmup fix ensures proper Transformer training**: warmup phase is no longer interrupted

See `reports/cycle_9/` for full metrics and details.
