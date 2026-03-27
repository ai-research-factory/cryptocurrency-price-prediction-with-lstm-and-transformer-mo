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
python3 -m src.cli run-experiment --config configs/default.yaml --output-dir reports/cycle_11

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

### Cycle 11 Evaluation Options

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
  early_stopping_patience: 5                      # Early stopping patience in epochs (Cycle 8)
  regime_threshold_sweep: [1.0, 1.25, 1.5, 2.0, 2.5]  # Per-ticker regime calibration (Cycle 8)
  hidden_size_sweep: [32, 64, 128]                # Per-model hidden size search (Cycle 8)
  adaptive_hidden_size: true                      # Auto-select best hidden size per model/ticker (Cycle 8)
  num_layers_sweep: [1, 2, 3]                     # Per-model num_layers search (Cycle 9)
  adaptive_num_layers: true                       # Auto-select best depth per model/ticker (Cycle 9)
  selective_ensemble: true                        # Drop models with negative Sharpe from ensemble (Cycle 9)
  selective_ensemble_threshold: 0.0               # Minimum Sharpe to include in selective ensemble (Cycle 9)
  n_seeds: 3                                      # Multi-seed prediction averaging (Cycle 10, increased Cycle 11)
  joint_search: true                              # Joint hyperparameter search (Cycle 10)
  joint_search_samples: 12                        # Number of random configs to evaluate (Cycle 10, increased Cycle 11)
  adaptive_mode: true                             # Auto-select regression vs classification per model/ticker (Cycle 10)
  confidence_weighted: true                       # Scale position by prediction confidence (Cycle 11)
  sharpe_loss: true                               # Sharpe-aware training loss for regression (Cycle 11)
  sharpe_loss_weight: 0.3                         # Blend weight: (1-w)*MSE + w*(-Sharpe) (Cycle 11)
  dropout_sweep: [0.1, 0.2, 0.3]                  # Dropout levels in joint search (Cycle 11)
```

## Project Structure

```
src/
  data.py            -- Data fetching, validation, cleaning, and caching
  preprocessing.py   -- Feature normalization, outlier clipping, stationarity transforms
  indicators.py      -- Technical indicators with frequency-adaptive periods (Cycle 6)
  models.py          -- LSTM, GRU, and Transformer architectures (Cycle 6: +GRU)
  training.py        -- Dataset preparation, training with LR scheduling + warm-up + early stopping + Sharpe-aware loss (Cycle 11)
  evaluation.py      -- Purged walk-forward, cost sensitivity, risk metrics, regime short, sweeps, multi-seed, joint search, mode selection, confidence weighting (Cycle 11)
  cli.py             -- CLI experiment runner (adaptive architecture, multi-seed, joint search, adaptive mode, confidence weighting, Sharpe loss)
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
reports/cycle_10/    -- Cycle 10 experiment results
reports/cycle_11/    -- Cycle 11 experiment results
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
- **Multi-seed prediction averaging:** Averages predictions across N random seeds to reduce initialization variance (Cycle 10)
- **Joint hyperparameter search:** Random search over (hidden_size, num_layers, seq_len, dropout) space replacing sequential sweeps (Cycle 10/11)
- **Adaptive mode selection:** Auto-selects regression vs classification per model/ticker (Cycle 10)
- **Confidence-weighted position sizing:** Scale positions by prediction magnitude instead of binary (Cycle 11)
- **Position change threshold:** Suppresses micro-trades (< 10% position change) from confidence weighting (Cycle 11)
- **Sharpe-aware loss:** Blended MSE + differentiable Sharpe ratio loss for regression models (Cycle 11)
- **Dropout in joint search:** Adds regularization tuning [0.1, 0.2, 0.3] to hyperparameter search space (Cycle 11)

## Cycle 11 Results

### Per-Ticker Best Models (5 tickers, n_seeds=3, joint_search=12, dropout search, Sharpe loss, confidence-weighted + position threshold)

| Ticker   | Best Model          | Sharpe | Return  | Baseline | Stability |
|----------|---------------------|--------|---------|----------|-----------|
| AAPL     | Ensemble (inv-var)  | -0.00  | -0.2%   | 0.35     | --        |
| SPY      | Ensemble (equal)    | 0.98   | +33.3%  | 0.98     | --        |
| MSFT     | GRU (reg)           | 0.61   | +33.3%  | 1.01     | 1.00      |
| BTC/USDT | LSTM (cls)          | 2.17   | +0.2%   | -0.76    | 0.75      |
| ETH/USDT | LSTM (cls)          | 5.67   | +0.7%   | 0.46     | 0.50      |

### Key Findings (Cycle 11):
- **Fixed dropout search bug**: `dropout_sweep` config was defined but never passed to joint search. Now properly varies dropout [0.1, 0.2, 0.3] per model.
- **Position change threshold reduces micro-trades**: Confidence-weighted positions now suppress changes < 10%, dramatically reducing trade costs from continuous position sizing.
- **SPY ensemble matches baseline exactly**: With position threshold, confidence-weighted ensemble holds constant position, correctly reflecting no alpha.
- **MSFT GRU achieves perfect stability**: HS=64, NL=3, SL=10 — deep GRU with short lookback. 3/3 positive windows, Sharpe 0.61.
- **Crypto classification improved**: BTC/USDT LSTM achieves Sharpe 2.17 (previously all-negative). ETH/USDT LSTM reaches 5.67, but likely overfitting with only 661 samples.
- **No statistical significance**: After 11 cycles, no model reliably outperforms buy-and-hold (p > 0.05), consistent with weak-form EMH.

See `reports/cycle_11/` for full metrics and details.
