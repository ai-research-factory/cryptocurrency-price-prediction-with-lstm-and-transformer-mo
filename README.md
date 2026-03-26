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
python3 -m src.cli run-experiment --config configs/default.yaml --output-dir reports/cycle_7

# Run tests
python3 -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to configure:
- **Data:** Tickers list, interval, period (fetched from ARF Data API), per-ticker overrides
- **Models:** LSTM, GRU, and/or Transformer with architecture parameters
- **Training:** Sequence length, epochs, batch size, learning rate
- **Evaluation:** Walk-forward window sizes, purge gap, adaptive windowing, cost sensitivity levels, minimum holding period, feature importance, ensemble, classification mode, long/short strategy, seq_len sweep, adaptive hold, adaptive seq_len, warm-up, regime short

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

### Cycle 7 Evaluation Options

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
```

## Project Structure

```
src/
  data.py            -- Data fetching, validation, cleaning, and caching
  preprocessing.py   -- Feature normalization, outlier clipping, stationarity transforms
  indicators.py      -- Technical indicators with frequency-adaptive periods (Cycle 6)
  models.py          -- LSTM, GRU, and Transformer architectures (Cycle 6: +GRU)
  training.py        -- Dataset preparation, training with LR scheduling + warm-up (Cycle 7)
  evaluation.py      -- Purged walk-forward, cost sensitivity, risk metrics, regime short (Cycle 7)
  cli.py             -- CLI experiment runner (5-ticker, adaptive seq_len, regime short)
configs/
  default.yaml       -- Default experiment configuration
tests/               -- Unit tests for all modules (69 tests)
reports/cycle_1/     -- Cycle 1 experiment results
reports/cycle_2/     -- Cycle 2 experiment results
reports/cycle_3/     -- Cycle 3 experiment results
reports/cycle_4/     -- Cycle 4 experiment results
reports/cycle_5/     -- Cycle 5 experiment results
reports/cycle_6/     -- Cycle 6 experiment results
reports/cycle_7/     -- Cycle 7 experiment results
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
- **Adaptive per-model seq_len:** Auto-selects optimal lookback per model/ticker (Cycle 7)
- **Transformer warm-up:** Linear LR warm-up for better Transformer convergence (Cycle 7)
- **Volatility-regime short toggling:** Disables shorts in high-volatility regimes (Cycle 7)

## Cycle 7 Results

### BTC/USDT (Hourly, adaptive_hold=3, adaptive seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -1.05 | -2.1% | 16 | seq_len=10 |
| GRU (regression) | -7.64 | -14.3% | 20 | seq_len=10 |
| **Transformer (regression)** | **2.35** | **+3.4%** | 13 | seq_len=30, beats baseline |
| Buy-Hold | 0.16 | +0.3% | -- | |

### ETH/USDT (Hourly, adaptive_hold=2, adaptive seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| **LSTM (classification)** | **3.82** | **+3.7%** | 7 | **Beats baseline** |
| Transformer (regression) | -3.47 | -3.2% | 18 | seq_len=50 |
| Buy-Hold | 0.82 | +0.8% | -- | |

### AAPL (Daily, adaptive_hold=5, adaptive seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| **Transformer (regression)** | **0.56** | **+38.5%** | 5 | seq_len=20, near baseline |
| GRU (regression) | -0.03 | -1.8% | 37 | seq_len=30 |
| Buy-Hold | 0.59 | +43.1% | -- | |

### SPY (Daily, adaptive_hold=5) — NEW

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (classification) | 0.41 | +12.8% | 47 | Best model |
| Buy-Hold | 0.98 | +33.3% | -- | |

### MSFT (Daily, adaptive_hold=1) — NEW

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| Transformer (regression) | 0.11 | +5.6% | 40 | Best regression |
| Regime short Transformer | 0.84 | -- | -- | +47 shorts disabled |
| Buy-Hold | 1.01 | +69.9% | -- | |

### Key findings:
- **No model consistently beats buy-and-hold** across 5 tickers, consistent with EMH
- **Adaptive seq_len confirms model-specific preferences**: Transformer prefers 20-30; LSTM/GRU vary widely
- **Regime short toggling helps equities**: MSFT Transformer improves 0.11 → 0.84 with regime-based short disabling
- **BTC Transformer remains best single-ticker result** (Sharpe 2.35), but lower than Cycle 6 (5.30) due to different data period
- **SPY and MSFT results confirm limited predictability** for trending equities

See `reports/cycle_7/` for full metrics and details.
