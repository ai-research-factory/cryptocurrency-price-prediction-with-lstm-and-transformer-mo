# Cryptocurrency Price Prediction with LSTM and Transformer Models

Predicts asset returns using LSTM and Transformer models with momentum and volatility technical indicators, evaluated via walk-forward validation with transaction costs.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run experiment (multi-ticker)
python3 -m src.cli run-experiment --config configs/default.yaml

# Run tests
python3 -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to configure:
- **Data:** Tickers list, interval, period (fetched from ARF Data API), per-ticker overrides
- **Models:** LSTM and/or Transformer with architecture parameters
- **Training:** Sequence length, epochs, batch size, learning rate
- **Evaluation:** Walk-forward window sizes, transaction cost (bps)

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

## Project Structure

```
src/
  data.py            — Data fetching, validation, cleaning, and caching
  preprocessing.py   — Feature normalization, outlier clipping, stationarity transforms
  indicators.py      — Technical indicators (RSI, MACD, ATR, Bollinger, etc.)
  models.py          — LSTM and Transformer architectures
  training.py        — Dataset preparation, training loop, prediction
  evaluation.py      — Walk-forward validation, trading metrics, baseline
  cli.py             — CLI experiment runner (multi-ticker)
configs/
  default.yaml       — Default experiment configuration
tests/               — Unit tests for all modules
reports/cycle_1/     — Cycle 1 experiment results
reports/cycle_2/     — Cycle 2 experiment results
```

## Features

- **Multi-ticker support:** Run experiments across multiple assets (crypto + equities)
- **Data quality pipeline:** Validation, cleaning, outlier removal, gap handling
- **Feature preprocessing:** Price-indicator normalization for stationarity, extreme value clipping
- **12 technical indicators:** RSI, MACD (line/signal/histogram), ROC, Stochastic %K, ATR, Bollinger Bandwidth, Bollinger %B, Historical Volatility, Log Return, Volume Change
- **Walk-forward validation:** No future information leakage; train-only scaling
- **Cost-aware metrics:** Sharpe ratio, max drawdown, win rate — all net of transaction costs
- **Baseline comparison:** Buy-and-hold benchmark included

## Cycle 2 Results

### BTC/USDT (Hourly, 1-year, 2 windows)

| Model | Sharpe (net) | Return | Max DD | Stability |
|-------|-------------|--------|--------|-----------|
| LSTM | -0.79 | -0.7% | 2.0% | 50% |
| Transformer | 1.54 | +3.1% | 1.9% | 50% |
| Buy-and-Hold | 1.41 | +3.1% | 1.9% | — |

### AAPL (Daily, 5-year, 11 windows)

| Model | Sharpe (net) | Return | Max DD | Stability |
|-------|-------------|--------|--------|-----------|
| LSTM | -1.27 | -30.8% | 56.5% | 45.5% |
| Transformer | 0.08 | +2.1% | 24.2% | 54.5% |
| Buy-and-Hold | -0.55 | -16.9% | 48.3% | — |

Transformer outperforms buy-and-hold on both tickers. Preprocessing improvements (indicator normalization) improved AAPL stability from 27.3% (Cycle 1) to 54.5%. See `reports/cycle_2/` for details.
