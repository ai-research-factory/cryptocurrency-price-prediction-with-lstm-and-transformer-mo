# Cryptocurrency Price Prediction with LSTM and Transformer Models

Predicts asset returns using LSTM and Transformer models with momentum and volatility technical indicators, evaluated via walk-forward validation with transaction costs.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run experiment
python3 -m src.cli run-experiment --config configs/default.yaml

# Run tests
python3 -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to configure:
- **Data:** Ticker, interval, period (fetched from ARF Data API)
- **Models:** LSTM and/or Transformer with architecture parameters
- **Training:** Sequence length, epochs, batch size, learning rate
- **Evaluation:** Walk-forward window sizes, transaction cost (bps)

## Project Structure

```
src/
  data.py          — Data fetching and caching from ARF API
  indicators.py    — Technical indicators (RSI, MACD, ATR, Bollinger, etc.)
  models.py        — LSTM and Transformer architectures
  training.py      — Dataset preparation, training loop, prediction
  evaluation.py    — Walk-forward validation, trading metrics, baseline
  cli.py           — CLI experiment runner
configs/
  default.yaml     — Default experiment configuration
tests/             — Unit tests for all modules
reports/cycle_1/   — Cycle 1 experiment results
```

## Features

- **12 technical indicators:** RSI, MACD (line/signal/histogram), ROC, Stochastic %K, ATR, Bollinger Bandwidth, Bollinger %B, Historical Volatility, Log Return, Volume Change
- **Walk-forward validation:** No future information leakage; train-only scaling
- **Cost-aware metrics:** Sharpe ratio, max drawdown, win rate — all net of transaction costs
- **Baseline comparison:** Buy-and-hold benchmark included

## Cycle 1 Results (AAPL Daily, 11 windows)

| Model | Sharpe (net) | Return | Max DD | Stability |
|-------|-------------|--------|--------|-----------|
| LSTM | -2.02 | -40.9% | 58.2% | 18.2% |
| Transformer | -0.25 | -3.7% | 12.7% | 27.3% |
| Buy-and-Hold | -0.55 | -16.9% | 48.3% | — |

Transformer outperforms LSTM and matches buy-and-hold on a bearish data period. See `reports/cycle_1/` for details.
