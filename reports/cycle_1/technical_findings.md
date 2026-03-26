# Cycle 1 Technical Findings

## Implementation Summary

Implemented the core pipeline for cryptocurrency/equity price prediction using LSTM and Transformer models with momentum and volatility technical indicators.

### Modules Implemented

1. **`src/data.py`** — Data loading from ARF Data API with local caching, log return computation.
2. **`src/indicators.py`** — 12 technical indicator features:
   - **Momentum:** RSI(14), MACD(12,26,9), MACD Signal, MACD Histogram, ROC(10), Stochastic %K(14)
   - **Volatility:** ATR(14), Bollinger Bandwidth(20), Bollinger %B, Historical Volatility(20)
   - **Price:** Log return, Volume change
3. **`src/models.py`** — LSTM (2-layer, hidden=64) and Transformer encoder (2-layer, d_model=64, 4 heads).
4. **`src/training.py`** — Sliding-window dataset, train-only normalization (no data leakage), training loop with gradient clipping.
5. **`src/evaluation.py`** — Walk-forward validation with configurable window sizes, cost-aware trading metrics (10 bps), naive buy-and-hold baseline.
6. **`src/cli.py`** — YAML-configured experiment runner.

### Design Decisions

- **Target:** Next-period log return (regression task).
- **Strategy:** Long when predicted return > 0, else flat (no short positions).
- **Scaling:** Z-score normalization using train-window statistics only — no future information leakage.
- **Walk-forward:** Rolling window with train_size=500, test_size=60, step_size=60.
- **Cost model:** 10 basis points per trade (applied at each position change).

## Results

### AAPL Daily (5-year, 1222 samples, 11 walk-forward windows)

| Metric | LSTM | Transformer | Buy-and-Hold |
|--------|------|-------------|--------------|
| Sharpe Ratio (net) | -2.02 | -0.25 | -0.55 |
| Total Return (net) | -40.9% | -3.7% | -16.9% |
| Max Drawdown | 58.2% | 12.7% | 48.3% |
| Win Rate | 47.3% | 50.4% | — |
| Stability (% positive windows) | 18.2% | 27.3% | — |
| N Trades | 34 | 35 | 1 |

### BTC/USDT Hourly (1-year, 686 samples, 2 walk-forward windows)

| Metric | LSTM | Transformer | Buy-and-Hold |
|--------|------|-------------|--------------|
| Sharpe Ratio (net) | 2.14 | -1.11 | 1.41 |
| Total Return (net) | 3.7% | -1.3% | 3.1% |
| Stability | 50% | 0% | — |

## Observations

1. **Models underperform on AAPL daily data.** Both LSTM and Transformer produce negative Sharpe ratios, though Transformer (-0.25) is significantly better than LSTM (-2.02) and comparable to the buy-and-hold baseline (-0.55) on this bearish period.

2. **Transformer has lower drawdown.** The Transformer model's maximum drawdown (12.7%) is far lower than LSTM (58.2%) or buy-and-hold (48.3%), suggesting it learned to stay flat during volatile periods.

3. **LSTM on BTC hourly shows promise.** The LSTM achieved a positive Sharpe (2.14) beating buy-and-hold (1.41), but with only 2 walk-forward windows this is not statistically significant.

4. **Low stability across windows.** Only 2/11 (LSTM) and 3/11 (Transformer) windows had positive Sharpe on AAPL, indicating inconsistent performance.

5. **Model convergence.** Training losses decrease smoothly. Transformer converges slower but to comparable levels. Both models show no obvious overfitting based on loss curves.

## Cycle 1 Conclusion

The core infrastructure is fully operational: data pipeline, feature engineering, model training, walk-forward validation, and cost-aware evaluation. Both models run end-to-end and produce valid predictions. Performance is weak on the tested data, which is expected for a baseline Cycle 1 implementation without hyperparameter tuning or advanced features.
