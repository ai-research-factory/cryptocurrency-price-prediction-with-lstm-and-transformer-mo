# Cycle 2: Technical Findings

## Objective
Add real data fetching and preprocessing for multi-ticker experiments, including cryptocurrency (BTC/USDT) and equity (AAPL) data.

## Implementation

### New Module: `src/preprocessing.py`
- **Price-indicator normalization**: MACD and ATR are divided by close price to make them stationary and comparable across tickers and time periods. This addresses the open question from Cycle 1 about non-stationary indicators.
- **Extreme value clipping**: Rolling z-score clipping (5-std) using expanding windows to avoid look-ahead bias.
- **Inf/NaN handling**: Replaces infinite values (e.g., from division-by-zero in volume change) with forward-filled values.
- **Feature statistics**: Reports per-feature mean, std, skew, kurtosis for diagnostics.

### Enhanced Module: `src/data.py`
- **`fetch_multiple_tickers()`**: Batch fetching for multiple tickers with error handling per ticker.
- **`validate_ohlcv()`**: Data quality checks — duplicate indices, negative prices, high/low violations, zero-volume rows, large time gaps.
- **`clean_ohlcv()`**: Automated cleaning — deduplication, negative price removal, high/low swap fix, forward-fill of small gaps, extreme return outlier removal (|z| > 10).
- **`compute_data_summary()`**: Statistical summary (return mean/std/skew/kurtosis, volatility, price range).

### Updated Module: `src/cli.py`
- Multi-ticker pipeline: iterates over a list of tickers, running full preprocessing + walk-forward per ticker.
- Per-ticker config overrides: allows different interval/period per ticker (e.g., hourly for crypto, daily for equities).
- Results saved with per-ticker structure in metrics.json.

### Configuration
- `configs/default.yaml` now supports `tickers` list and `ticker_overrides` for per-ticker interval/period.
- BTC/USDT uses 1h interval, 1y period (720 data points).
- AAPL uses 1d interval, 5y period (1256 data points).

## Results

### BTC/USDT (1-hour, 1-year, 686 usable samples, 2 walk-forward windows)

| Metric | LSTM | Transformer | Buy-and-Hold |
|--------|------|-------------|--------------|
| Sharpe (net) | -0.79 | **1.54** | 1.41 |
| Total Return | -0.7% | **3.1%** | 3.1% |
| Max Drawdown | 2.0% | 1.9% | 1.9% |
| Win Rate | 43.8% | 53.8% | — |
| Stability | 50% (1/2) | 50% (1/2) | — |
| Trades | 8 | 7 | — |

- Transformer **outperforms buy-and-hold** on Sharpe (1.54 vs 1.41) with comparable return.
- LSTM underperforms significantly with negative Sharpe.
- Only 2 windows due to limited data; results need more data for statistical significance.
- BTC hourly data shows lower annualized volatility (9.0%) than AAPL daily (27.3%) over this specific period.

### AAPL (Daily, 5-year, 1222 usable samples, 11 walk-forward windows)

| Metric | LSTM | Transformer | Buy-and-Hold |
|--------|------|-------------|--------------|
| Sharpe (net) | -1.27 | **0.08** | -0.55 |
| Total Return | -30.8% | **2.1%** | -16.9% |
| Max Drawdown | 56.5% | 24.2% | 48.3% |
| Win Rate | 48.2% | 52.4% | — |
| Stability | 45.5% (5/11) | **54.5% (6/11)** | — |
| Trades | 52 | 38 | — |

- Transformer beats buy-and-hold substantially (Sharpe 0.08 vs -0.55; return +2.1% vs -16.9%).
- Transformer stability improved from Cycle 1 (27.3% → 54.5%), likely due to normalized indicators.
- LSTM remains weak with -1.27 Sharpe, though stability improved (18.2% → 45.5%).

## Comparison with Cycle 1 (AAPL)

| Metric | Cycle 1 Transformer | Cycle 2 Transformer | Change |
|--------|--------------------|--------------------|--------|
| Sharpe (net) | -0.25 | 0.08 | +0.33 |
| Total Return | -3.7% | +2.1% | +5.8pp |
| Max Drawdown | 12.7% | 24.2% | +11.5pp |
| Stability | 27.3% | 54.5% | +27.2pp |

The preprocessing pipeline (especially indicator normalization) improved Sharpe and stability noticeably. The higher drawdown may be due to the model taking more positions.

## Observations

1. **Indicator normalization matters**: Dividing ATR and MACD by price makes them comparable across time and tickers. This likely contributed to the Transformer stability improvement.
2. **BTC hourly vs daily**: The API returned only 365 daily bars for BTC (30 days). Hourly data (720 bars) provided enough for 2 walk-forward windows. More historical data would be needed for robust evaluation.
3. **Data quality**: Both datasets passed validation with no issues (no missing data, no OHLC violations).
4. **Transformer consistently outperforms LSTM**: Across both tickers and both cycles.
5. **Limited crypto windows**: 2 windows is insufficient for reliable conclusions about BTC performance.

## Next Steps (Cycle 3)
- Add walk-forward validation improvements (expanding window, multiple window sizes)
- Implement proper cost modeling (different cost structures for crypto vs equity)
- Address the limited crypto data window issue with more historical data or shorter intervals
