# Open Questions

## Addressed in Cycle 2

- ~~**Stationarity of features (#10):** MACD and ATR normalized by dividing by close price. Contributed to improved stability.~~
- ~~**Multiple tickers (#8):** Multi-ticker pipeline implemented. BTC/USDT (hourly) and AAPL (daily) evaluated.~~

## Remaining

1. **Feature selection:** Are all 12 indicators contributing, or would a reduced set improve generalization? Feature importance analysis needed.

2. **Sequence length sensitivity:** Currently fixed at 30. How sensitive are the models to this parameter? Should it differ between LSTM and Transformer?

3. **Classification vs regression:** Direction prediction (up/down) with cross-entropy loss might produce stronger trading signals than return regression.

4. **Learning rate scheduling:** No LR schedule is used. Cosine annealing or ReduceLROnPlateau may improve convergence.

5. **Walk-forward window sizing:** train_size=500 may be too large for some datasets. BTC/USDT hourly yielded only 2 windows. Adaptive window sizing could help.

6. **Short positions:** Current strategy is long-or-flat. Allowing short positions could improve Sharpe on bearish periods.

7. **Ensemble methods:** Combining LSTM and Transformer predictions may improve stability.

8. **Transaction cost sensitivity:** 10 bps is assumed. Real crypto exchange fees vary (Binance ~2-7 bps for makers). How sensitive are results?

## New from Cycle 2

9. **Crypto data availability:** ARF API returns limited BTC/USDT daily history (~365 days). Hourly interval gives 720 bars but only 2 walk-forward windows. Need to evaluate whether shorter intervals (e.g., 15m) or different crypto tickers yield more data.

10. **Indicator periods for different frequencies:** RSI(14), MACD(12,26,9), etc. were designed for daily data. For hourly data, these periods may need adjustment (e.g., RSI(24) for daily cycle in hourly data).

11. **Annualized volatility calculation:** `historical_volatility()` uses sqrt(252) for annualization. For hourly data, this should be sqrt(252*24) ≈ sqrt(6048). Currently this is not adjusted per frequency.

12. **Transformer convergence:** Transformer loss is much higher initially than LSTM, suggesting the architecture needs more epochs or different learning rates for crypto data.
