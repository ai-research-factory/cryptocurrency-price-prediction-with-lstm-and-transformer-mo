# Open Questions

## Addressed in Cycle 2

- ~~**Stationarity of features (#10):** MACD and ATR normalized by dividing by close price. Contributed to improved stability.~~
- ~~**Multiple tickers (#8):** Multi-ticker pipeline implemented. BTC/USDT (hourly) and AAPL (daily) evaluated.~~

## Addressed in Cycle 3

- ~~**Walk-forward window sizing (#5):** Adaptive window sizing implemented. BTC/USDT now gets 4 windows (vs 2). Guarantees minimum 3 windows per ticker.~~
- ~~**Transaction cost sensitivity (#8):** Sweep across [0, 2, 5, 10, 20, 50] bps implemented. Both tickers show break-even around 7 bps (BTC) and 17 bps (AAPL).~~
- ~~**Annualized volatility calculation (#11):** Fixed — hourly data now uses sqrt(6048), daily uses sqrt(252). Per-interval annualization factors in evaluation.py.~~

## Remaining

1. **Feature selection:** Are all 12 indicators contributing, or would a reduced set improve generalization? Feature importance analysis needed.

2. **Sequence length sensitivity:** Currently fixed at 30. How sensitive are the models to this parameter? Should it differ between LSTM and Transformer?

3. **Classification vs regression:** Direction prediction (up/down) with cross-entropy loss might produce stronger trading signals than return regression.

4. **Learning rate scheduling:** No LR schedule is used. Cosine annealing or ReduceLROnPlateau may improve convergence.

5. **Short positions:** Current strategy is long-or-flat. Allowing short positions could improve Sharpe on bearish periods.

6. **Ensemble methods:** Combining LSTM and Transformer predictions may improve stability.

7. **Indicator periods for different frequencies:** RSI(14), MACD(12,26,9), etc. were designed for daily data. For hourly data, these periods may need adjustment (e.g., RSI(24) for daily cycle in hourly data).

8. **Transformer convergence:** Transformer loss is much higher initially than LSTM, suggesting the architecture needs more epochs or different learning rates for crypto data.

## New from Cycle 3

9. **No statistical significance:** No model achieves p < 0.05 vs buy-and-hold on either ticker. With only 3-4 windows, the test has low statistical power. More data or expanding to additional tickers would help.

10. **Trade frequency control:** Transformer generates ~2x more trades than LSTM, amplifying cost drag. A position-change penalty or minimum holding period could reduce unnecessary trades.

11. **Break-even cost analysis:** Strategies are only viable at very low costs (< 7 bps BTC, < 17 bps AAPL). Need to investigate whether execution costs in practice can be kept below these thresholds, especially for crypto.

12. **Crypto data limitations persist:** Even with adaptive windowing, BTC/USDT yields only 4 hourly windows (720 total samples). Consider using longer history (daily BTC) or multiple crypto tickers to increase sample size.
