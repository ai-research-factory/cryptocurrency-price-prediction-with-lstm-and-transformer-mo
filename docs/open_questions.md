# Open Questions

## Addressed in Cycle 2

- ~~**Stationarity of features (#10):** MACD and ATR normalized by dividing by close price. Contributed to improved stability.~~
- ~~**Multiple tickers (#8):** Multi-ticker pipeline implemented. BTC/USDT (hourly) and AAPL (daily) evaluated.~~

## Addressed in Cycle 3

- ~~**Walk-forward window sizing (#5):** Adaptive window sizing implemented. BTC/USDT now gets 4 windows (vs 2). Guarantees minimum 3 windows per ticker.~~
- ~~**Transaction cost sensitivity (#8):** Sweep across [0, 2, 5, 10, 20, 50] bps implemented. Both tickers show break-even around 7 bps (BTC) and 17 bps (AAPL).~~
- ~~**Annualized volatility calculation (#11):** Fixed -- hourly data now uses sqrt(6048), daily uses sqrt(252). Per-interval annualization factors in evaluation.py.~~

## Addressed in Cycle 4

- ~~**Feature selection (#1):** Permutation feature importance implemented. All 12 indicators contribute; RSI and MACD family are most important. No clear candidate for removal.~~
- ~~**Learning rate scheduling (#4):** ReduceLROnPlateau added (factor=0.5, patience=5). Active during training -- LR reduced from 1e-3 to 5e-4 in some windows.~~
- ~~**Ensemble methods (#6):** LSTM+Transformer ensemble (prediction averaging) implemented. Provides middle-ground performance between individual models.~~
- ~~**Trade frequency control (#10):** Minimum holding period (default=3) added. Reduced Transformer trades by 60-67%. AAPL Transformer improved from Sharpe -0.09 to +0.21.~~
- ~~**Crypto data limitations (#12):** ETH/USDT added as third ticker. Confirms models struggle with crypto -- ETH results even weaker than BTC.~~
- ~~**No statistical significance (#9):** Added ETH/USDT for more data points. Still no significance achieved (p > 0.05 everywhere), consistent with EMH expectations.~~

## Addressed in Cycle 5

- ~~**Classification vs regression (#3):** BCE classification mode implemented and compared. Regression is consistently better -- classification discards magnitude information and produces noisier signals. BTC Transformer: 3.92 (reg) vs -5.81 (cls). Regression is the preferred approach.~~
- ~~**Short positions (#5):** Long/short strategy implemented. Improves AAPL ensemble (Sharpe 0.39 vs baseline 0.35, first time exceeding buy-and-hold). Amplifies losses on ETH. Useful for equities, harmful for crypto.~~
- ~~**Minimum holding period tuning (#7 new):** Swept [1,2,3,5,10]. Optimal varies by ticker: BTC LSTM optimal at hold=5 (Sharpe 1.71 vs -0.40 at hold=3), AAPL Transformer optimal at hold=3. Global default of 3 is reasonable but suboptimal per-ticker.~~
- ~~**Ensemble weighting (#8 new):** Inverse-variance weighting implemented. Produced same weights as equal (0.50/0.50) because LSTM and Transformer have similar return variance. Need more model diversity for this to differentiate.~~

## Remaining

1. **Sequence length sensitivity (#2):** Currently fixed at 30. How sensitive are the models to this parameter? Should it differ between LSTM and Transformer?

2. **Indicator periods for different frequencies (#7):** RSI(14), MACD(12,26,9), etc. were designed for daily data. For hourly data, these periods may need adjustment (e.g., RSI(24) for daily cycle in hourly data).

3. **Transformer convergence (#8):** Transformer still shows higher initial loss than LSTM. Could benefit from pre-training or different architecture (e.g., temporal convolutional network).

## New from Cycle 5

4. **Per-ticker holding period optimization:** The sweep showed each ticker/model has a different optimal min_hold. An adaptive per-ticker parameter (or even per-window) could improve results, especially for BTC LSTM (hold=5: Sharpe 1.71 vs hold=3: -0.40).

5. **BTC Transformer low-trade regime:** BTC Transformer achieved Sharpe 3.92 with only 2 trades. This needs investigation -- is the model confidently picking a single regime, or is it failing to produce meaningful predictions most of the time?

6. **Long/short asymmetry:** Short positions helped AAPL but hurt crypto. A per-asset or volatility-regime-based toggle (short only in low-vol or trending markets) could be more effective.

7. **Ensemble model diversity:** Inverse-variance weighting produced equal weights due to similar LSTM/Transformer variance. Adding a third model type (e.g., GRU, temporal CNN, or a simple momentum baseline) could provide genuine diversification.

8. **AAPL ensemble beats buy-and-hold:** The AAPL ensemble (Sharpe 0.39 vs baseline 0.35, drawdown 26.9% vs 40.6%) is the first positive result. Needs more walk-forward windows and out-of-sample testing to confirm this isn't data snooping.
