# Open Questions

## Addressed in Cycle 2

- ~~**Stationarity of features (#10):** MACD and ATR normalized by dividing by close price. Contributed to improved stability.~~
- ~~**Multiple tickers (#8):** Multi-ticker pipeline implemented. BTC/USDT (hourly) and AAPL (daily) evaluated.~~

## Addressed in Cycle 3

- ~~**Walk-forward window sizing (#5):** Adaptive window sizing implemented. BTC/USDT now gets 4 windows (vs 2). Guarantees minimum 3 windows per ticker.~~
- ~~**Transaction cost sensitivity (#8):** Sweep across [0, 2, 5, 10, 20, 50] bps implemented. Both tickers show break-even around 7 bps (BTC) and 17 bps (AAPL).~~
- ~~**Annualized volatility calculation (#11):** Fixed — hourly data now uses sqrt(6048), daily uses sqrt(252). Per-interval annualization factors in evaluation.py.~~

## Addressed in Cycle 4

- ~~**Feature selection (#1):** Permutation feature importance implemented. All 12 indicators contribute; RSI and MACD family are most important. No clear candidate for removal.~~
- ~~**Learning rate scheduling (#4):** ReduceLROnPlateau added (factor=0.5, patience=5). Active during training — LR reduced from 1e-3 to 5e-4 in some windows.~~
- ~~**Ensemble methods (#6):** LSTM+Transformer ensemble (prediction averaging) implemented. Provides middle-ground performance between individual models.~~
- ~~**Trade frequency control (#10):** Minimum holding period (default=3) added. Reduced Transformer trades by 60-67%. AAPL Transformer improved from Sharpe -0.09 to +0.21.~~
- ~~**Crypto data limitations (#12):** ETH/USDT added as third ticker. Confirms models struggle with crypto — ETH results even weaker than BTC.~~
- ~~**No statistical significance (#9):** Added ETH/USDT for more data points. Still no significance achieved (p > 0.05 everywhere), consistent with EMH expectations.~~

## Remaining

1. **Sequence length sensitivity (#2):** Currently fixed at 30. How sensitive are the models to this parameter? Should it differ between LSTM and Transformer? Could be explored in a future hyperparameter sweep.

2. **Classification vs regression (#3):** Direction prediction (up/down) with cross-entropy loss might produce stronger trading signals than return regression. The current MSE loss doesn't penalize direction errors specifically.

3. **Short positions (#5):** Current strategy is long-or-flat. Allowing short positions could improve Sharpe on bearish periods, especially for crypto.

4. **Indicator periods for different frequencies (#7):** RSI(14), MACD(12,26,9), etc. were designed for daily data. For hourly data, these periods may need adjustment (e.g., RSI(24) for daily cycle in hourly data).

5. **Transformer convergence (#8):** Transformer still shows higher initial loss than LSTM. Could benefit from pre-training or different architecture (e.g., temporal convolutional network).

## New from Cycle 4

6. **Feature pruning experiment:** While permutation importance shows all 12 features contribute, a formal ablation study removing the bottom 3-4 features might improve generalization by reducing overfitting.

7. **Minimum holding period tuning:** Current min_hold=3 is a default. A sweep across [1, 2, 3, 5, 10] could identify the optimal balance between trade reduction and signal responsiveness per ticker.

8. **Ensemble weighting:** Current ensemble uses equal averaging. Inverse-variance weighting or a stacking approach (train a meta-model on individual predictions) could improve ensemble quality.

9. **ETH model failure mode:** LSTM on ETH/USDT shows extreme negative Sharpe (-6.51). Investigation needed — possible overfitting to training regime that inverts in test, or ETH's different momentum characteristics require different architecture/features.

10. **Cross-asset signals:** Using BTC returns/volatility as features for ETH prediction (and vice versa) could capture cross-asset momentum and improve crypto predictions.
