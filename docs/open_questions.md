# Open Questions

1. **Feature selection:** Are all 12 indicators contributing, or would a reduced set improve generalization? Feature importance analysis needed.

2. **Sequence length sensitivity:** Currently fixed at 30. How sensitive are the models to this parameter? Should it differ between LSTM and Transformer?

3. **Classification vs regression:** The paper may frame this as a direction prediction (up/down) classification task rather than return regression. Classification with cross-entropy loss might produce stronger trading signals.

4. **Learning rate scheduling:** No LR schedule is used. Cosine annealing or ReduceLROnPlateau may improve convergence.

5. **Walk-forward window sizing:** train_size=500 may be too large for hourly data or too small for daily. Adaptive window sizing could help.

6. **Short positions:** Current strategy is long-or-flat. Allowing short positions could improve Sharpe on bearish periods.

7. **Ensemble methods:** Combining LSTM and Transformer predictions (e.g., averaging or stacking) may improve stability.

8. **Multiple tickers:** The paper focuses on cryptocurrency. Should the model be trained on a basket of crypto assets or evaluated on a broader universe?

9. **Transaction cost sensitivity:** 10 bps is assumed. Real crypto exchange fees vary (Binance ~2-7 bps for makers). How sensitive are results to this parameter?

10. **Stationarity of features:** Some indicators (MACD, ATR) are in price-space and non-stationary. Normalizing them relative to price may help.
