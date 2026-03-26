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

## Addressed in Cycle 6

- ~~**Sequence length sensitivity (#2):** Swept seq_len=[10,20,30,50] per model per ticker. Highly sensitive: BTC models prefer 30, ETH LSTM prefers 50 (Sharpe 7.31 vs -6.94), AAPL Transformer prefers 20 (Sharpe 1.72 vs 0.46). No universal optimum exists.~~
- ~~**Indicator periods for different frequencies (#7):** Implemented frequency-adaptive periods. Hourly uses RSI(24), MACD(24,48,12), BB(48), ATR(24) aligned with daily cycles. Effect confounded by other Cycle 6 changes but represents correct methodology.~~
- ~~**Ensemble model diversity (#7 from Cycle 5):** Added GRU as third model type. 3-model ensemble achieves differentiated inverse-variance weights. BTC ensemble improved from 3.85 to 4.85 Sharpe.~~
- ~~**Per-ticker holding period optimization (#4 from Cycle 5):** Adaptive per-ticker hold period selection implemented. BTC=1, ETH=10, AAPL=5 selected automatically from sweep results.~~

## Remaining

1. **Per-model-ticker seq_len optimization:** Cycle 6 sweep shows large sensitivity to seq_len, but currently all models use the same value. Using per-model-ticker optimal seq_len could significantly improve results (e.g., ETH LSTM at seq_len=50 vs 30).

2. **AAPL Cycle 5 result fragility:** The Cycle 5 AAPL ensemble (Sharpe 0.39) did not replicate in Cycle 6 (-0.97). The result was sensitive to hold period changes (3→5) and additional model (GRU). Needs investigation into what made the Cycle 5 configuration work.

3. **Transformer convergence (#8 from prior):** Transformer still shows higher initial loss than LSTM/GRU. Could benefit from pre-training, warm-up scheduling, or different initialization.

4. **BTC low-trade regime persistence:** BTC Transformer continues to achieve high Sharpe (5.30) with very few trades (2). This pattern persists across cycles, suggesting it's picking a single profitable regime rather than overfitting. But low trade count makes statistical evaluation impossible.

5. **Cross-validated seq_len selection:** Current seq_len sweep uses full walk-forward per level, which is expensive. A lighter-weight inner validation loop for seq_len selection could be more practical.

6. **Additional tickers for robustness:** Only 3 tickers tested. Adding more equities (SPY, MSFT) and crypto (SOL, XRP) would help distinguish signal from noise in the results.

7. **Volatility-regime-based short toggling:** Short positions still hurt crypto. A regime detector that only enables shorts in low-volatility trending markets could reduce losses.
