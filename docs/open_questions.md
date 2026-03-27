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

## Addressed in Cycle 7

- ~~**Per-model-ticker seq_len optimization (#1):** Implemented adaptive seq_len selection per model per ticker. Transformer consistently prefers 20-30 across equities; LSTM/GRU selections vary widely (10-50). Eliminates one-size-fits-all seq_len.~~
- ~~**Transformer convergence (#3):** Added 5-epoch linear warm-up (lr/10 to lr). AAPL Transformer improved from Cycle 6 (-1.13) to 0.56, suggesting warm-up helps equity convergence. Effect confounded by adaptive seq_len and different data period.~~
- ~~**Additional tickers for robustness (#6):** Added SPY and MSFT (daily, 5y). Both confirm models struggle vs buy-and-hold on trending equities. SPY best model (LSTM CLS, 0.41) well below baseline (0.98). MSFT similar pattern.~~
- ~~**Volatility-regime-based short toggling (#7):** Implemented regime detection using rolling vs expanding mean volatility. Improves equity results (MSFT Transformer: 0.11 → 0.84 with 47 shorts disabled). No effect on crypto (consistently high volatility).~~

## Addressed in Cycle 8

- ~~**Ensemble with adaptive seq_len (#1):** Fixed tail-alignment logic for prediction arrays of different lengths. Ensembles now work correctly with adaptive seq_len. BTC ensemble achieves Sharpe 7.10, best project result.~~
- ~~**Regime detection calibration (#5):** Implemented per-ticker threshold sweep across [1.0, 1.25, 1.5, 2.0, 2.5]. Lower thresholds (1.0-1.25) generally better for equities. SPY Transformer prefers 2.0. MSFT GRU improves from 0.67 to 1.14 with threshold=1.0.~~
- ~~**Model architecture search (#6):** Implemented per-model hidden size sweep [32, 64, 128]. Each model-ticker gets optimal architecture. No universal preference; smaller models (32) often selected, suggesting prior cycles were overfitting.~~

## Addressed in Cycle 9

- ~~**Selective ensemble (#5):** Implemented selective ensemble that drops models with Sharpe below threshold (default: 0.0). Most tickers have 0-1 positive models, preventing selective ensemble formation. MSFT is the only ticker where all 3 models are positive, producing selective ensemble Sharpe 1.05. The mechanism is sound but requires more model diversity.~~
- ~~**num_layers search (#6):** Implemented per-model num_layers sweep [1, 2, 3]. GRU consistently prefers 3 layers across all tickers. Transformer prefers 1-2 layers. LSTM is mixed (1-3). No universal depth preference; confirms value of per-model-ticker adaptation.~~
- ~~**Early stopping + warmup interaction (#7):** Fixed by skipping early stopping checks during warmup phase (epoch < warmup_epochs). Transformer now always completes warmup before early stopping can trigger.~~

## Addressed in Cycle 10

- ~~**Result instability across cycles (#1):** Multi-seed averaging (n_seeds=2) introduced to reduce initialization variance. Modest impact at n=2; higher seed counts (3-5) recommended for meaningful variance reduction.~~
- ~~**Cross-validated sweep efficiency (#2):** Joint hyperparameter search replaces sequential sweeps. 6 random samples from the (hidden_size x num_layers x seq_len) space instead of 30 sequential evaluations. Reduces compute by ~50% but may miss optima.~~
- ~~**Classification vs regression on SPY (#4):** Adaptive mode selection per model/ticker implemented. SPY correctly selects classification for all 3 models; AAPL shows heterogeneity (GRU=classification, LSTM/Transformer=regression). Validates per-model-ticker mode selection.~~

## Addressed in Cycle 11

- ~~**Joint search budget too small (#2):** Increased from 6 to 12 samples. Also expanded search space to include dropout [0.1, 0.2, 0.3], making 108 total combinations with 12 sampled (~11%).~~
- ~~**Multi-seed averaging needs higher n (#3):** Increased from 2 to 3 seeds, providing ~42% variance reduction (vs 30% at n=2).~~
- ~~**Ticker coverage reduced (#5):** Restored all 5 tickers: AAPL, SPY, MSFT, BTC/USDT, ETH/USDT.~~
- ~~**Ensemble approaches buy-and-hold (#6):** Addressed with confidence-weighted position sizing + position change threshold (0.1). When models lack conviction, positions shrink. The threshold prevents micro-trade cost accumulation. Ensemble now correctly reflects near-zero alpha rather than tracking benchmark.~~
- ~~**Confidence weighting increases trade costs (#4, previous):** Fixed with position change threshold in `_apply_min_holding_period()`. Changes < 10% of position size are suppressed, dramatically reducing trade count without losing meaningful signals.~~
- ~~**Dropout search not functional (bug):** The `dropout_sweep` config parameter was defined but never passed to `joint_hyperparam_search()`. Fixed by adding `dropout_levels` parameter extraction and passing in cli.py.~~

## Remaining

1. **Statistical significance gap:** After 11 cycles, no model achieves statistically significant outperformance vs buy-and-hold (p < 0.05). Both t-test and bootstrap methods confirm. Walk-forward window count (3-4) limits statistical power. The fundamental issue may be that technical indicators do not contain sufficient predictive signal for individual asset timing.

2. **Selective ensemble limited by model count:** With 3 model types and varying per-ticker performance, selective ensemble often has insufficient models above threshold. Adding more model architectures (e.g., TCN, attention-LSTM) could help.

3. **Sharpe loss + classification conflict:** Sharpe loss is designed for regression (continuous positions) but the system also tests classification mode. Currently Sharpe loss is only applied during final training (not during mode sweep), which is correct. However, the loss may still bias regression models to produce different signals than what the mode sweep evaluated.

4. **Ensemble confidence weighting neutralizes positions:** Averaging 3 models and then scaling by prediction magnitude results in very small positions. SPY and BTC ensembles match baseline exactly because the confidence threshold suppresses all changes. Consider using confidence weighting on individual models pre-ensemble rather than post-ensemble.

5. **Crypto results are unstable:** ETH/USDT LSTM achieves Sharpe 5.67 but with only 661 samples and 4 windows, this is likely overfitting. BTC/USDT LSTM achieves 2.17 but previous run had all-negative. Results are highly sensitive to random search sampling.

6. **Architecture preferences remain unstable:** MSFT best model switches from LSTM (previous run) to GRU (current run). SPY mode preferences change between runs. This instability suggests the search space has many near-equivalent configurations and results are sensitive to random sampling.
