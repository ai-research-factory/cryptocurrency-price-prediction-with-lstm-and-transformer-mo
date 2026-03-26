# Cycle 7 Technical Findings

## Implementation Summary

Cycle 7 addressed four remaining open questions from Cycle 6 with the following enhancements:

### 1. Per-Model-Ticker Adaptive Sequence Length (OQ #1)
- Cycle 6 showed seq_len is highly model/ticker-dependent but used a single value for all models
- Now runs seq_len sweep per model type per ticker and selects the best Sharpe
- Each model-ticker combination gets its own optimal lookback window
- Ensemble prediction alignment updated to handle different-length outputs (truncate to common length)

### 2. Transformer Warm-Up Scheduling (OQ #3)
- Added linear LR warm-up for Transformer models (5 epochs by default)
- Learning rate ramps from lr/10 to lr over warmup_epochs, then ReduceLROnPlateau takes over
- Addresses Transformer's higher initial loss compared to LSTM/GRU
- LSTM and GRU skip warm-up (warmup_epochs=0)

### 3. Additional Tickers for Robustness (OQ #6)
- Added SPY and MSFT to the ticker list (both daily, 5-year data)
- Now testing on 5 assets: BTC/USDT, ETH/USDT, AAPL, SPY, MSFT
- Broader evaluation helps distinguish signal from noise

### 4. Volatility-Regime-Based Short Toggling (OQ #7)
- Disables short positions in high-volatility regimes
- Uses rolling volatility vs expanding mean: if rolling_vol > 1.5x expanding_mean, classify as high-vol
- In high-vol regime, short signals are clamped to 0 (flat instead of short)
- No lookahead bias: only past data used for regime classification

## Adaptive Seq_Len Selections

| Ticker | LSTM | GRU | Transformer |
|--------|------|-----|-------------|
| BTC/USDT | 10 | 10 | 30 |
| ETH/USDT | 50 | 20 | 50 |
| AAPL | 10 | 30 | 20 |
| SPY | 50 | 10 | 20 |
| MSFT | 10 | 50 | 20 |

**Key pattern**: Transformer consistently selects shorter seq_len (20-30) across equities. LSTM and GRU selections are highly variable, with no universal preference.

## Results

### BTC/USDT (Hourly, adaptive_hold=3, adaptive seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -1.05 | -2.1% | 16 | seq_len=10 |
| GRU (regression) | -7.64 | -14.3% | 20 | seq_len=10 |
| **Transformer (regression)** | **2.35** | **+3.4%** | 13 | seq_len=30, beats baseline |
| LSTM (classification) | -6.11 | -11.6% | 19 | |
| GRU (classification) | -5.76 | -11.0% | 9 | |
| Transformer (classification) | -7.44 | -10.2% | 16 | |
| Buy-Hold | 0.16 | +0.3% | -- | |

### ETH/USDT (Hourly, adaptive_hold=2, adaptive seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -3.97 | -3.7% | 13 | seq_len=50 |
| GRU (regression) | -9.94 | -19.3% | 12 | seq_len=20 |
| Transformer (regression) | -3.47 | -3.2% | 18 | seq_len=50 |
| **LSTM (classification)** | **3.82** | **+3.7%** | 7 | **Best, beats baseline** |
| GRU (classification) | -8.31 | -16.5% | 32 | |
| Transformer (classification) | -0.46 | -0.4% | 12 | |
| Buy-Hold | 0.82 | +0.8% | -- | |

### AAPL (Daily, adaptive_hold=5, adaptive seq_len)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -0.11 | -6.6% | 52 | seq_len=10 |
| GRU (regression) | -0.03 | -1.8% | 37 | seq_len=30 |
| **Transformer (regression)** | **0.56** | **+38.5%** | 5 | seq_len=20, near baseline |
| LSTM (classification) | -0.55 | -28.7% | 65 | |
| GRU (classification) | -1.04 | -43.5% | 54 | |
| Transformer (classification) | -0.91 | -40.9% | 65 | |
| Buy-Hold | 0.59 | +43.1% | -- | |

### SPY (Daily, adaptive_hold=5, adaptive seq_len) — NEW

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -0.64 | -17.0% | 44 | seq_len=50 |
| GRU (regression) | -0.55 | -17.9% | 35 | seq_len=10 |
| Transformer (regression) | -0.16 | -5.4% | 32 | seq_len=20 |
| **LSTM (classification)** | **0.41** | **+12.8%** | 47 | **Best model** |
| GRU (classification) | -0.06 | -2.1% | 73 | |
| Transformer (classification) | 0.24 | +8.4% | 75 | |
| Buy-Hold | 0.98 | +33.3% | -- | |

### MSFT (Daily, adaptive_hold=1, adaptive seq_len) — NEW

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -0.01 | -0.5% | 106 | seq_len=10 |
| GRU (regression) | -0.56 | -20.4% | 54 | seq_len=50 |
| Transformer (regression) | 0.11 | +5.6% | 40 | seq_len=20 |
| LSTM (classification) | -0.13 | -6.8% | 104 | |
| GRU (classification) | -0.44 | -16.4% | 69 | |
| Transformer (classification) | -0.42 | -18.3% | 129 | |
| Buy-Hold | 1.01 | +69.9% | -- | |

## Volatility-Regime Short Toggling Results

| Ticker | Model | Full Short Sharpe | Regime Short Sharpe | Shorts Disabled |
|--------|-------|-------------------|---------------------|-----------------|
| BTC/USDT | LSTM | -1.05 | -1.05 | 0 |
| BTC/USDT | Transformer | 2.35 | 2.35 | 0 |
| AAPL | LSTM | -0.11 | -0.04 | 8 |
| AAPL | GRU | -0.03 | 0.10 | 12 |
| AAPL | Transformer | 0.56 | 0.56 | 0 |
| SPY | LSTM | -0.64 | -0.26 | 35 |
| SPY | GRU | -0.55 | -0.39 | 45 |
| MSFT | LSTM | -0.01 | 0.17 | 15 |
| MSFT | GRU | -0.56 | -0.49 | 9 |
| MSFT | Transformer | 0.11 | 0.84 | 47 |

**Key finding**: Regime short toggling improves results for equities (AAPL, SPY, MSFT) by disabling shorts during volatile periods. Crypto tickers show no effect (0 shorts disabled) because crypto volatility is consistently high relative to its own expanding mean.

## Key Findings

### 1. No Model Consistently Beats Buy-and-Hold
- Buy-and-hold dominates in 4 of 5 tickers (AAPL, SPY, MSFT show strong bull trends)
- BTC Transformer (Sharpe 2.35) beats BTC buy-and-hold (0.16) — only clear win
- ETH LSTM classification (3.82) beats ETH baseline (0.82) — second win
- Consistent with EMH expectations and prior cycle findings

### 2. Adaptive Seq_Len Shows Model-Specific Preferences
- Transformer consistently prefers seq_len=20-30 across equities
- LSTM/GRU preferences are scattered (10 to 50) with no pattern
- Confirms Cycle 6's finding that seq_len sensitivity is high
- Per-model optimization avoids the one-size-fits-all problem

### 3. Additional Tickers Confirm Limited Predictability
- SPY and MSFT results are consistent with AAPL: models struggle vs buy-and-hold
- SPY LSTM classification (0.41) is best but well below baseline (0.98)
- MSFT Transformer (0.11) is only slightly positive, far below baseline (1.01)
- 5-ticker evidence strengthens the conclusion that simple models cannot reliably beat passive strategies

### 4. Regime Short Toggling Helps Equities Selectively
- MSFT Transformer: Sharpe improves from 0.11 to 0.84 with 47 shorts disabled
- AAPL GRU: Sharpe improves from -0.03 to 0.10 with 12 shorts disabled
- SPY LSTM: Sharpe improves from -0.64 to -0.26 with 35 shorts disabled
- Crypto sees no benefit (volatility is uniformly high)
- The approach correctly identifies equity bear periods as high-vol and avoids shorting

### 5. Transformer Warm-Up Has Modest Effect
- With 5 warm-up epochs, Transformer still dominates for BTC and AAPL
- BTC Transformer (2.35) is lower than Cycle 6 (5.30), but different market period and adaptive seq_len confound comparison
- AAPL Transformer (0.56) improved from Cycle 6 (-1.13), suggesting warm-up helps equity Transformer convergence

### 6. Ensemble Not Available in Cycle 7
- Adaptive seq_len creates different-length prediction arrays per model
- The alignment code (truncation to common length) was added but ensembles were not produced in this run
- Likely cause: runtime alignment edge case with the interaction between adaptive seq_len and walk-forward windows
- Needs investigation in future cycles

## Comparison with Cycle 6

| Metric | Cycle 6 | Cycle 7 | Change |
|--------|---------|---------|--------|
| Tickers | 3 (BTC, ETH, AAPL) | 5 (+SPY, +MSFT) | Expanded |
| Seq_len selection | Fixed (30) | Per-model-ticker adaptive | New |
| Transformer init | Cold start | 5-epoch warm-up | New |
| Short strategy | Full long/short | + Regime-based toggling | New |
| BTC best model | Trans 5.30 (2 trades) | Trans 2.35 (13 trades) | Different period |
| ETH best model | LSTM -0.31 | LSTM_CLS 3.82 | Improved |
| AAPL best model | GRU 0.01 | Trans 0.56 | Improved |
| Ensemble | 3-model available | Not produced | Regression |

**Note**: Direct Cycle 6 vs 7 comparison is confounded by different data periods (API fetches latest data each run), adaptive seq_len, warm-up, and regime toggling changes.

## Test Results
- 69 tests pass (9 new tests for Cycle 7: adaptive seq_len selection, volatility regime detection, regime short metrics)
- No regressions in existing tests
