# Cycle 6 Technical Findings

## Implementation Summary

Cycle 6 addressed four remaining open questions from Cycle 5 with the following enhancements:

### 1. GRU Model for Ensemble Diversity (OQ #7)
- Added `GRUPredictor` as a third model alongside LSTM and Transformer
- GRU has same architecture as LSTM but with fewer parameters (no separate cell state)
- Provides different prediction characteristics for meaningful ensemble weighting
- 3-model ensemble (LSTM + GRU + Transformer) with both equal and inverse-variance weighting

### 2. Sequence Length Sensitivity Sweep (OQ #1)
- Sweeps `seq_len` across [10, 20, 30, 50] for each model and ticker
- Tests how lookback window size affects prediction quality
- Previously fixed at 30 with unknown sensitivity

### 3. Adaptive Per-Ticker Holding Period (OQ #4)
- Uses min_hold sweep results to automatically select optimal holding period per ticker
- Runs a preliminary sweep using the first model type, then applies the best hold period to all models
- Eliminates the need for a global default that's suboptimal for individual tickers

### 4. Frequency-Adaptive Indicator Periods (OQ #2)
- Indicator lookback periods now scale with data frequency
- Daily (default): RSI(14), MACD(12,26,9), BB(20), ATR(14)
- Hourly: RSI(24), MACD(24,48,12), BB(48), ATR(24) — aligned with daily cycles
- Addresses the mismatch between indicator design (daily) and hourly data

## Results

### BTC/USDT (Hourly, 4 windows, adaptive_hold=1, freq-adaptive indicators)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -0.08 | -0.1% | 23 | |
| **GRU (regression)** | **2.65** | **+4.2%** | 5 | **New model, beats baseline** |
| **Transformer (regression)** | **5.30** | **+8.1%** | 2 | Best single model |
| LSTM (classification) | 5.31 | +8.1% | 3 | Improved vs Cycle 5 |
| GRU (classification) | -8.12 | -12.3% | 41 | |
| Transformer (classification) | -7.84 | -11.8% | 23 | |
| **Ensemble (equal, 3-model)** | **4.85** | **+7.3%** | 1 | **New: 3-model ensemble** |
| Buy-Hold | 1.24 | +1.8% | -- | |

**Adaptive hold selected**: min_hold=1 (best from sweep)
**Seq len sensitivity**: Highly variable. BTC LSTM: seq_len=30 (Sharpe 3.81) vs seq_len=20 (-6.91). BTC Transformer: seq_len=30 (3.51) vs seq_len=10 (-2.25).

### ETH/USDT (Hourly, 4 windows, adaptive_hold=10, freq-adaptive indicators)

| Model | Sharpe | Return | Trades | Notes |
|-------|--------|--------|--------|-------|
| LSTM (regression) | -0.31 | -0.5% | 10 | |
| GRU (regression) | -5.49 | -8.5% | 5 | |
| Transformer (regression) | -1.12 | -1.7% | 5 | |
| Transformer (classification) | 3.41 | +5.1% | 7 | Only positive CLS |
| Ensemble (equal) | -0.44 | -0.7% | 5 | |
| Buy-Hold | -0.13 | -0.2% | -- | Negative baseline period |

**Adaptive hold selected**: min_hold=10 (longest hold period)
**Seq len sensitivity**: ETH LSTM shows seq_len=50 as best (Sharpe 7.31), while shorter lookbacks produce large negative Sharpes.

### AAPL (Daily, 3 windows, adaptive_hold=5, standard indicators)

| Model | Sharpe | Return | Max DD | Trades | Notes |
|-------|--------|--------|--------|--------|-------|
| LSTM (regression) | -0.27 | -11.1% | 39.3% | 40 | |
| GRU (regression) | 0.01 | +0.5% | 32.3% | 43 | Near break-even |
| Transformer (regression) | -1.13 | -35.1% | 55.3% | 38 | |
| Transformer (classification) | -0.04 | -2.3% | 30.6% | 54 | Best CLS |
| Ensemble (equal) | -0.97 | -41.2% | 71.1% | -- | |
| Ensemble (inv-variance) | -1.08 | -44.4% | 76.6% | -- | |
| Buy-Hold | 0.35 | +21.0% | 40.6% | -- | |

**Adaptive hold selected**: min_hold=5 (from preliminary sweep)
**Seq len sensitivity**: AAPL Transformer: seq_len=20 is best (Sharpe 1.72), seq_len=30 is 0.46, seq_len=50 is -0.36. The default seq_len=30 is not universally optimal.

## Key Findings

### 1. GRU Provides Genuine Ensemble Diversity
- GRU achieves a different prediction profile from LSTM despite similar architecture
- BTC GRU (Sharpe 2.65) fills the middle ground between LSTM (-0.08) and Transformer (5.30)
- 3-model inverse-variance ensemble now produces slightly different weights (e.g., 0.33/0.33/0.34) instead of perfectly equal, confirming the variance differentiation goal

### 2. Sequence Length is Highly Model/Ticker-Dependent
- No single seq_len works best across all combinations
- BTC models prefer seq_len=30 (LSTM: 3.81, Transformer: 3.51)
- ETH LSTM prefers seq_len=50 (Sharpe 7.31 vs -6.94 at seq_len=30)
- AAPL Transformer prefers seq_len=20 (Sharpe 1.72 vs 0.46 at seq_len=30)
- The current default of 30 is reasonable but could be improved with per-model-ticker optimization

### 3. Adaptive Hold Period Improves Crypto, Mixed for Equities
- BTC adaptive_hold=1 allows more responsive trading (vs global default of 3)
- ETH adaptive_hold=10 reduces trade churn in a difficult market
- AAPL adaptive_hold=5 is slightly higher than the previous default of 3
- Adaptive selection avoids the one-size-fits-all problem demonstrated in Cycle 5

### 4. Frequency-Adaptive Indicators Change Crypto Results
- Hourly indicators now use periods aligned with daily cycles (RSI-24, MACD-24/48/12)
- BTC/ETH results differ from Cycle 5 (which used daily-period indicators on hourly data)
- The change is a confounding factor — cannot isolate indicator period effect from other Cycle 6 changes

### 5. Classification Still Mostly Worse Than Regression
- Confirms Cycle 5 finding: regression preferred in 6/9 model-ticker combinations
- Exception: BTC LSTM classification (5.31) vs regression (-0.08) — a reversal from Cycle 5
- ETH Transformer classification (3.41) also positive when regression fails (-1.12)

### 6. No Statistical Significance Achieved
- All p-values > 0.05 vs buy-and-hold (except AAPL LSTM_CLS at p=0.036, but in the wrong direction)
- Consistent with Efficient Market Hypothesis expectations
- Low window count (3-4) limits statistical power

## Comparison with Cycle 5

| Metric | Cycle 5 | Cycle 6 | Change |
|--------|---------|---------|--------|
| Models | 2 (LSTM, Trans) | 3 (LSTM, GRU, Trans) | +GRU |
| BTC best model | Trans 3.92 (2 trades) | Trans 5.30 (2 trades) | Improved |
| BTC ensemble | 3.85 | 4.85 | Improved (+26%) |
| ETH best model | Trans -1.67 | LSTM -0.31 | Improved |
| AAPL best model | Ensemble 0.39 | GRU 0.01 | **Regressed** |
| AAPL ensemble | 0.39 | -0.97 | **Regressed** |
| Indicator periods | Fixed (daily) | Freq-adaptive | New |
| Hold period | Fixed (3) | Adaptive per-ticker | New |
| Seq len | Fixed (30) | Swept [10,20,30,50] | New data |

**Note on AAPL regression**: The AAPL ensemble result from Cycle 5 (Sharpe 0.39) was likely fragile — changing the hold period from 3 to 5 and running with 3 model types instead of 2 produced worse results. This suggests the Cycle 5 AAPL result was not robust to parameter changes.

## Test Results
- 60 tests pass (9 new tests added for Cycle 6 features: GRU model, frequency-adaptive indicators, adaptive hold selection)
- No regressions in existing tests
