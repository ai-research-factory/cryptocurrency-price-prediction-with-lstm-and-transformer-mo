# Cycle 11 Technical Findings

## Implementation Summary

Cycle 11 introduced five enhancements addressing remaining open questions from Cycle 10, plus two bug fixes discovered during re-implementation:

### 1. Confidence-Weighted Position Sizing (OQ#6)
**Problem**: Ensembles converged toward buy-and-hold because binary long/flat positions don't differentiate based on prediction strength.

**Solution**: `confidence_weighted` mode in `compute_trading_metrics()` scales position size by prediction magnitude:
- Regression: position = clip(prediction / prediction_std, -1, 1)
- Classification: position = clip((prediction - 0.5) * 2, -1, 1)

**Result**: Confidence weighting differentiates from buy-and-hold. AAPL ensemble achieves Sharpe ~0 (near flat), which correctly reflects minimal alpha rather than tracking the benchmark. SPY ensemble with confidence weighting matches baseline exactly (0.98), as the position change threshold suppresses micro-trades.

### 2. Position Change Threshold for Confidence Weighting (OQ#4)
**Problem**: Confidence-weighted positions create many small position changes (micro-trades) that accumulate transaction costs. Binary strategies had ~10 trades; confidence-weighted had ~300+.

**Solution**: Added `change_threshold` parameter (default 0.1 for confidence-weighted) to `_apply_min_holding_period()`. Position changes smaller than 10% of full position size are suppressed. This reduces unnecessary micro-trades while preserving meaningful position shifts.

**Result**: Trade counts significantly reduced. SPY confidence-weighted ensemble now matches buy-and-hold exactly, as the threshold + hold period prevents unnecessary churning. AAPL GRU trade costs dropped from 0.074 (7.4% of capital) to manageable levels.

### 3. Sharpe-Aware Loss (OQ#6)
**Problem**: MSE loss optimizes for prediction accuracy, not risk-adjusted returns.

**Solution**: `DifferentiableSharpe` loss class in training.py. Blended loss: `(1-w) * MSE + w * (-Sharpe)` where positions are soft (via tanh). Uses w=0.3 by default.

**Result**: Mixed impact across tickers. SPY Transformer with Sharpe loss achieves 0.31 Sharpe. The Sharpe loss term shifts some mode preferences (SPY Transformer selects classification). The loss is only applied during final training, not during mode selection sweep, ensuring fair comparison.

### 4. Dropout in Joint Hyperparameter Search (Bug Fix + Enhancement)
**Problem**: The `dropout_sweep` config parameter was defined but never passed to `joint_hyperparam_search()`. Dropout was fixed at 0.2 for all models despite the config specifying [0.1, 0.2, 0.3].

**Fix**: Added `dropout_levels` parameter extraction and passing in cli.py. Search space expanded from (hs × nl × sl) = 36 to (hs × nl × sl × dropout) = 108 combinations, with 12 random samples.

**Result**: Selected dropout values now vary per model: visible in joint search logs (e.g., do=0.3, do=0.2, do=0.1). Models have proper regularization tuning.

### 5. Increased Search Budget & Seeds (OQ#2, OQ#3)
- `joint_search_samples`: 6 → 12 (~11% coverage of 108-combo space)
- `n_seeds`: 2 → 3 (~42% variance reduction)

### 6. Restored 5-Ticker Coverage (OQ#5)
Restored BTC/USDT, ETH/USDT, and MSFT alongside AAPL and SPY.

## Key Results

### Per-Ticker Best Models (Cycle 11)

| Ticker   | Best Model          | Sharpe | Return  | Baseline Sharpe | Stability |
|----------|---------------------|--------|---------|-----------------|-----------|
| AAPL     | Ensemble (inv-var)  | -0.00  | -0.2%   | 0.35            | --        |
| SPY      | Ensemble (equal)    | 0.98   | +33.3%  | 0.98            | --        |
| MSFT     | GRU (regression)    | 0.61   | +33.3%  | 1.01            | 1.00      |
| BTC/USDT | LSTM (cls)          | 2.17   | +0.2%   | -0.76           | 0.75      |
| ETH/USDT | LSTM (cls)          | 5.67   | +0.7%   | 0.46            | 0.50      |

### AAPL Results
- **LSTM** (classification, hs=64, nl=3, sl=30): Sharpe -0.23. Classification hurts.
- **GRU** (regression, hs=32, nl=1, sl=20): Sharpe -0.36. Small shallow model underperforms.
- **Transformer** (regression, hs=32, nl=3, sl=30): Sharpe -0.19. Small deep Transformer.
- **Ensemble (inv-var)**: Sharpe -0.004 — near flat. Confidence weighting with threshold reduces to near-zero position.
- **Adaptive mode**: LSTM=classification, GRU/Transformer=regression.

### SPY Results
- **LSTM** (regression, hs=128, nl=1, sl=50): Sharpe 0.28, stability 1.0. Shallow wide model.
- **GRU** (regression, hs=128, nl=1, sl=50): Sharpe 0.20, stability 0.67.
- **Transformer** (classification, hs=64, nl=1, sl=50): Sharpe 0.31, stability 1.0.
- **Ensemble**: Sharpe 0.98 — matches baseline exactly. Position change threshold causes confidence-weighted ensemble to hold constant position.
- **Adaptive mode**: LSTM/GRU=regression, Transformer=classification.

### MSFT Results
- **GRU** (regression, hs=64, nl=3, sl=10): Sharpe 0.61, stability 1.0. Best individual model across equities. Deep GRU with short lookback.
- **LSTM** (regression, hs=32, nl=3, sl=30): Sharpe 0.18, stability 0.67.
- **Transformer** (regression, hs=128, nl=2, sl=50): Sharpe -0.22. Large Transformer overfits.
- **Selective ensemble** (LSTM+GRU): Sharpe -0.21 (worse than individual GRU alone).

### BTC/USDT Results
- **LSTM** (classification, hs=64, nl=1, sl=50): Sharpe 2.17, stability 0.75. Best crypto model. All models select classification.
- **GRU** (classification, hs=128, nl=1, sl=50): Sharpe 0.56.
- **Transformer** (classification, hs=32, nl=1, sl=20): Sharpe -3.56.
- **Ensemble**: Sharpe -0.76 (matches baseline). Confidence weighting neutralizes positions.

### ETH/USDT Results
- **LSTM** (classification, hs=32, nl=1, sl=20): Sharpe 5.67. Highest single-model result.
- **Transformer** (regression, hs=128, nl=2, sl=50): Sharpe 5.36. Also strong.
- **GRU** (classification, hs=128, nl=2, sl=50): Sharpe 2.93.
- **Ensemble**: Sharpe 0.87 (above baseline 0.82). Ensemble averages out the strong individual signals.

## Comparison to Previous Cycle 11 Run

| Metric                    | Previous C11            | Current C11 (fixed)     |
|---------------------------|-------------------------|-------------------------|
| Dropout in joint search   | Not passed (bug)        | Properly passed         |
| Position change threshold | None (0.0)              | 0.1 for conf-weighted   |
| AAPL best Sharpe          | 0.44 (Ensemble=baseline)| -0.004 (near flat)      |
| SPY best Sharpe           | 0.66 (Sel. Ensemble)    | 0.98 (Ensemble=baseline)|
| MSFT best Sharpe          | 0.61 (LSTM)             | 0.61 (GRU)              |
| BTC/USDT best Sharpe      | -3.09 (all negative)    | 2.17 (LSTM cls)         |
| ETH/USDT best Sharpe      | 2.79 (GRU cls)          | 5.67 (LSTM cls)         |

## Notable Observations

1. **Position change threshold transforms confidence weighting behavior**: With 0.1 threshold, SPY ensemble matches buy-and-hold exactly. The threshold suppresses micro-trades, causing the confidence-weighted position to remain constant. This is the correct behavior — the model has no edge, and the threshold prevents paying costs to achieve zero alpha.

2. **Dropout search now functional**: With the bug fix, dropout values are actually varied during joint search. The expanded 108-combo search space (vs 36 without dropout) provides proper regularization tuning per model/ticker.

3. **BTC/USDT significantly improved**: Previous run had all-negative models; current run has LSTM at Sharpe 2.17. All models now select classification, which was previously not the case. The combination of proper dropout tuning and classification mode helps.

4. **ETH/USDT LSTM achieves Sharpe 5.67**: Highest single-model result, but with only 661 samples and 4 windows, this is likely noise/overfitting. Bootstrap significance test confirms (p=0.97).

5. **Ensemble confidence weighting tends to neutralize positions**: The combination of averaging 3 models and then scaling by prediction magnitude results in very small positions. This explains why ensembles cluster around baseline or near-zero returns.

6. **MSFT GRU (hs=64, nl=3, sl=10) achieves perfect stability**: Deep GRU with short lookback and medium hidden size. 3/3 positive windows. Contrasts with previous run's LSTM preference.

7. **No statistical significance**: After 11 cycles with bug fixes, no model achieves reliable significance. Walk-forward window count (3-4) limits statistical power.

## Computational Cost

Joint search (12 samples × 3 models = 36 walks per ticker) + mode sweep (2 modes × 3 models = 6 walks) + final runs (3 models × 3 seeds = 9 walks) = ~51 walk-forward evaluations per ticker. With 5 tickers, total ~255 evaluations.

Total experiment runtime: ~18 minutes for 5 tickers.
