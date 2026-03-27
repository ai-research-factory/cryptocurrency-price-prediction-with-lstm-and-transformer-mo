# Cycle 11 Technical Findings

## Implementation Summary

Cycle 11 introduced four enhancements addressing the remaining open questions from Cycle 10:

### 1. Confidence-Weighted Position Sizing (OQ#6)
**Problem**: Ensembles converged toward buy-and-hold because binary long/flat positions don't differentiate based on prediction strength.

**Solution**: `confidence_weighted` mode in `compute_trading_metrics()` scales position size by prediction magnitude:
- Regression: position = clip(prediction / prediction_std, -1, 1)
- Classification: position = clip((prediction - 0.5) * 2, -1, 1)

**Result**: Confidence weighting did differentiate from buy-and-hold but generally reduced performance. The variable position sizing creates more frequent trades and costs. AAPL ensemble Sharpe exactly matches baseline (0.44) — models predict the right direction but not enough to overcome costs with partial positions.

### 2. Sharpe-Aware Loss (OQ#6)
**Problem**: MSE loss optimizes for prediction accuracy, not risk-adjusted returns. Models learn to predict returns close to zero (mean-reverting signal) rather than generating tradeable alpha.

**Solution**: `DifferentiableSharpe` loss class in training.py. Blended loss: `(1-w) * MSE + w * (-Sharpe)` where positions are soft (via tanh). Uses w=0.3 by default.

**Result**: Mixed impact. SPY Transformer with Sharpe loss achieves 0.46 Sharpe (perfect stability), but SPY adaptive mode now selects regression over classification for all models (Cycle 10 selected classification). The Sharpe loss term may be conflicting with the classification objective.

### 3. Dropout in Joint Hyperparameter Search (new)
**Problem**: Dropout was fixed at 0.2 across all models. Different architectures and datasets may benefit from different regularization.

**Solution**: Added `dropout_levels` parameter to `joint_hyperparam_search()`. Search space expanded from (hidden_size x num_layers x seq_len) = 36 combinations to (hs x nl x sl x dropout) = 108 combinations, with 12 random samples.

**Result**: The expanded search space with 12 samples covers ~11% of configurations. Selected dropout values vary: MSFT LSTM selects 32/3/10 (small/deep/short), while SPY Transformer selects 128/2/30 (large/medium/medium). Dropout selection is implicit in the joint params — the best config's dropout is used.

### 4. Increased Search Budget & Seeds (OQ#2, OQ#3)
**Problem**: 6 joint search samples explored only 17% of the 36-combo space. n_seeds=2 provided only ~30% variance reduction.

**Solution**:
- `joint_search_samples`: 6 → 12 (33% coverage of 36-combo base space)
- `n_seeds`: 2 → 3 (~42% variance reduction vs 30%)

### 5. Restored 5-Ticker Coverage (OQ#5)
Restored BTC/USDT, ETH/USDT, and MSFT alongside AAPL and SPY.

## Key Results

### Per-Ticker Best Models (Cycle 11)

| Ticker   | Best Model          | Sharpe | Return  | Baseline Sharpe | Stability |
|----------|---------------------|--------|---------|-----------------|-----------|
| AAPL     | Ensemble (equal)    | 0.44   | +24.2%  | 0.44            | --        |
| SPY      | Sel. Ensemble (eq)  | 0.66   | +19.1%  | 1.12            | --        |
| MSFT     | LSTM (regression)   | 0.61   | +23.2%  | 1.01            | 1.00      |
| BTC/USDT | LSTM (regression)   | -3.09  | -1.3%   | 0.17            | 0.50      |
| ETH/USDT | GRU (classification)| 2.79   | +1.7%   | 0.82            | 0.50      |

### AAPL Results
- **LSTM** (regression, hs=128, nl=3, sl=50): Sharpe -0.63. Deep LSTM overfits on AAPL.
- **GRU** (classification, hs=128, nl=1, sl=30): Sharpe -0.21. Shallow classifier fails.
- **Transformer** (classification, hs=64, nl=1, sl=10): Sharpe -0.21. Small Transformer underperforms.
- **Ensemble**: Sharpe 0.44 — exactly matches buy-and-hold baseline. Confidence weighting causes models to take partial positions that average out to buy-and-hold.
- **Adaptive mode**: LSTM=regression, GRU/Transformer=classification (mixed, same as Cycle 10 but GRU/Transformer swapped).

### SPY Results
- **Transformer** (regression, hs=128, nl=2, sl=30): Sharpe 0.46, stability 1.0. Best individual model.
- **GRU** (regression, hs=64, nl=3, sl=20): Sharpe 0.25, stability 0.67.
- **LSTM** (regression, hs=128, nl=3, sl=30): Sharpe -0.26, stability 0.0.
- **Selective ensemble** (GRU + Transformer): Sharpe 0.66 — best SPY result.
- **Adaptive mode**: All regression (contrast with Cycle 10 all-classification). Sharpe loss may have shifted the preference.

### MSFT Results
- **LSTM** (regression, hs=32, nl=3, sl=10): Sharpe 0.61, stability 1.0. Small architecture with short lookback works best. Perfect stability across windows.
- **Transformer** (regression, hs=128, nl=2, sl=30): Sharpe 0.06. Marginal.
- **GRU** (classification, hs=128, nl=1, sl=30): Sharpe -0.45. Classification hurts.
- **Ensemble**: Sharpe 0.63 (full), but selective ensemble -0.08 (drops GRU but LSTM+Transformer worse together).

### BTC/USDT Results
- All models negative. LSTM best at -3.09. Confidence weighting amplifies losses on crypto.
- Hourly crypto remains very challenging. Only 661 samples with high noise.

### ETH/USDT Results
- **GRU** (classification, hs=128, nl=3, sl=50): Sharpe 2.79. Only positive model.
- All others strongly negative. GRU classification finds a pattern others miss.

## Comparison to Cycle 10

| Metric                    | Cycle 10               | Cycle 11               |
|---------------------------|------------------------|------------------------|
| Tickers evaluated         | 2 (AAPL, SPY)          | 5 (AAPL, SPY, MSFT, BTC, ETH) |
| Joint search samples      | 6                      | 12                     |
| Search dimensions          | 3 (hs, nl, sl)         | 4 (hs, nl, sl, dropout)|
| Multi-seed                | n=2                    | n=3                    |
| Confidence-weighted       | No                     | Yes                    |
| Sharpe-aware loss         | No                     | Yes (w=0.3)            |
| AAPL best Sharpe          | 0.37 (Transformer)     | 0.44 (Ensemble, = baseline) |
| SPY best Sharpe           | 0.67 (LSTM CLS)        | 0.66 (Sel. Ensemble)   |
| MSFT best Sharpe          | N/A                    | 0.61 (LSTM)            |

## Notable Observations

1. **Confidence weighting converges to buy-and-hold on equities**: When models are uncertain (predictions close to zero), position sizes shrink, reducing exposure. The ensemble averaging further smooths positions. Net effect: the strategy approximates buy-and-hold. This is actually a rational outcome — the model correctly identifies that it has limited alpha.

2. **Sharpe-aware loss shifts mode preferences**: SPY models now prefer regression (Cycle 10: all classification). The Sharpe loss term rewards models that produce calibrated magnitude signals, which regression does better than binary classification.

3. **MSFT LSTM (hs=32, nl=3, sl=10) is a new strong result**: Small, deep architecture with short lookback achieves perfect stability (3/3 positive windows) and 0.61 Sharpe. Contradicts the Cycle 10 finding that larger hidden sizes (128) are preferred.

4. **ETH/USDT GRU classification outperforms**: Sharpe 2.79, the highest single-model result across all tickers. However, hourly crypto results are volatile and likely overfitting with only 661 samples and 4 windows.

5. **Crypto models still struggle**: BTC/USDT all-negative, ETH/USDT only 1 of 3 positive. The confidence-weighted approach amplifies losses when models are confidently wrong.

6. **Selective ensemble works well on SPY**: Dropping LSTM (negative Sharpe) and keeping GRU + Transformer improves from 0.53 → 0.66. Validates the selective ensemble mechanism.

7. **Bootstrap significance added**: The hook added bootstrap significance testing alongside the existing t-test. All models remain non-significant (p > 0.05) across all tickers and models.

## Computational Cost

Joint search (12 samples x 3 models = 36 walks per ticker) + mode sweep (2 modes x 3 models = 6 walks) + final runs (3 models x 3 seeds = 9 walks) = ~51 walk-forward evaluations per ticker. With 5 tickers, total ~255 evaluations.

Total experiment runtime: ~45 minutes for 5 tickers.
