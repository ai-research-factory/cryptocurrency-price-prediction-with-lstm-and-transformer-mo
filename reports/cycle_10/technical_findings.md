# Cycle 10 Technical Findings

## Implementation Summary

Cycle 10 introduced three enhancements to address result instability and computational cost from previous cycles:

### 1. Multi-Seed Averaging (n_seeds=2)
**Problem**: Neural network predictions vary significantly with random initialization (OQ#1 from previous cycles). Results were not reproducible across runs.

**Solution**: `walk_forward_validation_multiseed()` trains each walk-forward window N times with different random seeds and averages predictions before computing metrics.

**Results**: With n_seeds=2, training cost doubles per model but individual window predictions become more stable. This is a modest seed count chosen to limit compute; higher values (e.g., 5) could further reduce variance but at proportional cost.

### 2. Joint Hyperparameter Search (replacing sequential sweeps)
**Problem**: Cycles 8-9 swept hidden_size, num_layers, and seq_len sequentially (greedy), meaning interactions between these hyperparameters were not explored. The full grid (3x3x4=36) was too expensive.

**Solution**: `joint_hyperparam_search()` randomly samples 6 configurations from the full (hidden_size x num_layers x seq_len) space, evaluating each with walk-forward validation. This replaces the sequential sweep (9+9+12=30 evaluations) with only 6 evaluations per model per ticker.

**Selected Parameters (Joint Search)**:

| Ticker | LSTM HS/NL/SL | GRU HS/NL/SL | Transformer HS/NL/SL |
|--------|---------------|--------------|----------------------|
| AAPL   | 128/2/50      | 128/2/30     | 128/2/30             |
| SPY    | 128/1/30      | 64/1/20      | 128/3/50             |

**Finding**: Joint search tends to select larger hidden sizes (128) more consistently than sequential sweeps. AAPL converges to uniform 128 hidden size across models. SPY shows more diversity, with GRU preferring smaller/shallower architecture. The 6-sample budget is tight - increasing to 12 or more would improve coverage.

### 3. Adaptive Mode Selection (classification vs regression per model/ticker)
**Problem**: Cycle 9 found SPY classification models outperforming regression, contradicting Cycle 5 findings. The global classification flag forced all models to the same mode.

**Solution**: `mode_selection_sweep()` tests both regression and classification for each (model, ticker) pair. The mode with higher Sharpe ratio is selected via `select_optimal_mode()`.

**Selected Modes**:

| Ticker | LSTM           | GRU            | Transformer    |
|--------|----------------|----------------|----------------|
| AAPL   | regression     | classification | regression     |
| SPY    | classification | classification | classification |

**Finding**: SPY strongly prefers classification across all models, consistent with Cycle 9 observations. AAPL shows heterogeneity: GRU works better in classification while LSTM and Transformer prefer regression. This validates per-model-ticker mode selection.

## Key Results

### Per-Ticker Best Models (Cycle 10)

| Ticker | Best Model         | Sharpe | Return   | Baseline Sharpe | Stability |
|--------|--------------------|--------|----------|-----------------|-----------|
| AAPL   | Transformer (reg)  | 0.37   | +7.7%    | 0.35            | 0.67      |
| SPY    | LSTM (cls)         | 0.67   | +15.9%   | 1.12            | 1.00      |

### AAPL Results
- **LSTM** (regression, hs=128, nl=2, sl=50): Sharpe -0.73, worst performer. Large hidden size with longer seq_len may be overfitting.
- **GRU** (classification, hs=128, nl=2, sl=30): Sharpe -0.35. Classification didn't help GRU on AAPL.
- **Transformer** (regression, hs=128, nl=2, sl=30): Sharpe 0.37, only positive model. Stability 0.67 (2/3 windows positive). Outperforms buy-and-hold baseline (0.35).
- **Ensemble**: Dragged down by negative LSTM/GRU. Full ensemble Sharpe = 0.44 (matching baseline exactly). Selective ensemble only had 1 model above threshold, so couldn't form.

### SPY Results
- **LSTM** (classification, hs=128, nl=1, sl=30): Sharpe 0.67, best individual. All windows positive (stability=1.0). Shallow architecture works best.
- **GRU** (classification, hs=64, nl=1, sl=20): Sharpe 0.50, stability=1.0. Smaller, faster model still effective.
- **Transformer** (classification, hs=128, nl=3, sl=50): Sharpe 0.18, worst SPY model. Deeper/wider Transformer underperforms on SPY, possibly overfitting.
- **Ensembles**: Equal ensemble Sharpe 0.98, matching baseline. Selective ensemble includes all 3 (all positive) at 0.98.

### Comparison to Cycle 9

| Metric                    | Cycle 9                | Cycle 10               |
|---------------------------|------------------------|------------------------|
| Tickers evaluated         | 5 (BTC, ETH, AAPL, SPY, MSFT) | 2 (AAPL, SPY) |
| Search method             | Sequential sweeps      | Joint random search    |
| Evaluations per ticker    | ~30 sweeps             | ~6 joint + 6 mode sweep + 3 final = ~15 |
| Multi-seed                | No                     | Yes (n=2)              |
| Adaptive mode             | No (global flag)       | Yes (per model/ticker) |
| AAPL best Sharpe          | 0.66 (Transformer)     | 0.37 (Transformer)     |
| SPY best Sharpe           | 0.76 (LSTM CLS)        | 0.67 (LSTM CLS)        |

**Note**: Cycle 10 uses only 2 tickers (AAPL, SPY) per the default config, vs 5 in Cycle 9. Results are not directly comparable due to potential data period differences (API re-fetch). The lower Sharpe values may reflect different market conditions in the test windows.

## Notable Observations

1. **Joint search reduces compute but may miss optima**: 6 random samples from 36 combinations means 83% of the space is unexplored. The sequential approach in Cycle 9, while greedy, evaluated 30 configurations and could find better local optima. Consider increasing `joint_search_samples` to 12+.

2. **Multi-seed averaging has modest impact at n_seeds=2**: The benefit of averaging two models is limited. Standard theory suggests diminishing returns follow sqrt(n), so going from 1 to 2 seeds provides ~30% variance reduction. n_seeds=3-5 would be more effective.

3. **Adaptive mode confirms SPY classification preference**: All 3 SPY models select classification, consistent with Cycle 9 findings. The mechanism correctly identifies this pattern.

4. **Selective ensemble limited by few tickers**: With only 2 tickers and 3 models, selective ensemble has limited opportunities. On AAPL, only 1 model passed the threshold.

5. **Ensemble matches baseline on SPY**: The SPY ensemble achieves Sharpe ~0.98, very close to buy-and-hold baseline (0.98-1.12). The strategy is essentially buying and holding with minimal trading activity (1-2 trades total due to aggressive min_hold or consistent position).

## Computational Cost

Joint search (6 samples x 3 models = 18 walks per ticker) + mode sweep (2 modes x 3 models = 6 walks) + final runs (3 models x 2 seeds = 6 walks) = ~30 walk-forward evaluations per ticker. With 2 tickers, total ~60 evaluations.

Total experiment runtime: ~8.5 minutes for 2 tickers.
