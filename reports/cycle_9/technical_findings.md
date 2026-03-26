# Cycle 9 Technical Findings

## Implementation Summary

Cycle 9 addressed three remaining open questions from Cycle 8:

### 1. Selective Ensemble (OQ#5)
**Problem**: Full ensemble (all 3 models) was being dragged down by underperforming models. In Cycle 8, SPY ensemble (-1.27) was much worse than SPY Transformer alone (1.70).

**Solution**: Added selective ensemble that filters out models with Sharpe ratio below a configurable threshold (default: 0.0). Only models with positive Sharpe are included.

**Results**:
- **BTC/USDT**: All models negative Sharpe → no selective ensemble possible
- **ETH/USDT**: Only GRU positive (3.66) → only 1 model, selective ensemble skipped
- **AAPL**: Only Transformer positive (0.66) → only 1 model, selective ensemble skipped
- **SPY**: All models negative Sharpe → no selective ensemble possible
- **MSFT**: All 3 models positive → selective ensemble formed, Sharpe 1.05 (all included)

**Finding**: With the current data period, most tickers have only 0-1 positive models, preventing selective ensemble formation. When all models are positive (MSFT), selective ensemble equals full ensemble. The mechanism is sound but requires more model diversity or different market conditions to show improvement.

### 2. num_layers Search (OQ#6)
**Problem**: num_layers was fixed at 2 across all models. Sweeping [1, 2, 3] could reveal model-specific depth preferences.

**Solution**: Added `num_layers_sweep()` in evaluation.py and `adaptive_num_layers` pipeline in cli.py. Sweeps [1, 2, 3] per model per ticker using walk-forward validation.

**Selected num_layers per model per ticker**:

| Ticker    | LSTM | GRU | Transformer |
|-----------|------|-----|-------------|
| BTC/USDT  |  2   |  3  |      3      |
| ETH/USDT  |  1   |  3  |      1      |
| AAPL      |  2   |  3  |      1      |
| SPY       |  3   |  3  |      1      |
| MSFT      |  3   |  3  |      2      |

**Finding**:
- GRU consistently prefers 3 layers across all tickers. More depth helps GRU capture complex patterns.
- Transformer tends to prefer fewer layers (1-2). The attention mechanism may already capture long-range dependencies without depth.
- LSTM is mixed (1-3), suggesting sensitivity depends on data characteristics.
- No universal optimal depth exists, confirming the value of per-model-ticker adaptation.

### 3. Early Stopping + Warmup Interaction (OQ#7)
**Problem**: With early_stopping_patience=7 and warmup_epochs=5, Transformer could stop training before warmup effects fully manifest. Early stopping was monitoring validation loss during warmup when LR was still ramping up.

**Solution**: Modified `train_model()` to skip early stopping checks during warmup phase. The condition `epoch >= warmup_epochs` ensures early stopping only activates after warmup completes.

**Effect**: Transformer now always trains through at least the warmup phase before early stopping can trigger. This is most impactful for cases where validation loss is noisy during the LR ramp-up period.

## Key Results

### Per-Ticker Best Models (Cycle 9)

| Ticker    | Best Model         | Sharpe | Return  | Baseline Sharpe | vs Cycle 8 |
|-----------|--------------------|--------|---------|-----------------|------------|
| BTC/USDT  | LSTM CLS           | -0.11  | -0.2%   | 1.24            | Worse      |
| ETH/USDT  | GRU (regression)   | 3.66   | +8.2%   | 0.46            | Improved   |
| AAPL      | Transformer (reg)  | 0.66   | +43.6%  | 0.35            | Improved   |
| SPY       | LSTM CLS           | 0.76   | +29.7%  | 1.16            | Different  |
| MSFT      | Selective Ensemble  | 1.05   | +64.9%  | 1.05            | Comparable |

### Notable Observations

1. **BTC/USDT regression**: All regression models show negative Sharpe this cycle, a significant deterioration from Cycle 8 (where ensemble achieved 7.10). This likely reflects different data periods (API re-fetch) and confirms BTC result fragility noted in previous cycles.

2. **ETH/USDT GRU strength**: GRU achieves Sharpe 3.66 with adaptive num_layers=3, showing deeper GRU architecture captures crypto patterns better.

3. **AAPL Transformer**: Sharpe 0.66 with num_layers=1, stability=1.0 (all windows positive). Shallow Transformer with regime threshold=1.0 reaches 0.96.

4. **SPY classification models outperform regression**: LSTM CLS (0.76) and Transformer CLS (0.35) beat all regression models. First statistical significance achieved: SPY GRU p=0.003, SPY Transformer p=0.019 (though these indicate underperformance vs baseline, not outperformance).

5. **MSFT broad model success**: All 3 models positive Sharpe, enabling the only selective ensemble. Transformer (0.86) is best individual, with regime short improving GRU from 0.76 to 1.08.

### Adaptive Architecture Selections

| Ticker    | Hold | LSTM HS/NL/SL  | GRU HS/NL/SL   | Trans HS/NL/SL  |
|-----------|------|----------------|-----------------|-----------------|
| BTC/USDT  | 10   | 64/2/30        | 64/3/30         | 64/3/30         |
| ETH/USDT  | 3    | 128/1/20       | 32/3/20         | 128/1/10        |
| AAPL      | 5    | 32/2/30        | 64/3/20         | 32/1/30         |
| SPY       | 3    | 32/3/20        | 64/3/10         | 64/1/10         |
| MSFT      | 2    | 32/3/10        | 128/3/10        | 32/2/20         |

(HS=hidden_size, NL=num_layers, SL=seq_len)

## Computational Cost

Adding num_layers sweep (3 levels) per model per ticker adds 9 additional walk-forward passes per ticker (3 models × 3 num_layers). Combined with hidden_size sweep (9 passes) and seq_len sweep (12 passes), the total pre-experiment sweeps are now 30 per ticker, up from 21 in Cycle 8.

Total experiment runtime: ~17 minutes for 5 tickers.
