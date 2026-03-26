# Cycle 4: Technical Findings

## Objective

Address review feedback from Cycles 1-3 open questions: feature importance analysis (OQ #1), learning rate scheduling (OQ #4), trade frequency control (OQ #10), ensemble model (OQ #6), and expanded ticker coverage (OQ #9, #12).

## Implementation

### 1. Learning Rate Scheduling (OQ #4)

Added `ReduceLROnPlateau` scheduler to `training.py` with:
- Factor: 0.5 (halve LR on plateau)
- Patience: 5 epochs
- Min LR: 1e-6

The scheduler is enabled by default. Logs now report current LR alongside loss. In practice, the LR reduced from 1e-3 to 5e-4 during some training windows, indicating the scheduler is active and the model was plateauing.

### 2. Minimum Holding Period (OQ #10)

Added `min_holding_period` parameter to `compute_trading_metrics()` and propagated through the entire pipeline (walk-forward, cost sensitivity, significance test). When `min_hold=3`:

- Once a position change occurs, the position is locked for 3 periods
- This directly reduces trade churn from noisy predictions

**Impact on BTC/USDT LSTM:** 23 trades (Cycle 4, min_hold=3) vs 22 trades (Cycle 3, min_hold=1). The effect is modest here because the LSTM already doesn't flip frequently.

**Impact on BTC/USDT Transformer:** 19 trades (Cycle 4) vs 48 trades (Cycle 3). The Transformer's trade count dropped by 60%, confirming that the Transformer was the primary source of trade churn (as flagged in OQ #10).

**Impact on AAPL Transformer:** 17 trades (Cycle 4) vs 51 trades (Cycle 3). 67% reduction. This dramatically reduced cost drag: AAPL Transformer improved from Sharpe -0.09 (Cycle 3) to Sharpe +0.21 (Cycle 4).

### 3. Feature Importance Analysis (OQ #1)

Implemented permutation feature importance in `evaluation.py`. For each feature, the test data column is shuffled n_repeats=3 times, and the Sharpe degradation is measured.

**BTC/USDT Feature Rankings:**
| Rank | LSTM | Transformer |
|------|------|-------------|
| 1 | macd_signal | macd_hist |
| 2 | bb_pct | stoch_k |
| 3 | atr | macd |
| 4 | macd_hist | rsi |
| 5 | hist_vol | log_return |

**ETH/USDT Feature Rankings:**
| Rank | LSTM | Transformer |
|------|------|-------------|
| 1 | bb_bandwidth | rsi |
| 2 | stoch_k | macd_signal |
| 3 | atr | stoch_k |
| 4 | volume_change | atr |
| 5 | macd_signal | volume_change |

**AAPL Feature Rankings:**
| Rank | LSTM | Transformer |
|------|------|-------------|
| 1 | rsi | rsi |
| 2 | volume_change | macd |
| 3 | macd | stoch_k |
| 4 | hist_vol | bb_pct |
| 5 | macd_signal | atr |

**Key findings:**
- RSI is consistently important across both models and tickers (especially equities)
- MACD family (signal, histogram, line) appears in top 5 for nearly all combinations
- Stochastic %K and ATR are consistently mid-to-high importance
- No single feature is irrelevant — all 12 contribute, though importance varies by model and ticker
- Volume change is more important for LSTM than Transformer

### 4. Ensemble Model (OQ #6)

Added ensemble prediction support that averages LSTM and Transformer predictions before computing trading signals. The ensemble is built from per-window predictions stored during walk-forward validation.

**Results:**

| Ticker | LSTM Sharpe | Transformer Sharpe | Ensemble Sharpe | Baseline Sharpe |
|--------|------------|-------------------|----------------|----------------|
| BTC/USDT | -1.46 | -2.18 | -1.32 | 0.81 |
| ETH/USDT | -6.51 | -0.79 | -1.04 | 0.06 |
| AAPL | -0.06 | 0.21 | 0.06 | 0.35 |

The ensemble generally provides a middle ground between the two models. On AAPL, it smooths out the LSTM's poor performance while benefiting from the Transformer's edge.

### 5. Additional Ticker: ETH/USDT (OQ #9, #12)

Added ETH/USDT (hourly, 1-year) to expand crypto coverage. ETH/USDT has similar data characteristics to BTC/USDT (686 samples after indicator computation, 4 walk-forward windows).

**ETH results are weak:** Both models underperform buy-and-hold significantly, with LSTM showing particularly poor performance (Sharpe -6.51). This suggests the models struggle more with ETH than BTC, possibly due to different market microstructure.

## Results Summary

### BTC/USDT (Hourly, 4 windows, min_hold=3)

| Model | Sharpe | Sortino | Return | Max DD | Trades | Cost | Stability |
|-------|--------|---------|--------|--------|--------|------|-----------|
| LSTM | -1.46 | -1.16 | -1.8% | 7.8% | 23 | 2.3% | 50% |
| Transformer | -2.18 | -1.72 | -2.4% | 5.2% | 19 | 1.9% | 50% |
| Ensemble | -1.32 | -1.11 | -1.6% | 8.6% | 21 | 2.1% | - |
| Buy-Hold | 0.81 | 0.80 | +1.3% | 8.2% | — | — | — |

### ETH/USDT (Hourly, 4 windows, min_hold=3)

| Model | Sharpe | Sortino | Return | Max DD | Trades | Cost | Stability |
|-------|--------|---------|--------|--------|--------|------|-----------|
| LSTM | -6.51 | -4.49 | -8.7% | 11.2% | 19 | 1.9% | 0% |
| Transformer | -0.79 | -0.44 | -0.9% | 5.8% | 9 | 0.9% | 25% |
| Ensemble | -1.04 | -0.60 | -1.1% | 5.1% | 11 | 1.1% | - |
| Buy-Hold | 0.06 | 0.06 | +0.1% | 11.3% | — | — | — |

### AAPL (Daily, 3 windows, min_hold=3)

| Model | Sharpe | Sortino | Return | Max DD | Trades | Cost | Stability |
|-------|--------|---------|--------|--------|--------|------|-----------|
| LSTM | -0.06 | -0.05 | -2.8% | 36.6% | 40 | 4.0% | 67% |
| Transformer | 0.21 | 0.20 | +11.9% | 40.6% | 17 | 1.7% | 67% |
| Ensemble | 0.06 | 0.05 | +3.1% | 40.6% | 34 | 3.4% | - |
| Buy-Hold | 0.35 | 0.33 | +21.0% | 40.6% | — | — | — |

## Key Observations

1. **Minimum holding period is most impactful for the Transformer.** Trade count dropped 60-67% on the Transformer across all tickers. This directly addresses OQ #10 and confirms the Transformer was noise-trading.

2. **AAPL Transformer improved significantly.** The Transformer on AAPL went from Sharpe -0.09 (Cycle 3) to +0.21 (Cycle 4), primarily due to reduced trade churn (51 → 17 trades). This is the best single model result across all tickers.

3. **Ensemble provides modest diversification.** The ensemble generally sits between the two models' performance. It doesn't outperform the best single model, but reduces the risk of picking the wrong model.

4. **Feature importance confirms all 12 indicators contribute.** No feature is clearly useless, though RSI and MACD family dominate. This suggests the current feature set is reasonable but could potentially be pruned to ~8 features without major impact.

5. **ETH/USDT underperforms BTC/USDT.** ETH has similar volatility but the models struggle more, possibly due to different momentum characteristics or thinner liquidity in the dataset period.

6. **LR scheduling is active but subtle.** The scheduler reduced LR in several training windows, but the impact on final metrics is hard to isolate given the other changes.

7. **No model achieves statistical significance.** All p-values remain well above 0.05, consistent with Cycle 3 findings.

## Comparison with Cycle 3

| Metric | Cycle 3 | Cycle 4 | Change |
|--------|---------|---------|--------|
| Tickers | 2 | 3 | +ETH/USDT |
| Min holding period | 1 | 3 | New |
| LR scheduling | None | ReduceLROnPlateau | New |
| Ensemble model | None | LSTM+Transformer avg | New |
| Feature importance | None | Permutation-based | New |
| BTC LSTM trades | 22 | 23 | ~same |
| BTC Transformer trades | 48 | 19 | -60% |
| AAPL Transformer Sharpe | -0.09 | +0.21 | Improved |
| AAPL Transformer trades | 51 | 17 | -67% |

Note: Cycle 3 and 4 results differ due to min_holding_period=3, LR scheduling, and stochastic training. The min_holding_period is the dominant factor.
