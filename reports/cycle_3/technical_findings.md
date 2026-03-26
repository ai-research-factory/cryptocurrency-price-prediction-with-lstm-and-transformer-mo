# Cycle 3: Technical Findings

## Objective

Add robust walk-forward validation with purged gaps, interval-aware annualization, cost sensitivity analysis, and additional risk-adjusted metrics (Sortino, Calmar). Address Cycle 2 open questions #5 (window sizing), #8 (cost sensitivity), and #11 (annualization).

## Implementation

### 1. Purged Walk-Forward Validation

Added a configurable `purge_gap` parameter (default: 5 periods) that creates a buffer zone between training and test data in each walk-forward window. This prevents information leakage from overlapping sequences — the last `seq_len` training samples could otherwise share temporal proximity with early test samples.

**File:** `src/evaluation.py` — `walk_forward_validation()` now accepts `purge_gap` parameter.

### 2. Interval-Aware Annualization

Fixed a critical bug from Cycle 2: all Sharpe/Sortino calculations used `sqrt(252)` regardless of data frequency. Now the annualization factor is derived from the data interval:

| Interval | Annualization Factor |
|----------|---------------------|
| 1d       | 252                 |
| 1h       | 6,048               |
| 4h       | 1,512               |
| 15m      | 24,192              |

This resolves open question #11. BTC/USDT hourly data now uses `sqrt(6048)` instead of `sqrt(252)`.

### 3. Adaptive Window Sizing

When `adaptive_window: true`, the system automatically computes window sizes to guarantee at least 3 walk-forward windows. This directly addresses open question #5 — BTC/USDT previously had only 2 windows with fixed `train_size=500`.

**Result:** BTC/USDT now runs 4 windows (train=212, test=91) vs. 2 windows in Cycle 2. AAPL runs 3 windows (train=461, test=198) vs. 11 smaller windows in Cycle 2 (the larger test windows provide more meaningful per-window statistics).

### 4. Cost Sensitivity Analysis

Sweep across 6 cost levels: [0, 2, 5, 10, 20, 50] bps. This addresses open question #8.

### 5. Additional Risk Metrics

- **Sortino ratio:** Uses downside deviation instead of total volatility — better captures asymmetric risk
- **Calmar ratio:** Annualized return / max drawdown — captures recovery risk
- **Total cost:** Explicit tracking of total transaction costs paid
- **Significance test:** Paired t-test of strategy returns vs buy-and-hold baseline (p-value)

### 6. Walk-Forward Config Reporting

Each result now includes the exact walk-forward configuration used (train_size, test_size, step_size, purge_gap, annualization_factor), enabling reproducibility.

## Results

### BTC/USDT (Hourly, 4 windows, purge=5)

| Model | Sharpe | Sortino | Return | Max DD | Cost | Stability |
|-------|--------|---------|--------|--------|------|-----------|
| LSTM | -0.68 | -0.48 | -0.7% | 4.8% | 2.2% | 50% |
| Transformer | -0.88 | -0.76 | -1.1% | 5.2% | 4.8% | 50% |
| Buy-Hold | 0.81 | 0.80 | +1.3% | 8.2% | — | — |

Neither model significantly outperforms the baseline (p=0.73 LSTM, p=0.65 Transformer).

**Cost sensitivity (LSTM):** Profitable at 0-5 bps (Sharpe 0.37-1.44), unprofitable at 10+ bps. Break-even around 7 bps.
**Cost sensitivity (Transformer):** Profitable at 0-5 bps (Sharpe 1.11-3.09), unprofitable at 10+ bps. Break-even around 7 bps. Higher gross alpha but 2x more trades → higher cost drag.

### AAPL (Daily, 3 windows, purge=5)

| Model | Sharpe | Sortino | Return | Max DD | Cost | Stability |
|-------|--------|---------|--------|--------|------|-----------|
| LSTM | 0.08 | 0.07 | +4.0% | 34.9% | 5.8% | 67% |
| Transformer | -0.09 | -0.07 | -4.0% | 40.6% | 5.1% | 33% |
| Buy-Hold | 0.35 | 0.33 | +21.0% | 40.6% | — | — |

Neither model significantly outperforms the baseline (p=0.33 LSTM, p=0.24 Transformer).

**Cost sensitivity (LSTM):** Profitable at 0-10 bps (Sharpe 0.08-0.19), break-even around 17 bps.
**Cost sensitivity (Transformer):** Only marginally profitable at 0 bps (Sharpe 0.02), negative at all higher cost levels.

## Key Observations

1. **Statistical significance:** No model achieves statistical significance vs buy-and-hold at the 5% level on either ticker. This is consistent with efficient market hypothesis expectations and the limited number of walk-forward windows.

2. **Proper annualization matters:** BTC/USDT hourly Sharpe ratios changed substantially with correct `sqrt(6048)` annualization vs the incorrect `sqrt(252)` used in Cycle 2. Cycle 2 BTC results were effectively understating risk-adjusted returns by a factor of ~4.9x.

3. **Cost sensitivity is critical:** Both tickers show strategies are only viable at low cost levels (< 7 bps for BTC, < 17 bps for AAPL). Real-world execution costs (including slippage, market impact) often exceed these thresholds, especially for crypto.

4. **Transformer trades more:** The Transformer generates ~2x more trades than LSTM on BTC, creating proportionally higher cost drag. This suggests the Transformer is more sensitive to noise.

5. **Adaptive windowing helped BTC:** Going from 2 to 4 windows provides more reliable statistics, though still limited.

6. **Purge gap effect:** With purge_gap=5, there is a small reduction in available test data per window, but this is a worthwhile tradeoff for preventing temporal leakage.

## Comparison with Cycle 2

| Metric | Cycle 2 | Cycle 3 | Change |
|--------|---------|---------|--------|
| BTC windows | 2 | 4 | +2 (adaptive) |
| AAPL windows | 11 | 3 | -8 (larger windows) |
| Annualization (BTC) | sqrt(252) | sqrt(6048) | Fixed |
| Purge gap | 0 | 5 | Added |
| Risk metrics | Sharpe only | Sharpe + Sortino + Calmar | Expanded |
| Cost analysis | Single level | 6 levels sweep | Expanded |
| Significance test | None | Paired t-test | Added |

Note: Cycle 2 and Cycle 3 results are not directly comparable due to the annualization fix and different window configurations.
