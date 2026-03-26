# Cycle 5: Technical Findings

## Objective

Address remaining open questions: direction classification (OQ #3), long/short positions (OQ #5), minimum holding period tuning (OQ #7 new), and inverse-variance ensemble weighting (OQ #8 new).

## Implementation

### 1. Direction Classification Mode (OQ #3)

Added optional classification mode where models output sigmoid probabilities instead of return forecasts, trained with BCE loss. The dataset converts targets to binary labels (1 if return > 0, else 0).

- Models: Added `classification=True` parameter to both `LSTMPredictor` and `TransformerPredictor`, applying sigmoid to the output
- Training: Switches from MSE to BCE loss when classification is enabled
- Evaluation: Threshold at 0.5 for position signals

**Results:**
Classification mode generally *underperformed* regression:

| Ticker | Model | Regression Sharpe | Classification Sharpe |
|--------|-------|------------------|-----------------------|
| BTC/USDT | LSTM | -0.40 | -0.10 |
| BTC/USDT | Transformer | **3.92** | -5.81 |
| ETH/USDT | LSTM | -5.37 | -8.63 |
| ETH/USDT | Transformer | -1.67 | -5.23 |
| AAPL | LSTM | 0.09 | -1.02 |
| AAPL | Transformer | 0.17 | -0.12 |

Classification was worse in 5 of 6 cases, and dramatically worse for the BTC Transformer (3.92 -> -5.81). The regression approach preserves magnitude information that helps distinguish strong vs weak signals.

### 2. Long/Short Strategy (OQ #5)

Enabled short positions: the strategy now goes long (+1) when predicted return > 0 and short (-1) when < 0, instead of long-or-flat. Short costs are symmetric (same bps per trade).

**Impact:**
With allow_short=True, the strategy is more aggressive. The BTC/USDT Transformer benefited significantly (Sharpe 3.92 vs Cycle 4's -2.18), but this is partly due to the model making very few trades (only 2). Most models saw increased costs from more frequent position changes.

The AAPL ensemble achieved Sharpe 0.39 with long/short, surpassing the buy-and-hold Sharpe of 0.35 for the first time. Max drawdown was also lower (0.27 vs 0.41).

### 3. Minimum Holding Period Sweep (OQ #7 new)

Swept min_hold across [1, 2, 3, 5, 10] to find optimal trade frequency per model/ticker.

**Key findings:**

| Ticker | Model | Optimal Hold | Sharpe at Optimal | Sharpe at hold=3 |
|--------|-------|-------------|-------------------|------------------|
| BTC/USDT | LSTM | 5 | 1.71 | -0.40 |
| BTC/USDT | Transformer | 1-10 (all same) | 3.92 | 3.92 |
| ETH/USDT | LSTM | 1 | -4.62 | -5.37 |
| ETH/USDT | Transformer | 1 | -1.21 | -1.67 |
| AAPL | LSTM | 5 | 0.19 | 0.09 |
| AAPL | Transformer | 3 | 0.17 | 0.17 |

The optimal holding period varies significantly by model and ticker. The BTC LSTM improves dramatically from hold=3 (Sharpe -0.40) to hold=5 (Sharpe 1.71), suggesting the current default of 3 is sub-optimal for crypto LSTM.

### 4. Inverse-Variance Ensemble Weighting (OQ #8 new)

Implemented inverse-variance weighting: each model's weight is proportional to 1/variance of its per-period returns. Models with lower return volatility get higher weight.

**Results:**
In practice, the LSTM and Transformer had very similar return variances on all tickers, resulting in near-equal (0.50/0.50) weights. The inverse-variance ensemble produced identical results to equal weighting. This suggests more heterogeneous models (e.g., different architectures or feature sets) would be needed to see differentiation.

## Results Summary

### BTC/USDT (Hourly, 4 windows, min_hold=3, allow_short=True)

| Model | Sharpe | Sortino | Return | Max DD | Trades |
|-------|--------|---------|--------|--------|--------|
| LSTM (reg) | -0.40 | -0.40 | -0.6% | 8.1% | 23 |
| Transformer (reg) | **3.92** | **4.54** | **+6.5%** | 7.8% | 2 |
| LSTM (cls) | -0.10 | -0.10 | -0.2% | 8.6% | 32 |
| Transformer (cls) | -5.81 | -6.09 | -8.9% | 14.1% | 30 |
| Ensemble (equal) | 3.85 | 4.33 | +6.4% | 5.9% | 4 |
| Ensemble (inv-var) | 3.85 | 4.33 | +6.4% | 5.9% | 4 |
| Buy-Hold | 0.81 | 0.80 | +1.3% | 8.2% | -- |

### ETH/USDT (Hourly, 4 windows, min_hold=3, allow_short=True)

| Model | Sharpe | Sortino | Return | Max DD | Trades |
|-------|--------|---------|--------|--------|--------|
| LSTM (reg) | -5.37 | -4.27 | -10.2% | 12.5% | 19 |
| Transformer (reg) | -1.67 | -1.33 | -3.3% | 7.5% | 28 |
| LSTM (cls) | -8.63 | -6.19 | -15.8% | 18.2% | 18 |
| Transformer (cls) | -5.23 | -5.17 | -9.9% | 17.4% | 24 |
| Ensemble (equal) | -9.95 | -8.91 | -18.0% | 21.9% | 26 |
| Buy-Hold | 0.06 | 0.06 | +0.1% | 11.3% | -- |

### AAPL (Daily, 3 windows, min_hold=3, allow_short=True)

| Model | Sharpe | Sortino | Return | Max DD | Trades |
|-------|--------|---------|--------|--------|--------|
| LSTM (reg) | 0.09 | 0.09 | +4.8% | 38.4% | 50 |
| Transformer (reg) | 0.17 | 0.17 | +9.5% | 37.1% | 33 |
| LSTM (cls) | -1.02 | -1.03 | -42.8% | 68.4% | 61 |
| Transformer (cls) | -0.12 | -0.13 | -6.5% | 34.3% | 71 |
| Ensemble (equal) | **0.39** | **0.40** | **+23.7%** | **26.9%** | 48 |
| Ensemble (inv-var) | **0.39** | **0.40** | **+23.7%** | **26.9%** | 48 |
| Buy-Hold | 0.35 | 0.33 | +21.0% | 40.6% | -- |

## Key Observations

1. **Classification mode is generally worse than regression.** The BCE loss discards magnitude information, leading to noisier signals. The Transformer on BTC went from Sharpe 3.92 (regression) to -5.81 (classification). This conclusively answers OQ #3: regression is preferred.

2. **Long/short with ensemble outperforms buy-and-hold on AAPL.** The AAPL ensemble achieved Sharpe 0.39 vs baseline 0.35, with lower max drawdown (26.9% vs 40.6%). This is the first result in the project to exceed buy-and-hold on a risk-adjusted basis.

3. **BTC Transformer found a strong regime.** Sharpe 3.92 from only 2 trades is noteworthy but likely not robust -- the model is effectively making a single long bet. This should not be interpreted as a reliable strategy.

4. **Min holding period tuning reveals ticker-specific optima.** BTC LSTM optimal at hold=5 (Sharpe 1.71 vs -0.40 at hold=3). The current global default of 3 is reasonable on average but suboptimal per-ticker.

5. **ETH remains the hardest ticker.** All models and configurations underperform buy-and-hold on ETH. The long/short strategy amplified losses rather than helping.

6. **Inverse-variance weighting matched equal weighting.** The LSTM and Transformer have similar return variance, so the weights are approximately equal. More model diversity is needed to benefit from this approach.

7. **No classification model achieves statistical significance in the right direction.** AAPL LSTM classification showed p=0.024 significance, but in the *negative* direction -- it's significantly worse than buy-and-hold.

## Comparison with Cycle 4

| Metric | Cycle 4 | Cycle 5 | Change |
|--------|---------|---------|--------|
| Strategy | Long-or-flat | Long/short | New |
| Classification mode | None | BCE + sigmoid | New |
| Ensemble methods | Equal avg | Equal + inv-variance | New |
| Min hold sweep | None | [1,2,3,5,10] | New |
| Best AAPL Sharpe | 0.21 (Transformer) | **0.39 (Ensemble)** | +0.18 |
| Best BTC Sharpe | -1.46 (LSTM) | **3.92 (Transformer)** | +5.38 |
| AAPL Ensemble vs baseline | 0.06 vs 0.35 | **0.39 vs 0.35** | Exceeds baseline |
