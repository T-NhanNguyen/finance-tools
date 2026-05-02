# Computation & Logic: Rule-Based Option Strike Optimizer

## 1. Abstract
This implementation provides a deterministic, multi-stage optimization engine for selecting option strikes and expirations. It integrates Black-Scholes mathematical foundations with market structure indicators (Gamma Exposure, Open Interest) and technical analysis (Support/Resistance) to score and rank trade candidates across various market scenarios.

---

## 2. Data Ingestion & Market Structure
The engine utilizes `yfinance` for raw data retrieval, wrapped in a thread-safe, atomic caching layer `[bulk_data_loader.py, L17-67]`.

### 2.1 Gamma Exposure (GEX) Calculation
GEX measures the dollar-gamma exposure of dealers at specific price levels.
```python
# [gex_provider.py, L109-118]
GEX_call = Gamma_call * OI_call * 100 * SpotPrice
GEX_put  = -Gamma_put * OI_put * 100 * SpotPrice
Net_GEX  = sum(GEX_call + GEX_put)
```
Aggregate density is then used to filter liquid expirations `[bulk_data_loader.py, L152-163]`.

### 2.2 Support & Resistance (S/R) Identification
S/R levels are derived through a hybrid approach of K-Means clustering on historical closing prices and local extrema detection `[option_strike_optimizer.py, L200-288]`.
- **Clustering**: Groups historical price action into $k$ optimal clusters.
- **Extrema**: Identifies local swing highs/lows from the last 20 trading days.

---

## 3. Mathematical Models

### 3.1 Expected Move Estimation
The engine predicts the high/low price boundaries using two primary methods `[option_strike_optimizer.py, L331-411]`:
1. **Options-Based (ATM Straddle)**: `Move = Call_ATM + Put_ATM`.
2. **Volatility-Based (1σ)**: `Move = S * IV * sqrt(DTE / 365)`.

The range is then dynamically adjusted toward GEX "Walls" (Call/Put walls) if they are within a 25% proximity to the spot price `[option_strike_optimizer.py, L389-404]`.

### 3.2 Probability Modeling
Profit probability is calculated using the standard Normal Cumulative Distribution Function (CDF) for the risk-neutral probability of the underlying being above (for calls) or below (for puts) the break-even point at expiration `[option_strike_optimizer.py, L551-553]`.
```python
d2 = (log(S/K) + (-0.5 * IV^2) * T) / (IV * sqrt(T))
Prob_Profit = norm.cdf(d2)
```

---

## 4. Scoring & Optimization Engine
Candidates are generated based on Delta thresholds (typically 0.15 to 0.70) and filtered by liquidity (OI > Scenario Min or Volume > 500) `[option_strike_optimizer.py, L311-325]`.

### 4.1 Scoring Algorithm
The Final Score ($S$) is a weighted sum of four primary components `[option_strike_optimizer.py, L574-581]`:

$$S = (W_{ev} \cdot E[PnL]_{norm}) + (W_{prob} \cdot P_{profit}) + (W_{gex} \cdot GEX_{bonus}) + ROI_{bonus}$$

- **EV Score**: Normalized expected P&L using hyperbolic tangent to cap outliers.
- **Prob Score**: Raw probability of profit.
- **GEX Bonus**: A fixed +0.2 bonus if the strike aligns with a major GEX wall (+/- 2% proximity).
- **ROI Bonus**: Rewarding higher return-on-capital (up to +0.1 for 50% ROI).

### 4.2 Penalties
- **Deep ITM Penalty**: -0.15 score reduction for Delta > 0.80 to avoid low leverage trades.
- **Spread Safety Penalty**: -0.10 for debit spreads where the cost exceeds 60% of the width.

---

## 5. Conclusion
By combining probabilistic forecasting with structural market positioning, this implementation effectively prunes the noise of the option chain to highlight trades with the highest confluence of mathematical edge and market support.
