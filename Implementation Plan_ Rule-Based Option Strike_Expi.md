
# Implementation Plan: Rule-Based Option Strike/Expiration Optimizer for Debit Spreads \& Long Legs

## Overview

Build a two-stage, rule-based optimizer that:

1. Uses **Black-Scholes** as the mathematical foundation for Greeks and theoretical prices.
2. Uses **OI, volume, and Gamma Exposure (GEX)** to prune noise and identify where market makers are positioned.
3. Feeds **expected move, support/resistance, and price targets** (from GEX + technicals) into a scoring model to select the best strike/expiration for:
    - Bull debit spreads (bull call / bear put)
    - Long single-leg calls/puts

The output is a ranked list of candidate trades per scenario (bullish 3-month, earnings, TA-level-based).

***

## 1. Data Requirements

### 1.1 Option Chain Data (per underlying)

For each expiration date and strike:

- Call/Put bid, ask, last, mid
- Open Interest (OI)
- Volume
- Implied Volatility (IV)
- Greeks: Delta, Gamma, Theta, Vega, Rho (or compute via Black-Scholes)
- Expiration date (DTE)
- Underlying spot price


### 1.2 Underlying Data

- Current spot price
- Historical prices (at least 1–2 years daily):
    - For realized volatility
    - For technical levels (support/resistance, moving averages, trendlines)
- Earnings dates and past earnings move data (if available)


### 1.3 Derived Quant Levels

From the above, compute:

- **Total OI by expiration**
- **Net GEX by expiration and by strike**
- **Expected move** (from options or historical)
- **Support/resistance levels** from:
    - GEX clusters (put/call walls)
    - Technical analysis (TA) levels (swing highs/lows, moving averages, Fibonacci, VWAP, etc.)
- **Price target zones** for each scenario

***

## 2. Architecture Overview

### High-Level Modules

1. **Data Engine**
    - Ingests and normalizes option chain, underlying prices, earnings data.
    - Computes Black-Scholes Greeks if not provided.
2. **Market-Structure Engine**
    - Computes OI, volume aggregates per expiration.
    - Computes GEX per strike and per expiration.
    - Identifies:
        - High-OI expirations
        - Put walls / call walls
        - Net positive/negative GEX zones
        - Max pain / pinning zones
3. **Thesis \& Scenario Engine**
    - Defines scenario parameters:
        - Bullish 3-month: target horizon, expected move, directional bias.
        - Earnings: earnings date, historical IV crush, expected post-earnings move.
        - TA-level-based: support/resistance, breakout/breakdown targets.
    - Produces:
        - Expected price range
        - Target zone
-_stop-loss / invalidation levels_
4. **Scoring \& Optimization Engine**
    - For each candidate strike/expiry:
        - Compute theoretical value and Greeks via Black-Scholes.
        - Compute scenario-based expected payoff and probability.
        - Apply rule-based filters and weights.
        - Output a **trade score**.
    - Rank candidates and select top trades per strategy type.
5. **Output \& Reporting**
    - Ranked list of trades per scenario and strategy type.
    - Key metrics for each trade:
        - Debit, max profit, max loss, break-even
        - Delta, theta, vega, gamma
        - Probability of profit, expected payoff
        - GEX alignment score, OI/volume score
    - Visualizations: GEX profile, support/resistance, P\&L diagrams.

***

## 3. Step-by-Step Implementation

### Step 1: Data Ingestion \& Normalization

**Tasks:**

- Connect to your data source (API, CSV, database).
- Normalize fields:
    - `underlying`, `expiration`, `strike`, `type` (call/put), `bid`, `ask`, `last`, `oi`, `volume`, `iv`.
- Compute mid price: `mid = (bid + ask) / 2`.
- Ensure consistent datetime handling (timezones, DTE calculation).

**Black-Scholes Greeks (if not provided):**
Implement a standard Black-Scholes module:

Inputs:

- `S` = spot price
- `K` = strike
- `T` = time to expiration in years
- `r` = risk-free rate
- `q` = dividend yield (if any)
- `sigma` = implied volatility

Outputs:

- Call/Put price
- Delta, Gamma, Theta, Vega, Rho

Store these as part of the option record.

***

### Step 2: Expiration-Level Filtering (Stage 1)

**Goal:** Identify “high-OI” expirations that matter.

**Steps:**

1. For each expiration date:
    - Compute:
        - `total_OI = sum(OI_call + OI_put)` over all strikes
        - `total_volume` similarly
        - `net_GEX_expiration = sum(GEX_call + GEX_put)` (see below)
    - Compute DTE (days to expiration).
2. Filter expirations:
    - Keep expirations where:
        - `total_OI` is in top N% (e.g., top 20%) or above a threshold.
        - DTE is within your thesis window:
            - For 3-month bullish: keep expirations from ~30 to ~120 DTE.
            - For earnings: keep expirations around the earnings date (e.g., 0–45 DTE).
            - For TA-based: DTE that covers the expected breakout window.
3. Rank remaining expirations by:
    - `total_OI`
    - `|net_GEX_expiration|`
    - Proximity to your target horizon.

**Output:** A list of candidate expirations per scenario.

***

### Step 3: Compute Gamma Exposure (GEX)

**Definition (practical):**

For each option:

- Gamma (from Black-Scholes): `Gamma_call`, `Gamma_put`
- OI: `OI_call`, `OI_put`
- For a given underlying move `dS`, the dealer’s gamma hedging impact is proportional to gamma × OI.

Define:

$$
\text{GEX}_{\text{call}}(K) = \text{Gamma}_{\text{call}}(K) \times \text{OI}_{\text{call}}(K) \times 100
$$

$$
\text{GEX}_{\text{put}}(K) = -\text{Gamma}_{\text{put}}(K) \times \text{OI}_{\text{put}}(K) \times 100
$$

(negative sign for puts because dealers are short puts → short gamma on the downside; sign conventions can vary, just be consistent).

Then:

$$
\text{net\_GEX}(K) = \text{GEX}_{\text{call}}(K) + \text{GEX}_{\text{put}}(K)
$$

Aggregate by expiration:

$$
\text{net\_GEX}_{\text{exp}} = \sum_{K} \text{net\_GEX}(K)
$$

**Implementation steps:**

- For each candidate expiration:
    - For each strike:
        - Compute Gamma via Black-Scholes.
        - Compute GEX_call, GEX_put, net_GEX.
    - Aggregate net_GEX over strikes.
    - Identify:
        - **Put wall**: strike with large negative net_GEX (heavy put OI + gamma).
        - **Call wall**: strike with large positive net_GEX (heavy call OI + gamma).
        - **GEX clusters**: ranges where |net_GEX| is high.

**Use:**

- Put walls → potential support.
- Call walls → potential resistance.
- High net positive GEX → market may be more stable (dealers hedging in a way that dampens moves).
- High net negative GEX → potential for accelerated moves (dealer hedging amplifies price moves).

***

### Step 4: Derive Expected Move, Support, Resistance, and Targets

#### 4.1 Expected Move

Two common approaches:

**A. Options-based expected move (earnings or event):**

For a given expiration around an event:

$$
\text{Expected Move} \approx \text{Price of ATM Straddle} = \text{Call}_{ATM} + \text{Put}_{ATM}
$$

Or as a percentage:

$$
\text{Expected \% Move} = \frac{\text{ATM Straddle Price}}{S}
$$

Define:

- `upper_expected = S + Expected Move`
- `lower_expected = S - Expected Move`

**B. Volatility-based expected move (for 3-month bullish):**

Use implied volatility for the chosen expiration:

$$
\text{Expected Move}_{1\sigma} = S \times \sigma \times \sqrt{\frac{T}{1}}
$$

Where:

- `σ` = IV (annualized)
- `T` = time to expiration in years

Define:

- `upper_move = S + Expected Move_{1σ}`
- `lower_move = S - Expected Move_{1σ}`

You can also compute 2σ ranges.

#### 4.2 Support \& Resistance from GEX + TA

Combine:

- **GEX-based levels:**
    - Put wall strike(s): support candidates.
    - Call wall strike(s): resistance candidates.
    - Max pain strike: potential magnet.
    - GEX cluster zones: areas where price may stall or reverse.
- **TA-based levels:**
    - Swing highs/lows
    - Moving averages (e.g., 50/200EMA)
    - Trendlines
    - Fibonacci retracements/extensions
    - VWAP bands (if intraday)

**Algorithm:**

1. Create a list of candidate support levels:
    - TA support levels
    - Put wall strikes
    - GEX clusters on the downside
2. Create a list of candidate resistance levels:
    - TA resistance levels
    - Call wall strikes
    - GEX clusters on the upside
3. For each level, compute:
    - Distance from spot: `(level - S) / S`
    - “Strength” score:
        - OI at nearby strikes
        - |GEX| at that level
        - Frequency of TA respect (how many times price bounced/tested)
4. Choose:
    - **Primary support**: highest-strength support below spot.
    - **Primary resistance**: highest-strength resistance above spot.
    - **Secondary levels**: next strongest.

#### 4.3 Price Targets per Scenario

**Scenario A: Bullish regime for next 3 months**

- Direction: up
- Target: next major resistance above spot (TA + call wall).
- Invalidations: below primary support or key moving average.

**Scenario B: Earnings**

- Directional bias: user-specified (bullish/bearish/neutral).
- Target: `S + expected move` (for bullish) or `S - expected move` (for bearish).
- Consider IV crush: if buying single-leg, be aware extrinsic value may drop sharply post-earnings.
- Often better to use **debit spreads** to offset IV crush.

**Scenario C: Technical price levels**

- Direction: breakout above resistance or breakdown below support.
- Target: next TA/Fibonacci target beyond breakout level.
- Use GEX to confirm: breakout level should align with a call wall (for upside) or put wall (for downside) turning into support/resistance.

***

### Step 5: Strategy-Specific Candidate Generation

#### 5.1 Long Single-Leg Calls / Puts

For each candidate expiration (from Step 2):

- Generate candidate strikes:
    - For long calls (bullish):
        - Strikes from ~20 delta to ~70 delta.
        - Focus: 45–65 delta is often optimal.
    - For long puts (bearish):
        - Same delta range, but on the put side.

For each candidate:

- Compute:
    - Price (mid)
    - Delta, Gamma, Theta, Vega
    - Probability of expiring ITM:
        - Approximation: `Prob ITM ≈ Delta` (for short-dated) or use Black-Scholes risk-neutral probability.
    - Expected payoff at target price `S_target`:
        - For call: `max(S_target - K, 0)`
        - For put: `max(K - S_target, 0)`
    - Expected P\&L:
        - `E[P&L] = Prob(reach target) * payoff_at_target - cost`
        - Use a simplified probability model (e.g., lognormal with IV) or Monte Carlo if desired.


#### 5.2 Debit Spreads (Bull Call Spread / Bear Put Spread)

For each candidate expiration:

- Generate pairs:
    - **Bull call spread**:
        - Long call at `K_long` (ATM or slightly ITM).
        - Short call at `K_short` > `K_long` (OTM), typically at or beyond target resistance.
    - **Bear put spread**:
        - Long put at `K_long` (ATM or slightly ITM).
        - Short put at `K_short` < `K_long` (OTM), at or beyond target support.

Rules:

- `K_short - K_long` = width (e.g., \$2, \$5, \$10).
- Narrower width = higher probability, lower max profit.
- Wider width = more risk, more reward.

For each spread:

- Compute:
    - Debit = `price_long - price_short`
    - Max profit = `width - debit`
    - Max loss = `debit`
    - Break-even:
        - Bull call: `K_long + debit`
        - Bear put: `K_long - debit`
    - Probability of profit:
        - Approx as probability the underlying is above (below) break-even at expiration.
    - Expected payoff at target:
        - If target ≥ `K_short` (for bull call), payoff = `width - debit`.
        - If target between `K_long` and `K_short`, interpolate linearly.
    - Greeks:
        - Spread Delta = `Delta_long - Delta_short`
        - Spread Gamma = `Gamma_long - Gamma_short`
        - Spread Theta = `Theta_long + Theta_short` (both negative, so more negative)
        - Spread Vega = `Vega_long - Vega_short`

***

### Step 6: Scoring Model (Rule-Based Optimization)

Define a **score function** for each candidate trade:

$$
\text{Score} = w_1 \cdot \text{EV\_score} + w_2 \cdot \text{Prob\_score} + w_3 \cdot \text{GEX\_alignment} + w_4 \cdot \text{Liquidity\_score} - w_5 \cdot \text{Theta\_penalty} - w_6 \cdot \text{SpreadCost\_penalty}
$$

Where:

1. **EV_score** (expected value)
    - Normalize expected P\&L across candidates:
        - `EV_score = (E[P&L] - min_E) / (max_E - min_E)`
    - Cap outliers to avoid dominance.
2. **Prob_score** (probability of profit)
    - Use probability of profit (POP) from Step 5.
    - `Prob_score = POP` (0–1), or normalized similarly.
3. **GEX_alignment**
    - For bullish trades:
        - Bonus if:
            - Target resistance aligns with call wall or GEX cluster.
            - Long strike is near or slightly above strong support (put wall).
    - For bearish trades:
        - Bonus if:
            - Target support aligns with put wall.
            - Long strike is near or slightly below strong resistance (call wall).
    - Define:
        - `gex_bonus` = function of distance from key GEX levels.
        - Normalize to 0–1.
4. **Liquidity_score**
    - Based on OI and volume:
        - `liquidity = (OI + volume) / max(OI + volume)` across candidates.
    - Also penalize wide bid-ask:
        - `spread_ratio = (ask - bid) / mid`
        - Reduce score if `spread_ratio` is high.
5. **Theta_penalty**
    - For long single-leg:
        - More negative theta = faster decay.
        - `theta_penalty = |Theta| normalized`.
    - For spreads:
        - Net theta is still negative; penalize excessively fast decay relative to DTE.
6. **SpreadCost_penalty** (for spreads only)
    - If debit is too high relative to width (low return/risk):
        - `cost_ratio = debit / width`
        - Penalize high `cost_ratio`.

**Weights (example starting point, tune per scenario):**

- Bullish 3-month:
    - `w1 = 0.25` (EV)
    - `w2 = 0.25` (POP)
    - `w3 = 0.20` (GEX alignment)
    - `w4 = 0.15` (Liquidity)
    - `w5 = 0.10` (Theta)
    - `w6 = 0.05` (Spread cost)
- Earnings:
    - Increase weight on EV and GEX alignment (event-driven).
    - Increase penalty on theta if holding through earnings.
    - Possibly reduce weight on long-term theta if very short-dated.
- TA-based:
    - Increase `w3` (GEX + TA alignment).
    - Emphasize liquidity to ensure clean entries/exits.

You can implement this as a simple function:

```python
def score_trade(trade, scenario):
```

