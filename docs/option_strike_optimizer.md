# Option Strike Optimizer: Institutional Alpha via Market Microstructure

## The Philosophy: Trading with Structural Footprints

Most traders select option strikes based on "gut feel" or simple Delta thresholds (e.g., "I always buy 30-delta calls"). While Delta represents a mathematical probability, it ignores the **structural forces** that actually move or pin the price of an underlying stock.

The **Option Strike Optimizer** is designed to move beyond "blind" probability. It treats the stock market as a liquidity landscape where **Market Makers (MMs)** and **Institutional Anchors** dictate the path of least resistance. By identifying where these players are positioned, we can select strikes that aren't just "likely" to be profitable, but are **structurally supported** by the market's own mechanics.

---

## 1. The Structural Pillars (The "Where")

The optimizer identifies specific price levels where the market is likely to pause, pivot, or accelerate. We look for **confluence**—the overlap of multiple independent signals.

### Gamma Exposure (GEX) Walls & Clusters
Market Makers hedge their options portfolios by buying or selling the underlying stock. 
- **The Walls**: We identify the "Call Wall" and "Put Wall"—specific strikes with the highest absolute Gamma. These act as "magnets" (attracting price) or "sticky zones" (pinning price) as expiration approaches.
- **The Clusters**: Beyond single strikes, we detect **GEX Zones** (strikes above the 60th percentile of exposure). A strike inside a cluster is more robust than an isolated level because it represents a broad band of institutional hedging interest.

### Institutional Anchors (SMA 20/50)
We incorporate the **20-day and 50-day Simple Moving Averages (SMAs)**. These are not just "indicators"; they are levels where institutional rebalancing and algorithmic "buy/sell programs" are triggered. When a GEX wall coincides with a 50-day SMA, we consider that level a "Primary Anchor."

### Multi-Method Support & Resistance
The engine uses three distinct layers to find price floors and ceilings:
1. **K-Means Clustering**: Statistical grouping of historical price action to find "memory zones."
2. **Local Extrema**: Identifying recent swing highs and lows (last 20 days).
3. **Rounded Psych Levels**: Natural human/algorithmic attraction to whole numbers (e.g., $450, $500).

---

## 2. Expected Move Dynamics (The "How Far")

Before picking a strike, we must define the "Playing Field."

### Implied vs. Realized Volatility
The optimizer computes the **Expected Move** by blending two perspectives:
1. **Options Pricing (The Straddle)**: What the market is *currently* pricing in via the At-The-Money (ATM) straddle.
2. **Statistical Volatility (1σ)**: What the stock's actual price movement history suggests is a 68% probability range.

### Earnings Intelligence (The Move Richness)
During earnings season, implied volatility (IV) skyrockets. The optimizer performs a deep-dive into the last 8 earnings cycles:
- **Priced-in vs. Realized**: If the market is pricing in a 5% move but the stock historically moves 8%, the options are "Cheap" (High Move Richness). 
- **Bound Expansion**: If a stock historically over-delivers on its move, the optimizer **widens the target range** to ensure we aren't selecting strikes that "cap" our potential profit too early.

---

## 3. The Scoring Engine (The "Why")

Every candidate is scored from **0.0 to 1.0**. The score is a weighted hierarchy reflecting our core priorities:

| Factor | Weight | Rationale |
|---|---|---|
| **GEX Alignment** | 30% | **The North Star.** If the strike isn't aligned with MM hedging zones, it's a "floating" trade with no structural support. |
| **Liquidity** | 25% | **The Crowd.** High Open Interest and Volume act as a "consensus" filter. We avoid "ghost strikes" with wide bid-ask spreads. |
| **Probability (POP)** | 20% | **The Math.** Using Normal CDF to ensure the trade has a statistically sound chance of expiring in-the-money. |
| **Expected Value (EV)** | 12% | **The Reward.** A secondary signal to ensure the risk-to-reward ratio makes sense after all structural filters are applied. |
| **Theta Penalty** | 8% | **The Clock.** We penalize high-decay candidates to ensure we aren't "paying too much for the time" we have. |
| **Spread Cost** | 5% | **The Efficiency.** For spreads, we penalize debits that exceed 40% of the spread width. |

---

## 4. Strategy Mechanics

### Single-Leg (Long Calls/Puts)
- **LEAPS Protection**: For long-dated options (>9 months), we tighten the Delta cap to **0.70**. This ensures we are buying high-leverage delta, not "overpaying for deep-ITM insurance" that behaves like a stock substitute.
- **Directional Bias**: The engine automatically flips its logic for Bearish scenarios, identifying Put Walls and structural resistance.

### Debit Spreads (Bull Call / Bear Put)
- **Symmetric Spreads**: We generate spreads where the **Short Strike** is positioned at or near a GEX wall. This lets the Market Maker's "pinning" behavior work *for* us by keeping the short strike OTM while the long strike gains value.
- **Invalidation Levels**: For every spread, we surface a structural "Exit Point." If the price breaks the nearest Primary Support/Resistance, the structural thesis for the trade is dead, and the optimizer suggests an exit.

---

## 5. Mental Model for the Trader

When you look at a **Score of 0.85**, you aren't just seeing a "good trade." You are seeing a trade where:
1. The **Market Makers** are positioned to support the price move.
2. The **Institutional SMAs** provide a floor for your entry.
3. The **Liquidity** ensures you can get out at a fair price.
4. The **Earnings History** suggests the move is underpriced by the crowd.

**Goal**: Don't just trade the ticker; trade the **market structure**.