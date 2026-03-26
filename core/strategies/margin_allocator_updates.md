# MARGIN ALLOCATOR ARCHITECTURE UPDATE

This document outlines the changes made to the portfolio margin allocation strategy to ensure accurate risk modeling and compliance with broker margin requirements.

## CORE CHANGES AND BUG FIXES

### DUPLICATE ENTRY RESOLUTION
The strategy configuration contained two entries for RKLB with conflicting margin rates. This caused the system to silently default to the lower of the two settings. We have removed the duplicate and retained the more conservative requirement of thirty-two percent for initial margin. This prevents the system from over-leveraging based on stale or incorrect data.

### MARGIN KEY CORRECTION
A logic error was identified where Cash-Secured Put positions were being evaluated using long margin requirements meant for holding stock. Selling options is a short-premium strategy and follows different broker rules. We updated the logic in both the scanner and the allocator to strictly use initial short and maintenance short requirements. This correction ensures the system correctly identifies that selling puts typically requires more collateral than holding the underlying stock in certain accounts.

### OPTIMIZER PRECISION UPGRADE
The previous version of the optimizer used a global approximation based on the single tightest requirement found in the portfolio. This was replaced with a per-position weight calculation. The system now treats every ticker as having its own unique capital weight. This allows for a more diverse mix of assets where low-risk tickers and high-risk tickers can coexist without the entire account being penalized by the single most restrictive asset.

### DUAL LAYER GATING
We introduced a distinction between initial margin and maintenance margin. The system now uses the initial margin requirement as the gate for opening new positions. This reflects the reality that most brokers require more cash up front to open a trade than they do to maintain it. Once a position is open, the system tracks the maintenance margin to determine the ongoing health and buffer of the collateral pool.

## CONSIDERATION POINTS

### ASSIGNMENT EXPOSURE
The system now tracks assignment exposure. This represents the total dollar amount needed to purchase the shares if all options are exercised. Even if the maintenance margin looks healthy, the gap between total equity and assignment exposure indicates the level of margin debt you would incur upon a full exercise event.

### PREMIUM AS EQUITY
The model currently counts premium collected as immediate equity. While this is mathematically true for net liquidation value, many brokers encumber a portion of that premium as collateral for the open position itself. Users should treat the remaining buying power as an estimate rather than an absolute limit.

## CRITICISM OF THE CURRENT MODEL

### CONCENTRATION RISK
The current allocator treats all tickers as independent units within a shared pool. It does not account for concentration risk. If a single ticker makes up more than fifty percent of the total volume, many brokers will increase the margin requirement for that specific ticker. The model currently assumes a flat requirement regardless of individual position size.

### ASSIGNMENT CASCADE
While we track assignment exposure, the model does not simulate the interest costs of being assigned. In a real scenario, being assigned on a significant portion of the portfolio would lead to high margin interest charges that reduce equity over time. The current simulator calculates a static snapshot of the entry moment rather than a multi-day holding period after assignment.

### LINEAR BUYING POWER ESTIMATE
The remaining buying power is estimated using a simple average requirement. This is a linear projection and may not perfectly reflect broker calculations if further positions are opened in highly volatile or specialized tickers that the broker flags for higher requirements.
