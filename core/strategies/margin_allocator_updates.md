# Portfolio Margin Allocator Updates: Feature Breakdown and Technical Scaling

This document details the architectural shifts and mathematical considerations implemented in the PortfolioMarginAllocator to support a unified, risk-adjusted collateral pool.

## 1. Multiple-Choice Knapsack (MCK) Strategy
The core optimization engine has been upgraded from a 0-1 Knapsack to an MCK model. 

The Challenge: A standard knapsack would either take a ticker or leave it. When multiple strikes are available for the same ticker (e.g., strike 16.00 versus strike 17.50 for UUUU), the solver must choose the single best strike that fits the global portfolio context.

The Solution: We implemented Mutual Exclusivity Groups. The Branch-and-Bound search now groups candidates by ticker. Once a strike is selected for a specific underlying, all other strikes for that underlying are pruned from the current search branch.

Result: The optimizer can now dynamically choose between a safer Cash Engine strike or a higher-yielding Wheel Engine strike based on which one maximizes the global Risk-Adjusted Premium.

## 2. Structural Priority and Safety Filters
To prevent the algorithm from chasing high-yield naked strikes with low or no GEX support, we refined the scoring weights and established a structural floor.

### Engine Weighting Shifts
Cash Engine:
- GEX Density: 70 percent
- Floor Depth: 20 percent
- Capital Efficiency: 10 percent

Wheel Engine:
- Capital Efficiency: 40 percent
- GEX Density: 40 percent
- Floor Depth: 20 percent

### The Support Multiplier
A 75 percent score penalty is applied to any strike where the normalized GEX density is below 0.15. This effectively reduces the objective value of yield traps, ensuring they are rejected in favor of verified GEX walls.

## 3. Margin Math and Layering
The allocator treats the portfolio as a tiered capital structure.

Cash Layer:
Total Equity (Cash plus Accumulated Premiums) serves as the first dollar of protection. One hundred percent of this capital is deployed against notional value before broker margin is engaged.

Margin Layer:
The most restrictive maintenance requirement among all active positions sets the global margin boundary. The Margin Limit is calculated as Total Equity divided by the tightest maintenance requirement. 

## 4. Reporting and Visibility Enhancement
The CLI and API responses now provide a forensic breakdown of the allocation.

Price Floor: The margin call threshold for every individual strike, calculated using standard safe loan logic.

Buying Power Remaining: Estimated additional notional headroom available at a conservative 50 percent average requirement.

Assignment Exposure: The total cash liability of the notional value if one hundred percent of the short positions were to be assigned today.

Clamped Utilization: The tightest maintenance requirement is now calculated dynamically only across active positions, eliminating false-positive margin breaches in reports.

## 5. Decision Flow Logic
In scenarios where a 17.50 strike with high yield and no support competes with a 16.0 strike with a GEX wall:
First, the Support Penalty reduces the 17.50 score by 75 percent.
Second, the Weighting Shift increases the 16.0 score by 40 percent.
Finally, the Optimizer calculates that the risk-adjusted value of the 16.0 strike is significantly higher, selecting the wall strike even at a lower raw premium.
