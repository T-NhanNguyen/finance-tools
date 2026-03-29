# Securities Margin Requirements for Portfolio Allocation

This document details the transition from flat-rate margin approximations to the CBOE and FINRA Reg-T short option margin formula, integrated with broker-specific requirements from the strategy configuration.

## Background on Allocation Errors

The move to a precise margin formula was triggered by a discrepancy observed in ticker BE. With approximately 113,000 in total equity, the initial flat-rate algorithm suggested selling 17 contracts at a 125 strike. However, the broker (IBKR) rejected the order, suggesting only 11 contracts were permissible.

The flat-rate approximation calculated capacity as notional divided by maintenance requirement (approximately 348,000 capacity). In reality, brokers enforce a non-linear margin gate at order entry that is significantly more restrictive than the maintenance requirement. For BE, the actual margin impact was approximately 96 per share, or 77 percent of the strike price, far exceeding the 32 percent used in the simulation.

## Core Margin Formula

The correct calculation for a single naked short put contract follows the CBOE strategy-based margin rules, substituted with ticker-specific rates from the internal margin requirement data.

The initial margin per contract is the maximum of two separate legs:

Leg 1: Option Premium + (Short Rate * Underlying Price) - Out of the Money Amount
Leg 2: Option Premium + (Floor Rate * Strike Price)

Where:
- Option Premium is the mid-price per share.
- Short Rate is the initial_short requirement (e.g., 0.3012 for UUUU).
- Floor Rate is the initial_long requirement (the 10 percent regulatory floor).
- Out of the Money Amount (OTM) is max(Underlying Price - Strike, 0) for puts.

## Common Pitfalls in Implementation

The most frequent error in margin modeling is using a flat percentage against the total notional value. This fails to account for the premium received (which offsets the requirement) and the OTM credit (which reduces the risk as the underlying moves away from the strike).

A second common mistake occurred during the regression phase: using the Reg-T formula directly to determine contract counts in the scanner. While the formula represents the hard ceiling the broker enforces, using it as an allocation target provides zero safety buffer. This leads to over-allocation where the theoretical capacity is filled to the absolute dollar, leaving the portfolio vulnerable to small price fluctuations that trigger immediate margin calls.

Others may also go wrong by failing to thread the specific ticker context through the math engine. Without the ticker, the algorithm defaults to standard 20 percent and 10 percent coefficients, which will significantly underestimate the requirements for high-volatility or high-maintenance tickers.

## Bifurcated Sizing Logic

To correct these errors, the implementation separates the concern of safety from the concern of precision.

### The Allocator Gate
The Portfolio Margin Allocator uses the Reg-T formula as a hard gate. This ensures that the suggested combination of trades will pass the broker's initial margin check. It serves as the authoritative boundary for total portfolio capacity.

### The Scanner Buffer
The Contract Selling Analyst (the scanner) uses a more conservative calculation for suggested contract sizing. It incorporates a safety buffer target (e.g., 20 percent) between the entry price and the liquidation threshold. This ensures the scanner suggests fewer contracts than the hard ceiling allows, creating a protective cushion.

### ROI Consistency
Despite using a conservative buffer for contract counts, the scanner uses the Reg-T formula for yield and return on investment (ROI) calculations. This ensures that the reported return accurately reflects the capital the broker will actually lock in the account, rather than a theoretical 100 percent cash-secured amount.

## Implementation Steps

Centralize all margin logic in a core math engine (e.g., csp_math_engine.py). This avoids duplicated logic across the scanner and allocator and ensures a single source of truth for the formula.

Always thread the ticker symbol into the math functions. This allows the engine to pull specific broker rates from the configuration. If the ticker is unknown, the engine must fallback to conservative regulatory defaults (20 percent short rate and 10 percent floor rate) rather than a flat percentage.

When calculating the maintenance requirement, substitute the initial_short and initial_long rates with maint_short and maint_long respectively. This allows the algorithm to predict the price floor where a margin call would occur.
