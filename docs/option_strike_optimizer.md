# Full Project Recap: Rule-Based Option Strike/Expiration Optimizer

## Project Overview

Built a **two-stage optimizer** for debit spreads and long single-leg options that leverages:
- Black-Scholes pricing and Greeks (Gamma, Delta, Theta)
- Open Interest (OI), volume, and Gamma Exposure (GEX) for market structure analysis
- Support/resistance levels via subprocess call to external module
- Scenario-based scoring for 3 different trading scenarios

---

## Implementation Breakdown

### 1. Core Architecture

**File**: `core/strategies/option_strike_optimizer.py`

**Key Classes/Functions**:

| Component | Purpose |
|-----------|---------|
| `OptionStrikeOptimizer` | Main class mirroring `ContractSellingAnalyst` pattern |
| `fetch_gex_single(ticker)` | Fetches GEX-extended option chain data |
| `filter_expirations(chain_data, scenario)` | Stage 1: Filters expirations by OI, volume, GEX density |
| `calculate_expected_move(chain_data, expiration, scenario)` | Calculates expected price movement range |
| `generate_single_leg_candidates(chain_data, scenario)` | Stage 2: Generates single-leg candidates |
| `generate_debit_spread_candidates(chain_data, scenario)` | Stage 2: Generates debit spread candidates |
| `calculate_metrics_single_leg(...)`, `calculate_metrics_debit_spread(...)` | Scoring model with EV, prob, GEX alignment |
| `analyze_strike(ticker, scenario, strike_type, option_type)` | Main entry point |
| `run_single_ticker(...)`, `run_multi_ticker(...)` | Scanner support |

### 2. Two-Stage Pipeline

**Stage 1: Expiration Filtering**
- Filters by minimum OI (default: $100K)
- Filters by minimum volume (default: 100 contracts)
- Prioritizes expirations with high GEX density
- Handles negative `daysToExpiration` from data source

**Stage 2: Strike Scoring**
- Black-Scholes Greeks calculation (Delta, Gamma, Theta, Vega)
- Support/resistance alignment scoring
- GEX density scoring
- Theta decay scoring
- EV (Expected Value) scoring

### 3. Scenario Support

| Scenario | Direction | Time Horizon | Use Case |
|----------|-----------|--------------|----------|
| `bullish_3month` | Bullish | ~90 days | General bullish outlook |
| `earnings` | Directional | ~14 days | Earnings play |
| `ta_breakout` | Directional | Variable | Technical breakout |

Each scenario has different scoring weights:
- `prob_weight`: Probability of hitting target
- `ev_weight`: Expected value score
- `gex_weight`: GEX alignment score
- `theta_weight`: Theta decay score

### 4. Unique Implementation Considerations

| Consideration | Implementation Detail | Rationale |
|--------------|----------------------|-----------|
| **Negative DTE Handling** | `if dte <= 0: dte = 30` | yfinance sometimes returns -1 for daysToExpiration |
| **Subprocess SR Integration** | Calls `python-stock-support-resistance/main.py` via subprocess | Leverages existing k-means clustering for support/resistance detection |
| **NaN Propagation** | Added `isinstance(x, float) and (np.isnan(x) or np.isinf(x))` checks | Prevents NaN from breaking scoring calculations |
| **GEX Data Structure** | Converts `chain_data["strikes"]` (list) to DataFrame | Standardizes data format across codebase |
| **Delta Range** | Expanded from 0.20-0.70 to 0.05-0.95 | Captures far OTM and deep ITM opportunities |
| **Separate Buying/Selling** | Built as standalone module (not integrated with `ContractSellingAnalyst`) | Buying strategies have different constraints than selling |

### 5. Scoring Model Details

**Single-Leg Scoring**:
```
Score = EV_Score * ev_weight + Prob_Score * prob_weight + GEX_Bonus * gex_weight + Theta_Adjustment * theta_weight
```

Where:
- **EV Score**: Normalized expected value (0.0 to 1.0)
- **Prob Score**: Probability of hitting target price
- **GEX Bonus**: Multiplier for GEX alignment (+0.1 if aligned with GEX peak)
- **Theta Adjustment**: Penalty for high negative theta

**Debit Spread Scoring**:
- Uses break-even analysis instead of target price
- Considers both long and short leg deltas
- Calculates max loss vs max profit ratio

---

## How to Run

### Single-Ticker Analysis

**Command**:
```bash
docker-compose run --rm finance-tools python -c "
from core.strategies import OptionStrikeOptimizer
opt = OptionStrikeOptimizer()
result = opt.analyze_strike('SPY', scenario='bullish_3month', strike_type='single_leg', option_type='call')
print(result['top_candidates'])
"
```

### Multi-Ticker Analysis

**Command**:
```bash
docker-compose run --rm finance-tools python -c "
from core.strategies import OptionStrikeOptimizer
opt = OptionStrikeOptimizer()
for ticker in ['SPY', 'QQQ', 'IWM']:
    result = opt.analyze_strike(ticker, scenario='bullish_3month', strike_type='single_leg', option_type='call')
    print(f'{ticker}: {len(result[\"top_candidates\"])} candidates')
"
```

### Test Script (Recommended)

**Command**:
```bash
docker-compose run --rm finance-tools python /app/scripts/test_optimizer.py
```

This runs all three test scenarios:
1. Single-leg call analysis on SPY
2. Debit spread analysis on SPY
3. Multi-ticker analysis on SPY and QQQ

---

## Key Command Differences from Selling Strategy

| Operation | Contract Selling Analyst | Option Strike Optimizer |
|-----------|-------------------------|------------------------|
| **Entry Point** | `ContractSellingAnalyst.analyze()` | `OptionStrikeOptimizer.analyze_strike()` |
| **Data Source** | Standard yfinance chain | GEX-extended chain via `fetch_gex_single()` |
| **Stage 1** | No expiration filtering | Expiration filtering by OI/volume/GEX |
| **Scoring** | Premium collected, time decay | Expected value, prob of target, GEX alignment |
| **SR Integration** | None | Subprocess call to external SR module |

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `core/strategies/option_strike_optimizer.py` | Main implementation |
| `core/strategies/__init__.py` | Export class |
| `core/data/__init__.py` | Export data functions |
| `core/analysis/__init__.py` | Export analysis functions |
| `scripts/test_optimizer.py` | Test script |
| `scripts/README.md` | Test script documentation |
| `Implementation Plan_ Rule-Based Option Strike_Expi.md` | Technical spec |

---

## Bug Fixes Applied

### Issue 1: NaN Values in Expected P&L

**Problem**: `expected_pnl` and `payoff_at_target` showed NaN values.

**Root Cause**:
1. `chain_data.get("daysToExpiration", 30)` returned -1 (negative), causing `sqrt(negative)` to return NaN
2. `expirations` was defined AFTER `expected_move` calculation

**Fix**:
1. Added dte validation: `if dte is None or dte <= 0: dte = 30`
2. Moved `filter_expirations` BEFORE `calculate_expected_move`

**Result**: SPY top candidate now shows Expected P&L: $7.80

---

## Test Results

```
SPY Top Candidate:
  Expected P&L: $7.80
  Delta: 0.35
  Probability of Profit: 79%
  Expiration: 2026-06-05

QQQ Top Candidate:
  Expected P&L: $7.17
  Delta: 0.50
  Probability of Profit: 50%
  Expiration: 2026-06-18
```