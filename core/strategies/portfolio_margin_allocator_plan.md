# Portfolio Margin Allocator - Implementation Plan

## Date: 2026-05-09

## Status: ✅ IMPLEMENTATION COMPLETE

## Executive Summary

Update the `PortfolioMarginAllocator` to address two critical issues:
1. **Over-concentration** - Current algorithm shows strict single strike per ticker with full deployment
2. **Liquidity blindness** - Ignores OI, volume, and GEX cluster data when ranking options

The new implementation will:
- Rank tickers by capital efficiency (highest ROI per dollar of collateral)
- Show top 3 cash engine options per ticker with liquidity metrics
- Provide diversified allocation recommendations
- Highlight best ticker(s) for the week

---

## Current Problems

### Problem 1: Over-Concentration
**Current behavior:**
```
FIG     586x $ 20.00 (notional $1,172,000.00) → FULL DEPLOYMENT
```

**Issues:**
- Puts 100% of capital into one ticker (FIG)
- Ignores diversification benefits
- No visibility into alternative opportunities

### Problem 2: Liquidity Blindness
**Current behavior:**
- Algorithm selects single strike without checking OI, volume, or GEX clusters
- No distinction between "thick" options vs "thin" options
- Missing liquidity metrics in output

**Issues:**
- May select options with low OI (<500) or low volume (<200)
- No visibility into GEX clustering around strikes
- Illiquid options can't be exited efficiently

---

## New Requirements

### Requirement 1: Ticker Ranking
- Rank tickers by **capital efficiency** (ROI per dollar of collateral)
- Capital efficiency = `Trade ROI / Safety Margin %`
- Identify "best ticker" for the week
- Show top N tickers (default: 3-5) for consideration

### Requirement 2: Multi-Strike Allocation Per Ticker
For each ticker, display:
- **Top 3 ranked cash engine options** (sorted by ROI)
- Each option should include:
  - Strike price
  - Premium
  - ROI %
  - Capital efficiency ratio
  - Open Interest (OI)
  - Volume
  - GEX cluster magnitude

### Requirement 3: Diversified Output Format
Replace single-line concentration with structured output:

```
BEST TICKER FOR WEEK: SPY (Capital Efficiency: 18.5%)
------------------------------------------------------

SPY (Top Ticker) - 4 Options Selected
  1. $280 strike @ $2.50 (ROI: 12.3%) - 50 contracts
     OI: 5,200 | Vol: 3,100 | GEX: +$125M
  2. $282 strike @ $2.30 (ROI: 11.8%) - 30 contracts
     OI: 4,800 | Vol: 2,900 | GEX: +$98M
  3. $278 strike @ $2.45 (ROI: 11.5%) - 25 contracts
     OI: 5,500 | Vol: 3,300 | GEX: +$110M
  4. $285 strike @ $2.15 (ROI: 10.9%) - 20 contracts
     OI: 3,800 | Vol: 2,400 | GEX: +$75M

QQQ (Second Choice) - 3 Options Selected
  1. $420 strike @ $2.10 (ROI: 10.8%) - 40 contracts
     OI: 6,200 | Vol: 3,800 | GEX: +$180M
  2. $415 strike @ $1.95 (ROI: 10.2%) - 35 contracts
     OI: 5,900 | Vol: 3,500 | GEX: +$165M
  3. $425 strike @ $2.25 (ROI: 11.5%) - 30 contracts
     OI: 5,100 | Vol: 3,000 | GEX: +$140M
...
```

---

## Implementation Plan

### Phase 1: Metrics Enhancement (Files: `portfolio_margin_allocator.py`, `strategy_config.py`)

#### Task 1.1: Add Liquidity Thresholds
**File:** `core/strategies/strategy_config.py`

```python
# Add to strategy_config.py
OPTION_LIQUIDITY_THRESHOLDS = {
    "min_open_interest": 500,      # Minimum OI to consider liquid
    "min_volume": 200,             # Minimum daily volume
    "min_gex_magnitude": 10_000_000,  # $10M minimum GEX clustering
}
```

#### Task 1.2: Calculate Capital Efficiency
**File:** `core/strategies/contract_selling_analyst.py`

Update `analyze_strike()` to return capital efficiency:
```python
def analyze_strike(...) -> Dict:
    # ... existing metrics calculation ...
    
    capital_efficiency_ratio = trade_roi_true / max(MIN_MONEYNESS_PCT, safety_margin_float)
    
    return {
        # ... existing returns ...
        "capital_efficiency_ratio": round(capital_efficiency_ratio, 4),
        "open_interest": oi_value,
        "volume": volume_value,  # Add volume from chain_data
        "gex_magnitude": abs(gex_value),
    }
```

#### Task 1.3: Calculate Ticker-Level Capital Efficiency
**File:** `core/strategies/portfolio_margin_allocator.py`

Add method to rank tickers:
```python
def rank_tickers_by_capital_efficiency(self, positions: List[Position]) -> List[Dict]:
    """
    Aggregate positions by ticker and calculate weighted capital efficiency.
    Returns list of {ticker, total_notional, avg_capital_efficiency, position_count}
    """
    ticker_data = {}
    for p in positions:
        if p.contracts <= 0: continue
        if p.ticker not in ticker_data:
            ticker_data[p.ticker] = {
                "total_notional": 0.0,
                "capital_efficiency_sum": 0.0,
                "position_count": 0,
                "positions": []
            }
        ticker_data[p.ticker]["total_notional"] += p.notional
        ticker_data[p.ticker]["capital_efficiency_sum"] += p.capital_efficiency_ratio
        ticker_data[p.ticker]["position_count"] += 1
        ticker_data[p.ticker]["positions"].append(p)
    
    ranked = []
    for ticker, data in ticker_data.items():
        avg_efficiency = data["capital_efficiency_sum"] / data["position_count"]
        ranked.append({
            "ticker": ticker,
            "total_notional": data["total_notional"],
            "capital_efficiency": round(avg_efficiency, 4),
            "position_count": data["position_count"],
            "positions": data["positions"]
        })
    
    return sorted(ranked, key=lambda x: x["capital_efficiency"], reverse=True)
```

### Phase 2: Output Format Refactoring

#### Task 2.1: Refactor `print_summary()`
**File:** `core/strategies/portfolio_margin_allocator.py`

Replace current single-line output with structured format:
```python
def print_summary(self, ranked_tickers: List[Dict] = None):
    """Prints diversified allocation view."""
    print(f"\n{'='*82}")
    print(f"PORTFOLIO MARGIN ALLOCATOR — Diversified Capital Allocation")
    print(f"{'='*82}")
    
    # Best ticker header
    if ranked_tickers:
        best = ranked_tickers[0]
        print(f"\n⭐ BEST TICKER FOR WEEK: {best['ticker']} (Capital Efficiency: {best['capital_efficiency']*100:.1f}%)")
        print(f"-"*82)
    
    # Ticker breakdown
    for rank, ticker_data in enumerate(ranked_tickers or [], 1):
        ticker = ticker_data["ticker"]
        positions = ticker_data["positions"]
        
        print(f"\n{ticker} {'(Top Pick)' if rank == 1 else f'(Rank {rank})'}")
        print(f"  Total Notional: ${ticker_data['total_notional']:,.2f} | Positions: {ticker_data['position_count']}")
        print(f"  Capital Efficiency: {ticker_data['capital_efficiency']*100:.1f}%")
        print(f"  {'':>4}{'Strike':<10}{'Premium':<12}{'Contracts':<10}{'ROI %':<10}{'OI':<12}{'Vol':<12}{'GEX':<15}")
        print(f"  {'-'*70}")
        
        for p in positions[:3]:  # Top 3 options per ticker
            oi = getattr(p, 'open_interest', 0)
            vol = getattr(p, 'volume', 0)
            gex = getattr(p, 'gex_magnitude', 0)
            oi_str = f"{int(oi/1000):.0f}k" if oi >= 1000 else f"{int(oi):.0f}"
            vol_str = f"{int(vol/1000):.0f}k" if vol >= 1000 else f"{int(vol):.0f}"
            gex_str = f"${gex/1e6:.1f}M" if gex >= 1e6 else f"${gex/1e3:.0f}k"
            
            roi = getattr(p, 'trade_roi_pct', 0)
            print(f"  {'':>4}${p.strike:<9.2f}${p.premium_collected:<11.2f}{p.contracts:<10}{roi:<10.2f}{oi_str:<12}{vol_str:<12}{gex_str:<15}")
        
        # Show not selected positions if any
        not_selected = [p for p in positions if p.contracts == 0]
        if not_selected:
            print(f"  {'':>4}...")
            print(f"  {'':>4}({len(not_selected)} lower-ranked options not selected)")
    
    # Portfolio summary
    print(f"\n{'='*82}")
    print(f"PORTFOLIO SUMMARY")
    print(f"{'='*82}")
    print(f"  Cash (Lender):         ${self.cash:>14,.2f}")
    print(f"  Accumulated Premiums:  ${self.accumulated_premiums:>14,.2f}")
    print(f"  Total Equity:          ${self.total_equity:>14,.2f}")
    # ... rest of existing summary ...
```

#### Task 2.2: Update `simulate_multi_asset_portfolio()`
**File:** `core/strategies/portfolio_margin_allocator.py`

Add ticker ranking after optimization:
```python
def simulate_multi_asset_portfolio(...) -> PortfolioMarginAllocator:
    # ... existing code ...
    
    # Run Knapsack Solver
    optimal_allocations = optimize_allocation(portfolio, candidate_positions)
    
    # Post-process for final portfolio state
    for orig, opt in zip(candidate_positions, optimal_allocations):
        if opt.contracts > 0:
            opt.status = "SELECTED" if opt.contracts == orig.contracts else "SCALED DOWN"
            portfolio.positions.append(opt)
            portfolio.accumulated_premiums += opt.premium_collected * opt.contracts * 100
            # Add liquidity metrics to position for output
            opt.open_interest = orig.open_interest if hasattr(orig, 'open_interest') else 0
            opt.volume = orig.volume if hasattr(orig, 'volume') else 0
            opt.gex_magnitude = orig.gex_magnitude if hasattr(orig, 'gex_magnitude') else 0
        else:
            opt.status = "REJECTED"
            opt.open_interest = orig.open_interest if hasattr(orig, 'open_interest') else 0
            opt.volume = orig.volume if hasattr(orig, 'volume') else 0
            opt.gex_magnitude = orig.gex_magnitude if hasattr(orig, 'gex_magnitude') else 0
    
    # Rank tickers by capital efficiency
    ranked_tickers = portfolio.rank_tickers_by_capital_efficiency(portfolio.positions)
    
    return portfolio, ranked_tickers
```

### Phase 3: CLI Output Update

#### Task 3.1: Update `__main__` block
**File:** `core/strategies/portfolio_margin_allocator.py`

```python
if __name__ == "__main__":
    # ... existing arg parsing ...
    
    print(f"\nWorking Capital: ${cash_equity:,.2f}")
    print(f"Strategy: {args.strategy}")
    print(f"\nScanning options chain for {len(tickers)} tickers...")
    
    portfolio, ranked_tickers = simulate_multi_asset_portfolio(
        tickers=tickers, 
        strategy_type=args.strategy, 
        expiration=args.expiration, 
        cash_equity=cash_equity
    )
    
    # Print diversified allocation
    print("\nOptimal combination calculated successfully:")
    portfolio.print_summary(ranked_tickers)
```

---

## Phase 3: Testing & Validation

### Test Case 1: Single Ticker (Control)
**Input:** `python portfolio_margin_allocator.py SPY`

**Expected:**
- SPY should show multiple options (top 3 cash engine)
- Not 100% concentration to single strike
- Liquidity metrics (OI, Vol, GEX) displayed

### Test Case 2: Multiple Tickers
**Input:** `python portfolio_margin_allocator.py SPY QQQ NVDA`

**Expected:**
- Top ticker ranked by capital efficiency
- Top 3 options per ticker displayed
- Portfolio summary shows diversified allocation

### Test Case 3: Liquidity Filtering
**Input:** Apply minimum OI threshold (e.g., 500)

**Expected:**
- Options with OI < 500 should not appear in top 3
- Lower-ranked options (below threshold) should be excluded from display

---

## Implementation Order

1. **Phase 1** (Metrics) → 2 hours
   - Task 1.1: Add liquidity thresholds
   - Task 1.2: Calculate capital efficiency
   - Task 1.3: Ticker ranking method

2. **Phase 2** (Output Format) → 3 hours
   - Task 2.1: Refactor print_summary
   - Task 2.2: Update simulate_multi_asset_portfolio

3. **Phase 3** (CLI Update) → 1 hour
   - Task 3.1: Update main block

**Total Estimate:** 6 hours

---

## Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| Tickers allocated | 1 (100% concentration) | 2-4 (diversified) |
| Options per ticker | 1 | 3 (top cash engine) |
| Liquidity info | None | OI, Vol, GEX displayed |
| Capital efficiency ranking | None | Tickers ranked |
| User guidance | Single strike only | Best ticker + alternatives |

---

## Known Limitations

1. **Knapsack constraint**: Current optimizer still enforces one strike per ticker
   - **Mitigation**: We display top 3 options per ticker in output, let user decide
   - **Future**: Modify MCK to allow multiple strikes per ticker within same engine

2. **Liquidity data availability**: GEX/OI data may not be available for all tickers
   - **Mitigation**: Use fallback values or display "N/A"
   - **Future**: Add data quality validation

3. **Scalability**: print_summary with ranked_tickers could be slow for 50+ tickers
   - **Mitigation**: Default to top 5 tickers in output
   - **Future**: Add `--max-tickers` CLI option

---

## Related Files

| File | Change Type | Description |
|------|-------------|-------------|
| `core/strategies/strategy_config.py` | Add | Liquidity thresholds config |
| `core/strategies/contract_selling_analyst.py` | Add | Capital efficiency calculation |
| `core/strategies/portfolio_margin_allocator.py` | Major | Output refactoring, ticker ranking |
| `core/analysis/csp_math_engine.py` | Unchanged | No changes needed |

---

## Next Steps

1. ✅ Implementation plan approved
2. Implement Phase 1: Metrics Enhancement
3. Implement Phase 2: Output Format Refactoring
4. Implement Phase 3: CLI Update
5. Test with sample tickers
6. Update documentation
7. Deploy to production