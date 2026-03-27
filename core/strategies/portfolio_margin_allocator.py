"""
Portfolio Margin Allocator

Models a single shared collateral pool across multiple open CSP/CC positions.
Implements a 0-1 Knapsack (Branch and Bound) optimizer to find the perfect contract sizes
across different tickers to maximize premium without breaching cash or margin limits.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from core.strategies.strategy_config import MARGIN_REQS, DEFAULT_MARGIN_REQ, LENDERS
from core.strategies.contract_selling_analyst import ContractSellingAnalyst

# =============================================================================
# POSITION
# =============================================================================

@dataclass
class Position:
    """A single open option-selling position in the shared pool."""
    ticker: str
    strike: float
    contracts: int
    premium_collected: float
    pillar_score: float = 1.0
    price_floor: float = 0.0
    strategy_type: str = "CSP"  # CSP or CC
    maint_req: Optional[float] = None
    status: Optional[str] = None # NEW: Stores simulation outcome

    def __post_init__(self):
        if self.maint_req is None:
            margin_info = MARGIN_REQS.get(self.ticker.upper(), {})
            key = "maint_long" if self.strategy_type.upper() == "CSP" else "maint_short"
            self.maint_req = margin_info.get(key, DEFAULT_MARGIN_REQ)

    @property
    def notional(self) -> float:
        return self.strike * self.contracts * 100


# =============================================================================
# PORTFOLIO MARGIN ALLOCATOR
# =============================================================================

@dataclass
class PortfolioMarginAllocator:
    """
    Manages shared collateral pool across all open positions.
    Enforces that margin is only deployed *after* total cash equity is fully utilized.
    """
    cash: float
    accumulated_premiums: float = 0.0
    positions: List[Position] = field(default_factory=list)

    @property
    def total_equity(self) -> float:
        return self.cash + self.accumulated_premiums

    @property
    def tightest_req(self) -> float:
        """
        Calculates the tightest maintenance requirement across ONLY active positions.
        """
        active_positions = [p for p in self.positions if p.contracts > 0]
        if not active_positions:
            return DEFAULT_MARGIN_REQ
        return max(p.maint_req for p in active_positions)

    @property
    def margin_limit(self) -> float:
        """
        Margin boundary established by the single most restrictive active maint_req.
        """
        return self.total_equity * (1 / self.tightest_req - 1)

    @property
    def total_notional(self) -> float:
        return sum(p.notional for p in self.positions)

    @property
    def cash_utilized(self) -> float:
        """Cash is utilized first. It caps out at total_equity."""
        return min(self.total_notional, self.total_equity)

    @property
    def margin_utilized(self) -> float:
        """Margin is only tapped for the notional that exceeds cash equity."""
        return max(0.0, self.total_notional - self.total_equity)

    def print_summary(self):
        """Prints a structured view of the shared collateral pool state."""
        print(f"\n{'='*65}")
        print(f"PORTFOLIO MARGIN ALLOCATOR — Shared Collateral Pool")
        print(f"{'='*65}")
        print(f"  Cash (Lender):         ${self.cash:>14,.2f}")
        print(f"  Accumulated Premiums:  ${self.accumulated_premiums:>14,.2f}")
        print(f"  Total Equity:          ${self.total_equity:>14,.2f}")
        print(f"  Margin Limit:          ${self.margin_limit:>14,.2f}  (tightest req: {self.tightest_req*100:.1f}%)")
        print(f"{'─'*65}")
        print(f"  {'Ticker':<8} {'Contracts':>9} {'Strike':>8} {'Notional':>12} {'Premium':>11} {'Floor':>8}")
        print(f"{'─'*65}")
        for p in self.positions:
            if p.contracts > 0:
                total_prem = p.contracts * p.premium_collected * 100
                print(f"  {p.ticker:<8} {p.contracts:>9} {p.strike:>8.2f} {p.notional:>12,.2f} {total_prem:>11,.2f} {p.price_floor:>8.2f}")
        print(f"{'─'*65}")
        print(f"  {'TOTALS':<8} {'':>9} {'':>8} {self.total_notional:>12,.2f} {self.accumulated_premiums:>11,.2f} {'':>8}")
        print(f"{'─'*65}")
        print(f"  Cash Layer:   ${self.cash_utilized:>12,.2f} / ${self.total_equity:,.2f}  ({(self.cash_utilized / self.total_equity * 100) if self.total_equity else 0:.1f}% used)")
        print(f"  Margin Layer: ${self.margin_utilized:>12,.2f} / ${self.margin_limit:,.2f}  ({(self.margin_utilized / self.margin_limit * 100) if self.margin_limit else 0:.1f}% used)")
        print(f"{'='*65}\n")


def optimize_allocation(allocator: PortfolioMarginAllocator, candidates: List[Position]) -> List[Position]:
    """
    Solves the multidimensional knapsack problem for optimal contract sizing across overlapping constraints.
    Maximizes total premium collected while strictly respecting the tiered cash and margin limits.
    """
    best_premium = -1.0
    best_counts = {i: 0 for i in range(len(candidates))}
    E0 = allocator.cash

    unique_reqs = sorted(list(set(c.maint_req for c in candidates)))

    for M in unique_reqs:
        valid_indices = [i for i, c in enumerate(candidates) if c.maint_req <= M]
        if not valid_indices: continue

        items = []
        for i in valid_indices:
            c = candidates[i]
            # Weight: adjusted notional requirement (buying power consumed)
            w = 100 * (c.strike - c.premium_collected / M)
            # Value: Risk-Adjusted Premium (Premium * Pillar Score)
            v = 100 * c.premium_collected * c.pillar_score
            cap = c.contracts
            items.append((i, w, v, cap))

        items.sort(key=lambda x: x[2] / x[1] if x[1] > 0 else float('inf'), reverse=True)
        capacity = E0 / M

        # Group items by ticker to enforce mutual exclusivity
        ticker_groups = {}
        for i, w, v, cap in items:
            t = candidates[i].ticker
            if t not in ticker_groups: ticker_groups[t] = []
            ticker_groups[t].append((i, w, v, cap))
        
        group_list = list(ticker_groups.values())

        def dfs(group_idx: int, current_w: float, current_v: float, counts: dict):
            nonlocal best_premium, best_counts
            if group_idx == len(group_list):
                if current_v > best_premium:
                    best_premium = current_v
                    for i in range(len(candidates)):
                        best_counts[i] = counts.get(i, 0)
                return

            # Upper bound heuristic for pruning
            rem_cap = capacity - current_w
            ub = current_v
            # Simple greedy bound across remaining groups
            for next_g_idx in range(group_idx, len(group_list)):
                max_v_in_group = 0
                for _, nw, nv, ncap in group_list[next_g_idx]:
                    take = min(ncap, rem_cap / nw) if nw > 0 else ncap
                    max_v_in_group = max(max_v_in_group, take * nv)
                ub += max_v_in_group
            
            if ub <= best_premium: return

            # Option 1: Skip this ticker entirely
            dfs(group_idx + 1, current_w, current_v, counts)

            # Option 2: Try each strike in this ticker group
            for orig_i, w, v, cap in group_list[group_idx]:
                max_take = cap
                if w > 0: 
                    max_take = min(max_take, int((capacity - current_w) / w))
                
                if max_take > 0:
                    counts[orig_i] = max_take
                    dfs(group_idx + 1, current_w + max_take * w, current_v + max_take * v, counts)
                    counts[orig_i] = 0

        dfs(0, 0.0, 0.0, {})

    optimal_positions = []
    for i, c in enumerate(candidates):
        optimal_size = best_counts[i]
        optimal_positions.append(
            Position(
                ticker=c.ticker,
                strike=c.strike,
                contracts=optimal_size,
                premium_collected=c.premium_collected,
                pillar_score=c.pillar_score,
                price_floor=c.price_floor,
                strategy_type=c.strategy_type,
                maint_req=c.maint_req
            )
        )
    return optimal_positions


# =============================================================================
# BUSINESS LOGIC: PORTFOLIO SIMULATION
# =============================================================================

def simulate_multi_asset_portfolio(
    tickers: List[str],
    strategy_type: str = "CSP",
    expiration: Optional[str] = None,
    cash_equity: Optional[float] = None
) -> PortfolioMarginAllocator:
    """
    Core function to analyze multiple tickers and solve for optimal collateral allocation.
    This is the authoritative representation for both CLI and API consumers.
    """
    if cash_equity is None:
        cash_equity = sum(LENDERS)

    portfolio = PortfolioMarginAllocator(cash=cash_equity)
    analyst = ContractSellingAnalyst(cash_equity=cash_equity)
    
    def process_ticker(t):
        try:
            return t, analyst.scan(t.upper(), expiration_input=expiration, strategy_type=strategy_type.upper(), engine_mode="BOTH")
        except Exception as e:
            return t, {"error": str(e)}

    # Scan in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
         collected_results = list(executor.map(process_ticker, tickers))

    candidate_positions = []
    for t, res in collected_results:
        if "error" in res: continue
            
        pillars_dict = res.get("pillars", {})
        # Feed all candidates from both engines to let the optimizer decide relative safety vs yield
        for eng_key in ["Top_Wheel_Engine", "Top_Cash_Engine"]:
            for p_data in pillars_dict.get(eng_key, []):
                candidate_positions.append(
                    Position(
                        ticker=t,
                        strike=p_data['Strike'],
                        contracts=p_data['Contracts'],
                        premium_collected=p_data['Premium_Raw'],
                        pillar_score=p_data.get('Pillar_Score', 1.0),
                        price_floor=p_data.get('Floor_P_call', 0.0),
                        strategy_type=strategy_type.upper(),
                        maint_req=res.get("maint_req")
                    )
                )

    # Run Knapsack Solver
    optimal_allocations = optimize_allocation(portfolio, candidate_positions)

    # Post-process for final portfolio state and status outcomes
    for orig, opt in zip(candidate_positions, optimal_allocations):
        if opt.contracts > 0:
            opt.status = "FULL DEPLOYMENT" if opt.contracts == orig.contracts else f"SCALED DOWN from {orig.contracts}"
            portfolio.positions.append(opt)
            portfolio.accumulated_premiums += opt.premium_collected * opt.contracts * 100
        else:
            opt.status = "REJECTED"
            portfolio.positions.append(opt) # Add even if empty so consumer sees the rejection

    return portfolio


# =============================================================================
# SIMULATION SCANNER CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a shared portfolio margin allocator across multiple tickers.")
    parser.add_argument("tickers", nargs="*", help="List of tickers (e.g., SPMX QQQ AAPL)")
    parser.add_argument("--strategy", type=str.upper, choices=["CSP", "CC"], default="CSP", help="Strategy type: CSP or CC")
    parser.add_argument("--expiration", help="Expiration date (YYYY-MM-DD, partial string, or index)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers] if args.tickers else ["UUUU", "RKLB", "SPXL", "ASTS", "NVDA"]
    cash_equity = sum(LENDERS)

    print(f"\nWorking Capital: ${cash_equity:,.2f}")
    print(f"Strategy: {args.strategy}")
    print(f"\nScanning options chain for {len(tickers)} tickers...")

    portfolio = simulate_multi_asset_portfolio(
        tickers=tickers, 
        strategy_type=args.strategy, 
        expiration=args.expiration, 
        cash_equity=cash_equity
    )

    print("\nOptimal combination calculated successfully:")
    print("-" * 75)
    for p in portfolio.positions:
        if p.contracts > 0:
            print(f"  {p.ticker:<6} {p.contracts:>4}x ${p.strike:>6.2f} (notional ${p.notional:,.2f}) → {p.status}")

    portfolio.print_summary()
