"""
Portfolio Margin Allocator

Models a single shared collateral pool across multiple open CSP/CC positions.
Implements a 0-1 Knapsack (Branch and Bound) optimizer to find the perfect contract sizes
across different tickers to maximize premium without breaching cash or margin limits.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional
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
    strategy_type: str = "CSP"  # CSP or CC
    maint_req: Optional[float] = None

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
        if not self.positions:
            return DEFAULT_MARGIN_REQ
        return max(p.maint_req for p in self.positions)

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
        print(f"  {'Ticker':<8} {'Contracts':>9} {'Strike':>8} {'Notional':>12} {'Premium':>11}")
        print(f"{'─'*69}")
        for p in self.positions:
            if p.contracts > 0:
                total_prem = p.contracts * p.premium_collected * 100
                print(f"  {p.ticker:<8} {p.contracts:>9} {p.strike:>8.2f} {p.notional:>12,.2f} {total_prem:>11,.2f}")
        print(f"{'─'*69}")
        print(f"  {'TOTALS':<8} {'':>9} {'':>8} {self.total_notional:>12,.2f} {self.accumulated_premiums:>11,.2f}")
        print(f"{'─'*69}")

        cash_pct = (self.cash_utilized / self.total_equity * 100) if self.total_equity else 0
        margin_pct = (self.margin_utilized / self.margin_limit * 100) if self.margin_limit else 0
        
        print(f"  Cash Layer:   ${self.cash_utilized:>12,.2f} / ${self.total_equity:,.2f}  ({cash_pct:.1f}% used)")
        print(f"  Margin Layer: ${self.margin_utilized:>12,.2f} / ${self.margin_limit:,.2f}  ({margin_pct:.1f}% used)")
        print(f"{'='*65}\n")


def optimize_allocation(allocator: PortfolioMarginAllocator, candidates: List[Position]) -> List[Position]:
    """
    Solves the multidimensional knapsack problem for optimal contract sizing across overlapping constraints.
    Maximizes total premium collected while strictly respecting the tiered cash and margin limits.
    """
    best_premium = -1.0
    best_counts = {i: 0 for i in range(len(candidates))}
    E0 = allocator.cash

    # Group by potential binding limits: we must test every candidate's maint_req acting as the ceiling
    unique_reqs = sorted(list(set(c.maint_req for c in candidates)))

    for M in unique_reqs:
        # If M is the tightest constraint, we can only safely mix in tickers with req <= M
        valid_indices = [i for i, c in enumerate(candidates) if c.maint_req <= M]
        if not valid_indices:
            continue

        items = []
        for i in valid_indices:
            c = candidates[i]
            # Formulated mathematically from: Sum(notional) <= (E0 + Sum(Premium)) / M
            w = 100 * (c.strike - c.premium_collected / M)
            v = 100 * c.premium_collected
            cap = c.contracts
            items.append((i, w, v, cap))

        # Sort by best intrinsic capital efficiency for high-speed branch pruning
        items.sort(key=lambda x: x[2] / x[1] if x[1] > 0 else float('inf'), reverse=True)
        
        capacity = E0 / M

        # DFS Branch & Bound search
        def dfs(item_idx: int, current_w: float, current_v: float, counts: dict):
            nonlocal best_premium, best_counts
            if item_idx == len(items):
                if current_v > best_premium:
                    best_premium = current_v
                    for i in range(len(candidates)):
                        best_counts[i] = counts.get(i, 0)
                return

            orig_i, w, v, cap = items[item_idx]

            # Relaxation Bound (best case projection)
            rem_cap = capacity - current_w
            ub = current_v
            for next_idx in range(item_idx, len(items)):
                _, nw, nv, ncap = items[next_idx]
                if nw <= 0:
                    ub += ncap * nv
                else:
                    take = min(ncap, rem_cap / nw)
                    if take > 0:
                        ub += take * nv
                        rem_cap -= take * nw
                    if rem_cap <= 0:
                        break

            if ub <= best_premium:
                return  # Prune search branch

            # Search down from max feasible size
            max_take = cap
            if w > 0:
                max_take = min(max_take, int((capacity - current_w) / w))

            for take in range(max_take, -1, -1):
                counts[orig_i] = take
                dfs(item_idx + 1, current_w + take * w, current_v + take * v, counts)
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
                strategy_type=c.strategy_type,
                maint_req=c.maint_req
            )
        )
    return optimal_positions


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
    portfolio = PortfolioMarginAllocator(cash=cash_equity)
    analyst = ContractSellingAnalyst(cash_equity=cash_equity)

    print(f"\nWorking Capital: ${cash_equity:,.2f}")
    print(f"Strategy: {args.strategy}")
    print(f"\nScanning options chain for {len(tickers)} tickers to find optimal multi-asset allocation...")

    def process_ticker(t):
        try:
            return t, analyst.scan(t, expiration_input=args.expiration, strategy_type=args.strategy, engine_mode="BOTH")
        except Exception as e:
            return t, {"error": str(e)}

    collected_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
         future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}
         for future in future_to_ticker:
              collected_results.append(future.result())

    candidate_positions = []
    for t, res in collected_results:
        if "error" in res:
            print(f"  [ERROR] {t}: {res['error']}")
            continue
            
        pillars = res.get("pillars", {})
        candidate_pillar = None
        for eng_key in ["Top_Wheel_Engine", "Top_Cash_Engine"]:
            if eng_key in pillars and pillars[eng_key]:
                candidate_pillar = pillars[eng_key][0]
                break
                 
        if candidate_pillar:
            candidate_positions.append(
                Position(
                    ticker=t,
                    strike=candidate_pillar['Strike'],
                    contracts=candidate_pillar['Contracts'],
                    premium_collected=candidate_pillar['Premium_Raw'],
                    strategy_type=args.strategy,
                    maint_req=res.get("maint_req")
                )
            )

    # Run DP optimal fit solver
    optimal_allocations = optimize_allocation(portfolio, candidate_positions)

    print("\nOptimal combination calculated successfully:")
    print("-" * 75)
    for orig, opt in zip(candidate_positions, optimal_allocations):
        if opt.contracts > 0:
            status = "FULL DEPLOYMENT" if opt.contracts == orig.contracts else f"SCALED DOWN from {orig.contracts}"
            print(f"  {opt.ticker:<6} {opt.contracts:>4}x ${opt.strike:>6.2f} (notional ${opt.notional:,.2f}) → {status}")
        else:
            print(f"  {opt.ticker:<6}    0x ${opt.strike:>6.2f} → REJECTED — Suboptimal for structural capacity")

        if opt.contracts > 0:
            portfolio.positions.append(opt)
            portfolio.accumulated_premiums += opt.premium_collected * opt.contracts * 100

    portfolio.print_summary()
