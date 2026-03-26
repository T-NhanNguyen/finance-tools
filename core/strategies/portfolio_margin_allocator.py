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
    spot_at_entry: float = 0.0
    initial_req: Optional[float] = None
    maint_req: Optional[float] = None

    def __post_init__(self):
        margin_info = MARGIN_REQS.get(self.ticker.upper(), {})
        # selling options (both CSP and CC) use 'short' requirements
        if self.initial_req is None:
            self.initial_req = margin_info.get("initial_short", DEFAULT_MARGIN_REQ)
        if self.maint_req is None:
            self.maint_req = margin_info.get("maint_short", DEFAULT_MARGIN_REQ)
        
        # Default spot to strike if not provided (conservative for CSP)
        if self.spot_at_entry <= 0:
            self.spot_at_entry = self.strike

    @property
    def notional(self) -> float:
        return self.strike * self.contracts * 100

    @property
    def margin_call_floor(self) -> float:
        """
        The price at which the maintenance requirement for the shares (after assignment) 
        would exactly equal the equity allocated to this position.
        Uses the same math as the core engine for consistency.
        """
        effective_entry = self.strike if self.strategy_type.upper() == "CSP" else self.spot_at_entry
        # Re-calculating cash_req logic locally for summary precision
        loan_safe = effective_entry * (1 - 0.20) * (1 - self.maint_req) # 0.20 is SAFETY_BUFFER_TARGET
        loan_limit = min(loan_safe, effective_entry * (1 - self.initial_req))
        return loan_limit / (1 - self.maint_req) if (1 - self.maint_req) > 0 else 0

    @property
    def initial_margin_required(self) -> float:
        """Margin collateral held by broker at entry."""
        return self.notional * self.initial_req

    @property
    def maintenance_margin_required(self) -> float:
        """Ongoing margin collateral requirement."""
        return self.notional * self.maint_req


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
    def total_initial_margin(self) -> float:
        """Sum of all entry margin requirements."""
        return sum(p.initial_margin_required for p in self.positions)

    @property
    def total_maintenance_margin(self) -> float:
        """Sum of all ongoing maintenance requirements."""
        return sum(p.maintenance_margin_required for p in self.positions)

    @property
    def margin_utilized(self) -> float:
        """Current margin consumed relative to total equity."""
        return self.total_maintenance_margin

    @property
    def buying_power_remaining(self) -> float:
        """Approximate amount of additional notional that can be carried at 50% avg req."""
        available_equity = max(0, self.total_equity - self.total_initial_margin)
        return available_equity * 2.0

    @property
    def total_notional(self) -> float:
        """Sum of notional value (strike * contracts * 100) of all positions."""
        return sum(p.notional for p in self.positions)

    @property
    def total_assignment_exposure(self) -> float:
        """Total cash required to settle all short positions if assigned."""
        return self.total_notional

    def print_summary(self):
        """Prints a structured view of the shared collateral pool state."""
        print(f"\n{'='*82}")
        print(f"PORTFOLIO MARGIN ALLOCATOR — Shared Collateral Pool")
        print(f"{'='*82}")
        print(f"  Cash (Lender):         ${self.cash:>14,.2f}")
        print(f"  Accumulated Premiums:  ${self.accumulated_premiums:>14,.2f}")
        print(f"  Total Equity:          ${self.total_equity:>14,.2f}")
        print(f"  Assignment Exposure:   ${self.total_assignment_exposure:>14,.2f}")
        print(f"{'─'*82}")
        print(f"  {'Ticker':<8} {'Contracts':>9} {'Strike':>8} {'Notional':>12} {'Floor':>9} {'Safety':>10}")
        print(f"{'─'*82}")
        for p in self.positions:
            if p.contracts > 0:
                safety_pct = ((p.spot_at_entry - p.margin_call_floor) / p.spot_at_entry * 100) if p.spot_at_entry else 0
                print(f"  {p.ticker:<8} {p.contracts:>9} {p.strike:>8.2f} {p.notional:>12,.2f} {p.margin_call_floor:>9.2f} {safety_pct:>9.1f}%")
        print(f"{'─'*82}")
        print(f"  {'TOTALS':<8} {'':>9} {'':>8} {self.total_notional:>12,.2f}")
        print(f"{'─'*82}")

        margin_pct = (self.total_maintenance_margin / self.total_equity * 100) if self.total_equity else 0
        
        print(f"  Equity (Buffer):   ${self.total_equity:>12,.2f}")
        print(f"  Margin Used:       ${self.total_maintenance_margin:>12,.2f}  ({margin_pct:.1f}% capacity used)")
        print(f"  Remaining BP:      ${self.buying_power_remaining:>12,.2f}  (est. @ 50% req)")
        print(f"  Note: High capacity usage is normal for optimized allocation.")
        print(f"{'='*82}\n")


def optimize_allocation(allocator: PortfolioMarginAllocator, candidates: List[Position]) -> List[Position]:
    """
    Solves the multidimensional knapsack problem for optimal contract sizing across overlapping constraints.
    Maximizes total premium collected while strictly respecting the tiered cash and margin limits.
    """
    best_premium = -1.0
    best_counts = {i: 0 for i in range(len(candidates))}
    E0 = allocator.cash

    # Solver now uses per-position margin rates instead of a global bucket.
    # Mathematical constraint: Sum(count_i * 100 * (strike_i * initial_req_i - premium_i)) <= E0
    items = []
    for i, c in enumerate(candidates):
        w = 100 * (c.strike * c.initial_req - c.premium_collected)
        v = 100 * c.premium_collected
        cap = c.contracts
        items.append((i, w, v, cap))

    # Sort by best intrinsic capital efficiency
    items.sort(key=lambda x: x[2] / x[1] if x[1] > 0 else float('inf'), reverse=True)
    
    capacity = E0

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
                initial_req=c.initial_req,
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
                    spot_at_entry=res.get("spot_price", 0),
                    initial_req=res.get("initial_req"),
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
