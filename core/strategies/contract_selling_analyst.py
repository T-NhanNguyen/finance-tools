"""
Contract Selling Analyst Strategy Module

Formulates a detailed breakdowns of option selling opportunities using
market makers structural hedging supports (GEX) and risk-mitigation ratios.
Contains:
1. Engine (Data Scientist): analyzes single strikes
2. Filter (Trader): finds high-density pillars
3. Runner (Scanner): connects pipeline to real-time data
"""

from typing import List, Dict, Optional
import pandas as pd

from api.api_handlers import getGEXData
from core.strategies.strategy_config import (
    W_DENSITY, W_EFF, W_FLOOR,
    VELOCITY_EXPANSION, VELOCITY_COMPRESSION, SKEW_ADJUSTMENT,
    TOP_N_PILLARS, INITIAL_MARGIN_REQ, MAINTENANCE_MARGIN_REQ
)


class ContractSellingAnalyst:
    """
    Analyzes contracts against collateral and structural floors.
    """


    def __init__(
        self, 
        cash_equity: float, 
        margin_loan: float, 
        initial_req: float = INITIAL_MARGIN_REQ, 
        maintenance_req: float = MAINTENANCE_MARGIN_REQ
    ):
        self.cash_equity = cash_equity
        self.margin_loan = margin_loan
        self.initial_req = initial_req
        self.maintenance_req = maintenance_req
        self.total_working_capital = cash_equity + margin_loan

    def analyze_strike(
        self,
        strike: float,
        premium: float,
        underlying_price: float,
        days_to_expiry: int,
        gex_value: float,
        oi_value: float,
        atm_weekly_premium: float
    ) -> Dict:
        """
        Function 1: The 'Data Scientist'
        Performs the complete breakdown of a single contract scenario.
        """
        # 1. CAPITAL & POSITION CALCULATIONS
        shares_assigned = self.total_working_capital / strike if strike > 0 else 0
        contracts = shares_assigned / 100
        
        # 2. YIELD & VELOCITY (ROI/EOY)
        trade_roi = (premium / (strike * self.initial_req)) * 100 if strike > 0 else 0
        annual_cycles = 365 / days_to_expiry if days_to_expiry > 0 else 365
        eoy_projection_compounded = ((1 + (trade_roi/100))**annual_cycles - 1) * 100
        
        # 3. SAFETY & MARGIN CALL FLOOR
        margin_call_floor = self.margin_loan / (shares_assigned * (1 - self.maintenance_req)) if shares_assigned > 0 else 0
        safety_margin_pct_float = abs(underlying_price - strike) / underlying_price if underlying_price > 0 else 0
        safety_margin = safety_margin_pct_float * 100
        
        # 4. STRUCTURAL & EFFICIENCY METRICS
        prem_yield = premium / underlying_price if underlying_price > 0 else 0
        # Original efficiency formula
        efficiency_score = prem_yield / (safety_margin_pct_float if safety_margin_pct_float > 0 else 0.001)
        
        structural_score = abs(gex_value * oi_value)
        eff_cost_basis = strike - premium
        
        # Capital Efficiency Ratio: ROI / Safety Margin
        capital_efficiency_ratio = trade_roi / (safety_margin if safety_margin > 0 else 0.01)

        # === Refined Repair Velocity Logic (CC Proxy) ===
        velocity_factor = 1.0
        if gex_value < 0:
            velocity_factor = VELOCITY_EXPANSION
        elif gex_value > 0:
            velocity_factor = VELOCITY_COMPRESSION

        skew_adjusted_base = atm_weekly_premium * SKEW_ADJUSTMENT
        predicted_p_call = skew_adjusted_base * velocity_factor

        weeks_to_zero = eff_cost_basis / (predicted_p_call if predicted_p_call > 0 else 1)

        return {
            "strike": strike,
            "premium": premium,
            "contracts": round(contracts, 2),
            "trade_roi_pct": round(trade_roi, 2),
            "eoy_projection_pct": round(eoy_projection_compounded, 2),
            "margin_call_floor": round(margin_call_floor, 2),
            "safety_margin_pct": round(safety_margin, 2),
            "structural_score": round(structural_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "capital_efficiency_ratio": round(capital_efficiency_ratio, 4),
            "weeks_to_zero": round(weeks_to_zero, 1),
            "eff_cost_basis": round(eff_cost_basis, 2),
            "predicted_p_call": round(predicted_p_call, 2)
        }

    def get_actionable_pillars(self, analyzed_list: List[Dict]) -> List[Dict]:
        """
        Function 2: The 'Trader'
        Filters noise and ranks results using a blended Pillar Score.
        """
        if not analyzed_list:
            return []
            
        # 1. Normalize variables across the list
        max_density = max([x.get('structural_score', 0) for x in analyzed_list]) or 1.0
        max_cap_eff = max([x.get('capital_efficiency_ratio', 0) for x in analyzed_list]) or 1.0
        
        floors = [x.get('margin_call_floor', 0) for x in analyzed_list]
        max_floor = max(floors) if floors else 1.0
        min_floor = min(floors) if floors else 0.0
        floor_range = max_floor - min_floor if max_floor != min_floor else 1.0

        # 2. Weights imported from strategy_config

        for x in analyzed_list:
            norm_density = x.get('structural_score', 0) / max_density
            norm_cap_eff = x.get('capital_efficiency_ratio', 0) / max_cap_eff
            # Normalize floor: Lower floor is better.
            norm_floor = (max_floor - x.get('margin_call_floor', 0)) / floor_range
            
            x['blended_pillar_score'] = (
                W_DENSITY * norm_density +
                W_EFF * norm_cap_eff +
                W_FLOOR * norm_floor
            )
            
        # 3. Sort by the blended "Pillar Score"
        sorted_list = sorted(analyzed_list, key=lambda x: x['blended_pillar_score'], reverse=True)
        
        pillars_scored = []
        for i, p in enumerate(sorted_list):
             pillars_scored.append({
                  "Rank": i + 1,
                  "Strike": p['strike'],
                  "Pillar_Score": round(p['blended_pillar_score'], 4),
                  "Pillar_Density": p['structural_score'],
                  "Floor_P_call": p['margin_call_floor'],
                  "Safety_Buffer": f"{p['safety_margin_pct']}%",
                  "Trade_ROI": f"{p['trade_roi_pct']}%",
                  "WTZ_Weeks": p['weeks_to_zero'],
                  "Cap_Efficiency": p.get('capital_efficiency_ratio', 0)
             })
        return pillars_scored

    def scan(
        self, 
        ticker: str, 
        expiration_input: Optional[str] = None, 
        strategy_type: str = "CSP", 
        top_n_pillars: int = TOP_N_PILLARS
    ) -> Dict:
        """
        Runs complete scanner pipeline.
        Calls GEX data endpoints and extracts actionable benchmarks.
        """
        data = getGEXData(ticker, expiration_input)
        if "error" in data:
            return data
            
        spot_price = data.get("spotPrice")
        strikes = data.get("strikes", [])
        if not strikes:
             return {"error": "No strikes returned by GEX data", "ticker": ticker}
             
        # Extract ATM Weekly Premium Benchmark (Always from nearest-term weekly)
        if expiration_input is None:
            nearest_data = data
        else:
            nearest_data = getGEXData(ticker, None)
            if "error" in nearest_data:
                nearest_data = data  # Fallback to current if nearest fails
                
        nearest_strikes = nearest_data.get("strikes", [])
        if nearest_strikes:
            # Sort by proximity to spot
            sorted_nearest = sorted(nearest_strikes, key=lambda x: abs(x['strike'] - spot_price))
            atm_strike_data = sorted_nearest[0]
            atm_weekly_premium = (atm_strike_data['putBid'] + atm_strike_data['putAsk']) / 2
        else:
            atm_weekly_premium = 0.0
            
        if atm_weekly_premium <= 0:
            # Absolute fallback from first available Put Bid in nearest chain
            if nearest_strikes:
                 atm_weekly_premium = nearest_strikes[0].get('putBid', 1.0) or 1.0
            else:
                 atm_weekly_premium = 1.0             
        analyzed_results = []
        for s_data in strikes:
             strike = s_data['strike']
             
             if strategy_type.upper() == "CSP":
                 premium = (s_data['putBid'] + s_data['putAsk']) / 2
                 if premium <= 0: continue
             else: # CC
                 premium = (s_data['callBid'] + s_data['callAsk']) / 2
                 if premium <= 0: continue
                 
             gex_raw = s_data['gexMillions'] * 1e6
             oi_raw = s_data['openInterestThousands'] * 1e3
             days_to_expiry = data.get("daysToExpiration", 30)
             
             res = self.analyze_strike(
                 strike=strike,
                 premium=premium,
                 underlying_price=spot_price,
                 days_to_expiry=days_to_expiry,
                 gex_value=gex_raw,
                 oi_value=oi_raw,
                 atm_weekly_premium=atm_weekly_premium
             )
             analyzed_results.append(res)
             
        pillars = self.get_actionable_pillars(analyzed_results)
        
        return {
             "ticker": ticker,
             "spot_price": spot_price,
             "strategy_type": strategy_type,
             "atm_premium_benchmark": atm_weekly_premium,
             "pillars": pillars[:top_n_pillars],
             "all_strikes": analyzed_results
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan multiple tickers for option selling pillars.")
    parser.add_argument("tickers", nargs="*", help="List of tickers (e.g., SPY QQQ AAPL)")
    args = parser.parse_args()
    
    tickers = args.tickers if args.tickers else ["ASTS", "QQQ", "RKLB", "NBIS"]
    
    analyst = ContractSellingAnalyst(cash_equity=100000, margin_loan=80000)
    for t in tickers:
        print(f"\n{'='*40}\nScanning {t.upper()}...\n{'='*40}")
        try:
             res = analyst.scan(t.upper())
             if "error" in res:
                 print(f"Error for {t}: {res['error']}")
                 continue
             print(f"Spot: ${res['spot_price']:.2f} | Benchmark Put Premium: ${res['atm_premium_benchmark']:.2f}")
             print("\nActionable Pillars (Top 5):")
             for p in res["pillars"]:
                 print(f"Rank {p['Rank']}: Strike {p['Strike']} | Score: {p['Pillar_Score']:.4f} | WTZ: {p['WTZ_Weeks']} Weeks | CapEff: {p['Cap_Efficiency']:.4f}")
        except Exception as e:
             print(f"Unexpected Exception for {t}: {e}")