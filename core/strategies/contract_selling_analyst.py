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

# API handlers used for scanning
from api.api_handlers import getGEXData


class ContractSellingAnalyst:
    """
    Analyzes contracts against collateral and structural floors.
    """
    def __init__(
        self, 
        cash_equity: float, 
        margin_loan: float, 
        initial_req: float = 0.20, 
        maintenance_req: float = 0.25
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
            velocity_factor = 1.25  # Expected 25% premium expansion
        elif structural_score > 1000:
            velocity_factor = 0.90  # Expected premium compression due to pinning

        skew_adjusted_base = atm_weekly_premium * 0.88
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
        Filters noise and ranks results combining density and capital efficiency.
        """
        sorted_list = sorted(
            analyzed_list,
            key=lambda x: (
                x.get('structural_score', 0),
                x.get('capital_efficiency_ratio', 0),
                -x.get('margin_call_floor', 999999)
            ),
            reverse=True
        )
        
        pillars_scored = []
        for i, p in enumerate(sorted_list):
             pillars_scored.append({
                  "Rank": i + 1,
                  "Strike": p['strike'],
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
        top_n_pillars: int = 5
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
             
        # Extract ATM strike
        sorted_strikes = sorted(strikes, key=lambda x: abs(x['strike'] - spot_price))
        atm_strike_data = sorted_strikes[0]
        atm_weekly_premium = (atm_strike_data['putBid'] + atm_strike_data['putAsk']) / 2
        
        if atm_weekly_premium <= 0:
             atm_weekly_premium = atm_strike_data['putBid'] or 1.0 
             
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