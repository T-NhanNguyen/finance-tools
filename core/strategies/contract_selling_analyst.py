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
    CASH_W_DENSITY, CASH_W_FLOOR, CASH_W_EFF,
    WHEEL_W_EFF, WHEEL_W_DENSITY, WHEEL_W_FLOOR,
    VELOCITY_EXPANSION, VELOCITY_COMPRESSION, SKEW_ADJUSTMENT,
    TOP_N_PILLARS, INITIAL_MARGIN_REQ, MAINTENANCE_MARGIN_REQ,
    SAFETY_MARGIN_THRESHOLD
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
        atm_weekly_premium: float,
        strategy_type: str = "CSP"
    ) -> Dict:
        """
        Function 1: The 'Data Scientist'
        Performs the complete breakdown of a single contract scenario.
        """
        # 1. THE INTRINSIC/EXTRINSIC CORRECTION
        if strategy_type.upper() == "CSP":
            intrinsic_value = max(0, strike - underlying_price)
        else: # CC
            intrinsic_value = max(0, underlying_price - strike)
        extrinsic_premium = premium - intrinsic_value
        
        # 2. CAPITAL & POSITION CALCULATIONS
        shares_assigned = self.total_working_capital / strike if strike > 0 else 0
        contracts = shares_assigned / 100
        
        # 3. TRUE YIELD (Using Extrinsic Only)
        trade_roi_true = (extrinsic_premium / (strike * self.initial_req)) * 100 if strike > 0 else 0
        annual_cycles = 365 / days_to_expiry if days_to_expiry > 0 else 365
        eoy_projection_compounded = ((1 + (trade_roi_true/100))**annual_cycles - 1) * 100
        
        # 4. SAFETY & MARGIN CALL FLOOR (No absolute value)
        margin_call_floor = self.margin_loan / (shares_assigned * (1 - self.maintenance_req)) if shares_assigned > 0 else 0
        safety_margin_float = (underlying_price - strike) / underlying_price if underlying_price > 0 else 0
        safety_margin = safety_margin_float * 100
        
        # Determine Strategy Category
        strategy_tag = "Cash Engine" if safety_margin_float >= SAFETY_MARGIN_THRESHOLD else "Wheel Engine"
        
        # 5. STRUCTURAL & EFFICIENCY METRICS
        prem_yield = extrinsic_premium / underlying_price if underlying_price > 0 else 0
        efficiency_score = prem_yield / (max(0.001, safety_margin_float) if safety_margin_float > 0 else 0.001)
        
        structural_score = abs(gex_value * oi_value)
        eff_cost_basis = strike - premium
        
        # CapEff is now based purely on extrinsic yield vs normalized risk floor
        risk_divisor = max(SAFETY_MARGIN_THRESHOLD, safety_margin_float) if strategy_tag == "Cash Engine" else SAFETY_MARGIN_THRESHOLD
        capital_efficiency_ratio = trade_roi_true / risk_divisor

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
            "strategy_tag": strategy_tag,
            "premium": premium,
            "premium_extrinsic": round(extrinsic_premium, 2),
            "contracts": round(contracts, 2),
            "trade_roi_pct": round(trade_roi_true, 2),
            "eoy_projection_pct": round(eoy_projection_compounded, 2),
            "margin_call_floor": round(margin_call_floor, 2),
            "safety_margin_pct": round(safety_margin, 2),
            "structural_score": structural_score,
            "efficiency_score": round(efficiency_score, 4),
            "capital_efficiency_ratio": round(capital_efficiency_ratio, 4),
            "weeks_to_zero": round(weeks_to_zero, 1),
            "eff_cost_basis": round(eff_cost_basis, 2),
            "predicted_p_call": round(predicted_p_call, 2)
        }

    def get_actionable_pillars(self, analyzed_list: List[Dict], engine_mode: str = "BOTH") -> Dict[str, List[Dict]]:
        """
        Function 2: The 'Trader'
        Filters noise and ranks results into bifurcated Wheel & Cash mandates.
        """
        if not analyzed_list:
            return {"Top_Wheel_Engine": [], "Top_Cash_Engine": []}

        # 1. Normalize variables across the list
        max_density = max([x.get('structural_score', 0) for x in analyzed_list]) or 1.0
        max_cap_eff = max([x.get('capital_efficiency_ratio', 0) for x in analyzed_list]) or 1.0
        
        floors = [x.get('margin_call_floor', 0) for x in analyzed_list]
        max_floor = max(floors) if floors else 1.0
        min_floor = min(floors) if floors else 0.0
        floor_range = max_floor - min_floor if max_floor != min_floor else 1.0

        for x in analyzed_list:
            norm_density = x.get('structural_score', 0) / max_density
            norm_cap_eff = x.get('capital_efficiency_ratio', 0) / max_cap_eff
            norm_floor = (max_floor - x.get('margin_call_floor', 0)) / floor_range
            
            if x.get('strategy_tag') == "Cash Engine":
                x['blended_pillar_score'] = (norm_density * CASH_W_DENSITY) + (norm_floor * CASH_W_FLOOR) + (norm_cap_eff * CASH_W_EFF)
            else:
                x['blended_pillar_score'] = (norm_cap_eff * WHEEL_W_EFF) + (norm_density * WHEEL_W_DENSITY) + (norm_floor * WHEEL_W_FLOOR)

        # 2. Separate and Sort
        cash_strikes = [p for p in analyzed_list if p.get('strategy_tag') == "Cash Engine"]
        wheel_strikes = [p for p in analyzed_list if p.get('strategy_tag') == "Wheel Engine"]

        cash_ranked = sorted(cash_strikes, key=lambda x: x['blended_pillar_score'], reverse=True)
        wheel_ranked = sorted(wheel_strikes, key=lambda x: x['blended_pillar_score'], reverse=True)

        def format_output(sorted_list):
             pillars_scored = []
             for i, p in enumerate(sorted_list):
                  pillars_scored.append({
                       "Rank": i + 1,
                       "Strike": p['strike'],
                       "Strategy_Tag": p['strategy_tag'],
                       "Pillar_Score": round(p['blended_pillar_score'], 4),
                       "Pillar_Density": p['structural_score'],
                       "Floor_P_call": p['margin_call_floor'],
                       "Safety_Buffer": f"{p['safety_margin_pct']}%",
                       "Trade_ROI": f"{p['trade_roi_pct']}%",
                       "WTZ_Weeks": p['weeks_to_zero'],
                       "Cap_Efficiency": p.get('capital_efficiency_ratio', 0),
                       "Extrinsic_Premium": p.get('premium_extrinsic'),
                       "Total_Premium": round(p['premium'] * 100 * p['contracts'], 2),
                       "Eff_Cost_Basis": p['eff_cost_basis']
                  })
             return pillars_scored

        m_upper = engine_mode.upper()
        output_dict = {}
        
        if m_upper in ["BOTH", "SPLIT"]:
            output_dict["Top_Wheel_Engine"] = format_output(wheel_ranked[:3])
            output_dict["Top_Cash_Engine"] = format_output(cash_ranked[:3])
        elif m_upper == "WHEEL":
            output_dict["Top_Wheel_Engine"] = format_output(wheel_ranked[:3])
        elif m_upper == "CASH":
            output_dict["Top_Cash_Engine"] = format_output(cash_ranked[:3])
            
        return output_dict

    def scan(
        self, 
        ticker: str, 
        expiration_input: Optional[str] = None, 
        strategy_type: str = "CSP", 
        top_n_pillars: int = TOP_N_PILLARS,
        engine_mode: str = "BOTH"
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
                 atm_weekly_premium=atm_weekly_premium,
                 strategy_type=strategy_type
             )
             analyzed_results.append(res)
        
        pillars = self.get_actionable_pillars(analyzed_results, engine_mode=engine_mode)
        
        return {
             "ticker": ticker,
             "spot_price": spot_price,
             "strategy_type": strategy_type,
             "atm_premium_benchmark": atm_weekly_premium,
             "expiration": data.get("expiration"),
             "pillars": pillars,
             "all_strikes": analyzed_results
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan multiple tickers for option selling pillars.")
    parser.add_argument("tickers", nargs="*", help="List of tickers (e.g., SPY QQQ AAPL)")
    parser.add_argument("--strategy", choices=["CSP", "CC"], default="CSP", help="Strategy type: CSP or CC")
    parser.add_argument("--engine", choices=["BOTH", "CASH", "WHEEL"], default="BOTH", help="Engine filter mode")
    args = parser.parse_args()
    
    tickers = args.tickers if args.tickers else ["ASTS", "QQQ", "RKLB", "NBIS"]
    
    analyst = ContractSellingAnalyst(cash_equity=100000, margin_loan=80000)
    print(f"Working Capital: ${analyst.total_working_capital:,.2f} (${analyst.cash_equity/1000:.0f}k Cash + ${analyst.margin_loan/1000:.0f}k Margin)")
    print(f"Strategy: {args.strategy.upper()} | Engine Mode: {args.engine.upper()}")
    
    for t in tickers:
        try:
             res = analyst.scan(t.upper(), strategy_type=args.strategy, engine_mode=args.engine)
             if "error" in res:
                 print(f"\nScanning {t.upper()}... Error: {res['error']}")
                 continue
                 
             print(f"\n{'='*70}\nScanning {t.upper()} (Expiration Chain: {res.get('expiration')})\n{'='*70}")
             print(f"Spot: ${res['spot_price']:.2f} | Benchmark Put Premium: ${res['atm_premium_benchmark']:.2f}")
             print("-" * 70)
             
             for engine, p_list in res["pillars"].items():
                 print(f"\n[{engine.replace('_', ' ')}]")
                 for p in p_list:
                      print(f"  Rank {p['Rank']}: Strike ${p['Strike']:.2f} | Score: {p['Pillar_Score']:.4f} | WTZ: {p['WTZ_Weeks']} Weeks")
                      print(f"    -> Price Flow: [Cost Basis: ${p['Eff_Cost_Basis']:.2f}] <- [Margin: ${p['Floor_P_call']:.2f}]")
                      print(f"    -> Extrinsic: ${p.get('Extrinsic_Premium', 0):.2f} | Total Prem: ${p['Total_Premium']:,.2f} | CapEff: {p['Cap_Efficiency']:.4f}\n")
        except Exception as e:
             print(f"Unexpected Exception for {t}: {e}")