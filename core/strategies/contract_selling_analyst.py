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

from core.data.bulk_data_loader import fetch_gex_single
from core.strategies.strategy_config import (
    CASH_W_DENSITY, CASH_W_FLOOR, CASH_W_EFF,
    WHEEL_W_EFF, WHEEL_W_DENSITY, WHEEL_W_FLOOR,
    VELOCITY_EXPANSION, VELOCITY_COMPRESSION, SKEW_ADJUSTMENT,
    TOP_N_PILLARS, INITIAL_MARGIN_REQ, MAINTENANCE_MARGIN_REQ,
    LENDERS, MARGIN_REQS, DEFAULT_MARGIN_REQ,
    MIN_MONEYNESS_PCT, WHEEL_MONEYNESS_MAX
)
from core.analysis.csp_math_engine import calculate_option_metrics


def _extract_atm_premium(strikes: List[Dict], spot_price: float, strategy_type: str) -> float:
    """Finds nearest-to-spot strike in a chain and returns its mid-market premium."""
    if not strikes:
        return 1.0
    sorted_strikes = sorted(strikes, key=lambda x: abs(x['strike'] - spot_price))
    atm = sorted_strikes[0]
    if strategy_type.upper() == "CSP":
        mid = (atm.get('putBid', 0) + atm.get('putAsk', 0)) / 2
        fallback_key = 'putBid'
    else:
        mid = (atm.get('callBid', 0) + atm.get('callAsk', 0)) / 2
        fallback_key = 'callBid'
    if mid > 0:
        return mid
    return atm.get(fallback_key, 1.0) or 1.0


class ContractSellingAnalyst:
    """
    Analyzes contracts against collateral and structural floors.
    """


    def __init__(
        self, 
        cash_equity: float, 
        initial_req: float = INITIAL_MARGIN_REQ, 
        maintenance_req: float = MAINTENANCE_MARGIN_REQ
    ):
        self.cash_equity = cash_equity
        self.initial_req = initial_req
        self.maintenance_req = maintenance_req
        self.total_working_capital = cash_equity

    def analyze_strike(
        self,
        strike: float,
        premium: float,
        underlying_price: float,
        days_to_expiry: int,
        gex_value: float,
        oi_value: float,
        atm_weekly_premium: float,
        strategy_type: str = "CSP",
        initial_req: Optional[float] = None,
        maintenance_req: Optional[float] = None,
        ticker: str = ""
    ) -> Dict:
        """
        Function 1: The 'Data Scientist'
        Performs the complete breakdown of a single contract scenario.
        """
        init_req = initial_req if initial_req is not None else self.initial_req
        maint_req = maintenance_req if maintenance_req is not None else self.maintenance_req

        metrics = calculate_option_metrics(
            strike=strike,
            premium=premium,
            underlying_price=underlying_price,
            days_to_expiry=days_to_expiry,
            gex_value=gex_value,
            oi_value=oi_value,
            strategy_type=strategy_type,
            total_working_capital=self.total_working_capital,
            cash_equity=self.cash_equity,
            init_req=init_req,
            maint_req=maint_req,
            ticker=ticker
        )
        
        # Unpack commonly used variables for backward compatibility
        extrinsic_premium = metrics["extrinsic_premium"]
        contracts = metrics["contracts"]
        shares_assigned = metrics["shares_assigned"]
        trade_roi_true = metrics["trade_roi_true"]
        trade_roi_net = metrics["trade_roi_net"]
        trade_roi_post_tax = metrics["trade_roi_post_tax"]
        eoy_projection_compounded = metrics["eoy_projection_compounded"]
        margin_call_floor = metrics["margin_call_floor"]
        safety_margin_float = metrics["safety_margin_float"]
        safety_margin = metrics["safety_margin"]
        strategy_tag = metrics["strategy_tag"]
        efficiency_score = metrics["efficiency_score"]
        structural_score = metrics["structural_score"]
        eff_cost_basis = metrics["eff_cost_basis"]
        capital_efficiency_ratio = metrics["capital_efficiency_ratio"]

        # === Refined Repair Velocity Logic (CC Proxy) ===
        velocity_factor = 1.0
        if gex_value < 0:
            velocity_factor = VELOCITY_EXPANSION
        elif gex_value > 0:
            velocity_factor = VELOCITY_COMPRESSION

        skew_adjusted_base = premium * SKEW_ADJUSTMENT
        predicted_p_call = skew_adjusted_base * velocity_factor

        weeks_to_zero = eff_cost_basis / (predicted_p_call if predicted_p_call > 0 else 1)

        return {
            "strike": strike,
            "strategy_tag": strategy_tag,
            "premium": premium,
            "premium_extrinsic": round(extrinsic_premium, 2),
            "contracts": contracts,
            "trade_roi_pct": round(trade_roi_true, 2),
            "trade_roi_net_pct": round(trade_roi_net, 2),
            "trade_roi_post_tax_pct": round(trade_roi_post_tax, 2),
            "eoy_projection_pct": round(eoy_projection_compounded, 2),
            "margin_call_floor": round(margin_call_floor, 2),
            "safety_margin_pct": round(safety_margin, 2),
            "structural_score": structural_score,
            "efficiency_score": round(efficiency_score, 4),
            "capital_efficiency_ratio": round(capital_efficiency_ratio, 4),
            "weeks_to_zero": round(weeks_to_zero, 1),
            "eff_cost_basis": round(eff_cost_basis, 2),
            "predicted_p_call": round(predicted_p_call, 2),
            "capital_deployed": round(strike * 100 * contracts, 2)
        }

    def get_actionable_pillars(self, analyzed_list: List[Dict], engine_mode: str = "BOTH") -> Dict[str, List[Dict]]:
        """
        Function 2: The 'Trader'
        Filters noise and ranks results into bifurcated Wheel & Cash mandates.
        Tiered classification based on moneyness.
        """
        if not analyzed_list:
            return {"Top_Wheel_Engine": [], "Top_Cash_Engine": []}

        # 0. Filter and Classify
        filtered_list = []
        for x in analyzed_list:
            smf = x.get('safety_margin_pct', 0) / 100.0
            if smf < MIN_MONEYNESS_PCT:
                continue
            elif smf <= WHEEL_MONEYNESS_MAX:
                x['strategy_tag'] = "Wheel Engine"
            else:
                x['strategy_tag'] = "Cash Engine"
            filtered_list.append(x)
            
        if not filtered_list:
            return {"Top_Wheel_Engine": [], "Top_Cash_Engine": []}
            
        analyzed_list = filtered_list

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

            # --- Structural Support Multiplier ---
            # Penalize strikes that have negligible GEX density (not a wall)
            # This ensures the optimizer prefers a 'Wall' even if ROI is slightly lower.
            if norm_density < 0.15:
                x['blended_pillar_score'] *= 0.25

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
                       "Net_ROI": f"{p['trade_roi_net_pct']}%",
                       "Post_Tax_ROI": f"{p['trade_roi_post_tax_pct']}%",
                       "WTZ_Weeks": p['weeks_to_zero'],
                       "Cap_Efficiency": p.get('capital_efficiency_ratio', 0),
                       "Extrinsic_Premium": p.get('premium_extrinsic'),
                       "Total_Premium": round(p['premium'] * 100 * p['contracts'], 2),
                       "Eff_Cost_Basis": p['eff_cost_basis'],
                       "Contracts": p.get('contracts', 0),
                       "Capital_Deployed": p.get('capital_deployed', 0),
                       "Premium_Raw": p.get('premium', 0)
                  })
             return pillars_scored

        m_upper = engine_mode.upper()
        output_dict = {}
        
        if m_upper in ["BOTH", "SPLIT"]:
            output_dict["Top_Wheel_Engine"] = format_output(wheel_ranked[:5])
            output_dict["Top_Cash_Engine"] = format_output(cash_ranked[:5])
        elif m_upper == "WHEEL":
            output_dict["Top_Wheel_Engine"] = format_output(wheel_ranked[:5])
        elif m_upper == "CASH":
            output_dict["Top_Cash_Engine"] = format_output(cash_ranked[:5])
            
        return output_dict

    def scan_from_chain(
        self, 
        chain_data: Dict, 
        strategy_type: str = "CSP", 
        engine_mode: str = "BOTH"
    ) -> Dict:
        """
        Runs analysis pipeline on a pre-loaded option chain.
        Skips network I/O.
        """
        ticker = chain_data.get("ticker", "UNKNOWN")
        spot_price = chain_data.get("spotPrice")
        strikes = chain_data.get("strikes", [])
        if not strikes:
             return {"error": "No strikes in chain data", "ticker": ticker}
             
        # Extract ATM Weekly Premium Benchmark from the chain
        atm_weekly_premium = _extract_atm_premium(strikes, spot_price, strategy_type)
        margin_info = MARGIN_REQS.get(ticker.upper(), MARGIN_REQS.get("DEFAULT", {}))
        init_req = margin_info.get("initial_short", DEFAULT_MARGIN_REQ)
        maint_req = margin_info.get("maint_short", DEFAULT_MARGIN_REQ)

        from core.analysis.csp_math_engine import calc_short_put_initial_margin_per_contract
        atm_init_margin = calc_short_put_initial_margin_per_contract(
            spot_price, spot_price, atm_weekly_premium, ticker
        )
        effective_capital = (self.cash_equity / atm_init_margin) * spot_price * 100 if atm_init_margin > 0 else self.total_working_capital
        effective_bp_pct = (atm_init_margin / (spot_price * 100)) * 100

        analyzed_results = []
        for s_data in strikes:
             strike = s_data['strike']
             
             if strategy_type.upper() == "CSP":
                 premium = (s_data.get('putBid', 0) + s_data.get('putAsk', 0)) / 2
             else: # CC
                 premium = (s_data.get('callBid', 0) + s_data.get('callAsk', 0)) / 2
                 
             if premium <= 0: continue
                 
             gex_raw = s_data.get('gexMillions', 0) * 1e6
             oi_raw = s_data.get('openInterestThousands', 0) * 1e3
             days_to_expiry = chain_data.get("daysToExpiration", 30)
             
             res = self.analyze_strike(
                 strike=strike,
                 premium=premium,
                 underlying_price=spot_price,
                 days_to_expiry=days_to_expiry,
                 gex_value=gex_raw,
                 oi_value=oi_raw,
                 atm_weekly_premium=atm_weekly_premium,
                 strategy_type=strategy_type,
                 initial_req=init_req,
                 maintenance_req=maint_req,
                 ticker=ticker
             )
             analyzed_results.append(res)
        
        pillars = self.get_actionable_pillars(analyzed_results, engine_mode=engine_mode)
        
        return {
             "ticker": ticker,
             "spot_price": spot_price,
             "strategy_type": strategy_type,
             "atm_premium_benchmark": atm_weekly_premium,
             "effective_capital": effective_capital,
             "effective_bp_pct": effective_bp_pct,
             "init_req": init_req,
             "maint_req": maint_req,
             "expiration": chain_data.get("expiration"),
             "pillars": pillars,
             "all_strikes": analyzed_results
        }

    def scan(
        self, 
        ticker: str, 
        expiration_input: Optional[str] = None, 
        strategy_type: str = "CSP", 
        top_n_pillars: int = TOP_N_PILLARS,
        engine_mode: str = "BOTH"
    ) -> Dict:
        """
        Runs complete scanner pipeline with synchronous fetch.
        """
        data = fetch_gex_single(ticker, expiration_input)
        if "error" in data:
            return data
            
        return self.scan_from_chain(data, strategy_type=strategy_type, engine_mode=engine_mode)


def run_scanner():
    import argparse
    parser = argparse.ArgumentParser(description="Scan multiple tickers for option selling pillars.")
    parser.add_argument("tickers", nargs="*", help="List of tickers (e.g., SPY QQQ AAPL)")
    parser.add_argument("--strategy", type=str.upper, choices=["CSP", "CC"], default="CSP", help="Strategy type: CSP or CC")
    parser.add_argument("--engine", type=str.upper, choices=["BOTH", "CASH", "WHEEL"], default="BOTH", help="Engine filter mode")
    parser.add_argument("--expiration", help="Expiration date (YYYY-MM-DD, partial string, or index)")
    args = parser.parse_args()
    
    parsed_tickers = []
    parsed_expiration = args.expiration
    for t in args.tickers:
        # If it looks like a date or numerical index, treat as expiration
        if (any(char.isdigit() for char in t) and any(sep in t for sep in ["/", "-", "."])) or (t.isdigit() and len(t) <= 3):
            parsed_expiration = t
        else:
            parsed_tickers.append(t)
            
    tickers = [t.upper() for t in parsed_tickers] if parsed_tickers else ["ASTS", "QQQ", "RKLB", "NBIS"]
    
    cash_equity = sum(LENDERS)
    analyst = ContractSellingAnalyst(cash_equity=cash_equity)
    print(f"Working Capital: ${analyst.total_working_capital:,.2f} (${analyst.cash_equity/1000:.0f}k Cash)")
    print(f"Strategy: {args.strategy.upper()} | Engine Mode: {args.engine.upper()}")
    
    from concurrent.futures import ThreadPoolExecutor
    def process_ticker(t):
        try:
             return t, analyst.scan(t.upper(), expiration_input=parsed_expiration, strategy_type=args.strategy, engine_mode=args.engine)
        except Exception as e:
             return t, {"error": str(e)}

    collected_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
         future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}
         for future in future_to_ticker:
              collected_results.append(future.result())

    for t, res in collected_results:
        try:
             if "error" in res:
                 print(f"\nScanning {t.upper()}... Error: {res['error']}")
                 continue
                 
             print(f"\n{'='*70}\nScanning {t.upper()} (Expiration Chain: {res.get('expiration')})\n{'='*70}")
             is_cc = res.get("strategy_type", "CSP").upper() == "CC"
             benchmark_label = "Call" if is_cc else "Put"
             print(f"Spot: ${res['spot_price']:.2f} | Benchmark {benchmark_label} Premium: ${res['atm_premium_benchmark']:.2f}")
             print(f"Effective Capital / BP: ${res['effective_capital']:,.2f} (Reg-T formula margin: {res.get('effective_bp_pct', 0):.1f}% of notional)")
             print("-" * 70)
             
             for engine, p_list in res["pillars"].items():
                 print(f"\n[{engine.replace('_', ' ')}]")
                 for p in p_list:
                      print(f"  Rank {p['Rank']}: Strike ${p['Strike']:.2f} | Score: {p['Pillar_Score']:.4f} | WTZ: {p['WTZ_Weeks']} Weeks")
                      if is_cc:
                          break_even = res['spot_price'] - p['Premium_Raw']
                          shares_capital = res['spot_price'] * 100 * p['Contracts']
                          print(f"    -> Price Flow: [Break-Even: ${break_even:.2f}] <- [Strike: ${p['Strike']:.2f}]")
                          print(f"    -> ROI: {p['Trade_ROI']} ({p['Post_Tax_ROI']} Net Post-Tax)")
                          print(f"    -> Premium: ${p['Premium_Raw']:.2f} | Total Prem: ${p['Total_Premium']:,.2f} ({p['Contracts']} Contracts)")
                          print(f"    -> Capital Deployed (Shares): ${shares_capital:,.2f} | CapEff: {p['Cap_Efficiency']:.4f}\n")
                      else:
                          print(f"    -> Price Flow: [Cost Basis: ${p['Eff_Cost_Basis']:.2f}] <- [Margin Price floor: ${p['Floor_P_call']:.2f}]")
                          print(f"    -> ROI: {p['Trade_ROI']} ({p['Post_Tax_ROI']} Net Post-Tax)")
                          print(f"    -> Premium: ${p['Premium_Raw']:.2f} | Total Prem: ${p['Total_Premium']:,.2f} ({p['Contracts']} Contracts)")
                          print(f"    -> Capital Deployed: ${p['Capital_Deployed']:,.2f} | CapEff: {p['Cap_Efficiency']:.4f}\n")
        except Exception as e:
             print(f"Unexpected Exception for {t}: {e}")

if __name__ == "__main__":
    run_scanner()