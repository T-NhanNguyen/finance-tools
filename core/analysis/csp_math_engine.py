"""
CSP Math Engine

Contains mathematical formulas for calculating options selling strategies metrics.
Extracted from contract_selling_analyst.py
"""

from core.strategies.strategy_config import (
    FEDERAL_TAX_BRACKET, APPLIES_NIIT,
    LOAN_RATE, MARGIN_RATE, SAFETY_MARGIN_THRESHOLD
)

def calculate_option_metrics(
    strike: float,
    premium: float,
    underlying_price: float,
    days_to_expiry: int,
    gex_value: float,
    oi_value: float,
    strategy_type: str,
    total_working_capital: float,
    cash_equity: float,
    margin_loan: float,
    init_req: float,
    maint_req: float
) -> dict:
    """
    Calculates detailed breakdown metrics for an option contract.
    """
    
    # 1. THE INTRINSIC/EXTRINSIC CORRECTION
    if strategy_type.upper() == "CSP":
        intrinsic_value = max(0.0, strike - underlying_price)
    else: # CC
        intrinsic_value = max(0.0, underlying_price - strike)
    extrinsic_premium = premium - intrinsic_value
    
    # 2. CAPITAL & POSITION CALCULATIONS
    shares_assigned_float = total_working_capital / (strike * init_req) if strike > 0 and init_req > 0 else 0
    contracts = int(shares_assigned_float / 100)
    shares_assigned = contracts * 100
    
    # 3. YIELD & CARRY SPREAD / TAX OVERLAYS
    effective_tax_rate = FEDERAL_TAX_BRACKET + (0.038 if APPLIES_NIIT else 0)
    annual_capital_cost = (cash_equity * LOAN_RATE) + (margin_loan * MARGIN_RATE)
    holding_period_cost = annual_capital_cost * (days_to_expiry / 365)
    
    net_extrinsic_premium = extrinsic_premium - holding_period_cost
    
    trade_roi_true = (extrinsic_premium / (strike * init_req)) * 100 if strike > 0 else 0
    trade_roi_net = (net_extrinsic_premium / (strike * init_req)) * 100 if strike > 0 else 0
    trade_roi_post_tax = trade_roi_net * (1 - effective_tax_rate)
    
    annual_cycles = 365 / days_to_expiry if days_to_expiry > 0 else 365
    eoy_projection_compounded = ((1 + (trade_roi_true/100))**annual_cycles - 1) * 100
    
    # 4. SAFETY & MARGIN CALL FLOOR (No absolute value)
    margin_call_floor = margin_loan / (shares_assigned * (1 - maint_req)) if shares_assigned > 0 else 0
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

    return {
        "intrinsic_value": intrinsic_value,
        "extrinsic_premium": extrinsic_premium,
        "contracts": contracts,
        "shares_assigned": shares_assigned,
        "net_extrinsic_premium": net_extrinsic_premium,
        "trade_roi_true": trade_roi_true,
        "trade_roi_net": trade_roi_net,
        "trade_roi_post_tax": trade_roi_post_tax,
        "eoy_projection_compounded": eoy_projection_compounded,
        "margin_call_floor": margin_call_floor,
        "safety_margin_float": safety_margin_float,
        "safety_margin": safety_margin,
        "strategy_tag": strategy_tag,
        "efficiency_score": efficiency_score,
        "structural_score": structural_score,
        "eff_cost_basis": eff_cost_basis,
        "risk_divisor": risk_divisor,
        "capital_efficiency_ratio": capital_efficiency_ratio
    }
