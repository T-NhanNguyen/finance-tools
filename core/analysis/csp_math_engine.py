"""
CSP Math Engine

Contains mathematical formulas for calculating options selling strategies metrics.
Extracted from contract_selling_analyst.py
"""

from core.strategies.strategy_config import (
    FEDERAL_TAX_BRACKET, APPLIES_NIIT,
    LOAN_RATE, MIN_MONEYNESS_PCT,
    SAFETY_BUFFER_TARGET
)

def calculate_cash_requirement(
    effective_entry: float,
    init_req: float,
    maint_req: float
) -> float:
    """
    Calculates the cash requirement per share with optimal safety cushions.
    """
    loan_safe = effective_entry * (1 - SAFETY_BUFFER_TARGET) * (1 - maint_req)
    loan_limit = min(loan_safe, effective_entry * (1 - init_req))
    return effective_entry - loan_limit


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
    effective_entry = strike if strategy_type.upper() == "CSP" else underlying_price
    cash_req = calculate_cash_requirement(effective_entry, init_req, maint_req)
    shares_assigned_float = cash_equity / cash_req if cash_req > 0 else total_working_capital / strike
    contracts = int(shares_assigned_float / 100)
    shares_assigned = contracts * 100
    
    # 3. YIELD & CARRY SPREAD / TAX OVERLAYS
    effective_tax_rate = FEDERAL_TAX_BRACKET + (0.038 if APPLIES_NIIT else 0)
    
    # Weighted Average Cost of Capital (WACC) to allocate carry costs proportionally
    wacc = LOAN_RATE
    holding_period_cost = (strike * init_req) * wacc * (days_to_expiry / 365)
    
    net_extrinsic_premium = extrinsic_premium - holding_period_cost
    
    trade_roi_true = (extrinsic_premium / (strike * init_req)) * 100 if strike > 0 else 0
    trade_roi_net = (net_extrinsic_premium / (strike * init_req)) * 100 if strike > 0 else 0
    trade_roi_post_tax = trade_roi_net * (1 - effective_tax_rate)
    
    annual_cycles = 365 / days_to_expiry if days_to_expiry > 0 else 365
    eoy_projection_compounded = ((1 + (trade_roi_true/100))**annual_cycles - 1) * 100
    
    # 4. SAFETY & MARGIN CALL FLOOR (Position-Specific)
    margin_call_floor = (effective_entry - cash_req) / (1 - maint_req) if (1 - maint_req) > 0 else 0
    safety_margin_float = (underlying_price - strike) / underlying_price if underlying_price > 0 else 0
    safety_margin = safety_margin_float * 100
    
    # Strategy Category is determined in get_actionable_pillars
    strategy_tag = None
    
    # 5. STRUCTURAL & EFFICIENCY METRICS
    prem_yield = extrinsic_premium / underlying_price if underlying_price > 0 else 0
    efficiency_score = prem_yield / (max(0.001, safety_margin_float) if safety_margin_float > 0 else 0.001)
    
    structural_score = abs(gex_value * oi_value)
    eff_cost_basis = strike - premium
    
    # CapEff is now based purely on extrinsic yield vs normalized risk floor
    risk_divisor = max(MIN_MONEYNESS_PCT, safety_margin_float)
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
