"""
CSP Math Engine

Contains mathematical formulas for calculating options selling strategies metrics.
Extracted from contract_selling_analyst.py
"""

from core.strategies.strategy_config import (
    FEDERAL_TAX_BRACKET, APPLIES_NIIT,
    LOAN_RATE, MIN_MONEYNESS_PCT,
    SAFETY_BUFFER_TARGET, MARGIN_REQS,
    USE_STATIC_CONSERVATIVE_MARGIN, DEFAULT_MARGIN_REQ
)

# =============================================================================
# CBOE/FINRA REG-T SHORT PUT MARGIN FORMULA
# Ref: CBOE strategy-based margin rules for naked short equity puts.
# Substitutes broker-specific rates from MARGIN_REQS for the standard 20%/10%.
# Default 20% (underlying component) and 10% (floor component) when ticker unknown.
# =============================================================================

REG_T_DEFAULT_SHORT = 0.20
REG_T_DEFAULT_FLOOR = 0.10


def calc_short_put_initial_margin_per_contract(
    underlying: float, strike: float, premium: float, ticker: str
) -> float:
    # Check strategy_config.py to change static margin requirement
    if USE_STATIC_CONSERVATIVE_MARGIN:
         return strike * DEFAULT_MARGIN_REQ * 100

    margin_info = MARGIN_REQS.get(ticker.upper(), {})
    rate_short = margin_info.get("initial_short", REG_T_DEFAULT_SHORT)
    rate_floor = margin_info.get("initial_long",  REG_T_DEFAULT_FLOOR)
    
    otm = max(underlying - strike, 0)
    leg1 = premium + rate_short * underlying - otm
    leg2 = premium + rate_floor * strike
    return max(leg1, leg2) * 100


def calc_short_put_maint_margin_per_contract(
    underlying: float, strike: float, option_market_value: float, ticker: str
) -> float:
    # Check strategy_config.py to change static margin requirement
    if USE_STATIC_CONSERVATIVE_MARGIN:
         return strike * DEFAULT_MARGIN_REQ * 100

    margin_info = MARGIN_REQS.get(ticker.upper(), {})
    rate_short = margin_info.get("maint_short", REG_T_DEFAULT_SHORT)
    rate_floor = margin_info.get("initial_long",  REG_T_DEFAULT_FLOOR)
    
    otm = max(underlying - strike, 0)
    leg1 = option_market_value + rate_short * underlying - otm
    leg2 = option_market_value + rate_floor * strike
    return max(leg1, leg2) * 100

def calculate_cash_requirement(
    effective_entry: float,
    init_req: float,
    maint_req: float
) -> float:
    """Cash required per share using safety-cushioned loan logic (used for floor/assignment math only)."""
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
    maint_req: float,
    ticker: str = ""
) -> dict:
    """Calculates detailed breakdown metrics for an option contract."""

    # Override requirements if global conservative toggle is active.
    if USE_STATIC_CONSERVATIVE_MARGIN:
        init_req = DEFAULT_MARGIN_REQ
        maint_req = DEFAULT_MARGIN_REQ

    # 1. THE INTRINSIC/EXTRINSIC CORRECTION
    if strategy_type.upper() == "CSP":
        intrinsic_value = max(0.0, strike - underlying_price)
    else: # CC
        intrinsic_value = max(0.0, underlying_price - strike)
    extrinsic_premium = premium - intrinsic_value

    # 2. CAPITAL & POSITION CALCULATIONS
    effective_entry = strike if strategy_type.upper() == "CSP" else underlying_price

    # Contract count uses the conservative cash_req (includes SAFETY_BUFFER_TARGET).
    # This ensures the scanner always suggests fewer contracts than the Reg-T hard ceiling.
    cash_req = calculate_cash_requirement(effective_entry, init_req, maint_req)
    contracts = int((cash_equity / cash_req) / 100) if cash_req > 0 else 0
    shares_assigned = contracts * 100

    # ROI and carry cost use the Reg-T formula margin as the denominator — the actual
    # capital locked per share by the broker.
    if ticker and strategy_type.upper() == "CSP":
        initial_margin_per_contract = calc_short_put_initial_margin_per_contract(
            underlying_price, strike, premium, ticker
        )
        maint_margin_per_contract = calc_short_put_maint_margin_per_contract(
            underlying_price, strike, premium, ticker
        )
    else:
        initial_margin_per_contract = init_req * effective_entry * 100
        maint_margin_per_contract   = maint_req * effective_entry * 100

    margin_per_share = initial_margin_per_contract / 100
    maint_per_share  = maint_margin_per_contract  / 100

    # 3. YIELD & CARRY SPREAD / TAX OVERLAYS
    effective_tax_rate = FEDERAL_TAX_BRACKET + (0.038 if APPLIES_NIIT else 0)

    wacc = LOAN_RATE
    holding_period_cost = margin_per_share * wacc * (days_to_expiry / 365)

    net_extrinsic_premium = extrinsic_premium - holding_period_cost

    trade_roi_true     = (extrinsic_premium     / margin_per_share) * 100 if margin_per_share > 0 else 0
    trade_roi_net      = (net_extrinsic_premium / margin_per_share) * 100 if margin_per_share > 0 else 0
    trade_roi_post_tax = trade_roi_net * (1 - effective_tax_rate)

    annual_cycles = 365 / days_to_expiry if days_to_expiry > 0 else 365
    eoy_projection_compounded = ((1 + (trade_roi_true / 100)) ** annual_cycles - 1) * 100

    # 4. SAFETY & MARGIN CALL FLOOR (Position-Specific)
    # Uses cash_req-based floor for consistency with the safety cushion model.
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
