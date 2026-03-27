"""
Strategy Configuration Parameters
"""

# config.py
"""
Hardcoded scenario variables, lender profiles, and tax configurations.
Edit this file to model different scenarios without touching the core logic.

All values in this file are for the DEFAULT scenario. Override them in main.py
as needed for alternative scenarios.
"""

# ===========================================================
# LENDER PROFILES
# Loan amounts from individual lenders (in USD).
# The sum becomes your total working capital (P_0).
# ===========================================================
LENDERS = [
    100_000,   # Lender 1
]

# Annual interest rate on all loans (flat rate applied to total principal).
# e.g. 0.06 = 6% annual interest
LOAN_RATE = 0.06

# Broker margin terms
MARGIN_RATE = 0.08  # 8% Annual margin interest

# Month (k) at which the total interest obligation is paid out to lenders.
# e.g. 12 = pay at end of year
INTEREST_MONTH_K = 1

# Total duration of the strategy in months.
MONTHS_T = 3


# ===========================================================
# STRATEGY ENGINE
# ===========================================================

# Target quarterly return (every 3 months).
# e.g. 0.20 = 20% every 3 months
TARGET_QUARTERLY_RETURN = 0.20

# If True, use compound (geometric) monthly rate: r_m = (1 + R_3m)^(1/3) - 1
# If False, use simple monthly rate:               r_m = R_3m / 3
COMPOUND_GAINS = True


# ===========================================================
# TAX PROFILE  (Washington State — No State Income Tax)
# ===========================================================

# Federal income tax bracket.
# Common brackets: 0.22, 0.24, 0.32, 0.35, 0.37
FEDERAL_TAX_BRACKET = 0.24

# Net Investment Income Tax — applies at 3.8% if income exceeds IRS thresholds.
# True = add 3.8%; False = exempt
APPLIES_NIIT = False


# ===========================================================
# OPTIONS SCANNER
# ===========================================================

# Comprehensive broker margin requirements per ticker
MARGIN_REQS = {
    "MP": {"initial_long": 0.2806, "maint_long": 0.2551, "initial_short": 0.30, "maint_short": 0.30},
    "UUUU": {"initial_long": 0.3012, "maint_long": 0.2739, "initial_short": 0.3012, "maint_short": 0.30},
    "ASTS": {"initial_long": 0.4319, "maint_long": 0.3927, "initial_short": 0.4319, "maint_short": 0.3927},
    "TSLA": {"initial_long": 0.3653, "maint_long": 0.3200, "initial_short": 0.30, "maint_short": 0.30},
    "SPXL": {"initial_long": 0.75, "maint_long": 0.75, "initial_short": 0.90, "maint_short": 0.90},
    "NVDA": {"initial_long": 0.25, "maint_long": 0.25, "initial_short": 0.30, "maint_short": 0.30},
    "NBIS": {"initial_long": 0.3162, "maint_long": 0.2875, "initial_short": 0.3162, "maint_short": 0.30},
    "NEBX": {"initial_long": 0.30, "maint_long": 0.30, "initial_short": 0.30, "maint_short": 0.30},
    "IREN": {"initial_long": 0.3962, "maint_long": 0.3602, "initial_short": 0.3962, "maint_short": 0.3602},
    "BE": {"initial_long": 0.3650, "maint_long": 0.3318, "initial_short": 0.3650, "maint_short": 0.3318},
    "WULF": {"initial_long": 0.4202, "maint_long": 0.3822, "initial_short": 0.4202, "maint_short": 0.3822},
    "AAOI": {"initial_long": 0.7167, "maint_long": 0.4809, "initial_short": 0.5289, "maint_short": 0.4809},
    "PLTR": {"initial_long": 0.2500, "maint_long": 0.2500, "initial_short": 0.3000, "maint_short": 0.3000},
    "ONDS": {"initial_long": 0.4509, "maint_long": 0.4099, "initial_short": 0.4970, "maint_short": 0.4970},
    "BWXT": {"initial_long": 0.2500, "maint_long": 0.2500, "initial_short": 0.3000, "maint_short": 0.3000},
    "NOK": {"initial_long": 0.2500, "maint_long": 0.2500, "initial_short": 0.6266, "maint_short": 0.6266},
    "SOFI": {"initial_long": 0.2500, "maint_long": 0.2500, "initial_short": 0.3000, "maint_short": 0.3000},
    "HIMS": {"initial_long": 0.3526, "maint_long": 0.3206, "initial_short": 0.3526, "maint_short": 0.3206},
    "RKLB": {"initial_long": 0.3282, "maint_long": 0.2984, "initial_short": 0.3282, "maint_short": 0.3000},
    "RKLB": {"initial_long": 0.2500, "maint_long": 0.2500, "initial_short": 0.3000, "maint_short": 0.3000},
    "AAPL": {"initial_long": 0.2500, "maint_long": 0.2500, "initial_short": 0.3000, "maint_short": 0.3000}
}
# Default margin requirement if a ticker is missing from the dictionary
DEFAULT_MARGIN_REQ = 0.50

# Optimal Margin Sizing Safety Cushion target distance from entry to liquidation (e.g., 0.15 = 15%)
SAFETY_BUFFER_TARGET = 0.20

# Minimum premium-to-underlying yield required to qualify a trade (the "2% rule").
# e.g. 0.02 = 2%
MIN_YIELD_THRESHOLD = 0.02


# Cash Engine Weights (Prioritizes Structural Safety/GEX Walls)
CASH_W_DENSITY = 0.70
CASH_W_FLOOR   = 0.20
CASH_W_EFF     = 0.10

# Wheel Engine Weights (Prioritizes structural support alongside premium)
WHEEL_W_EFF     = 0.40
WHEEL_W_DENSITY = 0.40
WHEEL_W_FLOOR   = 0.20

# Repair Velocity Factors (CC Proxy)
VELOCITY_EXPANSION = 1.25   # if GEX < 0
VELOCITY_COMPRESSION = 0.90  # if GEX > 0
SKEW_ADJUSTMENT = 0.88      # safety factor overlay on ATM benchmark

# Runner & Scanner Defaults
TOP_N_PILLARS = 5
INITIAL_MARGIN_REQ = 0.20
MAINTENANCE_MARGIN_REQ = 0.25

# Safety & Moneyness thresholds
MIN_MONEYNESS_PCT = 0.02  # 2% floor to qualify for any engine (eliminates ATM pin risk)
WHEEL_MONEYNESS_MAX = 0.05  # 5% ceiling for Wheel Engine, above is Cash Engine (separates income trading from wheel preparation)
