"""
Strategy Configuration Parameters
"""

# Cash Engine Weights (Prioritizes Safety and GEX Structure)
CASH_W_DENSITY = 0.50
CASH_W_FLOOR   = 0.30
CASH_W_EFF     = 0.20

# Wheel Engine Weights (Prioritizes Extrinsic Efficiency and Premium)
WHEEL_W_EFF     = 0.60
WHEEL_W_DENSITY = 0.20
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
SAFETY_MARGIN_THRESHOLD = 0.01  # 1% buffer to qualify as OTM Cash Engine vs ATM Wheel
