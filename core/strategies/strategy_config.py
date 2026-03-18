"""
Strategy Configuration Parameters
"""

# Weights for blending Actionable Pillar Scores (Sum should equal 1.0)
W_DENSITY = 0.40
W_EFF     = 0.35
W_FLOOR   = 0.25

# Repair Velocity Factors (CC Proxy)
VELOCITY_EXPANSION = 1.25   # if GEX < 0
VELOCITY_COMPRESSION = 0.90  # if GEX > 0
SKEW_ADJUSTMENT = 0.88      # safety factor overlay on ATM benchmark

# Runner & Scanner Defaults
TOP_N_PILLARS = 5
INITIAL_MARGIN_REQ = 0.20
MAINTENANCE_MARGIN_REQ = 0.25
