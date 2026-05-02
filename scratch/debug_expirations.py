import sys
import os
sys.path.append('/app')
from core.data.bulk_data_loader import fetch_gex_all_expirations
from core.strategies.strategy_config import SCENARIOS

ticker = 'SPY'
data = fetch_gex_all_expirations(ticker)
print(f"Ticker: {data.get('ticker')}")
print(f"Spot Price: {data.get('spotPrice')}")
expirations = data.get('expirations', [])
print(f"Total expirations found: {len(expirations)}")

scenario = SCENARIOS['bullish_3month']
target_days = scenario.time_horizon_days
tolerance = max(7, int(target_days * 0.25))
min_days = max(1, target_days - tolerance)
max_days = target_days + (tolerance * 3)

print(f"Scenario: {scenario.name}")
print(f"Target Days: {target_days}")
print(f"Min Days: {min_days}, Max Days: {max_days}")
print(f"Min OI: {scenario.min_open_interest}")

for exp in expirations:
    dte = exp.get('daysToExpiration')
    oi = exp.get('totalOI')
    match_dte = min_days <= dte <= max_days
    match_oi = oi >= scenario.min_open_interest
    if match_dte or dte > 0: # Print everything with DTE > 0 to see what's available
        status = "MATCH" if (match_dte and match_oi) else "MISS"
        print(f"  Exp: {exp['expiration']} | DTE: {dte:3} | OI: {oi:8} | Status: {status}")
