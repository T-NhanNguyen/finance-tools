import pandas as pd
from core.data.get_options_data import getOptionChain
ticker = 'SPY'
exp = '2026-06-18'
chain = getOptionChain(ticker, expiration=exp)
print(f"Strikes found: {len(chain.get('calls', []))}")
if not chain.get('calls', pd.DataFrame()).empty:
    print(f"Min Strike: {chain['calls']['strike'].min()}")
    print(f"Max Strike: {chain['calls']['strike'].max()}")
