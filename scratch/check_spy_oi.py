import pandas as pd
from core.data.get_options_data import getOptionChain
ticker = 'SPY'
exp = '2026-05-15'
chain = getOptionChain(ticker, expiration=exp)
calls = chain.get('calls')
if calls is not None and not calls.empty:
    print(f"Max OI: {calls['openInterest'].max()}")
    print(f"Total OI Sum: {calls['openInterest'].sum()}")
else:
    print("No calls found")
