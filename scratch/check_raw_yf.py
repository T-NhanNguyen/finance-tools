import yfinance as yf
import pandas as pd
ticker = 'SPY'
stock = yf.Ticker(ticker)
exp = stock.options[10]
print(f"Checking SPY expiration: {exp}")
chain = stock.option_chain(exp)
calls = chain.calls
print(f"Calls found: {len(calls)}")
print("Sample data:")
print(calls[['strike', 'openInterest', 'volume']].head(10))
print(f"Total Call OI: {calls['openInterest'].sum()}")
print(f"Total Call Volume: {calls['volume'].sum()}")
