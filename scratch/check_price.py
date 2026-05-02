import yfinance as yf
from core.data.get_stock_price import getCurrentPrice
print(f"Current Price AMZN: {getCurrentPrice('AMZN')}")
