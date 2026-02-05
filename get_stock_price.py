"""
Stock Price Retrieval Module

This module provides efficient stock price retrieval with support for bulk queries
to minimize API calls and optimize performance.
"""

import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from enum import Enum


class PriceInterval(Enum):
    """Valid intervals for stock price data"""
    ONE_MINUTE = "1m"
    TWO_MINUTES = "2m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    SIXTY_MINUTES = "60m"
    NINETY_MINUTES = "90m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"


class PricePeriod(Enum):
    """Valid periods for historical data"""
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    YEAR_TO_DATE = "ytd"
    MAX = "max"


def getCurrentPrice(ticker: str) -> Optional[float]:
    """
    Get the current price for a single stock ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        Current price as float, or None if retrieval fails
    """
    try:
        stock = yf.Ticker(ticker)
        currentPrice = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice')
        return currentPrice
    except Exception as error:
        print(f"Error fetching current price for {ticker}: {error}")
        return None


def getCurrentPricesBulk(tickers: List[str]) -> Dict[str, Optional[float]]:
    """
    Get current prices for multiple tickers in a single bulk operation.
    This is significantly more efficient than calling getCurrentPrice() multiple times.
    
    Args:
        tickers: List of stock symbols
        
    Returns:
        Dictionary mapping ticker symbols to their current prices
    """
    priceMap = {}
    
    try:
        # Download data for all tickers at once - much more efficient
        tickerString = " ".join(tickers)
        data = yf.download(tickerString, period=PricePeriod.ONE_DAY.value, interval=PriceInterval.ONE_MINUTE.value, progress=False, auto_adjust=True)
        
        # Extract the most recent price for each ticker
        if len(tickers) == 1:
            # Single ticker returns different structure
            latestPrice = data['Close'].iloc[-1] if not data.empty else None
            priceMap[tickers[0]] = latestPrice
        else:
            # Multiple tickers
            for ticker in tickers:
                try:
                    latestPrice = data['Close'][ticker].iloc[-1] if ticker in data['Close'].columns else None
                    priceMap[ticker] = latestPrice
                except Exception:
                    priceMap[ticker] = None
                    
    except Exception as error:
        print(f"Error in bulk price fetch: {error}")
        # Fallback: return None for all tickers
        for ticker in tickers:
            priceMap[ticker] = None
            
    return priceMap


def getHistoricalPrices(
    ticker: str,
    period: PricePeriod = PricePeriod.ONE_MONTH,
    interval: PriceInterval = PriceInterval.ONE_DAY,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Get historical price data for a single ticker.
    
    Args:
        ticker: Stock symbol
        period: Time period for historical data (ignored if startDate/endDate provided)
        interval: Data interval/granularity
        startDate: Optional start date in 'YYYY-MM-DD' format
        endDate: Optional end date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume), or None if fails
    """
    try:
        stock = yf.Ticker(ticker)
        
        if startDate and endDate:
            historicalData = stock.history(start=startDate, end=endDate, interval=interval.value)
        else:
            historicalData = stock.history(period=period.value, interval=interval.value)
            
        return historicalData if not historicalData.empty else None
        
    except Exception as error:
        print(f"Error fetching historical data for {ticker}: {error}")
        return None


def getHistoricalPricesBulk(
    tickers: List[str],
    period: PricePeriod = PricePeriod.ONE_MONTH,
    interval: PriceInterval = PriceInterval.ONE_DAY,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Get historical price data for multiple tickers efficiently.
    Uses bulk download to minimize API calls.
    
    Args:
        tickers: List of stock symbols
        period: Time period for historical data
        interval: Data interval/granularity
        startDate: Optional start date in 'YYYY-MM-DD' format
        endDate: Optional end date in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary mapping ticker symbols to their historical data DataFrames
    """
    dataMap = {}
    
    try:
        tickerString = " ".join(tickers)
        
        if startDate and endDate:
            bulkData = yf.download(tickerString, start=startDate, end=endDate, 
                                  interval=interval.value, progress=False, group_by='ticker', auto_adjust=True)
        else:
            bulkData = yf.download(tickerString, period=period.value, 
                                  interval=interval.value, progress=False, group_by='ticker', auto_adjust=True)
        
        # Parse the multi-index DataFrame
        if len(tickers) == 1:
            # Single ticker returns simpler structure
            dataMap[tickers[0]] = bulkData if not bulkData.empty else None
        else:
            # Multiple tickers - data is grouped by ticker
            for ticker in tickers:
                try:
                    tickerData = bulkData[ticker]
                    dataMap[ticker] = tickerData if not tickerData.empty else None
                except Exception:
                    dataMap[ticker] = None
                    
    except Exception as error:
        print(f"Error in bulk historical data fetch: {error}")
        for ticker in tickers:
            dataMap[ticker] = None
            
    return dataMap


def getPriceComparison(tickers: List[str], period: PricePeriod = PricePeriod.ONE_MONTH) -> Optional[pd.DataFrame]:
    """
    Get normalized price comparison for multiple tickers.
    Useful for comparing relative performance.
    
    Args:
        tickers: List of stock symbols to compare
        period: Time period for comparison
        
    Returns:
        DataFrame with normalized prices (base 100) for easy comparison
    """
    try:
        historicalDataMap = getHistoricalPricesBulk(tickers, period=period)
        
        # Extract closing prices and normalize
        comparisonData = pd.DataFrame()
        
        for ticker, data in historicalDataMap.items():
            if data is not None and not data.empty:
                closePrices = data['Close']
                # Normalize to base 100
                normalizedPrices = (closePrices / closePrices.iloc[0]) * 100
                comparisonData[ticker] = normalizedPrices
                
        return comparisonData if not comparisonData.empty else None
        
    except Exception as error:
        print(f"Error creating price comparison: {error}")
        return None


# Example usage
if __name__ == "__main__":
    print("=== Single Ticker Example ===")
    applPrice = getCurrentPrice("AAPL")
    print(f"Apple current price: ${applPrice}")
    
    print("\n=== Bulk Ticker Example ===")
    techStocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    bulkPrices = getCurrentPricesBulk(techStocks)
    for ticker, price in bulkPrices.items():
        print(f"{ticker}: ${price:.2f}" if price else f"{ticker}: N/A")
    
    print("\n=== Historical Data Example ===")
    historicalData = getHistoricalPrices("AAPL", period=PricePeriod.ONE_MONTH)
    if historicalData is not None:
        print(f"Retrieved {len(historicalData)} days of data")
        print(historicalData.tail())
    
    print("\n=== Price Comparison Example ===")
    comparison = getPriceComparison(techStocks, period=PricePeriod.THREE_MONTHS)
    if comparison is not None:
        print("Normalized price performance (base 100):")
        print(comparison.tail())
