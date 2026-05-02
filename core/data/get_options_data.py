"""
Options Chain Retrieval Module

Provides efficient individual and bulk access to option chains with reusable filtering.
Built on yfinance with focus on readability, debuggability, and performance.
Greeks (delta, gamma, etc.) are not available via yfinance and are intentionally excluded.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
from .get_stock_price import getCurrentPrice


class OptionType(Enum):
    """Option contract types for filtering"""
    CALL = "calls"
    PUT = "puts"
    BOTH = "both"


def getOptionExpirations(ticker: str) -> List[str]:
    """
    Get available expiration dates for a ticker's option chain.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        List of expiration dates as strings (YYYY-MM-DD), or empty list if none/unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        return list(expirations) if expirations else []
    except Exception as error:
        print(f"Error fetching expirations for {ticker}: {error}")
        return []


def _getNearestExpiration(expirations: List[str]) -> Optional[str]:
    """Internal: Select the nearest future expiration date"""
    if not expirations:
        return None
    today = datetime.now()
    futureDates = [
        datetime.strptime(date, "%Y-%m-%d") for date in expirations
        if datetime.strptime(date, "%Y-%m-%d") >= today
    ]
    if not futureDates:
        return None
    nearest = min(futureDates, key=lambda d: (d - today).days)
    return nearest.strftime("%Y-%m-%d")
    

def getOptionChain(
    ticker: str,
    expiration: Optional[str] = None,
    optionType: OptionType = OptionType.BOTH
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Get option chain for a single ticker and expiration.
    
    Args:
        ticker: Stock symbol
        expiration: Specific expiration date (YYYY-MM-DD), or None for nearest
        optionType: CALL, PUT, or BOTH
        
    Returns:
        Dict with 'calls' and/or 'puts' DataFrames (or None if unavailable)
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        
        if not expirations:
            print(f"No option chain available for {ticker}")
            return {"calls": None, "puts": None}
        
        selectedExpiration = expiration or _getNearestExpiration(list(expirations))
        if selectedExpiration not in expirations:
            print(f"Expiration {selectedExpiration} not available for {ticker}. Available: {expirations[:5]}...")
            return {"calls": None, "puts": None}
        
        chain = stock.option_chain(selectedExpiration)
        calls = chain.calls.copy() if not chain.calls.empty else None
        puts = chain.puts.copy() if not chain.puts.empty else None
        
        result = {}
        if optionType in (OptionType.CALL, OptionType.BOTH):
            result["calls"] = calls
        if optionType in (OptionType.PUT, OptionType.BOTH):
            result["puts"] = puts
            
        return result
        
    except Exception as error:
        print(f"Error fetching option chain for {ticker} {expiration or 'nearest'}: {error}")
        return {"calls": None, "puts": None}


def getOptionChainsBulk(
    tickers: List[str],
    expiration: Optional[str] = None,
    optionType: OptionType = OptionType.BOTH
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """
    Get option chains for multiple tickers efficiently.
    
    Args:
        tickers: List of stock symbols
        expiration: Specific expiration, or None for nearest per ticker
        optionType: CALL, PUT, or BOTH
        
    Returns:
        Nested dict: {ticker: {'calls': df, 'puts': df}}
    """
    results = {}
    for ticker in tickers:
        results[ticker] = getOptionChain(ticker, expiration=expiration, optionType=optionType)
    return results


def getOptionsNearStrike(
    chainData: Dict[str, Optional[pd.DataFrame]],
    targetStrike: Optional[float] = None,
    tolerance: float = 0.10,
    usePercentage: bool = True,
    ticker: Optional[str] = None,
    underlyingPrice: Optional[float] = None
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Filter option chain to strikes near a target (or ATM).
    Adaptive tolerance scales with underlying price for volatile/high-priced tickers.
    
    Args:
        chainData: Output from getOptionChain
        targetStrike: Specific strike, or None for at-the-money (uses underlying price)
        tolerance: Absolute dollars or percentage (if usePercentage=True)
        usePercentage: If True, tolerance is % of underlying price (default/recommended)
        ticker: Required if underlyingPrice not provided and targetStrike is None
        underlyingPrice: Optional override price
        
    Returns:
        Filtered {'calls': df, 'puts': df}
    """
    if chainData is None:
        return {"calls": None, "puts": None}
    
    price = underlyingPrice or getCurrentPrice(ticker)
    if targetStrike is None and price is None:
        if ticker is None:
            return {"calls": None, "puts": None, "error": "empty ticker"}
        price = getCurrentPrice(ticker)
        if price is None:
            return {"calls": None, "puts": None, "error": "empty price"}
        target = price
    else:
        target = targetStrike or price or 0.0
    
    effectiveTolerance = tolerance
    if usePercentage and price:
        effectiveTolerance = price * tolerance
    
    lower = target - effectiveTolerance
    upper = target + effectiveTolerance
    
    if isinstance(chainData, pd.DataFrame):
        mask = (chainData["strike"] >= lower) & (chainData["strike"] <= upper)
        return chainData[mask].copy() if mask.any() else None
    
    filtered = {}
    for optType, df in chainData.items():
        if df is None or df.empty:
            filtered[optType] = None
        else:
            mask = (df["strike"] >= lower) & (df["strike"] <= upper)
            filtered[optType] = df[mask].copy() if mask.any() else None
            
    return filtered


def getOptionsNearStrikeBulk(
    chainData: Dict[str, Dict[str, Optional[pd.DataFrame]]],
    targetStrike: Optional[float] = None,
    tolerance: float = 0.05,
    usePercentage: bool = True
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """
    Bulk version of near-strike filtering with per-ticker price awareness.
    """
    results = {}
    for ticker, chain in chainData.items():
        price = getCurrentPrice(ticker)
        results[ticker] = getOptionsNearStrike(
            chain,
            targetStrike=targetStrike,
            tolerance=tolerance,
            usePercentage=usePercentage,
            underlyingPrice=price,
            ticker=ticker if targetStrike is None else None
        )
    return results


# Reusable filter functions (chainable with .pipe())

def filterByVolume(df: pd.DataFrame, minVolume: int = 100) -> pd.DataFrame:
    """Filter options with minimum trading volume"""
    if df is None or df.empty:
        return df
    return df[df["volume"] >= minVolume].copy()


def filterByOpenInterest(df: pd.DataFrame, minOI: int = 1000) -> pd.DataFrame:
    """Filter options with minimum open interest (liquidity proxy)"""
    if df is None or df.empty:
        return df
    return df[df["openInterest"] >= minOI].copy()


def filterByImpliedVolatility(
    df: pd.DataFrame,
    minIV: float = 0.0,
    maxIV: Optional[float] = None
) -> pd.DataFrame:
    """Filter by implied volatility range (fraction, e.g., 0.3 = 30%)"""
    if df is None or df.empty:
        return df
    mask = df["impliedVolatility"] >= minIV
    if maxIV is not None:
        mask &= df["impliedVolatility"] <= maxIV
    return df[mask].copy()


def filterInTheMoney(df: pd.DataFrame, itm_only: bool = True) -> pd.DataFrame:
    """Filter to in-the-money options (or out-of-money if False)"""
    if df is None or df.empty:
        return df
    if itm_only:
        return df[df["inTheMoney"]].copy()
    else:
        return df[~df["inTheMoney"]].copy()



def getEarningsContext(ticker: str, atm_straddle_price: float = 0.0,
                       spot_price: float = 0.0) -> Dict:
    """Fetch upcoming earnings date and compute historical realized move context.

    Returns a dict with:
      next_date           - next earnings date string (YYYY-MM-DD) or None
      days_to_earnings    - calendar days until next earnings
      avg_realized_move   - mean abs 1-day move across past earnings events
      move_richness       - priced_in_pct / avg_realized_move (>1 = expensive straddle)
      realized_moves      - list of individual past moves
    """
    EARNINGS_HISTORY_LIMIT = 8
    EARNINGS_WINDOW_DAYS = 1  # measure close-to-close on the announcement day

    result = {
        "next_date": None,
        "days_to_earnings": None,
        "avg_realized_move": None,
        "move_richness": None,
        "realized_moves": []
    }

    try:
        stock = yf.Ticker(ticker)
        earnings_dates_df = stock.get_earnings_dates(limit=EARNINGS_HISTORY_LIMIT)

        if earnings_dates_df is None or earnings_dates_df.empty:
            return result

        today = datetime.now().date()

        # Normalize index to date objects for comparison
        earnings_dates_df.index = pd.to_datetime(earnings_dates_df.index).date

        future_dates = [d for d in earnings_dates_df.index if d > today]
        past_dates = [d for d in earnings_dates_df.index if d <= today]

        if future_dates:
            next_earnings_date = min(future_dates)
            result["next_date"] = str(next_earnings_date)
            result["days_to_earnings"] = (next_earnings_date - today).days

        # Compute realized moves around each past earnings date
        realized_moves = []
        for earnings_date in past_dates:
            window_start = (datetime.combine(earnings_date, datetime.min.time()) -
                            timedelta(days=2)).strftime("%Y-%m-%d")
            window_end = (datetime.combine(earnings_date, datetime.min.time()) +
                          timedelta(days=EARNINGS_WINDOW_DAYS + 1)).strftime("%Y-%m-%d")
            hist = stock.history(start=window_start, end=window_end)

            if hist is None or len(hist) < 2:
                continue

            prev_close = float(hist["Close"].iloc[-2])
            event_close = float(hist["Close"].iloc[-1])
            if prev_close > 0:
                realized_move_pct = abs((event_close - prev_close) / prev_close)
                realized_moves.append(round(realized_move_pct, 4))

        if realized_moves:
            avg_realized = float(np.mean(realized_moves))
            result["avg_realized_move"] = round(avg_realized, 4)
            result["realized_moves"] = realized_moves

            if atm_straddle_price > 0 and spot_price > 0:
                priced_in_pct = atm_straddle_price / spot_price
                result["move_richness"] = round(priced_in_pct / avg_realized, 3) if avg_realized > 0 else None

    except Exception as error:
        print(f"Warning: Could not fetch earnings context for {ticker}: {error}")

    return result


# Example usage

if __name__ == "__main__":
    ticker = "IREN"
    
    print("=== Option Expirations ===")
    expirations = getOptionExpirations(ticker)
    print(f"Available: {expirations[:5]}...")
    
    print("\n=== Single Option Chain (Nearest Expiration) ===")
    chain = getOptionChain(ticker)
    calls = chain.get("calls")
    if calls is not None:
        print(f"Calls ({len(calls)} contracts):")
        print(calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].head())
    
    print("\n=== Filtered High-Liquidity ITM Calls Near ATM ===")
    if calls is not None:
        near_atm = getOptionsNearStrike(
            {"calls": calls, "puts": None},
            targetStrike=None,      # Auto ATM
            tolerance=0.05,          # 5% of price
            ticker=ticker
        )["calls"]
        
        if near_atm is not None:
            high_liquidity_itm = (
                near_atm
                .pipe(filterByVolume, minVolume=500)
                .pipe(filterByOpenInterest, minOI=2000)
                .pipe(filterInTheMoney)
                .pipe(filterByImpliedVolatility, minIV=0.2)
            )
            print(high_liquidity_itm[['strike', 'lastPrice', 'volume', 'openInterest', 'impliedVolatility']])
    
    print("\n=== Bulk Example ===")
    tech_tickers = ["AAPL", "NVDA", "TSLA"]
    bulk_chains = getOptionChainsBulk(tech_tickers)
    near_strikes = getOptionsNearStrikeBulk(bulk_chains, tolerance=0.05)
    for t, filtered in near_strikes.items():
        calls_count = len(filtered["calls"]) if filtered["calls"] is not None else 0
        print(f"{t}: {calls_count} calls near ATM")