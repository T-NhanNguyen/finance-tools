"""
GEX (Gamma Exposure) Data Provider
Core logic for fetching, calculating, and aggregating GEX data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from core.data import (
    getOptionChain, getOptionExpirations, getCurrentPrice
)
from core.analysis import calculateGamma

OPTION_CHAIN_LENGTH = 50

def parse_flexible_date(ticker: str, date_str: Optional[str]) -> Optional[str]:
    """
    Resolves a flexible date string or index to a valid expiration date (YYYY-MM-DD).
    """
    expirations = getOptionExpirations(ticker)
    if not expirations:
        return None
    
    # Handle empty/None -> nearest
    if not date_str:
        today = datetime.now().strftime("%Y-%m-%d")
        future_exps = [e for e in expirations if e >= today]
        return future_exps[0] if future_exps else expirations[0]

    date_str = str(date_str).strip().lower()

    # Handle numeric index
    if date_str.isdigit():
        idx = int(date_str)
        if 0 <= idx < len(expirations):
            return expirations[idx]
        return expirations[0]

    # Try parsing common formats
    parsed_dt = None
    for fmt in ("%m/%d/%y", "%m-%d-%y", "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            normalized = dt.strftime("%Y-%m-%d")
            if normalized in expirations:
                return normalized
            if not parsed_dt:
                parsed_dt = dt
        except ValueError:
            continue

    # Prefix or partial match
    normalized_input = date_str.replace("/", "-").replace(".", "-")
    matches = [e for e in expirations if e.startswith(normalized_input) or normalized_input in e]
    if matches:
        return matches[0]

    # If we parsed a date but it wasn't an exact match, find the nearest one globally
    if parsed_dt:
        return min(expirations, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - parsed_dt).total_seconds()))

    # Fallback to nearest future expiration (today or later)
    today = datetime.now().strftime("%Y-%m-%d")
    future_exps = [e for e in expirations if e >= today]
    return future_exps[0] if future_exps else expirations[0]


def fetch_gex_data_raw(ticker: str, expiration_input: Optional[str] = None) -> Dict:
    """
    Core GEX data fetching and processing logic.
    Returns a raw dictionary of results.
    """
    ticker = ticker.upper()
    
    # Get all available expirations
    available_expirations = getOptionExpirations(ticker)
    if not available_expirations:
        return {"error": "No option expirations found", "ticker": ticker}
    
    # Parse expiration date
    expiration = parse_flexible_date(ticker, expiration_input)
    if not expiration:
        return {"error": "Could not parse expiration date", "ticker": ticker}
    
    # Fetch spot price
    spot_price = getCurrentPrice(ticker)
    if not spot_price:
        return {"error": "Could not fetch current price", "ticker": ticker}
    
    # Fetch option chain
    chain = getOptionChain(ticker, expiration=expiration)
    calls = chain.get('calls')
    puts = chain.get('puts')
    
    if calls is None or puts is None:
        return {"error": "Could not fetch option chain", "ticker": ticker, "details": f"Expiration: {expiration}"}
    
    # Calculate time to expiration
    today = datetime.now()
    expiry_date = datetime.strptime(expiration, "%Y-%m-%d")
    dte_years = max(1e-6, (expiry_date - today).total_seconds() / (365 * 24 * 3600))
    days_to_expiration = (expiry_date - today).days
    
    # Calculate gammas for calls
    call_strikes = calls['strike'].values
    call_ivs = calls['impliedVolatility'].values
    call_gammas = calculateGamma(spot_price, call_strikes, dte_years, call_ivs)
    calls['gamma'] = call_gammas
    calls['gex'] = calls['gamma'] * calls['openInterest'] * 100 * spot_price
    
    # Calculate gammas for puts
    put_strikes = puts['strike'].values
    put_ivs = puts['impliedVolatility'].values
    put_gammas = calculateGamma(spot_price, put_strikes, dte_years, put_ivs)
    puts['gamma'] = put_gammas
    puts['gex'] = -puts['gamma'] * puts['openInterest'] * 100 * spot_price
    
    # Aggregate by strike
    calls_agg = calls[['strike', 'gex', 'openInterest', 'bid', 'ask', 'impliedVolatility']].groupby('strike').agg({
        'gex': 'sum',
        'openInterest': 'sum',
        'bid': 'first',
        'ask': 'first',
        'impliedVolatility': 'first'
    }).reset_index()
    
    puts_agg = puts[['strike', 'gex', 'openInterest', 'bid', 'ask', 'impliedVolatility']].groupby('strike').agg({
        'gex': 'sum',
        'openInterest': 'sum',
        'bid': 'first',
        'ask': 'first',
        'impliedVolatility': 'first'
    }).reset_index()

    # Merge call and put data on strike
    combined_agg = pd.merge(
        calls_agg,
        puts_agg,
        on='strike',
        how='outer',
        suffixes=('_call', '_put')
    ).fillna(0.0)
    
    # Calculate combined aggregates
    combined_agg['totalGEX'] = combined_agg['gex_call'] + combined_agg['gex_put']
    combined_agg['totalOI'] = combined_agg['openInterest_call'] + combined_agg['openInterest_put']
    
    # Filter for strikes around ATM
    combined_agg = combined_agg.sort_values('strike').reset_index(drop=True)
    atm_idx = (combined_agg['strike'] - spot_price).abs().idxmin()
    
    start_idx = max(0, atm_idx - OPTION_CHAIN_LENGTH)
    end_idx = min(len(combined_agg), atm_idx + OPTION_CHAIN_LENGTH + 1)
    plot_df = combined_agg.iloc[start_idx:end_idx].copy()
    
    # Find max values for normalization
    max_gex_absolute = float(plot_df['totalGEX'].abs().max() or 1)
    max_open_interest = float(plot_df['totalOI'].max() or 1)
    
    # Find ATM strike
    atm_strike = plot_df.iloc[(plot_df['strike'] - spot_price).abs().argsort()[:1]]['strike'].values[0]
    
    # Build strike data list
    strikes = []
    for _, row in plot_df.iterrows():
        strike = float(row['strike'])
        strikes.append({
            "strike": strike,
            "gexMillions": float(row['totalGEX'] / 1e6),
            "openInterestThousands": float(row['totalOI'] / 1e3),
            "isATM": (strike == atm_strike),
            "normalizedGEX": float(row['totalGEX'] / max_gex_absolute),
            "normalizedOI": float(row['totalOI'] / max_open_interest),
            "callBid": float(row['bid_call']),
            "callAsk": float(row['ask_call']),
            "callIV": float(row['impliedVolatility_call']),
            "callOI": float(row['openInterest_call'] / 1e3),
            "putBid": float(row['bid_put']),
            "putAsk": float(row['ask_put']),
            "putIV": float(row['impliedVolatility_put']),
            "putOI": float(row['openInterest_put'] / 1e3)
        })
    
    return {
        "ticker": ticker,
        "expiration": expiration,
        "spotPrice": float(spot_price),
        "daysToExpiration": int(days_to_expiration),
        "strikes": strikes,
        "maxGEXAbsolute": float(max_gex_absolute),
        "maxOpenInterest": float(max_open_interest),
        "availableExpirations": available_expirations
    }
