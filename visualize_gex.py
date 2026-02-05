import numpy as np
import pandas as pd
from get_options_data import getOptionChain, getOptionExpirations
from get_stock_price import getCurrentPrice
from calculate_gamma_delta import calculateGamma
from datetime import datetime
import argparse
import sys

OPTION_CHAIN_LENGTH = 10

def parse_flexible_date(ticker, date_str):
    """
    Resolves a flexible date string or index to a valid expiration date (YYYY-MM-DD).
    
    Supports:
    - None or empty string: Nearest future expiration.
    - Numeric index (e.g., '0' for nearest, '1' for next).
    - Partial strings (e.g., '2026-01' -> '2026-01-16').
    - Exact dates (YYYY-MM-DD).
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

    # Try parsing common formats like M/D/YY, M/D/YYYY, etc.
    for fmt in ("%m/%d/%y", "%m-%d-%y", "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"):
        try:
            parsed_dt = datetime.strptime(date_str, fmt)
            normalized = parsed_dt.strftime("%Y-%m-%d")
            if normalized in expirations:
                return normalized
        except ValueError:
            continue

    # Normalize separators for prefix/partial match logic
    normalized_input = date_str.replace("/", "-").replace(".", "-")

    # Exact match after normalization
    if normalized_input in expirations:
        return normalized_input

    # Prefix or partial match
    matches = [e for e in expirations if e.startswith(normalized_input) or normalized_input in e]
    if matches:
        return matches[0]

    # Fallback to nearest
    today = datetime.now().strftime("%Y-%m-%d")
    future_exps = [e for e in expirations if e >= today]
    return future_exps[0] if future_exps else expirations[0]

def visualize_gex(ticker, expiration_input=None):
    ticker = ticker.upper()
    expiration = parse_flexible_date(ticker, expiration_input)
    
    if not expiration:
        print(f"Error: Could not find any option expirations for {ticker}")
        return

    print(f"\nTargeting Expiration: {expiration} for {ticker}")
    # 1. Fetch Spot Price
    spot_price = getCurrentPrice(ticker)
    if not spot_price:
        print(f"Error: Could not fetch price for {ticker}")
        return

    # 2. Fetch Option Chain
    chain = getOptionChain(ticker, expiration=expiration)
    calls = chain.get('calls')
    puts = chain.get('puts')

    if calls is None or puts is None:
        print(f"Error: Could not fetch option chain for {expiration}")
        return

    # Calculate Time to Expiration
    today = datetime.now()
    expiry_date = datetime.strptime(expiration, "%Y-%m-%d")
    dte_years = max(1e-6, (expiry_date - today).total_seconds() / (365 * 24 * 3600))

    # 3. Process Data
    # Merge calls and puts on strike
    # We only care about strike, openInterest, and gamma
    
    # Calculate Gammas for all calls
    call_strikes = calls['strike'].values
    call_ivs = calls['impliedVolatility'].values
    call_gammas = calculateGamma(spot_price, call_strikes, dte_years, call_ivs)
    calls['gamma'] = call_gammas
    calls['gex'] = calls['gamma'] * calls['openInterest'] * 100 * spot_price

    # Calculate Gammas for all puts
    put_strikes = puts['strike'].values
    put_ivs = puts['impliedVolatility'].values
    put_gammas = calculateGamma(spot_price, put_strikes, dte_years, put_ivs)
    puts['gamma'] = put_gammas
    # Puts are negative gamma exposure from MM perspective (short puts)
    puts['gex'] = -puts['gamma'] * puts['openInterest'] * 100 * spot_price

    # Aggregate by strike (Include Open Interest)
    calls_agg = calls[['strike', 'gex', 'openInterest']].groupby('strike').sum()
    puts_agg = puts[['strike', 'gex', 'openInterest']].groupby('strike').sum()
    
    combined_agg = pd.concat([calls_agg, puts_agg]).groupby('strike').sum().reset_index()

    # Filter for strikes: 5 above and 5 below the strike closest to spot price
    combined_agg = combined_agg.sort_values('strike').reset_index(drop=True)
    
    # Find the index of the strike closest to spot price
    atm_idx = (combined_agg['strike'] - spot_price).abs().idxmin()
    
    # Select slicing range (5 below, strike itself, 5 above = 11 total)
    start_idx = max(0, atm_idx - OPTION_CHAIN_LENGTH)
    end_idx = min(len(combined_agg), atm_idx + OPTION_CHAIN_LENGTH + 1)
    
    plot_df = combined_agg.iloc[start_idx:end_idx].copy()

    # 4. ASCII Chart with Color Overlay
    # Terminal Colors
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\nGamma Exposure & Open Interest for {YELLOW}{ticker.upper()}{RESET} expiring {YELLOW}{expiration}{RESET}")
    print(f"Spot Price: {CYAN}${spot_price:.2f}{RESET}")
    print("-" * 85)
    print(f"{'Strike':<10} | {'GEX ($M)':<12} | {'OI (K)':<8} | {'GEX (Bar) vs OI (Dots)'}")
    print("-" * 85)

    max_gex = plot_df['gex'].abs().max() or 1
    max_oi = plot_df['openInterest'].max() or 1
    chart_width = 40
    mid = chart_width // 2

    # Find the strike closest to spot price to highlight ATM
    atm_strike = plot_df.iloc[(plot_df['strike'] - spot_price).abs().argsort()[:1]]['strike'].values[0]

    for _, row in plot_df.iterrows():
        strike = row['strike']
        gex_m = row['gex'] / 1e6
        oi_k = row['openInterest'] / 1e3
        
        # Color coding and highlighting
        is_atm = strike == atm_strike
        strike_str = f"{strike:<10.2f}"
        if is_atm:
            strike_str = f"{BOLD}{CYAN}>> {strike:<7.2f}{RESET}"
        else:
            strike_str = f"   {strike:<7.2f}"
            
        gex_color = GREEN if gex_m >= 0 else RED
        gex_str = f"{gex_color}{gex_m:>12.2f}{RESET}"
        oi_str = f"{BLUE}{oi_k:>8.1f}{RESET}"
        
        # Smart Overlay Logic
        norm_gex = int((row['gex'] / max_gex) * mid)
        norm_oi = int((row['openInterest'] / max_oi) * mid)
        
        # Create visual canvas
        canvas = [" "] * (chart_width + 1)
        canvas[mid] = "|"
        
        # Overlay Logic for overlapping bars (Positive GEX and OI both go right)
        if norm_gex >= 0:
            shorter = min(norm_gex, norm_oi)
            longer = max(norm_gex, norm_oi)
            
            # Determine which color/char gets the inner (top) segment
            # If OI < GEX, OI is on top. If OI > GEX, GEX is on top (as per user request)
            # Both cases result in the shorter bar being the inner segment
            inner_is_oi = norm_oi < norm_gex
            
            inner_color = BLUE if inner_is_oi else GREEN
            inner_char = "·" if inner_is_oi else "#"
            outer_color = GREEN if inner_is_oi else BLUE
            outer_char = "#" if inner_is_oi else "·"
            
            for i in range(1, shorter + 1):
                canvas[mid + i] = f"{inner_color}{inner_char}{RESET}"
            for i in range(shorter + 1, longer + 1):
                canvas[mid + i] = f"{outer_color}{outer_char}{RESET}"
        else:
            # Negative GEX (Left) and OI (Right) - No overlap
            for i in range(1, norm_oi + 1):
                canvas[mid + i] = f"{BLUE}·{RESET}"
            for i in range(1, abs(norm_gex) + 1):
                canvas[mid - i] = f"{RED}#{RESET}"
                
        bar_chart = "".join(canvas)
        print(f"{strike_str} | {gex_str} | {oi_str} |{bar_chart}")

    print("-" * 85)
    print(f"{GREEN}GEX (#){RESET} | {BLUE}OI (·){RESET} | {CYAN}>> At-The-Money{RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Gamma Exposure (GEX) for a ticker.")
    parser.add_argument("ticker", nargs="?", help="Stock ticker symbol (e.g., SPY, QQQ, SLV)")
    parser.add_argument("date", nargs="?", help="Expiration date (YYYY-MM-DD, partial string, or index 0, 1, 2...)")
    
    args = parser.parse_args()
    
    ticker = args.ticker
    date = args.date
    
    if not ticker:
        ticker = input("Enter Ticker (e.g. SPY, SLV): ").strip().upper() or "SPY"
        date = input("Enter Expiration/Index (leave blank for nearest): ").strip()
        if not date: date = None
        
    visualize_gex(ticker, date)
