"""
GEX Data Fetcher

Authoritative source for fetching, calculating, and structuring
Gamma Exposure data from the raw option chain.

Used by:
  - core/data/bulk_data_loader.py  (caching + batch scanner)
  - api/api_handlers.py            (HTTP response formatting)
  - visualizers/visualize_gex.py   (terminal chart rendering)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

from core.data.get_options_data import getOptionChain, getOptionExpirations
from core.data.get_stock_price import getCurrentPrice
from core.analysis import calculateGamma
from visualizers.visualize_gex import parse_flexible_date, OPTION_CHAIN_LENGTH


def fetch_gex_structured(ticker: str, expiration_input: Optional[str] = None) -> Dict:
    """
    Fetches and calculates all GEX data for a ticker and expiration.
    Returns a fully structured dict with strike-level breakdown suitable
    for both the API response layer and the local caching layer.
    """
    ticker = ticker.upper()

    available_expirations = getOptionExpirations(ticker)
    if not available_expirations:
        return {"error": "No option expirations found", "ticker": ticker}

    expiration = parse_flexible_date(ticker, expiration_input)
    if not expiration:
        return {"error": "Could not parse expiration date", "ticker": ticker, "details": str(expiration_input)}

    spot_price = getCurrentPrice(ticker)
    if not spot_price:
        return {"error": "Could not fetch current price", "ticker": ticker}

    chain = getOptionChain(ticker, expiration=expiration)
    calls = chain.get("calls")
    puts = chain.get("puts")

    if calls is None or puts is None:
        return {"error": "Could not fetch option chain", "ticker": ticker, "details": f"Expiration: {expiration}"}

    today = datetime.now()
    expiry_date = datetime.strptime(expiration, "%Y-%m-%d")
    dte_years = max(1e-6, (expiry_date - today).total_seconds() / (365 * 24 * 3600))
    days_to_expiration = (expiry_date - today).days

    call_strikes = calls["strike"].values
    call_ivs = calls["impliedVolatility"].values
    call_gammas = calculateGamma(spot_price, call_strikes, dte_years, call_ivs)
    calls["gamma"] = call_gammas
    calls["gex"] = calls["gamma"] * calls["openInterest"] * 100 * spot_price

    put_strikes = puts["strike"].values
    put_ivs = puts["impliedVolatility"].values
    put_gammas = calculateGamma(spot_price, put_strikes, dte_years, put_ivs)
    puts["gamma"] = put_gammas
    puts["gex"] = -puts["gamma"] * puts["openInterest"] * 100 * spot_price

    calls_agg = calls[["strike", "gex", "openInterest", "bid", "ask", "impliedVolatility"]].groupby("strike").agg({
        "gex": "sum", "openInterest": "sum",
        "bid": "first", "ask": "first", "impliedVolatility": "first"
    }).reset_index()

    puts_agg = puts[["strike", "gex", "openInterest", "bid", "ask", "impliedVolatility"]].groupby("strike").agg({
        "gex": "sum", "openInterest": "sum",
        "bid": "first", "ask": "first", "impliedVolatility": "first"
    }).reset_index()

    combined_agg = pd.merge(
        calls_agg, puts_agg,
        on="strike", how="outer", suffixes=("_call", "_put")
    ).fillna(0.0)

    combined_agg["totalGEX"] = combined_agg["gex_call"] + combined_agg["gex_put"]
    combined_agg["totalOI"] = combined_agg["openInterest_call"] + combined_agg["openInterest_put"]
    combined_agg = combined_agg.sort_values("strike").reset_index(drop=True)

    atm_idx = (combined_agg["strike"] - spot_price).abs().idxmin()
    start_idx = max(0, atm_idx - OPTION_CHAIN_LENGTH)
    end_idx = min(len(combined_agg), atm_idx + OPTION_CHAIN_LENGTH + 1)
    plot_df = combined_agg.iloc[start_idx:end_idx].copy()

    max_gex_abs = float(plot_df["totalGEX"].abs().max() or 1)
    max_oi = float(plot_df["totalOI"].max() or 1)
    atm_strike = float(plot_df.iloc[(plot_df["strike"] - spot_price).abs().argsort()[:1]]["strike"].values[0])

    strikes: List[Dict] = []
    for _, row in plot_df.iterrows():
        strike = float(row["strike"])
        strikes.append({
            "strike": strike,
            "gexMillions": float(row["totalGEX"] / 1e6),
            "openInterestThousands": float(row["totalOI"] / 1e3),
            "isATM": strike == atm_strike,
            "normalizedGEX": float(row["totalGEX"] / max_gex_abs),
            "normalizedOI": float(row["totalOI"] / max_oi),
            "callBid": float(row["bid_call"]),
            "callAsk": float(row["ask_call"]),
            "callIV": float(row["impliedVolatility_call"]),
            "callOI": float(row["openInterest_call"] / 1e3),
            "putBid": float(row["bid_put"]),
            "putAsk": float(row["ask_put"]),
            "putIV": float(row["impliedVolatility_put"]),
            "putOI": float(row["openInterest_put"] / 1e3),
        })

    return {
        "ticker": ticker,
        "expiration": expiration,
        "spotPrice": float(spot_price),
        "daysToExpiration": days_to_expiration,
        "strikes": strikes,
        "maxGEXAbsolute": max_gex_abs,
        "maxOpenInterest": max_oi,
        "availableExpirations": available_expirations,
    }
