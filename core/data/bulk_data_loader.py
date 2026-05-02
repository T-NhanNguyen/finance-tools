import os
import json
import time
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from core.data.gex_provider import fetch_gex_data_raw
from api.cache_manager import cache_manager

_TICKER_LOCKS: Dict[str, threading.Lock] = {}
_LOCK_ATTR_LOCK = threading.Lock()
FETCH_STAGGER_DELAY = 0.5
MAX_FETCH_RETRIES = 2
FAILURE_TTL = 300  # 5 minutes failure cache


def fetch_gex_single(ticker: str, expiration: Optional[str] = None) -> Dict:
    """Gets GEX data with local absolute caching and failure protection."""
    ticker = ticker.upper()
    
    # 1. Check Cache Manager first
    cached_data = cache_manager.get(ticker, expiration)
    if cached_data:
        # print(f"DEBUG: Cache HIT for {ticker} {expiration}")
        if "error" in cached_data:
            return cached_data
        return cached_data


    # Get or create per-ticker lock
    with _LOCK_ATTR_LOCK:
        if ticker not in _TICKER_LOCKS:
            _TICKER_LOCKS[ticker] = threading.Lock()
        lock = _TICKER_LOCKS[ticker]

    with lock:
        # Re-check cache after acquiring lock
        cached_data = cache_manager.get(ticker, expiration)
        if cached_data:
            return cached_data

        last_result = {"error": "Fetch failed", "ticker": ticker}
        for attempt in range(MAX_FETCH_RETRIES):
            if attempt > 0:
                time.sleep(FETCH_STAGGER_DELAY * (2 ** attempt))
            
            data = fetch_gex_data_raw(ticker, expiration)
            if "error" not in data:
                # Basic validation: ensure we have strikes and a valid price
                if data.get("strikes") and data.get("spotPrice", 0) > 0:
                    cache_manager.put(ticker, expiration, data)
                    return data
                else:
                    last_result = {"error": "Invalid data (0 price or empty strikes)", "ticker": ticker}
            else:
                last_result = data
        
        # Cache the failure briefly to avoid spamming the API
        # We wrap it in a special error dict so cache_manager can still see it
        # but OptionStrikeOptimizer knows it's a failure.
        # Actually, let's just use the error dict directly.
        # NOTE: CacheManager.put uses time.time() as timestamp.
        # If we put an error dict, it will be "fresh" for FAILURE_TTL if we adjust _is_expired
        # For now, let's just NOT cache failures but optimize the path.
        return last_result

def fetch_gex_bulk(tickers: List[str], expiration: Optional[str] = None, max_workers: int = 4) -> Dict[str, Dict]:
    """
    Concurrently fetches structured JSON targets utilizing ThreadPoolExecutor
    and transparent recursive fetch-layer caching underneath.
    """
    results = {}

    def worker(t):
        return t, fetch_gex_single(t, expiration)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(worker, t): t for t in tickers}
        for future in future_to_ticker:
            t, data = future.result()
            results[t] = data

    return results

def fetch_all_expirations(ticker: str) -> Dict[str, str]:
    """
    Fetches the nearest expiration first (to get availableExpirations list),
    then spawns background threads for the remaining expirations.
    Returns immediately with { expiration_date: "fetching"|"cached"|"error" } status map.
    """
    ticker = ticker.upper()
    # 1. Fetch nearest to get the list of all expirations
    nearest_data = fetch_gex_single(ticker, None)
    
    if "error" in nearest_data:
        return {"nearest": "error"}

    expirations = nearest_data.get("availableExpirations", [])
    if not expirations:
        return {"nearest": "cached"}

    status_map = {}
    to_fetch = []

    for exp in expirations:
        if cache_manager.is_fresh(ticker, exp):
            status_map[exp] = "cached"
        else:
            status_map[exp] = "fetching"
            to_fetch.append(exp)

    # 2. Spawn background threads for remaining expirations
    def background_worker(t, e):
        fetch_gex_single(t, e)

    if to_fetch:
        executor = ThreadPoolExecutor(max_workers=4)
        for exp in to_fetch:
            executor.submit(background_worker, ticker, exp)
        # Note: Executor is NOT shut down here to allow background work to continue
        # but we don't wait for results. 

    return status_map

def fetch_gex_all_expirations(ticker: str) -> Dict:
    """
    Fetches GEX data for all available expirations for a ticker.
    Aggregates high-level metrics for each expiration.
    Returns a dict structure compatible with OptionStrikeOptimizer.
    """
    ticker = ticker.upper()
    nearest_data = fetch_gex_single(ticker, None)
    if "error" in nearest_data:
        return nearest_data
        
    available_expirations = nearest_data.get("availableExpirations", [])
    if not available_expirations:
        return nearest_data
        
    expirations_metrics = []
    spot_price = nearest_data["spotPrice"]
    
    # We'll use a thread pool to fetch all expirations concurrently
    def fetch_metrics(exp):
        data = fetch_gex_single(ticker, exp)
        if "error" in data:
            return None
        
        strikes = data.get("strikes", [])
        total_oi = sum(s.get("openInterestThousands", 0) for s in strikes) * 1000
        total_gex = sum(s.get("gexMillions", 0) for s in strikes) * 1e6
        total_volume = sum(s.get("volume", 0) for s in strikes) # Assuming we add volume to gex_provider
        
        return {
            "expiration": exp,
            "daysToExpiration": data.get("daysToExpiration", 0),
            "totalOI": total_oi,
            "volume": total_volume,
            "netGEX": total_gex,
            "gex_density": abs(total_gex) / (total_oi if total_oi > 0 else 1),
            "spotPrice": spot_price
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_metrics, available_expirations))
        
    expirations_metrics = [r for r in results if r is not None]
    
    return {
        "ticker": ticker,
        "spotPrice": spot_price,
        "expirations": expirations_metrics,
        "availableExpirations": available_expirations
    }

def get_cached_option_chain(ticker: str, expiration: Optional[str]) -> Optional[Dict]:
    """
    Pure cache read — delegates to cache_manager.get().
    Returns None on miss; does NOT trigger a live fetch.
    """
    return cache_manager.get(ticker.upper(), expiration)


def run_bulk_loader():
    """
    CLI interface for bulk GEX data fetching and caching.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Bulk GEX Data Loader & Caching Utility")
    parser.add_argument("tickers", nargs="*", help="List of tickers to fetch (e.g., SPY QQQ)")
    parser.add_argument("--expiration", help="Expiration date (YYYY-MM-DD or index)")
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent workers")
    args = parser.parse_args()

    # Default tickers if none provided
    tickers = [t.upper() for t in args.tickers] if args.tickers else ["SPY", "QQQ"]
    print(f"\n{'='*60}")
    print(f"Bulk Fetching GEX for: {', '.join(tickers)}")
    print(f"Expiration Filter: {args.expiration or 'Nearest'}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    results = fetch_gex_bulk(tickers, expiration=args.expiration, max_workers=args.workers)
    elapsed = float(time.time() - start_time)
    
    print("-" * 60)
    for t in tickers:
        data = results.get(t, {})
        if "error" in data:
            status = f"\033[91mFAILED\033[0m - {data['error']}"
        else:
            status = f"\033[92mOK\033[0m - {len(data.get('strikes', []))} strikes cached"
        print(f"{t:<6}: {status}")
    print("-" * 60)
    print(f"Total time: {elapsed:.2f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_bulk_loader()
