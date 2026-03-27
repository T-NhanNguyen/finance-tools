"""
Bulk Data Loader & Caching Utility
Provides rate-limited concurrent I/O fetches utilizing ThreadPools and File Caching.
"""

import os
import json
import time
import threading
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from core.data.gex_provider import fetch_gex_data_raw

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".gex_cache"))
CACHE_EXPIRY = 900  # 15 minutes (900 seconds)

_FETCH_LOCK = threading.Lock()
FETCH_STAGGER_DELAY = 0.5  # seconds between live network requests to avoid rate limiting
MAX_FETCH_RETRIES = 3


def _get_cache_path(ticker: str, expiration: Optional[str]) -> str:
    """Calculates flat-file absolute path to workspace buffers"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    exp_hash = expiration if expiration else "nearest"
    return os.path.join(CACHE_DIR, f"{ticker.upper()}_{exp_hash.replace('-', '')}.json")

def _load_from_cache(path: str) -> Optional[Dict]:
    """Load JSON buffers assuming they are within TTL thresholds"""
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        if (time.time() - mtime) < CACHE_EXPIRY:
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
    return None

def _save_to_cache(path: str, data: Dict) -> None:
    """Commits node data payloads backwards into local buffer"""
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass

def fetch_gex_single(ticker: str, expiration: Optional[str] = None) -> Dict:
    """
    Gets GEX data with local absolute caching.
    Serializes live network requests via a lock to avoid rate limiting.
    Retries up to MAX_FETCH_RETRIES times with exponential backoff on failure.
    """
    cache_path = _get_cache_path(ticker, expiration)
    cached_data = _load_from_cache(cache_path)
    if cached_data:
        return cached_data

    with _FETCH_LOCK:
        # Re-check cache after acquiring lock — another thread may have populated it
        cached_data = _load_from_cache(cache_path)
        if cached_data:
            return cached_data

        last_result = {"error": f"All {MAX_FETCH_RETRIES} fetch attempts failed", "ticker": ticker}
        for attempt in range(MAX_FETCH_RETRIES):
            time.sleep(FETCH_STAGGER_DELAY * (2 ** attempt))
            data = fetch_gex_data_raw(ticker, expiration)
            if "error" not in data:
                strikes = data.get("strikes", [])
                total_bids = sum((s.get("putBid", 0) or 0) + (s.get("callBid", 0) or 0) for s in strikes)
                if strikes and total_bids > 0:
                    _save_to_cache(cache_path, data)
                return data
            last_result = data

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
