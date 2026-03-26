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
from core.data.get_gex_data import fetch_gex_structured

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
            data = fetch_gex_structured(ticker, expiration)
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
