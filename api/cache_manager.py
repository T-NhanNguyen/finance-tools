import os
import json
import time
import glob
import threading
from typing import Dict, List, Optional
from collections import OrderedDict
import pytz
from datetime import datetime, time as dtime, timedelta

# Constants
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".gex_cache"))
LIVE_TTL = 900  # 15 minutes during market hours
MAX_MEM_ENTRIES = 50
CACHE_EXPIRY = LIVE_TTL  # For backward compatibility

def get_market_status():
    """
    Returns (is_open, last_close, next_open) in Eastern Time.
    US Market: 9:30 AM - 4:00 PM ET, Mon-Fri.
    Accounting for ~30 min data delay, 'settlement' is at 4:30 PM ET.
    """
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    
    # Settlement time is 30 mins after close
    settle_time = dtime(16, 30)
    
    # Check if weekend
    if now.weekday() >= 5:
        is_open = False
    # Market is 'moving' until 4:30 PM due to delay
    elif now.time() < dtime(9, 30) or now.time() >= settle_time:
        is_open = False
    else:
        is_open = True
        
    # Calculate last close settlement (when we expect final daily data to be available)
    if now.time() < settle_time:
        # Last settlement was yesterday (or Friday)
        days_back = 3 if now.weekday() == 0 else 1
        if now.weekday() >= 5: days_back = now.weekday() - 4
        last_settle = now.replace(hour=16, minute=30, second=0, microsecond=0) - timedelta(days=days_back)
    else:
        # Last settlement was today
        last_settle = now.replace(hour=16, minute=30, second=0, microsecond=0)
        
    return is_open, last_settle, now

class CacheManager:
    """
    In-process LRU cache on top of the filesystem .gex_cache.
    Ensures hot tickers are served from memory while persisting all fetches to disk.
    """
    def __init__(self):
        self._mem_cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = threading.Lock()
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)

    def _get_cache_key(self, ticker: str, expiration: Optional[str]) -> str:
        """Standardized key for both memory and disk lookups."""
        exp_hash = expiration if expiration else "nearest"
        return f"{ticker.upper()}_{exp_hash.replace('-', '')}"

    def _is_expired(self, entry_timestamp: float) -> bool:
        """Determines if a cache entry is expired based on market hours."""
        is_open, last_close, now = get_market_status()
        
        # entry_dt in ET
        entry_dt = datetime.fromtimestamp(entry_timestamp, pytz.utc).astimezone(pytz.timezone('US/Eastern'))
        
        if is_open:
            # During market hours, use 15-minute TTL
            return (datetime.now(pytz.timezone('US/Eastern')) - entry_dt).total_seconds() > LIVE_TTL
        else:
            # When market is closed, data is fresh if it was fetched after the last market close
            # This ensures we have the final closing data and keep it until next open
            return entry_dt < last_close

    def _get_path(self, key: str) -> str:
        return os.path.join(CACHE_DIR, f"{key}.json")

    def get(self, ticker: str, expiration: Optional[str]) -> Optional[Dict]:
        """
        Retrieves data from memory or disk. 
        Updates LRU position on memory hit.
        """
        key = self._get_cache_key(ticker, expiration)
        
        with self._lock:
            # 1. Memory Hit
            if key in self._mem_cache:
                entry = self._mem_cache[key]
                if not self._is_expired(entry["timestamp"]):
                    self._mem_cache.move_to_end(key)
                    return entry["data"]
                else:
                    del self._mem_cache[key]

            # 2. Disk Check
            path = self._get_path(key)
            if os.path.exists(path):
                try:
                    mtime = os.path.getmtime(path)
                    if not self._is_expired(mtime):
                        with open(path, 'r') as f:
                            data = json.load(f)
                        # Put in memory for next time
                        self._mem_cache[key] = {
                            "data": data,
                            "timestamp": mtime
                        }
                        self._mem_cache.move_to_end(key)
                        return data
                except Exception:
                    pass
        
        return None

    def put(self, ticker: str, expiration: Optional[str], data: Dict, skip_disk: bool = False) -> None:
        """
        Stores data in memory and optionally disk.
        Evicts oldest entries if over MAX_MEM_ENTRIES.
        """
        key = self._get_cache_key(ticker, expiration)
        
        with self._lock:
            # Update memory
            self._mem_cache[key] = {
                "data": data,
                "timestamp": time.time()
            }
            self._mem_cache.move_to_end(key)
            
            if len(self._mem_cache) > MAX_MEM_ENTRIES:
                self._mem_cache.popitem(last=False)

            # Update disk atomically
            if not skip_disk:
                path = self._get_path(key)
                tmp_path = f"{path}.tmp"
                try:
                    with open(tmp_path, 'w') as f:
                        json.dump(data, f)
                    os.replace(tmp_path, path)
                    # print(f"DEBUG: Wrote {key} to {path}")
                except Exception as e:
                    print(f"DEBUG: Failed to write {key}: {e}")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    def is_fresh(self, ticker: str, expiration: Optional[str]) -> bool:
        """Checks if a fresh cache entry exists without loading full payload."""
        key = self._get_cache_key(ticker, expiration)
        
        # Check memory
        if key in self._mem_cache:
            if not self._is_expired(self._mem_cache[key]["timestamp"]):
                return True
        
        # Check disk
        path = self._get_path(key)
        if os.path.exists(path):
            return not self._is_expired(os.path.getmtime(path))
            
        return False

    def list_cached_expirations(self, ticker: str) -> List[str]:
        """
        Returns a list of YYYY-MM-DD strings for which we have cached data.
        Excludes the 'nearest' pseudo-expiration.
        """
        pattern = os.path.join(CACHE_DIR, f"{ticker.upper()}_*.json")
        files = glob.glob(pattern)
        expirations = []
        for f in files:
            basename = os.path.basename(f)
            # Format is TICKER_YYYYMMDD.json or TICKER_nearest.json
            parts = basename.replace(".json", "").split("_")
            if len(parts) < 2: continue
            date_str = parts[1]
            if date_str == "nearest": continue
            
            # Convert YYYYMMDD to YYYY-MM-DD
            if len(date_str) == 8:
                formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                expirations.append(formatted)
        return sorted(expirations)

# Global singleton
cache_manager = CacheManager()
