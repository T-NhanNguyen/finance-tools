import os
import json
import time
import glob
from typing import Dict, List, Optional
from collections import OrderedDict

# Constants mirrored from bulk_data_loader to avoid circular imports
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".gex_cache"))
CACHE_EXPIRY = 900  # 15 minutes
MAX_MEM_ENTRIES = 50

class CacheManager:
    """
    In-process LRU cache on top of the filesystem .gex_cache.
    Ensures hot tickers are served from memory while persisting all fetches to disk.
    """
    def __init__(self):
        self._mem_cache: OrderedDict[str, Dict] = OrderedDict()
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)

    def _get_cache_key(self, ticker: str, expiration: Optional[str]) -> str:
        """Standardized key for both memory and disk lookups."""
        exp_hash = expiration if expiration else "nearest"
        return f"{ticker.upper()}_{exp_hash.replace('-', '')}"

    def _get_path(self, key: str) -> str:
        return os.path.join(CACHE_DIR, f"{key}.json")

    def get(self, ticker: str, expiration: Optional[str]) -> Optional[Dict]:
        """
        Retrieves data from memory or disk. 
        Updates LRU position on memory hit.
        """
        key = self._get_cache_key(ticker, expiration)
        
        # 1. Memory Hit
        if key in self._mem_cache:
            entry = self._mem_cache[key]
            if (time.time() - entry["timestamp"]) < CACHE_EXPIRY:
                self._mem_cache.move_to_end(key)
                return entry["data"]
            else:
                del self._mem_cache[key]

        # 2. Disk Check
        path = self._get_path(key)
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if (time.time() - mtime) < CACHE_EXPIRY:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    self.put(ticker, expiration, data, skip_disk=True)
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
        
        # Update memory
        self._mem_cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
        self._mem_cache.move_to_end(key)
        
        if len(self._mem_cache) > MAX_MEM_ENTRIES:
            self._mem_cache.popitem(last=False)

        # Update disk
        if not skip_disk:
            path = self._get_path(key)
            try:
                with open(path, 'w') as f:
                    json.dump(data, f)
            except Exception:
                pass

    def is_fresh(self, ticker: str, expiration: Optional[str]) -> bool:
        """Checks if a fresh cache entry exists without loading full payload."""
        key = self._get_cache_key(ticker, expiration)
        
        # Check memory
        if key in self._mem_cache:
            if (time.time() - self._mem_cache[key]["timestamp"]) < CACHE_EXPIRY:
                return True
        
        # Check disk
        path = self._get_path(key)
        if os.path.exists(path):
            return (time.time() - os.path.getmtime(path)) < CACHE_EXPIRY
            
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
