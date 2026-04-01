# Backend Engineer Plan — Unified Ticker Cache

**Repo:** `finance-tools`
**Dependency:** Frontend engineer must not call the old `/api/contract-selling` endpoint once this branch ships.

---

## Files to Change

| Status | File | Summary |
|---|---|---|
| NEW | `api/cache_manager.py` | In-process LRU singleton on top of file cache |
| MODIFY | `core/data/bulk_data_loader.py` | Add `fetch_all_expirations`, `get_cached_option_chain` |
| MODIFY | `api/api_handlers.py` | Add 3 new handlers, remove `getContractSellingData` standalone re-fetch path |
| MODIFY | `api/api_server.py` | Register 3 new routes, remove `/api/contract-selling` route |
| MODIFY | `api/api_types.py` | Add request/response Pydantic models for new endpoints |
| MODIFY | `core/strategies/contract_selling_analyst.py` | Add `scan_from_chain(data, ...)` overload |

---

## Step 1 — `api/cache_manager.py` (NEW)

Single module-level singleton shared across all uvicorn worker threads.

```python
CACHE_DIR = ...              # same path as bulk_data_loader.CACHE_DIR
CACHE_EXPIRY = 900           # seconds, matches existing constant
MAX_MEM_ENTRIES = 50         # LRU cap

class CacheManager:
    def get(ticker: str, expiration: str | None) -> dict | None
        # 1. Check _mem_cache; if hit and fresh → return
        # 2. Try load from .gex_cache file; if fresh → populate _mem_cache, return
        # 3. Return None (cache miss)

    def put(ticker: str, expiration: str, data: dict) -> None
        # Write to _mem_cache + disk (delegates to _save_to_cache)
        # Evict LRU entry if over MAX_MEM_ENTRIES

    def is_fresh(ticker: str, expiration: str | None) -> bool
        # Check file mtime < CACHE_EXPIRY without loading full payload

    def list_cached_expirations(ticker: str) -> list[str]
        # Glob CACHE_DIR for TICKER_*.json and return parsed expiration strings

cache_manager = CacheManager()  # module-level singleton
```

**Key:** `expiration=None` key resolves to `"nearest"` in the cache filename, identical to existing `bulk_data_loader._get_cache_path` behavior — no file format change.

---

## Step 2 — `core/data/bulk_data_loader.py` (MODIFY)

Add two functions, keep existing `fetch_gex_single` and `fetch_gex_bulk` unchanged.

```python
def fetch_all_expirations(ticker: str) -> dict[str, str]:
    """
    Fetches the nearest expiration first (to get availableExpirations list),
    then spawns background threads for the remaining expirations.
    Returns immediately with { expiration_date: "fetching"|"cached"|"error" } status map.
    """

def get_cached_option_chain(ticker: str, expiration: str | None) -> dict | None:
    """
    Pure cache read — delegates to cache_manager.get().
    Returns None on miss; does NOT trigger a live fetch.
    """
```

**`fetch_all_expirations` implementation sketch:**
1. Call `fetch_gex_single(ticker, None)` (nearest) — this blocks briefly, populates cache.
2. Extract `data["availableExpirations"]`.
3. For remaining dates not yet cached, submit to a `ThreadPoolExecutor` with `max_workers=4`.
4. Return a status dict immediately (remaining threads run in background).

---

## Step 3 — `api/api_types.py` (MODIFY)

Add Pydantic models:

```python
class QueueTickersRequest(BaseModel):
    tickers: list[str]
    expiration: Optional[str] = None  # None = fetch all expirations

class TickerQueueStatus(BaseModel):
    status: Literal["cached", "fetching", "error"]
    cached_expirations: list[str]

# QueueTickersResponse = dict[str, TickerQueueStatus]  (return as plain dict)

class BatchAnalysisRequest(BaseModel):
    tickers: list[str]
    expiration: str
    strategy: str = "CSP"
    cash_equity: Optional[float] = None

# BatchAnalysisResponse = dict[str, ContractSellingResponse]
```

The `GET /api/option-chain/{ticker}` response reuses the existing `GEXResponse` model — no new type needed.

---

## Step 4 — `api/api_handlers.py` (MODIFY)

### 4a. Add `queueTickers` handler

```python
def queueTickers(tickers: list[str], expiration: Optional[str]) -> dict:
    result = {}
    for ticker in [t.upper() for t in tickers]:
        cached_exps = cache_manager.list_cached_expirations(ticker)
        if cache_manager.is_fresh(ticker, expiration):
            result[ticker] = {"status": "cached", "cached_expirations": cached_exps}
        else:
            # fire-and-forget background fetch
            threading.Thread(
                target=fetch_all_expirations, args=(ticker,), daemon=True
            ).start()
            result[ticker] = {"status": "fetching", "cached_expirations": cached_exps}
    return result
```

### 4b. Add `getOptionChain` handler

```python
def getOptionChain(ticker: str, expiration: Optional[str]) -> dict | None:
    data = get_cached_option_chain(ticker.upper(), expiration)
    if data is None:
        return None  # caller returns HTTP 404
    return GEXResponse(**data).model_dump()
```

### 4c. Add `batchAnalyzeContracts` handler

```python
def batchAnalyzeContracts(
    tickers: list[str],
    expiration: str,
    strategy: str,
    cash_equity: float
) -> dict:
    results = {}
    analyst = ContractSellingAnalyst(cash_equity=cash_equity)

    def process(ticker):
        chain = get_cached_option_chain(ticker, expiration)
        if chain is None:
            return ticker, {"error": "chain not cached", "ticker": ticker}
        result = analyst.scan_from_chain(chain, strategy_type=strategy)
        return ticker, ContractSellingResponse(**result).model_dump()

    with ThreadPoolExecutor(max_workers=8) as pool:
        for ticker, result in pool.map(lambda t: process(t), [t.upper() for t in tickers]):
            results[ticker] = result

    return results
```

### 4d. Remove `getContractSellingData` re-fetch path

Wire `getContractSellingData` through `CacheManager` instead of calling `fetch_gex_single` directly, so it benefits from cache even when called in isolation (e.g., from tests).

---

## Step 5 — `core/strategies/contract_selling_analyst.py` (MODIFY)

Add `scan_from_chain` as a pure-compute overload that accepts pre-loaded data:

```python
def scan_from_chain(
    self,
    chain_data: dict,          # pre-loaded from CacheManager
    strategy_type: str = "CSP",
    engine_mode: str = "BOTH"
) -> dict:
    """
    Same logic as scan() but skips the fetch_gex_single() call.
    chain_data must be a valid GEX data dict (same shape as fetch_gex_data_raw output).
    """
    # Extract spot_price, strikes, daysToExpiration from chain_data
    # Run analyze_strike() loop + get_actionable_pillars()
    # Return same dict shape as scan()
```

The existing `scan()` method is updated to delegate to `scan_from_chain` after fetching:

```python
def scan(self, ticker, expiration_input, strategy_type, ...):
    data = fetch_gex_single(ticker, expiration_input)  # unchanged
    if "error" in data:
        return data
    return self.scan_from_chain(data, strategy_type, ...)  # delegate
```

---

## Step 6 — `api/api_server.py` (MODIFY)

```python
# REMOVE
@app.post("/api/contract-selling", ...)

# ADD
@app.post("/api/queue-tickers")
async def queue_tickers(request: QueueTickersRequest, ...):
    return queueTickers(request.tickers, request.expiration)

@app.get("/api/option-chain/{ticker}")
async def get_option_chain(ticker: str, expiration: Optional[str] = Query(None), ...):
    result = getOptionChain(ticker, expiration)
    if result is None:
        raise HTTPException(status_code=404, detail={"status": "queued", "ticker": ticker})
    return result

@app.post("/api/option-analysis/batch")
async def batch_analyze(request: BatchAnalysisRequest, ...):
    equity = request.cash_equity if request.cash_equity is not None else sum(LENDERS)
    result = batchAnalyzeContracts(
        request.tickers, request.expiration, request.strategy, equity
    )
    return result
```

All three routes use the existing `api_dependencies` (API secret auth).

---

## Verification

```bash
# 1. Queue tickers (should return immediately)
curl -s -X POST http://localhost:8000/api/queue-tickers \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["NVDA", "SPY"]}' | python3 -m json.tool

# 2. Wait ~3s, then read chain from cache
curl -s "http://localhost:8000/api/option-chain/NVDA?expiration=2025-04-18" \
  | python3 -m json.tool | head -20

# 3. Batch analysis — must return spot_price per ticker
curl -s -X POST http://localhost:8000/api/option-analysis/batch \
  -H "Content-Type: application/json" \
  -d '{"tickers":["NVDA","SPY"],"expiration":"2025-04-18","strategy":"CSP","cash_equity":1420500}' \
  | python3 -m json.tool | grep -E '"ticker"|"spot_price"'

# 4. Confirm /api/contract-selling is gone (should 404 or 405)
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:8000/api/contract-selling \
  -H "Content-Type: application/json" \
  -d '{"ticker":"NVDA"}'
```

**Cache file check:**
```bash
ls -lh /path/to/finance-tools/.gex_cache/NVDA_*.json
# Should see one file per expiration date after queue-tickers completes
```

---

## Notes for Frontend Engineer

- The `GET /api/option-chain/{ticker}` response shape is identical to the existing `/api/gex/{ticker}` response — no changes needed on existing GEX page type imports.
- A `404` with body `{ "status": "queued" }` means the chain is still being fetched; frontend should retry after a short delay.
- The `/api/option-analysis/batch` response is a flat `dict[ticker → ContractSellingResponse]` — one key per requested ticker. If a ticker's chain is missing, its value will contain `{ "error": "chain not cached" }`.
