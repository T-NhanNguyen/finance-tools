# Unified Ticker Cache Architecture — Master Plan

## Problem Summary

Three critical flaws in the current architecture:

1. **Price lookup bug** — `spot_price` on ticker cards is sourced from portfolio simulation positions only. Tickers dropped by the margin knapsack never get hydrated, showing `---`.
2. **Silo'd data fetching** — GEX page and Option Selling page independently fetch and cache the same option chains with no shared data layer.
3. **Redundant re-computation** — Every expiration change re-fires N separate `/api/contract-selling` POST requests, re-fetching chains and re-running pillar scoring from scratch.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         BACKEND (finance-tools)                       │
│                                                                        │
│  POST /api/queue-tickers   ─── CacheManager ─── .gex_cache/ files    │
│  GET  /api/option-chain/{ticker}  (cache read)                        │
│  POST /api/option-analysis/batch  (compute-only, no I/O)              │
└──────────────────────────────────────────────────────────────────────┘
          ▲                       ▲
          │                       │
  GEX page                 Option Selling page
  (synchronous-feel,        (queue → expiry switch → batch compute)
   backed by cache)
```

---

## Resolved Design Decisions

| # | Question | Decision |
|---|---|---|
| Q1 | Keep `/api/contract-selling`? | **Remove it.** New batch endpoint is the enforced pattern. |
| Q2 | `queue-tickers` blocking or async? | **Return immediately.** Users see first-expiration data right away; remaining expirations fill in background. |
| Q3 | GEX page fetch feel? | **Synchronous-feel backed by cache.** Single ticker → call queue + immediately read chain (no polling UX). |

---

## Split Engineer Assignments

This work is divided into two independent tracks that can be parallelized after an agreed contract on the new API response shapes.

| Track | File | Owner |
|---|---|---|
| Backend | [backend_plan.md](file:///Users/nhan/.gemini/antigravity/brain/ffb6e955-1812-4699-a168-35a4029468bd/backend_plan.md) | Backend Engineer |
| Frontend | [frontend_plan.md](file:///Users/nhan/.gemini/antigravity/brain/ffb6e955-1812-4699-a168-35a4029468bd/frontend_plan.md) | Frontend Engineer |

### API Contract (shared between both engineers)

Both engineers must agree on these shapes before starting:

**`POST /api/queue-tickers`**
```jsonc
// Request
{ "tickers": ["NVDA", "SPY"], "expiration": null }

// Response
{
  "NVDA": { "status": "fetching", "cached_expirations": [] },
  "SPY":  { "status": "cached",   "cached_expirations": ["2025-04-18", "2025-05-02"] }
}
```

**`GET /api/option-chain/{ticker}?expiration=YYYY-MM-DD`**
```jsonc
// Response — same shape as existing GEXResponse (no breaking change for GEX page)
{ "ticker": "NVDA", "spotPrice": 875.00, "expiration": "2025-04-18", "strikes": [...], ... }
// 404 body when not yet cached:
{ "status": "queued", "ticker": "NVDA" }
```

**`POST /api/option-analysis/batch`**
```jsonc
// Request
{
  "tickers": ["NVDA", "SPY"],
  "expiration": "2025-04-18",
  "strategy": "CSP",
  "cash_equity": 1420500
}

// Response
{
  "NVDA": {
    "ticker": "NVDA", "spot_price": 875.00, "expiration": "2025-04-18",
    "atm_premium_benchmark": 12.40, "effective_capital": 1200000,
    "init_req": 0.20, "maint_req": 0.15,
    "pillars": { "Top_Wheel_Engine": [...], "Top_Cash_Engine": [...] }
  },
  "SPY": { ... }
}
```

---

## Data Flow (After)

```
1. User adds NVDA, AAPL, SPY
        │
        ▼
  POST /api/queue-tickers → returns immediately with per-ticker status
        │  Background: fetch_all_expirations() writes NVDA_*.json, AAPL_*.json ...
        ▼
2. User selects expiration "2025-04-18"
        │
        ▼
  POST /api/option-analysis/batch → CacheManager.get() × N (no I/O) → compute pillars
  POST /api/portfolio-simulation  → unchanged
        │
        ▼
  Frontend: spot_price sourced from batch result per ticker (bug fixed)

3. User switches to "2025-05-02"
        │
        ▼
  POST /api/option-analysis/batch (same tickers, new expiration)
  Backend: already in memory → pure re-compute ~50ms
```
