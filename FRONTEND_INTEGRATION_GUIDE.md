# Frontend Integration: Finance Tools API

This document provides the context and steps needed to link the frontend to the **Finance Tools** backend running on Vercel Serverless Functions.

## 1. Context & Setup

The Finance Tools backend provides Gamma Exposure (GEX) and Technical Indicator analysis. It is currently configured to run as a serverless function on Vercel.

### Project Files

Ensure the following files are copied from the backend repository to your frontend project (e.g., in `src/api/` or `src/lib/`):

- `api_types.ts`: TypeScript interfaces for the API responses.
- `api_client.ts`: The HTTP client class for making requests.

## 2. Environment Variables

To avoid naming collisions with other backends (like your main API), this project uses a specific environment variable prefix.

### Vercel Dashboard (Production)

In your frontend project's **Environment Variables** settings on Vercel, add:

- **Key**: `NEXT_PUBLIC_FINANCE_API_URL`
- **Value**: `https://your-finance-backend.vercel.app` (e.g., your deployed Python API URL)

### Local Development

Add this to your `.env.local` (Next.js) or `.env` (Vite):

```bash
NEXT_PUBLIC_FINANCE_API_URL=http://localhost:8000
```

## 3. Implementation

### Initialize the Client

The `createClient` function has been updated to automatically detect the unique finance environment variable.

```typescript
import { createClient } from "./api_client";

// Automatically uses NEXT_PUBLIC_FINANCE_API_URL or VITE_FINANCE_API_URL
const client = createClient();
```

### Fetching GEX Data

```typescript
const result = await client.getGEX("SPY");

if (result.success) {
  const { spotPrice, strikes } = result.data;
  // spotPrice: number
  // strikes: GEXStrikeData[]
} else {
  console.error(result.error);
}
```

### Fetching Technical Indicators (MACD, OBV, RSI)

```typescript
const result = await client.getIndicators("AAPL", "6mo", "1d", [
  "MACD",
  "OBV",
  "RSI",
]);

if (result.success) {
  const { dataPoints, trendSegments } = result.data;
  // dataPoints: IndicatorDataPoint[] (Time series)
  // trendSegments: OBVTrendSegment[] (OBV analysis)
}
```

## 4. Troubleshooting & Notes

- **Vercel Timeout (10s)**: On the Hobby plan, individual requests have a 10s limit. Large data fetches (e.g., 5y period) might occasionally time out.
- **CORS**: Currently set to `allow_origins=["*"]`. If you see CORS errors in the browser, ensure the backend's `api_server.py` allows your frontend's specific domain.
- **Node Types**: If your IDE flags `process.env` errors, ensure you have `@types/node` installed (`npm i --save-dev @types/node`).
