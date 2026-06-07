# Finance Tools API

RESTful API for gamma exposure (GEX) and technical indicator analysis with structured JSON responses.

## API Endpoints

### Health Check

```
GET /health
```

Returns server health status.

### Gamma Exposure (GEX)

#### GET Request

```
GET /api/gex/{ticker}?expiration={date}
```

**Parameters:**

- `ticker` (path): Stock ticker symbol (e.g., SPY, QQQ, AAPL)
- `expiration` (query, optional): Expiration date (YYYY-MM-DD, partial, or index)

**Example:**

```bash
curl http://localhost:8000/api/gex/SPY
curl http://localhost:8000/api/gex/SPY?expiration=2026-02-20
```

#### POST Request

```
POST /api/gex
Content-Type: application/json

{
  "ticker": "SPY",
  "expiration": "2026-02-20"
}
```

**Response:**

```json
{
  "ticker": "SPY",
  "expiration": "2026-02-20",
  "spotPrice": 580.25,
  "daysToExpiration": 22,
  "strikes": [
    {
      "strike": 575.0,
      "gexMillions": 125.5,
      "openInterestThousands": 45.2,
      "isATM": false,
      "normalizedGEX": 0.85,
      "normalizedOI": 0.72
    }
  ],
  "maxGEXAbsolute": 147500000.0,
  "maxOpenInterest": 62500.0,
  "availableExpirations": ["2026-01-30", "2026-02-06", "2026-02-20"]
}
```

### Technical Indicators

#### GET Request

```
GET /api/indicators/{ticker}?period={period}&interval={interval}&indicators={list}
```

**Parameters:**

- `ticker` (path): Stock ticker symbol
- `period` (query): Time period (1mo, 3mo, 6mo, 1y, 2y, 5y) - default: 6mo
- `interval` (query): Data interval (1d, 1h, 5m) - default: 1d
- `indicators` (query): Comma-separated list (MACD,OBV,RSI) - default: MACD,OBV,RSI

**Example:**

```bash
curl "http://localhost:8000/api/indicators/AAPL?period=6mo&interval=1d&indicators=MACD,OBV,RSI"
```

#### POST Request

```
POST /api/indicators
Content-Type: application/json

{
  "ticker": "AAPL",
  "period": "6mo",
  "interval": "1d",
  "indicators": ["MACD", "OBV", "RSI"]
}
```

**Response:**

```json
{
  "ticker": "AAPL",
  "period": "6mo",
  "interval": "1d",
  "dataPoints": [
    {
      "date": "2025-08-01",
      "close": 225.5,
      "volume": 45000000,
      "macd": 1.25,
      "macdSignal": 1.1,
      "macdHistogram": 0.15,
      "obv": 1500000000,
      "obvTrend": "Rising",
      "rsi": 65.5,
      "trendSummary": "ACCUMULATION",
      "rsiSignal": "NEUTRAL",
      "macdCrossover": "BULLISH"
    }
  ],
  "trendSegments": [
    {
      "segment": 1,
      "start": "2025-08-01",
      "end": "2025-09-15",
      "duration": "45 days",
      "slope": "25000000",
      "priceChangePct": "+5.25%",
      "obvTrend": "Rising",
      "trendSummary": "STRONG_ACCUMULATION"
    }
  ],
  "averageDailyVolume": 48500000.0,
  "dynamicSlopeThreshold": 1455000.0
}
```

## TypeScript Integration

### Install the Client

Copy `api_types.ts` and `api_client.ts` to your React/TypeScript project.

### Usage Example

```typescript
import { createClient } from "./api_client";

const client = createClient("http://localhost:8000");

// Fetch GEX data
const gexResult = await client.getGEX("SPY");
if (gexResult.success) {
  console.log("Spot Price:", gexResult.data.spotPrice);
  console.log("Strikes:", gexResult.data.strikes);
}

// Fetch technical indicators
const indicatorsResult = await client.getIndicators("AAPL", "6mo", "1d", [
  "MACD",
  "OBV",
  "RSI",
]);
if (indicatorsResult.success) {
  console.log("Data Points:", indicatorsResult.data.dataPoints);
  console.log("Trend Segments:", indicatorsResult.data.trendSegments);
}
```

### React Hook Example

```typescript
import { useState, useEffect } from 'react';
import { createClient } from './api_client';
import type { GEXResponse, ErrorResponse } from './api_types';

function useGEXData(ticker: string, expiration?: string) {
  const [data, setData] = useState<GEXResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ErrorResponse | null>(null);

  useEffect(() => {
    const client = createClient();

    client.getGEX(ticker, expiration).then(result => {
      if (result.success) {
        setData(result.data);
        setError(null);
      } else {
        setError(result.error);
        setData(null);
      }
      setLoading(false);
    });
  }, [ticker, expiration]);

  return { data, loading, error };
}

// Usage in component
function GEXChart({ ticker }: { ticker: string }) {
  const { data, loading, error } = useGEXData(ticker);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.error}</div>;
  if (!data) return null;

  return (
    <div>
      <h2>{data.ticker} - ${data.spotPrice}</h2>
      <p>Expiration: {data.expiration}</p>
      {/* Render chart with data.strikes */}
    </div>
  );
}
```

## File Structure

```
finance-tools/
├── api_server.py          # FastAPI server with endpoints
├── api_handlers.py        # Business logic handlers
├── api_types.py           # Pydantic models (Python)
├── api_types.ts           # TypeScript type definitions
├── api_client.ts          # TypeScript API client
├── visualize_gex.py       # Original GEX visualization (CLI)
├── visualize_indicators.py # Original indicators visualization (CLI)
├── get_technical_indicator.py
├── get_options_data.py
├── get_stock_price.py
└── requirements.txt
```

## Development

### Run with Auto-Reload

```bash
python api_server.py
```

The server will automatically reload when you modify the code.

### CORS Configuration

By default, CORS is configured to allow all origins (`*`). For production, update `api_server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Error Handling

All endpoints return structured error responses:

```json
{
  "error": "Error message",
  "ticker": "AAPL",
  "details": "Additional context"
}
```

HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid ticker, parameters, etc.)
- `500`: Internal server error

## Available Indicators

- `MACD`: Moving Average Convergence Divergence
- `OBV`: On-Balance Volume
- `RSI`: Relative Strength Index
- `SMA`: Simple Moving Average
- `EMA`: Exponential Moving Average
- `BOLLINGER_BANDS`: Bollinger Bands
- `ATR`: Average True Range
- `STOCHASTIC`: Stochastic Oscillator
