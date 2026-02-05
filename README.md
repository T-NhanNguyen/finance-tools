# Docker Headless Finance Tools

Dockerized collection of financial data retrieval and analysis tools. This suite is designed to provide "high-signal" market intelligence directly to AI agents or manual operators with zero API-key friction.

---

## 🚨 Quick Start: Implementation Guide

### 1. Tools Required

- **Docker Desktop**: Essential for running the containerized services.
- **Node.js & npm**: Required if you wish to run the MCP server locally outside of Docker (using `tsx`).
- **Python 3.11+**: Necessary if you prefer running analysis scripts directly on your host machine.

### 2. How to Build and Run Docker

The system is divided into an **Analysis CLI** and a **FastAPI REST Server**.

#### Build the Image

```powershell
# Standard Build
docker-compose build

# Clean Build (if dependency issues occur)
docker-compose build --no-cache
```

#### REST API Walkthrough (api_server.py)

The API server provides structured JSON endpoints, perfect for integrating financial data into custom dashboards or providing high-signal context to LLM apps via standard HTTP calls.

**1. Start the Server**

```bash
docker-compose up api-server
```

The server runs at `http://localhost:8000`. You can access the **interactive API documentation** (Swagger UI) at `http://localhost:8000/docs`.

**2. Key Endpoints for Model Integration**

- **Gamma Exposure (GEX)**: `GET /api/gex/{ticker}`
  - _Usage_: Provides strike-level gamma data, spot price, and expiration details. Essential for identifying "Gamma Walls" or price magnets.
  - _Example_: `curl http://localhost:8000/api/gex/SPY`
- **Technical Indicators**: `GET /api/indicators/{ticker}`
  - _Usage_: Returns time-series data for RSI, MACD, and OBV, including calculated trend segments.
  - _Example_: `curl "http://localhost:8000/api/indicators/TSLA?period=3mo&interval=1d"`

**3. Why use the API?**
While the MCP server is best for autonomous agents, the REST API is optimized for:

- **Frontend Dashboards**: Easily consumed by React/Next.js (see `api_client.ts`).
- **Prompt Augmentation**: Fetch data via server-side logic before sending a prompt to your model provider (OpenAI, Anthropic, etc.).
- **Validation**: All responses are strictly typed via Pydantic (`api_types.py`), ensuring your integration doesn't break due to unexpected data shapes.

#### Run Individual Analysis Scripts (CLI)

```powershell
# Get Stock Prices
docker-compose run --rm finance-tools python get_stock_price.py "AAPL,TSLA,NVDA"

# Get Financial Metrics
docker-compose run --rm finance-tools python get_financial_statement.py "MSFT"

# Get Technical Indicators (Visualizer Data)
docker-compose run --rm finance-tools python visualize_indicators.py "BTC-USD"
```

### 3. AI Agent Integration (MCP)

This project acts as an **MCP (Model Context Protocol) Server**, enabling AI agents to autonomously query market data.

#### Claude Desktop

Add this to your `claude_desktop_config.json` (usually in `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "finance-tools": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "finance-tools"]
    }
  }
}
```

### 4. How `mcp_server.ts` Works

`mcp_server.ts` is the TypeScript orchestration layer that functions as a "Bridge" between the AI agent and the Python logic. This is a work around having to pay for a hosting service for your mcp:

1.  **Tool Registration**: It defines a schema of available tools (`getCurrentPrice`, `getOptionChain`, etc.) that the AI can understand.
2.  **Request Dispatching**: When the AI calls a tool, the server captures the JSON arguments.
3.  **Python Execution**: It uses `child_process.exec` to call `mcp.py` with the specific command (e.g., `python mcp.py "getCurrentPrice({'ticker': 'AAPL'})"`).
4.  **Translation Layer**: `mcp.py` parses the string-based command, routes it to the correct internal Python function (`yfinance`, `pandas`), and returns a minified, high-signal JSON response.
5.  **Stdio Transport**: The final data is sent back to the AI via standard input/output (stdio), which is the standard protocol for MCP communication.

---

## Core Features & Architecture

The system is built on **Vectorized Operations** and **Bulk Query Logic**.

| Feature                  | Command Script               | Signal Type          |
| :----------------------- | :--------------------------- | :------------------- |
| **Real-time Pricing**    | `get_stock_price.py`         | Price & Volume       |
| **Technical Analysis**   | `get_technical_indicator.py` | Momentum & Trends    |
| **Fundamental Analysis** | `get_financial_statement.py` | Yield & Ratios       |
| **Options Intelligence** | `get_options_data.py`        | Liquidity & Vol      |
| **Gamma Exposure**       | `visualize_gex.py`           | Market Maker Hedging |

## Performance Tuning

- **Bulk Optimization**: Functions bypass standard loop-based fetching to reduce "Time-to-Data."
- **Vectorization**: The indicator engine uses NumPy for millisecond calculations on large datasets.
- **JSON/TSON Compression**: MCP responses use minified JSON and TSON (Tabular JSON) to save token context for LLMs.

---

_Created via Antigravity Agency._
