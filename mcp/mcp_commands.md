# Finance Tools MCP Commands

These commands are optimized for AI agents and human testing. They provide structured, high-signal JSON data.

## Running the MCP Server

To use these tools with an LLM (LM Studio, Claude Desktop, etc.) or an AI CLI, you can create an entry point (host) with the following:

```bash
docker-compose --rm run finance-tools
```

To add the MCP to your AI, in this case gemini, run

```bash
gemini mcp add finance-tools docker-compose run finance-tools
```

## Manual Testing via CLI

You can test individual commands using the `mcp.py` entry point. The new **Flexible Schema** allows you to pass arguments as a single dictionary.

### Fetch Current Price

```bash
docker-compose --rm run finance-tools python mcp.py "getCurrentPrice({'ticker': 'AMZN'})"
```

### Fetch Historical Prices

```bash
docker-compose --rm run finance-tools python mcp.py "getHistoricalPrices({'ticker': 'AAPL', 'startDate': '2023-01-01'})"
```

### Fetch Wykoff Trend Analysis

Provides OBV trend segments and technical snapshots for the last 6 months.

```bash
docker-compose --rm run finance-tools python mcp.py "getWykoffIndicator({'ticker': 'MSFT'})"
```

### Fetch Technical Snapshot

Get RSI, MACD, and Trading Signals (Defaults to daily).

```bash
docker-compose --rm run finance-tools python mcp.py "getIndicatorsSnapshot({'ticker': 'TSLA', 'interval': 'hourly'})"
```

### Fetch Filtered Option Chain

The most powerful command. Supports cumulative filtering for liquidity and proximity to the money.

```bash
docker-compose --rm run finance-tools python mcp.py "getOptionChain({'ticker': 'UBER', 'contractType': 'call', 'filters': {'minVolume': 500, 'minOpenInterest': 1000}})"
```

### Fetch Financial Statements

To test the script directly (bypassing the MCP server entrypoint):

```bash
docker-compose run finance-tools python mcp.py "getFinancialStatements({'ticker': 'AAPL', 'period': 'quarterly', 'latestReport': true})"
```

## Command reference

- `getCurrentPrice(ticker)`
- `getCurrentPricesBulk(tickers)`
- `getHistoricalPrices(ticker, startDate, endDate)`
- `getHistoricalPricesBulk(tickers, startDate, endDate)`
- `getWykoffIndicator(ticker, startDate, endDate)`
- `getIndicatorsSnapshot(ticker, interval)`
- `getOptionChain(ticker, contractType, expirationDate, filters, atmCharacteristic)`
- `getFinancialStatements(ticker, period, latestReport)`
