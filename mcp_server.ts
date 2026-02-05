import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    ListToolsRequestSchema,
    CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

const server = new Server({
    name: "finance-tools-mcp",
    version: "1.0.0"
}, {
    capabilities: { tools: {} }
});

const tools = [
    {
        name: "getCurrentPrice",
        description: "Get the current market price for a single ticker.",
        inputSchema: {
            type: "object",
            properties: {
                ticker: { type: "string", description: "Stock symbol (e.g., AMZN)" }
            },
            required: ["ticker"]
        }
    },
    {
        name: "getCurrentPricesBulk",
        description: "Get current prices for multiple tickers.",
        inputSchema: {
            type: "object",
            properties: {
                tickers: { type: "array", items: { type: "string" }, description: "List of stock symbols (e.g., [\"AMZN\", \"MSFT\"])", }
            },
            required: ["tickers"]
        }
    },
    {
        name: "getHistoricalPrices",
        description: "Get historical price data for a ticker. Defaults to 1 month of daily data.",
        inputSchema: {
            type: "object",
            properties: {
                ticker: { type: "string", description: "Stock symbol (e.g., AMZN)" },
                startDate: { type: "string", description: "Start date (YYYY-MM-DD)" },
                endDate: { type: "string", description: "End date (YYYY-MM-DD)" }
            },
            required: ["ticker"]
        }
    },
    {
        name: "getHistoricalPricesBulk",
        description: "Get historical price data for multiple tickers.",
        inputSchema: {
            type: "object",
            properties: {
                tickers: { type: "array", items: { type: "string" }, description: "List of stock symbols" },
                startDate: { type: "string", description: "Start date (YYYY-MM-DD)" },
                endDate: { type: "string", description: "End date (YYYY-MM-DD)" }
            },
            required: ["tickers"]
        }
    },
    {
        name: "getWykoffIndicator",
        description: "Get OBV trend segments and Wykoff analysis for a ticker. If no start and end dates are provided, defaults to last 6 months of daily data.",
        inputSchema: {
            type: "object",
            properties: {
                ticker: { type: "string", description: "Stock symbol (e.g., AMZN)" },
                startDate: { type: "string", description: "Start date (YYYY-MM-DD)" },
                endDate: { type: "string", description: "End date (YYYY-MM-DD)" }
            },
            required: ["ticker"]
        }
    },
    {
        name: "getIndicatorsSnapshot",
        description: "Get a snapshot of technical indicators (RSI, MACD, etc.) and signals. If no interval is provided, defaults to daily.",
        inputSchema: {
            type: "object",
            properties: {
                ticker: { type: "string", description: "Stock symbol (e.g., AMZN)" },
                interval: { type: "string", enum: ["hourly", "daily"], description: "Time interval" }
            },
            required: ["ticker"]
        }
    },
    {
        name: "getOptionChain",
        description: "Get filtered option chain data. Results might be missing expiration dates, but that could be extracted from contractSymbol",
        inputSchema: {
            type: "object",
            properties: {
                ticker: { type: "string", description: "Stock symbol (e.g., AMZN)" },
                contractType: { type: "string", enum: ["call", "put", "both"], description: "Type of options" },
                expirationDate: { type: "string", description: "Expiration date (YYYY-MM-DD)" },
                filters: {
                    type: "object",
                    description: "Filters for liquidity and IV",
                    properties: {
                        minVolume: { type: "number" },
                        minOpenInterest: { type: "number" },
                        minImpliedVolatility: { type: "number" },
                        maxImpliedVolatility: { type: "number" },
                        itmOnly: { type: "boolean" }
                    }
                },
                atmCharacteristic: {
                    type: "object",
                    description: "Filters for proximity to ATM",
                    properties: {
                        targetStrike: { type: "number" },
                        tolerance: { type: "number", description: "Tolerance as fraction of price (e.g. 0.05)" }
                    }
                }
            },
            required: ["ticker"]
        }
    },
    {
        name: "getFinancialStatements",
        description: "Retrieve comprehensive financial statements (income, balance sheet, cash flow) and key metrics.",
        inputSchema: {
            type: "object",
            properties: {
                ticker: { type: "string", description: "Stock ticker symbol (e.g., AAPL)" },
                period: {
                    type: "string",
                    enum: ["annual", "quarterly"],
                    description: "Frequency of statements (default: quarterly)"
                },
                latestReport: {
                    type: "boolean",
                    description: "If true, returns only the most recent reporting period instead of the full history."
                },
                fullData: {
                    type: "boolean",
                    description: "If true, returns all available line items. By default, only high-signal investment metrics are returned to save tokens. Avoid using this unless you need all data."
                }
            },
            required: ["ticker"]
        }
    },
    {
        name: "getContractInfoByStrike",
        description: "Lookup comprehensive option contract information for multiple strikes, enriched with Greeks (Delta/Gamma). Use this for quantitative analysis of localized Gamma Exposure (GEX), identifying 'Gamma Walls' or pinning magnets at specific strikes, and assessing market maker hedging pressure.",
        inputSchema: {
            type: "object",
            properties: {
                ticker: { type: "string", description: "Stock ticker symbol (e.g., AAPL)" },
                strikes: {
                    type: "array",
                    items: { type: "number" },
                    description: "List of strike prices to look up."
                },
                expiration: {
                    type: "string",
                    description: "Expiration date in YYYY-MM-DD format (e.g., '2025-12-26')."
                },
                contractType: {
                    type: "string",
                    enum: ["call", "put", "both"],
                    description: "Type of options to retrieve (default: 'both')."
                },
                properties: {
                    type: "array",
                    items: { type: "string" },
                    description: "Specific properties to return: 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'delta', 'gamma'. If omitted, returns all available including Greeks."
                }
            },
            required: ["ticker", "strikes", "expiration"]
        }
    }
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    // Universal dispatcher: Since mcp.py supports a single dictionary argument, 
    // we can pass all tool arguments directly as a JSON-like object.
    const pythonArgs = JSON.stringify(args || {}).replace(/"/g, "'");
    const pythonCmd = `${name}(${pythonArgs})`;

    try {
        // Run the python script. Use double quotes for the outer CLI argument.
        // The mcp.py translation layer will handle lowercase 'true'/'false'/'null' within pythonArgs.
        // Run the python script directly. We assume mcp.py is in the same directory.
        // We use double quotes for the outer shell command to avoid issues with internal nested quotes.
        const { stdout, stderr } = await execAsync(`python mcp.py "${pythonCmd}"`);

        if (stderr && stderr.trim()) {
            // Some libraries write to stderr even on success, so we check if stdout is empty
            if (!stdout.trim()) {
                return { content: [{ type: "text", text: stderr }], isError: true };
            }
        }

        return { content: [{ type: "text", text: stdout }] };
    } catch (error: any) {
        return { content: [{ type: "text", text: error.message }], isError: true };
    }
});

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("Finance Tools MCP Server started");