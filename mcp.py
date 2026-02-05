import get_options_data
import get_technical_indicator
import sys
import json
import re
import ast
import get_stock_price
import get_options_data
import pandas as pd
from typing import Any, Dict
import numpy as np
import get_financial_statement
import json_to_tson
import calculate_gamma_delta
from datetime import datetime

# --- Constants ---
DEFAULT_PRECISION = 4
PRICE_PRECISION = 2
COLUMNS_TO_DROP_GENERAL = ['Dividends', 'Stock Splits', 'OBV']
COLUMNS_TO_DROP_OPTIONS = ['index', 'lastTradeDate', 'inTheMoney', 'contractSize', 'currency']
DEFAULT_CURRENCY = "USD"
TOGGLE_MINIFIED_OUTPUT_FORMAT = True  # Set False for pretty-printed output (indent=2)
TOGGLE_TSON_OUTPUT_FORMAT = True     # Set False to bypass TSON tabular compression

HIGH_SIGNAL_METRICS = {
    'income': [
        'Total Revenue', 'Operating Income', 'Net Income', 'Diluted EPS',
        'Gross Profit', 'Research And Development', 'Selling General And Administration'
    ],
    'balance_sheet': [
        'Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity',
        'Invested Capital', 'Total Debt', 'Long Term Debt', 'Commercial Paper',
        'Cash And Cash Equivalents', 'Working Capital'
    ],
    'cash_flow': [
        'Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure',
        'Purchase Of PPE', 'Net Business Purchase And Sale', 'Stock Based Compensation'
    ]
}

def _preprocessData(data: Any) -> Any:
    """Recursively processes data to convert pandas objects, drop unneeded columns, and format numbers."""
    if isinstance(data, (pd.Timestamp, pd.DatetimeIndex)):
        return data.strftime('%Y-%m-%d')

    if isinstance(data, (float, np.float64, np.float32)):
        if pd.isna(data): return None
        val = float(data)
        absVal = abs(val)
        
        # High-signal financial formatting for large values
        if absVal >= 1e12: # Trillions
            return f"{val/1e12:.3f}T"
        if absVal >= 1e9:  # Billions
            return f"{val/1e9:.3f}B"
        if absVal >= 1e6:  # Millions
            return f"{val/1e6:.3f}M"
            
        # Keep smaller values (prices, ratios, tax rates) as precise floats
        return round(val, DEFAULT_PRECISION)

    if isinstance(data, pd.DataFrame):
        # Drop non-price signal columns to save tokens
        colsToDrop = [c for c in COLUMNS_TO_DROP_GENERAL if c in data.columns]
        if colsToDrop:
            data = data.drop(columns=colsToDrop)
        return _preprocessData(data.reset_index().to_dict(orient='records'))
    elif isinstance(data, pd.Series):
        return _preprocessData(data.to_dict())
    elif isinstance(data, dict):
        # Recursively process both keys and values, dropping entries with None values
        processed = {}
        for k, v in data.items():
            k_proc = _preprocessData(k)
            v_proc = _preprocessData(v)
            if k_proc is not None and v_proc is not None:
                processed[k_proc] = v_proc
        return processed
    elif isinstance(data, list):
        # Recursively process elements and drop None/null entries
        processedList = [res for i in data if (res := _preprocessData(i)) is not None]
        # Apply user-provided TSON optimization logic if enabled
        if TOGGLE_TSON_OUTPUT_FORMAT:
            return json_to_tson.convertToTSON(processedList)
        return processedList
    return data

def _formatMcpResponse(data: Any, error: str = None) -> str:
    """Standardized high-signal JSON response for AI agents."""
    # Formatting rules based on developer toggles
    dump_args = {"default": str}
    if TOGGLE_MINIFIED_OUTPUT_FORMAT:
        dump_args["separators"] = (',', ':')
    else:
        dump_args["indent"] = 2

    if error:
        return json.dumps({"error": error}, **dump_args)
    
    processedData = _preprocessData(data)
    # Ensure no scientific notation bleed through
    return json.dumps(processedData, **dump_args)

def _getParams(args: tuple, fieldMap: Dict[str, int]) -> Dict[str, Any]:
    """Extracts parameters from either a single dict or positional args."""
    if len(args) == 1 and isinstance(args[0], dict):
        return args[0]
    
    params = {}
    for field, index in fieldMap.items():
        if len(args) > index:
            params[field] = args[index]
    return params

def routeCommand(commandString: str) -> str:
    """Parses and routes the agent command string."""
    
    # Match pattern: functionName(args)
    match = re.match(r"(\w+)\((.*)\)", commandString)
    if not match:
        return _formatMcpResponse(None, error=f"Invalid command format: '{commandString}'")
    
    functionName = match.group(1)
    argsString = match.group(2)
    
    try:
        
        # Wrap args in a tuple-like string to parse safely
        if argsString.strip():
            # Many LLMs/JSON-based inputs use lowercase true/false/null which ast.literal_eval rejects.
            # Pre-process these common patterns to valid Python literals.
            # We use regex to ensure we only replace keywords, not substrings in ticker names.
            processedArgs = re.sub(r'\btrue\b', 'True', argsString)
            processedArgs = re.sub(r'\bfalse\b', 'False', processedArgs)
            processedArgs = re.sub(r'\bnull\b', 'None', processedArgs)
            
            # Use ast.literal_eval for safe parsing of strings, lists, etc.
            args = ast.literal_eval(f"({processedArgs})")
            if not isinstance(args, tuple):
                args = (args,)
        else:
            args = ()
    except Exception as e:
        return _formatMcpResponse(None, error=f"Failed to parse arguments: {str(e)}")

    # Command Registry
    if functionName == "getCurrentPrice":
        params = _getParams(args, {"ticker": 0})
        ticker = params.get("ticker")
        if not ticker or not isinstance(ticker, str):
            return _formatMcpResponse(None, error="getCurrentPrice expects a 'ticker' string")
        
        price = get_stock_price.getCurrentPrice(ticker)
        if price is None:
             return _formatMcpResponse(None, error=f"Could not retrieve price for {ticker}")
             
        return _formatMcpResponse({
            "ticker": ticker.upper(),
            "price": price,
            "currency": DEFAULT_CURRENCY
        })

    elif functionName == "getCurrentPricesBulk":
        params = _getParams(args, {"tickers": 0})
        tickers = params.get("tickers")
        if not isinstance(tickers, list):
            return _formatMcpResponse(None, error="getCurrentPricesBulk expects a 'tickers' list")
        
        prices = get_stock_price.getCurrentPricesBulk(tickers)
        return _formatMcpResponse({
            "results": prices
        })
    
    elif functionName == "getHistoricalPrices":
        fieldMap = {"ticker": 0, "startDate": 1, "endDate": 2}
        params = _getParams(args, fieldMap)
        ticker = params.get("ticker")
        if not ticker:
            return _formatMcpResponse(None, error="getHistoricalPrices expects a ticker")
        
        prices = get_stock_price.getHistoricalPrices(
            ticker, 
            startDate=params.get("startDate"), 
            endDate=params.get("endDate")
        )
        return _formatMcpResponse({
            "ticker": ticker.upper(),
            "results": prices
        })

    elif functionName == "getHistoricalPricesBulk":
        fieldMap = {"tickers": 0, "startDate": 1, "endDate": 2}
        params = _getParams(args, fieldMap)
        tickers = params.get("tickers")
        if not isinstance(tickers, list):
            return _formatMcpResponse(None, error="getHistoricalPricesBulk expects a tickers list")
            
        prices = get_stock_price.getHistoricalPricesBulk(
            tickers, 
            startDate=params.get("startDate"), 
            endDate=params.get("endDate")
        )
        return _formatMcpResponse({
            "results": prices,
            "currency": DEFAULT_CURRENCY
        })

    elif functionName == "getWykoffIndicator":
        fieldMap = {"ticker": 0, "startDate": 1, "endDate": 2}
        params = _getParams(args, fieldMap)
        ticker = params.get("ticker")
        if not ticker:
            return _formatMcpResponse(None, error="getWykoffIndicator expects a ticker")
        
        # Indicators essential for Wykoff
        indicatorsToCalculate = [
            get_technical_indicator.IndicatorType.SMA,
            get_technical_indicator.IndicatorType.RSI,
            get_technical_indicator.IndicatorType.MACD,
            get_technical_indicator.IndicatorType.OBV
        ]
        
        specificIndicators = get_technical_indicator.getIndicators(
            ticker, 
            indicatorsToCalculate,
            interval=get_technical_indicator.PriceInterval.ONE_DAY,
            period=get_technical_indicator.PricePeriod.SIX_MONTHS,
            startDate=params.get("startDate"),
            endDate=params.get("endDate")
        )
        
        if specificIndicators is not None:
            
            trendSegments = get_technical_indicator.getTrendSegments(specificIndicators)
            return _formatMcpResponse({
                "ticker": ticker.upper(),
                "obvTrendSegments": trendSegments,
                "latestSnapshot": specificIndicators.iloc[-1]
            })

    elif functionName == "getIndicatorsSnapshot":
        params = _getParams(args, {"ticker": 0, "interval": 1})
        ticker = params.get("ticker")
        if not ticker:
            return _formatMcpResponse(None, error="getIndicatorsSnapshot expects a ticker")
            
        intervalStr = params.get("interval", "daily")
        snapShotInterval = (
            get_technical_indicator.PriceInterval.ONE_HOUR 
            if intervalStr == "hourly" 
            else get_technical_indicator.PriceInterval.ONE_DAY
        )

        indicatorsToCalculate = [
            get_technical_indicator.IndicatorType.SMA,
            get_technical_indicator.IndicatorType.RSI,
            get_technical_indicator.IndicatorType.MACD,
            get_technical_indicator.IndicatorType.BOLLINGER_BANDS,
            get_technical_indicator.IndicatorType.OBV
        ]

        specificIndicators = get_technical_indicator.getIndicators(
            ticker, 
            indicatorsToCalculate,
            interval=snapShotInterval,
            period=get_technical_indicator.PricePeriod.THREE_MONTHS
        )
        
        if specificIndicators is not None:
            specificIndicators = get_technical_indicator.generateTradingSignals(specificIndicators)
            
            latestValues = specificIndicators.iloc[-1]
            latestSnapshot = latestValues.drop(labels=[c for c in COLUMNS_TO_DROP_GENERAL if c in latestValues.index])
            
            return _formatMcpResponse({
                "ticker": ticker.upper(),
                "snapshot": latestSnapshot
            })

    elif functionName == "getOptionChain":
        fieldMap = {
            "ticker": 0, 
            "contractType": 1, 
            "expirationDate": 2, 
            "filters": 3, 
            "atmCharacteristic": 4
        }
        params = _getParams(args, fieldMap)
        ticker = params.get("ticker")
        if not ticker:
            return _formatMcpResponse(None, error="getOptionChain expects a ticker")
            
        contractTypeArg = params.get("contractType", "both")
        expirationDate = params.get("expirationDate")
        filters = params.get("filters")
        atmCharacteristic = params.get("atmCharacteristic")
        
        itmOnly = None
        if isinstance(filters, dict):
            itmOnly = filters.get("itmOnly") if filters.get("itmOnly") is not None else filters.get("itmsOnly")

        if isinstance(atmCharacteristic, dict):
            targetStrike = atmCharacteristic.get("targetStrike")
            tolerance = atmCharacteristic.get("tolerance")

        if contractTypeArg and contractTypeArg.lower() == "call":
            contractType = get_options_data.OptionType.CALL
        elif contractTypeArg and contractTypeArg.lower() == "put":
            contractType = get_options_data.OptionType.PUT
        else:
            contractType = get_options_data.OptionType.BOTH

        # Private helper functions
        def _applyFiltersToChain(chain):
            if not isinstance(chain, dict): return chain
            result = {}

            # Have to run through each DataFrame in the dictionary because we can't just pipe it like in get_options_data.py, main
            for key, df in chain.items():
                if df is None or not hasattr(df, 'empty') or df.empty:
                    result[key] = df
                    continue
                
                filteredDf = df
                if isinstance(filters, dict):
                    if filters.get("minVolume") is not None:
                        filteredDf = get_options_data.filterByVolume(filteredDf, minVolume=filters["minVolume"])
                    if filters.get("minOpenInterest") is not None:
                        filteredDf = get_options_data.filterByOpenInterest(filteredDf, minOI=filters["minOpenInterest"])
                    if filters.get("minImpliedVolatility") is not None:
                        filteredDf = get_options_data.filterByImpliedVolatility(filteredDf, minIV=filters["minImpliedVolatility"])
                    if filters.get("maxImpliedVolatility") is not None:
                        filteredDf = get_options_data.filterByImpliedVolatility(filteredDf, maxIV=filters["maxImpliedVolatility"])
                    if itmOnly is not None:
                        filteredDf = get_options_data.filterInTheMoney(filteredDf, itmOnly=itmOnly)
                
                result[key] = filteredDf
            return result

        def _applyATMCharacteristicToChain(chain):
            if not isinstance(chain, dict): return chain
            result = {}

            for key, df in chain.items():
                if df is None or not hasattr(df, 'empty') or df.empty:
                    result[key] = df
                    continue
                
                atmDf = df
                if atmCharacteristic is not None:
                    nearStrikeArgs = {"chainData": atmDf, "ticker": ticker}
                    if atmCharacteristic.get("targetStrike"): 
                        nearStrikeArgs["targetStrike"] = atmCharacteristic["targetStrike"]
                    if atmCharacteristic.get("tolerance"): 
                        nearStrikeArgs["tolerance"] = atmCharacteristic["tolerance"]
                    
                    atmDf = get_options_data.getOptionsNearStrike(**nearStrikeArgs)
                result[key] = atmDf
            return result

        # Resume getOptionChain logic
        optionChain = get_options_data.getOptionChain(ticker, optionType=contractType, expiration=expirationDate)
        filteredChain = _applyFiltersToChain(optionChain)
        filteredChain = _applyATMCharacteristicToChain(filteredChain)
        
        if isinstance(filteredChain, dict):
            for key in filteredChain:
                df = filteredChain[key]
                if df is not None and hasattr(df, 'drop'):
                    filteredChain[key] = df.drop(columns=[c for c in COLUMNS_TO_DROP_OPTIONS if c in df.columns])
        
        return _formatMcpResponse(filteredChain)

    elif functionName == "getFinancialStatements":
        fieldMap = {
            "ticker": 0, 
            "period": 1, 
            "latestReport": 2, 
            "fullData": 3
        }
        params = _getParams(args, fieldMap)
        ticker = params.get("ticker")
        if not ticker:
            return _formatMcpResponse(None, error="getFinancialStatements expects a ticker")
            
        periodArg = params.get("period", "quarterly")
        latestReport = params.get("latestReport", False)
        fullData = params.get("fullData", False)

        # Map string to Enum
        if periodArg.lower() == "quarterly":
            period = get_financial_statement.StatementPeriod.QUARTERLY
        elif periodArg.lower() == "annual":
            period = get_financial_statement.StatementPeriod.ANNUAL
        else:
            return _formatMcpResponse(None, error="getFinancialStatements expects a period of 'quarterly' or 'annual'")

        metrics = get_financial_statement.extractKeyMetrics(ticker, period=period)
        statements = get_financial_statement.getAllStatements(ticker, period=period)

        signals = {}
        if latestReport:
            # Generate growth signals for all whitelist keys before we truncate the dataframes
            for stmt_type, df in statements.items():
                if df is not None and not df.empty and df.shape[1] >= 2:
                    whitelist = HIGH_SIGNAL_METRICS.get(stmt_type, [])
                    for metric in whitelist:
                        if metric in df.index:
                            # yfinance indices can be tricky with ambiguity, handle safely
                            vals = df.loc[metric]
                            if hasattr(vals, 'iloc') and len(vals) >= 2:
                                v0, v1 = vals.iloc[0], vals.iloc[1]
                                if pd.notna(v0) and pd.notna(v1) and v1 != 0:
                                    growth = ((v0 - v1) / abs(v1)) * 100
                                    signals[f"{metric} Growth Percent"] = round(growth, 2)

        # Apply High-Signal Filtering and denoise other information by default
        if not fullData:
            filtered_statements = {}
            for stmt_type, df in statements.items():
                if df is not None:
                    # Filter rows (metrics) by our high-signal whitelist
                    whitelist = HIGH_SIGNAL_METRICS.get(stmt_type, [])
                    # yfinance indices are the metric names
                    existing_keys = [k for k in whitelist if k in df.index]
                    if existing_keys:
                        filtered_statements[stmt_type] = df.loc[existing_keys]
            statements = filtered_statements

        if latestReport:
            for key in statements:
                df = statements[key]
                if df is not None and hasattr(df, 'empty') and not df.empty:
                    # yfinance DataFrames have dates as columns (most recent first)
                    # and metrics as rows. We want the most recent column.
                    statements[key] = df.iloc[:, 0]

        return _formatMcpResponse({
            "ticker": ticker.upper(),
            "signals": signals if latestReport else None,
            "metrics": metrics,
            "statements": statements
        })

    elif functionName == "getContractInfoByStrike":
        fieldMap = {
            "ticker": 0,
            "strikes": 1,
            "expiration": 2,
            "contractType": 3,
            "properties": 4 # ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        }
        params = _getParams(args, fieldMap)
        ticker = params.get("ticker")
        requestedStrikes = params.get("strikes")
        expirationDate = params.get("expiration")
        
        if not ticker or not requestedStrikes or not expirationDate:
            return _formatMcpResponse(None, error="getContractInfoByStrike requires ticker, strikes, and expiration")
        
        if not isinstance(requestedStrikes, list):
            requestedStrikes = [requestedStrikes]
        
        contractTypeArg = params.get("contractType", "both")
        requestedProperties = params.get("properties")
        
        if contractTypeArg and contractTypeArg.lower() == "call":
            optionType = get_options_data.OptionType.CALL
        elif contractTypeArg and contractTypeArg.lower() == "put":
            optionType = get_options_data.OptionType.PUT
        else:
            optionType = get_options_data.OptionType.BOTH
        
        fullChain = get_options_data.getOptionChain(ticker, expiration=expirationDate, optionType=optionType)
        
        # Calculate time to expiration for Greeks
        today = datetime.now()
        expiryDate = datetime.strptime(expirationDate, "%Y-%m-%d")
        timeToExpirationYears = max(1e-6, (expiryDate - today).total_seconds() / (365 * 24 * 3600))
        
        # Get current spot price for Greeks
        stockPrice = get_stock_price.getCurrentPrice(ticker)
        
        contractsList = {}
        
        for chainType in ['calls', 'puts']:
            if chainType not in fullChain or fullChain[chainType] is None:
                contractsList[chainType] = []
                continue
            
            chainDataFrame = fullChain[chainType]
            matchedContracts = chainDataFrame[chainDataFrame['strike'].isin(requestedStrikes)].copy()
            
            if matchedContracts.empty:
                contractsList[chainType] = []
                continue

            # Enrich with Greeks (Delta/Gamma)
            if not matchedContracts.empty and stockPrice:
                strikes = matchedContracts['strike'].values
                ivs = matchedContracts['impliedVolatility'].values
                
                # vectorized calculation for efficiency
                gammas = calculate_gamma_delta.calculateGamma(stockPrice, strikes, timeToExpirationYears, ivs)
                deltas = calculate_gamma_delta.calculateDelta(stockPrice, strikes, timeToExpirationYears, ivs, 'call' if chainType == 'calls' else 'put')
                
                matchedContracts['gamma'] = gammas
                matchedContracts['delta'] = deltas
            
            if requestedProperties:
                # Include Greeks in property filter if requested
                validProps = ['strike', 'delta', 'gamma'] + [p for p in requestedProperties if p in matchedContracts.columns]
                matchedContracts = matchedContracts[list(dict.fromkeys(validProps))] # unique props
            
            contractsAsListOfDicts = matchedContracts.to_dict('records')
            contractsList[chainType] = contractsAsListOfDicts
        
        return _formatMcpResponse({
            "ticker": ticker.upper(),
            "spotPrice": stockPrice,
            "expiration": expirationDate,
            "contracts": contractsList
        })

    else:
        return _formatMcpResponse(None, error=f"Unknown command: {functionName}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(_formatMcpResponse(None, error="No command provided"))
        sys.exit(1)
    
    # Concatenate all arguments incase the shell split the call string
    commandString = " ".join(sys.argv[1:])
    print(routeCommand(commandString))
