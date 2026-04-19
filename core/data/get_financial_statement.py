"""
Financial Statement Retrieval Module

This module provides efficient retrieval of financial statements including
income statements, balance sheets, and cash flow statements with bulk query support.
"""

import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Literal, Union
from enum import Enum


class StatementType(Enum):
    """Types of financial statements available"""
    INCOME = "income"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    ALL = "all"


class StatementPeriod(Enum):
    """Period types for financial statements"""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


def getIncomeStatement(
    ticker: Union[str, yf.Ticker],
    period: StatementPeriod = StatementPeriod.ANNUAL
) -> Optional[pd.DataFrame]:
    """
    Get income statement for a single ticker.
    
    Args:
        ticker: Stock symbol or yf.Ticker object
        period: Annual or quarterly data
        
    Returns:
        DataFrame with income statement data, or None if retrieval fails
    """
    try:
        stock = ticker if isinstance(ticker, yf.Ticker) else yf.Ticker(ticker)
        
        if period == StatementPeriod.ANNUAL:
            incomeStatement = stock.income_stmt
        else:
            incomeStatement = stock.quarterly_income_stmt
            
        return incomeStatement if not incomeStatement.empty else None
        
    except Exception as error:
        print(f"Error fetching income statement for {ticker}: {error}")
        return None


def getBalanceSheet(
    ticker: Union[str, yf.Ticker],
    period: StatementPeriod = StatementPeriod.ANNUAL
) -> Optional[pd.DataFrame]:
    """
    Get balance sheet for a single ticker.
    
    Args:
        ticker: Stock symbol or yf.Ticker object
        period: Annual or quarterly data
        
    Returns:
        DataFrame with balance sheet data, or None if retrieval fails
    """
    try:
        stock = ticker if isinstance(ticker, yf.Ticker) else yf.Ticker(ticker)
        
        if period == StatementPeriod.ANNUAL:
            balanceSheet = stock.balance_sheet
        else:
            balanceSheet = stock.quarterly_balance_sheet
            
        return balanceSheet if not balanceSheet.empty else None
        
    except Exception as error:
        print(f"Error fetching balance sheet for {ticker}: {error}")
        return None


def getCashFlowStatement(
    ticker: Union[str, yf.Ticker],
    period: StatementPeriod = StatementPeriod.ANNUAL
) -> Optional[pd.DataFrame]:
    """
    Get cash flow statement for a single ticker.
    
    Args:
        ticker: Stock symbol or yf.Ticker object
        period: Annual or quarterly data
        
    Returns:
        DataFrame with cash flow data, or None if retrieval fails
    """
    try:
        stock = ticker if isinstance(ticker, yf.Ticker) else yf.Ticker(ticker)
        
        if period == StatementPeriod.ANNUAL:
            cashFlow = stock.cashflow
        else:
            cashFlow = stock.quarterly_cashflow
            
        return cashFlow if not cashFlow.empty else None
        
    except Exception as error:
        print(f"Error fetching cash flow statement for {ticker}: {error}")
        return None


def getAllStatements(
    ticker: str,
    period: StatementPeriod = StatementPeriod.ANNUAL
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Get all financial statements for a single ticker in one call.
    More efficient than calling each statement function separately.
    
    Args:
        ticker: Stock symbol
        period: Annual or quarterly data
        
    Returns:
        Dictionary with keys 'income', 'balance_sheet', 'cash_flow'
    """
    statements = {
        'income': None,
        'balance_sheet': None,
        'cash_flow': None
    }
    
    try:
        stock = yf.Ticker(ticker)
        
        # reuse helper functions by passing the initialized stock object
        statements['income'] = getIncomeStatement(stock, period)
        statements['balance_sheet'] = getBalanceSheet(stock, period)
        statements['cash_flow'] = getCashFlowStatement(stock, period)
            
    except Exception as error:
        print(f"Error fetching statements for {ticker}: {error}")
        
    return statements


def getStatementsBulk(
    tickers: List[str],
    statementType: StatementType = StatementType.ALL,
    period: StatementPeriod = StatementPeriod.ANNUAL
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """
    Get financial statements for multiple tickers efficiently.
    Optimized to minimize redundant API calls.
    
    Args:
        tickers: List of stock symbols
        statementType: Type of statement(s) to retrieve
        period: Annual or quarterly data
        
    Returns:
        Nested dictionary: {ticker: {statement_type: DataFrame}}
    """
    bulkStatements = {}
    
    for ticker in tickers:
        if statementType == StatementType.ALL:
            bulkStatements[ticker] = getAllStatements(ticker, period)
        else:
            # Only fetch the requested statement type
            tickerStatements = {}
            
            if statementType == StatementType.INCOME:
                tickerStatements['income'] = getIncomeStatement(ticker, period)
            elif statementType == StatementType.BALANCE_SHEET:
                tickerStatements['balance_sheet'] = getBalanceSheet(ticker, period)
            elif statementType == StatementType.CASH_FLOW:
                tickerStatements['cash_flow'] = getCashFlowStatement(ticker, period)
                
            bulkStatements[ticker] = tickerStatements
            
    return bulkStatements


def safeExtract(df, rowName, colName):
    """
    When pandas has certain index structures, 'Total Revenue' in df.index doesn't return a simple True or False. 
    Instead, it returns a Series or DataFrame with multiple True/False values. 
    This function safely extract a scalar value from a DataFrame.
    
    Handles cases where:
    - Row doesn't exist
    - .loc returns a Series instead of scalar (duplicate indices)
    - Index checking causes ambiguity errors
    """
    try:
        # Use .isin().any() instead of 'in' to avoid DataFrame ambiguity error
        # 'rowName in df.index' can return a DataFrame/Series, causing the error
        # .isin() returns a boolean Series, .any() reduces it to a single bool
        if df.index.isin([rowName]).any():
            value = df.loc[rowName, colName]
            # Handle case where .loc returns a Series (duplicate indices)
            if isinstance(value, pd.Series):
                return value.squeeze() if len(value) == 1 else value.iloc[0]
            return value
        return None
    except (KeyError, IndexError, AttributeError):
        return None


def extractKeyMetrics(ticker: str, period: StatementPeriod = StatementPeriod.ANNUAL) -> Optional[Dict]:
    """
    Extract key financial metrics from statements for quick analysis.
    
    Args:
        ticker: Stock symbol
        period: Annual or quarterly data
        
    Returns:
        Dictionary with key metrics like revenue, net income, total assets, etc.
    """
    try:
        statements = getAllStatements(ticker, period)
        
        hasData = (statements['income'] is not None or 
                   statements['balance_sheet'] is not None or 
                   statements['cash_flow'] is not None)
        
        if not hasData:
            return None
            
        metrics = {}

        # Extract from income statement
        if statements['income'] is not None:
            incomeStmt = statements['income']
            latestColumn = incomeStmt.columns[0]
            prevColumn = incomeStmt.columns[1] if len(incomeStmt.columns) > 1 else None
            
            metrics['revenue'] = safeExtract(incomeStmt, 'Total Revenue', latestColumn)
            metrics['gross_profit'] = safeExtract(incomeStmt, 'Gross Profit', latestColumn)
            metrics['operating_income'] = safeExtract(incomeStmt, 'Operating Income', latestColumn)
            metrics['net_income'] = safeExtract(incomeStmt, 'Net Income', latestColumn)
            metrics['ebitda'] = safeExtract(incomeStmt, 'EBITDA', latestColumn)
            metrics['eps_diluted'] = safeExtract(incomeStmt, 'Diluted EPS', latestColumn)
            
            # Derived Margins
            if metrics.get('revenue'):
                if metrics.get('gross_profit'):
                    metrics['gross_margin_pct'] = (metrics['gross_profit'] / metrics['revenue']) * 100
                if metrics.get('operating_income'):
                    metrics['operating_margin_pct'] = (metrics['operating_income'] / metrics['revenue']) * 100
                if metrics.get('net_income'):
                    metrics['net_margin_pct'] = (metrics['net_income'] / metrics['revenue']) * 100
            
            # Growth (YoY or QoQ based on period)
            if prevColumn:
                prev_rev = safeExtract(incomeStmt, 'Total Revenue', prevColumn)
                if prev_rev and metrics.get('revenue'):
                    metrics['revenue_growth_pct'] = ((metrics['revenue'] - prev_rev) / abs(prev_rev)) * 100
                
                prev_ni = safeExtract(incomeStmt, 'Net Income', prevColumn)
                if prev_ni and metrics.get('net_income'):
                    metrics['net_income_growth_pct'] = ((metrics['net_income'] - prev_ni) / abs(prev_ni)) * 100
            
        # Extract from balance sheet
        if statements['balance_sheet'] is not None:
            balanceSheet = statements['balance_sheet']
            latestColumn = balanceSheet.columns[0]
            metrics['total_assets'] = safeExtract(balanceSheet, 'Total Assets', latestColumn)
            metrics['total_liabilities'] = safeExtract(balanceSheet, 'Total Liabilities Net Minority Interest', latestColumn)
            metrics['total_equity'] = safeExtract(balanceSheet, 'Stockholders Equity', latestColumn)
            metrics['cash'] = safeExtract(balanceSheet, 'Cash And Cash Equivalents', latestColumn)
            metrics['total_debt'] = safeExtract(balanceSheet, 'Total Debt', latestColumn)
            
            # Capital Structure
            if metrics.get('total_assets') and metrics.get('total_liabilities'):
                metrics['debt_to_assets_pct'] = (metrics['total_liabilities'] / metrics['total_assets']) * 100
            if metrics.get('total_equity') and metrics.get('total_debt'):
                metrics['debt_to_equity_pct'] = (metrics['total_debt'] / metrics['total_equity']) * 100
            if metrics.get('total_assets') and metrics.get('cash'):
                metrics['cash_to_assets_pct'] = (metrics['cash'] / metrics['total_assets']) * 100
            
        # Extract from cash flow
        if statements['cash_flow'] is not None:
            cashFlow = statements['cash_flow']
            latestColumn = cashFlow.columns[0]
            metrics['operating_cash_flow'] = safeExtract(cashFlow, 'Operating Cash Flow', latestColumn)
            metrics['free_cash_flow'] = safeExtract(cashFlow, 'Free Cash Flow', latestColumn)
            metrics['capital_expenditure'] = safeExtract(cashFlow, 'Capital Expenditure', latestColumn)
            
            # CF Quality
            if metrics.get('net_income') and metrics.get('operating_cash_flow'):
                metrics['cash_flow_conversion_pct'] = (metrics['operating_cash_flow'] / metrics['net_income']) * 100 if metrics['net_income'] != 0 else None
            
        return metrics
        
    except Exception as error:
        print(f"Error extracting key metrics for {ticker}: {error}")
        return None


def getValuationStats(ticker: str) -> Optional[Dict]:
    """
    Extract valuation and market stats from yfinance info.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with P/E, PEG, Market Cap, Beta, Dividend info, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        stats = {
            "ticker": ticker.upper(),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "enterprise_to_revenue": info.get("enterpriseToRevenue"),
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),
            "five_year_avg_div_yield": info.get("fiveYearAvgDividendYield"),
            "trailing_eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "book_value": info.get("bookValue"),
            "held_percent_institutions": info.get("heldPercentInstitutions"),
            "held_percent_insiders": info.get("heldPercentInsiders"),
            "short_ratio": info.get("shortRatio"),
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_day_average": info.get("fiftyDayAverage"),
            "two_hundred_day_average": info.get("twoHundredDayAverage")
        }
        
        # Clean up None values
        return {k: v for k, v in stats.items() if v is not None}
        
    except Exception as error:
        print(f"Error extracting valuation stats for {ticker}: {error}")
        return None


def compareFinancials(
    tickers: List[str],
    period: StatementPeriod = StatementPeriod.ANNUAL
) -> Optional[pd.DataFrame]:
    """
    Compare key financial metrics across multiple companies.
    
    Args:
        tickers: List of stock symbols to compare
        period: Annual or quarterly data
        
    Returns:
        DataFrame with companies as columns and metrics as rows
    """
    try:
        comparisonData = {}
        
        for ticker in tickers:
            metrics = extractKeyMetrics(ticker, period)
            if metrics:
                comparisonData[ticker] = metrics
                
        if not comparisonData:
            return None
            
        comparisonDf = pd.DataFrame(comparisonData)
        return comparisonDf
        
    except Exception as error:
        print(f"Error creating financial comparison: {error}")
        return None


# Example usage
if __name__ == "__main__":
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(description="Financial Statement Tool")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AMZN)")
    parser.add_argument("--mcp", action="store_true", help="Output in structured JSON for MCP usage")
    parser.add_argument("--period", choices=['annual', 'quarterly'], default='quarterly', help="Financial period")
    
    args = parser.parse_args()
    period = StatementPeriod.QUARTERLY if args.period == 'quarterly' else StatementPeriod.ANNUAL
    
    if args.mcp:
        # High-signal output for AI agents
        metrics = extractKeyMetrics(args.ticker)
        statements = getAllStatements(args.ticker, period=period)
        
        mcpOutput = {
            "ticker": args.ticker.upper(),
            "period": args.period,
            "metrics": metrics,
            "metadata": {
                "source": "Yahoo Finance (via yfinance)",
                "status": "success" if metrics else "failed"
            }
        }
        
        # Add basic growth signal if we have the income statement
        income = statements.get('income')
        if income is not None and not income.empty and income.shape[1] >= 2:
            try:
                # Calculate simple YoY/QoQ revenue growth from columns
                # Columns are dates (latest first)
                latest_rev = income.loc['Total Revenue'].iloc[0]
                prev_rev = income.loc['Total Revenue'].iloc[1]
                growth = ((latest_rev - prev_rev) / prev_rev) * 100
                mcpOutput["signals"] = {
                    "revenue_growth_percent": round(growth, 2)
                }
            except:
                pass
                
        print(json.dumps(mcpOutput, indent=2))
        
    else:
        # Traditional human-readable output
        print(f"Financial Statements for {args.ticker.upper()}")
        tickerStatements = getAllStatements(args.ticker, period=period)
        
        if tickerStatements['income'] is not None:
            print("\nIncome Statement (latest 4 periods):")
            print(tickerStatements['income'].head(10))
        
        print("\nKey Metrics Extraction")
        tickerMetrics = extractKeyMetrics(args.ticker)
        if tickerMetrics:
            for metric, value in tickerMetrics.items():
                if value is not None:
                    if isinstance(value, float) and abs(value) > 1000:
                        print(f"{metric}: ${value:,.0f}")
                    else:
                        print(f"{metric}: {value:.2f}" if isinstance(value, float) else f"{metric}: {value}")
        
        print("\nFinancial Comparison (Tech Giants)")
        techCompanies = ["IREN", "NBIS", "CRWV"]
        comparison = compareFinancials(techCompanies)
        
        if comparison is not None:
            print("\nKey metrics comparison:")
            # Use columns that actually exist in the comparison
            avail_metrics = [m for m in ['revenue', 'net_income', 'profit_margin', 'total_assets'] if m in comparison.index]
            print(comparison.loc[avail_metrics])
