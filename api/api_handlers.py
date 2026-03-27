"""
API Handler Functions
Converts visualization scripts to return structured JSON data for HTTP responses.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from core.data import (
    getOptionChain, getOptionExpirations, getCurrentPrice, PricePeriod, PriceInterval,
    getIndicators, IndicatorType, generateTradingSignals, getTrendSegments, _getPiecewiseBoundaries, calculateTheilSenSlope, MINIMUM_SEGMENT_LENGTH
)
from core.analysis import calculateGamma
from visualizers.visualize_indicators import SLOPE_SENSITIVITY_RATIO
from core.data.gex_provider import fetch_gex_data_raw, parse_flexible_date

from core.strategies.contract_selling_analyst import ContractSellingAnalyst
from core.strategies.portfolio_margin_allocator import (
    PortfolioMarginAllocator, 
    Position, 
    optimize_allocation, 
    simulate_multi_asset_portfolio
)
from core.strategies.strategy_config import LENDERS

from .api_types import (
    GEXResponse, GEXStrikeData, IndicatorsResponse, 
    IndicatorDataPoint, OBVTrendSegment, ErrorResponse,
    ContractSellingResponse, StrikeAnalysisData, PillarScoredPoint, ActionablePillars,
    PortfolioMarginResponse, PortfolioPositionData
)


# ============================================================================
# GEX Handler
# ============================================================================

def getGEXData(ticker: str, expiration_input: Optional[str] = None) -> Dict:
    """
    Fetch GEX (Gamma Exposure) data and return structured JSON.
    Delegates core fetching and processing to core.data.gex_provider.
    """
    try:
        data = fetch_gex_data_raw(ticker, expiration_input)
        
        if "error" in data:
            return ErrorResponse(
                error=data["error"],
                ticker=data.get("ticker", ticker),
                details=data.get("details")
            ).model_dump()
            
        # Wrap the raw data in GEXResponse model
        strikes = [GEXStrikeData(**s) for s in data["strikes"]]
        
        return GEXResponse(
            ticker=data["ticker"],
            expiration=data["expiration"],
            spotPrice=data["spotPrice"],
            daysToExpiration=data["daysToExpiration"],
            strikes=strikes,
            maxGEXAbsolute=data["maxGEXAbsolute"],
            maxOpenInterest=data["maxOpenInterest"],
            availableExpirations=data["availableExpirations"]
        ).model_dump()
        
    except Exception as exc:
        return ErrorResponse(
            error="Internal server error in getGEXData",
            ticker=ticker,
            details=str(exc)
        ).model_dump()


# ============================================================================
# Technical Indicators Handler
# ============================================================================

def getIndicatorsData(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    indicators: Optional[List[str]] = None
) -> Dict:
    """
    Fetch technical indicators and return structured JSON.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1mo, 3mo, 6mo, 1y, etc.)
        interval: Data interval (1d, 1h, etc.)
        indicators: List of indicator names (MACD, OBV, RSI, etc.)
        
    Returns:
        Dictionary matching IndicatorsResponse schema or ErrorResponse schema
    """
    try:
        ticker = ticker.upper()
        
        # Parse period
        periodMap = {
            "1d": PricePeriod.ONE_DAY,
            "5d": PricePeriod.FIVE_DAYS,
            "1mo": PricePeriod.ONE_MONTH,
            "3mo": PricePeriod.THREE_MONTHS,
            "6mo": PricePeriod.SIX_MONTHS,
            "1y": PricePeriod.ONE_YEAR,
            "2y": PricePeriod.TWO_YEARS,
            "5y": PricePeriod.FIVE_YEARS,
            "ytd": PricePeriod.YEAR_TO_DATE,
            "max": PricePeriod.MAX,
        }
        # Handle case-insensitive and common variants
        pricePeriod = periodMap.get(period.lower(), PricePeriod.SIX_MONTHS)
        
        # Parse interval
        intervalMap = {
            "1m": PriceInterval.ONE_MINUTE,
            "2m": PriceInterval.TWO_MINUTES,
            "5m": PriceInterval.FIVE_MINUTES,
            "15m": PriceInterval.FIFTEEN_MINUTES,
            "30m": PriceInterval.THIRTY_MINUTES,
            "60m": PriceInterval.SIXTY_MINUTES,
            "1h": PriceInterval.ONE_HOUR,
            "1d": PriceInterval.ONE_DAY,
            "1wk": PriceInterval.ONE_WEEK,
            "1mo": PriceInterval.ONE_MONTH,
        }
        priceInterval = intervalMap.get(interval.lower(), PriceInterval.ONE_DAY)
        
        # Parse indicators
        if indicators is None:
            indicators = ["MACD", "OBV", "RSI"]
        
        indicatorTypeMap = {
            "SMA": IndicatorType.SMA,
            "EMA": IndicatorType.EMA,
            "RSI": IndicatorType.RSI,
            "MACD": IndicatorType.MACD,
            "BOLLINGER_BANDS": IndicatorType.BOLLINGER_BANDS,
            "ATR": IndicatorType.ATR,
            "STOCHASTIC": IndicatorType.STOCHASTIC,
            "OBV": IndicatorType.OBV,
        }
        
        indicatorsToCalculate = [
            indicatorTypeMap[ind.upper()] 
            for ind in indicators 
            if ind.upper() in indicatorTypeMap
        ]
        
        if not indicatorsToCalculate:
            return ErrorResponse(
                error="No valid indicators specified",
                ticker=ticker,
                details=f"Requested: {indicators}"
            ).model_dump()
        
        # Fetch data with indicators
        df = getIndicators(ticker, indicatorsToCalculate, period=pricePeriod, interval=priceInterval)
        
        if df is None or df.empty:
            return ErrorResponse(
                error="Could not fetch data",
                ticker=ticker,
                details=f"Period: {period}, Interval: {interval}"
            ).model_dump()
        
        # Generate trading signals
        df = generateTradingSignals(df)
        
        # Calculate dynamic slope threshold
        averageDailyVolume = float(df['Volume'].mean())
        dynamicSlopeThreshold = float(averageDailyVolume * SLOPE_SENSITIVITY_RATIO)
        
        # Get trend segments
        trendSegments = getTrendSegments(df)
        
        
        # Calculate piecewise OBV trendlines for visualization
        # This mirrors the exact logic from visualize_indicators.py
        obvPiecewiseData = {}  # Dictionary to store trendline segments
        
        if 'OBV' in df.columns and 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            boundaries = _getPiecewiseBoundaries(df)
            totalSegments = len(boundaries) - 1
            
            for k in range(totalSegments):
                start, end = boundaries[k], boundaries[k+1]
                segmentLength = end - start
                
                if segmentLength >= MINIMUM_SEGMENT_LENGTH:
                    segY = df['OBV'].iloc[start:end]
                    slope = calculateTheilSenSlope(segY)
                    
                    # Calculate robust intercept: b = median(y - slope * x)
                    x_range = np.arange(len(segY))
                    intercept = np.median(segY.values - slope * x_range)
                    
                    # Calculate trendline points
                    trendY = slope * x_range + intercept
                    
                    # Determine color based on slope
                    if slope > dynamicSlopeThreshold:
                        color = '#22C55E'  # green
                    elif slope < -dynamicSlopeThreshold:
                        color = '#EF4444'  # red
                    else:
                        color = '#3B82F6'  # blue
                    
                    for idx_offset, actual_idx in enumerate(range(start, end)):
                        segmentKey = df.index[actual_idx].isoformat()
                        obvPiecewiseData[segmentKey] = {
                            'value': float(trendY[idx_offset]),
                            'color': color
                        }

        # Build data points
        dataPoints = []
        
        for idx, row in df.iterrows():
            # Use isoformat() for robust cross-platform date/time parsing
            dateKey = idx.isoformat()
            
            # Get piecewise trend data if available (using same key format)
            piecewiseInfo = obvPiecewiseData.get(dateKey, {})
            
            # Convert numpy types to Python native types
            dataPoint = IndicatorDataPoint(
                date=dateKey,
                close=float(row['Close']),
                volume=float(row['Volume']),
                macd=float(row['MACD']) if 'MACD' in row and pd.notna(row['MACD']) else None,
                macdSignal=float(row['MACD_Signal']) if 'MACD_Signal' in row and pd.notna(row['MACD_Signal']) else None,
                macdHistogram=float(row['MACD_Histogram']) if 'MACD_Histogram' in row and pd.notna(row['MACD_Histogram']) else None,
                obv=float(row['OBV']) if 'OBV' in row and pd.notna(row['OBV']) else None,
                obvTrend=str(row['OBV_Trend']) if 'OBV_Trend' in row and pd.notna(row['OBV_Trend']) else None,
                obvPiecewiseTrend=piecewiseInfo.get('value'),
                obvTrendColor=piecewiseInfo.get('color'),
                rsi=float(row['RSI']) if 'RSI' in row and pd.notna(row['RSI']) else None,
                trendSummary=str(row['Trend_Summary']) if 'Trend_Summary' in row and pd.notna(row['Trend_Summary']) else None,
                rsiSignal=str(row['RSI_Signal']) if 'RSI_Signal' in row and pd.notna(row['RSI_Signal']) else None,
                macdCrossover=str(row['MACD_Crossover']) if 'MACD_Crossover' in row and pd.notna(row['MACD_Crossover']) else None,
            )
            dataPoints.append(dataPoint)
        
        # Build trend segments
        segments = []
        for seg in trendSegments:
            # getTrendSegments already returns formatted strings and ISO dates
            segments.append(OBVTrendSegment(
                segment=seg['segment'],
                start=seg['start'],  # Already formatted as string
                end=seg['end'],      # Already formatted as string
                duration=seg['duration'],
                slope=seg['slope'],
                priceChangePct=seg['priceChangePct'],
                obvTrend=seg['obvTrend'],
                trendSummary=seg['trendSummary']
            ))
        
        return IndicatorsResponse(
            ticker=ticker,
            period=period,
            interval=interval,
            dataPoints=dataPoints,
            trendSegments=segments,
            averageDailyVolume=averageDailyVolume,
            dynamicSlopeThreshold=dynamicSlopeThreshold
        ).model_dump()
    except Exception as exc:
        return ErrorResponse(
            error="Internal server error in getIndicatorsData",
            ticker=ticker,
            details=str(exc)
        ).model_dump()


# ============================================================================
# Contract Selling Handler
# ============================================================================

def getContractSellingData(
    ticker: str, 
    strategy: str = "CSP", 
    engine: str = "BOTH", 
    expiration: Optional[str] = None
) -> Dict:
    """
    Fetch Contract Selling analysis and return structured JSON.
    """
    try:
        cash_equity = sum(LENDERS)
        analyst = ContractSellingAnalyst(cash_equity=cash_equity)
        result = analyst.scan(
            ticker.upper(), 
            expiration_input=expiration, 
            strategy_type=strategy.upper(), 
            engine_mode=engine.upper()
        )
        
        if "error" in result:
            return ErrorResponse(
                error=result["error"],
                ticker=ticker,
                details=result.get("details")
            ).model_dump()
            
        return ContractSellingResponse(**result).model_dump()
        
    except Exception as exc:
        return ErrorResponse(
            error="Internal server error in ContractSellingAnalyst",
            ticker=ticker,
            details=str(exc)
        ).model_dump()


# ============================================================================
# Portfolio Margin Simulation Handler
# ============================================================================

def getPortfolioSimulationData(
    tickers: List[str], 
    strategy: str = "CSP", 
    expiration: Optional[str] = None
) -> Dict:
    """
    Simulate shared portfolio margin allocation across multiple tickers.
    Delegates calculation and scan-execution to PortfolioMarginAllocator.simulate_multi_asset_portfolio
    """
    try:
        portfolio = simulate_multi_asset_portfolio(
            tickers=tickers,
            strategy_type=strategy,
            expiration=expiration
        )

        api_positions = [
            PortfolioPositionData(
                ticker=p.ticker,
                strike=p.strike,
                contracts=p.contracts,
                notional=p.notional,
                premium_collected=p.premium_collected,
                strategy_type=p.strategy_type,
                maint_req=p.maint_req,
                status=p.status
            )
            for p in portfolio.positions
        ]

        return PortfolioMarginResponse(
            cash=portfolio.cash,
            accumulated_premiums=portfolio.accumulated_premiums,
            total_equity=portfolio.total_equity,
            tightest_req=portfolio.tightest_req,
            margin_limit=portfolio.margin_limit,
            total_notional=portfolio.total_notional,
            cash_utilized=portfolio.cash_utilized,
            margin_utilized=portfolio.margin_utilized,
            positions=api_positions
        ).model_dump()
        
    except Exception as exc:
        return ErrorResponse(
            error="Portfolio simulation failed",
            details=str(exc)
        ).model_dump()
