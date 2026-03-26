"""
API Handler Functions
Converts core data and strategy modules into structured JSON responses.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from core.data import (
    PricePeriod, PriceInterval,
    getIndicators, IndicatorType, generateTradingSignals, getTrendSegments, _getPiecewiseBoundaries, calculateTheilSenSlope, MINIMUM_SEGMENT_LENGTH
)
from core.data.get_gex_data import fetch_gex_structured
from visualizers.visualize_indicators import SLOPE_SENSITIVITY_RATIO

from core.strategies.contract_selling_analyst import ContractSellingAnalyst
from core.strategies.portfolio_margin_allocator import PortfolioMarginAllocator, Position, optimize_allocation

from .api_types import (
    GEXResponse, GEXStrikeData, IndicatorsResponse,
    IndicatorDataPoint, OBVTrendSegment, ErrorResponse,
    ContractSellingResponse, PortfolioMarginResponse, PortfolioPositionData,
    ActionablePillars, PillarScoredPoint, StrikeAnalysisData
)


# ============================================================================
# GEX Handler
# ============================================================================

def getGEXData(ticker: str, expiration_input: Optional[str] = None) -> Dict:
    """Fetches GEX data and wraps the result in the API response envelope."""
    result = fetch_gex_structured(ticker, expiration_input)
    if "error" in result:
        return ErrorResponse(
            error=result["error"],
            ticker=result.get("ticker", ticker),
            details=result.get("details")
        ).model_dump()
    strikes = [GEXStrikeData(**s) for s in result["strikes"]]
    return GEXResponse(
        ticker=result["ticker"],
        expiration=result["expiration"],
        spotPrice=result["spotPrice"],
        daysToExpiration=result["daysToExpiration"],
        strikes=strikes,
        maxGEXAbsolute=result["maxGEXAbsolute"],
        maxOpenInterest=result["maxOpenInterest"],
        availableExpirations=result["availableExpirations"]
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
            error="Internal server error",
            ticker=ticker,
            details=str(exc)
        ).model_dump()


# ============================================================================
# Contract Selling Handler
# ============================================================================

def getContractSellingData(
    ticker: str, 
    strategy_type: str = "CSP", 
    expiration: Optional[str] = None, 
    engine_mode: str = "BOTH"
) -> Dict:
    """Analyze option selling opportunities for a single asset."""
    try:
        from core.strategies.strategy_config import LENDERS
        cash_equity = sum(LENDERS)
        analyst = ContractSellingAnalyst(cash_equity=cash_equity)
        
        raw_res = analyst.scan(
            ticker.upper(), 
            expiration_input=expiration, 
            strategy_type=strategy_type, 
            engine_mode=engine_mode
        )
        
        if "error" in raw_res:
            return raw_res
            
        return ContractSellingResponse(**raw_res).model_dump()
        
    except Exception as e:
        return ErrorResponse(error="Strategy Error", ticker=ticker, details=str(e)).model_dump()


# ============================================================================
# Portfolio Allocation Handler
# ============================================================================

def optimizePortfolioAllocation(cash: float, candidates: List[Dict]) -> Dict:
    """Solve for optimal multi-asset allocation across a shared collateral pool."""
    try:
        # 1. Setup allocator
        allocator = PortfolioMarginAllocator(cash=cash)
        
        # 2. Convert raw candidate dicts back into Position objects for core logic
        candidate_objs = []
        for c in candidates:
            candidate_objs.append(Position(
                ticker=c['ticker'],
                strike=c['strike'],
                contracts=c['contracts'],
                premium_collected=c['premium_collected'],
                strategy_type=c.get('strategy_type', 'CSP'),
                initial_req=c.get('initial_req'),
                maint_req=c.get('maint_req'),
                spot_at_entry=c.get('spot_at_entry', 0)
            ))
            
        # 3. Solve 0-1 Knapsack
        optimal_allocs = optimize_allocation(allocator, candidate_objs)
        
        # 4. Build return payload
        out_positions = []
        acc_prem = 0.0
        for orig, opt in zip(candidate_objs, optimal_allocs):
            if opt.contracts > 0:
                status = "FULL DEPLOYMENT" if opt.contracts == orig.contracts else f"SCALED DOWN from {orig.contracts}"
                p_data = PortfolioPositionData(
                    ticker=opt.ticker,
                    strike=opt.strike,
                    contracts=opt.contracts,
                    notional=opt.notional,
                    premium_collected=opt.premium_collected,
                    strategy_type=opt.strategy_type,
                    initial_req=opt.initial_req,
                    maint_req=opt.maint_req,
                    spot_at_entry=opt.spot_at_entry,
                    margin_call_floor=opt.margin_call_floor,
                    status=status
                )
                out_positions.append(p_data)
                acc_prem += opt.contracts * opt.premium_collected * 100
                allocator.positions.append(opt)
        
        allocator.accumulated_premiums = acc_prem
        
        return PortfolioMarginResponse(
            cash=allocator.cash,
            accumulated_premiums=allocator.accumulated_premiums,
            total_review_equity=allocator.total_equity,
            total_notional=allocator.total_notional,
            total_initial_margin=allocator.total_initial_margin,
            total_maintenance_margin=allocator.total_maintenance_margin,
            total_assignment_exposure=allocator.total_assignment_exposure,
            buying_power_remaining=allocator.buying_power_remaining,
            positions=out_positions
        ).model_dump()
        
    except Exception as e:
        return ErrorResponse(error="Allocation Error", details=str(e)).model_dump()
