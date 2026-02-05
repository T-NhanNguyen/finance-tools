"""
API Handler Functions
Converts visualization scripts to return structured JSON data for HTTP responses.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

from get_options_data import getOptionChain, getOptionExpirations
from get_stock_price import getCurrentPrice, PricePeriod, PriceInterval
from calculate_gamma_delta import calculateGamma
from get_technical_indicator import (
    getIndicators, IndicatorType, generateTradingSignals,
    getTrendSegments, _getPiecewiseBoundaries, calculateTheilSenSlope,
    MINIMUM_SEGMENT_LENGTH
)
from visualize_indicators import SLOPE_SENSITIVITY_RATIO
from visualize_gex import parse_flexible_date, OPTION_CHAIN_LENGTH

from api_types import (
    GEXResponse, GEXStrikeData, IndicatorsResponse, 
    IndicatorDataPoint, OBVTrendSegment, ErrorResponse
)


# ============================================================================
# GEX Handler
# ============================================================================

def getGEXData(ticker: str, expiration_input: Optional[str] = None) -> Dict:
    """
    Fetch GEX (Gamma Exposure) data and return structured JSON.
    
    Args:
        ticker: Stock ticker symbol
        expiration_input: Optional expiration date (YYYY-MM-DD, partial, or index)
        
    Returns:
        Dictionary matching GEXResponse schema or ErrorResponse schema
    """
    try:
        ticker = ticker.upper()
        
        # Get all available expirations first
        availableExpirations = getOptionExpirations(ticker)
        if not availableExpirations:
            return ErrorResponse(
                error="No option expirations found",
                ticker=ticker,
                details="This ticker may not have options available"
            ).model_dump()
        
        # Parse expiration date
        expiration = parse_flexible_date(ticker, expiration_input)
        if not expiration:
            return ErrorResponse(
                error="Could not parse expiration date",
                ticker=ticker,
                details=f"Input: {expiration_input}"
            ).model_dump()
        
        # Fetch spot price
        spotPrice = getCurrentPrice(ticker)
        if not spotPrice:
            return ErrorResponse(
                error="Could not fetch current price",
                ticker=ticker
            ).model_dump()
        
        # Fetch option chain
        chain = getOptionChain(ticker, expiration=expiration)
        calls = chain.get('calls')
        puts = chain.get('puts')
        
        if calls is None or puts is None:
            return ErrorResponse(
                error="Could not fetch option chain",
                ticker=ticker,
                details=f"Expiration: {expiration}"
            ).model_dump()
        
        # Calculate time to expiration
        today = datetime.now()
        expiryDate = datetime.strptime(expiration, "%Y-%m-%d")
        dteYears = max(1e-6, (expiryDate - today).total_seconds() / (365 * 24 * 3600))
        daysToExpiration = (expiryDate - today).days
        
        # Calculate gammas for calls
        callStrikes = calls['strike'].values
        callIVs = calls['impliedVolatility'].values
        callGammas = calculateGamma(spotPrice, callStrikes, dteYears, callIVs)
        calls['gamma'] = callGammas
        calls['gex'] = calls['gamma'] * calls['openInterest'] * 100 * spotPrice
        
        # Calculate gammas for puts (negative from MM perspective)
        putStrikes = puts['strike'].values
        putIVs = puts['impliedVolatility'].values
        putGammas = calculateGamma(spotPrice, putStrikes, dteYears, putIVs)
        puts['gamma'] = putGammas
        puts['gex'] = -puts['gamma'] * puts['openInterest'] * 100 * spotPrice
        
        # Aggregate by strike
        callsAgg = calls[['strike', 'gex', 'openInterest', 'bid', 'ask', 'impliedVolatility']].groupby('strike').agg({
            'gex': 'sum',
            'openInterest': 'sum',
            'bid': 'first',
            'ask': 'first',
            'impliedVolatility': 'first'
        }).reset_index()
        
        putsAgg = puts[['strike', 'gex', 'openInterest', 'bid', 'ask', 'impliedVolatility']].groupby('strike').agg({
            'gex': 'sum',
            'openInterest': 'sum',
            'bid': 'first',
            'ask': 'first',
            'impliedVolatility': 'first'
        }).reset_index()

        # Merge call and put data on strike
        combinedAgg = pd.merge(
            callsAgg,
            putsAgg,
            on='strike',
            how='outer',
            suffixes=('_call', '_put')
        ).fillna(0.0)
        
        # Calculate combined aggregates
        combinedAgg['totalGEX'] = combinedAgg['gex_call'] + combinedAgg['gex_put']
        combinedAgg['totalOI'] = combinedAgg['openInterest_call'] + combinedAgg['openInterest_put']
        
        # Filter for strikes around ATM
        combinedAgg = combinedAgg.sort_values('strike').reset_index(drop=True)
        atmIdx = (combinedAgg['strike'] - spotPrice).abs().idxmin()
        
        startIdx = max(0, atmIdx - OPTION_CHAIN_LENGTH)
        endIdx = min(len(combinedAgg), atmIdx + OPTION_CHAIN_LENGTH + 1)
        plotDf = combinedAgg.iloc[startIdx:endIdx].copy()
        
        # Find max values for normalization
        maxGEXAbsolute = float(plotDf['totalGEX'].abs().max() or 1)
        maxOpenInterest = float(plotDf['totalOI'].max() or 1)
        
        # Find ATM strike
        atmStrike = plotDf.iloc[(plotDf['strike'] - spotPrice).abs().argsort()[:1]]['strike'].values[0]
        
        # Build strike data
        strikes = []
        for _, row in plotDf.iterrows():
            strike = float(row['strike'])
            gexM = float(row['totalGEX'] / 1e6)
            oiK = float(row['totalOI'] / 1e3)
            
            # Normalized values for frontend charting
            normGEX = float(row['totalGEX'] / maxGEXAbsolute)
            normOI = float(row['totalOI'] / maxOpenInterest)
            
            strikes.append(GEXStrikeData(
                strike=strike,
                gexMillions=gexM,
                openInterestThousands=oiK,
                isATM=(strike == atmStrike),
                normalizedGEX=normGEX,
                normalizedOI=normOI,
                callBid=float(row['bid_call']),
                callAsk=float(row['ask_call']),
                callIV=float(row['impliedVolatility_call']),
                callOI=float(row['openInterest_call'] / 1e3),
                putBid=float(row['bid_put']),
                putAsk=float(row['ask_put']),
                putIV=float(row['impliedVolatility_put']),
                putOI=float(row['openInterest_put'] / 1e3)
            ))
        
        return GEXResponse(
            ticker=ticker,
            expiration=expiration,
            spotPrice=float(spotPrice),
            daysToExpiration=daysToExpiration,
            strikes=strikes,
            maxGEXAbsolute=maxGEXAbsolute,
            maxOpenInterest=maxOpenInterest,
            availableExpirations=availableExpirations
        ).model_dump()
        
    except Exception as exc:
        return ErrorResponse(
            error="Internal server error",
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
            error="Internal server error",
            ticker=ticker,
            details=str(exc)
        ).model_dump()
