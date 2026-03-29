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
        
        if indicators is None:
            indicators = ["MACD", "OBV", "RSI"]
        
        indicatorTypeMap = {
            "SMA": IndicatorType.SMA, "EMA": IndicatorType.EMA, "RSI": IndicatorType.RSI, "MACD": IndicatorType.MACD,
            "BOLLINGER_BANDS": IndicatorType.BOLLINGER_BANDS, "ATR": IndicatorType.ATR, "STOCHASTIC": IndicatorType.STOCHASTIC, "OBV": IndicatorType.OBV,
        }
        
        indicatorsToCalculate = [indicatorTypeMap[ind.upper()] for ind in indicators if ind.upper() in indicatorTypeMap]
        
        if not indicatorsToCalculate:
            return ErrorResponse(error="No valid indicators specified", ticker=ticker).model_dump()
        
        df = getIndicators(ticker, indicatorsToCalculate, period=pricePeriod, interval=priceInterval)
        if df is None or df.empty:
            return ErrorResponse(error="Could not fetch data", ticker=ticker).model_dump()
        
        df = generateTradingSignals(df)
        
        # Static sensitivity for dummy visualization
        SLOPE_SENSITIVITY_RATIO = 0.001
        dynamicSlopeThreshold = float(df['Volume'].mean() * SLOPE_SENSITIVITY_RATIO)
        trendSegments = getTrendSegments(df)
        
        dataPoints = []
        for idx, row in df.iterrows():
            dataPoint = IndicatorDataPoint(
                date=idx.isoformat(),
                close=float(row['Close']),
                volume=float(row['Volume']),
                macd=float(row['MACD']) if 'MACD' in row and pd.notna(row['MACD']) else None,
                macdSignal=float(row['MACD_Signal']) if 'MACD_Signal' in row and pd.notna(row['MACD_Signal']) else None,
                macdHistogram=float(row['MACD_Histogram']) if 'MACD_Histogram' in row and pd.notna(row['MACD_Histogram']) else None,
                obv=float(row['OBV']) if 'OBV' in row and pd.notna(row['OBV']) else None,
                obvTrend=str(row['OBV_Trend']) if 'OBV_Trend' in row and pd.notna(row['OBV_Trend']) else None,
                rsi=float(row['RSI']) if 'RSI' in row and pd.notna(row['RSI']) else None,
            )
            dataPoints.append(dataPoint)
        
        segments = [OBVTrendSegment(**seg) for seg in trendSegments]
        
        return IndicatorsResponse(
            ticker=ticker, period=period, interval=interval,
            dataPoints=dataPoints, trendSegments=segments,
            averageDailyVolume=float(df['Volume'].mean()),
            dynamicSlopeThreshold=dynamicSlopeThreshold
        ).model_dump()
    except Exception as exc:
        return ErrorResponse(error="Internal server error in getIndicatorsData", ticker=ticker, details=str(exc)).model_dump()


# ============================================================================
# Contract Selling Handler
# ============================================================================

def getContractSellingData(
    ticker: str, 
    strategy: str = "CSP", 
    engine: str = "BOTH", 
    expiration: Optional[str] = None,
    cash_equity: Optional[float] = None
) -> Dict:
    """
    Fetch Contract Selling analysis and return structured JSON.
    """
    try:
        current_equity = cash_equity if cash_equity is not None else sum(LENDERS)
        analyst = ContractSellingAnalyst(cash_equity=current_equity)
        result = analyst.scan(
            ticker.upper(), 
            expiration_input=expiration, 
            strategy_type=strategy.upper(), 
            engine_mode=engine.upper()
        )
        
        if "error" in result:
            return ErrorResponse(error=result["error"], ticker=ticker).model_dump()
            
        return ContractSellingResponse(**result).model_dump()
    except Exception as exc:
        return ErrorResponse(error="Internal server error in ContractSellingAnalyst", ticker=ticker, details=str(exc)).model_dump()


# ============================================================================
# Portfolio Simulation Handler
# ============================================================================

def getPortfolioSimulationData(
    tickers: List[str], 
    strategy: str = "CSP", 
    expiration: Optional[str] = None,
    cash_equity: Optional[float] = None
) -> Dict:
    """
    Simulate shared portfolio margin allocation across multiple tickers.
    """
    try:
        portfolio = simulate_multi_asset_portfolio(
            tickers=tickers,
            strategy_type=strategy,
            expiration=expiration,
            cash_equity=cash_equity
        )

        api_positions = [
            PortfolioPositionData(
                ticker=p.ticker,
                strike=p.strike,
                contracts=p.contracts,
                notional=p.notional,
                premium_collected=p.premium_collected,
                strategy_type=p.strategy_type,
                initial_req=p.initial_req or 0.0,
                maint_req=p.maint_req or 0.0,
                spot_at_entry=p.spot_at_entry,
                margin_call_floor=p.margin_call_floor,
                status=p.status
            )
            for p in portfolio.positions
        ]

        return PortfolioMarginResponse(
            cash=portfolio.cash,
            accumulated_premiums=portfolio.accumulated_premiums,
            total_review_equity=portfolio.total_equity,
            total_notional=portfolio.total_notional,
            total_initial_margin=portfolio.total_initial_margin,
            total_maintenance_margin=portfolio.total_maintenance_margin,
            total_assignment_exposure=portfolio.total_assignment_exposure,
            buying_power_remaining=portfolio.buying_power_remaining,
            positions=api_positions
        ).model_dump()
    except Exception as exc:
        import traceback
        return ErrorResponse(
            error="Portfolio simulation failed", 
            details=f"{str(exc)}\n{traceback.format_exc()}"
        ).model_dump()
