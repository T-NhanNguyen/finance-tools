"""
Type definitions for API responses.
Provides structured data models for finance tool outputs.
"""

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from datetime import date


# ============================================================================
# GEX (Gamma Exposure) Response Models
# ============================================================================

class GEXStrikeData(BaseModel):
    """Single strike price data point for GEX visualization"""
    strike: float = Field(description="Option strike price")
    gexMillions: float = Field(description="Gamma exposure in millions of dollars")
    openInterestThousands: float = Field(description="Open interest in thousands of contracts")
    isATM: bool = Field(description="Whether this strike is at-the-money")
    normalizedGEX: float = Field(description="Normalized GEX for chart scaling (-1 to 1)")
    normalizedOI: float = Field(description="Normalized OI for chart scaling (0 to 1)")
    
    # Per-side data for Option Chain visualization
    callBid: float = Field(0.0, description="Highest call bid")
    callAsk: float = Field(0.0, description="Lowest call ask")
    callIV: float = Field(0.0, description="Call implied volatility")
    callOI: float = Field(0.0, description="Call open interest in thousands")
    putBid: float = Field(0.0, description="Highest put bid")
    putAsk: float = Field(0.0, description="Lowest put ask")
    putIV: float = Field(0.0, description="Put implied volatility")
    putOI: float = Field(0.0, description="Put open interest in thousands")


class GEXResponse(BaseModel):
    """Complete GEX analysis response"""
    ticker: str = Field(description="Stock ticker symbol")
    expiration: str = Field(description="Option expiration date (YYYY-MM-DD)")
    spotPrice: float = Field(description="Current stock price")
    daysToExpiration: int = Field(description="Days until expiration")
    strikes: List[GEXStrikeData] = Field(description="Strike data sorted by strike price")
    maxGEXAbsolute: float = Field(description="Maximum absolute GEX value for scaling")
    maxOpenInterest: float = Field(description="Maximum open interest for scaling")
    availableExpirations: List[str] = Field(description="All available expiration dates")


# ============================================================================
# Technical Indicators Response Models
# ============================================================================

class IndicatorDataPoint(BaseModel):
    """Single time series data point with indicators"""
    date: str = Field(description="Date in ISO 8601 format")
    close: float = Field(description="Closing price")
    volume: float = Field(description="Trading volume")
    
    # MACD indicators
    macd: Optional[float] = Field(None, description="MACD line value")
    macdSignal: Optional[float] = Field(None, description="MACD signal line value")
    macdHistogram: Optional[float] = Field(None, description="MACD histogram value")
    
    # OBV indicators
    obv: Optional[float] = Field(None, description="On-Balance Volume")
    obvTrend: Optional[Literal["Rising", "Falling", "Flat", "Noise/Transition"]] = Field(
        None, description="OBV trend classification"
    )
    obvPiecewiseTrend: Optional[float] = Field(None, description="Piecewise trend line value")
    obvTrendColor: Optional[str] = Field(None, description="Color code for the trend segment")
    
    # RSI
    rsi: Optional[float] = Field(None, description="Relative Strength Index (0-100)")
    
    # Trading signals
    trendSummary: Optional[Literal[
        "NEUTRAL", "CONFIRMED_DOWNTREND", "BEARISH_DIVERGENCE", 
        "ACCUMULATION", "STRONG_ACCUMULATION"
    ]] = Field(None, description="Overall trend summary")
    rsiSignal: Optional[Literal["NEUTRAL", "OVERSOLD", "OVERBOUGHT"]] = Field(
        None, description="RSI signal"
    )
    macdCrossover: Optional[Literal["NEUTRAL", "BULLISH", "BEARISH"]] = Field(
        None, description="MACD crossover signal"
    )


class OBVTrendSegment(BaseModel):
    """OBV trend segment summary"""
    segment: int = Field(description="Segment number")
    start: str = Field(description="Segment start date (ISO 8601)")
    end: str = Field(description="Segment end date (ISO 8601)")
    duration: str = Field(description="Duration in days")
    slope: str = Field(description="Theil-Sen slope value (formatted)")
    priceChangePct: str = Field(description="Price change percentage (formatted)")
    obvTrend: Literal["Rising", "Falling", "Flat"] = Field(description="OBV trend direction")
    trendSummary: Literal[
        "NEUTRAL", "CONFIRMED_DOWNTREND", "BEARISH_DIVERGENCE", 
        "ACCUMULATION", "STRONG_ACCUMULATION"
    ] = Field(description="Trend interpretation")


class IndicatorsResponse(BaseModel):
    """Complete technical indicators response"""
    ticker: str = Field(description="Stock ticker symbol")
    period: str = Field(description="Time period analyzed")
    interval: str = Field(description="Data interval (e.g., '1d')")
    dataPoints: List[IndicatorDataPoint] = Field(description="Time series data with indicators")
    trendSegments: List[OBVTrendSegment] = Field(description="Major OBV trend segments")
    averageDailyVolume: float = Field(description="Average daily volume")
    dynamicSlopeThreshold: float = Field(description="Dynamic slope threshold for trend detection")


# ============================================================================
# Error Response Model
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response structure"""
    error: str = Field(description="Error message")
    ticker: Optional[str] = Field(None, description="Ticker that caused the error")
    details: Optional[str] = Field(None, description="Additional error details")


# ============================================================================
# Request Models
# ============================================================================

class GEXRequest(BaseModel):
    """Request parameters for GEX analysis"""
    ticker: str = Field(description="Stock ticker symbol")
    expiration: Optional[str] = Field(
        None, 
        description="Expiration date (YYYY-MM-DD, partial string, or index). Defaults to nearest."
    )


class IndicatorsRequest(BaseModel):
    """Request parameters for technical indicators"""
    ticker: str = Field(description="Stock ticker symbol")
    period: Optional[str] = Field("6mo", description="Time period (1mo, 3mo, 6mo, 1y, etc.)")
    interval: Optional[str] = Field("1d", description="Data interval (1d, 1h, etc.)")
    indicators: Optional[List[str]] = Field(
        ["MACD", "OBV", "RSI"], 
        description="List of indicators to calculate"
    )
