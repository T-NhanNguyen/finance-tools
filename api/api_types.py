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
# Contract Selling Analyst Response Models
# ============================================================================

class StrikeAnalysisData(BaseModel):
    """Deep metrics for a single option strike analysis"""
    strike: float = Field(description="Option strike price")
    strategy_tag: str = Field(description="Strategy bucket (Cash Engine / Wheel Engine)")
    premium: float = Field(description="Total premium price")
    premium_extrinsic: float = Field(description="Extrinsic value of the premium")
    contracts: int = Field(description="Number of affordable contracts")
    trade_roi_pct: float = Field(description="True Trade ROI percentage")
    trade_roi_net_pct: float = Field(description="Net Trade ROI percentage after carry costs")
    trade_roi_post_tax_pct: float = Field(description="Post-tax Trade ROI percentage")
    eoy_projection_pct: float = Field(description="Compounded annual ROI projection")
    margin_call_floor: float = Field(description="Calculated margin call price floor")
    safety_margin_pct: float = Field(description="Safety cushion distance from spot price")
    structural_score: float = Field(description="Structural support score (abs(GEX * OI))")
    efficiency_score: float = Field(description="Premium yield / normalized safety buffer")
    capital_efficiency_ratio: float = Field(description="Trade ROI vs risk divisor ratio")
    weeks_to_zero: float = Field(description="Estimated weeks to zero cost basis via repairs")
    eff_cost_basis: float = Field(description="Effective cost basis if assigned")
    predicted_p_call: float = Field(description="Predicted call/put premium proxy for repairs")
    capital_deployed: float = Field(description="Cash required for position size")

class PillarScoredPoint(BaseModel):
    """Ranked pillar prospects formatted for display summaries"""
    Rank: int = Field(description="Rank position")
    Strike: float = Field(description="Strike price of prospect")
    Strategy_Tag: str = Field(description="Classification profile (Cash/Wheel)")
    Pillar_Score: float = Field(description="Blended metric efficiency score")
    Pillar_Density: float = Field(description="GEX/OI structural density score")
    Floor_P_call: float = Field(description="Calculated price support floor benchmark")
    Safety_Buffer: str = Field(description="Formatted safety cushion string")
    Trade_ROI: str = Field(description="Formatted true trade yield string")
    Net_ROI: str = Field(description="Formatted net yield string after carry costs")
    Post_Tax_ROI: str = Field(description="Formatted post tax yield")
    WTZ_Weeks: float = Field(description="Estimated weeks to zero cost basis")
    Cap_Efficiency: float = Field(description="Capital efficiency ratio")
    Extrinsic_Premium: float = Field(description="Extrinsic premium value")
    Total_Premium: float = Field(description="Total premium Collected")
    Eff_Cost_Basis: float = Field(description="Effective cost basis Mapping break-even value")
    Contracts: int = Field(description="Number of contracts mappings")
    Capital_Deployed: float = Field(description="Amount mapping scalar deploying mapping mapped amount")
    Premium_Raw: float = Field(description="Raw contract premium price")
    
    # Helpful overlays serving view layer mapping logic
    Break_Even: Optional[float] = Field(None, description="Optional break-even calculation (CC only)")
    Capital_Deployed_Shares: Optional[float] = Field(None, description="Optional Collateral size value (CC only)")

class ActionablePillars(BaseModel):
    """Pillar ranking breakdowns supporting view dashboards"""
    Top_Wheel_Engine: List[PillarScoredPoint] = Field(default_factory=list, description="Top ranked targets scored targeting wheel execution")
    Top_Cash_Engine: List[PillarScoredPoint] = Field(default_factory=list, description="Top ranked targets scored targeting cash yield")

class ContractSellingResponse(BaseModel):
    """The response dataset covering breakdowns of scanning opportunities"""
    ticker: str = Field(description="Asset ticker symbol")
    spot_price: float = Field(description="Underlying price")
    strategy_type: str = Field(description="Strategy policy tag (CSP/CC)")
    atm_premium_benchmark: float = Field(description="Anchor benchmark yield payout scalar benchmark allocation rate")
    expiration: Optional[str] = Field(None, description="Expiration date scalar")
    pillars: ActionablePillars = Field(description=" actionables profiles setups configs responsive sets layouts")
    all_strikes: List[StrikeAnalysisData] = Field(default_factory=list, description="breaks Analytics dataset metrics mapped response")


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


class ContractSellingRequest(BaseModel):
    """Request parameters for Contract Selling analysis"""
    ticker: str = Field(description="Stock ticker symbol")
    strategy: Optional[Literal["CSP", "CC"]] = Field("CSP", description="Strategy type: CSP or CC")
    engine: Optional[Literal["BOTH", "CASH", "WHEEL"]] = Field("BOTH", description="Engine filter mode (BOTH, CASH, WHEEL)")
    expiration: Optional[str] = Field(None, description="Expiration date (YYYY-MM-DD or index)")
