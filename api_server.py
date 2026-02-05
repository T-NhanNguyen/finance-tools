"""
FastAPI Backend Server
Provides HTTP endpoints for finance tools with structured JSON responses.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import uvicorn

from api_types import GEXRequest, IndicatorsRequest, GEXResponse, IndicatorsResponse, ErrorResponse
from api_handlers import getGEXData, getIndicatorsData

# Initialize FastAPI app
app = FastAPI(
    title="Finance Tools API",
    description="RESTful API for gamma exposure and technical indicator analysis",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
async def healthCheck():
    """Health check endpoint"""
    return {"status": "healthy", "service": "finance-tools-api"}


# ============================================================================
# GEX Endpoints
# ============================================================================

@app.get("/api/gex/{ticker}", response_model=GEXResponse)
async def getGEX(
    ticker: str,
    expiration: Optional[str] = Query(None, description="Expiration date (YYYY-MM-DD, partial, or index)")
):
    """
    Get Gamma Exposure (GEX) analysis for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., SPY, QQQ, AAPL)
        expiration: Optional expiration date. Defaults to nearest expiration.
        
    Returns:
        GEXResponse with strike data and metadata
    """
    result = getGEXData(ticker, expiration)
    
    # Check if error response
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    
    return result


@app.post("/api/gex", response_model=GEXResponse)
async def postGEX(request: GEXRequest):
    """
    Get Gamma Exposure (GEX) analysis via POST request.
    
    Args:
        request: GEXRequest with ticker and optional expiration
        
    Returns:
        GEXResponse with strike data and metadata
    """
    result = getGEXData(request.ticker, request.expiration)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    
    return result


# ============================================================================
# Technical Indicators Endpoints
# ============================================================================

@app.get("/api/indicators/{ticker}", response_model=IndicatorsResponse)
async def getIndicators(
    ticker: str,
    period: str = Query("6mo", description="Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)"),
    interval: str = Query("1d", description="Data interval (1d, 1h, 5m)"),
    indicators: Optional[str] = Query(None, description="Comma-separated list of indicators (MACD,OBV,RSI)")
):
    """
    Get technical indicators analysis for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for analysis
        interval: Data granularity
        indicators: Comma-separated indicator names
        
    Returns:
        IndicatorsResponse with time series data and trend segments
    """
    # Parse indicators from comma-separated string
    indicatorList = None
    if indicators:
        indicatorList = [ind.strip() for ind in indicators.split(",")]
    
    result = getIndicatorsData(ticker, period, interval, indicatorList)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    
    return result


@app.post("/api/indicators", response_model=IndicatorsResponse)
async def postIndicators(request: IndicatorsRequest):
    """
    Get technical indicators analysis via POST request.
    
    Args:
        request: IndicatorsRequest with ticker and parameters
        
    Returns:
        IndicatorsResponse with time series data and trend segments
    """
    result = getIndicatorsData(
        request.ticker,
        request.period or "6mo",
        request.interval or "1d",
        request.indicators
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    
    return result


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )
