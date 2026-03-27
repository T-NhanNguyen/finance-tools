"""
FastAPI Backend Server
Provides HTTP endpoints for finance tools with structured JSON responses.
"""

from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import os

from .api_types import (
    GEXRequest, IndicatorsRequest, GEXResponse, IndicatorsResponse, ErrorResponse,
    ContractSellingRequest, ContractSellingResponse, PortfolioSimulationRequest, PortfolioMarginResponse, PortfolioMarginRequest
)
from .api_handlers import getGEXData, getIndicatorsData, getContractSellingData, getPortfolioSimulationData, optimizePortfolioAllocation

# Security Configuration
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

async def verify_api_secret(x_api_secret: Optional[str] = Header(None)):
    """Verify the shared secret if one is configured."""
    if API_SECRET_KEY and x_api_secret != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Secret")

# Create a dependency sequence to apply optionally
api_dependencies = [Depends(verify_api_secret)] if API_SECRET_KEY else []

# Initialize FastAPI app
app = FastAPI(
    title="Finance Tools API",
    description="RESTful API for gamma exposure and technical indicator analysis",
    version="1.0.0"
)

# Configure CORS for frontend access
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "https://stool-webapp.vercel.app,http://localhost:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
@app.get("/api/status")
async def healthCheck():
    """Health check endpoint"""
    print("Health check endpoint hit!")
    return {"status": "healthy", "service": "finance-tools-api"}


# ============================================================================
# GEX Endpoints
# ============================================================================

@app.get("/api/gex/{ticker}", response_model=GEXResponse, dependencies=api_dependencies)
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


@app.post("/api/gex", response_model=GEXResponse, dependencies=api_dependencies)
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

@app.get("/api/indicators/{ticker}", response_model=IndicatorsResponse, dependencies=api_dependencies)
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


@app.post("/api/indicators", response_model=IndicatorsResponse, dependencies=api_dependencies)
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
# Contract Strategy Endpoints
# ============================================================================

@app.get("/api/scanner/{ticker}")
@app.get("/api/contract-selling/{ticker}", response_model=ContractSellingResponse, dependencies=api_dependencies)
async def get_contract_selling(
    ticker: str,
    strategy: Optional[str] = Query("CSP", description="Strategy type (CSP or CC)"),
    expiration: Optional[str] = Query(None, description="Expiration date (YYYY-MM-DD or index)"),
    engine: Optional[str] = Query("BOTH", description="Engine filter mode (BOTH, CASH, WHEEL)")
):
    """Analyze a single asset for option selling opportunities."""
    result = getContractSellingData(ticker, strategy, engine, expiration)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/scanner")
@app.post("/api/contract-selling", response_model=ContractSellingResponse, dependencies=api_dependencies)
async def post_contract_selling(request: ContractSellingRequest):
    """Analyze a single asset for option selling opportunities (POST variant)."""
    result = getContractSellingData(
        request.ticker, 
        request.strategy or "CSP", 
        request.engine or "BOTH", 
        request.expiration
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


# ============================================================================
# Portfolio Allocation Endpoints
# ============================================================================

@app.post("/api/portfolio-simulation", response_model=PortfolioMarginResponse, dependencies=api_dependencies)
async def simulate_portfolio_margin(request: PortfolioSimulationRequest):
    """
    High-level: Simulate shared portfolio margin across multiple tickers by scanning chains.
    """
    result = getPortfolioSimulationData(
        request.tickers,
        request.strategy or "CSP",
        request.expiration
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/api/allocate")
@app.post("/api/portfolio-allocation", response_model=PortfolioMarginResponse, dependencies=api_dependencies)
async def allocate_portfolio_positions(request: PortfolioMarginRequest):
    """
    Low-level: Solve for optimal contract allocation across manually provided candidates.
    """
    result = optimizePortfolioAllocation(request.cash, [c.model_dump() for c in request.candidates])
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
        
    return result


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn # not compatible with vercel. moved here for local run
    uvicorn.run(
        "api.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )
