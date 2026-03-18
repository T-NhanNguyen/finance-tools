import yfinance as yf
from datetime import datetime, timedelta
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SHORT_TERM_RATE = 0.05
DEFAULT_LONG_TERM_RATE = 0.04
CACHE_REFRESH_INTERVAL_HOURS = 4
DAYS_IN_YEAR = 365.0
MIN_RISK_FREE_RATE = 0.001
MAX_RISK_FREE_RATE = 0.15

# Ticker Symbols
TREASURY_BILL_3M_TICKER = "^IRX"
TREASURY_NOTE_10Y_TICKER = "^TNX"

class RiskFreeRateManager:
    """
    Manages fetching and caching of risk-free rates from market data.
    Uses a singleton pattern to ensure consistent data across the application.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RiskFreeRateManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self.shortTermRate = None
        self.longTermRate = None
        self.lastUpdateTimestamp = None
        self.cacheDuration = timedelta(hours=CACHE_REFRESH_INTERVAL_HOURS)
    
    def _needsRefresh(self):
        if self.lastUpdateTimestamp is None:
            return True
        return datetime.now() - self.lastUpdateTimestamp > self.cacheDuration
    
    def _fetchMarketRates(self):
        """
        Fetches current interest rates for 3-month T-bills and 10-year Treasury notes.
        Includes fallback mechanism to default rates if the API call fails.
        """
        try:
            # Fetch short-term rate (3-month T-bill)
            treasuryBill3MData = yf.Ticker(TREASURY_BILL_3M_TICKER).history(period="1d")
            if not treasuryBill3MData.empty:
                self.shortTermRate = treasuryBill3MData['Close'].iloc[-1] / 100
            
            # Fetch long-term rate (10-year Treasury)
            treasuryNote10YData = yf.Ticker(TREASURY_NOTE_10Y_TICKER).history(period="1d")
            if not treasuryNote10YData.empty:
                self.longTermRate = treasuryNote10YData['Close'].iloc[-1] / 100
            
            self.lastUpdateTimestamp = datetime.now()
            logger.info(f"Risk-free rates updated: Short={self.shortTermRate:.4f}, Long={self.longTermRate:.4f}")
            
        except Exception as error:
            logger.warning(f"Failed to fetch market risk-free rates. Error: {error}. Using defaults.")
            # Set reasonable defaults if API fails or returns no data
            if self.shortTermRate is None:
                self.shortTermRate = DEFAULT_SHORT_TERM_RATE
            if self.longTermRate is None:
                self.longTermRate = DEFAULT_LONG_TERM_RATE
    
    def getRateByExpiration(self, daysToExpiration: int) -> float:
        """
        Retrieves the appropriate risk-free rate based on the time horizon.
        
        Args:
            daysToExpiration: Number of days until the option expires.
            
        Returns:
            The annualized risk-free interest rate.
        """
        with self._lock:
            if self._needsRefresh():
                self._fetchMarketRates()
            
            timeHorizonYears = daysToExpiration / DAYS_IN_YEAR
            
            # Use short-term rate for horizons <= 1 year, else use long-term rate
            if timeHorizonYears <= 1:
                rate = self.shortTermRate
            else:
                rate = self.longTermRate
            
            # Bound the rate within a reasonable range to prevent numerical instability
            return max(MIN_RISK_FREE_RATE, min(rate, MAX_RISK_FREE_RATE))

# Singleton instance for easy access across the application
riskFreeRateManagerInstance = RiskFreeRateManager()

def getRiskFreeRate(daysToExpiration: int) -> float:
    """
    Convenience function to access the current risk-free rate.
    
    Args:
        daysToExpiration: Number of days until the option expires.
        
    Returns:
        The annualized risk-free interest rate (e.g., 0.045 for 4.5%).
    """
    return riskFreeRateManagerInstance.getRateByExpiration(daysToExpiration)