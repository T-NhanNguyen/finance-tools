import numpy as np
from scipy.stats import norm
from typing import Union, List, Dict
from calculate_risk_free_rate import getRiskFreeRate

# Constants
DAYS_IN_YEAR = 365.0

class BlackScholesCalculator:
    """
    Unified Black-Scholes calculator for Gamma and Delta Greeks.
    Handles single values, lists, or NumPy arrays through vectorization.
    """

    def _validateInputs(self, stockPrice: float, strikePrice: Union[float, List[float], np.ndarray], timeToExpirationYears: float, riskFreeRate: float, impliedVolatility: Union[float, np.ndarray]) -> np.ndarray:
        """
        Validates mathematical constraints and returns strikes as a consistent NumPy array.
        """
        ivArray = np.asarray(impliedVolatility)
        
        if stockPrice <= 0 or timeToExpirationYears <= 0 or np.any(ivArray <= 0):
            raise ValueError(f"Stock price, time, and volatility must be positive.")
        
        strikePrices = np.asarray(strikePrice)
        if np.any(strikePrices <= 0):
            raise ValueError(f"All strike prices must be positive. Received strikes: {strikePrice}")
            
        return strikePrices

    def _calculateD1(self, stockPrice: float, strikePrices: np.ndarray, timeToExpirationYears: float, riskFreeRate: float, impliedVolatility: float) -> np.ndarray:
        """
        Internal vectorized d1 calculation.
        """
        numerator = np.log(stockPrice / strikePrices) + (riskFreeRate + 0.5 * impliedVolatility ** 2) * timeToExpirationYears
        denominator = impliedVolatility * np.sqrt(timeToExpirationYears)
        return numerator / denominator

    def calculateGamma(self, stockPrice: float, strikePrice: Union[float, List[float], np.ndarray], timeToExpirationYears: float, impliedVolatility: float, daysToExpiration: int = None) -> Union[float, np.ndarray]:
        """
        Calculates Gamma for one or many strikes.
        
        Args:
            stockPrice: Current underlying price.
            strikePrice: Single strike (float) or multiple strikes (list/array).
            timeToExpirationYears: Time to expiry in years.
            impliedVolatility: Annualized IV (e.g., 0.3 for 30%).
            
        Returns:
            A single float if strikePrice was a scalar, otherwise a NumPy array.
        """
        if daysToExpiration is None:
            daysToExpiration = int(timeToExpirationYears * DAYS_IN_YEAR)
            
        riskFreeRate = getRiskFreeRate(daysToExpiration)
        strikePricesArray = self._validateInputs(stockPrice, strikePrice, timeToExpirationYears, riskFreeRate, impliedVolatility)
        
        d1Values = self._calculateD1(stockPrice, strikePricesArray, timeToExpirationYears, riskFreeRate, impliedVolatility)
        pdfValues = norm.pdf(d1Values)
        
        gammaResult = pdfValues / (stockPrice * impliedVolatility * np.sqrt(timeToExpirationYears))
        
        # Return same type as input (scalar vs array)
        return float(gammaResult) if np.isscalar(strikePrice) else gammaResult

    def calculateDelta(self, stockPrice: float, strikePrice: Union[float, List[float], np.ndarray], timeToExpirationYears: float, impliedVolatility: float, optionType: str, daysToExpiration: int = None) -> Union[float, np.ndarray]:
        """
        Calculates Delta for one or many strikes.
        """
        if daysToExpiration is None:
            daysToExpiration = int(timeToExpirationYears * DAYS_IN_YEAR)
            
        riskFreeRate = getRiskFreeRate(daysToExpiration)
        strikePricesArray = self._validateInputs(stockPrice, strikePrice, timeToExpirationYears, riskFreeRate, impliedVolatility)
        
        d1Values = self._calculateD1(stockPrice, strikePricesArray, timeToExpirationYears, riskFreeRate, impliedVolatility)
        
        if optionType.lower() == 'call':
            deltaResult = norm.cdf(d1Values)
        elif optionType.lower() == 'put':
            deltaResult = norm.cdf(d1Values) - 1.0
        else:
            raise ValueError(f"Invalid optionType: '{optionType}'. Use 'call' or 'put'.")

        return float(deltaResult) if np.isscalar(strikePrice) else deltaResult

# Singleton instance
blackScholesCalculatorInstance = BlackScholesCalculator()

# Simplified Global Facade
def calculateGamma(stockPrice: float, strikePrice: Union[float, List[float], np.ndarray], timeToExpirationYears: float, impliedVolatility: float, daysToExpiration: int = None) -> Union[float, np.ndarray]:
    return blackScholesCalculatorInstance.calculateGamma(stockPrice, strikePrice, timeToExpirationYears, impliedVolatility, daysToExpiration)

def calculateDelta(stockPrice: float, strikePrice: Union[float, List[float], np.ndarray], timeToExpirationYears: float, impliedVolatility: float, optionType: str, daysToExpiration: int = None) -> Union[float, np.ndarray]:
    return blackScholesCalculatorInstance.calculateDelta(stockPrice, strikePrice, timeToExpirationYears, impliedVolatility, optionType, daysToExpiration)

if __name__ == "__main__":
    # Demo of clean API
    S, T, IV = 150.0, 45/365, 0.30
    
    print(f"--- Unified Black-Scholes Demo (S=${S}) ---")
    
    # 1. Single strike calculation
    g_single = calculateGamma(S, 155.0, T, IV)
    print(f"Single Strike (155) Gamma: {g_single:.6f} ({type(g_single)})")
    
    # 2. Bulk strike calculation (List)
    strikes = [145.0, 150.0, 155.0]
    g_bulk = calculateGamma(S, strikes, T, IV)
    print(f"Bulk Strikes {strikes} Gamma: {np.round(g_bulk, 6)} ({type(g_bulk)})")
    
    # 3. Delta Bulk (Call)
    d_bulk = calculateDelta(S, strikes, T, IV, 'call')
    print(f"Bulk Strikes {strikes} Delta: {np.round(d_bulk, 6)}")