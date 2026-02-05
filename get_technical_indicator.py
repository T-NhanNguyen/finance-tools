"""
Technical Indicator Calculation Module

This module provides efficient calculation of technical indicators with support
for bulk processing. Uses vectorized operations for optimal performance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from get_stock_price import getHistoricalPrices, getHistoricalPricesBulk, PricePeriod, PriceInterval


class IndicatorType(Enum):
    """Available technical indicators"""
    SMA = "simple_moving_average"
    EMA = "exponential_moving_average"
    RSI = "relative_strength_index"
    MACD = "moving_average_convergence_divergence"
    BOLLINGER_BANDS = "bollinger_bands"
    ATR = "average_true_range"
    STOCHASTIC = "stochastic_oscillator"
    OBV = "on_balance_volume"


# Constants for default indicator parameters
DEFAULT_SMA_PERIOD = 20
DEFAULT_EMA_PERIOD = 12
DEFAULT_RSI_PERIOD = 14
DEFAULT_RSI_OVERBOUGHT = 70
DEFAULT_RSI_OVERSOLD = 30
DEFAULT_RSI_TOLERANCE = 6
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BOLLINGER_PERIOD = 20
DEFAULT_BOLLINGER_STD = 2
DEFAULT_ATR_PERIOD = 14
DEFAULT_STOCHASTIC_K = 14
DEFAULT_STOCHASTIC_D = 3

# Constants for Piecewise Linear Segmentation
CROSSOVER_CONFIRMATION_THRESHOLD = 0.10  # 10% difference
INTERCEPT_PROXIMITY_THRESHOLD = 0.03     # 3% difference
MINIMUM_SEGMENT_LENGTH = 14              # Minimum points for a valid trend segment
DETERMINISTIC_NEIGHBORHOOD_LOOKAHEAD = 30 # Points to check for confirmation
CROSSOVER_DETECTION_SKIP_STEP = 5        # Avoid duplicate detection in same event
INITIAL_INDEX_OFFSET = -MINIMUM_SEGMENT_LENGTH # Ensure first point is always accepted
UNDERLYING_PRICE_CHANGE_THRESHOLD = 0.02 # 2% price change threshold for trend confirmation
USE_SIGN_CHANGE_CROSSOVERS = False        # Choose sensitivity for segment cutoff criteria. (True) more sensitve


def calculateSMA(prices: pd.Series, period: int = DEFAULT_SMA_PERIOD) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Series of price data
        period: Number of periods for the moving average
        
    Returns:
        Series with SMA values
    """
    return prices.rolling(window=period).mean()


def calculateEMA(prices: pd.Series, period: int = DEFAULT_EMA_PERIOD) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Series of price data
        period: Number of periods for the moving average
        
    Returns:
        Series with EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculateRSI(prices: pd.Series, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Series of price data
        period: Number of periods for RSI calculation
        
    Returns:
        Series with RSI values (0-100)
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses using EMA
    averageGains = gains.ewm(span=period, adjust=False).mean()
    averageLosses = losses.ewm(span=period, adjust=False).mean()
    
    # Calculate RS and RSI
    relativeStrength = averageGains / averageLosses
    rsi = 100 - (100 / (1 + relativeStrength))
    
    return rsi


def calculateMACD(
    prices: pd.Series,
    fastPeriod: int = DEFAULT_MACD_FAST,
    slowPeriod: int = DEFAULT_MACD_SLOW,
    signalPeriod: int = DEFAULT_MACD_SIGNAL
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series of price data
        fastPeriod: Fast EMA period
        slowPeriod: Slow EMA period
        signalPeriod: Signal line EMA period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fastEma = calculateEMA(prices, fastPeriod)
    slowEma = calculateEMA(prices, slowPeriod)
    
    macdLine = fastEma - slowEma
    signalLine = calculateEMA(macdLine, signalPeriod)
    histogram = macdLine - signalLine
    
    return macdLine, signalLine, histogram


def calculateBollingerBands(
    prices: pd.Series,
    period: int = DEFAULT_BOLLINGER_PERIOD,
    numStdDev: float = DEFAULT_BOLLINGER_STD
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of price data
        period: Number of periods for moving average
        numStdDev: Number of standard deviations for bands
        
    Returns:
        Tuple of (Upper band, Middle band/SMA, Lower band)
    """
    middleBand = calculateSMA(prices, period)
    standardDeviation = prices.rolling(window=period).std()
    
    upperBand = middleBand + (standardDeviation * numStdDev)
    lowerBand = middleBand - (standardDeviation * numStdDev)
    
    return upperBand, middleBand, lowerBand


def calculateATR(
    highPrices: pd.Series,
    lowPrices: pd.Series,
    closePrices: pd.Series,
    period: int = DEFAULT_ATR_PERIOD
) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        highPrices: Series of high prices
        lowPrices: Series of low prices
        closePrices: Series of close prices
        period: Number of periods for ATR
        
    Returns:
        Series with ATR values
    """
    # Calculate True Range components
    highLow = highPrices - lowPrices
    highClose = abs(highPrices - closePrices.shift())
    lowClose = abs(lowPrices - closePrices.shift())
    
    # True Range is the maximum of the three
    trueRange = pd.concat([highLow, highClose, lowClose], axis=1).max(axis=1)
    
    # ATR is the EMA of True Range
    atr = trueRange.ewm(span=period, adjust=False).mean()
    
    return atr


def calculateStochastic(
    highPrices: pd.Series,
    lowPrices: pd.Series,
    closePrices: pd.Series,
    kPeriod: int = DEFAULT_STOCHASTIC_K,
    dPeriod: int = DEFAULT_STOCHASTIC_D
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        highPrices: Series of high prices
        lowPrices: Series of low prices
        closePrices: Series of close prices
        kPeriod: Period for %K line
        dPeriod: Period for %D line (SMA of %K)
        
    Returns:
        Tuple of (%K line, %D line)
    """
    # Calculate %K
    lowestLow = lowPrices.rolling(window=kPeriod).min()
    highestHigh = highPrices.rolling(window=kPeriod).max()
    
    kLine = 100 * ((closePrices - lowestLow) / (highestHigh - lowestLow))
    
    # Calculate %D (SMA of %K)
    dLine = kLine.rolling(window=dPeriod).mean()
    
    return kLine, dLine


def calculateOBV(closePrices: pd.Series, volumes: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        closePrices: Series of close prices
        volumes: Series of trading volumes
        
    Returns:
        Series with OBV values
    """
    # Calculate the price change direction (1, -1, or 0)
    priceChangeDirection = np.sign(closePrices.diff())
    
    # Multiply by volume and calculate cumulative sum
    # Note: Using first day's volume as the base to follow standard approach
    # where the first day's OBV is the first day's volume.
    obvAccumulator = (priceChangeDirection * volumes)
    # The first row has no previous close, so diff() is NaN. 
    # Fill it with the first day's volume to start the accumulation.
    obvAccumulator.iloc[0] = volumes.iloc[0]
    
    return obvAccumulator.cumsum()


def _addIndicatorsToDataFrame(
    dataFrame: pd.DataFrame,
    indicators: List[IndicatorType]
) -> pd.DataFrame:
    """
    Internal helper that adds requested technical indicators to an existing DataFrame
    containing at least 'Close', 'High', 'Low' columns.

    Args:
        dataFrame: DataFrame with historical price data (must have Close, High, Low)
        indicators: List of IndicatorType to calculate
        
    Returns:
        The same DataFrame with new indicator columns added in-place for efficiency
    """
    closePrices = dataFrame['Close']

    for indicator in indicators:
        if indicator == IndicatorType.SMA:
            dataFrame['SMA_20'] = calculateSMA(closePrices)
            dataFrame['SMA_50'] = calculateSMA(closePrices, 50)
            
        elif indicator == IndicatorType.EMA:
            dataFrame['EMA_12'] = calculateEMA(closePrices, 12)
            dataFrame['EMA_26'] = calculateEMA(closePrices, 26)
            
        elif indicator == IndicatorType.RSI:
            dataFrame['RSI'] = calculateRSI(closePrices)
            
        elif indicator == IndicatorType.MACD:
            macdLine, signalLine, histogram = calculateMACD(closePrices)
            dataFrame['MACD'] = macdLine
            dataFrame['MACD_Signal'] = signalLine
            dataFrame['MACD_Histogram'] = histogram
            
        elif indicator == IndicatorType.BOLLINGER_BANDS:
            upperBand, middleBand, lowerBand = calculateBollingerBands(closePrices)
            dataFrame['BB_Upper'] = upperBand
            dataFrame['BB_Middle'] = middleBand
            dataFrame['BB_Lower'] = lowerBand
            
        elif indicator == IndicatorType.ATR:
            dataFrame['ATR'] = calculateATR(
                dataFrame['High'],
                dataFrame['Low'],
                dataFrame['Close']
            )
            
        elif indicator == IndicatorType.STOCHASTIC:
            kLine, dLine = calculateStochastic(
                dataFrame['High'],
                dataFrame['Low'],
                dataFrame['Close']
            )
            dataFrame['Stochastic_K'] = kLine
            dataFrame['Stochastic_D'] = dLine
            
        elif indicator == IndicatorType.OBV:
            dataFrame['OBV'] = calculateOBV(
                dataFrame['Close'],
                dataFrame['Volume']
            )
            
    return dataFrame


def calculateTheilSenSlope(y: pd.Series) -> float:
    """
    Calculate the Theil-Sen slope for a series.
    Theil-Sen is a robust linear regression estimator that calculates 
    the median of all slopes between all pairs of points.
    
    Args:
        y: Series of values (e.g., OBV)
        
    Returns:
        The median slope (scalar)
    """
    n = len(y)
    if n < 2:
        return 0.0
    
    # We use the index as x values (0, 1, 2, ...)
    yValues = y.values
    
    # Calculate slopes for all pairs (i, j) where i < j
    # For performance on larger segments, we could subsample, 
    # but for ~14-100 points, O(N^2) is acceptable.
    calculatedSlopes = []
    for i in range(n):
        for j in range(i + 1, n):
            # x_j - x_i is just j - i
            slopeValue = (yValues[j] - yValues[i]) / (j - i)
            calculatedSlopes.append(slopeValue)
            
    return float(np.median(calculatedSlopes))


def _getPiecewiseBoundaries(df: pd.DataFrame) -> List[int]:
    """
    Args:
        df: DataFrame with MACD, MACD_Signal, and OBV columns
    
    Returns:
        List[int]: List of indices where piecewise segments begin
        
    Internal helper to identify segment boundary indices based on MACD crossovers.
    Supports either sign-change detection or proximity-based intercept detection.
    """
    total_points = len(df)
    confirmed_starts = []
    
    if USE_SIGN_CHANGE_CROSSOVERS:
        # --- Approach A: Sign-Change Detection (More granular) ---
        macd_diff = df['MACD'] - df['MACD_Signal']
        average_macd_mag = (df['MACD'].abs() + df['MACD_Signal'].abs()) / 2 + 1e-9
        percent_difference = macd_diff.abs() / average_macd_mag
        
        i = 0
        while i < total_points - 1:
            if (macd_diff.iloc[i] * macd_diff.iloc[i+1] <= 0) and (macd_diff.iloc[i] != macd_diff.iloc[i+1]):
                found_confirmation = False
                for j in range(i + 1, min(i + DETERMINISTIC_NEIGHBORHOOD_LOOKAHEAD, total_points)):
                    if percent_difference.iloc[j] >= CROSSOVER_CONFIRMATION_THRESHOLD:
                        found_confirmation = True
                        break
                    if (macd_diff.iloc[i+1] * macd_diff.iloc[j] < 0):
                        break
                
                if found_confirmation:
                    confirmed_starts.append(i + 1)
                    i += CROSSOVER_DETECTION_SKIP_STEP 
                else:
                    i += 1
            else:
                i += 1
    else:
        # --- Approach B: Proximity Intercept Detection (Smoother/Legacy) ---
        averageMacdMagnitude = (df['MACD'].abs() + df['MACD_Signal'].abs()) / 2 + 1e-9
        percentDifference = (df['MACD'] - df['MACD_Signal']).abs() / averageMacdMagnitude
        potentialInterceptPoints = percentDifference <= INTERCEPT_PROXIMITY_THRESHOLD
        
        i = 0
        while i < total_points:
            if potentialInterceptPoints.iloc[i]:
                foundConfirmation = False
                for j in range(i + 1, min(i + DETERMINISTIC_NEIGHBORHOOD_LOOKAHEAD, total_points)):
                    if percentDifference.iloc[j] >= CROSSOVER_CONFIRMATION_THRESHOLD:
                        foundConfirmation = True
                        break
                    if potentialInterceptPoints.iloc[j]:
                        break
                
                if foundConfirmation:
                    confirmed_starts.append(i)
                    i += CROSSOVER_DETECTION_SKIP_STEP 
                else:
                    i += 1
            else:
                i += 1

    # 2. Refine boundaries based on MINIMUM_SEGMENT_LENGTH
    # Use forward-cut logic: cut at the first confirmed crossover after MIN_LEN
    refined_starts = [0]
    last_added_index = 0
    
    if not USE_SIGN_CHANGE_CROSSOVERS:
        # Legacy clustering for Approach B
        idx = 0
        while idx < len(confirmed_starts):
            clusterIdx = idx
            while clusterIdx + 1 < len(confirmed_starts) and \
                  confirmed_starts[clusterIdx + 1] - confirmed_starts[clusterIdx] <= 3:
                clusterIdx += 1
            
            startPoint = confirmed_starts[clusterIdx]
            if startPoint - last_added_index >= MINIMUM_SEGMENT_LENGTH:
                refined_starts.append(startPoint)
                last_added_index = startPoint
            idx = clusterIdx + 1
    else:
        # Simple earliest-cut for Approach A
        for start_point in confirmed_starts:
            if start_point - last_added_index >= MINIMUM_SEGMENT_LENGTH:
                refined_starts.append(start_point)
                last_added_index = start_point
            
    return sorted(list(set(refined_starts + [total_points])))


def calculateOBVTrendPiecewise(df: pd.DataFrame) -> pd.Series:
    """
    Args:
        df: DataFrame with MACD, MACD_Signal, and OBV columns
    
    Returns:
        pd.Series with OBV trend labels ("Rising", "Falling", "Flat", "Noise/Transition")
    
    Calculate OBV trend using Piecewise Linear Segmentation based on MACD crossovers.
    Uses Theil-Sen regression for slope calculation within segments.
    """
    if not all(col in df.columns for col in ['MACD', 'MACD_Signal', 'OBV']):
        return pd.Series("N/A", index=df.index)

    boundaries = _getPiecewiseBoundaries(df)
    trendLabels = pd.Series("Noise/Transition", index=df.index)
    
    for k in range(len(boundaries) - 1):
        start, end = boundaries[k], boundaries[k+1]
        segmentLength = end - start
        
        if segmentLength >= MINIMUM_SEGMENT_LENGTH:
            segmentY = df['OBV'].iloc[start:end]
            calculatedSlope = calculateTheilSenSlope(segmentY)
            
            if calculatedSlope > 0:
                trendLabel = "Rising"
            elif calculatedSlope < 0:
                trendLabel = "Falling"
            else:
                trendLabel = "Flat"
            
            trendLabels.iloc[start:end] = trendLabel
        else:
            trendLabels.iloc[start:end] = "Noise/Transition"
            
    return trendLabels


def getTrendSegments(df: pd.DataFrame) -> List[Dict]:
    """
    Identifies major OBV trend segments and returns a token-efficient summary for each.
    
    Args:
        df: DataFrame with Close, Volume, MACD, MACD_Signal, and OBV
        
    Returns:
        List of dictionaries containing segment details (dates, duration, slope, price change)
    """
    if not all(col in df.columns for col in ['MACD', 'MACD_Signal', 'OBV', 'Close']):
        return []

    boundaries = _getPiecewiseBoundaries(df)
    segments = []
    
    for k in range(len(boundaries) - 1):
        start, end = boundaries[k], boundaries[k+1]
        segmentLength = end - start
        
        if segmentLength >= MINIMUM_SEGMENT_LENGTH:
            segmentY = df['OBV'].iloc[start:end]
            slope = calculateTheilSenSlope(segmentY)
            
            startDate = df.index[start]
            endDate = df.index[end-1]
            
            startPrice = df['Close'].iloc[start]
            endPrice = df['Close'].iloc[end-1]
            priceChange = ((endPrice - startPrice) / startPrice) * 100
            
            obvTrend = "Rising" if slope > 0 else "Falling" if slope < 0 else "Flat"
            
            # Trend Summary Logic (matches generateTradingSignals but per-segment)
            trendSummary = "NEUTRAL"
            if obvTrend == "Falling":
                if priceChange < 0:
                    trendSummary = "CONFIRMED_DOWNTREND"
                elif priceChange > 0:
                    trendSummary = "BEARISH_DIVERGENCE"
            elif obvTrend == "Rising":
                if abs(priceChange) < (UNDERLYING_PRICE_CHANGE_THRESHOLD * 100):
                    trendSummary = "ACCUMULATION"
                elif priceChange > 0:
                    trendSummary = "STRONG_ACCUMULATION"
            
            segments.append({
                "segment": k + 1,
                "start": startDate.isoformat(),
                "end": endDate.isoformat(),
                "duration": f"{(endDate - startDate).days} days",
                "slope": f"{slope:,.0f}",
                "priceChangePct": f"{priceChange:+.2f}%",
                "obvTrend": obvTrend,
                "trendSummary": trendSummary
            })
            
    return segments


def getIndicators(
    ticker: str,
    indicators: List[IndicatorType],
    period: PricePeriod = PricePeriod.THREE_MONTHS,
    interval: PriceInterval = PriceInterval.ONE_DAY,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Calculate technical indicators for a single ticker.
    
    Args:
        ticker: Stock symbol
        indicators: List of indicators to calculate
        period: Time period (if no dates provided)
        interval: Data granularity (e.g., ONE_DAY)
        startDate: Optional 'YYYY-MM-DD'
        endDate: Optional 'YYYY-MM-DD'

    Returns:
        pd.DataFrame with indicators
    """
    try:
        historicalData = getHistoricalPrices(ticker, period, interval, startDate, endDate)
        
        if historicalData is None or historicalData.empty:
            return None
            
        resultDf = historicalData.copy()
        _addIndicatorsToDataFrame(resultDf, indicators)
        
        return resultDf
        
    except Exception as error:
        print(f"Error calculating indicators for {ticker}: {error}")
        return None


def getIndicatorsBulk(
    tickers: List[str],
    indicators: List[IndicatorType],
    period: PricePeriod = PricePeriod.THREE_MONTHS,
    interval: PriceInterval = PriceInterval.ONE_DAY,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Calculate technical indicators for multiple tickers using bulk data retrieval.
    """
    indicatorResults = {}
    
    try:
        bulkHistoricalData = getHistoricalPricesBulk(tickers, period, interval, startDate, endDate)
        
        for ticker, historicalData in bulkHistoricalData.items():
            if historicalData is None or historicalData.empty:
                indicatorResults[ticker] = None
                continue
                
            resultDf = historicalData.copy()
            _addIndicatorsToDataFrame(resultDf, indicators)
            indicatorResults[ticker] = resultDf
            
    except Exception as error:
        print(f"Error in bulk indicator calculation: {error}")
        
    return indicatorResults


def generateTradingSignals(indicatorData: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        indicatorData: DataFrame with calculated indicators
        
    Returns:
        DataFrame with added signal columns

    Generate complex trading signals based on technical indicators, 
    including OBV trend analysis via Piecewise Linear Segmentation.
    """
    signals = indicatorData.copy()
    
    # 1. OBV Piecewise Trend Analysis
    if 'OBV' in signals.columns:
        signals['OBV_Trend'] = calculateOBVTrendPiecewise(signals)
    
    # 2. Trend Summary & Divergence Logic
    if all(col in signals.columns for col in ['Close', 'OBV', 'OBV_Trend']):
        signals['Trend_Summary'] = 'NEUTRAL'
        
        # Calculate local price direction (over last 5 days for noise reduction)
        priceChangeMagnitude = signals['Close'].diff(5)
        
        # a. Confirmed Downtrend: Price falling and OBV falling
        signals.loc[(priceChangeMagnitude < 0) & (signals['OBV_Trend'] == "Falling"), 'Trend_Summary'] = 'CONFIRMED_DOWNTREND'
        
        # b. Bearish Divergence: OBV falls while Price rises (Rally is a lie)
        signals.loc[(priceChangeMagnitude > 0) & (signals['OBV_Trend'] == "Falling"), 'Trend_Summary'] = 'BEARISH_DIVERGENCE'
        
        # c. Accumulation: OBV rising while price consolidating (absolute price change < 2%)
        absolutePriceChangeRatio = (priceChangeMagnitude / signals['Close'].shift(5)).abs()
        signals.loc[(absolutePriceChangeRatio < UNDERLYING_PRICE_CHANGE_THRESHOLD) & (signals['OBV_Trend'] == "Rising"), 'Trend_Summary'] = 'ACCUMULATION'
        
        # d. Negative Volume Balance: Heavy volume on drops, light on bounces
        # (Approximate: OBV dropping faster than price implies aggressive exiting)
        # Note: This is more a characteristic of the "Falling" trend during certain phases.
        
    # 3. Standard Indicators
    # RSI signals (oversold/overbought)
    if 'RSI' in signals.columns:
        signals['RSI_Signal'] = 'NEUTRAL'
        signals.loc[signals['RSI'] < (DEFAULT_RSI_OVERSOLD + DEFAULT_RSI_TOLERANCE), 'RSI_Signal'] = 'OVERSOLD'
        signals.loc[signals['RSI'] > (DEFAULT_RSI_OVERBOUGHT - DEFAULT_RSI_TOLERANCE), 'RSI_Signal'] = 'OVERBOUGHT'
    
    # MACD signals (bullish/bearish crossover)
    if 'MACD' in signals.columns and 'MACD_Signal' in signals.columns:
        signals['MACD_Crossover'] = 'NEUTRAL'
        macdDifference = signals['MACD'] - signals['MACD_Signal']
        previousMacdDifference = macdDifference.shift(1)
        
        # Bullish crossover: MACD crosses above signal
        signals.loc[(macdDifference > 0) & (previousMacdDifference <= 0), 'MACD_Crossover'] = 'BULLISH'
        # Bearish crossover: MACD crosses below signal
        signals.loc[(macdDifference < 0) & (previousMacdDifference >= 0), 'MACD_Crossover'] = 'BEARISH'
    
    # Bollinger Bands signals
    if all(col in signals.columns for col in ['Close', 'BB_Upper', 'BB_Lower']):
        signals['BB_Signal'] = 'NEUTRAL'
        signals.loc[signals['Close'] <= signals['BB_Lower'], 'BB_Signal'] = 'OVERSOLD'
        signals.loc[signals['Close'] >= signals['BB_Upper'], 'BB_Signal'] = 'OVERBOUGHT'
    
    return signals


# Example usage
if __name__ == "__main__":
    # Single ticker example
    print("=== Single Ticker Technical Indicators ===")
    indicatorsToCalculate = [
        IndicatorType.SMA,
        IndicatorType.RSI,
        IndicatorType.MACD,
        IndicatorType.BOLLINGER_BANDS,
        IndicatorType.OBV
    ]
    
    # 1. Example: Specific Date Range & Daily Interval
    print("\n=== Specific Date Range (Daily Interval) ===")
    specificIndicators = getIndicators(
        "TE", 
        indicatorsToCalculate,
        startDate="2025-04-01",
        endDate="2025-12-31",
        interval=PriceInterval.ONE_DAY  # Explicitly setting daily interval
    )
    
    if specificIndicators is not None:
        # Generate trading signals (this adds OBV_Trend and Trend_Summary)
        specificIndicators = generateTradingSignals(specificIndicators)
        print(f"Retrieved {len(specificIndicators)} daily data points for 2025")
        
        # Create a clean display copy (preserves original precision in specificIndicators)
        displayDf = specificIndicators[['Close', 'OBV', 'OBV_Trend', 'Trend_Summary']].copy()
        
        # Format index to show ONLY date (no time/timezone)
        displayDf.index = displayDf.index.date
        
        # Format Close to round DOWN (floor)
        displayDf['Close'] = np.floor(displayDf['Close'])
        
        print(displayDf)

    # 2. Example: Recent 6 Months Analysis
    print("\n=== Recent 6 Months OBV Trend Analysis ===")
    longHistoryData = getIndicators("BE", indicatorsToCalculate, period=PricePeriod.SIX_MONTHS)
    if longHistoryData is not None:
        signalsData = generateTradingSignals(longHistoryData)
        print("\nLatest signals and OBV trend:")
        summaryColumnNames = ['Close', 'OBV', 'OBV_Trend', 'Trend_Summary']
        if 'RSI_Signal' in signalsData.columns:
            summaryColumnNames.append('RSI_Signal')
        print(signalsData[summaryColumnNames].tail(10))
    
    # Bulk indicators example
    print("\n=== Bulk Technical Indicators ===")
    techStocks = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    bulkIndicators = getIndicatorsBulk(
        techStocks,
        [IndicatorType.RSI, IndicatorType.MACD, IndicatorType.OBV],
        period=PricePeriod.ONE_MONTH
    )
    
    for ticker, data in bulkIndicators.items():
        if data is not None:
            latestRsiValue = data['RSI'].iloc[-1]
            latestMacdValue = data['MACD'].iloc[-1]
            latestObvValue = data['OBV'].iloc[-1]
            print(f"{ticker} - RSI: {latestRsiValue:.2f}, MACD: {latestMacdValue:.2f}, OBV: {latestObvValue:.2f}")
