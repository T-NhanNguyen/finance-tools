import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_technical_indicator import (
    getIndicators, IndicatorType, generateTradingSignals, 
    calculateTheilSenSlope, MINIMUM_SEGMENT_LENGTH, INITIAL_INDEX_OFFSET,
    _getPiecewiseBoundaries
)
from get_stock_price import PricePeriod, PriceInterval, getHistoricalPrices

# sensitivity ratio: what % of ADV must a slope exceed to be 'Rising' or 'Falling'?
SLOPE_SENSITIVITY_RATIO = 0.03

def plotIndicators(ticker: str, period: PricePeriod = PricePeriod.SIX_MONTHS):
    """
    Fetches data and creates a two-panel visualization for MACD and OBV with Trendlines.
    """
    print(f"Fetching data for {ticker}...")
    indicatorsToCalculate = [
        IndicatorType.MACD,
        IndicatorType.OBV,
        IndicatorType.RSI,
    ]
    priceAction = getHistoricalPrices(ticker, period=period, interval=PriceInterval.ONE_DAY)
    
    # Get the data with indicators
    df = getIndicators(ticker, indicatorsToCalculate, period=period)
    if df is None:
        print("Failed to retrieve data.")
        return

    # Process signals to get trend labels
    df = generateTradingSignals(df)
    
    # Calculate Dynamic Slope Threshold based on Average Daily Volume
    averageDailyVolume = df['Volume'].mean()
    dynamicSlopeThreshold = averageDailyVolume * SLOPE_SENSITIVITY_RATIO
    print(f"Average Daily Volume: {averageDailyVolume:,.0f}")
    print(f"Dynamic Slope Threshold ({SLOPE_SENSITIVITY_RATIO*100}% of ADV): {dynamicSlopeThreshold:,.0f}")
    
    # Setup the figure
    fig, (ax3, ax2, ax1) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # --- Plot 1: MACD ---
    ax1.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1.5)
    ax1.plot(df.index, df['MACD_Signal'], label='Signal', color='red', linestyle='--', linewidth=1)
    ax1.bar(df.index, df['MACD_Histogram'], label='Histogram', color='gray', alpha=0.3)
    ax1.set_title(f"{ticker} - MACD Analysis", fontsize=14, loc='left')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: OBV with Piecewise Trendline ---
    ax2.scatter(df.index, df['OBV'], label='Raw OBV', color='black', s=2, alpha=0.4)
    
    boundaries = _getPiecewiseBoundaries(df)
    totalSegments = len(boundaries) - 1
    
    for k in range(totalSegments):
        start, end = boundaries[k], boundaries[k+1]
        
        if (end - start) >= MINIMUM_SEGMENT_LENGTH:
            segY = df['OBV'].iloc[start:end]
            slope = calculateTheilSenSlope(segY)
            
            # To plot, we need a b value. 
            # Standard robust approach: b = median(y - slope * x)
            x_range = np.arange(len(segY))
            intercept = np.median(segY.values - slope * x_range)
            
            # Calculate trendline points
            trend_y = slope * x_range + intercept
            
            # Color coding based on slope with DYNAMIC tolerance
            if slope > dynamicSlopeThreshold:
                color = 'green'
            elif slope < -dynamicSlopeThreshold:
                color = 'red'
            else:
                color = 'blue'
            
            # Label exclusively the last 2 segments
            label = ""
            if k >= totalSegments - 2:
                # Calculate duration of the segment safely
                duration = df.index[end-1] - df.index[start]
                label = f"Trend {k+1} (Slope: {slope:,.0f}, {duration.days} days)"
            
            ax2.plot(df.index[start:end], trend_y, color=color, linewidth=3, label=label)

    ax2.set_title("On-Balance Volume (OBV) & Piecewise Trendlines", fontsize=14, loc='left')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    # ax2.set_xlabel("Date")

    # --- Plot 3: Close Price ---
    ax3.plot(priceAction.index, priceAction['Close'], label='Close', color='blue', linewidth=1.5)
    ax3.set_title(f"{ticker} - Close Price", fontsize=14, loc='left')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    # ax3.set_xlabel("Date")

    # Rotate dates for readability
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Save the output
    print("Saving visualization as 'indicator_visualization.png'...")
    plt.savefig('indicator_visualization.png', bbox_inches='tight', dpi=150)
    print("Visualization complete.")

if __name__ == "__main__":
    import matplotlib
    import sys
    matplotlib.use('Agg') # Headless mode
    plotIndicators(sys.argv[1])
