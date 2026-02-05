/**
 * TypeScript Type Definitions for Finance Tools API
 * Auto-generated interfaces matching Python Pydantic models
 */

// ============================================================================
// GEX (Gamma Exposure) Types
// ============================================================================

export interface GEXStrikeData {
  /** Option strike price */
  strike: number;
  /** Gamma exposure in millions of dollars */
  gexMillions: number;
  /** Open interest in thousands of contracts */
  openInterestThousands: number;
  /** Whether this strike is at-the-money */
  isATM: boolean;
  /** Normalized GEX for chart scaling (-1 to 1) */
  normalizedGEX: number;
  /** Normalized OI for chart scaling (0 to 1) */
  normalizedOI: number;
}

export interface GEXResponse {
  /** Stock ticker symbol */
  ticker: string;
  /** Option expiration date (YYYY-MM-DD) */
  expiration: string;
  /** Current stock price */
  spotPrice: number;
  /** Days until expiration */
  daysToExpiration: number;
  /** Strike data sorted by strike price */
  strikes: GEXStrikeData[];
  /** Maximum absolute GEX value for scaling */
  maxGEXAbsolute: number;
  /** Maximum open interest for scaling */
  maxOpenInterest: number;
  /** All available expiration dates */
  availableExpirations: string[];
}

// ============================================================================
// Technical Indicators Types
// ============================================================================

export type OBVTrend = "Rising" | "Falling" | "Flat" | "Noise/Transition";

export type TrendSummary = 
  | "NEUTRAL" 
  | "CONFIRMED_DOWNTREND" 
  | "BEARISH_DIVERGENCE" 
  | "ACCUMULATION" 
  | "STRONG_ACCUMULATION";

export type RSISignal = "NEUTRAL" | "OVERSOLD" | "OVERBOUGHT";

export type MACDCrossover = "NEUTRAL" | "BULLISH" | "BEARISH";

export interface IndicatorDataPoint {
  /** Date in YYYY-MM-DD format */
  date: string;
  /** Closing price */
  close: number;
  /** Trading volume */
  volume: number;
  
  // MACD indicators
  /** MACD line value */
  macd?: number | null;
  /** MACD signal line value */
  macdSignal?: number | null;
  /** MACD histogram value */
  macdHistogram?: number | null;
  
  // OBV indicators
  /** On-Balance Volume */
  obv?: number | null;
  /** OBV trend classification */
  obvTrend?: OBVTrend | null;
  
  // RSI
  /** Relative Strength Index (0-100) */
  rsi?: number | null;
  
  // Trading signals
  /** Overall trend summary */
  trendSummary?: TrendSummary | null;
  /** RSI signal */
  rsiSignal?: RSISignal | null;
  /** MACD crossover signal */
  macdCrossover?: MACDCrossover | null;
}

export interface OBVTrendSegment {
  /** Segment number */
  segment: number;
  /** Segment start date (YYYY-MM-DD) */
  start: string;
  /** Segment end date (YYYY-MM-DD) */
  end: string;
  /** Duration in days */
  duration: string;
  /** Theil-Sen slope value (formatted) */
  slope: string;
  /** Price change percentage (formatted) */
  priceChangePct: string;
  /** OBV trend direction */
  obvTrend: "Rising" | "Falling" | "Flat";
  /** Trend interpretation */
  trendSummary: TrendSummary;
}

export interface IndicatorsResponse {
  /** Stock ticker symbol */
  ticker: string;
  /** Time period analyzed */
  period: string;
  /** Data interval (e.g., '1d') */
  interval: string;
  /** Time series data with indicators */
  dataPoints: IndicatorDataPoint[];
  /** Major OBV trend segments */
  trendSegments: OBVTrendSegment[];
  /** Average daily volume */
  averageDailyVolume: number;
  /** Dynamic slope threshold for trend detection */
  dynamicSlopeThreshold: number;
}

// ============================================================================
// Error Response Type
// ============================================================================

export interface ErrorResponse {
  /** Error message */
  error: string;
  /** Ticker that caused the error */
  ticker?: string | null;
  /** Additional error details */
  details?: string | null;
}

// ============================================================================
// Request Types
// ============================================================================

export interface GEXRequest {
  /** Stock ticker symbol */
  ticker: string;
  /** Expiration date (YYYY-MM-DD, partial string, or index). Defaults to nearest. */
  expiration?: string | null;
}

export interface IndicatorsRequest {
  /** Stock ticker symbol */
  ticker: string;
  /** Time period (1mo, 3mo, 6mo, 1y, etc.) */
  period?: string;
  /** Data interval (1d, 1h, etc.) */
  interval?: string;
  /** List of indicators to calculate */
  indicators?: string[];
}

// ============================================================================
// API Client Configuration
// ============================================================================

export interface APIConfig {
  /** Base URL for the API (e.g., 'http://localhost:8000') */
  baseURL: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
}

// ============================================================================
// Utility Types
// ============================================================================

/** API response wrapper for error handling */
export type APIResponse<T> = 
  | { success: true; data: T }
  | { success: false; error: ErrorResponse };
