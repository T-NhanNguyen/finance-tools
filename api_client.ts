/**
 * Finance Tools API Client
 * Type-safe HTTP client for interacting with the Python backend
 */

import type {
  GEXRequest,
  GEXResponse,
  IndicatorsRequest,
  IndicatorsResponse,
  ErrorResponse,
  APIConfig,
  APIResponse,
} from './api_types';

export class FinanceAPIClient {
  private baseURL: string;
  private timeout: number;

  constructor(config: APIConfig) {
    this.baseURL = config.baseURL.replace(/\/$/, ''); // Remove trailing slash
    this.timeout = config.timeout || 30000; // Default 30s timeout
  }

  /**
   * Generic fetch wrapper with timeout and error handling
   */
  private async fetchWithTimeout<T>(
    url: string,
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData: ErrorResponse = await response.json();
        return { success: false, error: errorData };
      }

      const data: T = await response.json();
      return { success: true, data };
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          return {
            success: false,
            error: {
              error: 'Request timeout',
              details: `Request exceeded ${this.timeout}ms`,
            },
          };
        }

        return {
          success: false,
          error: {
            error: 'Network error',
            details: error.message,
          },
        };
      }

      return {
        success: false,
        error: {
          error: 'Unknown error',
          details: String(error),
        },
      };
    }
  }

  // ============================================================================
  // Health Check
  // ============================================================================

  async healthCheck(): Promise<APIResponse<{ status: string; service: string }>> {
    return this.fetchWithTimeout(`${this.baseURL}/health`);
  }

  // ============================================================================
  // GEX Endpoints
  // ============================================================================

  /**
   * Get Gamma Exposure analysis (GET request)
   */
  async getGEX(ticker: string, expiration?: string): Promise<APIResponse<GEXResponse>> {
    const params = new URLSearchParams();
    if (expiration) {
      params.append('expiration', expiration);
    }

    const url = `${this.baseURL}/api/gex/${ticker}${params.toString() ? `?${params}` : ''}`;
    return this.fetchWithTimeout<GEXResponse>(url);
  }

  /**
   * Get Gamma Exposure analysis (POST request)
   */
  async postGEX(request: GEXRequest): Promise<APIResponse<GEXResponse>> {
    return this.fetchWithTimeout<GEXResponse>(`${this.baseURL}/api/gex`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // ============================================================================
  // Technical Indicators Endpoints
  // ============================================================================

  /**
   * Get technical indicators analysis (GET request)
   */
  async getIndicators(
    ticker: string,
    period: string = '6mo',
    interval: string = '1d',
    indicators?: string[]
  ): Promise<APIResponse<IndicatorsResponse>> {
    const params = new URLSearchParams({
      period,
      interval,
    });

    if (indicators && indicators.length > 0) {
      params.append('indicators', indicators.join(','));
    }

    const url = `${this.baseURL}/api/indicators/${ticker}?${params}`;
    return this.fetchWithTimeout<IndicatorsResponse>(url);
  }

  /**
   * Get technical indicators analysis (POST request)
   */
  async postIndicators(request: IndicatorsRequest): Promise<APIResponse<IndicatorsResponse>> {
    return this.fetchWithTimeout<IndicatorsResponse>(`${this.baseURL}/api/indicators`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }
}

// ============================================================================
// Default Client Instance
// ============================================================================

/**
 * Create a default client instance
 * Update baseURL to match your deployment
 */
export const createClient = (baseURL: string = 'http://localhost:8000') => {
  return new FinanceAPIClient({ baseURL });
};

// ============================================================================
// Usage Examples (for documentation)
// ============================================================================

/*
// Example 1: Get GEX data
const client = createClient();
const gexResult = await client.getGEX('SPY');

if (gexResult.success) {
  console.log('Spot Price:', gexResult.data.spotPrice);
  console.log('Strikes:', gexResult.data.strikes);
} else {
  console.error('Error:', gexResult.error.error);
}

// Example 2: Get technical indicators
const indicatorsResult = await client.getIndicators('AAPL', '6mo', '1d', ['MACD', 'OBV', 'RSI']);

if (indicatorsResult.success) {
  console.log('Data Points:', indicatorsResult.data.dataPoints.length);
  console.log('Trend Segments:', indicatorsResult.data.trendSegments);
} else {
  console.error('Error:', indicatorsResult.error.error);
}

// Example 3: Using POST requests
const gexPostResult = await client.postGEX({
  ticker: 'QQQ',
  expiration: '2026-02-20'
});

// Example 4: React Hook
import { useState, useEffect } from 'react';

function useGEXData(ticker: string, expiration?: string) {
  const [data, setData] = useState<GEXResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ErrorResponse | null>(null);

  useEffect(() => {
    const client = createClient();
    
    client.getGEX(ticker, expiration).then(result => {
      if (result.success) {
        setData(result.data);
        setError(null);
      } else {
        setError(result.error);
        setData(null);
      }
      setLoading(false);
    });
  }, [ticker, expiration]);

  return { data, loading, error };
}
*/
