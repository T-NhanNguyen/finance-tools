#!/usr/bin/env python3
"""
Option Strike Optimizer - CLI Scanner for finding optimal option strikes.
Usage: python -m core.strategies.option_strike_optimizer TICKER --scenario SCENARIO
"""

import argparse
import sys
from datetime import datetime, date
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import json
import heapq
from tabulate import tabulate
from scipy.signal import argrelextrema
from scipy.stats import norm
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

# Constants for Support/Resistance
ROUNDING_BASES = {5000: 25, 2000: 10, 500: 5, 200: 2}
LEAPS_DTE_THRESHOLD = 270  # LEAPS are typically 9 months+

# Add app to path
sys.path.insert(0, '/app')

from core.data.get_options_data import getOptionChain, getEarningsContext
from core.data.bulk_data_loader import fetch_gex_all_expirations, fetch_gex_single
from core.data.get_gex_data import fetch_gex_structured
from core.data.get_stock_price import getHistoricalPrices, PricePeriod, PriceInterval
from core.analysis.calculate_gamma_delta import calculateDelta, calculateGamma, calculateBlackScholesPrice
from core.analysis.calculate_risk_free_rate import getRiskFreeRate
from core.strategies.strategy_config import SCENARIOS, ScenarioConfig, LENDERS
from core.strategies.contract_selling_analyst import ContractSellingAnalyst
from core.data.get_technical_indicator import getIndicators, calculateSMA


class OptionStrikeOptimizer:
    """Optimizer for finding optimal option strikes and expirations."""
    
    def __init__(self):
        """Initialize the optimizer."""
        self.support_resistance_cache = {}
    
    def analyze_strike(self, ticker: str, scenario: str = 'bullish_3month',
                      strike_type: str = 'single_leg', option_type: str = 'call',
                      spread_width: float = 5.0) -> Dict:
        """
        Analyze optimal strikes for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            scenario: Trading scenario (bullish_3month, earnings, ta_breakout)
            strike_type: single_leg or debit_spread
            option_type: call or put (for single leg)
            spread_width: Width for debit spreads
        
        Returns:
            Dictionary with analysis results
        """
        # Get scenario config
        if isinstance(scenario, str):
            scenario_config = SCENARIOS[scenario]
        else:
            scenario_config = scenario
        
        # Fetch GEX data
        chain_data = fetch_gex_all_expirations(ticker)
        if "error" in chain_data:
            return {"error": chain_data["error"]}
            
        spot_price = chain_data.get('spot_price', 0) or chain_data.get('spotPrice', 0)
        
        # Step 1: Get support/resistance levels
        support_resistance = self._get_support_resistance(ticker)
        
        # Step 2: Get expiration candidates
        expirations = filter_expirations_v2(chain_data, scenario_config)
        
        if not expirations:
            return {
                "error": "No suitable expirations found",
                "ticker": ticker,
                "spot_price": spot_price,
                "available_expirations": chain_data.get("availableExpirations", [])
            }
        
        # Step 3: Calculate expected move using first filtered expiration
        first_exp = expirations[0]["expiration"]

        # Earnings context: only fetched for earnings scenario (avoids unnecessary API calls)
        earnings_context = None
        if scenario_config.name == "Earnings Play":
            # Preliminary expected move needed to compute move_richness
            preliminary_move = calculate_expected_move(chain_data, first_exp, scenario_config)
            atm_straddle_price = preliminary_move.get("expected_move", 0)
            earnings_context = getEarningsContext(
                ticker,
                atm_straddle_price=atm_straddle_price,
                spot_price=spot_price
            )

        expected_move = calculate_expected_move(
            chain_data, first_exp, scenario_config, earnings_context=earnings_context
        )
        
        # Step 4: Generate strike candidates
        candidates = []
        
        if strike_type == "single_leg":
            candidates = generate_single_leg_candidates(
                chain_data, scenario_config, option_type, expirations
            )
        else:
            candidates = generate_debit_spread_candidates(
                chain_data, scenario_config, spread_width, expirations
            )
        
        # Step 5: Score candidates
        scored_candidates = []
        for candidate in candidates:
            if strike_type == "single_leg":
                scored = calculate_metrics_single_leg(
                    candidate, spot_price, scenario_config,
                    support_resistance, expected_move
                )
            else:
                scored = calculate_metrics_debit_spread(
                    candidate, spot_price, scenario_config,
                    support_resistance, expected_move
                )
            scored_candidates.append(scored)
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank
        for i, candidate in enumerate(scored_candidates, 1):
            candidate['rank'] = i
        
        # Take top N
        top_candidates = scored_candidates[:10]
        
        return {
            "ticker": ticker,
            "scenario": scenario,
            "spot_price": spot_price,
            "expiration_candidates": expirations,
            "support_resistance": support_resistance,
            "expected_move": expected_move,
            "top_candidates": top_candidates
        }
    
    def scan(self, tickers: List[str], scenario: str = 'bullish_3month',
            strike_type: str = 'single_leg', option_type: str = 'call',
            spread_width: float = 5.0, top_n: int = 5) -> Dict:
        """
        Scan multiple tickers for optimal strikes.
        
        Args:
            tickers: List of ticker symbols
            scenario: Trading scenario
            strike_type: single_leg or debit_spread
            option_type: call or put
            spread_width: Width for debit spreads
            top_n: Number of top results per ticker
        
        Returns:
            Dictionary of results per ticker
        """
        results = {}
        
        for ticker in tickers:
            result = self.analyze_strike(
                ticker,
                scenario,
                strike_type,
                option_type,
                spread_width
            )
            results[ticker] = result
        
        return results
    
    def scan_multiple(self, tickers: List[str], scenario: str = 'bullish_3month',
                     strike_type: str = 'single_leg', option_type: str = 'call',
                     spread_width: float = 5.0, top_n: int = 5) -> Dict:
        """
        Scan multiple tickers for optimal strikes.
        
        Args:
            tickers: List of ticker symbols
            scenario: Trading scenario
            strike_type: single_leg or debit_spread
            option_type: call or put
            spread_width: Width for debit spreads
            top_n: Number of top results per ticker
        
        Returns:
            Dictionary of results per ticker
        """
        results = {}
        
        for ticker in tickers:
            result = self.analyze_strike(
                ticker,
                scenario,
                strike_type,
                option_type,
                spread_width
            )
            results[ticker] = result
        
        return results
    
    def _get_support_resistance(self, ticker: str) -> Dict:
        """Get support/resistance levels for a ticker leveraging clustering and extrema."""
        if ticker in self.support_resistance_cache:
            return self.support_resistance_cache[ticker]
        # Get historical data for the last year
        df = getHistoricalPrices(ticker, period=PricePeriod.ONE_YEAR, interval=PriceInterval.ONE_DAY)
        
        if df is None or df.empty:
            return {"support": [], "resistance": [], "primary_support": None, "primary_resistance": None}
            
        # Standardize columns (yfinance might return multi-index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        last_close = df["Close"].iloc[-1]
        
        # 1. Clustering-based S/R (from python-stock-support-resistance)
        support_list = []
        resistance_list = []
        
        if KMeans is not None:
            try:
                # Find clusters
                price_data = df[['Close']]
                size = min(11, len(df.index))
                wcss = []
                k_models = []
                for i in range(1, size):
                    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
                    kmeans.fit(price_data)
                    wcss.append(kmeans.inertia_)
                    k_models.append(kmeans)
                
                # Find optimal k
                optimum_k = len(wcss) - 1
                for i in range(0, len(wcss) - 1):
                    if abs(wcss[i + 1] - wcss[i]) < 0.1:
                        optimum_k = i
                        break
                
                centers = k_models[optimum_k].cluster_centers_
                all_lines = sorted([item for sublist in centers.tolist() for item in sublist])
                
                # Filter Support/Resistance
                def get_recent(array, price, type_sr):
                    new_arr = [line for line in array if (line < price and type_sr == "S") or (line >= price and type_sr == "R")]
                    if not new_arr: return []
                    temp = heapq.nsmallest(3, new_arr, key=lambda x: abs(x - price))
                    res = [temp[0]]
                    for level in temp[1:]:
                        if abs(level - res[-1]) >= 5: res.append(level)
                    return res
                
                support_list = get_recent(all_lines, last_close, "S")
                resistance_list = get_recent(all_lines, last_close, "R")
            except Exception:
                pass

        # 2. Fallback to Local Extrema if clustering failed or found nothing
        if not support_list or not resistance_list:
            # Check last 20 days for local extrema
            recent_df = df.tail(20)
            if not support_list:
                min_idx = argrelextrema(recent_df["Close"].values, np.less_equal)[0]
                support_list = [recent_df["Close"].iloc[idx] for idx in min_idx if recent_df["Close"].iloc[idx] < last_close]
            if not resistance_list:
                max_idx = argrelextrema(recent_df["Close"].values, np.greater_equal)[0]
                resistance_list = [recent_df["Close"].iloc[idx] for idx in max_idx if recent_df["Close"].iloc[idx] >= last_close]

        # 3. SMA-based S/R: 50-day and 20-day act as institutional price anchors
        for sma_period in (50, 20):
            sma_val = calculateSMA(df["Close"], sma_period).iloc[-1]
            if not np.isnan(sma_val):
                if float(sma_val) < last_close:
                    support_list.append(float(sma_val))
                else:
                    resistance_list.append(float(sma_val))

        # 4. Round and Filter
        def round_p(price, close):
            base = next((v for k, v in sorted(ROUNDING_BASES.items(), reverse=True) if close >= k), 1)
            return float(base * round(price / base))

        support = sorted(list(set([round_p(x, last_close) for x in support_list])), reverse=True)
        resistance = sorted(list(set([round_p(x, last_close) for x in resistance_list])))
        
        sr_data = {
            "support": support,
            "resistance": resistance,
            "primary_support": support[0] if support else None,
            "primary_resistance": resistance[0] if resistance else None
        }
        
        self.support_resistance_cache[ticker] = sr_data
        return sr_data

def filter_expirations_v2(chain_data: Dict, scenario: ScenarioConfig) -> List[Dict]:
    """Filter expirations based on scenario time horizon and structural density."""
    expirations = chain_data.get("expirations", [])
    
    target_days = scenario.time_horizon_days
    # Narrow the window around target days
    tolerance = max(7, int(target_days * 0.25))
    min_days = max(1, target_days - tolerance)
    max_days = target_days + (tolerance * 3)
    
    passing = []
    for exp in expirations:
        dte = exp.get("daysToExpiration", 0)
        oi = exp.get("totalOI", 0)
        
        # Filter by DTE window and liquidity
        # Fallback to volume if OI is 0 (common during yfinance downtime)
        is_liquid = (oi >= scenario.min_open_interest) or (exp.get("volume", 0) > 500)
        
        if min_days <= dte <= max_days and is_liquid:
            passing.append({
                "expiration": exp["expiration"],
                "dte": dte,
                "daysToExpiration": dte,  # Synchronized key
                "oi": oi,
                "volume": exp.get("volume", 0),
                "gex_density": exp.get("gex_density", 0),
            })
    
    if not passing:
        return []

    # Normalize OI and GEX across passing candidates for blended ranking
    max_oi = max((e["oi"] for e in passing), default=1) or 1
    max_gex = max((e["gex_density"] for e in passing), default=1) or 1

    for exp in passing:
        proximity_score = (1.0 / (1.0 + abs(exp["dte"] - target_days) / target_days))**2
        norm_gex = exp["gex_density"] / max_gex
        norm_oi = exp["oi"] / max_oi
        # GEX is primary (market-maker influence), OI is secondary (crowd wisdom)
        exp["horizon_score"] = (0.60 * norm_gex + 0.40 * norm_oi) * proximity_score

    # Sort by horizon score descending
    passing.sort(key=lambda x: x["horizon_score"], reverse=True)
    return passing[:10]

def calculate_expected_move(chain_data: Dict, expiration: str,
                           scenario: ScenarioConfig,
                           earnings_context: Dict = None) -> Dict:
    """Calculate expected move and GEX dominant levels for an expiration."""
    spot_price = chain_data.get("spotPrice", 0)
    ticker = chain_data.get("ticker", "")
    
    chain = getOptionChain(ticker, expiration=expiration)
    
    # Find ATM strike
    call_df = chain.get("calls")
    put_df = chain.get("puts")
    
    if call_df is not None and not call_df.empty:
        strikes = call_df["strike"].values
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
    else:
        atm_strike = spot_price
    
    # 1. Basic Expected Move
    expected_move = 0
    if scenario.expected_move_type == "options":
        if call_df is not None and put_df is not None:
            atm_call = call_df[call_df["strike"] == atm_strike]
            atm_put = put_df[put_df["strike"] == atm_strike]
            if not atm_call.empty and not atm_put.empty:
                call_price = (atm_call.iloc[0].get("bid", 0) + atm_call.iloc[0].get("ask", 0)) / 2
                put_price = (atm_put.iloc[0].get("bid", 0) + atm_put.iloc[0].get("ask", 0)) / 2
                expected_move = call_price + put_price
    
    if expected_move == 0:
        iv = chain_data.get("impliedVolatility", 0.20)
        dte = chain_data.get("daysToExpiration", 30)
        expected_move = spot_price * iv * np.sqrt(max(dte, 1) / 365.0)
    
    # 2. GEX Dominant Levels (Specific to this expiration)
    gex_levels = {"call_wall": None, "put_wall": None, "max_pain": None, "call_zone": None, "put_zone": None}
    if call_df is not None and put_df is not None and not call_df.empty:
        # Use GEX if available, otherwise fallback to OI
        if "gex" in call_df.columns:
            gex_levels["call_wall"] = float(call_df.loc[call_df["gex"].idxmax()]["strike"])
            gex_levels["put_wall"] = float(put_df.loc[put_df["gex"].idxmin()]["strike"])

            # Cluster zones: strikes above the 60th-pct |GEX| on each side.
            # Used only as a fallback tolerance band in scoring — not a primary signal.
            call_above = call_df[call_df["strike"] > spot_price]
            if not call_above.empty:
                call_thresh = call_above["gex"].quantile(0.60)
                call_zone_strikes = call_above.loc[call_above["gex"] >= call_thresh, "strike"]
                if not call_zone_strikes.empty:
                    gex_levels["call_zone"] = (float(call_zone_strikes.min()), float(call_zone_strikes.max()))

            put_below = put_df[put_df["strike"] < spot_price]
            if not put_below.empty:
                put_thresh = put_below["gex"].abs().quantile(0.60)
                put_zone_strikes = put_below.loc[put_below["gex"].abs() >= put_thresh, "strike"]
                if not put_zone_strikes.empty:
                    gex_levels["put_zone"] = (float(put_zone_strikes.min()), float(put_zone_strikes.max()))
        else:
            gex_levels["call_wall"] = float(call_df.loc[call_df["openInterest"].idxmax()]["strike"])
            gex_levels["put_wall"] = float(put_df.loc[put_df["openInterest"].idxmax()]["strike"])
            
        # Simplified Max Pain
        all_s = sorted(list(set(call_df["strike"]) | set(put_df["strike"])))
        pains = []
        for s in all_s:
            c_p = call_df[call_df["strike"] < s].apply(lambda x: (s - x["strike"]) * x["openInterest"], axis=1).sum()
            p_p = put_df[put_df["strike"] > s].apply(lambda x: (x["strike"] - s) * x["openInterest"], axis=1).sum()
            pains.append(c_p + p_p)
        gex_levels["max_pain"] = float(all_s[np.argmin(pains)])

    # 3. Forecast Range biased by GEX (Only if walls are relevant/within 25% of spot)
    upper_expected = spot_price + expected_move
    lower_expected = spot_price - expected_move
    
    if gex_levels["call_wall"] and abs(gex_levels["call_wall"] - spot_price) / spot_price < 0.25:
        # If Call Wall is above spot, it acts as a magnet/resistance
        if gex_levels["call_wall"] > spot_price:
            upper_expected = (upper_expected + gex_levels["call_wall"]) / 2
        else:
            # If Call Wall is below spot, it might act as support
            lower_expected = max(lower_expected, gex_levels["call_wall"])
            
    if gex_levels["put_wall"] and abs(gex_levels["put_wall"] - spot_price) / spot_price < 0.25:
        # If Put Wall is below spot, it acts as support
        if gex_levels["put_wall"] < spot_price:
            lower_expected = (lower_expected + gex_levels["put_wall"]) / 2
        else:
            # If Put Wall is above spot, it might act as resistance
            upper_expected = min(upper_expected, gex_levels["put_wall"])
        
    # 4. Earnings Context: widen bounds when historical realized moves exceed priced-in move
    if earnings_context and earnings_context.get("avg_realized_move"):
        avg_realized = earnings_context["avg_realized_move"]
        priced_in_pct = expected_move / spot_price if spot_price > 0 else 0
        if avg_realized > priced_in_pct:
            # Historical move > straddle pricing: use realized avg to set targets
            realized_move_dollars = avg_realized * spot_price
            upper_expected = max(upper_expected, spot_price + realized_move_dollars)
            lower_expected = min(lower_expected, spot_price - realized_move_dollars)

    return {
        "expected_move": float(expected_move),
        "upper_expected": float(upper_expected),
        "lower_expected": float(lower_expected),
        "atm_strike": float(atm_strike),
        "gex_levels": gex_levels,
        "earnings_context": earnings_context or {}
    }


def generate_single_leg_candidates(chain_data: Dict, scenario: ScenarioConfig,
                                   option_type: str, filtered_expirations: List[Dict]) -> List[Dict]:
    """Generate single leg candidates using filtered expirations."""
    spot_price = chain_data.get("spotPrice", 0)
    candidates = []
    
    for exp in filtered_expirations:
        exp_date = exp["expiration"]
        dte = exp.get("daysToExpiration") or exp.get("dte") or 30
        
        if dte is None or dte <= 0: continue
        
        chain = getOptionChain(chain_data.get("ticker", ""), expiration=exp_date)
        options = chain.get("calls" if option_type == "call" else "puts")
        
        if options is None or options.empty: continue
        
        for _, row in options.iterrows():
            strike = row["strike"]
            price = (row.get("bid", 0) + row.get("ask", 0)) / 2
            if price <= 0: continue
            
            # ITM Filter: Avoid deep ITM unless LEAPS
            is_leaps = dte >= LEAPS_DTE_THRESHOLD
            if not is_leaps:
                if option_type == "call" and strike < spot_price * 0.95: continue
                if option_type == "put" and strike > spot_price * 1.05: continue
            
            delta = row.get("delta")
            if delta is None or np.isnan(delta) or delta == 0:
                delta = calculateDelta(spot_price, strike, dte/365.0, row.get("impliedVolatility", 0.2), option_type)
            
            delta_abs = abs(delta)
            # Filter for tradeable delta range
            # Non-LEAPS: 0.15 to 0.70
            # LEAPS: 0.20 to 0.70 (tighter cap to avoid near-delta-1 deep ITM)
            if is_leaps:
                if delta_abs < 0.20 or delta_abs > 0.70: continue
            else:
                if delta_abs < 0.15 or delta_abs > 0.70: continue
            
            candidates.append({
                "option_type": option_type,
                "strike": strike,
                "expiration": exp_date,
                "dte": dte,
                "price": price,
                "delta": delta,
                "gamma": row.get("gamma", 0),
                "theta": row.get("theta", 0),
                "vega": row.get("vega", 0),
                "iv": row.get("impliedVolatility", 0.20),
                "oi": row.get("openInterest", 0),
                "volume": row.get("volume", 0),
                "metrics": {}
            })
    
    candidates.sort(key=lambda x: abs(x["delta"] - 0.40))
    return candidates


def generate_debit_spread_candidates(chain_data: Dict, scenario: ScenarioConfig,
                                    spread_width: float = 5.0, filtered_expirations: List[Dict] = None) -> List[Dict]:
    """Generate debit spread candidates using filtered expirations."""
    spot_price = chain_data.get("spotPrice", 0)
    candidates = []
    
    if filtered_expirations is None:
        filtered_expirations = chain_data.get("expirations", [])
        
    is_bullish = scenario.direction == "bullish"
    opt_key = "calls" if is_bullish else "puts"
    
    for exp in filtered_expirations:
        exp_date = exp["expiration"]
        dte = exp.get("daysToExpiration") or exp.get("dte") or 30
        if dte is None or dte <= 0: continue
        
        chain = getOptionChain(chain_data.get("ticker", ""), expiration=exp_date)
        options = chain.get(opt_key)
        if options is None or options.empty: continue
        
        for _, row in options.iterrows():
            long_strike = row["strike"]
            
            # ITM Filter: Avoid deep ITM unless LEAPS
            is_leaps = dte >= LEAPS_DTE_THRESHOLD
            if is_leaps:
                # Even for LEAPS, avoid extreme deep ITM
                if is_bullish:
                    if long_strike < spot_price * 0.70: continue
                else:
                    if long_strike > spot_price * 1.30: continue
            else:
                if is_bullish:
                    if long_strike < spot_price * 0.95: continue
                else:
                    if long_strike > spot_price * 1.05: continue
            
            # Short strike selection based on direction
            if is_bullish:
                short_strike = long_strike + spread_width
            else:
                short_strike = long_strike - spread_width
                
            short_row = options[options["strike"] == short_strike]
            if short_row.empty: continue
            
            long_p = (row.get("bid", 0) + row.get("ask", 0)) / 2
            short_p = (short_row.iloc[0].get("bid", 0) + short_row.iloc[0].get("ask", 0)) / 2
            debit = long_p - short_p
            
            # Risk/Reward Filter: Debit shouldn't be > 80% of width for spreads
            if debit <= 0 or debit > spread_width * 0.80: continue
            
            # Calculate Greeks if missing
            long_iv = row.get("impliedVolatility", 0.20)
            short_iv = short_row.iloc[0].get("impliedVolatility", 0.20)
            t = dte / 365.0
            
            long_delta = row.get("delta")
            if long_delta is None or np.isnan(long_delta):
                long_delta = calculateDelta(spot_price, long_strike, t, long_iv, "call" if is_bullish else "put")
            
            short_delta = short_row.iloc[0].get("delta")
            if short_delta is None or np.isnan(short_delta):
                short_delta = calculateDelta(spot_price, short_strike, t, short_iv, "call" if is_bullish else "put")
            
            candidates.append({
                "option_type": "debit_spread",
                "long_strike": long_strike,
                "short_strike": short_strike,
                "expiration": exp_date,
                "dte": dte,
                "debit": debit,
                "long_iv": long_iv,
                "short_iv": short_iv,
                "long_delta": long_delta,
                "short_delta": short_delta,
                "long_gamma": row.get("gamma", 0),
                "short_gamma": short_row.iloc[0].get("gamma", 0),
                "long_theta": row.get("theta", 0),
                "short_theta": short_row.iloc[0].get("theta", 0),
                "long_vega": row.get("vega", 0),
                "short_vega": short_row.iloc[0].get("vega", 0),
                "long_oi": row.get("openInterest", 0),
                "short_oi": short_row.iloc[0].get("openInterest", 0),
                "metrics": {}
            })
    
    candidates.sort(key=lambda x: x["debit"])
    return candidates


def calculate_metrics_single_leg(candidate: Dict, spot_price: float,
                                 scenario: ScenarioConfig,
                                 support_resistance: Dict,
                                 expected_move: Dict) -> Dict:
    """Calculate metrics for single leg candidate using normal CDF probability."""
    weights = scenario.scoring_weights
    target_price = expected_move.get("upper_expected", spot_price) if scenario.direction == "bullish" else \
                   expected_move.get("lower_expected", spot_price)
    
    dte = max(candidate.get("dte", 30), 1)
    iv = candidate.get("iv", 0.20)
    
    # Probability of price being above strike at expiration (Normal CDF)
    t = dte / 365.0
    d2 = (np.log(spot_price / candidate["strike"]) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    prob_target = norm.cdf(d2) if candidate["option_type"] == "call" else 1 - norm.cdf(d2)
    
    # Payoff at target
    if candidate["option_type"] == "call":
        payoff = max(target_price - candidate["strike"], 0)
    else:
        payoff = max(candidate["strike"] - target_price, 0)
    
    expected_pnl = prob_target * payoff - candidate["price"]
    
    # Normalize scores
    ev_score = 0.5 + 0.5 * np.tanh(expected_pnl / (candidate["price"] + 1))
    prob_score = prob_target
    
    # GEX Alignment: primary wall → +0.20; cluster zone fallback → +0.08 (substitute tolerance only)
    gex_bonus = 0
    gex_levels = expected_move.get("gex_levels", {})
    is_bullish = scenario.direction == "bullish"
    relevant_wall = gex_levels.get("call_wall" if is_bullish else "put_wall")
    relevant_zone = gex_levels.get("call_zone" if is_bullish else "put_zone")
    if relevant_wall and abs(candidate["strike"] - relevant_wall) < (spot_price * 0.02):
        gex_bonus = 0.20
    elif relevant_zone:
        zone_low, zone_high = relevant_zone
        if zone_low <= candidate["strike"] <= zone_high:
            gex_bonus = 0.08
    
    # Liquidity Score: OI + volume normalized, penalized by wide bid-ask
    oi = candidate.get("oi", 0)
    volume = candidate.get("volume", 0)
    liquidity_raw = min(1.0, (oi + volume) / 5000.0)
    bid = candidate.get("bid", 0)
    ask = candidate.get("ask", candidate["price"] * 2)
    spread_ratio = (ask - bid) / (candidate["price"] + 0.01)
    liquidity_score = liquidity_raw * max(0.0, 1.0 - spread_ratio)

    # Theta Penalty: normalized decay rate relative to option cost
    theta = abs(candidate.get("theta", 0))
    theta_penalty = min(1.0, theta / (candidate["price"] + 0.01))

    score = (gex_bonus * weights.get("gex_weight", 0.30) +
            liquidity_score * weights.get("liquidity_weight", 0.25) +
            prob_score * weights.get("prob_weight", 0.20) +
            ev_score * weights.get("ev_weight", 0.12) -
            theta_penalty * weights.get("theta_weight", 0.08))
            
    # ROI Bonus: Reward better return on capital
    roi = expected_pnl / (candidate["price"] + 0.01)
    roi_score = 0.1 * min(1.0, roi / 0.5)
    score += roi_score
    
    # Delta Penalty: Penalize extremely deep ITM (Delta > 0.8)
    if abs(candidate.get("delta", 0)) > 0.80:
        score -= 0.15
    
    score = max(0.0, min(1.0, score))
    
    # Invalidation Level (Stop-Loss)
    invalidation_level = None
    if is_bullish:
        if support_resistance.get("support"):
            invalidation_level = support_resistance["support"][0]
    else:
        if support_resistance.get("resistance"):
            invalidation_level = support_resistance["resistance"][0]

    # Fill metrics
    candidate["score"] = round(score, 4)
    candidate["metrics"]["score"] = candidate["score"]
    candidate["metrics"]["ev_score"] = round(ev_score, 4)
    candidate["metrics"]["probability_of_profit"] = round(prob_score, 4)
    candidate["metrics"]["expected_pnl"] = round(expected_pnl, 2) if expected_pnl is not None else None
    candidate["metrics"]["payoff_at_target"] = round(payoff, 2)
    candidate["metrics"]["delta"] = round(candidate.get("delta", 0), 4)
    candidate["metrics"]["gamma"] = round(candidate.get("gamma", 0), 6)
    candidate["metrics"]["theta"] = round(candidate.get("theta", 0), 4)
    candidate["metrics"]["vega"] = round(candidate.get("vega", 0), 4)
    candidate["metrics"]["iv"] = round(candidate.get("iv", 0), 4)
    candidate["metrics"]["prob_target"] = round(prob_target, 4)
    candidate["metrics"]["invalidation_level"] = invalidation_level

    return candidate



def calculate_metrics_debit_spread(candidate: Dict, spot_price: float,
                                   scenario: ScenarioConfig,
                                   support_resistance: Dict,
                                   expected_move: Dict) -> Dict:
    """Calculate metrics for debit spread candidate with ITM filtering and GEX walls."""
    weights = scenario.scoring_weights
    target_price = expected_move.get("upper_expected", spot_price) if scenario.direction == "bullish" else \
                   expected_move.get("lower_expected", spot_price)
    
    dte = max(candidate.get("dte", 30), 1)
    iv = candidate.get("long_iv", 0.20)
    t = dte / 365.0
    
    # Probability of profit (Price > Breakeven for call spreads, Price < Breakeven for put spreads)
    is_bullish = scenario.direction == "bullish"
    if is_bullish:
        break_even = candidate["long_strike"] + candidate["debit"]
        d2 = (np.log(spot_price / break_even) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
        prob_profit = norm.cdf(d2)
    else:
        break_even = candidate["long_strike"] - candidate["debit"]
        d2 = (np.log(spot_price / break_even) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
        prob_profit = 1 - norm.cdf(d2)
    
    # Payoff at target
    if is_bullish:
        if target_price >= candidate["short_strike"]:
            payoff = candidate["short_strike"] - candidate["long_strike"] - candidate["debit"]
        elif target_price >= candidate["long_strike"]:
            payoff = target_price - candidate["long_strike"] - candidate["debit"]
        else:
            payoff = -candidate["debit"]
    else: # Bearish
        if target_price <= candidate["short_strike"]:
            payoff = candidate["long_strike"] - candidate["short_strike"] - candidate["debit"]
        elif target_price <= candidate["long_strike"]:
            payoff = candidate["long_strike"] - target_price - candidate["debit"]
        else:
            payoff = -candidate["debit"]
            
    expected_pnl = prob_profit * payoff
    
    # Normalize scores
    ev_score = 0.5 + 0.5 * np.tanh(expected_pnl / (candidate["debit"] + 1))
    prob_score = prob_profit
    
    # Max Profit/Loss Ratio
    max_profit = abs(candidate["short_strike"] - candidate["long_strike"]) - candidate["debit"]
    max_loss = candidate["debit"]
    pl_score = min(1.0, (max_profit / max_loss) / 3) if max_loss > 0 else 0
    
    # GEX Alignment: primary wall → +0.20; cluster zone fallback → +0.08 (substitute tolerance only)
    gex_bonus = 0
    gex_levels = expected_move.get("gex_levels", {})
    is_bullish = scenario.direction == "bullish"
    relevant_wall = gex_levels.get("call_wall" if is_bullish else "put_wall")
    relevant_zone = gex_levels.get("call_zone" if is_bullish else "put_zone")
    if relevant_wall and abs(candidate["long_strike"] - relevant_wall) < (spot_price * 0.02):
        gex_bonus = 0.20
    elif relevant_zone:
        zone_low, zone_high = relevant_zone
        if zone_low <= candidate["long_strike"] <= zone_high:
            gex_bonus = 0.08

    # Liquidity Score: combined long+short leg OI, penalized by debit spread width
    long_oi = candidate.get("long_oi", 0)
    short_oi = candidate.get("short_oi", 0)
    liquidity_raw = min(1.0, (long_oi + short_oi) / 10000.0)
    liquidity_score = liquidity_raw

    # Theta Penalty: net theta decay relative to debit cost
    net_theta = abs(candidate.get("long_theta", 0) + candidate.get("short_theta", 0))
    theta_penalty = min(1.0, net_theta / (candidate["debit"] + 0.01))

    # Spread cost penalty: debit too high relative to width
    spread_width = candidate["short_strike"] - candidate["long_strike"]
    cost_ratio = candidate["debit"] / spread_width if spread_width > 0 else 1.0
    spreadcost_penalty = max(0.0, cost_ratio - 0.4)  # penalize above 40% of width

    score = (gex_bonus * weights.get("gex_weight", 0.30) +
            liquidity_score * weights.get("liquidity_weight", 0.25) +
            prob_score * weights.get("prob_weight", 0.20) +
            ev_score * weights.get("ev_weight", 0.12) -
            theta_penalty * weights.get("theta_weight", 0.08) -
            spreadcost_penalty * weights.get("spreadcost_weight", 0.05))
            
    # ROI Bonus: Reward better return on capital
    roi = expected_pnl / (candidate["debit"] + 0.01)
    roi_score = 0.1 * min(1.0, roi / 1.0)
    score += roi_score
    
    # P/L ratio micro-bonus
    pl_bonus = 0.05 * min(1.0, (max_profit / max_loss) / 3) if max_loss > 0 else 0
    score += pl_bonus
    
    # ITM Penalty: Spread is too "safe" (debit > 60% of width)
    if candidate["debit"] > spread_width * 0.6:
        score -= 0.1
    
    score = max(0.0, min(1.0, score))
    
    # Invalidation Level (Stop-Loss)
    invalidation_level = None
    if is_bullish:
        if support_resistance.get("support"):
            invalidation_level = support_resistance["support"][0]
    else:
        if support_resistance.get("resistance"):
            invalidation_level = support_resistance["resistance"][0]

    # Fill metrics
    candidate["score"] = round(score, 4)
    candidate["metrics"]["score"] = candidate["score"]
    candidate["metrics"]["ev_score"] = round(ev_score, 4)
    candidate["metrics"]["probability_of_profit"] = round(prob_score, 4)
    candidate["metrics"]["expected_pnl"] = round(expected_pnl, 2) if expected_pnl is not None else None
    candidate["metrics"]["payoff_at_target"] = round(payoff, 2)
    candidate["metrics"]["long_strike"] = candidate["long_strike"]
    candidate["metrics"]["short_strike"] = candidate["short_strike"]
    candidate["metrics"]["net_debit"] = round(candidate["debit"], 2)
    candidate["metrics"]["max_profit"] = round(max_profit, 2)
    candidate["metrics"]["max_loss"] = round(max_loss, 2)
    candidate["metrics"]["break_even"] = round(break_even, 2)
    candidate["metrics"]["long_delta"] = round(candidate.get("long_delta", 0), 4)
    candidate["metrics"]["short_delta"] = round(candidate.get("short_delta", 0), 4)
    candidate["metrics"]["delta"] = round(candidate.get("long_delta", 0) + candidate.get("short_delta", 0), 4)
    candidate["metrics"]["gamma"] = round(candidate.get("long_gamma", 0) + candidate.get("short_gamma", 0), 6)
    candidate["metrics"]["theta"] = round(candidate.get("long_theta", 0) + candidate.get("short_theta", 0), 4)
    candidate["metrics"]["vega"] = round(candidate.get("long_vega", 0) + candidate.get("short_vega", 0), 4)
    candidate["metrics"]["invalidation_level"] = invalidation_level
    
    return candidate


# =============================================================================
# Efficiency Comparison (New Design)
# See EFFICIENCY_COLUMNS.md in this directory for a full column reference.
# =============================================================================

def compare_efficiency(ticker: str, deadline: str, top_n: int = 5,
                       target_price: float = None, near_date_cutoff: int = 30,
                       option_type: str = "call") -> Dict:
    """
    Compare ITM near-dated vs OTM further-out option contracts by
    cost-to-cover (extrinsic premium / expected move) and theta-adjusted ROI.

    Args:
        ticker: Stock ticker symbol (e.g., 'SPY')
        deadline: Deadline date in YYYY-MM-DD format (market regime end)
        top_n: Number of top candidates to display per category
        target_price: Optional price target. If provided, expected move =
                      abs(target_price - spot_price). Falls back to
                      calculate_expected_move() otherwise.
        near_date_cutoff: DTE threshold separating near-dated from further-out (default 30)
        option_type: 'call' (bullish) or 'put' (bearish) — defaults to 'call'

    Returns:
        Dictionary with results, candidates, and warnings
    """
    # ── Direction-specific field keys and formulas ──
    _opt_oi = "callOI" if option_type == "call" else "putOI"
    _opt_bid = "callBid" if option_type == "call" else "putBid"
    _opt_ask = "callAsk" if option_type == "call" else "putAsk"
    _opt_iv = "callIV" if option_type == "call" else "putIV"
    _intrinsic_fn = (lambda s, k: max(0.0, s - k)) if option_type == "call" else (lambda s, k: max(0.0, k - s))
    _payoff_fn = (lambda t, k: max(0.0, t - k)) if option_type == "call" else (lambda t, k: max(0.0, k - t))
    _scenario_direction = "bullish" if option_type == "call" else "bearish"
    _exp_move_sign = 1 if option_type == "call" else -1
    today = date.today()

    # Try multiple common date formats
    deadline_dt = None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%Y%m%d", "%d/%m/%Y",
                "%m-%d-%Y", "%d-%m-%Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            deadline_dt = datetime.strptime(deadline, fmt).date()
            break
        except ValueError:
            continue

    if deadline_dt is None:
        return {"error": f"Invalid deadline format: '{deadline}'."
                         f" Accepted formats: YYYY-MM-DD, MM/DD/YYYY, YYYYMMDD, "
                         f"'Jul 1, 2026', 'July 1, 2026', etc."}

    if deadline_dt <= today:
        return {"error": f"deadline {deadline} is in the past or today. Please provide a future date."}

    target_dte = (deadline_dt - today).days

    # Fetch option chain data (via GEX cache — avoids redundant yfinance calls)
    chain_data = fetch_gex_all_expirations(ticker)
    if "error" in chain_data:
        return {"error": chain_data["error"]}

    spot_price = chain_data.get('spot_price', 0) or chain_data.get('spotPrice', 0)
    if not spot_price:
        return {"error": f"Could not determine spot price for {ticker}"}

    # Determine expected move
    dollar_expected_move = None
    if target_price is not None:
        dollar_expected_move = abs(target_price - spot_price)
    else:
        # Fallback: use first available expiration's expected move
        available = chain_data.get("availableExpirations", [])
        if available:
            fallback_exp = available[0]
            em_result = calculate_expected_move(chain_data, fallback_exp, ScenarioConfig(
                name="Efficiency", direction=_scenario_direction, time_horizon_days=target_dte,
                min_open_interest=0, min_delta=0.0, max_delta=1.0,
                expected_move_type="options",
                scoring_weights={}))
            dollar_expected_move = em_result.get("expected_move", 0)
            if not dollar_expected_move or dollar_expected_move <= 0:
                # Try volatility-based fallback
                iv = chain_data.get("impliedVolatility", 0.20)
                dollar_expected_move = spot_price * iv * np.sqrt(target_dte / 365.0)

    if not dollar_expected_move or dollar_expected_move <= 0:
        return {"error": "Could not compute expected move. Provide --target-price or check data availability."}

    # Use available expirations from chain_data (avoid redundant yf.Ticker call)
    available = chain_data.get("availableExpirations", [])
    if not available:
        return {"error": f"No option expirations found for {ticker}"}

    # Filter expirations: only those on or after today, before or on deadline
    valid_expirations = []
    for exp_date_str in available:
        try:
            exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if exp_date >= today:
            valid_expirations.append((exp_date_str, (exp_date - today).days))

    if not valid_expirations:
        return {"error": f"No expirations found on or after {deadline} for {ticker}"}

    # Remove same-day expirations (DTE == 0 — no time value, can't compute Greeks)
    valid_expirations = [(e, d) for e, d in valid_expirations if d > 0]
    if not valid_expirations:
        return {"error": f"No future expirations found (all remaining ones expire today)"}

    # Sort by DTE
    valid_expirations.sort(key=lambda x: x[1])

    # Fetch chains via cached GEX pipeline to avoid redundant yfinance calls
    # fetch_gex_single uses cache_manager (disk+memory LRU cache, 15-min TTL during market)
    expiration_details = []
    max_oi = 0
    for exp_date_str, dte in valid_expirations:
        gex_data = fetch_gex_single(ticker, expiration=exp_date_str)
        if "error" in gex_data:
            continue

        strikes = gex_data.get("strikes", [])
        if not strikes:
            continue

        # Compute total OI from the GEX data (OI is in thousands in GEX data)
        total_oi = int(sum(s.get(_opt_oi, 0) for s in strikes) * 1000)

        expiration_details.append({
            "expiration": exp_date_str,
            "dte": dte,
            "total_oi": total_oi,
            "strikes": strikes,
            "spotPrice": gex_data.get("spotPrice", spot_price)
        })
        if total_oi > max_oi:
            max_oi = total_oi

    if not expiration_details:
        return {"error": f"No {option_type} option data found for {ticker}"}

    # OI filter: skip expirations with OI < 25% of max OI
    oi_threshold = max_oi * 0.25
    filtered_expirations = [e for e in expiration_details if e["total_oi"] >= oi_threshold]

    if not filtered_expirations:
        return {
            "ticker": ticker,
            "deadline": deadline,
            "spot_price": spot_price,
            "expected_move": dollar_expected_move,
            "warnings": [f"No expirations passed OI filter (>= {oi_threshold:.0f} OI, 25% of max {max_oi}). Try a more liquid ticker."],
            "candidates": [],
            "itm_near": [],
            "otm_far": []
        }

    # Scan candidates — prefer GEX cache data (no redundant yfinance calls),
    # but fall back to raw chain for lastPrice when GEX bid/ask are 0 (off-hours)
    candidates = []
    for exp_info in filtered_expirations:
        exp_date_str = exp_info["expiration"]
        dte = exp_info["dte"]
        strikes_data = exp_info["strikes"]
        t = dte / 365.0
        exp_spot = exp_info.get("spotPrice", spot_price)

        for s in strikes_data:
            strike = s["strike"]

            # Use GEX cached prices. The cache_manager handles TTL:
            # - During market hours: 15-min TTL, live bid/ask from yfinance
            # - After market close: data from last settlement is fresh and has bid/ask
            # - If never populated: 0 bid/ask → skip gracefully (no wasteful API call)
            bid = s.get(_opt_bid, 0) or 0
            ask = s.get(_opt_ask, 0) or 0
            if bid > 0 and ask > 0:
                effective_price = ask  # What you'd pay
                last_price = (bid + ask) / 2
            elif bid > 0:
                effective_price = bid
                last_price = bid
            elif ask > 0:
                effective_price = ask
                last_price = ask
            else:
                # Cache has no price data — don't fetch fresh (market closed, nothing new)
                continue

            # IV from GEX cache
            raw_iv = s.get(_opt_iv, 0)
            if raw_iv is None or np.isnan(raw_iv) or raw_iv <= 0.01:
                iv = 0.20
            else:
                iv = raw_iv

            delta = calculateDelta(exp_spot, strike, t, iv, option_type)
            delta_abs = abs(delta)

            # Classify by delta
            is_itm = 0.55 <= delta_abs <= 0.80
            is_otm = 0.15 <= delta_abs <= 0.45

            if not is_itm and not is_otm:
                continue

            # Determine category
            if is_itm and dte <= near_date_cutoff:
                category = "ITM_Near"
            elif is_otm and dte > near_date_cutoff:
                category = "OTM_Far"
            else:
                # Cross-category: ITM far-dated or OTM near-dated — still include in summary
                if is_itm:
                    category = "ITM_Far"
                else:
                    category = "OTM_Near"

            # Intrinsic value
            intrinsic = _intrinsic_fn(spot_price, strike)
            extrinsic = effective_price - intrinsic

            # Cost-to-Cover = extrinsic premium / dollar expected move
            cost_to_cover = extrinsic / dollar_expected_move if dollar_expected_move > 0 else float('inf')

            # Theta not available from yfinance — compute estimate using Black-Scholes
            theta_abs = 0.0
            if iv > 0.01:
                try:
                    bs_price = calculateBlackScholesPrice(spot_price, strike, t, iv, option_type)
                    # Approximate theta using finite difference: shift T by 1 day
                    t_minus_1 = max((dte - 1) / 365.0, 1/365.0)
                    bs_price_tomorrow = calculateBlackScholesPrice(spot_price, strike, t_minus_1, iv, option_type)
                    theta_abs = abs(bs_price_tomorrow - bs_price)
                except Exception:
                    theta_abs = effective_price * 0.01  # Rough estimate: 1% of price as daily theta
            else:
                # Crude theta estimate: deep ITM options decay slower, OTM decay faster
                moneyness = abs(strike - spot_price) / spot_price
                theta_abs = effective_price * 0.01 * (1 + moneyness * 2) if moneyness > 0 else effective_price * 0.005

            # Expected P&L at target
            computed_price = target_price if target_price is not None else (spot_price + dollar_expected_move * _exp_move_sign)
            payoff_at_target = _payoff_fn(computed_price, strike)
            expected_pnl = payoff_at_target - effective_price

            # Theta ROI = expected P&L / (theta burn over target holding period)
            theta_roi = None
            if theta_abs > 0 and target_dte > 0:
                theta_burn = theta_abs * target_dte  # Total theta decay over target holding period
                if theta_burn > 0:
                    theta_roi = expected_pnl / theta_burn

            candidate = {
                "strike": strike,
                "expiration": exp_date_str,
                "dte": dte,
                "delta": round(delta, 4),
                "bid": bid,
                "ask": ask,
                "last_price": last_price,
                "effective_price": round(effective_price, 2),
                "intrinsic": round(intrinsic, 2),
                "extrinsic": round(extrinsic, 2),
                "cost_to_cover": round(cost_to_cover, 4) if cost_to_cover != float('inf') else None,
                "theta_abs": round(theta_abs, 4),
                "theta_roi": round(theta_roi, 4) if theta_roi is not None else None,
                "expected_pnl": round(expected_pnl, 2),
                "payoff_at_target": round(payoff_at_target, 2),
                "category": category,
                "oi": int(s.get(_opt_oi, 0) * 1000),
                "volume": int(s.get("volume", 0)),
                "iv": round(iv, 4),
                "combined_score": 0.0,  # will be set after normalization
                "option_type": option_type
            }
            candidates.append(candidate)

    if not candidates:
        return {
            "ticker": ticker,
            "deadline": deadline,
            "spot_price": spot_price,
            "expected_move": dollar_expected_move,
            "warnings": [f"No candidates found matching delta ranges (ITM: 0.55-0.80, OTM: 0.15-0.45) for {ticker}"],
            "candidates": [],
            "itm_near": [],
            "otm_far": []
        }

    # Normalize and compute combined score
    cost_values = [c["cost_to_cover"] for c in candidates if c["cost_to_cover"] is not None]
    theta_values = [c["theta_roi"] for c in candidates if c["theta_roi"] is not None]

    cost_min = min(cost_values) if cost_values else 0
    cost_max = max(cost_values) if cost_values else 1
    cost_range = cost_max - cost_min if cost_max != cost_min else 1

    theta_min = min(theta_values) if theta_values else 0
    theta_max = max(theta_values) if theta_values else 1
    theta_range = theta_max - theta_min if theta_max != theta_min else 1

    for c in candidates:
        # Cost score: lower cost_to_cover is better (invert and normalize 0-1)
        cost_score = 1.0 - ((c["cost_to_cover"] - cost_min) / cost_range) if c["cost_to_cover"] is not None else 0.0
        # Theta ROI score: higher is better (normalize 0-1)
        theta_score = ((c["theta_roi"] - theta_min) / theta_range) if c["theta_roi"] is not None else 0.0
        # Safety: clamp scores to valid range
        cost_score = max(0.0, min(1.0, cost_score))
        theta_score = max(0.0, min(1.0, theta_score))
        # Combined: equal weight
        c["combined_score"] = round(0.5 * cost_score + 0.5 * theta_score, 4)

    # Sort by combined score descending
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)

    # Add rank
    for i, c in enumerate(candidates, 1):
        c["rank"] = i

    # Group by category for the two main groups
    itm_near = [c for c in candidates if c["category"] == "ITM_Near"][:top_n]
    otm_far = [c for c in candidates if c["category"] == "OTM_Far"][:top_n]

    # Warnings
    warnings = []
    is_bearish = option_type == "put"
    if target_price is not None and (
        (not is_bearish and target_price <= spot_price) or
        (is_bearish and target_price >= spot_price)
    ):
        direction_label = "bullish" if not is_bearish else "bearish"
        warnings.append(f"target_price (${target_price:.2f}) is {'below' if not is_bearish else 'above'} spot (${spot_price:.2f}). Expected move is against {direction_label} direction.")
    if not itm_near:
        warnings.append(f"No ITM near-dated candidates found (delta 0.55-0.80, DTE <= {near_date_cutoff}).")
    if not otm_far:
        warnings.append(f"No OTM further-out candidates found (delta 0.15-0.45, DTE > {near_date_cutoff}).")
    if len(candidates) < top_n:
        warnings.append(f"Only {len(candidates)} candidates found (requested top {top_n}).")
    if len(candidates) == 0:
        warnings.append("No candidates matching criteria were found.")

    return {
        "ticker": ticker,
        "deadline": deadline,
        "option_type": option_type,
        "target_dte": target_dte,
        "spot_price": spot_price,
        "expected_move": round(dollar_expected_move, 2),
        "expected_move_pct": round(dollar_expected_move / spot_price * 100, 2),
        "near_date_cutoff": near_date_cutoff,
        "oi_threshold": oi_threshold,
        "max_oi": max_oi,
        "expirations_scanned": len(expiration_details),
        "expirations_passed_oi_filter": len(filtered_expirations),
        "total_candidates": len(candidates),
        "candidates": candidates[:top_n],
        "itm_near": itm_near,
        "otm_far": otm_far,
        "warnings": warnings
    }


def _format_efficiency_json(result: Dict) -> str:
    """Build a compact JSON block for the efficiency comparison — candidates only, no group tables."""
    out = dict(result)
    # Strip group table fields that are redundant with the ranked candidates list
    for key in ("itm_near", "otm_far", "total_candidates"):
        out.pop(key, None)
    return json.dumps(out, separators=(",", ":"))


def _format_diagonal_json(diagonal_plans: List[Dict], ticker: str = "", option_type: str = "call",
                          deadline: str = "", target_price: float = None, spot_price: float = 0) -> str:
    """Build a compact JSON block for diagonal spread plans — no Ranked Summary."""
    out = {
        "ticker": ticker,
        "option_type": option_type,
        "deadline": deadline,
        "spot_price": spot_price,
        "plans": diagonal_plans
    }
    if target_price is not None:
        out["target_price"] = target_price
    return json.dumps(out, separators=(",", ":"))


def format_efficiency_results(result: Dict) -> str:
    """Format efficiency comparison results into readable tables."""
    from tabulate import tabulate

    lines = []

    ticker = result.get("ticker", "")
    spot = result.get("spot_price", 0)
    exp_move = result.get("expected_move", 0)
    exp_move_pct = result.get("expected_move_pct", 0)
    target_date = result.get("deadline", "")
    target_dte = result.get("target_dte", 0)
    boundary = result.get("near_date_cutoff", 30)

    lines.append(f"\n{'='*75}")
    option_label = result.get("option_type", "call").capitalize()
    lines.append(f"  {ticker.upper()} — Efficiency Comparison ({option_label})")
    lines.append(f"{'='*75}")
    lines.append(f"  Target Deadline: {target_date} (in ~{target_dte} days)")
    lines.append(f"  Spot: ${spot:.2f}  |  Expected Move: ${exp_move:.2f} ({exp_move_pct:.1f}%)")
    lines.append(f"  DTE Boundary: {boundary}d  |  Expirations: {result.get('expirations_passed_oi_filter', 0)} (of {result.get('expirations_scanned', 0)} passed OI filter)")
    lines.append(f"{'='*75}")

    # Warnings
    warnings = result.get("warnings", [])
    if warnings:
        lines.append(f"\n  ⚠  Warnings:")
        for w in warnings:
            lines.append(f"     • {w}")

    candidates = result.get("candidates", [])
    if not candidates:
        lines.append("\n  No candidates found.")
        return "\n".join(lines)

    # Ranked summary table with Category column
    lines.append(f"\n  ── Ranked Summary (Top {len(candidates)}) ──")
    summary_headers = ["Rank", "Category", "Expiry", "DTE", "Strike", "Price", "Intr", "Extr",
                       "Cost/Cov", "ThetaROI", "ExpPnL", "Delta", "Score"]
    summary_rows = []
    for c in candidates:
        cat_display = c["category"]
        if cat_display == "ITM_Near":
            cat_display = "I↑"
        elif cat_display == "ITM_Far":
            cat_display = "I↑↑"
        elif cat_display == "OTM_Far":
            cat_display = "O↓"
        elif cat_display == "OTM_Near":
            cat_display = "O↓↓"

        summary_rows.append([
            c["rank"],
            cat_display,
            c["expiration"][-5:],  # MM-DD
            c["dte"],
            f"${c['strike']:.0f}",
            f"${c['effective_price']:.2f}",
            f"${c['intrinsic']:.2f}",
            f"${c['extrinsic']:.2f}",
            f"{c['cost_to_cover']:.2f}x" if c['cost_to_cover'] is not None else "N/A",
            f"{c['theta_roi']:.2f}x" if c['theta_roi'] is not None else "N/A",
            f"${c['expected_pnl']:.2f}",
            f"{c['delta']:.3f}",
            f"{c['combined_score']:.4f}"
        ])

    lines.append(tabulate(summary_rows, headers=summary_headers, tablefmt="simple",
                          colalign=("right", "center", "center", "right", "right",
                                    "right", "right", "right", "right", "right",
                                    "right", "right", "right")))

    # ITM Near-Dated group table
    itm_near = result.get("itm_near", [])
    if itm_near:
        lines.append(f"\n  ── ITM Near-Dated (DTE ≤ {boundary}, Delta 0.55—0.80) ──")
        itm_headers = ["Rank", "Expiry", "DTE", "Strike", "Price", "Intrinsic",
                       "Extrinsic", "Cost/Cov", "ThetaROI", "ExpPnL", "Score"]
        itm_rows = []
        for c in itm_near:
            itm_rows.append([
                c["rank"], c["expiration"][-5:], c["dte"],
                f"${c['strike']:.0f}", f"${c['effective_price']:.2f}",
                f"${c['intrinsic']:.2f}", f"${c['extrinsic']:.2f}",
                f"{c['cost_to_cover']:.2f}x" if c['cost_to_cover'] is not None else "N/A",
                f"{c['theta_roi']:.2f}x" if c['theta_roi'] is not None else "N/A",
                f"${c['expected_pnl']:.2f}", f"{c['combined_score']:.4f}"
            ])
        lines.append(tabulate(itm_rows, headers=itm_headers, tablefmt="simple"))
    else:
        lines.append(f"\n  (No ITM near-dated candidates)")

    # OTM Further-Out group table
    otm_far = result.get("otm_far", [])
    if otm_far:
        lines.append(f"\n  ── OTM Further-Out (DTE > {boundary}, Delta 0.15—0.45) ──")
        otm_headers = ["Rank", "Expiry", "DTE", "Strike", "Price", "Intrinsic",
                       "Extrinsic", "Cost/Cov", "ThetaROI", "ExpPnL", "Score"]
        otm_rows = []
        for c in otm_far:
            otm_rows.append([
                c["rank"], c["expiration"][-5:], c["dte"],
                f"${c['strike']:.0f}", f"${c['effective_price']:.2f}",
                f"${c['intrinsic']:.2f}", f"${c['extrinsic']:.2f}",
                f"{c['cost_to_cover']:.2f}x" if c['cost_to_cover'] is not None else "N/A",
                f"{c['theta_roi']:.2f}x" if c['theta_roi'] is not None else "N/A",
                f"${c['expected_pnl']:.2f}", f"{c['combined_score']:.4f}"
            ])
        lines.append(tabulate(otm_rows, headers=otm_headers, tablefmt="simple"))
    else:
        lines.append(f"\n  (No OTM further-out candidates)")

    return "\n".join(lines)


# =============================================================================
# Diagonal Spread Planner
# =============================================================================

def _is_friday(date_obj) -> bool:
    """Check if a date is a Friday (weekday 4)."""
    return date_obj.weekday() == 4


def _get_next_fridays(expiration_dates: List[str], cutoff_dte: int) -> List[tuple]:
    """
    From a list of available expiration dates, extract those that are
    Fridays and fall within valid DTE range (<= cutoff_dte from today).
    Returns list of (expiration_str, dte) sorted by DTE ascending.
    """
    today = date.today()
    results = []
    for exp_str in expiration_dates:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if dte <= 0 or dte > cutoff_dte:
            continue
        if _is_friday(exp_date):
            results.append((exp_str, dte))
    results.sort(key=lambda x: x[1])
    return results


def _skip_current_week(friday_exp: str, friday_dte: int) -> bool:
    """
    Determine if the current week's Friday should be skipped.
    Skip if today is Thursday+ AND the Friday has ≤2 DTE.
    """
    today = date.today()
    # Thursday = weekday 3, Friday = 4, Saturday = 5, Sunday = 6
    if today.weekday() >= 3 and friday_dte <= 2:
        return True
    return False


def _find_top_short_legs(
    ticker: str,
    weekly_gex_data: Dict,
    long_strike: float,
    spot_price: float,
    option_type: str = "call",
    cash_equity: float = 100000.0,
    top_n: int = 5
) -> List[Dict]:
    """
    Use ContractSellingAnalyst to find the top-N OTM short
    strikes for a diagonal spread, ranked by highest premium.

    For calls (CC strategy): OTM strikes above both long_strike and spot_price.
    For puts (CSP strategy): OTM strikes below both long_strike and spot_price.

    Returns list of dicts with short leg metrics, sorted by premium descending.
    Each dict contains: strike, Premium_Raw, Trade_ROI, Post_Tax_ROI,
    Cap_Efficiency, Break_Even, Contracts, Total_Premium
    Returns empty list if no suitable strikes found.
    """
    is_call = option_type == "call"
    strategy_type = "CC" if is_call else "CSP"

    analyst = ContractSellingAnalyst(cash_equity=cash_equity)
    result = analyst.scan_from_chain(
        weekly_gex_data,
        strategy_type=strategy_type,
        engine_mode="BOTH"
    )

    if "error" in result:
        return []

    # Collect all analyzed strikes that are OTM relative to both long_strike and spot_price
    candidates = result.get("all_strikes", [])
    valid = [
        c for c in candidates
        if (
            (is_call and c.get("strike", 0) > max(long_strike, spot_price))
            or (not is_call and c.get("strike", 0) < min(long_strike, spot_price))
        )
        and c.get("premium", 0) > 0
    ]
    if not valid:
        return []

    # Sort by raw premium descending and take top N
    valid.sort(key=lambda c: c.get("premium", 0), reverse=True)
    top = valid[:top_n]

    spot = weekly_gex_data.get("spotPrice", 0)
    results = []
    for c in top:
        prem = c.get("premium", 0)
        contracts = c.get("contracts", 0)
        if is_call:
            break_even = spot - prem
        else:
            break_even = c["strike"] - prem
        results.append({
            "strike": c["strike"],
            "Premium_Raw": prem,
            "Trade_ROI": c.get("trade_roi_pct", 0),
            "Post_Tax_ROI": c.get("trade_roi_post_tax_pct", 0),
            "Cap_Efficiency": c.get("capital_efficiency_ratio", 0),
            "Contracts": contracts,
            "Total_Premium": prem * 100 * contracts,
            "Break_Even": break_even
        })

    return results


def plan_diagonal_spreads(efficiency_result: Dict, top_n: int = 5, min_oi_pct: float = 0.0,
                            option_type: str = "call") -> List[Dict]:
    """
    From the efficiency comparison results, build diagonal spread plans
    for the top-N candidates.

    For each long leg candidate:
    1. Find weekly Friday expirations until the long leg's DTE
    2. Apply the Wednesday/2DTE rule
    3. For each active week, find the OTM short strike relative to the long strike
    4. Project rolling premiums (80% retention, 10% weekly decay)
    
    Supports both calls (bullish diagonal) and puts (bearish diagonal).
    """
    is_call = option_type == "call"
    _oi_field = "callOI" if is_call else "putOI"
    _bid_field = "callBid" if is_call else "putBid"
    ticker = efficiency_result.get("ticker", "")
    if not ticker:
        return []

    target_dte = efficiency_result.get("target_dte", 0)

    # Get the full expiration list from chain_data
    chain_data = fetch_gex_all_expirations(ticker)
    all_expirations = chain_data.get("availableExpirations", [])
    if not all_expirations:
        return []

    # Gather candidates: efficiency top-N + ITM_Near + ITM contracts from all expirations
    seen_keys = set()
    candidates = []
    for c in efficiency_result.get("candidates", []):
        key = (c["strike"], c["expiration"])
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(c)
    for c in efficiency_result.get("itm_near", []):
        key = (c["strike"], c["expiration"])
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(c)

    # Also scan all available expirations for ITM contracts not yet in the pool
    # This catches longer-dated expirations that pass the OI filter but
    # were excluded by the efficiency DTE boundary.
    spot = chain_data.get('spot_price', 0) or chain_data.get('spotPrice', 0)
    bulk_added = 0
    for exp_str in all_expirations:
        if any(c["expiration"] == exp_str for c in candidates):
            continue  # already covered by efficiency results
        gex = fetch_gex_single(ticker, expiration=exp_str)
        if "error" in gex:
            continue
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - date.today()).days
        if dte <= 7:
            continue  # skip same-week, no room for rolls
        # Collect all valid ITM strikes for this expiration
        bulk_candidates = []
        for s in gex.get("strikes", []):
            strike = s["strike"]
            # ITM condition is direction-dependent:
            # Calls: ITM = strike < spot (right to buy below market)
            # Puts:  ITM = strike > spot (right to sell above market)
            itm_condition = (strike < spot and is_call) or (strike > spot and not is_call)
            # Also exclude deep ITM (>50% away from spot) and extreme strikes
            too_far = (strike < spot * 0.5) or (strike > spot * 2.0)
            if not itm_condition or too_far:
                continue
            bid = s.get(_bid_field, 0) or 0
            if bid <= 0:
                continue
            key = (strike, exp_str)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bulk_candidates.append({
                "strike": strike,
                "expiration": exp_str,
                "dte": dte,
                "effective_price": bid,
                "distance": abs(strike - spot),
                "category": "ITM_Bulk",
                "combined_score": 0.5
            })
        # Sort by distance to spot (closest to ATM = best leverage), take top 5
        bulk_candidates.sort(key=lambda x: x["distance"])
        for bc in bulk_candidates[:5]:
            del bc["distance"]
            candidates.append(bc)
            bulk_added += 1

    # Keep top_n long leg candidates total (efficiency-ranked + bulk ITM)
    efficient = [c for c in candidates if c.get("category") != "ITM_Bulk"][:top_n]
    bulk = [c for c in candidates if c.get("category") == "ITM_Bulk"]
    candidates = efficient + bulk
    if not candidates:
        return []

    warnings = []

    # Build plans. If no DTE ≥ 21 candidates produce plans, retry with DTE ≥ 14.
    def _build_plans(min_dte):
        results = []
        for long_candidate in candidates:
            if len(results) >= top_n:
                break
            long_strike = long_candidate["strike"]
            long_exp = long_candidate["expiration"]
            long_dte = long_candidate["dte"]
            if long_dte < min_dte:
                continue
            long_cost = long_candidate.get("effective_price", 0)

            # Optional OI filter: prune candidates with low relative OI
            if min_oi_pct > 0:
                gex_oi = fetch_gex_single(ticker, expiration=long_candidate["expiration"])
                if "error" not in gex_oi:
                    all_strikes = gex_oi.get("strikes", [])
                    option_oi = [int(s.get(_oi_field, 0) * 1000) for s in all_strikes if s.get(_oi_field, 0) > 0]
                    strike_oi = int(sum(s.get(_oi_field, 0) for s in all_strikes if s["strike"] == long_strike) * 1000)
                    max_oi_strike = max(option_oi) if option_oi else 0
                    if max_oi_strike > 0 and strike_oi < max_oi_strike * (min_oi_pct / 100.0):
                        continue

            category = long_candidate.get("category", "")
            score = long_candidate.get("combined_score", 0)

            fridays = _get_next_fridays(all_expirations, long_dte)
            if not fridays:
                continue

            active_weeks = []
            for exp_str, dte in fridays:
                if not active_weeks and _skip_current_week(exp_str, dte):
                    continue
                active_weeks.append((exp_str, dte))

            if not active_weeks:
                continue

            cash_equity = sum(LENDERS) if LENDERS else 100_000
            short_legs = []
            # Only process the first weekly expiration — future weeks are handled
            # by running ContractSellingAnalyst separately after this week resolves.
            exp_str, dte = active_weeks[0]
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            gex_data = fetch_gex_single(ticker, expiration=exp_str)
            if "error" not in gex_data and gex_data.get("strikes"):
                strikes = gex_data["strikes"]
                weekly_spot = gex_data.get("spotPrice", 0)
                top_short_legs = _find_top_short_legs(
                    ticker, gex_data, long_strike, weekly_spot, option_type=option_type,
                    cash_equity=cash_equity, top_n=5
                )
                for sl_entry in top_short_legs:
                    short_strike = sl_entry["strike"]
                    # Spread check: short must be OTM relative to long
                    spread_is_valid = (short_strike > long_strike) if is_call else (long_strike > short_strike)
                    spread_width = (short_strike - long_strike) if is_call else (long_strike - short_strike)
                    min_spread = max(5.0, long_strike * 0.02)
                    if not spread_is_valid or spread_width < min_spread:
                        continue
                    gross_premium = sl_entry["Premium_Raw"]
                    buyback_cost = gross_premium * 0.2
                    net_credit = gross_premium - buyback_cost
                    long_close_value = max(0, weekly_spot - long_strike) if is_call else max(0, long_strike - weekly_spot)
                    long_value_at_strike = max(0, short_strike - long_strike) if is_call else max(0, long_strike - short_strike)
                    combined_close = long_close_value + net_credit
                    short_legs.append({
                        "short_expiry": exp_str,
                        "short_strike": short_strike,
                        "gross_premium": round(gross_premium, 2),
                        "buyback_cost": round(buyback_cost, 2),
                        "net_credit": round(net_credit, 2),
                        "long_close_value": round(long_close_value, 2),
                        "long_value_at_strike": round(long_value_at_strike, 2),
                        "combined_close": round(combined_close, 2),
                        "break_even": round(sl_entry["Break_Even"], 2),
                        "contracts": sl_entry["Contracts"]
                    })

            if not short_legs:
                continue

            # short_legs are independent alternatives (pick one), not sequential rolls
            for sl in short_legs:
                sl["cumulative_gross_premium"] = sl["gross_premium"]

            top_net_credit = short_legs[0]["net_credit"]
            plan_spot = gex_data.get("spotPrice", 0)

            results.append({
                "long_strike": long_strike,
                "long_expiry": long_exp,
                "long_dte": long_dte,
                "long_cost": long_cost,
                "category": category,
                "score": score,
                "target_dte": target_dte,
                "total_net_credit": top_net_credit,
                "short_legs": short_legs,
                "num_rolls": len(short_legs),
                "option_type": option_type,
                "spot_price": round(plan_spot, 2)
            })
        return results

    diagonal_plans = _build_plans(21)
    if not diagonal_plans:
        diagonal_plans = _build_plans(14)
        if diagonal_plans:
            print("  ⚠  Illiquid warning: no long legs with 3+ weeks DTE. Using shorter expirations.")

    return diagonal_plans


def format_diagonal_results(diagonal_plans: List[Dict], ticker: str = "", deadline: str = "", target_price: float = None) -> str:
    """Format diagonal spread plans into readable tables."""
    from datetime import datetime
    from tabulate import tabulate

    lines = []

    option_label = diagonal_plans[0].get("option_type", "call") if diagonal_plans else "call"
    direction = "bullish" if option_label == "call" else "bearish"
    target_str = f" | Target: ${target_price:.0f}" if target_price else ""
    spot = diagonal_plans[0].get("spot_price", 0) if diagonal_plans else 0
    spot_str = f" | Spot: ${spot:.2f}" if spot else ""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"\n{'='*75}")
    lines.append(f"  Diagonal Spread Plans")
    lines.append(f"  {ticker.upper() if ticker else ''} — {option_label.capitalize()} ({direction}){target_str}{spot_str} | Deadline: {deadline} | Generated: {now_str}")
    lines.append(f"{'='*75}")

    for plan in diagonal_plans:
        lines.append(f"")
        option_label = plan.get("option_type", "call").capitalize()
        lines.append(f"Long: ${plan['long_strike']:.0f} {option_label} exp {plan['long_expiry']} (DTE {plan['long_dte']}) "
                     f"| Cost: ${plan['long_cost']:.2f} "
                     f"| Cat: {plan['category']} | Score: {plan['score']:.4f}")
        # lines.append(f"  {'─'*60}")

        pa_headers = ["Rank", "Short Exp", "Short Strike",
                      "Short Prem", "−20% Buybk", "= Net Kept",
                      "+ Long Value", "Close (Max, Stop)",
                      "Break-Even", "ROI (Max, Stop)"]
        pa_rows = []
        long_cost = plan.get("long_cost", 0)
        for rank_i, sl in enumerate(plan["short_legs"], 1):
            effective_cost = max(0.01, long_cost - sl['cumulative_gross_premium'])
            total_value_max = sl['combined_close']
            stop_close_value = sl['long_value_at_strike'] + sl['net_credit']
            roi_max = (sl['net_credit'] / effective_cost) * 100
            roi_stop = ((stop_close_value - long_cost) / effective_cost) * 100 if long_cost > 0 else 0
            pa_rows.append([
                rank_i,
                sl["short_expiry"][-5:],
                f"${sl['short_strike']:.0f}",
                f"${sl['gross_premium']:.2f}",
                f"−${sl['buyback_cost']:.2f}",
                f"${sl['net_credit']:.2f}",
                f"${sl['long_close_value']:.2f}",
                f"(${sl['combined_close']:.2f}, ${stop_close_value:.2f})",
                f"${sl['break_even']:.2f}",
                f"({roi_max:.1f}%, {roi_stop:.1f}%)"
            ])
        lines.append(tabulate(pa_rows, headers=pa_headers, tablefmt="simple",
                              colalign=("right", "center", "right",
                                        "right", "right", "right", "right",
                                        "right", "center", "center")))

    return "\n".join(lines)


# =============================================================================
# CLI Scanner (Old — commented out for reference / New Efficiency Scanner active)
# =============================================================================

def run_optimizer_scanner():
    """CLI scanner for the option efficiency comparison (new design)."""

    parser = argparse.ArgumentParser(
        prog="option_efficiency",
        description="Efficiency comparison and diagonal spread planner for call/put options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m core.strategies.option_strike_optimizer --ticker SPY --deadline-date 2026-06-15 --top-n 10
  python -m core.strategies.option_strike_optimizer --ticker SPY --deadline-date 2026-06-15 --target-price 560 --mode diagonal --top-n 5
  python -m core.strategies.option_strike_optimizer --ticker SPY --deadline-date 2026-06-15 --target-price 450 --mode diagonal --option-type put --top-n 5
  python -m core.strategies.option_strike_optimizer --ticker QQQ --deadline-date 2026-07-01 --near-date-cutoff 45"""
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="efficiency",
        choices=["efficiency", "diagonal"],
        help="Mode: 'efficiency' (default, compare cost-to-cover/theta-ROI) or 'diagonal' (build diagonal spreads from top candidates)"
    )

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Single ticker symbol to analyze (e.g., SPY)"
    )
    parser.add_argument(
        "--deadline-date",
        type=str,
        required=True,
        help="Deadline date. Accepted: YYYY-MM-DD, MM/DD/YYYY, YYYYMMDD, 'Jul 1, 2026', 'July 1, 2026', etc."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top candidates to display per category (default: 5)"
    )
    parser.add_argument(
        "--target-price",
        type=float,
        default=None,
        help="Optional price target. Script computes expected move = abs(target - spot). "
             "If omitted, falls back to calculate_expected_move()."
    )
    parser.add_argument(
        "--near-date-cutoff",
        type=int,
        default=30,
        help="DTE threshold separating near-dated from further-out (default: 30)"
    )
    parser.add_argument(
        "--option-type",
        type=str,
        default="call",
        choices=["call", "put"],
        help="Option type: 'call' for bullish diagonal (default) or 'put' for bearish diagonal"
    )

    parser.add_argument(
        "--min-oi-pct",
        type=float,
        default=0.0,
        help="Minimum open interest as %% of max OI in chain for long leg selection (default: 0 = no filter)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output compact JSON for AI consumption instead of formatted tables"
    )

    args = parser.parse_args()

    # Diagonal mode: extend DTE boundary to 90 to capture longer-dated long legs
    if args.mode == "diagonal":
        if args.near_date_cutoff < 90:
            args.near_date_cutoff = 90

    # Run efficiency comparison (always needed for both modes)
    result = compare_efficiency(
        ticker=args.ticker,
        deadline=args.deadline_date,
        top_n=args.top_n,
        target_price=args.target_price,
        near_date_cutoff=args.near_date_cutoff,
        option_type=args.option_type
    )

    if "error" in result:
        if args.json:
            print(json.dumps({"error": result["error"]}, separators=(",", ":")))
        else:
            print(f"\nError: {result['error']}")
        return

    if args.mode == "efficiency":
        if args.json:
            print(_format_efficiency_json(result))
        else:
            print(format_efficiency_results(result))
        return

    # Diagonal mode: build diagonal spreads from top candidates
    diagonal_results = plan_diagonal_spreads(result, top_n=args.top_n, min_oi_pct=args.min_oi_pct,
                                              option_type=args.option_type)

    if args.json:
        print(_format_diagonal_json(diagonal_results, result.get("ticker", ""), args.option_type,
                                      deadline=args.deadline_date, target_price=args.target_price,
                                      spot_price=result.get("spot_price", 0)))
    else:
        if diagonal_results:
            print(format_diagonal_results(diagonal_results, ticker=result.get("ticker", ""), 
                                          deadline=args.deadline_date, target_price=args.target_price))


if __name__ == "__main__":
    run_optimizer_scanner()