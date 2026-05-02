#!/usr/bin/env python3
"""
Option Strike Optimizer - CLI Scanner for finding optimal option strikes.
Usage: python -m core.strategies.option_strike_optimizer TICKER --scenario SCENARIO
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import pandas as pd
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

from core.data.get_options_data import getOptionChain
from core.data.bulk_data_loader import fetch_gex_all_expirations, fetch_gex_single
from core.data.get_gex_data import fetch_gex_structured
from core.analysis.calculate_gamma_delta import calculateDelta, calculateGamma, calculateBlackScholesPrice
from core.analysis.calculate_risk_free_rate import getRiskFreeRate
from core.strategies.strategy_config import SCENARIOS, ScenarioConfig
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
        expected_move = calculate_expected_move(chain_data, first_exp, scenario_config)
        
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
            
        from core.data.get_stock_price import getHistoricalPrices, PricePeriod, PriceInterval
        
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


# =============================================================================
# Import existing functions from option_strike_optimizer.py
# =============================================================================

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
                           scenario: ScenarioConfig) -> Dict:
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
        
    return {
        "expected_move": float(expected_move),
        "upper_expected": float(upper_expected),
        "lower_expected": float(lower_expected),
        "atm_strike": float(atm_strike),
        "gex_levels": gex_levels
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
        
    for exp in filtered_expirations:
        exp_date = exp["expiration"]
        dte = exp.get("daysToExpiration") or exp.get("dte") or 30
        if dte is None or dte <= 0: continue
        
        chain = getOptionChain(chain_data.get("ticker", ""), expiration=exp_date)
        options = chain.get("calls")
        if options is None or options.empty: continue
        
        for _, row in options.iterrows():
            long_strike = row["strike"]
            
            # ITM Filter: Avoid deep ITM unless LEAPS
            is_leaps = dte >= LEAPS_DTE_THRESHOLD
            if is_leaps:
                # Even for LEAPS, avoid extreme deep ITM
                if long_strike < spot_price * 0.70: continue
            else:
                if long_strike < spot_price * 0.95: continue
            
            short_strike = long_strike + spread_width
            short_row = options[options["strike"] == short_strike]
            if short_row.empty: continue
            
            long_p = (row.get("bid", 0) + row.get("ask", 0)) / 2
            short_p = (short_row.iloc[0].get("bid", 0) + short_row.iloc[0].get("ask", 0)) / 2
            debit = long_p - short_p
            
            # Risk/Reward Filter: Debit shouldn't be > 80% of width for spreads
            if debit <= 0 or debit > spread_width * 0.80: continue
            
            candidates.append({
                "option_type": "debit_spread",
                "long_strike": long_strike,
                "short_strike": short_strike,
                "expiration": exp_date,
                "dte": dte,
                "debit": debit,
                "long_iv": row.get("impliedVolatility", 0.20),
                "long_delta": row.get("delta", 0),
                "short_delta": short_row.iloc[0].get("delta", 0),
                "long_gamma": row.get("gamma", 0),
                "short_gamma": short_row.iloc[0].get("gamma", 0),
                "long_theta": row.get("theta", 0),
                "short_theta": short_row.iloc[0].get("theta", 0),
                "long_vega": row.get("vega", 0),
                "short_vega": short_row.iloc[0].get("vega", 0),
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
    
    # Probability of profit (Price > Breakeven at expiration)
    break_even = candidate["long_strike"] + candidate["debit"]
    d2 = (np.log(spot_price / break_even) + (- 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    prob_profit = norm.cdf(d2) if scenario.direction == "bullish" else 1 - norm.cdf(d2)
    
    # Payoff at target
    if scenario.direction == "bullish":
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
    max_profit = candidate["short_strike"] - candidate["long_strike"] - candidate["debit"]
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
    
    return candidate


# =============================================================================
# CLI Scanner
# =============================================================================

def run_optimizer_scanner():
    """CLI scanner for the option strike optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        prog="option_strikes",
        description="Find optimal option trades using Black-Scholes, GEX, and market structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Scenarios: bullish_3month (90d), earnings (14d), ta_breakout (var)"""
    )
    
    parser.add_argument(
        "tickers",
        nargs="*",
        help="List of ticker symbols to analyze (e.g., SPY QQQ MSFT)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="bullish_3month",
        choices=list(SCENARIOS.keys()),
        help="Trading scenario (default: bullish_3month)"
    )
    parser.add_argument(
        "--strike-type",
        type=str,
        default="single_leg",
        choices=["single_leg", "debit_spread"],
        help="Strike type: single_leg (single option) or debit_spread (two-leg spread)"
    )
    parser.add_argument(
        "--option-type",
        type=str,
        default="call",
        choices=["call", "put"],
        help="Option type for single leg trades (default: call)"
    )
    parser.add_argument(
        "--spread-width",
        type=float,
        default=5.0,
        help="Width between strikes for debit spreads (default: 5.0)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top candidates to display per ticker (default: 5)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for programmatic use)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed metrics for each candidate (delta, gamma, theta, vega)"
    )
    parser.add_argument(
        "--help-scenarios",
        action="store_true",
        help="Show detailed scenario descriptions and use cases"
    )
    parser.add_argument(
        "--help-examples",
        action="store_true",
        help="Show common use cases and examples"
    )
    
    args = parser.parse_args()
    
    if not args.tickers and not (args.help_scenarios or args.help_examples):
        parser.print_help()
        return
        
    if args.help_scenarios or args.help_examples:
        print("\n--- Scenarios ---")
        for k, v in SCENARIOS.items(): print(f"{k:15}: {v.direction} outlook ({v.time_horizon_days} days)")
        print("\n--- Examples ---")
        print("  python core/strategies/option_strike_optimizer.py SPY --strike-type single_leg")
        print("  python core/strategies/option_strike_optimizer.py MSFT --scenario earnings --strike-type debit_spread")
        return
    
    optimizer = OptionStrikeOptimizer()
    results = optimizer.scan_multiple(
        args.tickers,
        scenario=args.scenario,
        strike_type=args.strike_type,
        option_type=args.option_type,
        spread_width=args.spread_width,
        top_n=args.top_n
    )
    
    if args.json:
        import json
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, pd.Timestamp): return obj.isoformat()
                return super().default(obj)
        print(json.dumps(results, indent=2, cls=CustomEncoder))
        return

    # Print results
    for ticker, result in results.items():
        print(f"\n--- {ticker.upper()} | {args.scenario} | {args.strike_type} | Spot: ${result.get('spot_price', 0):.2f} ---")
        
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
            
        sr = result.get('support_resistance', {})
        em = result.get('expected_move', {})
        gl = em.get('gex_levels', {})
        
        print(f"S/R: {sr.get('support', [])[:2]} / {sr.get('resistance', [])[:2]} | Range: ${em.get('lower_expected', 0):.0f}-${em.get('upper_expected', 0):.0f}")
        print(f"Walls: C: ${gl.get('call_wall')} | P: ${gl.get('put_wall')} | MP: ${gl.get('max_pain')}")
        
        candidates = result.get("top_candidates", [])
        if not candidates:
            print("\nNo suitable candidates found within scenario constraints.")
            continue
            
        print(f"\nTop {len(candidates)} Candidates:")
        
        table_data = []
        if args.strike_type == "single_leg":
            headers = ["Rank", "Score", "Expiry", "DTE", "Strike", "Price", "Delta", "Prob Profit", "Exp P&L"]
            for c in candidates:
                m = c['metrics']
                table_data.append([
                    c['rank'],
                    f"{c['score']:.4f}",
                    c['expiration'],
                    c['dte'],
                    f"${c['strike']}",
                    f"${c['price']:.2f}",
                    f"{m.get('delta', 0):.4f}",
                    f"{m.get('probability_of_profit', 0):.1%}",
                    f"${m.get('expected_pnl', 0):.2f}"
                ])
        else:
            headers = ["Rank", "Score", "Expiry", "DTE", "Spread", "Debit", "Max Profit", "Prob Profit", "Exp P&L"]
            for c in candidates:
                m = c['metrics']
                table_data.append([
                    c['rank'],
                    f"{c['score']:.4f}",
                    c['expiration'],
                    c['dte'],
                    f"${m.get('long_strike')} / ${m.get('short_strike')}",
                    f"${m.get('net_debit'):.2f}",
                    f"${m.get('max_profit'):.2f}",
                    f"{m.get('probability_of_profit', 0):.1%}",
                    f"${m.get('expected_pnl', 0):.2f}"
                ])
        
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
        print()


if __name__ == "__main__":
    run_optimizer_scanner()