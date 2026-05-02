#!/usr/bin/env python3
"""
Test script for OptionStrikeOptimizer to debug NaN values in expected P&L
"""

import warnings
warnings.filterwarnings('ignore')

from core.strategies import OptionStrikeOptimizer

def test_single_leg():
    """Test single leg option analysis"""
    optimizer = OptionStrikeOptimizer()
    
    print("=== Testing Single-Leg Call Analysis ===\n")
    
    # Test on SPY
    result = optimizer.analyze_strike(
        'SPY', 
        scenario='bullish_3month', 
        strike_type='single_leg', 
        option_type='call'
    )
    
    print(f"Result keys: {result.keys()}")
    print(f"Number of candidates: {len(result.get('top_candidates', []))}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print('\n=== Top 5 Candidates ===')
    for i, candidate in enumerate(result.get('top_candidates', [])[:5], 1):
        print(f"\nRank {candidate['rank']}: {candidate['option_type'].upper()} at strike {candidate['strike']}")
        print(f"  Score: {candidate['score']}")
        print(f"  Expiration: {candidate['expiration']}")
        print(f"  Delta: {candidate['metrics'].get('delta', 'N/A')}")
        print(f"  Gamma: {candidate['metrics'].get('gamma', 'N/A')}")
        print(f"  Probability of Profit: {candidate['metrics'].get('probability_of_profit', 'N/A')}")
        print(f"  Expected P&L: {candidate['metrics'].get('expected_pnl', 'N/A')}")
        print(f"  Payoff at Target: {candidate['metrics'].get('payoff_at_target', 'N/A')}")

def test_debit_spread():
    """Test debit spread analysis"""
    optimizer = OptionStrikeOptimizer()
    
    print("\n\n=== Testing Debit Spread Analysis ===\n")
    
    # Test on SPY
    result = optimizer.analyze_strike(
        'SPY', 
        scenario='bullish_3month', 
        strike_type='debit_spread', 
        option_type='call'
    )
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Number of candidates: {len(result.get('top_candidates', []))}")
    
    if result.get('top_candidates'):
        candidate = result['top_candidates'][0]
        print(f"\nTop Candidate:")
        print(f"  Score: {candidate['score']}")
        print(f"  Expiration: {candidate['expiration']}")
        print(f"  Long strike: {candidate['metrics'].get('long_strike', 'N/A')}")
        print(f"  Short strike: {candidate['metrics'].get('short_strike', 'N/A')}")
        print(f"  Net debit: {candidate['metrics'].get('net_debit', 'N/A')}")
        print(f"  Expected P&L: {candidate['metrics'].get('expected_pnl', 'N/A')}")
    else:
        print("No candidates found!")

def test_scan_tickers():
    """Test multi-ticker scanning"""
    optimizer = OptionStrikeOptimizer()
    
    print("\n\n=== Testing Multi-Ticker Analysis ===\n")
    
    # Analyze SPY, QQQ
    tickers = ['SPY', 'QQQ']
    
    for ticker in tickers:
        print(f"\n=== Analyzing {ticker} ===")
        try:
            result = optimizer.analyze_strike(
                ticker, 
                scenario='bullish_3month', 
                strike_type='single_leg', 
                option_type='call'
            )
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                candidates = result.get('top_candidates', [])
                print(f"Found {len(candidates)} candidates")
                if candidates:
                    top = candidates[0]
                    print(f"  Top candidate: {top['option_type']} at {top['strike']}, score={top['score']}")
                    print(f"  Expiration: {top['expiration']}")
                    print(f"  Expected P&L: {top['metrics'].get('expected_pnl', 'N/A')}")
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == '__main__':
    print("OptionStrikeOptimizer Test Script")
    print("=" * 50)
    
    test_single_leg()
    test_debit_spread()
    test_scan_tickers()
    
    print("\n\n=== Test Complete ===")