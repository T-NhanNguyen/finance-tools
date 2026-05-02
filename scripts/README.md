# Option Strike Optimizer Test Script

## Location
`/scripts/test_optimizer.py`

## Usage

```bash
# Run the test script
docker-compose run --rm finance-tools python /app/scripts/test_optimizer.py
```

## What It Tests

1. **Single-Leg Call Analysis**: Tests the `analyze_strike` method with:
   - Ticker: SPY
   - Scenario: bullish_3month
   - Strike type: single_leg
   - Option type: call

2. **Debit Spread Analysis**: Tests the `analyze_strike` method with:
   - Ticker: SPY
   - Scenario: bullish_3month
   - Strike type: debit_spread
   - Option type: call

3. **Multi-Ticker Analysis**: Tests individual ticker analysis for:
   - SPY
   - QQQ

## Output

The script outputs ranked candidates with:
- Rank
- Option type and strike price
- Score
- Expiration date
- Delta, Gamma, Probability of Profit
- **Expected P&L** (was previously NaN, now working)
- Payoff at Target

## Known Issues

### Debit Spread Net Debit Missing
The `net_debit` field shows "N/A" because it's calculated differently for debit spreads. This is a UI issue, not a calculation issue.

## Development

To add new test cases:
1. Add a new test function to `test_optimizer.py`
2. Call it from the `if __name__ == '__main__':` block
3. Test with various tickers and scenarios