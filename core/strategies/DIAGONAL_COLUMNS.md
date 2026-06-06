# Diagonal Spread Planner — Column Reference

Activated with `--mode diagonal` on the existing CLI. Runs the efficiency
comparison first, then builds diagonal spread plans for the top-N candidates.

---

## How It Works

For each long call candidate from the efficiency scan:

1. **Scan weekly expirations**: Find all Friday expiration dates between today
   and the long leg's expiry date.
2. **Apply the Wednesday rule**: If today is Thursday or later AND the current
   week's Friday has ≤2 DTE, skip it and start from the next week.
3. **Select the short leg**: For each active week, find the nearest OTM call
   strike strictly above the long leg's strike. Read its ask price from the
   GEX cache.
4. **Project rolling scenario**:
   - Week 1: gross premium = actual ask from GEX cache
   - Week 2+: gross premium = previous week × 0.9 (10% decay)
   - Buyback cost = 80% of gross (net kept = 20%)
   - Stop rolling if no weekly exists before the long leg expires
5. **Long leg depreciation**: `theta_abs × target_dte` as a flat theta burn.

---

## Price Action Table

One table per long candidate. Rows = weeks in the rolling schedule.

| Column | Description |
|--------|-------------|
| **Week** | Roll number (1 = first short leg) |
| **Short Exp** | Weekly Friday expiration date (MM-DD) |
| **DTE** | Days to expiration for this short leg |
| **Short Strike** | OTM call strike above the long leg strike |
| **Gross Premium** | Short leg ask price (actual for week 1, 10% decay for subsequent) |
| **Net Credit (20%)** | Gross × 20% (after 80% buyback cost) |
| **Long Close Value** | Long leg's intrinsic value if closed at this week's expiry: `max(0, spot_at_expiry − long_strike)`. Uses the weekly's GEX spot price. |
| **Combined Close** | `long_close_value + gross_premium`. What you'd net if closing the whole diagonal at this expiry. |

---

## Summary Section

Below each price action table:

| Metric | Description |
|--------|-------------|
| **Total Net Credit** | Sum of net credits from all short rolls |
| **Long Theta Depreciation** | `theta_abs × target_dte`. Flat projection of time decay cost on the long leg over the full holding period. Theta is from Black-Scholes (finite difference estimate). |
| **Net Position** | `total_net_credit − theta_burn`. Positive = rolling income covers long decay. Negative = theta burn eats the premium. |
| **ROI vs Long Cost** | `(net_position / long_cost) × 100`. Return on the long leg's purchase cost. |

---

## Wednesday / 2-Day Rule

```
if today.weekday() >= 3 (Thursday) AND weekly_friday_dte <= 2:
    skip this week → start from next Friday
```

| Today | This Friday | DTE | Use it? |
|-------|-------------|-----|---------|
| Monday | Friday | 4 | ✅ Yes |
| Tuesday | Friday | 3 | ✅ Yes |
| Wednesday | Friday | 2 | ✅ Yes (Wednesday is the last day) |
| Thursday | Friday | 1 | ❌ Skipped (past Wednesday + ≤2 DTE) |
| Friday | Friday | 0 | ❌ Skipped |

---

## Assumptions & Limitations

- **Short leg always OTM**: The short strike is the nearest call strike above
  the long strike. If no such strike exists in the weekly chain, the week is
  skipped.
- **10% decay per week**: Subsequent week premiums are estimated at 90% of
  the previous week's. This is a conservative assumption — actual premiums
  depend on realized volatility.
- **80% buyback cost**: Flat assumption that closing the short leg costs 80%
  of the premium collected. This accounts for bid-ask friction and adverse
  price movement.
- **Long close value = intrinsic only**: At each short expiry, the long leg's
  value is approximated as `max(0, spot − strike)`. This is accurate for ITM
  calls where intrinsic dominates, but understates value for deep OTM longs.
- **No monthly fallback**: If no Friday weekly exists before the long leg
  expires, rolling stops. No monthly option is substituted.
- **Theta is a BS estimate**: Computed via Black-Scholes finite difference
  (price at T vs T−1 day), not a real market theta. Use as directional guide.
- **GEX cache dependency**: Short leg prices come from the GEX cache. During
  off-hours, cached prices from the last market close are used. If the cache
  has never been populated, first run will fetch from yfinance.
