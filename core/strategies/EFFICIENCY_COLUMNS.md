# Efficiency Comparison — Column Reference

Located in `option_strike_optimizer.py`, the `compare_efficiency()` function
scans all liquid option expirations and ranks call contracts by how efficiently
they capture a given expected move.

---

## Category Column

Appears in the **Ranked Summary** table. Each candidate is classified by
delta range and time-to-expiration relative to the `--dte-boundary` (default 30).

| Symbol | Full Name | Delta | DTE | Description |
|--------|-----------|-------|-----|-------------|
| **I↑** | ITM_Near | 0.55–0.80 | ≤ boundary | **In-the-money, near-dated.** Low extrinsic premium, high intrinsic. Usually the lower cost-to-cover group. |
| **O↓** | OTM_Far | 0.15–0.45 | > boundary | **Out-of-the-money, further-out.** High time premium, zero intrinsic. The contrasting group. |
| **I↑↑** | ITM_Far | 0.55–0.80 | > boundary | ITM delta but longer-dated. Cross-category — uncommon. |
| **O↓↓** | OTM_Near | 0.15–0.45 | ≤ boundary | OTM delta but short-dated. Cross-category — uncommon. |

**Arrow convention:**
- **I / O** → In-the-money (higher delta ↑) / Out-of-the-money (lower delta ↓)
- **One arrow** → near-dated (within DTE boundary)
- **Two arrows** → further-out (beyond DTE boundary)

---

## Ranked Summary Columns

The top table shows every candidate sorted by combined score. Columns:

| Column | Description |
|--------|-------------|
| **Rank** | Position by combined score (1 = best). Lower cost-to-cover + higher theta ROI = higher rank. |
| **Category** | See table above. |
| **Expiry** | Option expiration date (MM-DD). |
| **DTE** | Days-to-expiration from today. |
| **Strike** | Option strike price. |
| **Price** | Effective price used for calculations. Priority: ask (market hours) > lastPrice (off-hours) > bid. |
| **Intr** | **Intrinsic value.** For calls: `max(0, spot - strike)`. The amount of option price that would be realized if exercised immediately. |
| **Extr** | **Extrinsic (time) value.** `effective_price - intrinsic`. The premium paid beyond intrinsic — decays to zero at expiration. |
| **Cost/Cov** | **Cost-to-cover ratio.** `extrinsic / expected_move`. Lower is better. Measures how much of the expected move you pay for in time premium. A value of 0.10× means you pay 10¢ in time premium per $1 of expected move. |
| **ThetaROI** | **Theta-adjusted ROI.** `expected_pnl / (theta_abs × target_dte)`. How many times the expected P&L covers the total theta decay over the holding period. Positive means your expected profit exceeds time decay. Negative means theta burns through your expected gain. |
| **ExpPnL** | **Expected P&L.** `payoff_at_target - effective_price`. Dollar profit or loss if the price target is reached, assuming you hold to expiration. |
| **Delta** | Option delta from Black-Scholes (fallback IV=0.20 when yfinance IV is unreasonably low). |
| **Score** | **Combined efficiency score** (0–1). `0.5 × cost_score + 0.5 × theta_score`, where each metric is normalized across all candidates. Higher = more efficient. |

---

## Group Table Columns (ITM Near-Dated / OTM Further-Out)

The two grouped tables below the summary show a subset of columns:

| Column | Description |
|--------|-------------|
| **Rank** | Position from the full ranked summary (not re-ranked within the group). |
| **Expiry** | Option expiration date (MM-DD). |
| **DTE** | Days-to-expiration. |
| **Strike** | Option strike price. |
| **Price** | Effective price (see above). |
| **Intrinsic** | Intrinsic value (`max(0, spot - strike)`). |
| **Extrinsic** | Time value (`effective_price - intrinsic`). |
| **Cost/Cov** | Cost-to-cover ratio (see above). |
| **ThetaROI** | Theta-adjusted ROI (see above). |
| **ExpPnL** | Expected P&L at price target. |
| **Score** | Combined efficiency score. |

---

## Metrics Detail

### Expected Move Source

All efficiency metrics are computed from the **same** expected move — they
never mix sources.

| `--target-price` provided? | `dollar_expected_move` | `payoff_at_target` |
|----------------------------|------------------------|---------------------|
| Yes | `abs(target_price − spot_price)` | `max(0, target_price − strike)` |
| No (fallback) | `calculate_expected_move()` (ATM straddle or IV-based) | `max(0, spot_price + expected_move − strike)` |

When you provide a **price target**, both Cost-to-Cover and Theta ROI measure
efficiency relative to *your* predicted move. When omitted, they measure
efficiency relative to the **market's implied move** (via ATM straddle).

### Cost-to-Cover

```
cost_to_cover = (effective_price - intrinsic) / dollar_expected_move
```

- Dollar expected move is either `abs(target_price − spot)` or the market-implied
  ATM straddle / IV-based calculation.
- **ITM near-dated** typically has low cost-to-cover (0.05–0.20×) because
  most of the price is intrinsic, leaving little extrinsic premium.
- **OTM further-out** typically has higher cost-to-cover (0.30–2.0×) because
  the full price is extrinsic (zero intrinsic) and farther expirations carry more
  time premium.

### Theta-Adjusted ROI

```
theta_roi = expected_pnl / (theta_abs × target_dte)
```

- `expected_pnl = payoff_at_target − effective_price` — uses the same price
  target or implied move as Cost-to-Cover (they never split).
- `theta_abs` is estimated via Black-Scholes finite difference (price at T vs T−1 day).
- `target_dte` = days from today to the `--target_date` deadline.
- A value of **2.0×** means the expected profit covers 2× the time decay cost
  over the holding period.
- A **negative** value means time decay exceeds expected profit — a losing
  trade regardless of direction.

### Combined Score

```
cost_score = inverted and normalized cost_to_cover (0–1, lower cost = higher score)
theta_score = normalized theta_roi (0–1, higher ROI = higher score)
combined_score = 0.5 × cost_score + 0.5 × theta_score
```

Both metrics are min-max normalized across all candidates before combining,
so the score is relative to the current set — not an absolute rating.

---

## Edge Case Notes

- **`N/A` in Cost/Cov or ThetaROI**: The metric could not be computed
  (zero expected move, zero theta, or division by near-zero).
- **Empty group tables**: No candidates matched the delta + DTE criteria
  for that category. The category is printed with "(No ... candidates)" instead.
- **Price shown as lastPrice**: During off-hours when bid/ask are 0, the
  last traded price is used as a proxy. This data comes from yfinance's
  previous close — it may not reflect after-hours movement.
- **`--target-price` below spot**: Warning is emitted. Expected P&L will be
  negative for call options since the target is below current price.
