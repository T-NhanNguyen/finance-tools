Below is a clean way to update your algorithm for **Reg T naked short puts** on **equity / narrow-based index options**.

## Formula

For one short put contract:

$$
\text{Margin per contract} = \max\Big(
P + 0.20S - \text{OTM},\;
P + 0.10K
\Big) \times 100
$$

Where:

- $P$ = option premium per share
- $S$ = current underlying stock price per share
- $K$ = put strike price per share
- $\text{OTM} = \max(S - K, 0)$ for a put
- 100 = standard contract multiplier

This matches the Cboe strategy-based margin rule for short equity puts: **100% of option proceeds plus 20% of underlying value less any out-of-the-money amount, with a minimum of option proceeds plus 10% of the put exercise price**.[^1]

FINRA Rule 4210 allows broker-dealers to impose requirements that are **higher than the minimums**, so your broker may still reject a trade even if this formula says it should fit.[^2][^3]

## Position-level version

For $N$ contracts:

$$
\text{Total Margin} = N \times \text{Margin per contract}
$$

You should then compare that to your available margin equity or buying power after any existing positions, because the broker checks the **post-trade** requirement, not just the standalone trade.[^2][^1]

## Algorithm pseudocode

```text
input:
  S = underlying price
  K = strike price
  P = option premium per share
  N = number of short put contracts
  multiplier = 100

OTM = max(S - K, 0)

req1 = P + 0.20 * S - OTM
req2 = P + 0.10 * K

margin_per_share = max(req1, req2)
margin_per_contract = margin_per_share * multiplier
total_margin = margin_per_contract * N
```


## Recommended implementation notes

- Use **per-share** inputs internally, then multiply by 100 at the end.
- Make sure your algorithm uses the **current underlying price at order entry**, not your cost basis.
- Apply the result as a **required margin increase**, then test whether the account still satisfies broker house rules after the trade.
- If you support multiple broker modes, keep a separate **house-margin override** layer, since brokers can tighten requirements above the Reg T baseline.[^3][^2]


## Example with your BE trade

Using the values you provided earlier:

- $S \approx 133.24$[^4]
- $K = 125$
- $P = 5$

Then:

- $\text{OTM} = 133.24 - 125 = 8.24$
- $P + 0.20S - \text{OTM} = 5 + 26.648 - 8.24 = 23.408$
- $P + 0.10K = 5 + 12.5 = 17.5$

So the margin floor is:

$$
23.408 \times 100 = 2340.80 \text{ per contract}
$$

and for 17 contracts:

$$
2340.80 \times 17 = 39,793.60
$$

That is the **baseline minimum** under the published short-put formula, not necessarily your broker’s actual requirement.[^1][^4]
