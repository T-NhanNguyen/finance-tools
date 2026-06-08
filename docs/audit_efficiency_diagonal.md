# Audit Summary: Efficiency Comparison & Diagonal Spread Planner

**File audited:** `core/strategies/option_strike_optimizer.py` (1942 lines)
**Mode:** Code audit only — no changes made.

---

## Dimension 1: Break-Even Price Emphasis

**Status: ⚠️ PARTIAL GAP**

### `compare_efficiency()` (L820–1230)
The scoring formula is:
```
combined_score = 0.5 * cost_score + 0.5 * theta_score
```
- `cost_score` = inverted normalized cost-to-cover (extrinsic / expected move)
- `theta_score` = normalized theta ROI (expected P&L per theta dollar)
- **No break-even price is computed or considered in this formula.**
- Ranking is purely about extrinsic premium efficiency and theta burn efficiency.

### `plan_diagonal_spreads()` (L1470–1700)
- Break-even IS computed per short leg (`Break_Even = spot - premium` for calls) and displayed in the price action table.
- But there's **no combined break-even** for the full diagonal. `effective_cost = max(0.01, long_cost - cumulative_gross_premium)` is used only for ROI calculation, not displayed or ranked.

**Gap:** A strike with a lower break-even (closer to spot) might score worse than one with artificially low cost-to-cover because the formula doesn't penalize candidates where break-even is far from the underlying.

---

## Dimension 2: OI/Liquidity Filtering for Long Legs

**Status: ❌ FULL GAP**

| Location | Lines | Filter | Default |
|---|---|---|---|
| `compare_efficiency()` OI filter | ~L1040–1055 | Per-expiration OI ≥ 25% of max OI in chain | Always on |
| `plan_diagonal_spreads()` long leg OI filter | ~L1562–1573 | Strike-level OI ≥ `min_oi_pct`% of max OI | **Off (0.0)** |
| `_find_top_short_legs()` | ~L1440–1470 | **None** | N/A |
| `ContractSellingAnalyst` (via scan_from_chain) | — | **None** (no OI/volume filtering) | N/A |

### Key findings:
1. **Short leg has zero OI/liquidity filter** — `_find_top_short_legs` delegates to `ContractSellingAnalyst`, which scores purely by premium, ROI, capital efficiency, and structural density (GEX). No OI or volume check exists.
2. **Long leg OI filter is optional and off by default** — `--min-oi-pct 0.0` means the bulk ITM scan (L1500–1530) adds strikes with near-zero OI. The only requirement is `bid > 0`.
3. For an ITM short-dated long leg, low-OI strikes could have wide bid-ask spreads on both entry and exit.

---

## Dimension 3: Capital Efficiency vs Raw ROI/Premium

**Status: ⚠️ PARTIAL ALIGNMENT**

### Positive aspects:
- `compare_efficiency()` avoids raw premium or highest delta. It uses two capital-efficiency-aware metrics:
  - `cost_to_cover` = extrinsic premium per dollar of expected move (lower = better)
  - `theta_roi` = expected P&L per dollar of theta decay (higher = better)
  - This aligns with "expected move per dollar spent" philosophy.

### Gaps:
- **`_find_top_short_legs()` (L1440–1470):** Sorts short leg candidates by **raw premium descending** — not by ROI, capital efficiency, or any efficiency measure. The `ContractSellingAnalyst` results include `trade_roi_pct`, `capital_efficiency_ratio`, etc., but these aren't used for ranking.
- **`generate_single_leg_candidates()` (L474):** No OI filter — only delta range (0.15–0.70).
- **`generate_debit_spread_candidates()` (L530):** No OI filter — only debit cost ratio (<80% of width).

---

## Dimension 4: Short Call as Yield Overlay

**Status: ⚠️ PARTIAL GAP**

### Positive aspects:
- Architecture correctly separates long leg selection (efficiency-driven) from short leg overlay.
- Short leg metrics include gross premium, 80% buyback cost (20% net retention), break-even, and ROI.
- Long leg is always selected first, then short legs are overlaid — matches "building block + yield overlay" vision.

### Gaps:
- **Only one weekly short leg is modeled** (L1595 comment: "Only process the first weekly expiration — future weeks are handled by running ContractSellingAnalyst separately after this week resolves"). The `total_net_credit` is from one week only, understating lifecycle costs.
- **No multi-week roll projection** — the 10% decay mentioned in your philosophy isn't implemented.
- **`_build_plans()`** tries 21 DTE first, falls back to 14 DTE with warning (L1681–1685). This is reasonable but could be improved.

---

## Other Findings

### Phantom Filter Interaction (L1884)
- Diagonal mode calls `compare_efficiency` with `skip_phantom_filter=False`
- This means strikes past the target price are excluded from comparison
- For a diagonal, the long leg could validly be above target price (for calls) — the phantom filter may silently remove useful candidates

### No Combined Break-Even for Diagonals
- `effective_cost` is used only for ROI calculation, not displayed alongside the plan
- A user sees each short leg's break-even but not the combined "at what underlying price does this diagonal become profitable?"

### Scoring/Ranking Summary Table

| Dimension | Status | Key Finding |
|---|---|---|
| Break-even emphasis | ⚠️ Partial gap | Not in efficiency scoring; per-short-leg only in diagonal, no combined break-even |
| OI/liquidity for long legs | ❌ Full gap | Long leg OI filter optional/off; short leg has zero OI filter at all layers |
| Capital efficiency | ⚠️ Partial align | Efficiency scoring is good; short leg ranking uses raw premium |
| Yield overlay treatment | ⚠️ Partial gap | Architecture is correct; single-week modeling understates lifecycle returns |

---

## Evidence Details

### Functions & Line Ranges
- `compare_efficiency()` — L820–1230 (scoring formula at L1195–1215)
- `plan_diagonal_spreads()` — L1470–1700
- `_find_top_short_legs()` — L1430–1470
- `_skip_current_week()` — L1410–1420
- `_get_next_fridays()` — L1380–1408
- `ContractSellingAnalyst.scan_from_chain()` — called at L1445
- `format_efficiency_results()` — L1300–1420
- `format_diagonal_results()` — L1700–1800
- `run_optimizer_scanner()` — L1800–1942
- Per-expiration OI filter — L1040–1055
- Phantom filter (bid-ask + strike past target) — L1082–1125
- Delta classification — L1136–1142
- Combined score normalization — L1210–1240
- Bulk ITM scan — L1500–1530
- Long leg OI filter (optional) — L1562–1573
- Short leg ranking by raw premium — L1460–1470
- Single-week modeling — L1595 comment
- DTE fallback 21→14 — L1681–1685
- CLI diagonal mode near_date_cutoff override to 90 — L1877
