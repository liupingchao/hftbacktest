# Cross-Exchange Maker Shortfall Plan

Date: 2026-07-16

## 2026-07-28 Implementation Status

This status table reconciles the original ordered plan with accepted task and
QA evidence. It does not change the original sequencing or authorize live
work.

| Item | Current status | Accepted boundary |
| --- | --- | --- |
| P0 Fill source / liquidity role | Partial | Source, attribution, reconciliation, and fail-closed role contracts exist; `0726T068` produced zero fills, so no role-known row exists. |
| P1 Production-equivalent public shadow | Complete | `0625T004`/`0625T005` passed and deterministic public shadow is established. |
| P2 Price taxonomy | Complete | `0718T015` accepted typed pricing config, price fields, hashes, and quote eligibility semantics. |
| P3 Basis regression | Complete for public shadow | `0722T061` accepted the candidate and `0722T063` accepted strict shared-kernel shadow wiring; no live/economics claim. |
| P4 Inventory-aware reservation | Structural completion | `0718T016` passed; inventory skew remains disabled pending real lifecycle evidence. |
| P5 Fill/adverse buckets | Partial | Public/proxy/censored evidence exists, but role-known fill-conditioned calibration is absent. |
| P6 Dynamic spread / intensity | Mechanism completion | Estimator, bounded activation, public multi-distance seed, and strict seeded path passed; fill-calibrated economics remain absent. |
| P7 Isolated policy candidates | Substantially complete | Dynamic spread, fill feedback, multi-level, basis, and seeded paths were isolated behind explicit gates; not all are live-enabled. |
| P8 Fee/PnL calibration | Blocked | Requires confirmed maker/taker fill, fee records, and inventory transition. |
| P9 Controlled tiny-live | Executed but blocked | `0726T068` completed three bounded windows and passed mechanism/safety evidence, but zero fills blocked role/economics acceptance. |

The remaining shortfall is narrow but material: acquire an independently
accepted role-known fill and reconcile its fee, inventory, markout, and PnL
without weakening the accepted source/account/lifecycle/terminal gates.

## Purpose

This plan synthesizes the current repository state against two maker-market-making principles:

1. Strategy structure:
   - market data
   - alpha / bucket features
   - forecast mid / fair value
   - quote placement, spread, inventory, fill rate
   - order generation and ID/lifecycle management
   - risk, kill-switch, monitoring
2. Price taxonomy:
   - mid / last
   - micro / fair
   - forecast mid
   - reservation price
   - AS-style inventory skew

The current repository already has strong evidence infrastructure, live/replay audit gates, and a QA-controlled workflow. The main shortfall is not the absence of cross-exchange data or signal research. The shortfall is that accepted signal and evidence artifacts have not yet been promoted into a role-aware, inventory-aware, fill-calibrated cross-exchange maker quote policy.

This document is a controller-level plan only. It does not authorize live retry, parameter changes, quote-envelope changes, fee/PnL calibration, maker viability claims, T012, promotion, or final MVP pass.

## Current Baseline

### Strong Areas

| Area | Current status |
| --- | --- |
| Dual venue public data | Binance lead / Hyperliquid lag public top5/L2 samples exist, with as-of joins, source age, basis context, and future labels. |
| Signal acceptance | `0625T003` accepted `binance_lead_composite` for public shadow: Binance top5 imbalance, Binance microprice-minus-mid, and Binance short mid move. |
| Shared kernel | `0625T004` accepted a pure signal/fair-mid/quote-intent kernel for the next production-equivalent public shadow task. |
| Single-venue price stack | Binance tick MM has `mid -> fair -> reservation -> target_bid/target_ask` with inventory and volatility terms. |
| Evidence discipline | Recent 0716 tasks repaired fill attribution, liquidity-role evidence contract, and quote-policy prework while keeping unsupported claims blocked. |
| Safety / audit | Live safety, shutdown proof, audit replay, action-path parity, and QA reports are strong relative to strategy maturity. |

### Main Shortfalls

| Shortfall | Why it matters |
| --- | --- |
| Basis is not the accepted main alpha | `basis_mid_ticks` exists but is currently context/conditioning, not the frozen production alpha or a regression coefficient in quote policy. |
| Forecast mid is not production-calibrated | The T004 kernel maps signal z-score to expected ticks, but the mapping is still fixture/infrastructure evidence, not a live-calibrated forecast model. |
| Cross-exchange reservation price lacks inventory | The shared kernel has forecast/fair-mid and touch quote intent, but no `reservation = forecast_mid - lambda * position` path. |
| No GLFT/intensity spread | There is no accepted trade-arrival/depth-intensity fit or dynamic spread model based on fill probability. |
| Fill-rate evidence is thin | 0716T002 has `5` submitted attempts, `3` post-only rejects, and `2` corrected fills, but maker/taker role and exact fill lifecycle remain incomplete. |
| Fee/PnL remains blocked | Unknown liquidity role and incomplete exchange-native fill lifecycle block fee/rebate and realized PnL calibration. |
| Quote policy is still prework | 0716T004 produced design candidates only: adverse-flow suppression, post-only reject drift precheck, and fill source-path capture. |

## Ordered Plan

### P0. Finish Fill Source And Liquidity-Role Evidence

Priority: highest

Why first:

- Fee/PnL, fill-rate calibration, maker viability, and quote policy promotion all depend on knowing whether fills are maker/taker and when/why they happened.
- 0716T001 and 0716T003 repaired future artifact contracts, but the next controlled evidence package still needs to prove the repaired source path works.

Scope:

- Use the 0716T003 liquidity-role contract in the next controlled evidence design/preflight.
- Require `fill_liquidity_role_evidence.csv`, `user_fills_pullback_audit.json`, exact fill timestamp fields, and attempt-keyed lifecycle joins.
- Preserve strict boundaries: no threshold/quote-envelope/order-size/max-submission expansion.

Deliverables:

- Controlled evidence preflight design.
- Artifact schema checklist for maker/taker role, fill timestamp, oid/client-id linkage, fee fields, and lifecycle interval.
- Acceptance gate that blocks fee/PnL if any fill has `unknown_liquidity_role`.

Verification:

- Focused artifact parser tests.
- Existing watcher/fill attribution regression tests.
- Boundary manifest proving no strategy behavior or live envelope change.

Exit criteria:

- Future controlled evidence can distinguish `confirmed_maker`, `confirmed_taker`, and `unknown_liquidity_role` without relying on external manual trade-history reconciliation.

### P1. Run Production-Equivalent Public Shadow With The T004 Kernel

Priority: highest

Why second:

- The accepted `binance_lead_composite` signal and shared kernel are not yet exercised in the production-equivalent public shadow path.
- This is the clean bridge from research alpha to a real-time decision path without placing orders.

Scope:

- Implement or dispatch the planned T005-style public-only shadow using `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`.
- Consume the T003 contract and explicit normalization stats.
- Preserve warning bucket visibility, especially source-age and basis buckets with negative adjusted proxy.

Deliverables:

- Multi-window public shadow artifacts:
  - signal components
  - fair-mid / forecast-mid
  - side
  - quote intent
  - edge
  - block reason
  - basis/source-age warning buckets
  - counterfactual markout
- Would-submit sample count and adjusted edge summary.

Verification:

- No private/order/cancel endpoints.
- No live orders.
- Deterministic replay of shadow decisions.
- Public-only boundary manifest.

Exit criteria:

- Would-submit count is large enough for bucket analysis.
- Counterfactual adjusted edge is not systematically negative.
- Source-age / basis warning buckets are explicit, not hidden.

### P2. Define The Cross-Exchange Price Taxonomy Contract

Priority: high

Why third:

- The repo currently has price concepts split across Binance tick MM and cross-exchange kernel. A formal contract prevents future tasks from confusing fair, forecast mid, edge, quote, and reservation price.

Scope:

- Create a design contract for cross-exchange price fields:
  - `hl_mid_px`
  - `hl_microprice_px`
  - `basis_mid_ticks`
  - `signal_score`
  - `forecast_mid_px`
  - `fair_mid_px`
  - `reservation_px`
  - `quote_bid_px`
  - `quote_ask_px`
  - `edge_ticks`
- Explicitly state which fields are decision inputs, diagnostics, or future labels.

Recommended definitions:

```text
hl_mid_px = (hl_best_bid + hl_best_ask) / 2
hl_microprice_px = weighted BBO or top5 microprice
signal_score = accepted Binance lead composite z-score
forecast_mid_px = hl_mid_px + beta_signal * signal_score * tick_size
basis_context_ticks = (binance_mid_px - hl_mid_px) / tick_size
fair_mid_px = forecast_mid_px + optional_context_adjustment
reservation_px = fair_mid_px - lambda_inventory * position
```

Important rule:

- `basis_context_ticks` must remain context-only unless a future formal task proves and accepts a basis regression coefficient.

Deliverables:

- Price taxonomy markdown contract.
- Audit schema delta proposal.
- Synthetic fixtures covering missing/stale/invalid fields.

Verification:

- Schema validation.
- No strategy behavior change.
- No threshold or parameter tuning.

Exit criteria:

- Future tasks can say exactly whether they are changing `forecast_mid`, `fair_mid`, `reservation`, or quote placement.

### P3. Evaluate Basis Regression As A Separate Candidate Alpha

Priority: high

Why fourth:

- Expert guidance specifically recommends cross-exchange basis regression.
- The repo already has basis context, but it is not accepted as a main alpha. It deserves a clean, isolated test rather than being smuggled into the current kernel.

Scope:

- Run offline/public-only basis regression research on accepted synchronized windows.
- Candidate models should stay simple and decision-time clean:
  - `future_hl_mid_move_ticks ~ basis_mid_ticks`
  - `future_hl_mid_move_ticks ~ basis_mid_ticks + binance_lead_composite`
  - regime-conditioned variants by source age, spread, volatility, and HL liquidity state.
- Use leave-one-window-out or stricter split.
- Compare against the accepted `binance_lead_composite` baseline.

Deliverables:

- Regression coefficient table.
- Held-out performance table.
- Stability by basis/source-age/liquidity bucket.
- Model comparison:
  - baseline composite
  - pure basis regression
  - composite plus basis
- Recommendation:
  - `accept_basis_regression_for_shadow`
  - `basis_context_only_keep`
  - `reject_basis_alpha`
  - `needs_more_samples`

Verification:

- No future leakage.
- No same-window coefficient backfill.
- Effective-horizon row condition preserved.
- Negative bucket warnings surfaced.

Exit criteria:

- Basis is either formally upgraded from context to model input, or explicitly kept out of quote policy.

### P4. Add Inventory-Aware Reservation Price To The Cross-Exchange Kernel

Priority: high

Why fifth:

- Current cross-exchange kernel can output forecast/fair-mid and touch quote intent, but it does not include inventory skew.
- Expert guidance centers quoting around `reservation = fmp - lambda * q`.

Prerequisite:

- P1 public shadow passes.
- P2 price taxonomy is accepted.
- P3 either accepts basis as model input or confirms it remains context-only.

Scope:

- Extend the shared kernel in design/offline mode to compute:

```text
reservation_px = round_to_tick(fair_mid_px - lambda_inventory * position)
```

- Add quote intent around reservation:
  - buy quote no more aggressive than current touch unless separately authorized.
  - sell quote no more aggressive than current touch unless separately authorized.
- Separate add-side and reduce-side behavior.

Deliverables:

- Kernel fixture update.
- Inventory state fixture cases:
  - flat
  - long add-side block/skew
  - long reduce-side allowed
  - short add-side block/skew
  - short reduce-side allowed
- Audit fields for inventory skew and reservation.

Verification:

- Deterministic fixtures.
- Boundary manifest: no live/order behavior.
- Tests proving no quote crosses post-only constraints.

Exit criteria:

- Cross-exchange decision path has a formal reservation price layer, still offline/public-shadow only.

### P5. Build Fill-Rate And Adverse-Selection Buckets From Controlled Evidence

Priority: medium-high

Why sixth:

- A maker strategy cannot optimize on alpha alone. It needs fill rate and fill quality conditioned on quote distance, source age, adverse flow, post-only reject drift, and inventory side.

Scope:

- After P0 controlled evidence produces reliable role/timestamp/fill lifecycle, synthesize buckets:
  - quote distance
  - edge vs fair/forecast/reservation
  - source age
  - basis bucket
  - adverse public-flow state
  - post-only reject drift state
  - inventory add/reduce side
  - fill/no-fill/reject/cancel outcome
  - maker/taker role
  - markout

Deliverables:

- Bucket-level fill-rate table.
- Bucket-level adverse markout table.
- Post-only reject drift table.
- Candidate quote-policy table:
  - safe touch
  - backoff/suppress
  - needs more evidence
  - reject

Verification:

- Artifact parser tests.
- Minimum sample gates.
- No default-on quote changes.

Exit criteria:

- The repo can identify where touch quoting is acceptable, where it is adverse, and where sample size is insufficient.

### P6. Design GLFT / Intensity-Based Spread As Offline Research

Priority: medium

Why seventh:

- Expert guidance recommends using trade-arrival depth/intensity to drive dynamic spread. The repo currently has queue/fill infrastructure but no accepted GLFT/intensity spread.

Scope:

- Start offline with public/trade data and accepted fill lifecycle artifacts.
- Estimate arrival intensity by depth bucket and time window.
- Fit simple intensity curves before considering full GLFT.
- Compare against current fixed/linear half-spread.

Deliverables:

- Intensity-by-depth table.
- Candidate `half_spread_dynamic` formula.
- Inventory interaction analysis.
- Simulation-only replay result with fail-closed assumptions.

Verification:

- No live behavior.
- No parameter promotion.
- Sensitivity report across windows.

Exit criteria:

- Either a dynamic-spread candidate is accepted for public shadow, or the current evidence remains too thin.

### P7. Implement Quote Policy Preflight Candidates One At A Time

Priority: medium

Why eighth:

- 0716T004 identified good candidates, but they must be tested one at a time to avoid mixing causes.

Candidate order:

1. Baseline no-change touch-only control.
2. Fill source-path capture improvements.
3. Post-only reject drift precheck.
4. Adverse public-flow suppression.
5. Inventory-aware reservation skew.
6. Dynamic spread / GLFT candidate.

Rule:

- Do not combine policy changes until each candidate has isolated shadow/replay evidence.

Deliverables:

- Candidate matrix.
- Pre/post funnel comparison.
- Counterfactual markout comparison.
- Boundary/unsupported claims manifest.

Verification:

- Public shadow first.
- Replay second.
- Tiny-live only after QA accepts prior stages and controller authorizes the exact envelope.

Exit criteria:

- A single policy candidate has enough evidence to justify a controlled live task without changing multiple variables at once.

### P8. Fee, Rebate, Inventory, And Realized PnL Calibration

Priority: medium, blocked until P0/P5

Why ninth:

- Profitability claims require maker/taker role, fee/rebate, inventory transition, and realized PnL source lines.

Prerequisite:

- Confirmed maker/taker role for fills.
- Exchange-native fill timestamps and fee records.
- Inventory before/after reconciliation.

Scope:

- Calibrate:
  - spread capture
  - fee/rebate
  - adverse markout
  - inventory MTM
  - cancel/fill race cost

Deliverables:

- Role-aware PnL ledger.
- Replay/live economics reconciliation.
- Fail-closed unknown-role handling.

Verification:

- Existing economics and account-inventory source validators.
- Same-window replay acceptance.
- No maker viability claim unless gates pass.

Exit criteria:

- Fee/PnL calibration can be used as an optimization objective without overclaiming.

### P9. Controlled Tiny-Live Re-Entry

Priority: last

Why last:

- Live retry before price, role, fill-quality, and PnL gates are accepted would mix evidence repair with strategy risk.

Prerequisite:

- P0 role evidence path accepted.
- P1 public shadow accepted.
- P5 fill/adverse buckets accepted.
- P8 fee/PnL source path accepted.
- Controller creates a fresh formal task with exact UTC schedule and envelope.

Scope:

- Same conservative envelope unless explicitly changed:
  - post-only maker only
  - tiny size
  - bounded submissions
  - tracked cancel
  - final open-orders proof
  - fail-closed max-loss monitor

Deliverables:

- Live evidence package.
- Same-window replay package.
- Role-aware fill/PnL package.
- QA acceptance report.

Verification:

- Public/private artifact reconciliation.
- Open-orders final proof.
- No unsupported maker viability claim.

Exit criteria:

- Either controlled live evidence supports the next replay/PnL calibration task, or the strategy remains blocked without parameter expansion.

## Summary Sequence

| Order | Work item | Main output | Unlocks |
| --- | --- | --- | --- |
| 1 | Fill source and liquidity-role evidence | Reliable maker/taker and fill timestamp source path | Fee/PnL and fill-quality calibration |
| 2 | T004 kernel public shadow | Real production-equivalent no-submit signal/fair-mid evidence | Quote-policy candidate evaluation |
| 3 | Price taxonomy contract | Clear definitions for mid/micro/fair/forecast/reservation/quote | Safe future implementation boundaries |
| 4 | Basis regression candidate test | Decision on basis as alpha vs context | Basis-adjusted fair/forecast design |
| 5 | Inventory-aware reservation kernel | `reservation = fair_mid - lambda * position` in cross-exchange path | Inventory-skew quote policy |
| 6 | Fill-rate/adverse buckets | Quote-distance and flow-conditioned fill quality | Policy preflight choices |
| 7 | GLFT/intensity spread research | Dynamic spread candidate or rejection | Spread optimization |
| 8 | Isolated quote-policy candidates | One change at a time, shadow/replay evidence | Controlled live task eligibility |
| 9 | Fee/PnL calibration | Role-aware economics ledger | Optimization objective |
| 10 | Controlled tiny-live re-entry | Real lifecycle/fill/PnL evidence | Next MVP gate |

## Non-Goals Until Gates Pass

- No default-on strategy change.
- No quote-envelope widening or tightening without a formal task.
- No order-size or max-submission expansion.
- No fee/PnL calibration while liquidity role is unknown.
- No maker viability, stable PnL, T012, promotion, or final MVP claim.
- No combined quote-policy bundle before isolated candidate evidence.
