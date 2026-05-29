# Binance Tick MM Next Policy Design

Task: `0529T001`

## Decision

The next Binance maker policy work should be fill-quality-first. Do not continue by tuning the rejected inventory-aware quote placement skeleton, and do not return to the current min-move projected-suppression grid as the main route.

The immediate next task should be a read-only fill-quality bucket synthesis runner. Existing artifacts are enough to reject the recent skeleton and define the design boundary, but they are not enough to implement a new strategy policy.

## Accepted Facts

- `0528T002` fixed Stage 6 audit input scaling by making Stage 6 consume `audit_bt_audit_replay.compact_lifecycle.csv`. This is an infrastructure/input fix, not maker edge evidence.
- `0528T001` rejected the fixed `inventory_aware_quote_placement_request` skeleton. Clean request buckets had enough mass, but worse quality:
  - request fills: `528`
  - request 5s markout: `-85.21` ticks
  - no-change 5s markout: `-70.20` ticks
  - request spread capture: `6.85` ticks
  - no-change spread capture: `16.78` ticks
- `0526T004` found no promising `min_move_quote_age_churn_guard` parameter set:
  - `sweep_seed_promising=0`
  - `reject=80`
  - `not_decisionable=676`
- `0526T005` still found maker edge families worth studying, but `0528T001` showed inventory must not be used as a direct quote-placement trigger. Inventory is context for side pressure, sizing, and risk, not the trigger base.

## Rejected Tracks

- Fixed inventory-aware quote placement skeleton from `0528T001`: rejected because it found more fills but worse fills.
- Current min-move projected-suppression grid from `0526T004`: rejected as a near-term main route because the read-only sweep found no promising parameter sets.
- Direct tiny-live / default-on / promotion: not supported by current evidence.
- Further work on audit bloat after `0528T002`: not useful unless a concrete compact-audit regression appears.

## Evidence Interpretation

The current Stage 9I decomposition shows a tradeoff, not a ready policy.

High-participation/touch buckets:

- `request_side_priority / allow_touch` has high fill rate: `0.0768`.
- It has weak spread capture: `1.02` ticks.
- It still has adverse markout: `-88.13` ticks.
- This bucket increases participation but does not prove profitable fill quality.

More passive quote-distance buckets:

- `request_quote_adjustment / prefer_one_tick_tight` has better spread capture: `30.77` ticks.
- It has lower fill rate: `0.0124`.
- It still has adverse markout: `-81.22` ticks and higher fill-after-cancel rate: `0.656`.

The best-looking fine buckets are not strategy-ready:

- `flat / weak_or_neutral / step_back_gt1 / no_change` has relatively least-bad markout (`-8.72` ticks) and good spread capture (`28.05` ticks), but it is a no-change bucket, not a new policy action.
- `large_skew_or_low_score / add_side / weak_or_neutral / step_back_gt1 / request_size_adjustment` has good spread capture (`32.89` ticks) and better markout than most request buckets (`-46.86` ticks), but it is add-side/inventory-increasing, not a safe inventory-improvement policy by itself.

Conclusion: existing artifacts identify the axes of interest, but they do not yet identify a positive-quality trigger base that can safely drive quote behavior.

## Policy Contract

The next policy shape must obey these rules:

- Trigger base comes from positive fill-quality buckets, not raw fill count or inventory state alone.
- Inventory is a risk, sizing, and side-pressure context.
- Quote distance defines the participation frontier.
- Stale, latency, post-only, reject, throttle, and churn fields are safety filters.
- Exact queue position, hidden queue assumptions, and future outcomes are not allowed as decision inputs.

## Candidate Shapes

These are design candidates only. Neither is authorized for implementation yet.

### Shape A: Passive Quality Gate With Inventory Sizing

Trigger conditions:
- Only consider buckets with acceptable side-adjusted markout, spread capture, and fill-after-cancel sensitivity.
- Prefer step-back / one-tick passive states over touch participation unless touch states prove positive quality.

Inventory role:
- Scale size down on add-side inventory pressure.
- Preserve or modestly prioritize reduce-side only when quality gates pass.

Quote-distance role:
- Defines whether the quote can be one-tick tight or must remain step-back.
- Does not force touch simply to increase fill count.

Safety filters:
- Exclude stale / unsafe anchor contexts.
- Require post-only recheck risk `0`.
- Reject buckets with elevated reject/throttle/churn or fill-after-cancel sensitivity.

Validation labels / metrics:
- clean-only fill mass
- 5s side-adjusted markout
- spread capture
- fill-after-cancel rate
- inventory increasing/reducing fill split
- quote-distance, latency, stale, and post-only strata stability

Difference from `0528T001`:
- Inventory does not directly generate quote-placement requests. It only scales size or side pressure after fill-quality gates pass.

### Shape B: Reduce-Side Participation Gate With Spread-Capture Floor

Trigger conditions:
- Only consider reduce-side participation when spread capture and markout meet a floor.
- Touch / allow-touch is not allowed unless it passes quality gates across clean samples.

Inventory role:
- Reduce-side bias is allowed only as an inventory repair action with fill-quality proof.
- Add-side is suppressed or sized down when inventory risk is high unless its own bucket is quality-positive.

Quote-distance role:
- Allow tighter quote distance only for reduce-side states with acceptable spread capture.
- Otherwise remain one-tick or step-back.

Safety filters:
- Same as Shape A, plus explicit check that inventory repair does not come from systematically worse adverse-selection fills.

Validation labels / metrics:
- reduce-side fill mass
- reduce-side markout and spread capture
- inventory recovery quality
- fill-after-cancel rate
- stale / latency / post-only strata stability
- reject/throttle/churn safety

Difference from `0528T001`:
- `0528T001` used inventory state to request side priority broadly. Shape B requires quality proof before any reduce-side participation priority.

## Immediate Next Task

Create a read-only fill-quality bucket synthesis runner.

Minimum runner contract:

- Inputs:
  - Stage 5 execution labels
  - Stage 5C quote-anchor safety diagnostics
  - Stage 6 calibration outputs
  - Stage 9B/9D/9H/9I artifacts
  - current-format sample directories
- Outputs:
  - bucket-level fill-quality table
  - clean-only and caveated sensitivity summaries
  - candidate trigger tables for Shape A and Shape B
  - rejected bucket table
  - recommendation markdown
- Required strata:
  - inventory bucket
  - add/reduce/flat side
  - quote distance
  - fair/reservation edge
  - latency/stale state
  - post-only/reject/throttle/churn safety
  - fill-after-cancel sensitivity
- Required verdicts:
  - `ready_for_policy_design`
  - `needs_more_clean_fills`
  - `reject_quality_negative`
  - `not_decisionable`

This runner should remain read-only/default-off. It must not implement strategy behavior, parameter search, live trading, default-on behavior, guard relaxation, or promotion.
