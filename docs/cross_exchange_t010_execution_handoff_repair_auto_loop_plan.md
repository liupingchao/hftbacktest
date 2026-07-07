# Cross-Exchange T010 Execution-Handoff Repair Auto-Loop Plan

## Purpose

This plan splits the post-`0706T010` blocker into three sequential tasks so that each task changes or measures only one layer.

`0706T010 / 0625T010-CONTROLLED-LIVE-EVIDENCE` proved:

- a controlled live watcher can run safely inside the low-risk envelope;
- trigger/pre-submit evidence exists;
- no live order was submitted;
- final open-orders remained `0`;
- full `0625T010` remains blocked.

The accepted blockers are:

- `edge_gate_source_status=missing_live_compatible_source`
- `post_open_orders_public_state_timeout`

Separately, trigger frequency is low and needs distribution diagnosis before changing anti-drift or touch-stability thresholds.

This plan intentionally separates:

1. live-compatible fair-mid / edge source binding repair;
2. post-open-orders public-state resync guard repair;
3. anti-drift / touch-stability live distribution diagnosis.

Do not combine these tasks. Combining them would make it hard to know which change caused any later improvement or regression.

## Current Fact Sources

Accepted QA facts:

- `0706T008` produced the long-window no-submit evidence that routed to controlled live evidence:
  - candidates: `2480`
  - fresh-touch allowed: `125`
  - edge pass / would-submit: `1`
- `0706T010` executed controlled live evidence:
  - candidates: `1872`
  - anti-drift pass/block: `5` / `107`
  - trigger found: `true`
  - trigger count: `1`
  - live submissions: `0`
  - real order endpoint called: `false`
  - final open-orders: `0`
  - independent final open-orders: `0`
  - blocker: `post_open_orders_public_state_timeout`
  - source blocker: `missing_live_compatible_source`

Accepted artifact roots:

- `local_live_analysis/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/`
- `local_live_analysis/cross_exchange_t010_controlled_live_evidence_0706T010_20260706T110202Z/`

## Auto-Loop Overview

```text
Task A: live-compatible fair-mid / edge source binding repair
  -> QA pass required
Task B: post-open-orders public-state resync guard repair
  -> QA pass required
Task C: anti-drift / touch-stability live distribution diagnosis
  -> QA pass required before any threshold change task
```

Default sequence:

1. Run Task A first.
2. Run Task B only after Task A QA passes.
3. Run Task C only after Task B QA passes.

No task in this plan authorizes a live order by itself. Any later controlled live-submit evidence attempt still requires a separate formal task and explicit envelope.

## Task A: `0707T001 / T010-LIVE-COMPATIBLE-EDGE-SOURCE-BINDING`

Goal:

- Repair the live decision path so that the event-driven live runner has a live-compatible fair-mid / edge source equivalent to the accepted no-submit source.
- Remove `edge_gate_source_status=missing_live_compatible_source` in dry-run/no-submit or mocked-live verification.

Problem statement:

- `0706T008` no-submit path can produce a fair-mid / edge pass.
- `0706T010` live path reports `missing_live_compatible_source`.
- This means the live runner cannot yet bind the accepted cross-exchange decision-time source into the final pre-submit edge gate.

Allowed scope:

- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- fair-mid / edge source provider code used by the live runner
- focused tests in `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- local/no-submit/mock-live artifacts proving the source is available

Not allowed:

- live-submit
- private/account/order/cancel endpoint use
- anti-drift threshold changes
- post-open-orders timeout changes
- quote-envelope changes
- size changes
- full T010/T011/T012 claims

Acceptance criteria:

- A focused test proves `--event-driven-edge-gate-live` can receive a live-compatible fair-mid / edge source.
- A local or mock-live artifact shows:
  - `edge_gate_live_compatible_source_available=true`
  - `edge_gate_source_status` is not `missing_live_compatible_source`
  - edge gate rows are populated from the live-compatible source
  - no order endpoint is called
  - no cancel endpoint is called
- Existing no-submit and watcher tests still pass.

Expected output:

- task file: `.workflow/tasks/0707T001.md`
- business report: `.workflow/reports/0707T001-business.md`
- QA report: `.workflow/reports/0707T001-qa.md`
- latest QA copied to `docs/qa-acceptance-report.md`
- local artifacts under `local_live_analysis/cross_exchange_t010_live_compatible_edge_source_0707T001/`

Route after QA:

- If QA passes, create Task B.
- If QA fails or blocks, do not start Task B.

## Task B: `0707T002 / T010-POST-OPEN-ORDERS-PUBLIC-STATE-RESYNC-REPAIR`

Goal:

- Repair the submit-before-order public-state resync guard so it remains fail-closed on stale state but does not systematically block because of a fixed `0.2s` wait.

Problem statement:

- `0706T010` observed `post_open_orders_public_state_timeout` in `5/5` post-open-orders resync attempts.
- The current guard requires an L2 state observed after read-only `open_orders()` and before submit.
- The safety requirement is correct, but the mechanism is too brittle if it only waits a short fixed interval for a fresh l2Book event.

Allowed scope:

- post-open-orders resync logic in `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- deterministic tests for:
  - pass when a fresh post-open-orders L2 exists
  - fail-closed when L2 is stale
  - fail-closed when no post-open-orders state can be proven
- local/mock-live artifacts showing resync pass/fail cases

Not allowed:

- weakening stale-state protection
- submitting an order without a fresh public state
- live-submit
- changing anti-drift thresholds
- changing fair-mid / edge thresholds
- quote-envelope or size changes

Acceptance criteria:

- Focused tests prove the guard can pass with valid post-open-orders L2 evidence.
- Focused tests prove stale/no-proof cases still fail closed.
- Local/mock-live artifact shows:
  - `post_open_orders_public_state_pass_count > 0` in the positive case
  - `post_open_orders_public_state_timeout` or stale-state reason in the negative case
  - no order endpoint is called
  - no cancel endpoint is called
- `0706T010` blocker is explicitly addressed without changing risk envelope.

Expected output:

- task file: `.workflow/tasks/0707T002.md`
- business report: `.workflow/reports/0707T002-business.md`
- QA report: `.workflow/reports/0707T002-qa.md`
- latest QA copied to `docs/qa-acceptance-report.md`
- local artifacts under `local_live_analysis/cross_exchange_t010_post_open_orders_resync_0707T002/`

Route after QA:

- If QA passes, create Task C.
- If QA fails or blocks, do not start Task C.

## Task C: `0707T003 / T010-ANTI-DRIFT-TOUCH-STABILITY-LIVE-DISTRIBUTION-DIAGNOSIS`

Goal:

- Diagnose live distribution of anti-drift and touch-stability filters before any threshold change.

Problem statement:

- `0706T008` and `0706T010` show candidate volume is not zero, but the funnel is narrow.
- Large portions are filtered by:
  - `missing_same_side_strict_through_support`
  - `missing_recent_same_side_at_or_through_throughput`
  - `outside_quality_a_b_queue_bands`
  - `touch_stability_below_minimum`
  - `fair_mid_source_stale`
- We need distribution evidence before deciding whether anti-drift or touch-stability thresholds are too conservative.

Allowed scope:

- read-only analysis over accepted `0706T008` and `0706T010` artifacts
- optional public-only/no-submit additional diagnostic window if existing artifacts are insufficient
- distribution reports, histograms, quantiles, retention curves

Not allowed:

- changing thresholds
- live-submit
- private/account/order/cancel endpoint use
- quote-envelope changes
- size changes
- strategy promotion

Required diagnostics:

- candidate counts by funnel stage
- touch stability quantiles
- anti-drift pass/block distribution
- top reasons and co-occurring reasons
- retention curve if changing stability threshold hypothetically from current value to nearby candidate thresholds
- separate public-only proxy evidence from execution-layer evidence
- explicit recommendation:
  - keep thresholds
  - propose a later threshold-change task
  - collect more no-submit data

Acceptance criteria:

- The report uses accepted artifacts and/or a clearly bounded public-only/no-submit diagnostic.
- It does not modify thresholds.
- It provides enough distribution evidence for a later controller decision.
- It states whether another controlled live evidence task is justified only after Task A and Task B are both accepted.

Expected output:

- task file: `.workflow/tasks/0707T003.md`
- business report: `.workflow/reports/0707T003-business.md`
- QA report: `.workflow/reports/0707T003-qa.md`
- latest QA copied to `docs/qa-acceptance-report.md`
- artifacts under `local_live_analysis/cross_exchange_t010_anti_drift_distribution_0707T003/`

Route after QA:

- If thresholds look reasonable, next step may be a separately authorized controlled live evidence task using repaired A+B code.
- If thresholds look too conservative, create a separate threshold-change design task. Do not silently change thresholds inside Task C.
- If evidence is inconclusive, collect another public-only/no-submit diagnostic window.

## Stop Conditions

Stop the auto-loop and return to controller decision if:

- Task A cannot provide a live-compatible edge source without weakening decision-time constraints.
- Task B cannot pass post-open-orders resync without allowing stale public state.
- Task C shows that trigger frequency is too low but threshold changes would require a strategy decision.
- Any task would require live-submit, size increase, quote-envelope relaxation, taker/crossing behavior, or default-on operation.

## Live Boundary

This plan itself authorizes no live order.

The next live-submit-capable task must be separate from this plan and must include:

- explicit task file
- explicit live envelope
- max submissions
- max size
- post-only `Alo`
- tracked cancel
- independent final open-orders proof
- QA acceptance

## Immediate Next Action

Create and dispatch only:

- `0707T001 / T010-LIVE-COMPATIBLE-EDGE-SOURCE-BINDING`

Do not create `0707T002` or `0707T003` until the preceding task has QA-passed.
