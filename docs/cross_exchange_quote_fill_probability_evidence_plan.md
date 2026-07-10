# Cross-Exchange Quote/Fill Probability Evidence Plan

## Purpose

This is a planning document only. It does not create or dispatch a formal workflow task.

The accepted `0709T003` T011 route is:

- `route_to_quote_fill_probability_evidence`

Controller review narrows the T011 conclusion:

- T011 supports multi-window live artifact, lifecycle, safety, and non-optimistic consistency evidence.
- T011 does not prove full multi-window replay-engine regression.
- T011 does not prove fill probability, fee/rebate, realized PnL, stable PnL, maker viability, T012 readiness, promotion, or final MVP pass.
- The four-row T011 synthesis contains three newly collected `0709T001` live windows plus one prior `0708T001` QA reference through `0708T002`.

The next task should answer a smaller question:

```text
Given the accepted submitted/rejected and submitted/resting/no-fill windows, what evidence explains the observed no-fill outcome and what does it imply about quote/fill probability under the current conservative quote policy?
```

## Suggested Formal Task

Suggested title:

- `T011-QUOTE-FILL-PROBABILITY-EVIDENCE`

Suggested formal ID if dispatched on 2026-07-10:

- `0710T001`

Do not create this task file until the controller explicitly dispatches it.

## Scope

Default scope is offline-only analysis over accepted local artifacts.

Primary inputs:

- `local_live_analysis/cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z/`
- `local_live_analysis/cross_exchange_t011_batch_same_window_replay_acceptance_0709T002/`
- `local_live_analysis/cross_exchange_t011_multi_window_robustness_synthesis_0709T003/`
- `.workflow/reports/0709T001-qa.md`
- `.workflow/reports/0709T002-qa.md`
- `.workflow/reports/0709T003-qa.md`
- `.workflow/reports/0708T002-qa.md`

Useful artifact families:

- `order_intent_audit.csv`
- `quote_attempt_matrix.csv`
- `private_order_response_audit.json`
- `market_markout_snapshot.json`
- `public_stream_summary.json`
- `public_state_freshness_matrix.csv`
- `event_driven_latency_matrix.csv`
- `inline_reprice_latency_matrix.csv`
- `edge_gate_matrix.csv`
- `current_candidate_audit.csv`
- `rolling_flow_state.csv`
- `live_fill_ledger.csv`

## Hard Boundaries

This plan does not authorize:

- live-submit;
- remote/AWS execution;
- credential reads;
- private/account/order/cancel endpoint calls;
- new market-data collection;
- threshold changes;
- quote-envelope changes;
- order-size or max-submission changes;
- strategy behavior changes;
- T012;
- live expansion;
- stable PnL, maker viability, promotion, or final MVP claims.

If later evidence suggests live data collection is needed, that must be a separate formal task with an explicit envelope.

## Required Analysis

The formal task should separate these mechanisms:

1. Quote placement / priority:
   - Was the quote at touch, behind touch, or effectively rejected because it would cross?
   - For rejected attempts, identify whether the reject was consistent with post-only protection versus malformed placement.
   - For resting attempts, record side, limit price, size, TIF, quote age, and cancel horizon.

2. Same-side depth and queue proxy:
   - Estimate same-side visible depth at or ahead of the quote when the order rested.
   - Express visible depth as multiples of order size.
   - Mark attempts with insufficient public depth support as `depth_proxy_missing`, not as pass/fail.

3. Trade-through / depletion evidence:
   - Count public trades at or through the quote price during the resting interval.
   - Estimate visible same-side depletion before cancel/shutdown.
   - Separate "no fill because no trade-through" from "possible queue-ahead no-fill despite trade-through".

4. Time-to-fill censoring:
   - Record observation horizon and quote hold time per attempt.
   - Mark no-fill conclusions as censored when hold time is too short to infer low fill probability.
   - Avoid fitting a fill-probability model from censored samples unless the assumptions are explicit.

5. Opportunity cost:
   - Compare observed no-fill windows against nearby public flow after cancel.
   - Report missed-fill opportunity only when public same-side depletion/trade-through support exists.
   - Do not infer realized PnL or rebate from missed opportunity.

6. Route decision:
   - If no-fill is mostly explained by insufficient trade-through or large queue-ahead depth, route to quote policy design or more conservative evidence.
   - If evidence is missing because public depth/trade fields are insufficient, route to public-flow artifact repair.
   - If fills remain absent but visible depletion suggests quote placement is too passive, route to quote placement design, not threshold loosening by default.

## Required Outputs

The formal task should produce a local artifact package, for example:

- `local_live_analysis/cross_exchange_quote_fill_probability_evidence_<TASK_ID>/`

Expected files:

- `quote_fill_probability_manifest.json`
- `attempt_level_fill_probability_matrix.csv`
- `same_side_depth_proxy_matrix.csv`
- `trade_through_depletion_matrix.csv`
- `censoring_and_horizon_matrix.csv`
- `boundary_manifest.json`
- `validation_report.md`
- `sha256_manifest.csv`

## Recommendation Enum

The task must emit exactly one recommendation:

- `route_to_public_flow_artifact_repair`
- `route_to_more_conservative_evidence`
- `route_to_quote_policy_design`
- `route_to_controlled_same_envelope_live_evidence`
- `route_to_fee_inventory_pnl_calibration`
- `stop_for_human_strategy_decision`

`route_to_fee_inventory_pnl_calibration` is allowed only if the task finds accepted fill-supported evidence. With the current T011 no-fill set, the expected route should not be fee/PnL calibration.

## Acceptance Criteria

The task can pass only if:

- all outputs are generated and parseable;
- every accepted live attempt has a row in the attempt-level matrix;
- no-fill, reject, depth-missing, and censoring states are explicitly separated;
- any quote/fill probability statement is supported by public flow/depth evidence and includes censoring caveats;
- no synthetic fill, fee, rebate, realized PnL, queue priority, maker viability, promotion, T012, or final MVP claim is made;
- boundary manifest confirms offline-only behavior and no forbidden endpoint or parameter change.

## Verification

Expected verification for the formal task:

- focused pytest for any new runner;
- `python -m py_compile` for any new runner;
- runner `--help`;
- JSON/CSV parse and row-count checks;
- deterministic rerun check if the task writes generated artifacts;
- `git diff --check`.

## Current Controller State

No formal task has been created from this plan. The next controller action, if continuing, is to create exactly one task file under `.workflow/tasks/` using the scope above.
