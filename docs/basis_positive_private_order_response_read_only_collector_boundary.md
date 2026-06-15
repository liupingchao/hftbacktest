# Basis-Positive Private Order Response Read-Only Collector Boundary

Task: `0615T002`

## Purpose

This document defines the boundary for a future private order response read-only collector implementation task. It is the first non-local source-line direction after the `0615T001` convergence policy. It remains design and local artifact work only.

The future source path may collect private order response artifacts only under a separately dispatched and QA-accepted no-trading task. This task does not implement or call endpoints, implement a collector, read credentials, sign requests, handle nonce values, subscribe to user streams, read real private/order/account/live/economics data, place orders, cancel orders, feed runners, compute execution metrics, run strategy logic, run live/default-on/tiny-live, search parameters, deploy, promote, or prove maker viability.

## Prerequisite Status

- `0615T001` QA is `已通过` with final recommendation `real_source_line_readiness_boundary_ready_for_qa`.
- `0611T002` accepted final recommendation is `private_order_response_artifact_skeleton_ready_for_qa`. The direct QA file is still absent in this workspace snapshot, but downstream QA/tracking records cite the accepted result; this remains a fact-source completeness caveat, not an execution blocker.
- `0610T006` QA is `已通过` with final recommendation `private_order_response_contract_ready_for_qa`.
- `0611T001` QA is `已通过` with final recommendation `source_line_synthesis_gate_ready_for_qa`.

## Endpoint And Permission Boundary

A later implementation task may document and enforce read-only private-order-response authority. It must be separated from trading authority:

- Allowed future purpose: read or receive order response facts for orders already created by an independently authorized context.
- Forbidden future purpose in this line: place an order, cancel an order, amend an order, emit order side, quote price, quote size, executable trigger, strategy decision, deployment flag, or promotion flag.
- Credential handling remains outside this task. A later implementation task must fail closed if any action-capable method is reachable from the collector path.
- User-stream and signed-request options remain implementation choices for a later task only after a security and QA gate. This design does not choose or implement either option.

## Schema Handoff

Future real source fields must be transformed into the accepted `0611T002` schema in `examples/binance_tick_mm/private_order_response_source.py`. The handoff target fields include:

- Source provenance: `task_id`, `source_task_id`, `source_line_id`, `venue`, `instrument`, `artifact_source_class`, `source_policy`.
- Event identity: `artifact_event_id`, `event_sequence_index`, `client_order_ref_hash`, `exchange_order_ref_hash`.
- Timing: `exchange_event_time`, `local_receive_time`, `artifact_generated_time`, `validation_or_reconciliation_time`.
- Labels: `response_category`, `lifecycle_state_label`, `post_only_reject_class`, `terminal_state_marker`, `unknown_missing_conflicting_flag`.
- Validation: `validation_gate_id`, `validation_status`, `fail_closed_reason`, `overclaim_rejection_id`, `allowed_future_use`, `forbidden_current_interpretation`.

The mapping is an artifact contract, not proof. Passing schema validation does not prove fill probability, post-only reject behavior, real order lifecycle, queue priority, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, PnL, live readiness, deployment readiness, promotion readiness, or maker execution viability.

## Redaction And Local Storage

Future artifacts must be local, redacted, and reproducible:

- Raw account identifiers, API keys, secrets, signatures, nonce payloads, raw client order IDs, and raw exchange order IDs must not be stored in task artifacts.
- Order references must be hashed or otherwise made opaque before artifact persistence.
- Timestamp domains must remain separate; local artifact generation time must not be substituted for exchange event time.
- Provenance must record source task, collector task, venue, instrument, local run id, schema version, validation gate id, and fail-closed reason when applicable.

## No-Trading Safety Gates

A later collector implementation task must include no-trading gates before any endpoint or stream code is allowed:

- Dry-run or read-only mode must be the only supported mode for the collector path.
- Any order placement, cancellation, amend, quote, strategy hook, runner-consumption hook, or action-capable method reachable from the collector path must be a hard failure.
- Collector output must be local artifacts only.
- Real artifacts must not be used by a runner until a separately scoped runner-consumption gate is QA accepted.

## Future QA Gates

A later implementation QA must verify:

- The collector output validates against `private_order_response_source.py`.
- Forbidden fields and action-capable methods are absent or fail closed.
- Credential, signing, nonce, and user-stream concerns are isolated from persisted artifacts.
- Negative tests cover forbidden trading actions, raw identifiers, secrets, merged timestamp domains, unknown enum values, unsupported source policy, and overclaim text.
- Boundary text preserves the prohibition on metrics, PnL, strategy, live, deployment, promotion, and maker viability claims.

## Next Task

The next actionable step after QA is a separately scoped read-only collector implementation task for private order response artifacts. It must remain no-trading, local-artifact-only, validator-backed, and QA-gated. It still must not authorize runner consumption, strategy/live behavior, metrics, PnL, deployment, promotion, or maker viability proof.

Final recommendation: `private_order_response_read_only_collector_boundary_ready_for_qa`.
