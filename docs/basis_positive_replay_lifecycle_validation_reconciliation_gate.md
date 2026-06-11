# Basis-Positive Replay Lifecycle Validation / Reconciliation Gate

Task: `0611T003`

This document describes the local-only replay lifecycle validation / reconciliation gate implemented for `0611T003`. It is based on the accepted `0610T007` `replay_lifecycle_semantics_source_line` contract, the accepted `0611T001` source-line synthesis gate, and the accepted `0611T002` private-order response local skeleton as context only.

## Scope

The implementation is limited to:

- local schema constants for replay lifecycle artifacts;
- disk-only CSV/JSON fixture loading;
- fail-closed validation over required fields, enum domains, timestamp-domain separation, same-order lifecycle ordering, terminal-state singleton policy, duplicate event identity, cross-order causal overclaim, replay-as-execution-proof overclaim, queue-priority overclaim, and cancel-fill-race metric overclaim;
- synthetic task-local fixtures;
- validation summaries, ordering/reconciliation policy, manifest, and boundary validation.

It does not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, replay/live semantic implementation, private/order/account/live data, remote execution, collection, runner consumption, real execution metrics, real economics metrics, PnL proof, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Implementation

The implementation is `examples/binance_tick_mm/replay_lifecycle_validation_gate.py`.

The CLI supports:

- `python examples/binance_tick_mm/replay_lifecycle_validation_gate.py --help`
- `python examples/binance_tick_mm/replay_lifecycle_validation_gate.py validate --input <local-csv-or-json>`
- `python examples/binance_tick_mm/replay_lifecycle_validation_gate.py generate-artifacts --output-dir local_live_analysis/basis_positive_replay_lifecycle_validation_gate_0611T003`

The validator accepts only local CSV/JSON paths. It has no network, endpoint, credential, user stream, connector, source collector, runner, strategy, or live behavior.

## Validation Gates

The validator fails closed for:

- missing required fields;
- unknown enum values or unsupported source policy;
- missing, merged, or non-ordered timestamp domains;
- non-monotonic same-order causal sequence;
- invalid terminal lifecycle ordering;
- ambiguous, conflicting, or out-of-order events accepted as usable;
- duplicate lifecycle event identity;
- duplicate terminal state for the same opaque order reference;
- cross-order causal proof overclaim;
- replay-as-execution-proof overclaim;
- queue-priority or exact queue-position proof overclaim;
- cancel-fill-race metric overclaim.

These checks are local validation gates only. Passing them does not prove queue priority, exact queue position, cancel-fill race, real order lifecycle, real execution metrics, PnL, maker execution viability, live readiness, deployment readiness, or promotion readiness.

## Synthetic Fixtures

Official fixtures are under:

- `local_live_analysis/basis_positive_replay_lifecycle_validation_gate_0611T003/fixtures/`

Covered cases:

- `valid_same_order_cancel_lifecycle`
- `valid_fill_before_cancel_context`
- `missing_required_field`
- `unknown_enum_value`
- `merged_timestamp_domain`
- `non_monotonic_same_order_sequence`
- `ambiguous_conflicting_out_of_order_event`
- `duplicate_lifecycle_event_identity`
- `duplicate_terminal_state`
- `cross_order_causal_overclaim`
- `replay_as_execution_proof_overclaim`
- `queue_priority_proof_overclaim`
- `cancel_fill_race_metric_overclaim`

The official validation summary records `2` pass cases and `11` fail-closed cases. All expected statuses match.

## Artifacts

Official artifacts:

- `schema_columns.csv`
- `fixture_case_catalog.csv`
- `validator_result_summary.csv`
- `validator_result_details.csv`
- `ordering_reconciliation_policy.csv`
- `replay_lifecycle_validation_manifest.json`
- `boundary_validation.csv`

Final recommendation: `replay_lifecycle_validation_gate_ready_for_qa`.

This recommendation means only that the local replay lifecycle validation / reconciliation gate is ready for QA/controller review. It does not authorize endpoint work, source collection, runner consumption, replay/live semantic implementation, metric proof, strategy use, live readiness, deployment, or promotion.
