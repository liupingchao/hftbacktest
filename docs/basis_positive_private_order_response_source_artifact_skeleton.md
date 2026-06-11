# Basis-Positive Private Order Response Local Artifact Skeleton

Task: `0611T002`

This document describes the local-only private order response artifact skeleton and validator implemented for `0611T002`. It is based on the accepted `0610T006` `private_order_response_source_line` contract and the accepted `0611T001` source-line synthesis gate.

## Scope

The implementation is limited to:

- local schema constants for a private-order response artifact;
- disk-only CSV/JSON fixture loading;
- fail-closed validation over required fields, enum domains, timestamps, terminal-state consistency, duplicate event identity, unsupported source policy, and conflicting terminal outcomes;
- synthetic task-local fixtures;
- validation summaries, manifest, and boundary validation.

It does not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, private/order/account/live data, remote execution, collection, runner consumption, real execution metrics, real economics metrics, PnL proof, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Implementation

The implementation is `examples/binance_tick_mm/private_order_response_source.py`.

The CLI supports:

- `python examples/binance_tick_mm/private_order_response_source.py --help`
- `python examples/binance_tick_mm/private_order_response_source.py validate --input <local-csv-or-json>`
- `python examples/binance_tick_mm/private_order_response_source.py generate-artifacts --output-dir local_live_analysis/basis_positive_private_order_response_source_artifact_skeleton_0611T002`

The validator accepts only local CSV/JSON paths. It has no network, endpoint, credential, user stream, connector, source collector, runner, strategy, or live behavior.

## Validation Gates

The validator fails closed for:

- missing required fields;
- unknown enum values;
- unsupported artifact source class or source policy;
- missing or non-ordered timestamps;
- terminal response without terminal marker;
- terminal marker incompatible with lifecycle state;
- accepted validation status with fail-closed flags;
- terminal rows without opaque order references;
- duplicate artifact event identity;
- incompatible terminal outcomes for the same opaque order reference.

These checks are skeleton gates only. Passing them does not prove fill probability, post-only reject behavior, real order lifecycle, PnL, maker execution viability, live readiness, deployment readiness, or promotion readiness.

## Synthetic Fixtures

Official fixtures are under:

- `local_live_analysis/basis_positive_private_order_response_source_artifact_skeleton_0611T002/fixtures/`

Covered cases:

- `valid_accepted_lifecycle`
- `valid_post_only_reject`
- `missing_required_field`
- `unknown_enum_value`
- `conflicting_terminal_state`
- `bad_timestamp`
- `duplicate_event_identity`
- `unsupported_evidence_source`
- `incomplete_lifecycle_evidence`

The official validation summary records `2` pass cases and `7` fail-closed cases. All expected statuses match.

## Artifacts

Official artifacts:

- `schema_columns.csv`
- `fixture_case_catalog.csv`
- `validator_result_summary.csv`
- `validator_result_details.csv`
- `private_order_response_skeleton_manifest.json`
- `boundary_validation.csv`

Final recommendation: `private_order_response_artifact_skeleton_ready_for_qa`.

This recommendation means only that the local artifact skeleton / validator is ready for QA/controller review. It does not authorize endpoint work, source collection, runner consumption, metric proof, strategy use, live readiness, deployment, or promotion.
