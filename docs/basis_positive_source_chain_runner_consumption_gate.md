# Basis-Positive Source-Chain Runner-Consumption Gate

Task: `0615T006`

This design defines the gate between accepted read-only source artifacts and a later proof-limited execution evidence runner. It is not a runner implementation and it does not authorize live trading, endpoint use, credential handling, order actions, PnL proof, deployment, promotion, or maker viability claims.

## Accepted Source Inputs

The later runner may consume only artifacts from QA-accepted tasks:

- `0615T003`: private order response read-only collector artifacts, validated by `private_order_response_source.py`.
- `0611T003`: replay lifecycle validation / reconciliation gate artifacts, validated by `replay_lifecycle_validation_gate.py`.
- `0615T004`: account inventory read-only source artifacts, validated by `account_inventory_source.py`.
- `0615T005`: economics fee/rebate read-only source artifacts, validated by `economics_fee_rebate_source.py`.

Each source line remains independently bounded. A downstream row may be emitted only when its source path, timestamp domain, opaque identity policy, source-policy value, validator status, and proof-limit class are explicit.

## Runner Consumption Rule

The later `0615T007` runner may consume these artifacts only to produce proof-limited rows. It must fail closed for missing source lines, inconsistent timestamps, identity mismatches, local synthetic-only evidence, stale validation artifacts, forbidden fields, or any metric overclaim.

The runner must not convert local fixture evidence into real execution proof. Local/read-only artifacts may support schema and reconciliation mechanics only. Real execution proof remains unavailable unless a later task separately accepts real source-path evidence.

## Timestamp Reconciliation

Timestamp domains remain separate:

- private order response: local receive, exchange event, artifact generation, validation.
- replay lifecycle: decision, replay, exchange event, local receive, validation.
- account inventory: account state, transition, local receive, artifact generation, validation.
- economics fee/rebate: fill, settlement, conversion, receive, reconciliation, validation.

The later runner must preserve domain labels and must not merge them into a single causal timestamp. Same-order ordering may be checked only inside an accepted source line. Cross-source alignment is reconciliation context unless a later real-source task explicitly upgrades it.

## Identity And Redaction

All cross-source linkage must use opaque identifiers. Raw account IDs, raw client order IDs, raw exchange order IDs, API keys, secrets, signatures, nonces, and listen keys are forbidden in runner inputs and outputs.

Opaque references can link rows only when they are explicitly declared as future-fill, future-order, account-scope, or settlement context references. A missing or conflicting opaque reference must produce a fail-closed row.

## Proof Limits

The later runner may emit these classes:

- `unavailable_missing_source`
- `proof_limited_local_artifact_only`
- `proof_limited_replay_regression_only`
- `proof_limited_cross_source_context_only`
- `mechanically_validated_not_execution_proof`
- `blocked_overclaim_rejected`

It must not emit ready/live/deploy/promote statuses.

## Next Sequence

The next formal task is `0615T007`, a proof-limited read-only runner implementation. It must pass QA before `0615T008` live-test protocol design. The first task that may open a small-cap live test is `0615T009`, and only after `0615T008` QA and explicit total-control approval.
