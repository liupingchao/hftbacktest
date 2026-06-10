# Basis-Positive Account Inventory Source-Line Contract

Task: `0610T008`

This document defines a design-only contract for `account_inventory_source_line`. It does not implement source readers, source collectors, account/private/order/live endpoints, user streams, signing, nonce handling, runner behavior, real inventory metrics, real execution metrics, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Prerequisites

`0610T005` QA is accepted with final recommendation `private_order_source_design_ready_next`, and its source-line mapping assigns `inventory_lifecycle` to `account_inventory_source_line`.

`0610T006` QA is accepted with final recommendation `private_order_response_contract_ready_for_qa`. Its private-order response contract is future transition-input and future cross-check context only. It is not current account inventory proof.

## Covered Gap

This contract covers exactly one primary gap assigned by `0610T005`:

- `inventory_lifecycle`

The gap remains a design label only. Current artifacts do not prove inventory lifecycle, realized inventory, realized exposure, PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion.

## Core Boundary

Order fills alone cannot prove inventory lifecycle.

Order fills may later be candidate transition inputs, but inventory lifecycle requires account-state authority, before/after inventory snapshots or accepted account-observed transitions, conservation checks, and fail-closed reconciliation. A fill label without account-state evidence cannot prove available balance, locked balance, total balance, position quantity, settlement timing, fee/rebate inventory effects, transfers, funding, manual adjustments, or external adjustments.

## Artifact Schema Boundary

The future account/inventory artifact is a local accepted artifact contract. It is not an endpoint specification. The schema must not contain endpoint URLs, API keys, secrets, signing payloads, nonce fields, user stream subscription details, executable order side, quote price, quote size, strategy signal, live gate, deployment flag, or promotion flag.

Required schema groups:

- Account scope identity: venue, account scope, account alias or opaque account reference, subaccount scope when applicable, margin mode design label, and validation status.
- Asset and instrument identity: asset, instrument, quote asset, base asset, precision, unit, and conversion context where applicable.
- Inventory state fields: snapshot type, available quantity, locked quantity, total quantity, position quantity, notional context design label, and state completeness.
- Transition fields: transition type, before quantity, after quantity, delta quantity, delta attribution, related future opaque order reference when available, and transition validation status.
- Provenance: source artifact id, source class, source policy, source task id, prior source-line context, validation gate id, and fail-closed reason.
- Timing: account state time, exchange event time when present, local receive time, artifact generated time, and validation or reconciliation time as separate fields.
- Proof limits: allowed future use, forbidden current interpretation, overclaim rejection id, current proof status, and design-only marker.

## Snapshot Taxonomy

Inventory snapshots are design labels. Snapshot labels must distinguish complete evidence from fail-closed evidence:

- `initial_snapshot_design_label`
- `periodic_snapshot_design_label`
- `pre_transition_snapshot_design_label`
- `post_transition_snapshot_design_label`
- `reconciliation_snapshot_design_label`
- `missing_snapshot_fail_closed`
- `stale_snapshot_fail_closed`
- `partial_snapshot_fail_closed`
- `conflicting_snapshot_fail_closed`
- `unsupported_snapshot_fail_closed`

Missing, stale, partial, conflicting, and unsupported snapshots cannot support current inventory lifecycle proof.

## Transition Taxonomy

Inventory transitions are design labels. A fill-derived transition is only a candidate until account state and conservation checks accept it.

- `fill_derived_candidate_transition`
- `account_observed_transition`
- `transfer_adjustment_transition`
- `funding_settlement_adjustment_transition`
- `fee_rebate_inventory_adjustment_transition`
- `manual_external_adjustment_transition`
- `unknown_transition_fail_closed`
- `ambiguous_transition_fail_closed`
- `conflicting_transition_fail_closed`
- `unsupported_transition_fail_closed`

Unknown, ambiguous, conflicting, and unsupported transitions fail closed. A transition with no account-state observation or no accepted reconciliation remains unresolved.

## Conservation Checks

Conservation checks are mandatory before any later design may treat inventory lifecycle evidence as usable. The checks include:

- Before/after quantity consistency.
- Delta attribution to one accepted transition class.
- Available, locked, and total relationship checks.
- Per-asset and per-instrument consistency.
- Sign, unit, and precision checks.
- Duplicate transition detection.
- Out-of-order transition detection.
- Tolerance policy for venue precision and rounding.

Non-conserving, unit-inconsistent, precision-invalid, duplicate, out-of-order, or unattributed transitions must be marked unresolved and fail closed.

## Reconciliation Boundary

The contract separates these evidence authorities:

- Order/fill observations: future candidate transition inputs only.
- Account snapshots: future account-state authority for inventory state.
- Account transitions: future state-change authority after conservation checks.
- Economics settlement: future fees, rebates, funding, and conversion authority, not owned by this contract except as adjustment labels.
- Replay lifecycle observations: supporting regression or event-order context only, not account inventory proof.
- Future endpoint or collector responsibilities: explicitly out of scope for this task.

## Timestamp Policy

The contract separates these times:

- `account_state_time`: source-reported account state time when present.
- `exchange_event_time`: exchange event time for a related future order or account event when present.
- `local_receive_time`: local receive time for the account/inventory artifact.
- `artifact_generated_time`: local artifact generation time.
- `validation_or_reconciliation_time`: time when the artifact is validated or reconciled.

No future design may treat artifact generation time as account state time or exchange event time. Missing, stale, out-of-order, or contradictory timing evidence must fail closed.

## Validation Gates

Validation gates are fail-closed. The required gate classes are:

- prerequisite source-line gate
- schema forbidden-field gate
- account scope identity gate
- asset and instrument identity gate
- snapshot completeness gate
- transition taxonomy gate
- quantity conservation gate
- available locked total relationship gate
- unit precision and sign gate
- duplicate and out-of-order gate
- unknown ambiguous conflicting evidence gate
- reconciliation boundary gate
- overclaim rejection gate
- boundary preservation gate

## Overclaim Rules

This contract rejects current claims of:

- inventory lifecycle proof
- inventory lifecycle proof from order fills alone
- realized inventory or exposure proof
- PnL or realized economics proof
- maker execution viability proof
- live/default-on/tiny-live readiness
- deployment readiness
- promotion readiness
- future labels as strategy, shadow, case-library, parameter-search, live-decision, or deployment criteria

## Allowed Future Use

If accepted by QA, this contract may become an input to a later separately dispatched design task. It may define the format a future account/inventory artifact should satisfy. It does not authorize account endpoint use, source collection, runner extension, execution metric computation, or trading behavior.

Final recommendation: `account_inventory_contract_ready_for_qa`.
