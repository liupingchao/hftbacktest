# Basis-Positive Account Inventory Local Artifact Skeleton

Task: `0611T004`

This document describes a local-only account/inventory artifact skeleton and validator derived from the accepted `0610T008` `account_inventory_source_line` contract and the accepted `0611T001` source-line synthesis gate.

It does not implement account endpoints, source readers, source collectors, user streams, signing, nonce handling, private/order/account/live data reads, runner consumption, real inventory metrics, real execution metrics, economics metrics, PnL proof, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Scope

The implementation is `examples/binance_tick_mm/account_inventory_source.py`. It validates only task-local CSV/JSON fixtures and generated local artifacts. Its purpose is to make the `0610T008` schema/taxonomy/conservation contract mechanically checkable before any future source collection or cross-source reconciliation task exists.

Allowed current outputs:

- local schema columns
- synthetic valid and fail-closed fixture cases
- validator summary and detail CSVs
- conservation check policy
- reconciliation boundary policy
- boundary validation
- manifest

## Validation Model

The validator fails closed for:

- missing required fields
- unknown enum values
- unsupported source policy or source authority
- forbidden endpoint, credential, user stream, action, strategy, live, deployment, or promotion fields
- missing, stale, partial, conflicting, or unsupported snapshot labels accepted as valid
- unknown, ambiguous, conflicting, or unsupported transition labels accepted as valid
- invalid timestamp-domain separation
- non-conserving available/locked/total quantities
- non-conserving before/after/delta quantities
- unit, precision, or sign-policy inconsistencies
- duplicate transition identity
- out-of-order transition evidence
- order-fills-alone inventory proof overclaims
- current inventory lifecycle proof overclaims
- PnL or economics proof overclaims
- live, deployment, or promotion overclaims

## Conservation Boundary

Valid synthetic rows must satisfy:

- `available_quantity + locked_quantity == total_quantity`
- `before_quantity + delta_quantity == after_quantity`
- known quantity unit
- known precision policy
- stable account scope / asset / instrument identity
- ordered account-state time and validation time domains

These checks are local artifact checks only. Passing them does not prove real inventory lifecycle because no accepted account-state endpoint, collector, or real account data exists in this task.

## Reconciliation Boundary

Order fills remain future candidate transition inputs only. A fill-derived transition cannot prove inventory lifecycle without account-state authority and accepted conservation checks. Replay lifecycle observations are event-order context only. Economics settlement remains a separate future source line.

## Artifacts

Official task artifacts are under:

`local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004/`

The manifest records the accepted predecessor context:

- `source_task_id=0610T008`
- `source_final_recommendation=account_inventory_contract_ready_for_qa`
- `synthesis_task_id=0611T001`
- `synthesis_final_recommendation=source_line_synthesis_gate_ready_for_qa`
- `private_order_context_task_id=0611T002`
- `private_order_context_final_recommendation=private_order_response_artifact_skeleton_ready_for_qa`
- `replay_lifecycle_context_task_id=0611T003`
- `replay_lifecycle_context_final_recommendation=replay_lifecycle_validation_gate_ready_for_qa`

Final recommendation: `account_inventory_artifact_skeleton_ready_for_qa`.
