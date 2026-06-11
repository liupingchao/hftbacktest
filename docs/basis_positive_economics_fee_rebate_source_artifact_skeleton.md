# Basis-Positive Economics Fee/Rebate Local Artifact Skeleton

Task: `0612T001`

This document describes a local-only economics fee/rebate/spread-capture artifact skeleton and validator derived from the accepted `0610T009` `economics_fee_rebate_source_line` contract and the accepted `0611T001` source-line synthesis gate.

It does not implement economics endpoints, fee endpoints, income readers, account readers, order readers, source collectors, user streams, signing, nonce handling, private/order/account/live/economics data reads, runner consumption, real economics metrics, real execution metrics, PnL proof, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Scope

The implementation is `examples/binance_tick_mm/economics_fee_rebate_source.py`. It validates only task-local CSV/JSON fixtures and generated local artifacts. Its purpose is to make the `0610T009` schema, settlement taxonomy, maker/taker classification policy, currency/tick arithmetic policy, spread-capture consistency policy, timestamp policy, reconciliation boundary, and overclaim rejection rules mechanically checkable before any future endpoint, collector, runner, cross-source reconciliation, or metric task exists.

Allowed current outputs:

- local schema columns
- synthetic valid and fail-closed fixture cases
- validator summary and detail CSVs
- arithmetic validation policy
- reconciliation boundary policy
- boundary validation
- manifest

## Validation Model

The validator fails closed for:

- missing required fields
- unknown enum values
- unsupported source policy or source authority
- forbidden endpoint, credential, user stream, action, strategy, live, deployment, or promotion fields
- missing or conflicting settlement authority
- unsupported maker/taker classification
- settlement or spread fail-closed labels accepted as valid
- invalid timestamp-domain separation
- non-conserving fee/rebate/net-fee arithmetic
- currency conversion mismatch
- tick-value arithmetic mismatch
- fee-adjusted spread consistency mismatch
- duplicate settlement identity
- hypothetical spread overclaims
- fill-notional-alone overclaims
- order-fills-alone overclaims
- public-markout-alone overclaims
- account-inventory-alone overclaims
- replay-lifecycle-alone overclaims
- PnL proof overclaims
- live, deployment, or promotion overclaims

## Arithmetic Boundary

Valid synthetic rows must satisfy:

- `fee_amount + rebate_amount == net_fee_amount`
- `net_fee_amount * conversion_rate == net_fee_in_settlement_currency`
- `net_fee_in_settlement_currency / tick_value == net_fee_ticks`
- `realized_spread_ticks - net_fee_ticks == fee_adjusted_spread_ticks`
- known amount sign convention, precision policy, rounding policy, settlement authority, and conversion provenance

These checks are local artifact checks only. Passing them does not prove real fees, rebates, spread capture, realized economics, or PnL because no accepted economics endpoint, collector, real settlement source, or cross-source reconciliation exists in this task.

## Reconciliation Boundary

Private order/fill observations remain future fill dependency context only. Fill notional and order fills alone cannot prove fees, rebates, spread capture, or PnL. Account inventory is future reconciliation context only. Replay lifecycle is timestamp/order context only. Public markout and hypothetical spread are context only, not realized spread-capture or PnL proof.

## Artifacts

Official task artifacts are under:

`local_live_analysis/basis_positive_economics_fee_rebate_source_artifact_skeleton_0612T001/`

The manifest records the accepted predecessor context:

- `source_task_id=0610T009`
- `source_final_recommendation=economics_fee_rebate_contract_ready_for_qa`
- `synthesis_task_id=0611T001`
- `synthesis_final_recommendation=source_line_synthesis_gate_ready_for_qa`
- `private_order_context_task_id=0611T002`
- `private_order_context_final_recommendation=private_order_response_artifact_skeleton_ready_for_qa`
- `replay_lifecycle_context_task_id=0611T003`
- `replay_lifecycle_context_final_recommendation=replay_lifecycle_validation_gate_ready_for_qa`
- `account_inventory_context_task_id=0611T004`
- `account_inventory_context_final_recommendation=account_inventory_artifact_skeleton_ready_for_qa`

Final recommendation: `economics_fee_rebate_artifact_skeleton_ready_for_qa`.
