# Basis-Positive Execution Source-Line Synthesis Gate

Task: `0611T001`

This document defines a design-only synthesis and implementation-readiness gate over the accepted basis-positive execution source-line contracts. It does not implement endpoints, source readers, collectors, runners, user streams, signing, nonce handling, real execution metrics, real economics metrics, PnL proof, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Prerequisites

The following design-only source-line contracts are accepted by QA:

- `0610T006`: `private_order_response_source_line`, final recommendation `private_order_response_contract_ready_for_qa`.
- `0610T007`: `replay_lifecycle_semantics_source_line`, final recommendation `replay_lifecycle_contract_ready_for_qa`.
- `0610T008`: `account_inventory_source_line`, final recommendation `account_inventory_contract_ready_for_qa`.
- `0610T009`: `economics_fee_rebate_source_line`, final recommendation `economics_fee_rebate_contract_ready_for_qa`.

`0610T005` is accepted with final recommendation `private_order_source_design_ready_next` and maps exactly seven execution gaps to four source lines. The source-line `ready_for_qa` recommendations mean only that the design contracts are ready for QA/controller review. They do not mean implementation readiness, endpoint readiness, metric proof, PnL proof, or live readiness.

## Covered Source Lines

This synthesis covers exactly four source lines:

- `private_order_response_source_line`
- `replay_lifecycle_semantics_source_line`
- `account_inventory_source_line`
- `economics_fee_rebate_source_line`

It covers exactly seven execution gaps from `0610T005`:

- `fill_probability`
- `queue_priority`
- `post_only_reject_behavior`
- `cancel_fill_race`
- `fees_rebates_spread_capture`
- `inventory_lifecycle`
- `real_order_lifecycle`

All seven remain unproven in the current task.

## Gate Semantics

`0611T001` creates a gate, not an implementation.

Allowed gate statuses are:

- `contract_accepted_design_only`
- `eligible_for_later_scoped_implementation_task`
- `requires_additional_design_before_implementation`
- `blocked_fail_closed`
- `forbidden_current_task`

Even when a row is labeled `eligible_for_later_scoped_implementation_task`, that only means a later task can be drafted with separate scope, acceptance criteria, and QA. It does not authorize implementation in this task.

The gate must not use statuses such as implemented, proof available, live ready, default-on ready, tiny-live ready, deployable, or promotable.

## Source Dependency Boundary

The synthesis keeps evidence authorities separate:

- Private order/fill observations are future private/order response context only.
- Replay lifecycle observations are supporting regression and future timestamp/order context only.
- Account/inventory records are future inventory and reconciliation context only.
- Economics settlement records are future fees/rebates/spread-capture context only after arithmetic, maker/taker classification, conversion, and timestamp validation.
- Public markout context is future context only, not execution proof or PnL proof.
- Future endpoint and collector responsibilities are out of scope for this task.

Contradictions across these authorities must fail closed until a later separately accepted reconciliation policy exists.

## Implementation-Readiness Boundary

The next implementation sequence may be described only as future scoped work. A future implementation task must carry:

- Its own task file and explicit files scope.
- Separate source permission boundary.
- Separate endpoint/data-use statement.
- Separate validation oracle.
- Separate overclaim rejection rules.
- QA before any downstream use.

No future sequence row may be used as a strategy signal, shadow decision, case-library condition, parameter-search objective, live-decision criterion, deployment criterion, or promotion criterion.

## Forbidden Overclaim Boundary

Current artifacts do not prove:

- fill probability
- queue priority
- post-only reject behavior
- cancel-fill race
- fees/rebates/spread capture
- inventory lifecycle
- real order lifecycle
- PnL
- maker execution viability
- live/default-on/tiny-live readiness
- deployment readiness
- promotion readiness

These claims remain rejected until separately accepted evidence exists.

## Allowed Future Use

If accepted by QA, this synthesis may guide which future task files should be drafted. It may not be used to run or implement an endpoint, source reader, collector, runner, metric, strategy, live behavior, deployment, or promotion.

Final recommendation: `source_line_synthesis_gate_ready_for_qa`.
