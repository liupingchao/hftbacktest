# Basis-Positive Execution Source Design Decomposition

Task: `0610T005`

This document decomposes the seven basis-positive execution-evidence gaps into source-design lines. It is design-only. It does not implement source readers, collectors, runners, private/order/account/live endpoints, user streams, signing, nonce handling, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Prerequisite

`0610T004` QA is accepted with final recommendation `fail_closed_runner_skeleton_ready_for_qa`. That means the fail-closed/read-only skeleton is ready for QA review and emits only unavailable/proof-limited rows. It does not make any current source ready for execution-proof metrics.

## Split Gates

Execution gaps are grouped into one source-design line only when all six dimensions match:

- `truth_authority`: the source that can authoritatively support the label.
- `label_unit`: the event, transition, snapshot, or settlement unit that carries the label.
- `causal_time_semantics`: whether evidence is decision-time visible, exchange event-time, local receive-time, replay time, or settlement time.
- `permission_boundary`: public/local, private/order, account/inventory, or economics/fee source boundary.
- `validation_oracle`: the validation mechanism that can fail closed.
- `overclaim_failure_mode`: the specific proof claim that must be rejected.

If any dimension differs, the source line remains separate. Secondary dependencies are recorded only as future design dependencies, never as current proof sources.

## Source Lines

### `private_order_response_source_line`

Primary gaps:

- `fill_probability`
- `post_only_reject_behavior`
- `real_order_lifecycle`

Truth authority is an accepted private/order response artifact contract, not a live endpoint. This line can later define response labels, reject-code taxonomy, order lifecycle state labels, and terminal-state consistency. It must not authorize endpoint use, signing, nonce handling, user stream, collector implementation, live/default-on/tiny-live, or strategy behavior.

### `replay_lifecycle_semantics_source_line`

Primary gaps:

- `queue_priority`
- `cancel_fill_race`

Truth authority is accepted replay/live lifecycle semantics and timestamp policy, not private/order response alone. Replay remains supporting regression until a later accepted semantics design defines proof limits and validation gates. This line must reject exact queue position proof and cancel-fill race proof from current artifacts.

### `account_inventory_source_line`

Primary gap:

- `inventory_lifecycle`

Truth authority is an accepted account/inventory state artifact contract. Order fills alone are not inventory lifecycle proof. This line must define inventory state source, transition validation, conservation checks, and reconciliation boundaries before any implementation can be considered.

### `economics_fee_rebate_source_line`

Primary gap:

- `fees_rebates_spread_capture`

Truth authority is an accepted economics, fee, rebate, or settlement artifact contract. Fill price/quantity and hypothetical spread are not realized economics or PnL proof. This line may depend on private/order fill labels later, but it remains separate because its validation oracle is fee/rebate arithmetic and currency/tick conversion rather than response taxonomy.

## Recommended Sequence

1. `private_order_response_source_line` design-only contract.
2. `replay_lifecycle_semantics_source_line` design-only contract.
3. `account_inventory_source_line` design-only contract.
4. `economics_fee_rebate_source_line` design-only contract.

The first recommendation is `private_order_source_design_ready_next` because it unlocks the largest number of primary gaps for later design work. This recommendation means only that a later separately dispatched design-only task can define the private/order response artifact contract. It does not authorize private/order endpoint implementation or any execution-proof claim.

## Boundary

Current T010/T011/0610T001/0610T002/0610T003/0610T004 artifacts remain design context or fail-closed runner artifacts only. They do not prove fill probability, exact queue position, exchange post-only reject behavior, cancel-fill race, realized fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion.
