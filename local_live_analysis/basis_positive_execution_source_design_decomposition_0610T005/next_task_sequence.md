# 0610T005 Next Task Sequence

Final recommendation: `private_order_source_design_ready_next`

This recommendation authorizes only a later separately dispatched design-only task for the `private_order_response_source_line`. It does not authorize private/order endpoint implementation, signing, nonce handling, user stream, collector implementation, runner implementation, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

Recommended sequence:

1. `private_order_response_source_line` design-only contract.
   - Scope: response artifact schema, label taxonomy, timestamp policy, terminal-state consistency, no endpoint.
   - Primary gaps: `fill_probability`, `post_only_reject_behavior`, `real_order_lifecycle`.
2. `replay_lifecycle_semantics_source_line` design-only contract.
   - Scope: queue semantics, replay/live proof-limit policy, cancel/fill race event ordering.
   - Primary gaps: `queue_priority`, `cancel_fill_race`.
3. `account_inventory_source_line` design-only contract.
   - Scope: inventory state artifact contract, transition validation, conservation checks.
   - Primary gap: `inventory_lifecycle`.
4. `economics_fee_rebate_source_line` design-only contract.
   - Scope: fee/rebate/settlement artifact contract, maker/taker classification, currency/tick conversion, PnL boundary.
   - Primary gap: `fees_rebates_spread_capture`.

Do not skip the design-only contracts and do not turn any source line into endpoint implementation until a later task explicitly dispatches and QA accepts that scope.
