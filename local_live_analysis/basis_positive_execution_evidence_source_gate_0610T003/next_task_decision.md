# 0610T003 Next Task Decision

Final recommendation: `runner_skeleton_ready_with_fail_closed_sources`.

## Meaning

A later separately dispatched task may implement a fail-closed read-only skeleton over the accepted `0610T002` contract and `0610T003` source gate. The skeleton may validate prerequisites, source policies, output schema restrictions, gap coverage, and overclaim rejection. It may emit unavailable/proof-limited status rows for the seven gaps.

## Not Authorized

This recommendation does not authorize:

- execution metric proof claims;
- private/account/order endpoint use;
- account inventory access;
- live/default-on/tiny-live behavior;
- strategy behavior;
- order side, quote price, quote size, submit/cancel/fill action fields;
- case-library implementation;
- source-row case catalogs;
- shadow decisions;
- parameter search;
- deployment recommendation;
- promotion;
- execution-layer maker viability proof.

## Required Later Source Designs

- `fill_probability`: private/order response or accepted fill-label source design.
- `queue_priority`: queue/priority source semantics and replay/live proof-limit design.
- `post_only_reject_behavior`: private/order response source and reject-code taxonomy design.
- `cancel_fill_race`: lifecycle source semantics and terminal-state policy design.
- `fees_rebates_spread_capture`: economics source design covering fees, rebates, spread capture, and currency conversion.
- `inventory_lifecycle`: account/inventory source and transition validation design.
- `real_order_lifecycle`: private/order lifecycle source and reconciliation policy design.
