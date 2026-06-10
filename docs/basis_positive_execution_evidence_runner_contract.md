# Basis-Positive Read-Only Execution Evidence Runner Contract

Task: `0610T002`

## Scope

This document is a runner contract/design only. It defines a future read-only execution-evidence runner interface, source policy, output schema, gap-to-metric mapping, validation plan, and fail-closed/overclaim rules.

It does not implement a runner. It does not read private/order/account/live data. It does not authorize case-library implementation, source-row case catalogs, shadow decisions, executable triggers, trading instructions, order side, quote price/size, strategy behavior, live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Source Policy

- `t010_t011_public_proxy_artifacts`: design context only, not execution proof.
- `replay_simulation_artifacts`: `supporting_regression_not_execution_proof`.
- `private_order_response_artifacts`: `forbidden_current_task / future_requires_separate_design`.
- `account_inventory_artifacts`: `forbidden_current_task / future_requires_separate_design`.
- `live_default_on_tiny_live_artifacts`: forbidden in this contract path.

## Future Runner Shape

A later separately scoped task may use this contract to implement a read-only runner only after QA/controller approval. This contract does not state implementation readiness.

The future runner must:

- fail closed unless all prerequisite QA reports and source manifests pass;
- read only allowlisted source artifacts from a separately accepted source contract;
- preserve future-label isolation;
- emit aggregate read-only metrics and validation artifacts only;
- never emit action-capable fields such as order side, quote price, quote size, submit/cancel/fill action, strategy signal, shadow decision, live gate, deployment fields, promotion fields, or execution-viability proof fields;
- report all execution-layer proof caveats explicitly.

## Required Execution Gaps

The future runner contract covers these seven gaps:

1. `fill_probability`
2. `queue_priority`
3. `post_only_reject_behavior`
4. `cancel_fill_race`
5. `fees_rebates_spread_capture`
6. `inventory_lifecycle`
7. `real_order_lifecycle`

## Final Recommendation

Final recommendation: `read_only_execution_evidence_runner_design_ready`.

This means only that the current runner contract/design artifacts are ready for QA/controller review. It does not indicate implementation readiness and does not authorize runner implementation, private/order endpoint use, account/inventory access, live/default-on/tiny-live behavior, strategy behavior, case-library implementation, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
