# Basis-Positive Execution Evidence Requirements

Task: `0610T001`

## Scope

This document is a design-only requirements contract. It translates the QA-accepted `0609T011` read-only proxy synthesis into minimum evidence requirements for any later execution-evidence contract/design task.

It does not implement a runner, case library, source-row case catalog, shadow decisions, executable triggers, trading instructions, strategy behavior, private/order endpoint use, live/default-on/tiny-live behavior, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.

## Source Facts

- `0609T011` QA status: `已通过`.
- `0609T011` final recommendation: `continue_to_execution_evidence_design`.
- T011 artifacts remain read-only proxy synthesis only.
- T010/T011 proxy evidence must not be interpreted as fill probability, exact queue position, exchange post-only reject behavior, cancel-fill race, realized fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion proof.

## Required Evidence Gaps

Any later task that attempts execution evidence must explicitly cover these seven gaps:

1. `fill_probability`
2. `queue_priority`
3. `post_only_reject_behavior`
4. `cancel_fill_race`
5. `fees_rebates_spread_capture`
6. `inventory_lifecycle`
7. `real_order_lifecycle`

## Source Class Policy

- Public proxy artifacts from T010/T011 are allowed only as design context. They are not execution proof.
- Replay/simulation artifacts are `supporting_regression_not_execution_proof` unless a later separately scoped and QA-accepted task defines stronger semantics.
- Private/order response artifacts are `forbidden_current_task / future_requires_separate_design`.
- Account/inventory artifacts are `forbidden_current_task / future_requires_separate_design`.
- Live/default-on/tiny-live artifacts are forbidden for this path until separate QA-accepted prerequisites exist.

## Minimum Later Contract Requirements

A later read-only execution-evidence runner contract/design task must define:

- Source artifact allowlist and explicit forbidden source classes.
- Label schema and unit of analysis for each execution gap.
- Timestamp and join policy for any source-path evidence.
- Future-label isolation policy.
- No-action-field output policy.
- Per-gap fail-closed checks.
- Per-gap overclaim reject rules.
- Boundary validation proving no strategy/private/order/live/default-on/tiny-live authorization.

## Recommendation

Final recommendation: `execution_evidence_runner_contract_ready`.

This recommendation means only that a later separately scoped read-only runner contract/design task can be considered after QA. It does not authorize runner implementation, private/order endpoint use, strategy behavior, case-library or shadow decisions, live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.
