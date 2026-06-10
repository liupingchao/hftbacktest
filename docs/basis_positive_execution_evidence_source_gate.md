# Basis-Positive Execution Evidence Source Gate

Task: `0610T003`

## Scope

This document is a source availability and runner implementation gate only. It translates the QA-passed `0610T002` read-only execution-evidence runner contract into a current-source readiness decision for a later separately scoped runner task.

It does not implement a runner. It does not read private/order/account/live data. It does not authorize case-library implementation, source-row case catalogs, shadow decisions, executable triggers, trading instructions, order side, quote price/size, strategy behavior, live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Input Fact Source

- `0610T002` QA status: `已通过`.
- `0610T002` final recommendation: `read_only_execution_evidence_runner_design_ready`.
- `0610T002` source policy remains binding:
  - T010/T011 public proxy artifacts are design context only, not execution proof.
  - Replay/simulation artifacts are `supporting_regression_not_execution_proof`.
  - Private/order response artifacts are `forbidden_current_task / future_requires_separate_design`.
  - Account/inventory artifacts are `forbidden_current_task / future_requires_separate_design`.

## Current Gate Decision

Current local artifacts are sufficient to define and later validate a fail-closed read-only runner skeleton. They are not sufficient to compute execution-proof metrics for any of the seven execution gaps.

Allowed next implementation shape:

- A later separately dispatched task may implement a fail-closed skeleton only.
- The skeleton may parse accepted contract inputs, validate source policies, emit unavailable/proof-limited status rows, and reject overclaims.
- The skeleton must not emit action-capable fields, source-row cases, shadow decisions, strategy signals, live gates, deployment fields, promotion fields, or execution viability proof fields.

Blocked next implementation shapes:

- No actual fill-probability metric can be claimed until accepted fill/outcome labels or response-source design exists.
- No queue/priority proof can be claimed from current public proxy or replay artifacts.
- No post-only reject metric can be claimed without separate private/order response source design.
- No cancel-fill race or real order lifecycle metric can be claimed without separate lifecycle source semantics.
- No fee/rebate/spread-capture or PnL metric can be claimed without accepted economics source design.
- No inventory lifecycle metric can be claimed without separate account/inventory source design.

## Final Recommendation

Final recommendation: `runner_skeleton_ready_with_fail_closed_sources`.

This means only that a later separately scoped task may implement a fail-closed/read-only skeleton that preserves source-policy checks and overclaim rejection. It does not authorize execution metric claims, runner implementation inside `0610T003`, private/order endpoint use, account/inventory access, live/default-on/tiny-live behavior, strategy behavior, case-library implementation, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
