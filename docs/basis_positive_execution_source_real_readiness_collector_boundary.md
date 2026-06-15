# Basis-Positive Execution Source Real-Readiness / Collector Boundary

Task: `0615T001`

## Purpose

This design closes the local-only artifact skeleton phase for the basis-positive execution evidence chain. The accepted local validators now define the target artifact shapes for:

- `private_order_response_source_line`
- `replay_lifecycle_semantics_source_line`
- `account_inventory_source_line`
- `economics_fee_rebate_source_line`

The next useful work must move toward a real source-line implementation or a read-only collector boundary. This document does not authorize endpoint implementation, source collector implementation, runner consumption, strategy behavior, live behavior, deployment, promotion, or metric proof.

## Prerequisite Status

`0612T001` and `0611T004` have direct QA reports in the current workspace and are accepted. `0611T002` and `0611T003` QA reports are not present in this workspace snapshot, but their accepted final recommendations are recorded by downstream QA reports and tracking files. That is sufficient for this design task, but the missing QA files are recorded as a repository fact-source completeness caveat.

## Source-Line Readiness

The four local skeletons are ready as validator targets, not as proof sources. The real-readiness result is:

- Private order response: best next non-local step. A read-only private order response artifact collector boundary can be designed first, with no order placement and no strategy action.
- Replay lifecycle: should consume real order-response artifacts later for reconciliation. It is not enough by itself to prove queue priority or cancel-fill race.
- Account inventory: must remain account-state authority only. Order fills alone still cannot prove inventory lifecycle.
- Economics fee/rebate: requires settlement authority, maker/taker classification, conversion policy, and reconciliation to private fills and account state before any economics metric can be considered.

## Permission Boundary

The next non-local task may only be a separately scoped read-only source-line task. It may define endpoint contracts and permission requirements, but implementation must remain no-trading, no-strategy, no-parameter-search, no-deployment, and no-promotion unless a later task explicitly authorizes otherwise.

Credentials, signing, nonce handling, user streams, account identifiers, and remote private data are not inputs to this task. A future collector task must define redaction, local artifact storage, clock/timestamp policy, and fail-closed behavior before any real data is consumed by a runner.

## Runner Consumption Gate

No execution-evidence runner may consume real source-line artifacts until all required source-line validators pass, provenance is recorded, timestamp domains remain separated, cross-source reconciliation is explicit, and forbidden overclaim checks remain fail-closed.

Runner consumption is separate from collection. Collection success does not imply metrics, PnL, maker viability, live readiness, deployment readiness, or promotion readiness.

## Convergence Policy

After `0615T001` QA, at most one additional local-only task may be dispatched. That exception is allowed only if `0615T001` names a concrete blocker. The only named local-only blocker is repository fact-source completeness: the current workspace lacks `.workflow/reports/0611T002-qa.md` and `.workflow/reports/0611T003-qa.md` even though downstream QA reports cite those tasks as accepted.

If total control chooses not to repair that record-completeness issue, or after one such repair task completes, the next formal execution-proof task must move to real source-line implementation or read-only collector work.

## Next Recommended Task

Recommended next non-local task:

`0615T002`: Basis-positive private order response read-only collector boundary / implementation design.

It should define the no-trading private order response source path, endpoint/permission contract, artifact schema handoff into `private_order_response_source.py`, redaction policy, timestamp/provenance policy, and QA gates. It must not place orders, cancel orders, modify strategy behavior, run live/default-on/tiny-live, compute metrics, or claim execution proof.

## Forbidden Interpretation

This task does not prove fill probability, post-only reject behavior, queue priority, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, maker execution viability, live readiness, deployment readiness, or promotion readiness.
