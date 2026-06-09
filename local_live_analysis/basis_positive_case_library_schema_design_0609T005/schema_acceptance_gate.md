# Schema Acceptance Gate

Task: `0609T005`

This gate applies only to the read-only schema/design contract for a possible future `basis_positive_clean_context` case-library artifact. Passing this gate does not authorize case-library implementation, row-level case entries, shadow decisions, executable triggers, private/account/order endpoints, order lifecycle, strategy implementation, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.

## Required Gates

1. `0609T003` QA must be `已通过`.
2. `0609T004` QA must be `已通过`.
3. `0609T005` QA must be `已通过` before any later row-level read-only artifact generator discussion.
4. T004 clean-context evidence gates must remain true:
   - clean context samples at least `7`
   - max sample row share below `0.40`
   - clean p95 wrong-way loss improvement versus raw basis-positive context is positive
   - no sample/horizon/conditioning negative-mean reversal
5. The schema must remain a non-executable design artifact.
6. The schema must not include or authorize row-level case entries, case catalogs, shadow decisions, order side, quote price, quote size, leverage, stop/take-profit, private/order endpoint behavior, order lifecycle logic, strategy implementation, live/default-on/tiny-live, parameter search, promotion, or deployment recommendation.
7. Future-label fields must remain research outputs only and must not be used as inputs, filters, triggers, case conditions, shadow-decision fields, or live decision fields.
8. Execution-layer gaps must remain explicitly unproven unless separately dispatched and QA-accepted.
9. Validator content must remain requirements/design-only in T005; executable validator implementation is outside this task.

## Current Evidence Snapshot

- Clean context rows: `3545`
- Clean context samples: `7`
- Clean context mean future move: `43.39492243` ticks
- Clean context p95 wrong-way loss: `86` ticks
- Clean context p95 wrong-way loss improvement versus raw: `34` ticks
- Clean context max sample row share: `0.31480959`
- Sample/horizon/conditioning negative-mean reversal: `false`

## Later Row-Level Discussion Gate

A future row-level read-only artifact generator discussion may be considered only after:

1. T003, T004, and T005 have passed QA.
2. A new task file explicitly scopes row-level read-only artifact generation.
3. The future task defines source manifest allowlists, no-executable-field checks, future-label leakage checks, and QA acceptance steps before any rows are generated.

This is a prerequisite list, not approval.
