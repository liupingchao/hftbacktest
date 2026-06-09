# Case Acceptance Gate

Task: `0609T004`

This gate is for a later separately dispatched read-only case-library design discussion. It does not authorize case-library implementation, shadow decisions, trading instructions, private/order endpoints, strategy changes, live/default-on/tiny-live, parameter search, or promotion.

## Required Gates

1. `0609T003` QA must be `已通过`.
2. `0609T004` QA must be `已通过`.
3. The clean context must remain represented across at least `7` samples.
4. The clean context max sample row share must remain below `0.40`.
5. Clean p95 wrong-way loss improvement versus raw basis-positive context must remain positive.
6. There must be no negative-mean reversal across sample, horizon, or conditioning checks.
7. Future-label fields must remain `future_label_for_research_only` and must not be used as inputs, triggers, case conditions, or live decisions.
8. Execution-layer gaps remain explicitly unproven unless separately dispatched and QA-accepted.

## Current T003 Evidence Snapshot

- Clean context rows: `3545`
- Clean context samples: `7`
- Clean context max sample row share: `0.31480959`
- Clean p95 wrong-way loss improvement versus raw: `34` ticks
- Sample/horizon/conditioning negative-mean reversal: `false`

## Boundary

Passing this gate only allows discussion of a later read-only design task. It does not permit row-level case entries, executable triggers, order side, quote price, quote size, leverage, stop/take-profit, live instruction, strategy implementation, private/account/order endpoint usage, parameter search, default-on, tiny-live, deployment, or promotion.
