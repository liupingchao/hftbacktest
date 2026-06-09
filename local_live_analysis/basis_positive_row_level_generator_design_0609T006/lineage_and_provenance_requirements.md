# Lineage And Provenance Requirements

Task: `0609T006`

This is a design-only requirement document for a possible future row-level read-only artifact generator. It does not implement a generator and does not generate row-level case entries, case catalogs, source-row catalogs, shadow decisions, executable triggers, or trading instructions.

## Required Lineage Chain

A later generator must preserve this accepted chain:

1. `0609T003` evidence and row-level input policy from `filtered_context_viability_manifest.json`.
2. `0609T004` field taxonomy, label taxonomy, acceptance gate, reject conditions, and execution-gap map.
3. `0609T005` schema contract, validator requirements, row-level prerequisite register, and execution-gap boundary register.
4. `0609T006` generator design contract, input allowlist, output schema contract, leakage guards, no-action guards, acceptance gate, and reject conditions.

All four tasks must have QA status `已通过` before any later implementation discussion.

## Source Manifest Rules

- Every source must appear in `input_manifest_allowlist.csv`.
- Every source must be local, read-only, and tied to a QA-accepted public/canonical observation-layer artifact.
- A later generator must fail closed if the source manifest is missing, mutable, outside the allowlist, or not tied to a QA-accepted task.
- A later generator must not discover new source files by directory walking except inside an explicitly allowlisted artifact class.
- A later generator must not read remote paths, private/account/order endpoint outputs, live bot logs, user streams, account state, positions, signing material, nonce handling, production configs, or deployment artifacts.

## Future Row Provenance Columns

A future row-level read-only artifact must include immutable provenance fields before any research context fields:

- artifact schema version
- future implementation task id
- source task id
- source artifact path
- source sample id
- source row reference
- source observation timestamp
- source QA status
- T004 field taxonomy version
- T005 schema version
- T006 generator design version

These columns prove traceability only. They must not become action timing, order timing, live scheduling, strategy gating, or deployment instructions.

## Determinism Requirements

A later implementation task must define deterministic row ordering and stable row ids, but row ids must not encode action, side, quote price, quote size, leverage, stop/take-profit, submit, cancel, fill, or shadow-decision information.

## Execution Gap Preservation

Every future row must carry or inherit execution-gap markers showing that fill probability, queue/queue-ahead, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, and real order lifecycle remain unproven.

No generated row may claim maker execution viability. Any execution-layer evidence task must be separately dispatched and QA-accepted.
