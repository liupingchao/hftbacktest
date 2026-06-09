# Generator Acceptance Gate

Task: `0609T006`

This gate is design-only. It does not authorize generator implementation, row-level case entries, case catalogs, source-row catalogs, shadow decisions, executable triggers, strategy implementation, private/account/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.

## Preconditions For Later Implementation Discussion

All must be true before a later generator implementation task can be discussed:

1. `0609T003` QA status is `已通过`.
2. `0609T004` QA status is `已通过`.
3. `0609T005` QA status is `已通过` with final recommendation `read_only_case_library_schema_ready`.
4. `0609T006` QA status is `已通过` with final recommendation `row_level_read_only_generator_design_ready`.
5. A new separate task file explicitly scopes a generator implementation.
6. That task must state that implementation is local, read-only, public-observation-layer only, and not connected to live or strategy paths.
7. That task must define verification before row generation.

## Required Later Implementation Gates

A future generator implementation must fail closed unless it proves:

- every source appears in `input_manifest_allowlist.csv`
- every source task has QA status `已通过`
- all output columns match `proposed_row_level_output_schema.csv`
- future-label fields are output-only
- no action-capable fields exist
- no shadow-decision fields exist
- no private/account/order/live/default-on/tiny-live/deployment/promotion fields exist
- execution-layer maker viability remains unproven
- generated rows, if ever authorized later, are read-only research rows and not executable instructions

## Required Later QA Evidence

A future QA thread must receive:

- implementation task file
- implementation business report
- source manifest allowlist validation output
- schema validation output
- future-label leakage guard output
- no-action-field guard output
- lineage/provenance validation output
- execution-gap boundary validation output
- explicit statement that no strategy/private/order/live/default-on/tiny-live/parameter search/promotion behavior is authorized

## Current T006 Acceptance Line

T006 passes business execution only if it produces the design contract, manifest, input allowlist, proposed output schema, lineage/provenance requirements, future-label leakage guard requirements, no-action-field guard requirements, acceptance gate, reject conditions, and business report without implementing a generator or generating rows.

Execution-layer maker viability remains unproven.
