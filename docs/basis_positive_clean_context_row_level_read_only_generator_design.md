# Basis-Positive Clean Context Row-Level Read-Only Generator Design

Task: `0609T006`

This document defines a design-only contract for a possible future row-level read-only artifact generator for `basis_positive_clean_context`. It does not implement a generator, produce row-level case entries, produce a case catalog, generate source-row catalogs, create shadow decisions, define executable triggers, output order side, quote price, quote size, leverage, stop/take-profit rules, deployment instructions, or authorize private/order/live behavior.

## Source Boundary

The generator design is downstream of accepted read-only design artifacts:

- T003 evidence: `local_live_analysis/basis_positive_filtered_context_viability_0609T003/`
- T004 case-design contract: `docs/basis_positive_clean_context_case_design.md`
- T005 case-library schema contract: `docs/basis_positive_clean_context_read_only_case_library_schema.md`
- T005 QA result: `.workflow/reports/0609T005-qa.md`

T005 final recommendation is `read_only_case_library_schema_ready`. That recommendation allows this design discussion only. It does not authorize generator implementation or row generation.

## Design Goal

The future generator, if separately dispatched and QA-accepted, would create a read-only row-level research artifact that preserves:

- immutable lineage to accepted public/canonical source artifacts
- T004 field taxonomy
- future-label output-only separation
- no-action-field proof
- no shadow-decision sections
- execution-gap boundaries from T003/T004/T005

T006 only defines the contract a later implementation must satisfy.

## Allowed Future Input Manifests

A later generator may read only source manifests explicitly allowlisted in `input_manifest_allowlist.csv`.

Allowed input classes are:

- accepted T003 filtered-context aggregate and manifest artifacts
- accepted T004 field and label contracts
- accepted T005 schema, validator requirement, acceptance gate, reject condition, row-level prerequisite, and execution-gap boundary artifacts
- accepted upstream public/canonical pricing-signal row references named by T003 lineage

The future generator must reject ad hoc local files, remote paths, private/account/order endpoint outputs, live bot logs, production configs, account state, positions, user streams, signing material, nonce handling, and any source not tied to a QA-accepted task.

## Proposed Future Output Schema

The proposed schema is recorded in `proposed_row_level_output_schema.csv`. It defines design-level columns only. It is not a row-level artifact and contains no real source rows.

A future row-level artifact may contain only these categories:

- `row_identity`: immutable research row ids and source row references
- `lineage`: source task, artifact, sample, timestamp, and row provenance fields
- `decision_time_visible_context`: T004-approved context fields only
- `diagnostic_context`: T004-approved diagnostic fields only
- `read_only_label`: T004 read-only design labels
- `future_label_for_research_only`: future labels kept strictly as offline outputs
- `execution_gap_reference`: unproven execution-layer gap markers
- `validation_trace`: validator version, schema version, and pass/fail trace fields for artifact QA

The schema forbids order side, quote price, quote size, leverage, stop/take-profit, submit, cancel, fill, executable trigger, shadow side, shadow action, strategy config, private/order endpoint, live/default-on/tiny-live, parameter search, deployment recommendation, and promotion fields.

## Leakage Guard

Future labels must remain output-only. A later generator must fail validation if any future label appears in:

- input feature sections
- row filters
- case conditions
- executable trigger fields
- shadow decision fields
- live decision fields
- strategy or deployment recommendations

The required checks are listed in `future_label_leakage_guard_requirements.csv`.

## No-Action Guard

A later generator must prove that no action-capable field exists in the output schema or output rows. This includes direct fields, aliases, derived fields, free-text instructions, and deployment claims. The required checks are listed in `no_action_field_guard_requirements.csv`.

## Acceptance Gate

Before any later implementation discussion:

1. `0609T003`, `0609T004`, `0609T005`, and `0609T006` must pass QA.
2. A separate task file must explicitly authorize generator implementation.
3. That later task must keep the generator local, read-only, public-observation-layer only, and default-off from any live or strategy path.
4. The implementation task must run schema validation before row generation.
5. QA must review the generated schema, lineage checks, future-label leakage proof, no-action-field proof, and execution-gap boundary preservation.

The full gate is recorded in `generator_acceptance_gate.md`.

## Reject Conditions

The generator design direction must be rejected if a proposal:

- implements generator code inside T006
- generates row-level case entries or case catalogs inside T006
- creates shadow decisions or executable triggers
- introduces order side, quote price, quote size, leverage, stop/take-profit, submit, cancel, fill, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion fields
- uses future labels as inputs, filters, case conditions, triggers, shadow-decision fields, or live decision fields
- overclaims execution-layer maker viability or execution-layer proof
- weakens T004/T005 field taxonomy, gates, or reject conditions

The full list is recorded in `generator_reject_conditions.csv`.

## Execution Gap Boundary

Execution-layer maker viability remains unproven. The future generator design does not close these gaps:

- fill probability
- queue position or queue-ahead
- post-only reject behavior
- cancel-fill race behavior
- fees, rebates, or spread capture
- inventory lifecycle
- real order lifecycle

Any future task that claims to address these gaps must be separately dispatched, must define its own verification, and must pass QA before changing the interpretation of `basis_positive_clean_context`.

## Final Recommendation

`row_level_read_only_generator_design_ready`

This means the generator design contract is ready for QA as a read-only design artifact. It does not authorize generator implementation, row generation, case-library implementation, row-level case entries, source-row case catalog generation, shadow decisions, strategy implementation, private/account/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.
