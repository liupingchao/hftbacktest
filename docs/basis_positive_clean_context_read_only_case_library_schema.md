# Basis-Positive Clean Context Read-Only Case-Library Schema

Task: `0609T005`

This document defines a read-only schema contract for a possible future `basis_positive_clean_context` case-library artifact. It is a schema/design artifact only. It does not implement a case library, generate row-level case entries, produce shadow decisions, define executable triggers, output order side, quote price, quote size, leverage, stop/take-profit rules, deployment instructions, or authorize private/order/live behavior.

## Source Contract

This schema inherits the accepted `0609T004` case-design contract:

- T004 design doc: `docs/basis_positive_clean_context_case_design.md`
- T004 artifacts: `local_live_analysis/basis_positive_clean_case_design_0609T004/`
- T004 QA result: `.workflow/reports/0609T004-qa.md`
- T004 final recommendation: `case_design_contract_ready_for_qa`

The evidence rationale comes from `0609T003`:

- Clean context rows: `3545`
- Clean context samples: `7`
- Mean future mid move: `43.39492243` ticks
- Wrong-way count: `38`
- p95 wrong-way loss: `86` ticks
- p95 wrong-way loss improvement versus raw basis-positive context: `34` ticks
- Max sample row share: `0.31480959`
- Negative-mean reversal across sample, horizon, and conditioning checks: `false`

These values justify only a read-only schema discussion. They do not prove maker execution-layer viability.

## Schema Scope

The minimum allowed schema is a document-level container with references, field contracts, label contracts, evidence snapshots, QA gates, reject conditions, and explicit execution-gap boundaries.

Allowed sections:

- `schema_metadata`: task id, schema version, source task references, generated-at timestamp, final recommendation, and boundary flags.
- `lineage_references`: pointers to accepted T003/T004/T002 artifacts and QA reports.
- `field_contract_reference`: inherited T004 field taxonomy and allowed/forbidden use by field.
- `read_only_label_contract`: inherited T004 read-only labels and definitions.
- `evidence_snapshot`: aggregate evidence metrics only, not row-level case rows.
- `future_label_research_output`: future outcome fields kept only as offline research outputs.
- `validator_requirement_reference`: design-only checks a later validator would need.
- `acceptance_gate_reference`: schema-level QA gates for any later discussion.
- `reject_condition_reference`: conditions that reject the schema direction or any later proposal.
- `execution_gap_reference`: explicit unproven execution-layer gaps.

Forbidden sections:

- `row_level_case_entries`
- `case_catalog`
- `shadow_decisions`
- `executable_triggers`
- `order_side_output`
- `quote_price_output`
- `quote_size_output`
- `live_instruction`
- `strategy_or_order_lifecycle_config`
- `parameter_search_result`
- `promotion_or_deployment_recommendation`

## Field Inheritance

T005 does not create new trading fields. It inherits T004 categories:

- `decision_time_visible_context`: may appear as read-only context metadata and design rationale.
- `diagnostic_context`: may appear only in diagnostics and gate summaries.
- `future_label_for_research_only`: may appear only as offline research outputs, never as inputs, triggers, case conditions, shadow-decision fields, or live decision fields.
- `execution_gap_reference`: may appear only to mark what remains unproven.

Any future row-level read-only artifact proposal must preserve this field taxonomy and must prove, before generation, that future labels cannot be joined back into decision-time input sections.

## Validator Requirements

T005 does not implement validator code. A later separately dispatched validator discussion or implementation would need to reject:

- executable trigger fields or language
- row-level case entries generated inside a schema-only task
- shadow decision fields or outputs
- order side, quote price, quote size, leverage, stop, or take-profit fields
- private/account/order endpoint references
- strategy implementation or live/default-on/tiny-live claims
- parameter search, promotion, deployment, or production recommendation claims
- future labels used as inputs, conditions, filters, triggers, or shadow-decision fields
- any execution-layer proof claim not backed by a separate QA-accepted task

Validator requirements are recorded in `local_live_analysis/basis_positive_case_library_schema_design_0609T005/validator_requirements.csv`.

## Acceptance Gate

The schema direction is acceptable only if:

1. `0609T003`, `0609T004`, and `0609T005` pass QA before any later row-level read-only artifact generator discussion.
2. T004 clean-context gates remain true: at least `7` samples, max sample row share below `0.40`, positive p95 wrong-way loss improvement, and no sample/horizon/conditioning negative-mean reversal.
3. The schema remains non-executable and cannot express order side, quote price, quote size, leverage, stop/take-profit, strategy behavior, private/order endpoint usage, live behavior, shadow decisions, parameter search, promotion, or deployment recommendations.
4. Future-label fields remain research outputs only.
5. Execution-layer gaps remain explicitly unproven unless a separate task is dispatched and passes QA.

The full gate is recorded in `schema_acceptance_gate.md`.

## Row-Level Artifact Prerequisites

T005 does not authorize row-level artifact generation. A future discussion may define a row-level read-only generator only after a separate task file and QA path exist. That future task would need:

- explicit source manifest allowlist
- immutable provenance references for each source row
- field-level category checks inherited from T004
- proof that future-label fields are output-only
- proof that no executable action fields exist
- empty or forbidden sections for shadow decisions and order behavior
- QA review of generated schema before any rows are produced

These are prerequisites, not approval.

## Execution Gap Boundary

The schema cannot close execution-layer gaps. The following remain unproven:

- fill probability
- queue position or queue-ahead
- post-only reject behavior
- cancel-fill race behavior
- fees, rebates, or spread capture
- inventory lifecycle
- real order lifecycle

Current evidence is public observation-layer research. It does not prove execution-layer maker viability.

## Final Recommendation

`read_only_case_library_schema_ready`

This means the schema/design contract is ready for QA as a read-only design artifact. It does not authorize case-library implementation, row-level case entries, shadow decisions, strategy implementation, private/account/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.
