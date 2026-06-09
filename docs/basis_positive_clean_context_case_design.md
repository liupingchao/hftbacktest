# Basis-Positive Clean Context Case Design

Task: `0609T004`

This document is a read-only design contract for `basis_positive_clean_context`. It is based on `0609T003` public observation-layer artifacts and does not implement a case library, generate shadow decisions, produce trading instructions, or authorize private/order/live behavior.

## Source Evidence

- T003 filtered-context artifacts: `local_live_analysis/basis_positive_filtered_context_viability_0609T003/`
- T003 QA result: `.workflow/reports/0609T003-qa.md`
- T003 final recommendation: `candidate_for_read_only_case_design`
- T003 clean context: `3545` rows across `7` samples
- T003 clean context p95 wrong-way loss improvement versus raw: `34` ticks
- T003 clean context max sample row share: `0.31480959`

## Clean Context Definition

`basis_positive_clean_context` is a read-only design label:

```text
context_basis_mid_ticks > 0
AND NOT (
  basis_magnitude_bucket = basis_positive_small
  OR hl_top5_imbalance_bucket = hl_top5_imbalance_negative_small
  OR hl_microprice_minus_mid_bucket = hl_microprice_minus_mid_negative_small
)
```

The definition is observation-layer research context only. It is not an executable trigger and must not be used as an order side, quote price, quote size, leverage rule, stop/take-profit rule, deployment recommendation, live instruction, or shadow decision.

## Field Taxonomy

Fields in this design must be classified as one of:

- `decision_time_visible_context`: derived only from decision-time public observations and allowed as context in later read-only design discussion.
- `diagnostic_context`: allowed only for analysis and reporting, not for trigger construction.
- `future_label_for_research_only`: outcome labels used only for offline evaluation; forbidden as inputs, triggers, or case conditions.
- `execution_gap_reference`: explicit markers of unproven execution-layer requirements.

The authoritative field list is `local_live_analysis/basis_positive_clean_case_design_0609T004/case_field_contract.csv`.

## Label Taxonomy

Every label in this contract is a `read_only_design_label`.

Allowed labels:

- `basis_positive_clean_context`
- `basis_positive_raw_context`
- `basis_positive_tail_risk_context`

Forbidden uses for all labels:

- row-level case-library entry generation
- shadow decision generation
- executable trading instruction
- actual order side
- quote price or quote size
- leverage
- stop rule or take-profit rule
- private/account/order endpoint behavior
- live/default-on/tiny-live/promotion

The authoritative label list is `local_live_analysis/basis_positive_clean_case_design_0609T004/case_label_contract.csv`.

## Acceptance Gates

A later separately dispatched read-only case-library design discussion may start only if all gates in `case_acceptance_gate.md` remain true:

- T003 and T004 QA have passed.
- Clean context remains represented across at least `7` samples.
- Max sample row share remains below `0.40`.
- Clean p95 wrong-way loss improvement versus raw remains positive.
- No sample, horizon, or conditioning negative-mean reversal is present.
- Execution-layer gaps remain explicitly unproven unless separately dispatched and QA-accepted.

## Reject Conditions

The direction must be rejected if any proposal attempts to turn this design into executable trading behavior, private/order endpoint usage, strategy implementation, shadow decisions, live/default-on/tiny-live, parameter search, promotion, or row-level case-library entries.

The authoritative rejection list is `local_live_analysis/basis_positive_clean_case_design_0609T004/case_reject_conditions.csv`.

## Execution Gap Boundary

The T003 execution gap register remains binding. Current public observation-layer evidence does not prove:

- fill probability
- queue position or queue-ahead
- post-only reject behavior
- cancel-fill race behavior
- fees, rebates, or spread capture
- inventory lifecycle
- real order lifecycle

Future evidence tasks may be designed to study these gaps, but T004 does not fill them.

## Final Recommendation

`case_design_contract_ready_for_qa`

This recommendation means only that the read-only design contract is complete enough for QA. It does not authorize strategy implementation, private/account/order endpoints, order lifecycle, case-library implementation, shadow decision generation, live/default-on/tiny-live, parameter search, deployment, or promotion.
