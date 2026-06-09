# Basis-Positive Clean Context Execution Evidence Contract

Task: `0609T009`

## Purpose

This contract defines the next read-only execution-evidence proxy runner for the `basis_positive_clean_context` row-level artifacts accepted in `0609T008`.

The source rows are observation-layer research rows only. They do not prove fill probability, queue position, post-only rejects, cancel-fill races, fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, live readiness, or maker execution viability.

## Accepted Source

The later runner may read only QA-accepted local public/canonical observation-layer artifacts:

- `local_live_analysis/basis_positive_row_level_generator_0609T008/row_level_generator_manifest.json`
- `local_live_analysis/basis_positive_row_level_generator_0609T008/row_level_read_only_cases.csv`
- `local_live_analysis/basis_positive_row_level_generator_0609T008/source_artifact_manifest.csv`
- `local_live_analysis/basis_positive_row_level_generator_0609T008/*_check.csv`
- inherited T006/T007 design and validator reports as boundary references only

The accepted T008 source has `3545` generated rows across `7` samples, with the label `basis_positive_clean_context` and primary horizon `1000ms`.

## Evidence Classes

Each execution question must be classified into exactly one proof class:

- `proxy_available_from_public_observation_rows`: can be estimated from decision-time public book/context rows, but remains a proxy.
- `proxy_available_with_strict_caveat`: can be partially estimated, but the output must carry a non-proof caveat.
- `not_provable_without_separate_execution_evidence`: cannot be proven without a later accepted execution-data task.
- `forbidden_for_current_research_stage`: must not be inferred or emitted by the proxy runner.

## Proxy Metric Contract

The later runner may design metrics for:

- public-book post-only feasibility proxy
- spread-capture / fee-rebate proxy under explicit configurable assumptions
- adverse move after hypothetical passive quote, using future labels only as output research labels
- touch/proximity opportunity proxy
- queue/priority proxy with explicit non-proof caveat

The later runner must not claim real fill probability, exact queue position, real post-only rejection behavior, cancel-fill race behavior, realized fee/rebate outcome, realized spread capture, real inventory lifecycle, real order lifecycle, realized PnL, or maker execution viability.

## Future Label Rule

Future labels may be output-only research labels. They may not be used as inputs, row filters, trigger conditions, case conditions, shadow-decision fields, live-decision fields, or deployment criteria.

## Later Runner Recommendation Taxonomy

The later runner may end only with:

- `read_only_proxy_runner_ready_for_implementation`
- `needs_more_proxy_contract_detail`
- `reject_proxy_runner_direction`

`read_only_proxy_runner_ready_for_implementation` means only that a separate read-only proxy runner implementation task may be dispatched. It does not authorize case-library implementation, shadow decisions, strategy implementation, private/account/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, deployment, promotion, or execution-layer proof.

## Required Fail-Closed Checks

The later runner must fail closed if it:

- reads non-allowlisted sources
- reads private/account/order/live artifacts
- emits action-capable fields
- emits order side, quote price, quote size, leverage, stop/take-profit, submit/cancel/fill, or order lifecycle fields
- emits case-library entries, source-row case catalogs, shadow decisions, executable triggers, live gates, or deployment recommendations
- uses future labels as inputs or filters
- weakens any T008 execution-gap marker
- claims execution-layer proof from public observation rows

## Boundary

This contract is design-only. It does not implement a proxy runner, generate proxy metrics, generate case-library entries, generate source-row case catalogs, produce shadow decisions, output executable trading instructions, change strategy behavior, use private/account/order endpoints, touch order lifecycle logic, run live/default-on/tiny-live, run parameter search, recommend deployment, claim promotion, or prove maker execution viability.
