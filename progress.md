# Progress

## 0611T001 Prepared Task

- `0611T001` has been created as the current next formal task after `0610T009` QA passed.
- Scope: design-only source-line synthesis / implementation-readiness gate over the accepted `0610T006` / `0610T007` / `0610T008` / `0610T009` source-line contracts and `0610T005` decomposition.
- Required coverage includes source-line contract registry, implementation-readiness gate matrix, source dependency reconciliation matrix, forbidden overclaim matrix, next-task sequence, manifest, boundary validation, design document, and business report.
- It may recommend future scoped implementation tasks, but it must not authorize endpoint/source reader/collector/runner implementation inside `0611T001`.
- It must preserve current proof rejection for all seven execution gaps, PnL, maker execution viability, live/default-on/tiny-live readiness, deployment, and promotion.
- It does not authorize endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real execution metrics, real economics metrics, PnL proof, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T009 QA Update

- `0610T009` business execution is complete and QA is `已通过`.
- Scope: design-only `economics_fee_rebate_source_line` contract.
- It may cover only `fees_rebates_spread_capture` as the primary gap.
- Required contract coverage includes economics/fee/rebate/spread-capture artifact schema, fee/rebate settlement taxonomy, spread-capture taxonomy, maker/taker classification policy, currency conversion / tick-value policy, settlement timestamp policy, reconciliation boundary, validation gates, and overclaim reject rules.
- Inputs are restricted to accepted local `0610T008` / `0610T007` / `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` design artifacts, manifests, and QA/business reports.
- `0610T006` private-order response artifacts may be used only as future fill dependency context, `0610T007` replay lifecycle artifacts only as future timestamp/order consistency context, and `0610T008` account inventory artifacts only as future reconciliation context.
- It must explicitly reject hypothetical spread, fill notional, order fills alone, public markout alone, account inventory alone, or replay lifecycle alone as proof of fees/rebates/spread capture or PnL.
- It does not authorize economics endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real economics metrics, real execution metrics, PnL proof, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Official artifacts: `local_live_analysis/basis_positive_economics_fee_rebate_source_line_contract_0610T009/`.
- Source-line contract: `docs/basis_positive_economics_fee_rebate_source_line_contract.md`.
- Final recommendation: `economics_fee_rebate_contract_ready_for_qa`, meaning only that the design contract is ready for QA/controller review.
- QA report: `.workflow/reports/0610T009-qa.md`.
- Latest QA acceptance document now records `0610T009` as the latest effective QA result.

## 0610T007 / 0610T008 QA Update

- `0610T007` and `0610T008` were executed in parallel as design-only contracts.
- `0610T007` business execution is complete and QA is `已通过`.
- `0610T007` official artifacts: `local_live_analysis/basis_positive_replay_lifecycle_semantics_source_line_contract_0610T007/`.
- `0610T007` source-line contract: `docs/basis_positive_replay_lifecycle_semantics_source_line_contract.md`.
- `0610T007` final recommendation: `replay_lifecycle_contract_ready_for_qa`.
- `0610T007` covers only `queue_priority` and `cancel_fill_race` as future design labels.
- `0610T007` commits: `139b76a` (`0610T007 replay lifecycle contract`) and `692f445` (`0610T007 business report metadata`).
- `0610T008` business execution is complete and QA is `已通过`.
- `0610T008` official artifacts: `local_live_analysis/basis_positive_account_inventory_source_line_contract_0610T008/`.
- `0610T008` source-line contract: `docs/basis_positive_account_inventory_source_line_contract.md`.
- `0610T008` final recommendation: `account_inventory_contract_ready_for_qa`.
- `0610T008` covers only `inventory_lifecycle` as a future design label and explicitly states that order fills alone cannot prove inventory lifecycle.
- `0610T008` commits: `66127b7` (`0610T008 account inventory contract`) and `156f6da` (`0610T008 business report metadata`).
- Parallel write-scope check passed: the business-thread commits for T007/T008 touched only their own task/report/doc/artifact paths and did not include shared tracking files.
- QA reports: `.workflow/reports/0610T007-qa.md` and `.workflow/reports/0610T008-qa.md`.
- Latest QA acceptance document now records `0610T008` as the latest effective QA result.
- Neither task authorizes source reader/collector implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T008 Prepared Task

- `0610T008` has been created as a prepared parallel-eligible design-only task.
- Scope: design-only `account_inventory_source_line` contract.
- It may cover only `inventory_lifecycle` as the primary gap.
- Required contract coverage includes account/inventory artifact schema, inventory snapshot / transition taxonomy, conservation checks, reconciliation boundary, fail-closed gates, and explicit rejection that order fills alone can prove inventory lifecycle.
- Inputs are restricted to accepted local `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` design artifacts, manifests, and QA/business reports.
- `0610T006` private order response artifacts may be used only as future transition input / future cross-check context, not current inventory lifecycle proof.
- It does not authorize account endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- It is parallel-eligible with `0610T007` because both are design-only contracts, their primary gaps and output paths are disjoint, and neither business thread may modify shared tracking files (`task_plan.md`, `progress.md`, `findings.md`, `docs/qa-acceptance-report.md`). Shared tracking must be updated later by total control in a single serial step.

## 0610T006 Business Update

- `0610T006` business execution is complete and QA is `已通过`.
- It created the design-only `private_order_response_source_line` contract after `0610T005` QA.
- Official artifacts: `local_live_analysis/basis_positive_private_order_response_source_line_contract_0610T006/`.
- Source-line contract: `docs/basis_positive_private_order_response_source_line_contract.md`.
- Final recommendation: `private_order_response_contract_ready_for_qa`.
- It covers only `fill_probability`, `post_only_reject_behavior`, and `real_order_lifecycle` as future design labels.
- It produced artifact schema, response/reject/lifecycle taxonomies, timestamp policy, validation gates, overclaim reject rules, manifest, boundary validation, and a business report.
- It does not authorize endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T007 Prepared Task

- `0610T007` has been created and dispatched as the next formal task.
- Scope: design-only `replay_lifecycle_semantics_source_line` contract after `0610T006` QA.
- It may cover only `queue_priority` and `cancel_fill_race` as future design labels.
- Inputs are restricted to accepted local `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` design artifacts, manifests, and QA/business reports.
- Required outputs are the replay lifecycle source-line design doc, replay lifecycle event schema, queue boundary matrix, cancel/fill race ordering policy, timestamp policy, replay/live proof-limit rules, validation gates, overclaim reject rules, manifest, boundary validation, and business report.
- It does not authorize replay/live semantic implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, exact queue position proof, cancel-fill race metric proof, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- It is parallel-eligible with `0610T008` because both are design-only contracts, their primary gaps and output paths are disjoint, and neither business thread may modify shared tracking files. Shared tracking must be updated later by total control in a single serial step.

## 0610T006 Prepared Task

- `0610T006` has been created, dispatched, executed, and accepted by QA.
- Scope: design-only `private_order_response_source_line` contract after `0610T005` QA.
- It may cover only `fill_probability`, `post_only_reject_behavior`, and `real_order_lifecycle` as future design labels.
- It must define response artifact schema, response/reject/lifecycle label taxonomy, timestamp policy, terminal-state consistency, fail-closed validation gates, and overclaim rejection rules.
- It does not authorize private/order endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T005 QA Update

- `0610T005` QA is `已通过`.
- It split the seven execution gaps into four source-design lines using truth authority, label unit, causal time semantics, permission boundary, validation oracle, and overclaim failure mode.
- Official artifacts: `local_live_analysis/basis_positive_execution_source_design_decomposition_0610T005/`.
- Source decomposition design: `docs/basis_positive_execution_source_design_decomposition.md`.
- Final recommendation: `private_order_source_design_ready_next`.
- This means only that a later separately dispatched design-only task may define the `private_order_response_source_line` contract.
- It does not authorize source implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T005 Business Update

- `0610T005` business execution is complete and QA is `已通过`.
- It split the seven execution gaps into source-design lines using truth authority, label unit, causal time semantics, permission boundary, validation oracle, and overclaim failure mode.
- Official artifacts: `local_live_analysis/basis_positive_execution_source_design_decomposition_0610T005/`.
- Source decomposition design: `docs/basis_positive_execution_source_design_decomposition.md`.
- Final recommendation: `private_order_source_design_ready_next`.
- It produced design docs, matrices, dependency graph, boundary validation, manifest, next-task sequence, and a business report.
- It does not authorize source implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T004 QA Update

- `0610T004` QA is `已通过`.
- It implemented a local fail-closed/read-only execution-evidence runner skeleton over accepted `0610T003`/`0610T002` contract and gate artifacts.
- Official artifacts: `local_live_analysis/basis_positive_execution_evidence_fail_closed_runner_0610T004/`.
- Final recommendation: `fail_closed_runner_skeleton_ready_for_qa`.
- It emits proof-limited unavailable status rows for all seven execution gaps under current sources.
- It does not authorize real execution metrics, private/order/account/live data use, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T003 Business Update

- `0610T003` business execution is complete and QA is `已通过`.
- It created the design/gate-only source availability and runner implementation gate over QA-passed `0610T002` artifacts.
- Required outputs are `docs/basis_positive_execution_evidence_source_gate.md` and `local_live_analysis/basis_positive_execution_evidence_source_gate_0610T003/`.
- Final recommendation: `runner_skeleton_ready_with_fail_closed_sources`.
- It classified all seven execution gaps as fail-closed placeholder only under current sources; actual execution-proof metrics remain blocked by private/order response, replay/lifecycle semantics, account/inventory, or economics source-design requirements.
- It does not authorize runner implementation, private/order, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T002 Business Update

- `0610T002` business execution is complete and QA is `已通过`.
- It created the design-only read-only basis-positive execution-evidence runner contract.
- Official artifacts: `local_live_analysis/basis_positive_execution_evidence_runner_contract_0610T002/`.
- Runner contract: `docs/basis_positive_execution_evidence_runner_contract.md`.
- Final recommendation: `read_only_execution_evidence_runner_design_ready`.
- This means only that the current runner contract/design artifacts are ready for QA/controller review. It does not indicate implementation readiness and does not authorize runner implementation, private/order, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T001 Business Update

- `0610T001` business execution is complete and QA is `已通过`.
- It created the design-only basis-positive execution-evidence requirements contract.
- Official artifacts: `local_live_analysis/basis_positive_execution_evidence_requirements_0610T001/`.
- Design contract: `docs/basis_positive_execution_evidence_requirements.md`.
- Final recommendation: `execution_evidence_runner_contract_ready`.
- This means only that a later separately scoped read-only runner contract/design task can be considered after QA. It does not authorize runner implementation, case-library implementation, shadow decisions, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0609T011 Business Update

- `0609T011` business execution is complete and is now `待验收`.
- It implemented the local read-only proxy evidence synthesis runner over QA-passed T010 artifacts.
- Official artifacts: `local_live_analysis/basis_positive_proxy_evidence_synthesis_0609T011/`.
- Final recommendation: `continue_to_execution_evidence_design`.
- This means only that a later separately scoped design task can define execution-layer evidence requirements. It does not authorize case-library implementation, source-row case catalogs, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0609T010 Prepared Task

- `0609T010` QA has passed.
- It implemented the local read-only maker-viability proxy runner over T008/T009 allowlisted artifacts.
- Official artifacts: `local_live_analysis/basis_positive_maker_viability_proxy_0609T010/`.
- Final recommendation: `read_only_proxy_evidence_ready_for_qa`.
- It does not authorize case-library implementation, source-row case catalogs, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.

## 0609T009 Prepared Task

- `0609T009` business execution is complete and is now `待验收`.
- It created the design/contract artifacts for `Basis-positive clean context execution-evidence gap planning / read-only maker-viability proxy contract`.
- Final recommendation: `read_only_proxy_runner_ready_for_implementation`.
- Scope remained design/contract only: translate `0609T008` row-level read-only artifacts into a later read-only maker-viability proxy runner contract.
- It does not authorize proxy runner implementation, case-library implementation, source-row case catalogs, shadow decisions, strategy/private/order/live/default-on/tiny-live, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.

## 0609T001 Business Update

- `0609T001` business execution is complete and is now `待验收`.
- New runner: `examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py`.
- Focused tests: `examples/hyperliquid/test_canonical_basis_positive_wrong_way_decomposition.py`.
- Official artifacts: `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/`.
- Final recommendation: `targeted_collection_ready`. This only means a later separately scoped collection task can be designed; T001 collected no new data and does not authorize strategy implementation, private/order endpoints, order lifecycle, case-library, shadow decisions, live/default-on/tiny-live, parameter search, or promotion.

## Current Focus

- Use `workflow-kit` and the local dashboard as the persistent development workflow for the hftbacktest Binance maker MM work.
- Latest accepted robustness source: `0604T003` has passed QA and makes event-mode + de-aliased future-row-delta evidence canonical while downgrading ordinary synthetic fixed-grid short-horizon alias evidence to diagnostic-only.
- Latest accepted workflow fact source: `0605T004` passed QA as the shutdown final proof no-order dry-run validation. It proves local fake no-network proof/log/audit observability only; it does not prove a real exchange shutdown run and does not authorize live/default-on/tiny-live or promotion.
- `0608T001` has reconciled workflow tracking so the controller no longer treats `0604T015` as the active task.
- Current implementation focus is no longer `0604T015`. The shutdown safety chain has moved through `0604T016` failed QA, `0605T001` partial-fill local proof repair, `0605T002` local-only proof diagnosis, `0605T003` exchange-reconciliation/final-proof hardening, and `0605T004` fake no-order dry-run validation.
- T003 now constrains the follow-up path: after QA, retain only a narrow Step 5C default-off / diagnostic-first quote-anchor safety layer. It should not repair audit_depth/bookTicker/top5 row-exact drift, promote top5 to hard anchor, redesign quote placement, change replay lifecycle, or start live.
- `0518T004` has completed business-thread execution and passed QA.
- `0519T001` QA passed; `0519T002` QA passed and closed Step 6 for roadmap progression.
- `0519T003` QA passed and closed Step 7 as a design-only inventory / execution model contract.
- `0519T004` completed Step 8 design-only quote-update / API-limit hygiene and passed QA.
- `0519T005` completed the narrow Step 8B read-only diagnostic / implementation-planning task and passed QA.
- `0519T006` passed QA as Step 8C default-off quote-update helper / instrumentation implementation.
- `0519T007` passed QA as the Step 9A design-only task.
- `0519T008` passed QA as the Step 9B default-off offline runner implementation task.
- `0519T009` passed QA as a narrow current-format live control collection plus T008 rerun task.
- `0519T010` passed QA as the planning-only Step 9C task.
- `0519T011` completed business-thread execution and passed QA.
- `0520T001` QA 已通过 as the read-only Step 9C multi-sample validation task.
- `0520T002` has been created as the narrow Step 9C runner/artifact hardening follow-on and is now `已通过`.
- `0521T001` has completed live collection, replay, and archive work and has passed QA.
- `0521T002` has completed the read-only Step 9C candidate x scenario bucket multi-sample determination task and passed QA.
- `0525T001` has completed business execution as the current Step 9D fine-bucket refinement task and passed QA.
- `0529T004` has completed the Hyperliquid read-only public market-data evidence hardening task and passed QA.
- `0529T005` has completed the Binance Stage 9L read-only rejection decomposition / bucket coarsening task and passed QA.
- `0530T001` has passed QA as a Hyperliquid design-only/read-only public market-data research consumer contract task.
- `0530T002` has passed QA after total controller explicitly ratified / accepted the already collected Stage 9M artifact.
- `0531T001` has passed QA as the Hyperliquid read-only consumer implementation task.
- `0531T002` has been created as the next Binance Stage 9N read-only evidence viability refinement task, is unblocked by `0530T002` QA, has completed business execution, and has passed QA.
- `0601T001` has passed QA as the Hyperliquid lag-venue public BTC sample collection and alignment task.
- `0602T001` has passed QA as the synchronized public-only Binance lead / Hyperliquid lag collection task.
- `0601T002` has passed QA as the read-only Binance lead / Hyperliquid lag as-of joined-feature input task.
- `0601T003` has passed QA as the read-only Binance-to-Hyperliquid lead-lag stability analyzer task.
- `0601T004` has passed QA as the Binance-led Hyperliquid maker data input contract.
- `0601T005` has passed QA as the Binance-led Hyperliquid read-only pricing-signal runner implementation.
- `0601T006` has passed QA as public-only collection / initial synthetic-grid aggregate evidence; its formal robustness interpretation is superseded by `0604T003` canonical event-mode artifacts.
- `0604T001`, `0604T002`, and `0604T003` have passed QA. Future Binance-led Hyperliquid pricing-signal robustness decisions should use the `0604T003` canonical event-mode artifacts.
- `0604T009` has passed QA and contributes one read-only Milestone 3 executability candidate regime: `regime_011_1000_spread_10_20_ticks`. It does not authorize maker action, strategy, live/default-on/tiny-live, parameter search, or promotion.
- `0604T015` / `0604T016` are historical shutdown-proof intermediate tasks. `0604T016` QA failed, and its defect chain was closed by `0605T001` / `0605T003` / `0605T004`.
- Latest completed milestones: `0515T003` QA 已通过，`0516T001` QA 已通过，`0516T002` QA 已通过，`0518T001` QA 已通过，`0518T002` QA 已通过，`0518T003` QA 已通过，`0518T004` QA 已通过。

## Current Status

- Workflow files: initializing.
- Active task: `0611T001`
- Active task status: `待执行`
- Parallel condition result: both business threads avoided shared tracking writes; total control has updated tracking after both business reports became available.
- Latest QA source of truth: `0610T009` (`已通过`)
- Latest business result awaiting QA: none
- Latest workflow housekeeping: `0608T001` (`已通过`, no QA)
- Prepared independent task: `0530T001` (`已通过`)
- Prepared Binance task: `0530T002` (`已通过`)
- Prepared Hyperliquid task: `0531T001` (`已通过`, unblocked by `0530T001` QA)
- Prepared Binance follow-up: `0531T002` (`已通过`, unblocked by `0530T002` QA)
- Prepared Hyperliquid lag sample: `0601T001` (`已通过`)
- Prepared synchronized cross-exchange sample: `0602T001` (`已通过`)
- Prepared cross-exchange join: `0601T002` (`已通过`)
- Prepared lead-lag stability analyzer: `0601T003` (`已通过`)
- Prepared read-only pricing-signal runner: `0601T005` (`已通过`, unblocked by `0601T004` QA)
- Prepared multi-sample robustness validation: `0601T006` (`已通过`, superseded for formal robustness interpretation by `0604T003`)
- Prepared horizon-alias canonicalization repair: `0604T003` (`已通过`, canonical event-mode robustness source)
- Prepared canonical evidence loader foundation: `0604T004` (`已通过`, shared foundation for subsequent Milestone 0 / Milestone 1 workers)
- Prepared canonical evidence source lock / guard hardening: `0604T005` (`已通过`, canonical source-lock guard foundation)
- Prepared canonical signal quality ranking: `0604T006` evidence is accepted for downstream use after `0604T008` repaired the report bucket-consistency QA defect.
- Prepared canonical horizon / regime diagnostics: `0604T007` (`已通过`, parallel Milestone 1 worker)
- Prepared T006 report consistency repair: `0604T008` (`已通过`)
- Prepared canonical signal + horizon/regime synthesis: `0604T009` (`已通过`, one read-only Milestone 3 executability candidate)
- Prepared Milestone 3 maker executability assessment: `0608T002` (`已通过`, final recommendation `reject_not_maker_executable`)
- Regime 011 directional momentum viability assessment: `0608T003` (`已通过`, final recommendation `reject_directional_edge_unstable`)
- Regime 011 feature-conditioned validity diagnosis: `0608T004` (`已通过`, final recommendation `watch_needs_contract_visibility_clarification`, no supported/watch-valid patterns)
- Regime 011 basis-context visibility / lineage diagnosis: `0608T005` (`已通过`, final contract decision `upgrade_to_context_only_supported` for `context_basis_mid_ticks > 0` as read-only context only)
- Basis-positive independent robustness diagnosis outside Regime 011: `0608T006` (`待验收`, final recommendation `needs_more_samples`, read-only only)
- Historical live shutdown bounded-wait fix: `0604T013` (`作废` as an open queue item; superseded by later proof-semantics tasks)
- Historical live shutdown wait-result ambiguity diagnosis: `0604T015` (`作废` as an open queue item; diagnosis consumed by later repair chain)
- Historical live shutdown cancel proof semantics fix: `0604T016` (`未通过`; defect repaired by `0605T001` and later proof hardening)
- Shutdown partial-fill local proof repair: `0605T001` (`已通过`)
- Shutdown local-only proof diagnosis: `0605T002` (`已通过`)
- Shutdown exchange-reconciliation/final-proof hardening: `0605T003` (`已通过`)
- Shutdown final proof no-order dry-run validation: `0605T004` (`已通过`, latest shutdown/live-safety QA fact source)
- Workflow current-state reconciliation: `0608T001` (`已通过`, no QA; tracking cleanup only)
- Current blocker: none.

## Next Task

- Controller state cleanup `0608T001` reconciles the stale queue created by `0604T013` / `0604T015` / `0604T016`. After this cleanup, do not dispatch from the old `0604T015` pending state.
- If continuing shutdown/live-safety work, the next task must be a separate, explicitly scoped real-environment proof design or no-order observation task. `0605T004` was fake/no-network only and does not authorize live/default-on/tiny-live/promotion.
- `0608T003` passed QA with `reject_directional_edge_unstable`. `0608T004` passed QA with no supported/watch-valid feature-conditioned case pattern. `0608T005` passed QA with `upgrade_to_context_only_supported` for `context_basis_mid_ticks > 0` as read-only context only. `0608T006` business execution is complete and awaiting QA with `needs_more_samples`: broad basis-positive evidence is positive but sample-concentrated and cost/tail-caveated. Regime 011 and basis-positive context still should not progress to maker/directional case-library, shadow decisions, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.
- `0526T007` now has a complete derived chain on `5-26-active-makeredge-control-180min-a`: T009 sidecar/join, Stage 5, Step 5C, Stage 6, Step 9B, and Step 9D are present. It remains a current-format no-rule/default-off control sample only.
- `0526T008` has passed QA as a 30min current-format no-rule/default-off control run with run id `5-26-active-minmove-control-30min-b`. It remains diagnostic/control data only and does not authorize candidate enablement, guard relaxation, parameter sweep, tiny live, default-on, or promotion.
- `0526T006` passed QA. Its output is a design-only `inventory_aware_quote_placement_request` contract and a later read-only runner contract; it did not implement strategy behavior, run sweep, start live, or make promotion claims.
- `0528T001` passed QA. It produced Stage 9I read-only inventory-aware quote-placement artifacts and a `reject` recommendation; it does not authorize strategy behavior changes, parameter search, live/default-on, guard relaxation, or promotion.
- `0527T001` passed QA. It found the `0526T008` bloat is primarily replay audit lifecycle export / order-state tracking repetition in audit replay mode, not a Stage 6 label-generation problem; repeated `cancel_ack` rows are redundant for Stage 6 after keeping first terminal lifecycle fact per order.
- `0528T002` passed QA. It implements compact replay lifecycle audit export for Stage 6 input, with terminal lifecycle rows de-duplicated inside the compact artifact by Stage 6 semantics. Bounded `0526T008` verification shows the 21GB full-prefix bloat path drops `250,000` scanned rows to `23,345` compact rows by skipping `226,655` duplicate terminal rows, and Stage 6 consumes `audit_bt_audit_replay.compact_lifecycle.csv` by contract. It does not authorize strategy, live, parameter, fill/cancel semantic, guard, default-on, or promotion changes.
- `0529T001` passed QA. It defines the next Binance maker policy direction as fill-quality-first and recommends a read-only fill-quality bucket synthesis runner before any new strategy policy implementation. It rejects continuing the fixed `0528T001` inventory skeleton, the current `0526T004` min-move grid, tiny-live/default-on/promotion, and more audit-bloat work unless a regression appears.
- `0529T002` passed QA. Stage 9K found `0` Shape A candidates and `0` Shape B candidates; clean-only decision-visible trigger buckets are mostly `reject_quality_negative` (`246`) or `needs_more_clean_fills` (`68`), so no policy-design follow-up is supported yet.
- `0529T004` passed QA. It collected a fresh 120s public-only Hyperliquid BTC sample, generated subscription/session/recovery evidence, ran the existing alignment runner, and classified the sample as `passes_pricing_research_market_view`. This remains public market-data-only evidence and does not authorize private connector, order lifecycle, strategy live logic, parameter search, default-on, tiny-live, or promotion.
- `0529T005` passed QA. It implemented the read-only Stage 9L runner, generated artifacts under `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/`, and produced final classification `needs_targeted_clean_fills`. Shape A / Shape B candidates remain `0`; churn-warning sensitivity alone creates no ready candidates. It does not authorize strategy implementation, live/default-on, parameter search, guard relaxation, tiny-live, or promotion.
- `0530T001` passed QA. It wrote `docs/hyperliquid_public_market_data_research_consumer_design.md`, uses `0529T004` as the fresh public-sample entry point, rechecked official Hyperliquid public docs successfully, and recommends only a later read-only local consumer implementation. QA rechecked the official public docs URLs and received HTTP 200. It continues to forbid private connector, order lifecycle, strategy live, parameter search, default-on, tiny-live, and promotion.
- `0530T002` passed QA after total controller ratified / accepted the already collected Stage 9M artifact. The data/artifact chain is reproducible: existing accepted current-format samples were scanned first; no existing sample could add top-gap evidence; one `120min` no-rule/default-off control sample was collected as `5-31-stage9m-cleanfill-control-120min-a`; maker acceptance and market-view passed; T009 join quality is clean; Stage 5 has `5742` submits / `106` fills; Stage 6 is `methodology_valid_single_sample`; and Stage 9K/9L were rerun. Aggregate clean-only Stage 9K fills increased `994 -> 1102`, but the top Stage 9L gap only improved `34 -> 36` fills and remains below threshold. Stage 9L final classification is still `needs_targeted_clean_fills`, coarsened ready bucket count is `0`, Shape A / Shape B candidate rows are `0`, and policy design remains blocked. Caveat: original fixed logs/reports did not prove pre-start approval; acceptance is based on current controller ratification.
- `0531T001` passed QA. It implements only a read-only local artifact consumer over `0529T004`, producing market-view time series, pricing features, quality summaries, and a recommendation markdown. It does not authorize fresh collection, private connector, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.
- `0531T002` is now prepared as the next Binance formal task and is unblocked by `0530T002` QA. It should read only existing Stage 9M/9K/9L artifacts, explain why the new sample added `+108` aggregate fills but only `+2` top-gap fills, estimate collection duration for the `36 -> 40` and `+20` top-gap targets, and recommend whether to stop, briefly continue for threshold-crossing only, or pivot to alternative read-only regime refinement. It does not authorize collection, policy design, strategy implementation, candidate enablement, guard relaxation, parameter search, tiny-live/default-on, or promotion.
- `0601T001` passed QA. Its Hyperliquid public-only lag-venue sample reached `passes_pricing_research_market_view` and remains venue-state / execution-context evidence only.
- `0602T001` passed QA. Its synchronized public-only sample has `1800.105s` overlap and can feed a later read-only `0601T002` join, but it does not establish a Binance-lead / Hyperliquid-lag statistical effect.
- `0601T002` passed QA. It generated read-only joined features over the synchronized public sample with `primary_usable_row_count=3596`, `future_join_count=0`, and trade pressure disabled.
- `0601T003` passed QA. It generated read-only lead-lag stability evidence with verdict counts `18 stable / 6 watch / 30 unstable`, and it explicitly records effective future age because the current Hyperliquid decision cadence is roughly 500ms.
- `0601T004` passed QA. The accepted next boundary is a later read-only pricing-signal runner only; primary Binance lead allowlist is `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty`. It does not authorize strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.
- `0601T005` passed QA. The read-only runner generated `21541` pricing signal rows, `4` feature quality rows, `30` horizon label summary rows, `540` feature/regime rows, and `54` venue-state conditioning rows from `3596` primary rows. Recommendation is `keep_for_read_only_research` with `single_public_sample_caveat=true`; this remains public-artifact read-only research only and does not authorize strategy/private/order/live/parameter/default-on/tiny-live/promotion.
- `0601T006` has passed QA as the public-only collection / initial aggregate step. Its ordinary synthetic fixed-grid aggregate remains diagnostic-only after `0604T003`.
- `0604T003` has passed QA and is the formal robustness source: canonical event-mode aggregate has `canonical_sample_count=3` and recommendation `continue_read_only_runner_refinement`; ordinary synthetic `xemm_0603_quiet_a/b/c` comparison has `canonical_sample_count=0`, `diagnostic_synthetic_sample_count=3`, and recommendation `needs_more_public_samples`.
- `0604T004` passed QA. It implemented the narrow canonical event-mode evidence loader / validator foundation before any parallel Milestone 0 / Milestone 1 runners. The accepted `0604T003` canonical aggregate validates to `canonical_sample_count=3`, and the synthetic diagnostic comparison validates to `canonical_sample_count=0` with `diagnostic_rejection_count=3`. It did not perform signal ranking, regime selection, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.
- `0604T005` passed QA. It added reusable canonical source-lock guard checks over the `0604T003` canonical event-mode aggregate through the `0604T004` foundation, produced task-scoped source-lock artifacts, accepted canonical evidence with `canonical_sample_count=3`, and negative-validated the synthetic diagnostic comparison with `canonical_sample_count=0` / `diagnostic_rejection_count=3`.
- `0604T006` completed read-only canonical signal quality ranking over the four `0601T004` allowlist features through the `0604T005` guard path. Its first QA found a report bucket-consistency defect, but `0604T008` repaired that defect and QA passed; downstream work should use the T008-refreshed artifacts under `local_live_analysis/canonical_signal_quality_ranking_0604T006/`. Current ranking remains: `binance_mid_move_ticks_from_prev=keep_for_read_only_research`; `binance_top5_imbalance`, `binance_top5_bid_qty`, and `binance_microprice_minus_mid_ticks=watch_regime_dependent`.
- `0604T007` passed QA as read-only canonical horizon / regime diagnostics over the `0604T003` event-mode aggregate through the `0604T004` loader path. It classified `100/250ms` horizons as watch-only, kept `500/1000/5000/10000ms` as diagnostic-supported only, and left all regime buckets watch-only with no final regime selection, strategy, private/order, live/default-on/tiny-live, parameter search, or promotion.
- `0604T008` passed QA as the narrow T006 report-consistency repair. It did not change ranking scoring, allowlist, source-lock guard, canonical loader, strategy, private/order, live/default-on/tiny-live, parameter search, or promotion.
- `0604T009` business execution is complete and awaiting QA. It combines `0604T006/T008` signal ranking and `0604T007` horizon/regime diagnostics into decision-time-visible candidate/watch/reject regime definitions only. Current result: `candidate_for_milestone3_executability=1`, `watch_needs_more_samples=6`, `reject_unstable_direction=9`, `reject_concentrated_or_aliased=2`. The only candidate is anchored by `binance_mid_move_ticks_from_prev`; `watch_regime_dependent` features remain secondary filter/context only.

## Next Step

Current controller decision point after `0518T004` QA:

```text
0513T002 QA passed.
0513T003 QA passed.
0512T008 QA passed.
0513T004 QA passed.
0513T005 QA passed.
0513T006 QA passed.
0513T007 QA failed due to Binance snapshot bootstrap bug in the sidecar reconstructed book.
0513T008 QA passed. It collected one no-rule control run and did not enable new strategy rules, promote live, modify strategy behavior, modify canonical audit schema, or modify core/connector APIs.
0513T009 QA passed. It used the existing 5-13-day-control-30min sample only, fixed the T007 snapshot/bootstrap bug, and did not start live or change strategy/core/connector/schema behavior.
`0519T009` QA passed. The new sample verifies all 15 T006 quote-update audit fields and reruns T008, but Step 9 remains default-off offline replay only and cannot be promoted from a single sample.
`0519T008` QA passed. The runner / artifact mechanics are accepted, while the old `5-13-day-control-30min` result remains `needs_more_instrumentation` because it lacks T006 fields.
`0519T010` QA passed as planning-only Step 9C.
`0518T004` implemented only anchor arbitration, side-conservative rounding, clamp, post-clamp re-check, guarded fallback, stale/join-age suppression, and diagnostic counters. It did not widen into source-level drift repair, generic quote-control redesign, or live promotion.
```

Current formal task:

```text
0513T005 QA passed.
0513T006 generated latency / market-data integrity / provenance / sample usability artifacts over existing local samples only.
Classification:
- 5-13-day-control-15min: pricing_research_candidate, limited to compressed BBO/mid sanity.
- 5-11-night-active: compressed_action_path_only.
- 5-10-day-control-1h-06: compressed_action_path_only.
- 5-9-noon: compressed_action_path_only.
- 5-9-small: compressed_action_path_only.
No current sample qualifies as queue_fill_research_candidate.
Current task: `0525T001` is `待验收`. `0521T002` has passed QA. `0520T002` is complete. T008 runner is accepted, T009 supplied one current-format T006 sample, T010 defines the multi-sample validation contract, T011 collected 3 separated 30min current-format no-rule/default-off samples with run ids beginning `5-19-night-active`, and T001 validated the multi-sample set.
Step 9C plan direction:
- Treat data scenario coverage as the immediate blocker.
- Require current-format samples with T006 fields, maker acceptance, sidecar/join quality, Stage 5 labels, Step 5C diagnostics, and Step 9B outputs.
- Compare candidate stability across volatility, spread, trade intensity, stale/latency, API/churn, inventory, post-only safety, cancel-fill, and market-view quality regimes.
- `0520T001` business execution is complete: accepted-set meets Step 9C research-comparison mass, clean-only sensitivity is under threshold, no candidate is `ready_for_tiny_live_design`, and no live/default-on/promotion or runner change was made.
- `0520T002` completed the validator hardening step and passed QA. `0521T001` collected `5-21-day-control-60min` successfully and passed QA. `0521T002` extended the multi-sample set with that sample and still found no `ready_for_tiny_live_design` candidate.
Completed prerequisites:
- `0514T003` passed QA after implementing and running Stage 4 read-only pricing-model research.
- `0514T004` passed QA as the maker execution outcome requirements contract.
- `0514T005` passed QA after implementing the read-only execution outcome label runner and dataset validation on `5-13-day-control-30min`.
Completed prerequisites:
- `0514T006` QA passed. Stage 6A now fixes the comparison unit, common labels, strata, sample policy, and Stage 6B boundary.
- `0514T007` QA passed. It proves methodology on one sample and shows replay fill/cancel lifecycle mismatch remains the dominant blocker.
- `0514T008` QA passed. It fixes the next-step direction: diagnosis first, repair second, sample expansion later.
- `0515T001` QA passed. It provides read-only diagnosis tables and confirms the root issue is replay lifecycle semantics, not sample coverage.
- `0515T002` QA passed. It narrows the repair to cancel-requested fill eligibility, terminal-state semantics, and long-horizon persistence bias, and explicitly defers sample expansion until after repair regression.
- Prepared next task:
  - `0515T005` should be a narrow repair task limited to the `cancel_race_window_too_short` residual class identified by `0515T004`.
  - It should explicitly exclude `4948` / `residual_replay_fill_trigger_uncertain`.
  - After `0515T005`, `0515T006` should remain a single-case read-only diagnosis for `4948` rather than a broad new repair.
  - `0516T001` adds top5 visible queue and same-price trade-quantity evidence for `4948`; QA should decide whether this is enough to plan a future queue-proxy repair design task. It does not authorize implementation.
  - `0516T002` measured repeatability of queue-ahead proxy mismatch: proxy-only no-fill pattern repeats, but replay-fill false-positive remains single-case.
  - `0518T001` should convert this into a conservative gate design and explicitly defer implementation until more data / more replay false-positive cases exist.
```

Prepared next task:

```text
0511T001 - adverse-selection timing rule 设计规格与验收合同
Dispatch to 业务线程-python for design only. Do not modify strategy code in T001.

0511T002 - adverse-selection timing guard default-off 实现与 Stage 6J replay
Dispatch only after 0511T001 design contract is accepted.

0511T003 - 升级 workflow dashboard 为实验决策看板
Can run independently as a workflow display-layer improvement. Do not modify strategy code.

0511T004 - adverse timing trigger 未命中原因诊断
Dispatch to 测试线程 after 0511T002 outputs are available. Diagnose whether target_deterioration did not trigger, triggered off-path, or replay/source-path metrics are insensitive.

0512T001 - 对齐 Stage 6J replay 与 live adverse-selection source-path
Dispatch to 测试线程 after 0511T004 QA. Diagnose why live/current-format risk diagnostics show positive adverse-selection counts while Stage 6J replay summary shows zero.

0512T002 - add-side submit/re-add toxic timing rule 设计合同
Dispatch to 业务线程-python only after 0512T004 defines the replay/live observability gate and sample split. Design only; do not modify strategy code.

0512T003 - 5-11-night-active live 样本 replay/acceptance/cancel-fill 分析
Dispatch to 测试线程 after the 4H sample has been pulled locally. Complete audit replay, maker acceptance, cancel-fill risk analysis, and archive refresh.

0512T004 - Stage 6J / live adverse-selection 观测门禁改进合同
Dispatch to 测试线程 after 0512T001 QA. It must run before 0512T002 and define which evidence is action-path coverage, replay-model regression, or live-derived source-path proof. It must also define `5-11-night-active` as the main development/diagnostic sample and `5-10-day-control-1h-06` / `5-9-noon` / `5-9-small` as cross-sample sanity checks.

0512T005 - add-side toxic timing guard default-off 实现与离线 replay
Dispatch to 业务线程-python only after 0512T002 QA passes. Implement the default-off add-side toxic timing guard, audit fields, unit tests, Stage 6J replay, and action-path coverage reporting. Do not start live or default-enable the rule.

0512T006 - T005 blocked-row attribution 分析计划
Dispatch to 测试线程 after T005 QA, or earlier only if total controller explicitly allows a planning-only task while T005 is waiting for QA. Plan blocked-row stratification, actual submit-removal analysis, and reason attribution only. Do not implement, do not run experiments, do not design stricter candidates, and do not start live.
```

## Runner Status

- `0510T002` runner has been implemented and executed.
- Current `0510T002` status: `待验收`.
- Output directory: `local_live_analysis/stage6j_cross_sample_0510T002/`.
- Result: `diagnostic_only_no_promotion`, samples `4`, candidates `6`, hard failures `0`.
- Next action: QA should review `.workflow/reports/0510T002-business.md`.
- `0511T001` has been narrowed to a design-contract task only.
- `0511T002` has been created as the later implementation/replay task.
- `0511T003` has been created to upgrade the static dashboard from a task index into an experiment decision board.
- `0511T003` has been executed and is waiting for QA. The dashboard now shows decision summaries, key metrics, business results, QA conclusions, and next steps.
- `0511T001` has been executed and is waiting for QA. It recommends `0511T002` only as a default-off implementation/replay task, not as live promotion.
- `0511T002` has been executed and is waiting for QA. Result: `diagnostic_only_no_promotion`, samples `4`, candidates `7`, hard failures `0`; no live micro test allowed.
- `0511T002` is suitable for QA on implementation/replay mechanics, but it has not identified why the pure adverse timing strategy produced no incremental replay effect. Current known fact: the pure `adverse_timing_target_deterioration_*` candidates matched baseline; root cause is still open.
- `0511T004` has been created as a follow-up diagnostic task for the pure adverse timing candidates that matched baseline in `0511T002`.
- `0511T004` has been executed and is waiting for QA. Diagnosis: `target_deterioration` fired heavily, but it did not overlap `submit_buy` / `submit_sell`; baseline and pure adverse timing candidates had zero row-level action-path differences. Current Stage 6J replay also shows `guard_candidate_adverse_selection_count=0`, while live/current-format risk diagnostics show positive adverse-selection candidate counts. No live micro test.
- `0512T001` has been created as the next diagnostic task. It must align Stage 6J replay source-path observability before any further adverse timing implementation.
- `0512T002` has been updated as a later design-contract task for add-side submit/re-add toxic timing. It is blocked on `0512T004` and does not modify strategy code.
- `0512T001` has been executed and is waiting for QA. Diagnosis: Stage 6J replay does not replay live order lifecycle/cancel-to-fill races; it regenerates simulated order lifecycles through `run_backtest`. Current Stage 6J can remain a replay-model regression gate, but cannot alone prove live adverse-selection source-path improvement. `0512T002` may start as design-only and must include this limitation.
- `0512T003` has been executed and is waiting for QA. `5-11-night-active` passed maker acceptance with action/planned/reject/throttle all `1.0`, working semantic/blocking mismatch `0/0`, API/throttle mismatch `0`, strict replay lag breach/drop/fail `0/0/0`, and post-startup outside dual gate rows `0`. Cancel-fill risk repeated at 4H scale: `391/915` fills after cancel request, notional rate `0.427300`, add-side candidates `201`, adverse-selection candidates `190`. Cross-sample risk decision is `proceed_to_stage6j_narrow_rule`; no live micro test.
- `0512T004` has been created as the replay/live observability-gate task that must run before `0512T002`.
- `0512T004` and `0512T002` now explicitly use `5-11-night-active` as the main development/diagnostic sample and `5-10-day-control-1h-06` / `5-9-noon` / `5-9-small` as cross-sample sanity checks.
- `0512T001` QA passed. Stage 6J replay remains a replay-model regression gate only and cannot alone prove live adverse-selection source-path improvement.
- `0512T003` QA passed. `5-11-night-active` is accepted as the current-format main development/diagnostic sample; it does not authorize live micro test.
- `0512T004` has been executed and is waiting for QA. It defines the three-layer evidence contract: action-path coverage, replay-model regression, and live-derived source-path proof. It allows `0512T002` to start after QA only as a design-contract task; no implementation or live micro test is authorized.
- `0512T004` QA passed. It authorizes `0512T002` to start only as a design-contract task.
- `0512T002` has been executed and is waiting for QA. It defines a default-off add-side submit/re-add toxic timing guard contract, with shared helper, config, audit fields, Stage 6J replay matrix, action-path coverage requirements, and no-live boundary. It allows a later implementation/offline replay task after QA, but does not authorize implementation or live micro test.
- `0512T005` has been created as the post-`0512T002` implementation/offline replay task. It is `待执行` and blocked on `0512T002` QA. It does not authorize live micro test.
- `0512T002` QA passed. It authorized only the post-design implementation/offline replay task, not live micro test.
- `0512T005` QA passed. Commit `34c954e` implements the default-off `add_side_toxic_timing_guard`, shared live/backtest helper, audit fields, Stage 6J candidate matrix, action-path coverage reporting, and focused tests. Stage 6J result: `diagnostic_only_no_promotion`, samples `4`, candidates `7`, hard failures `0`; blocked reduce-side total `0`; pure toxic timing candidates show action-path coverage but no replay risk improvement, so no live micro test.
- `0512T006` has been created as a planning-only follow-up for the T005 phenomenon. It covers only blocked-row stratification, actual submit-removal analysis, and reason attribution plans. It explicitly excludes stricter candidate design, implementation, new replay runs, and live.
- `0512T006` has been executed and is waiting for QA. It produced a planning-only T007 attribution contract: blocked-row strata, true submit-removal risk linkage, and reason/window attribution. It did not implement scripts, run new replay, design stricter candidates, or start live.
- `0512T006` QA passed. It authorizes creation of T007 as read-only attribution / experiment implementation only; no stricter rule redesign or live micro test is authorized.
- `0512T007` QA passed. Result: representative 100ms pure toxic blocked rows `594`, true submit removals `56` row-level / `42` unique submit-order keys, blocked reduce-side `0`, removed-submit overlap with baseline cancel-fill risk events `0/56`, and 50/100/200ms blocked key equivalence `100%`. The direct explanation is that coverage mostly did not remove baseline submits, and the submits it did remove were not the replay risk orders.
- `0512T008` QA passed. Result: live/backtest both call `hbt.depth(0)`, but live reads connector-maintained depth while Stage 6J no-overlay reads replay-reconstructed depth. Audit replay overlay forces compressed market/fair/target fields but not top5 strings. Existing audit/npz is sufficient for compressed action-path alignment, but not sufficient for full L2 / queue / OFI / microprice equivalence.
- `0513T001` QA passed. It authorizes only a bounded `0513T002` implementation for strategy-layer MarketView provenance / top5 audit transparency. It does not authorize live, replay candidates, core API changes, converter/npz changes, or microprice/OFI/queue work.
- `0513T002` QA passed. It adds strategy-layer MarketView provenance and top5 source fields without changing strategy behavior, core API, configs, replay candidates, or live scripts.
- `0513T003` QA passed. It used git commit `192470f` to sync T002/T003 validation code to `awsserver1`, collected `5-13-day-control-15min` from `2026-05-13T08:35:45+0900` to `2026-05-13T08:50:58+0900`, ran `align_live_run.py`, ran `maker_acceptance.py`, and confirmed T002 provenance fields: live decision rows `live_depth`, normal replay decision rows `replay_depth`, and audit replay decision rows `market_view_source=audit_overlay` with `top5_source=replay_depth`. Acceptance hard gates passed, but top5/full L2 remain not fully aligned; no live promotion.
- `0513T004` QA passed. It added `examples/binance_tick_mm/deploy/preflight_live_run.py`, integrated it into `run_live.sh`, added focused tests, and verified startup preflight manifest generation. The gate records commit/dirty status/config hashes/key code hashes/schema compatibility/start-stop marker paths and fails before tmux/live if `AUDIT_FIELDS` is incompatible with strategy audit rows. No live run, AWS change, strategy semantic change, PnL proof, or full L2 proof.
- `0513T005` QA passed. It is planning-only for Step 2 and defines latency metrics, market-data integrity checks, sample priority, artifact outputs, sample usability classification, acceptance criteria, and follow-up tasks. It does not implement code, run replay, start live, or authorize pricing/queue research.
- `0513T006` QA passed. It generated the required Step 2 output artifacts under `local_live_analysis/step2_market_data_baseline_0513T006/`. Result: only `5-13-day-control-15min` is a limited `pricing_research_candidate`; the other four samples are `compressed_action_path_only`; no current sample supports queue/fill research. Existing samples remain legacy/pre-T004 and lack full raw provenance in converted npz.
- `0513T007` has been executed and is waiting for QA. It keeps the standard npz `data` main event array unchanged, adds Binance provenance/top5 sidecars, and proves `raw_seq -> final npz rows -> reconstructed top5 book -> decision rows` mapping with as-of join-age acceptance. Smoke metrics: final data row mapping coverage `1.0`, future join count `0`, depth `pu` mismatch `0`, bookTicker/depth BBO match/mismatch `151/1`; bounded slice also correctly reports unusable sync/join quality with `first_valid_update_aligned=false`, stale joins `243/244`, and gap-crossed joins `244/244`. It is explicitly top5-only and does not authorize live, replay/sweep, strategy changes, full L2 persistence, exact queue-position claims, `align_live_run.py` integration, canonical audit schema changes, or core/connector API changes.
- `0513T008` QA passed. It synced commit `f228950` through a clean remote git worktree, collected `5-13-day-control-30min` from `2026-05-13T17:30:12+0900` to `2026-05-13T18:00:25+0900`, passed T004 preflight with `dirty=false`, ran `align_live_run.py`, passed `maker_acceptance.py`, generated T007 full-run sidecar/join artifacts, refreshed archive sha256 `d7291afe0cb547663d4aa8e4cc9a175bfd06a3e7fcffea9d9eab82cc70b5c033`, and classified the sample as limited `pricing_research_candidate`. It is not live promotion and does not authorize strategy-rule changes.
- `0513T007` QA failed after T008 full-run validation exposed a sidecar bootstrap bug. On `5-13-day-control-30min`, snapshot `raw_seq=6` has `lastUpdateId=10537138804218`; buffered depth `raw_seq=5` has `U=10537138802913, u=10537138805036` and covers `lastUpdateId+1=10537138804219`; snapshot-after depth `raw_seq=7` has `pu=10537138805036` and should chain from `raw_seq=5`. Current T007 ignores the buffered update and starts at `raw_seq=7`, causing `first_valid_update_aligned=false` and `gap_crossed_join_count=47499/47499`.
- `0513T009` QA passed. It replays the buffered pre-snapshot `raw_seq=5` after snapshot `raw_seq=6`, then chains future `raw_seq=7` by `pu == previous_u`. Fixed full-run metrics on `5-13-day-control-30min`: first valid update aligned `true`, depth `pu` mismatch `0`, final data row mapping coverage `1.0`, decision join coverage `1.0`, future join count `0`, and gap-crossed join count `0`. The sample is upgraded for top5 microprice / OFI proxy / imbalance pricing research candidates, but still not for full L2 equivalence, exact live-audit-vs-sidecar top5 equality, or exact queue/fill proof.
- Step 2 is complete enough to proceed to Step 3. The remaining top5 tick/qty non-exact matches should be handled as market-view acceptance thresholds/classification, not more T009 bootstrap repair.
- `0514T001` QA passed. It adds optional Stage 3 market-view gates to `maker_acceptance.py`, keeps the existing action-path gates intact, and classifies `5-13-day-control-30min` as `passes_pricing_research_market_view` using T009 fixed sidecar/join metrics. It does not authorize full L2 equivalence, exact queue proof, strategy changes, or live promotion.
- Stage 4 preconditions are satisfied for read-only pricing-model research. The next task should study fair-value candidates and markouts over accepted samples, not implement a strategy rule or start live.
- `0514T002` QA passed. It defines candidates, filters, horizons, outputs, acceptance criteria, and authorizes `0514T003` as the implementation task for a read-only pricing research runner.
- `0514T003` QA passed. It generated read-only research artifacts under `local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003/`: `47499` decision rows, `47067` primary non-stale rows, `432` stale rows excluded from primary, and `14` candidate_for_followup signals. QA grouped the discovered signals by importance and documented duplicate signals.
- `0514T004` QA passed. It is a requirements-only follow-up for maker execution outcome research: fill probability, time-to-fill, adverse selection after fill, spread capture, queue/priority proxy, cancel-to-fill race, reject/throttle/churn, inventory impact, quote placement/distance, missed-fill opportunity cost, realized PnL decomposition, tail risk, partial-fill lifecycle, inventory cycle, and sample validity/censoring. It requires later analysis to use statistics appropriate to each label type rather than a single universal correlation metric.
- `0514T005` QA passed. It implements the read-only execution outcome label runner, focused tests, and dataset validation on `5-13-day-control-30min`. Result: the first execution-outcome label layer now exists, and it pushes Stage 6 toward replay/live fill-cancel lifecycle proxy calibration rather than broad exact-queue language.
- `0514T006` QA passed. Result: Stage 6 is now explicitly framed as replay/live fill-cancel lifecycle proxy calibration; comparison should be keyed on matched submit opportunities rather than raw cross-domain `order_id`; `5-13-day-control-30min` is sufficient for single-sample methodology but not enough alone for quote-adjustment promotion; `0514T007` remains a read-only implementation task.
