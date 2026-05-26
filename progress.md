# Progress

## Current Focus

- Use `workflow-kit` and the local dashboard as the persistent development workflow for the hftbacktest Binance maker MM work.
- Current implementation focus: `0526T004` narrow min-move parameter sweep and `0526T005` maker edge family read-only triage have passed QA; `0526T006` focused maker-edge design is created but not yet dispatched.
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
- Latest completed milestones: `0515T003` QA 已通过，`0516T001` QA 已通过，`0516T002` QA 已通过，`0518T001` QA 已通过，`0518T002` QA 已通过，`0518T003` QA 已通过，`0518T004` QA 已通过。

## Current Status

- Workflow files: initializing.
- Active task: `0526T006`
- Active task status: `待执行`
- Current blocker: none.

## Next Task

- `0526T006` is prepared as the next design-only task, but it has not been dispatched. Its task contract now requires a concrete candidate policy skeleton and a later read-only runner input/output contract.

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
