# Progress

## Current Focus

- Use `workflow-kit` and the local dashboard as the persistent development workflow for the hftbacktest Binance maker MM work.
- Current implementation focus: `0513T003 - 5-13-day-control-15min T002 live-data 验证`.

## Current Status

- Workflow files: initializing.
- Active task: `0513T003`.
- Active task status: `执行中`.
- Current blocker: none.

## Next Step

Current controller decision point after `0513T001` QA:

```text
0513T002 has been implemented and is waiting for QA.
0513T003 is now validating T002 on a fresh 15-minute no-rule control live sample, `5-13-day-control-15min`, after explicit controller authorization to update `awsserver1` and start live collection.
This validation must not change strategy rules or promote live behavior; it only checks that T002 provenance fields are emitted and interpretable in live/replay/audit-overlay outputs.
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
- `0512T008` has been created and executed. It is waiting for QA. Result: live/backtest both call `hbt.depth(0)`, but live reads connector-maintained depth while Stage 6J no-overlay reads replay-reconstructed depth. Audit replay overlay forces compressed market/fair/target fields but not top5 strings. Existing audit/npz is not sufficient for full L2 / queue / OFI / microprice research; the previously named T009 is now `0513T001` planning-only, followed by a separate implementation task if QA passes.
- `0513T001` QA passed. It authorizes only a bounded `0513T002` implementation for strategy-layer MarketView provenance / top5 audit transparency. It does not authorize live, replay candidates, core API changes, converter/npz changes, or microprice/OFI/queue work.
- `0513T002` has been implemented and is waiting for QA. It adds strategy-layer MarketView provenance and top5 source fields without changing strategy behavior, core API, configs, replay candidates, or live scripts.
- `0513T003` has been created and started. It will sync the already-implemented T002 strategy-layer files to `awsserver1`, collect `5-13-day-control-15min` for 15 minutes as a no-rule control sample, then run `align_live_run.py`, `maker_acceptance.py`, archive generation, and provenance field checks.
