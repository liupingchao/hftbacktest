```md
执行线程：
- 测试线程

任务ID：
- 0511T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- .workflow/tasks/0511T004.md
- .workflow/reports/0511T004-business.md
- local_live_analysis/stage6j_cross_sample_0511T002/*
- local_live_analysis/stage6j_regime_control_cancel_fill_risk/stage6i_cancel_fill_summary.csv
- examples/binance_tick_mm/strategy_core.py
- examples/binance_tick_mm/analyze_cancel_fill_risk.py
- findings.md
- progress.md

action：
- 读取 `0511T002` cross-sample Stage 6J 输出目录，覆盖 4 个 run、7 个 candidate。
- 聚合 pure adverse timing candidates 的 `adverse_timing_guard_*` active count、reason count、target move tick 和 action overlap。
- 对 baseline 与 `adverse_timing_target_deterioration_50/100/200ms` 做 row-level 对比，检查 action、planned_action、position、target/working ticks 是否发生差异。
- 核对 `strategy_core.py` 中 `adverse_timing_guard_side_blocks(...)` 与 `decide_actions(...)` 的实际语义。
- 对比 live/current-format 风险诊断中的 `guard_candidate_adverse_selection_count` 与 Stage 6J replay summary 中的 source-path 指标。
- 未修改策略代码，未启动 live，未连接交易所。

verify：
- `find local_live_analysis/stage6j_cross_sample_0511T002 -maxdepth 3 -type f` -> 找到 4 个 run x 7 个 candidate 的 `audit_bt_stage6j.csv`、`summary.json` 和 config。
- CSV header 检查 -> `audit_bt_stage6j.csv` 包含 `adverse_timing_guard_buy_active`、`adverse_timing_guard_sell_active`、reason、until_ts、target move ticks、action、planned_action、target/working ticks。
- Trigger 聚合检查 -> pure adverse timing candidates 每个候选跨 4 样本都有大量 active rows：`buy_active=118429`，`sell_active=160062`。
- Action overlap 检查 -> pure adverse timing candidates 的 active rows 与 `submit_buy` / `submit_sell` overlap 均为 `0`；与 cancel-or-submit overlap 仅 `buy=19`、`sell=12`。
- Row-level baseline diff 检查 -> 12 个 run-candidate 对比中，baseline vs `adverse_timing_target_deterioration_50/100/200ms` 的 `action`、`planned_action`、position、target/working ticks 差异数均为 `0`。
- Source-path 检查 -> Stage 6J replay summary 中 baseline 和 adverse timing candidates 的 `guard_candidate_adverse_selection_count` 总和均为 `0`；live/current-format 风险诊断中同一批样本存在 positive adverse-selection candidate count。
- `python3 .workflow/build_dashboard.py` -> exit 0。

done：
- 结论 1：`target_deterioration` 不是没有触发。它在 replay audit 里大量触发，pure candidates 每个候选跨 4 样本为 `buy_active=118429`、`sell_active=160062`，reason 主要是 `target_deterioration`。
- 结论 2：当前 trigger 没有命中有效 add-side action path。所有 pure adverse timing candidates 的 active rows 与 `submit_buy` / `submit_sell` overlap 为 `0`，因此没有阻断实际 add-side submit。
- 结论 3：baseline 与 pure adverse timing candidates 在 row-level action path 上完全一致。4 个 run x 3 个 pure candidates 的 `action`、`planned_action`、position、target/working ticks 差异数均为 `0`。
- 结论 4：第一版 `target_deterioration` 语义与目标风险错位。helper 只在 `working.buy is not None` / `working.sell is not None` 时比较 working price 与 target price；但很多需要阻断的 add-side submit / re-add 场景发生在该 side 没有 working order 时，audit 中表现为 `working_*_tick=-1` 且 `adverse_timing_guard_*_active=0`。
- 结论 5：当 trigger active 时，它多数落在已有 stale working quote 的 cancel/keep 路径上；baseline 本来也会 cancel stale quote，所以没有增量 action delta。
- 结论 6：当前 Stage 6J replay 对 live adverse-selection source-path 不足够敏感。live/current-format 风险诊断显示 adverse-selection candidate 存在，例如 `5-10-day-control-1h-06=14`、`5-9-small=11`、`5-9-noon=13`、`5-8-stage3-15m-livetest-v4=25`；但 Stage 6J replay summary 中 baseline 和 adverse timing candidates 的 `guard_candidate_adverse_selection_count` 均为 `0`。
- H1 判定：`target_deterioration` 条件“没触发/过窄”被证伪；它不是没触发，而是触发太宽且不在目标 add-side submit path 上。
- H2 判定：成立。trigger 有 active rows，但没有覆盖 add-side submit action path，因此不改变 cancel/place 行为。
- H3 判定：部分成立。Stage 6J 当前 cancel-fill/source-path 指标能验证 same-side readd 类风险，但对 live adverse-selection / inventory-reducing cancel race 的 replay 表达不足。
- H4 判定：部分成立。live risk 诊断和 Stage 6J replay source-path 分布不一致，说明当前 replay 指标不能单独作为 adverse-selection timing rule 的晋级证据。
- 是否允许进入下一轮策略实现：不建议直接进入实现。下一步应先写新的诊断/设计合同，明确目标是“add-side submit/re-add 前的 toxic timing block”，并补 replay instrumentation 或 source-path 对齐证据。
- 是否允许 live micro test：不允许。当前仍是 `diagnostic_only_no_promotion`，没有证据支持 live promotion 或默认开启规则。

blockers：
- 无执行阻塞。本任务诊断已完成。
- 下一轮策略实现存在前置阻塞：需要补充可观测合同，至少能证明新 trigger 覆盖 add-side submit/re-add path，并且 replay 能观察 live adverse-selection source-path 或明确替代验收指标。

commit：
- 无

提交信息：
- 无
```
