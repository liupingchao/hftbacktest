```md
执行线程：
- 测试线程

任务ID：
- 0512T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查本报告是否解释了 live/current-format adverse-selection source-path 与 Stage 6J replay summary 的差异，并给出后续观测门禁。

files：
- .workflow/tasks/0512T001.md
- .workflow/reports/0512T001-business.md
- .workflow/reports/0511T004-business.md
- local_live_analysis/stage6j_cross_sample_0511T002/stage6j_replay_summary.csv
- local_live_analysis/stage6j_cross_sample_0511T002/*/01_baseline_inflight_only/audit_bt_stage6j.csv
- local_live_analysis/stage6j_regime_control_cancel_fill_risk/stage6i_cancel_fill_summary.csv
- local_live_analysis/stage6j_regime_control_cancel_fill_risk/*/cancel_fill_events.csv
- examples/binance_tick_mm/analyze_cancel_fill_risk.py
- examples/binance_tick_mm/stage6j_replay.py
- findings.md
- progress.md

action：
- 读取 `0512T001` 合同、`0511T004` 诊断报告、live/current-format risk summary、Stage 6J replay summary 和 4 个 baseline replay audit。
- 构建 live vs replay source-path 对比表，覆盖 4 个 run。
- 统计 baseline replay audit 中 `fill_after_cancel_request`、`cancel_request_ts`、`fill_ts`、event_type 和 lifecycle 字段。
- 从 live `cancel_fill_events.csv` 中抽样 inventory-reducing cancel race / adverse-selection candidate event，并按 order_id、side、price、timestamp 在 Stage 6J baseline replay audit 中查找对应关系。
- 核对 `stage6j_replay.py` 调用路径和 `analyze_cancel_fill_risk.py` source-path 分类逻辑。
- 未修改策略代码，未修改 replay 代码，未启动 live。

verify：
- `sed -n '1,80p' local_live_analysis/stage6j_regime_control_cancel_fill_risk/stage6i_cancel_fill_summary.csv` -> live/current-format risk summary 可读，并包含 `guard_candidate_adverse_selection_count`、source-path 和 cancel-fill 字段。
- `sed -n '1,80p' local_live_analysis/stage6j_cross_sample_0511T002/stage6j_replay_summary.csv` -> Stage 6J replay summary 可读，并包含 baseline/adverse timing candidate source-path 字段。
- live vs replay 对比脚本 -> 4 个 run 均显示 live adverse-selection count 为正数，baseline replay adverse-selection count 为 `0`。
- baseline replay lifecycle 统计 -> replay audit 有 cancel/fill lifecycle 字段，但 replay 自身只生成 `0/1/1/1` 条 `fill_after_cancel_request`，远少于 live 的 `21/51/28/18`。
- live event 对应检查 -> 4 个抽样 live adverse-selection order id 在 Stage 6J baseline replay audit 中均无对应 order/fill/lifecycle 行。
- near-window 检查 -> live adverse event 时间附近，replay audit 多数是 decision/keep；没有相同 side/price fill，说明 replay 未复现 live adverse-selection lifecycle。
- `stage6j_replay.py` 代码检查 -> Stage 6J 是调用 `run_backtest(...)` 重新生成 candidate replay audit，再调用同一个 `analyze_audit_csv(...)` 分析 replay audit。
- `analyze_cancel_fill_risk.py` 代码检查 -> `_guard_candidate_path(...)` 将 `inventory_reducing_cancel_race` 分类为 `adverse_selection_candidate`；分类逻辑对 live/replay 是同一个函数。

done：
- live vs replay source-path 对比表：

| run_id | live cancel-fill | replay baseline cancel-fill | live adverse count | replay baseline adverse count | live add-side count | replay add-side count | mismatch reason |
|---|---:|---:|---:|---:|---:|---:|---|
| `5-10-day-control-1h-06` | 21 | 0 | 14 | 0 | 7 | 0 | replay 未复现 live cancel-after-request fills |
| `5-8-stage3-15m-livetest-v4` | 51 | 1 | 25 | 0 | 26 | 1 | replay 只生成极少 cancel-after-request fills，且 source-path 分布不同 |
| `5-9-noon` | 28 | 1 | 13 | 0 | 15 | 1 | replay 只生成极少 cancel-after-request fills，且 source-path 分布不同 |
| `5-9-small` | 18 | 1 | 11 | 0 | 7 | 1 | replay 只生成极少 cancel-after-request fills，且 source-path 分布不同 |

- 抽样对应关系：
  - `5-10-day-control-1h-06` live event：order `154`，buy，price `80717.8`，cancel-to-fill `6.317824ms`，source `inventory_reducing_cancel_race`。replay baseline 中 order `154` 无对应行，且无同 side/price fill。
  - `5-9-small` live event：order `1430`，buy，price `80364.8`，cancel-to-fill `56.155648ms`，source `inventory_reducing_cancel_race`。replay baseline 中 order `1430` 无对应行，且无同 side/price fill。
  - `5-9-noon` live event：order `228`，buy，price `80307.9`，cancel-to-fill `4.73856ms`，source `inventory_reducing_cancel_race`。replay baseline 中 order `228` 无对应行，且无同 side/price fill。
  - `5-8-stage3-15m-livetest-v4` live event：order `81`，sell，price `79630.1`，cancel-to-fill `4.917248ms`，source `inventory_reducing_cancel_race`。replay baseline 中 order `81` 无对应行，且无同 side/price fill。
- 根因分类：主要是 fill model / replay lifecycle 差异，不是 `analyze_cancel_fill_risk.py` source-path 分类函数对 live/replay 使用了不同定义。
- 具体原因：Stage 6J 是策略重放和候选比较 replay。它重新运行 `run_backtest(...)` 生成新的 simulated order lifecycle；live order ids、live cancel-to-fill races 和 live exchange timing 不会被逐事件重放。因此 live adverse-selection / inventory-reducing cancel race 事件不会自然出现在 Stage 6J baseline audit 中。
- Stage 6J 当前能做什么：比较同一个 replay fill model 下不同 candidate 对 replay 内 cancel-fill、same-side readd、position、drop/churn/PnL 的相对影响。
- Stage 6J 当前不能单独做什么：不能单独证明一个 adverse-selection timing rule 已经覆盖或改善 live/current-format inventory-reducing cancel race source-path。
- 是否能作为 adverse-selection timing rule 晋级门禁：不能单独作为晋级门禁。它可以保留为候选回归/副作用门禁，但必须增加 live-event replay/instrumentation gate 或替代 observability gate。
- 建议下一步：
  - `0512T002` 可以启动为设计合同任务，但必须把 Stage 6J 的限制写进合同。
  - `0512T002` 不应要求当前 Stage 6J replay 直接降低 live adverse-selection count；应先要求 action-path coverage：eligible add-side submit/re-add count > 0、blocked submit count > 0、reduce-side allowed。
  - 若后续要把 adverse-selection source-path 作为 hard promotion gate，需要新增 instrumentation/replay 任务，例如 live-event-aligned cancel-fill replay、live event overlay analysis，或在 live audit 上做 counterfactual action-path tagging。
- 是否允许 live micro test：不允许。
- 是否允许启动 `0512T002`：允许启动设计合同，但不允许实现或 live。`0512T002` 必须显式承认当前 Stage 6J 只能做 replay-model regression，不能单独证明 live adverse-selection improvement。

blockers：
- 无执行阻塞。本任务诊断已完成。
- 后续实现阻塞仍存在：缺少能直接度量 live adverse-selection timing rule 改善的 hard gate。

commit：
- 无

提交信息：
- 无
```
