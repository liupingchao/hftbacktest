```md
执行线程：
- 测试线程

任务ID：
- 0512T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查本报告是否把 Stage 6J replay、action-path coverage、live-derived source-path proof 三类证据边界拆清楚，并能作为 `0512T002` 的前置观测合同。

files：
- .workflow/tasks/0512T004.md
- .workflow/reports/0512T004-business.md
- .workflow/reports/0512T001-business.md
- .workflow/reports/0512T001-qa.md
- .workflow/reports/0512T003-business.md
- .workflow/reports/0512T003-qa.md
- .workflow/reports/0511T004-business.md
- local_live_analysis/stage6j_cross_sample_0511T002/stage6j_replay_summary.csv
- local_live_analysis/stage6j_regime_control_cancel_fill_risk/stage6i_cancel_fill_summary.csv
- local_live_analysis/stage6j_regime_control_cancel_fill_risk_5-11/stage6i_cancel_fill_summary.csv
- findings.md
- progress.md
- task_plan.md

action：
- 确认 `0512T001` QA 已通过：Stage 6J replay 会重新生成 simulated order lifecycle，不能复现 live order id / cancel-to-fill race / exchange timing；Stage 6J 只能作为 replay-model regression gate，不能单独证明 live adverse-selection improvement。
- 确认 `0512T003` QA 已通过：`5-11-night-active` 是已通过 maker acceptance 的 current-format 4H night-active 样本，风险事件量足够作为主开发/诊断样本，但不授权 live micro test。
- 读取 `0511T004` 诊断：旧 `target_deterioration` 大量触发但与 `submit_buy` / `submit_sell` overlap 为 `0`，所以后续规则必须先证明 action-path coverage。
- 定义三层观测门禁：action-path coverage、replay-model regression、live-derived source-path proof。
- 定义 replay 与 live 可比字段/不可比字段，避免把 Stage 6J source-path 指标误用为 live source-path proof。
- 定义样本分层：`5-11-night-active` 为主开发/诊断样本；`5-10-day-control-1h-06`、`5-9-noon`、`5-9-small` 为 cross-sample sanity check。
- 设计 `0512T002` 可引用的最小验收指标表和后续 runner/instrumentation 需求。
- 未修改策略逻辑，未启动 live，未连接交易所。

verify：
- 读取 `.workflow/reports/0512T001-qa.md` -> `0512T001` 状态 `已通过`，QA 结论确认 Stage 6J 不能单独作为 live adverse-selection improvement proof。
- 读取 `.workflow/reports/0512T003-qa.md` -> `0512T003` 状态 `已通过`，QA 结论确认 `5-11-night-active` 可作为 current-format 主开发/诊断样本，但不允许 live micro test。
- 读取 `local_live_analysis/stage6j_cross_sample_0511T002/stage6j_replay_summary.csv` -> Stage 6J baseline/adverse timing replay 可用于同一 replay fill model 内的候选对比；旧 pure adverse timing candidates 未产生 action-path delta。
- 读取 `local_live_analysis/stage6j_regime_control_cancel_fill_risk/stage6i_cancel_fill_summary.csv` 和 `local_live_analysis/stage6j_regime_control_cancel_fill_risk_5-11/stage6i_cancel_fill_summary.csv` -> current-format live 风险样本存在正的 add-side/adverse-selection candidates，且 `5-11-night-active` 事件量最大。
- `python3 .workflow/build_dashboard.py` -> exit 0，Loaded 10 tasks and 17 reports，写入 `.workflow/dashboard.html` 和 `.workflow/dispatch_suggestions.md`。

done：
- 三层观测门禁定义：

| 证据层 | 能证明什么 | 必须报告的字段 | 不能证明什么 |
|---|---|---|---|
| action-path coverage | 新 rule 是否真的命中 add-side submit/re-add 决策路径 | eligible add-side submit count、blocked submit count、blocked reason、action/planned_action diff、reduce-side allowed rows、blocked reduce-side count | 不能证明 live cancel-to-fill race 会下降 |
| replay-model regression | 在同一 Stage 6J replay fill model 下，candidate 相对 baseline 是否有副作用或回归 | PnL、max position、avg abs position、drop latency/API、action churn、replay fill/cancel-fill、same-side worsening、replay source-path 辅助指标 | 不能跨模型证明 live adverse-selection / inventory-reducing cancel race 已改善 |
| live-derived source-path proof | 真实 live audit 中 cancel-to-fill race/source-path 是否存在、是否在后续 live 里下降 | fill_after_cancel_request、inventory_reducing_cancel_race、adverse_selection_candidate、add-side candidate、cancel-to-fill latency、side-adjusted markout、post-rule live current-format risk summary | 不能由当前 Stage 6J replay 单独替代；历史 live counterfactual 只能证明覆盖，不等于真实成交生命周期改善 |

- 样本角色定义：

| role | run_id | current evidence | use in later work |
|---|---|---|---|
| main development / diagnostic | `5-11-night-active` | maker acceptance 已通过；fills `915`，cancel-fill `391`，notional rate `0.427300`，add-side candidates `201`，adverse-selection candidates `190` | 用于主诊断、rule action-path coverage、主要 replay/regression 表 |
| cross-sample sanity | `5-10-day-control-1h-06` | cancel-fill `21/46`，notional rate `0.456445`，add-side `7`，adverse-selection `14` | 检查 daytime/control 窗口不回归 |
| cross-sample sanity | `5-9-noon` | cancel-fill `28/75`，notional rate `0.373264`，add-side `15`，adverse-selection `13` | 检查 noon/daytime regime 不回归 |
| cross-sample sanity | `5-9-small` | cancel-fill `18/51`，notional rate `0.353011`，add-side `7`，adverse-selection `11` | 保持 Stage 6G acceptance anchor 的 sanity check |

- Stage 6J replay 在后续规则验收中的角色：
  - 可以作为 replay-model regression gate：同一 replay fill model 内比较 baseline、add-side guard、new toxic timing、combined guard、broad cooldown control。
  - 可以辅助观察 replay 内 cancel-fill/source-path 是否恶化。
  - 不能把 Stage 6J 的 `guard_candidate_adverse_selection_count=0` 解释为 live adverse-selection 不存在。
  - 不能把 Stage 6J source-path 改善单独当成 live inventory-reducing cancel race 改善证明。

- replay/live 可比字段：
  - 可比：candidate config、action/planned_action/reject/throttle、row-level action diff、position/max position、drop latency/API rate、PnL/max drawdown、replay 内 cancel-fill/source-path 相对变化。
  - 不可比：live order id 与 replay order id、live cancel_request_ts/fill_ts exact lifecycle、live cancel-to-fill latency 分布与 replay simulated latency、live fill markout 与 replay simulated fill markout、live adverse-selection candidate count 与 Stage 6J replay adverse-selection count 的绝对值。

- `0512T002` 可引用的最小验收指标表：

| sample role | run_id | candidate | action coverage | replay regression | live-derived source-path evidence | promotion status |
|---|---|---|---|---|---|---|
| main | `5-11-night-active` | `baseline` | report eligible add-side submit baseline only | baseline replay metrics | current-format risk: cancel-fill `391`, add-side `201`, adverse-selection `190` | diagnostic only |
| main | `5-11-night-active` | `new_toxic_timing_*` | must show eligible > 0, blocked > 0, submit overlap > 0, blocked reduce-side = 0 | must not breach PnL/position/drop/churn/cancel-fill sanity vs baseline | historical live counterfactual may show event overlap; true improvement requires later post-rule live-derived analysis | no live from T002 |
| sanity | `5-10-day-control-1h-06` | `new_toxic_timing_*` | must show nonzero or explicitly explain low opportunity count | must not regress vs baseline | current-format risk baseline exists: cancel-fill `21`, adverse-selection `14` | no live from T002 |
| sanity | `5-9-noon` | `new_toxic_timing_*` | must show nonzero or explicitly explain low opportunity count | must not regress vs baseline | current-format risk baseline exists: cancel-fill `28`, adverse-selection `13` | no live from T002 |
| sanity | `5-9-small` | `new_toxic_timing_*` | must show nonzero or explicitly explain low opportunity count | must not regress vs baseline | current-format risk baseline exists: cancel-fill `18`, adverse-selection `11` | no live from T002 |

- `0512T002` 启动口径：
  - 允许在 `0512T004` QA 通过后启动 `0512T002`，且只允许作为设计合同任务。
  - `0512T002` 必须设计 add-side submit/re-add toxic timing rule，不得继续把旧 `target_deterioration` threshold tuning 当主方案。
  - `0512T002` 必须把 action-path coverage 作为必要条件，把 Stage 6J regression 作为辅助门禁，把 live-derived source-path proof 留给后续只读 overlay/instrumentation 或 future QA-approved live micro test。
  - `0512T002` 不得声明 live adverse-selection source-path improvement，不得授权实现，不得授权 live micro test。

- 后续只读 runner / instrumentation 建议：
  - 需要在后续实现/验证任务中补充 action-path summary：eligible add-side submit、blocked add-side submit、blocked reason、reduce-side allowed、row-level action/planned_action diff。
  - 建议增加 live-audit counterfactual overlay/tagger：只用 decision-time-visible inputs 在历史 live audit 上标记 “would block” rows，并与 live cancel-fill/adverse-selection events 做时间和 side/path overlap。该 overlay 只能证明覆盖潜在风险路径，不能证明真实 live 改善。
  - 任何策略实现、default enable、live micro test 都不由 `0512T004` 授权。

- 是否允许 live micro test：不允许。

blockers：
- 无执行阻塞。
- 后续实现/晋级阻塞仍存在：缺少 post-rule live-derived source-path proof；必须先完成 `0512T002` 设计、后续实现/replay/QA、只读 overlay 或 instrumentation，再由总控另行判断是否具备 live micro test 前置条件。

commit：
- 无

提交信息：
- 无
```
