```md
执行线程：
- 业务线程-python

任务ID：
- 0512T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查本报告是否只产出 add-side submit/re-add toxic timing rule 设计合同，是否使用 `0512T004` 的三层观测门禁，是否明确 no implementation / no live micro test。

files：
- .workflow/tasks/0512T002.md
- .workflow/reports/0512T002-business.md
- .workflow/reports/0511T004-business.md
- .workflow/reports/0512T001-business.md
- .workflow/reports/0512T003-business.md
- .workflow/reports/0512T004-business.md
- .workflow/reports/0512T004-qa.md
- examples/binance_tick_mm/strategy_core.py
- examples/binance_tick_mm/stage6j_replay.py
- examples/binance_tick_mm/analyze_cancel_fill_risk.py
- findings.md
- progress.md
- task_plan.md

action：
- 确认 `0512T004` QA 已通过，允许 `0512T002` 作为设计合同启动，但不授权实现或 live micro test。
- 读取 `0511T004` 结论：旧 `target_deterioration` 大量触发，但与 `submit_buy` / `submit_sell` overlap 为 `0`，不能作为主 trigger。
- 读取 `0512T001` 结论：Stage 6J replay 重新生成 simulated lifecycle，不能单独证明 live adverse-selection source-path improvement。
- 读取 `0512T003` 结论：`5-11-night-active` 已通过 maker acceptance，且提供主要事件量；它是主开发/诊断样本，不是单样本 promotion 证据。
- 读取 `0512T004` 观测门禁：验收证据必须拆成 action-path coverage、replay-model regression、live-derived source-path proof。
- 设计新的 default-off add-side submit/re-add toxic timing guard 合同，覆盖 side 当前无 working order 或 cancel+re-add 的 submit path。
- 定义 helper 合同、配置合同、audit 字段合同、Stage 6J replay 合同、QA 验收合同和 no-live 边界。
- 未修改策略代码，未启动 live，未连接交易所。

verify：
- `python3 .workflow/build_dashboard.py` -> exit 0，Loaded 10 tasks and 19 reports，写入 `.workflow/dashboard.html` 和 `.workflow/dispatch_suggestions.md`。
- 合同检查：本报告包含 H/E/F/A 表、helper 合同、配置合同、audit 字段合同、Stage 6J replay 合同、QA 验收合同。
- 边界检查：本报告明确区分 action-path coverage、Stage 6J replay regression、live adverse-selection source-path proof。
- 样本检查：本报告明确 `5-11-night-active` 为主开发/诊断样本，`5-10-day-control-1h-06`、`5-9-noon`、`5-9-small` 为 cross-sample sanity check。
- no-live 检查：本报告不授权策略实现、默认开启或 live micro test。

done：
- H/E/F/A：

| 假设 | 证据 | 证伪条件 | 证伪动作 |
|---|---|---|---|
| H1：旧 `target_deterioration` 不能作为主 trigger | `0511T004` 显示 pure candidates 有大量 active rows，但 `submit_buy` / `submit_sell` overlap 为 `0`，row-level action diff 为 `0` | 新实现仍以 working quote deterioration 为主，且无法证明 submit overlap > 0 | 不进入实现；退回重新定义 submit/re-add eligibility |
| H2：新规则必须覆盖 add-side submit/re-add path | 目标风险多发生在 side 无 working order 或 cancel 后 re-add，旧 helper只看已有 working quote | 主样本 eligible add-side submit count = 0，或 blocked submit count = 0，或 blocked rows 与目标风险 path 无 overlap | 增加只读 action-path instrumentation / overlay；不能进入 live |
| H3：Stage 6J 只能做 replay-model regression | `0512T001` 证明 Stage 6J replay 不复现 live order id、cancel-to-fill race、exchange timing | 报告把 Stage 6J source-path 改善当成 live proof | QA 失败，重写验收口径 |
| H4：`5-11-night-active` 适合主诊断但不能单样本 promotion | `0512T003` 显示该样本 fills `915`、cancel-fill `391`、add-side `201`、adverse-selection `190` | 成功条件只依赖 `5-11-night-active`，没有 sanity checks | 增加 `5-10-day-control-1h-06`、`5-9-noon`、`5-9-small` 结果表 |

- 旧 trigger 结论：
  - `target_deterioration` 不再作为主 trigger。
  - 它可以保留为辅助 signal，但必须基于 submit/re-add eligibility 计算；不能只在 existing working quote path 上触发。
  - 后续实现不得只调 `min_target_move_ticks` 或 cooldown threshold 来复用旧方案。

- 新规则名称和目标：
  - 建议命名：`add_side_toxic_timing_guard`。
  - 默认关闭：`risk.add_side_toxic_timing_guard_enabled = false`。
  - 目标：只阻断 add-side submit/re-add；保留 reduce-side submit、cancel stale working quote、terminal cleanup 和 safety cancellation。
  - 目标路径：side 当前无 working order 准备 submit，或 same-side stale working quote 被 cancel 后准备立即 re-add submit。

- add-side / reduce-side 判定：
  - buy 为 add-side：`position >= 0.0` 且准备 `submit_buy`。
  - sell 为 add-side：`position <= 0.0` 且准备 `submit_sell`。
  - buy 为 reduce-side：`position < 0.0` 且准备 `submit_buy`。
  - sell 为 reduce-side：`position > 0.0` 且准备 `submit_sell`。
  - flat position 下 buy/sell submit 都按 add-side 处理，因为都会打开新 inventory。

- helper 合同：
  - 新 helper 应在 live/backtest 共享路径实现，禁止复制两套逻辑。
  - 建议接口：`add_side_toxic_timing_guard_side_blocks(context, state, config) -> AddSideToxicTimingResult`。
  - `context` 至少包含：`ts_local`、`position`、`target_bid_tick`、`target_ask_tick`、`working`、planned submit eligibility、in-flight/cancel-requested exposure、last same-side cancel request ts、last same-side cancel-fill ts、latency signals。
  - `state` 至少包含 per-side `last_quote_or_cancel_target_tick`、`last_cancel_request_ts`、`last_cancel_fill_ts`、`guard_until_ts`、rolling recent toxic event counters。
  - result 至少包含 per-side `eligible`、`blocked`、`reason`、`guard_until_ts`、`target_move_since_last_quote_or_cancel`、`last_cancel_request_age_ms`、`last_cancel_fill_age_ms`、`reduce_side_allowed`。

- block 语义：
  - 必须先计算 submit eligibility，再决定是否 suppress submit。
  - 对 cancel+re-add path，允许 cancel stale same-side working quote，但 toxic window 内 suppress paired re-add submit。
  - 对 no-working-order path，若 side add-side 且 toxic signal 命中，则 suppress new submit。
  - 对 reduce-side submit，必须允许通过，并在 audit 中证明 blocked reduce-side count = `0`。
  - 不得通过把 `desired_buy` / `desired_sell` 粗暴改成 false 来造成额外 cancel 或 reduce-side block，除非实现能证明 action diff 只来自 add-side submit suppression。

- toxic signal 第一版合同：
  - `recent_same_side_cancel_request_age_ms <= window_ms` 或 `last_same_side_cancel_fill_age_ms <= window_ms` 是必要触发域。
  - 至少再满足一个 timing/toxicity signal：side-adverse target move >= min ticks，或 latency signal >= configured threshold，或 rolling recent cancel-requested fill count > 0。
  - buy side-adverse target move：`last_quote_or_cancel_target_bid_tick - target_bid_tick >= min_ticks`。
  - sell side-adverse target move：`target_ask_tick - last_quote_or_cancel_target_ask_tick >= min_ticks`。
  - future markout、future mid、replay overlay state、live 当时不可见诊断输出禁止作为 live 决策输入。

- 配置合同：
  - `risk.add_side_toxic_timing_guard_enabled = false`
  - `risk.add_side_toxic_timing_guard_window_ms = 50/100/200`
  - `risk.add_side_toxic_timing_guard_min_target_move_ticks = 1/2`
  - `risk.add_side_toxic_timing_guard_latency_threshold_ms = optional`
  - `risk.add_side_toxic_timing_guard_pending_cancel_enabled = true`
  - `risk.add_side_toxic_timing_guard_post_cancel_fill_enabled = true`
  - `risk.add_side_toxic_timing_guard_target_move_enabled = true`
  - `risk.add_side_toxic_timing_guard_block_mode = "add_side_submit_only"`

- audit 字段合同：
  - `add_side_submit_eligible_buy`
  - `add_side_submit_eligible_sell`
  - `add_side_submit_blocked_buy`
  - `add_side_submit_blocked_sell`
  - `add_side_submit_block_reason_buy`
  - `add_side_submit_block_reason_sell`
  - `add_side_submit_reduce_side_allowed_buy`
  - `add_side_submit_reduce_side_allowed_sell`
  - `target_move_since_last_quote_or_cancel_buy`
  - `target_move_since_last_quote_or_cancel_sell`
  - `last_cancel_request_age_ms_buy`
  - `last_cancel_request_age_ms_sell`
  - `last_cancel_fill_age_ms_buy`
  - `last_cancel_fill_age_ms_sell`
  - `toxic_timing_guard_until_ts_buy`
  - `toxic_timing_guard_until_ts_sell`

- Stage 6J replay 合同：
  - samples：主样本 `5-11-night-active`；sanity checks `5-10-day-control-1h-06`、`5-9-noon`、`5-9-small`。
  - candidates：`baseline_inflight_only`、`add_side_guard_only`、`add_side_toxic_timing_50ms`、`add_side_toxic_timing_100ms`、`add_side_toxic_timing_200ms`、`add_side_guard_plus_toxic_timing_100ms`、`broad_add_side_cooldown_200ms_control`。
  - overlays must stay `off/off/off` for optimization-style replay.
  - 每个 candidate 必须报告 action-path coverage、replay-model regression 和 live-derived/counterfactual evidence 状态。
  - Stage 6J source-path 指标只能作为 replay 内辅助诊断，不能单独作为 live source-path proof。

- replay 中必须观察的 action-path coverage：
  - eligible add-side submit rows > 0。
  - blocked add-side submit rows > 0。
  - blocked rows 与 `submit_buy` / `submit_sell` planned path overlap > 0。
  - blocked reduce-side submit rows = 0。
  - row-level action/planned_action diff 必须集中在 add-side submit suppression 或 cancel+re-add 的 submit leg suppression。

- replay-model regression 指标：
  - PnL / max drawdown 不作为唯一晋级指标。
  - 必须报告 max abs position、avg abs position、drop latency/API rate、action churn、submit/cancel counts、replay fill/cancel-fill、same-side worsening。
  - 不允许通过大幅减少交易量、削弱 reduce-side、或扩大 drops 来制造表面改善。

- live-derived source-path proof：
  - 历史 live counterfactual overlay 可作为只读证据：用 decision-time-visible inputs 标记 would-block rows，并与 live cancel-fill/adverse-selection events 做 side/time overlap。
  - counterfactual overlay 只能证明覆盖潜在 live 风险路径，不能证明真实 live fill lifecycle 会改善。
  - 真正的 source-path improvement proof 只能来自后续 QA 批准的 post-rule live micro test 之后的 current-format live audit risk analysis。
  - `0512T002` 不授权该 live micro test。

- QA 验收合同：
  - 本任务 QA 只验设计合同是否完整，不验策略效果。
  - 后续实现任务 QA 必须验 default-off、shared helper、audit 字段、unit tests、Stage 6J replay、action-path coverage、no-live。
  - 后续 live micro test 必须另建任务，且前置条件至少包括实现 QA 通过、Stage 6J regression 通过、historical live counterfactual coverage 通过、总控显式批准。

- 是否允许创建后续实现任务：
  - 允许在 `0512T002` QA 通过后创建一个后续 implementation + offline replay task。
  - 不允许直接实现于本任务。
  - 不允许 live micro test。

blockers：
- 无执行阻塞。
- 晋级阻塞仍存在：还没有实现 action-path instrumentation，也没有 post-rule live-derived source-path proof。

commit：
- 无

提交信息：
- 无
```
