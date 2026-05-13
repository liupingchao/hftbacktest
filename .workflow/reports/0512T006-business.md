```md
执行线程：
- 测试线程

任务ID：
- 0512T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。T006 是计划任务，只验收是否完成 blocked-row stratification、actual submit-removal rows、reason attribution 三类分析计划，并确认未实现脚本、未运行新 replay、未设计更严格候选规则、未启动 live。

files：
- .workflow/tasks/0512T006.md
- .workflow/reports/0512T006-business.md
- .workflow/reports/0512T005-business.md
- local_live_analysis/stage6j_cross_sample_0512T005/stage6j_replay_summary.csv
- local_live_analysis/stage6j_cross_sample_0512T005/STAGE6J_B_REPLAY_SUMMARY.md
- local_live_analysis/stage6j_cross_sample_0512T005/stage6j_replay_decision.json
- task_plan.md
- progress.md
- findings.md

action：
- 读取 `0512T006` 任务文件、`0512T005` business report、Stage 6J summary markdown/csv 和 audit CSV header。
- 整理 T005 已知事实表，保持三层证据边界：action-path coverage、replay-model regression、live-derived source-path proof。
- 制定 blocked rows 分层分析计划。
- 制定 actual submit-removal rows 分析计划。
- 制定 blocked reason attribution 分析计划。
- 明确 T007 可执行输入、输出表和决策指标。
- 未修改策略代码，未实现 attribution runner，未运行新的 Stage 6J replay，未启动 live。

verify：
- 人工检查本报告包含三类计划：blocked rows 分层、actual submit-removal rows、reason attribution。
- 人工检查本报告明确不做第四点更严格候选规则设计。
- 人工检查本报告给出 T007 可执行输入清单、输出表和下一步决策依据。
- `python3 .workflow/build_dashboard.py` -> exit 0，Loaded 12 tasks and 23 reports，已刷新 `.workflow/dashboard.html` 与 `.workflow/dispatch_suggestions.md`。

done：
- T006 只做计划，没有实现、没有实验、没有 live。
- T006 只覆盖前三点分析，不做第四点更严格候选规则。
- T007 仍应是只读 attribution / experiment implementation，除非总控另建规则设计任务。
- no live micro test。

## 已知事实表

| 事实 | 数值 / 状态 | 解释边界 |
|---|---:|---|
| Stage 6J decision | `diagnostic_only_no_promotion` | replay-model regression gate，不是 live promotion |
| samples | `4` | `5-11-night-active` 为主样本，另外三个为 sanity checks |
| candidates | `7` | baseline、add-side guard、pure toxic 50/100/200ms、combined、broad cooldown |
| hard failures | `0` | 离线 replay 机械运行通过 |
| pure toxic blocked add-side submit | `594` | action-path coverage 存在 |
| pure toxic blocked reduce-side | `0` | 没有误伤 reduce-side |
| pure toxic baseline action/planned diff | `199` | 只有部分 blocked rows 改变 action/planned_action |
| pure toxic blocked submit overlap | `56` | 只有部分 blocked rows 与 baseline submit 重叠 |
| pure toxic submit removed | `56` | 真正删除 baseline submit 的 rows 只有这部分 |
| baseline cancel-fill / same-side worsening | `4 / 3` | replay risk 基线 |
| pure toxic cancel-fill / same-side worsening | `4 / 3` | pure toxic 没有改善 replay risk |
| live-derived source-path proof | 缺失 | T005 没有 post-rule live 样本，Stage 6J 不能单独证明 live source-path improvement |

## Per-run 解释焦点

| run | blocked add-side submit | baseline action/planned diff | blocked submit overlap | submit removed | risk 解释焦点 |
|---|---:|---:|---:|---:|---|
| `5-11-night-active` | 299 | 0 | 0 | 0 | blocked 发生了，但未改变 baseline submit；T007 必须解释 eligibility 命中为何不转化为 action change |
| `5-10-day-control-1h-06` | 6 | 0 | 0 | 0 | 小样本同样只有 eligibility 命中，无 submit removal |
| `5-9-noon` | 46 | 17 | 2 | 2 | 有少量真实 submit removal，但 replay risk 未改善 |
| `5-9-small` | 243 | 182 | 54 | 54 | submit removal 主要集中样本，但 cancel-fill / same-side worsening 仍未改善 |

## 计划 1：blocked rows 分层分析

目标：
- 解释 `594` 个 blocked add-side submit rows 到底是实际 action-path 改变，还是仅在 pre-submit eligibility 层命中。
- 特别解释 `5-11-night-active` blocked `299` 但 submit removed `0` 的原因。

输入：
- 每个 run 的 baseline audit：`*/01_baseline_inflight_only/audit_bt_stage6j.csv`。
- 每个 run 的 pure toxic audit：`*/03_add_side_toxic_timing_50ms/audit_bt_stage6j.csv`、`*/04_add_side_toxic_timing_100ms/audit_bt_stage6j.csv`、`*/05_add_side_toxic_timing_200ms/audit_bt_stage6j.csv`。
- `stage6j_replay_summary.csv` 中的 baseline compare counters。

行对齐方法：
- 先沿用 T005/Stage 6J 当前比较口径：按 CSV 行顺序 zip baseline 与 candidate，并检查 `ts_local` mismatch。
- T007 需要额外输出 mismatch 数量；如 mismatch 非零，再使用 `strategy_seq + event_type + event_seq + ts_local` 做备选 join。
- 所有解释必须标注使用的 join key，避免把 lifecycle event 行误当作 decision row。

blocked row 记录粒度：
- 每个 blocked side 产出一条 normalized side record。
- 字段至少包括：`run_id`、`candidate`、`strategy_seq`、`event_type`、`ts_local`、`side`、`position`、baseline `action/planned_action`、candidate `action/planned_action`、`add_side_submit_eligible_*`、`add_side_submit_blocked_*`、`add_side_submit_reduce_side_allowed_*`、`add_side_submit_block_reason_*`、`target_move_since_last_quote_or_cancel_*`、`last_cancel_request_age_ms_*`、`last_cancel_fill_age_ms_*`、`latency_signal_ms`、`working_*_order_id/status/req/pending_cancel`、`throttle_reason`、`reject_reason`、`dropped_by_latency`、`dropped_by_api_limit`。

分层类别：
- `true_submit_removed`：candidate blocked，baseline `action` 或 `planned_action` 有同 side submit，candidate 已无该 submit。
- `blocked_but_baseline_no_submit`：candidate blocked，但 baseline 同行本来没有同 side submit。
- `blocked_but_submit_survived`：candidate blocked，baseline 有 submit，candidate 仍有 submit。若出现，说明 suppress 与 action composition 之间还有路径缺口。
- `blocked_with_action_diff_non_submit`：blocked 且 action/planned diff 存在，但差异不是同 side submit removal，可能是 cancel、keep、组合 action 或 lifecycle row 对齐影响。
- `blocked_no_action_diff`：blocked 但 action/planned 完全不变，是解释 `5-11-night-active` 的首要桶。

需要排查的 swallow / non-action 原因：
- baseline 本来没有同 side submit：eligibility 命中不等于最终 submit。
- position cap / soft inventory cap 已经阻止 submit：看 `position`、`pos_limit`、max abs notional 附近状态。
- quote throttle 或 API interval guard 已经阻止 submit：看 `throttle_reason`、`reject_reason`、`dropped_by_api_limit`。
- latency guard 已经阻止 submit：看 `dropped_by_latency`、`reject_reason=latency_guard`、`latency_signal_ms`。
- working order / pending cancel 状态导致本来不 submit：看 `working_*_order_id/status/req/pending_cancel`。
- cancel-race guard 或 add-side guard 已在其他 candidate 中覆盖同类路径：对比 `add_side_guard_only` 和 combined candidate。
- two-phase replace / cancel+re-add 被拆成不同行：同 `strategy_seq` 附近窗口需要聚合，而不是只看单行。

输出表：
- `t007_blocked_row_strata.csv`：一行一个 blocked side record。
- `t007_blocked_row_strata_summary.csv`：按 `run_id/candidate/side/category` 聚合。
- `t007_non_action_reason_summary.csv`：按 swallow reason 聚合 `blocked_no_action_diff` 和 `blocked_but_baseline_no_submit`。
- `t007_night_active_zero_removal.md`：专门解释 `5-11-night-active` 的 `299 -> 0 submit removed`。

判断标准：
- 如果大多数 blocked rows 落在 `blocked_but_baseline_no_submit` 或 `blocked_no_action_diff`，则 T005 的 coverage 主要是 eligibility coverage，不是 effective submit-path modification。
- 如果 blocked rows 被 throttle/API/latency/position/working-order 状态吞掉，T007 应先报告 ordering/path attribution；不能直接进入更严格候选规则。

## 计划 2：actual submit-removal rows 分析

目标：
- 只分析 `submit_removed_by_guard_count=56` 的真实 action 改变 rows。
- 判断这些 removed submit 是否命中 replay 中真正导致 cancel-fill / same-side worsening 的订单生命周期。

输入：
- baseline 与 pure toxic 100ms audit CSV。
- 50ms/200ms pure toxic audit CSV，用于确认三档窗口是否完全同一批 removal rows。
- `summary.json` 和 `stage6j_replay_summary.csv` 中的 cancel-fill/source-path 聚合。

submit removal row 定义：
- baseline 同行 `action` 或 `planned_action` 含同 side `submit_buy` / `submit_sell`。
- candidate 同行 `add_side_submit_blocked_* = 1`。
- candidate 同行 `action` 与 `planned_action` 均不再含同 side submit。

风险事件关联方法：
- 从 baseline audit 的 removed submit row 出发，记录 `strategy_seq`、`planned_order_id`、side、target tick、working state。
- 在 baseline 后续 lifecycle rows 中关联同 `linked_strategy_seq`、`linked_action`、`linked_order_id`、`order_side`、`order_price_tick` 或同 side/price 的 created order。
- 检查关联订单是否出现 `cancel_requested=1` 后的 `fill_after_cancel_request=1`。
- 对每个 removed submit 计算后续窗口内是否有 cancel-fill、same-side readd then cancel-fill、inventory worsening no readd、inventory-reducing cancel race proxy。
- 与 candidate audit 对比：该订单是否消失、是否由另一侧或稍后 submit 替代、风险事件是否只是迁移到别的 order。

输出字段：
- `run_id`、`candidate`、`side`、`ts_local`、`strategy_seq`、baseline action/planned、candidate action/planned。
- block reason、target move、last cancel request/fill age、latency signal、position、working/pending-cancel state。
- baseline linked order id、candidate linked order id、order price tick、cancel request ts、fill ts、cancel-to-fill latency。
- baseline source-path label、candidate source-path label、risk overlap flag、same-side worsening flag。
- local PnL / position before-after window summary，如现有 audit 足够支持则做，否则标注 unavailable。

输出表：
- `t007_submit_removed_rows.csv`：56 条 expected true removal 明细。
- `t007_submit_removed_risk_linkage.csv`：removed rows 与后续 risk events 的关联结果。
- `t007_submit_removed_summary.md`：按 run/side/reason/source-path 汇总。

判断标准：
- 如果 removed submit 与 baseline replay risk events overlap 很低，则 pure toxic timing 没改善是因为删掉的是非风险 submit。
- 如果 removed submit 与 risk events overlap 高但 replay risk count 不变，则需要报告 fill model / lifecycle replacement / risk migration 问题，而不是直接调强信号。
- 如果 removed submit 集中在 `5-9-small`，而主样本没有真实 removal，则不能用小样本 removal 支撑 live micro test。

## 计划 3：reason attribution 分析

目标：
- 拆解 blocked reason 对 blocked rows、effective submit removals、action diff、risk-event overlap 的贡献。
- 解释 50/100/200ms pure toxic timing 结果完全相同的原因。

输入：
- pure toxic 50/100/200ms audit CSV。
- audit 字段：`add_side_submit_block_reason_*`、`target_move_since_last_quote_or_cancel_*`、`last_cancel_request_age_ms_*`、`last_cancel_fill_age_ms_*`、`latency_signal_ms`、`toxic_timing_guard_until_ts_*`。
- blocked-row strata 与 submit-removal linkage 输出。

reason 解析口径：
- 以 `add_side_submit_block_reason_buy/sell` 为主字段。
- 若 reason 是组合字符串，T007 应拆分为 reason tokens；分隔符需兼容 `|`、`;`、`,`，并保留 raw reason。
- 若 reason 为空但 blocked=1，输出 `missing_reason`，作为 instrumentation gap。
- reason 维度至少覆盖：pending cancel / recent cancel request / recent cancel fill / target move / latency。

统计口径：
- 按 `run_id/candidate/window_ms/side/reason_token` 统计 blocked rows。
- 同时统计每个 reason 的 baseline action/planned diff、blocked submit overlap、submit removed、risk-event overlap。
- 对 age/strength 做分桶：cancel request age、cancel fill age、target move ticks、latency signal ms。
- 输出 reason 的 marginal contribution：`reason_rows / total_blocked`、`submit_removed_by_reason / total_submit_removed`、`risk_overlap_by_reason / total_risk_overlap`。

50/100/200ms 等价性检查：
- 逐行比较三档窗口的 blocked side set：`strategy_seq + ts_local + side + reason`。
- 如果三档 blocked set 完全相同，检查是否所有有效 cancel ages 都小于 50ms，或 reason 由 pending/inflight 状态主导，导致 window_ms 不生效。
- 如果三档 reason set 不同但 summary 相同，检查 action diff / risk overlap 是否被后续 guard 或 replay lifecycle 离散化吞掉。
- 如果 reason 多数来自 target_move 而 age window 无差异，说明 window 不是主变量，T007 只能报告 attribution，不得提出新 threshold 规则作为已验证结论。

输出表：
- `t007_reason_attribution.csv`：reason token x run x side x window 聚合。
- `t007_reason_window_equivalence.csv`：50/100/200ms blocked set 差异。
- `t007_reason_risk_overlap.csv`：reason token 到 risk-event overlap 的贡献。
- `t007_reason_attribution_summary.md`：解释 window 等价性和主导 reason。

判断标准：
- 若某个 reason 高 blocked、低 submit removal、低 risk overlap，则它是 noisy coverage，不是有效风险信号。
- 若某个 reason 高 submit removal 但低 risk overlap，则它改变行为但未命中 replay 风险源。
- 若某个 reason 高 submit removal 且高 risk overlap，才值得后续另建规则设计任务评估 signal 强弱。

## T007 可执行输入清单

必须读取：
- `local_live_analysis/stage6j_cross_sample_0512T005/stage6j_replay_summary.csv`
- `local_live_analysis/stage6j_cross_sample_0512T005/stage6j_replay_summary.json`
- `local_live_analysis/stage6j_cross_sample_0512T005/stage6j_replay_decision.json`
- `local_live_analysis/stage6j_cross_sample_0512T005/STAGE6J_B_REPLAY_SUMMARY.md`
- 4 个 run 的 baseline audit CSV：`*/01_baseline_inflight_only/audit_bt_stage6j.csv`
- 4 个 run 的 pure toxic 50/100/200ms audit CSV：`*/03_add_side_toxic_timing_50ms/audit_bt_stage6j.csv`、`*/04_add_side_toxic_timing_100ms/audit_bt_stage6j.csv`、`*/05_add_side_toxic_timing_200ms/audit_bt_stage6j.csv`
- 可选 comparator：`*/02_add_side_guard_only/audit_bt_stage6j.csv`、`*/06_add_side_guard_plus_toxic_timing_100ms/audit_bt_stage6j.csv`

建议输出：
- `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_blocked_row_strata.csv`
- `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_blocked_row_strata_summary.csv`
- `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_non_action_reason_summary.csv`
- `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_submit_removed_rows.csv`
- `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_submit_removed_risk_linkage.csv`
- `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_reason_attribution.csv`
- `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_reason_window_equivalence.csv`
- `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/T007_ATTRIBUTION_SUMMARY.md`

T007 决策指标：
- `blocked_to_baseline_submit_rate = baseline_submit_overlap_blocked / blocked_total`
- `true_submit_removal_rate = submit_removed / blocked_total`
- `submit_overlap_to_removal_rate = submit_removed / baseline_submit_overlap_blocked`
- `non_action_block_rate = blocked_no_action_diff / blocked_total`
- `risk_event_overlap_rate = removed_submit_risk_overlap / submit_removed`
- `reason_removed_share = submit_removed_by_reason / total_submit_removed`
- `reason_risk_overlap_share = risk_overlap_by_reason / total_risk_overlap`
- `window_equivalence_rate` for 50/100/200ms blocked sets
- `blocked_reduce_side_count` must remain `0`

T007 结论边界：
- T007 只能做只读 attribution / experiment implementation。
- T007 可以判断 pure toxic timing 为什么有 coverage 但没有 replay risk improvement。
- T007 不能提出、实现或推荐更严格候选规则作为已验证结论，除非总控另建规则设计任务。
- T007 不能启动 live，不能声明 live-derived source-path proof。

blockers：
- 无执行阻塞。
- 晋级阻塞仍存在：T006 只是计划，尚未执行 T007 attribution；没有 live-derived source-path proof，不能授权 live micro test。

commit：
- 无

提交信息：
- 无
```
