```md
执行线程：
- 测试线程

任务ID：
- 0512T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查 T007 是否只读实现了 T006 的三类 attribution：blocked-row strata、submit-removal risk linkage、reason/window attribution；是否没有改策略代码、没有新 Stage 6J replay、没有规则 redesign、没有 live。

files：
- .workflow/tasks/0512T007.md
- .workflow/reports/0512T007-business.md
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/T007_ATTRIBUTION_SUMMARY.md
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_blocked_row_strata.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_blocked_row_strata_summary.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_non_action_reason_summary.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_submit_removed_rows.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_submit_removed_unique_orders.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_submit_removed_risk_linkage.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_reason_attribution.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_reason_risk_overlap.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_reason_window_equivalence.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_baseline_risk_events.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_candidate_100ms_risk_events.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_risk_event_summary.csv
- local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/t007_summary_metrics.json
- task_plan.md
- progress.md
- findings.md

action：
- 新建 `0512T007` 任务，基于 `0512T006` 计划执行只读 attribution。
- 读取既有 `0512T005` Stage 6J baseline / pure toxic 50ms / 100ms / 200ms audit CSV。
- 生成 blocked-row strata：按 baseline/candidate 行对齐，分出 `true_submit_removed`、`blocked_no_action_diff`、`blocked_with_action_diff_non_submit`。
- 生成 submit-removal linkage：把 true submit removal 的 submit order id 与 baseline / candidate 100ms replay cancel-fill risk events 按 order id 关联。
- 生成 reason/window attribution：拆 `pending_cancel`、`target_move`、`recent_cancel_request`，并比较 50/100/200ms blocked set。
- 输出 `T007_ATTRIBUTION_SUMMARY.md` 和 CSV 明细表。
- 未修改策略代码，未运行新的 Stage 6J replay，未设计更严格候选规则，未启动 live。

verify：
- 只读 attribution 生成完成 -> output dir `local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/`。
- `wc -l local_live_analysis/stage6j_cross_sample_0512T005/t007_attribution/*.csv` -> 12 个 CSV，总计 2293 行。
- 核对代表性 100ms 口径：blocked rows `594`，true submit removals `56`，blocked reduce-side `0`，与 T005 summary 对齐。
- 核对三档窗口：50/100/200ms unique blocked key equivalence rate 都是 `100%`。
- 核对 risk linkage：100ms true submit removals 与 baseline cancel-fill risk event order id overlap 为 `0 / 56`。
- `python3` consistency check on `t007_summary_metrics.json` -> passed。
- `python3 .workflow/build_dashboard.py` -> exit 0，Loaded 13 tasks and 25 reports，已刷新 `.workflow/dashboard.html` 与 `.workflow/dispatch_suggestions.md`。

done：
- 实验结果：
  - representative 100ms pure toxic blocked rows = `594`。
  - true submit removals = `56` row-level removals，占 blocked rows `9.43%`；去重后是 `42` 个 unique submit-order keys。
  - blocked reduce-side = `0`。
  - removed-submit overlap with baseline cancel-fill risk events = `0 / 56`。
  - baseline risk events 与 candidate 100ms risk events 的 order ids 未变化：`5-11-night-active` 为 `2|1`，`5-9-noon` 为 `3`，`5-9-small` 为 `13`。

- blocked-row strata：

| category | 100ms rows |
|---|---:|
| `blocked_no_action_diff` | 463 |
| `blocked_with_action_diff_non_submit` | 75 |
| `true_submit_removed` | 56 |

- per-run 结果：

| run | blocked | true submit removed | no action diff | non-submit diff |
|---|---:|---:|---:|---:|
| `5-11-night-active` | 299 | 0 | 299 | 0 |
| `5-10-day-control-1h-06` | 6 | 0 | 6 | 0 |
| `5-9-noon` | 46 | 2 | 42 | 2 |
| `5-9-small` | 243 | 54 | 116 | 73 |

- reason/window attribution：
  - 100ms reason token `pending_cancel` blocked `594`，submit removed `56`，risk overlap `0`。
  - 100ms reason token `target_move` blocked `594`，submit removed `56`，risk overlap `0`。
  - 100ms reason token `recent_cancel_request` only `4` rows，submit removed `0`，risk overlap `0`。
  - 50/100/200ms unique blocked key sets are identical across all four runs，window equivalence rate `100%`。

- 核心分析：
  - T005 pure toxic timing 有 action-path coverage，但大部分是 eligibility-only coverage；100ms 中 `538 / 594` blocked rows 没有真正删除 baseline submit。
  - 主样本 `5-11-night-active` blocked `299` rows 但 true submit removed `0`，所以它不会改变 replay cancel-fill / same-side worsening。
  - 真正删除的 `56` row-level submits 没有命中任何 baseline cancel-fill risk order；Stage 6J replay 中的 risk order ids 在 baseline 与 candidate 100ms 之间保持不变，因此 replay risk count 不改善是预期结果。
  - 50/100/200ms 完全等价，因为 blocked reason 被 `pending_cancel+target_move` 主导；在这些 replay 样本中，window_ms 不是实际区分变量。

- 证据边界：
  - 这是 replay-model attribution，不是 live-derived source-path proof。
  - T007 没有授权 stricter candidate rule design、default enable、new replay、live micro test 或 promotion。

- 限制：
  - 行级 strata 沿用 T005/Stage 6J 的 baseline/candidate row-order compare 口径；`5-9-small` 和 `5-9-noon` 存在部分 `ts_local` mismatch，所以 row strata 解释应以 T005 action-path coverage 口径为准。
  - risk linkage 使用 order id 关联 cancel-fill risk events；在本轮输出中 true submit removal 与 risk event order id overlap 为 `0`，这个结论不依赖 live source-path。

blockers：
- 无执行阻塞。
- 晋级阻塞仍存在：没有 live-derived source-path proof；不能授权 live micro test。
- 后续如要改规则，必须另建规则设计任务，且不能把 T007 attribution 直接当成 live improvement proof。

commit：
- 无

提交信息：
- 无
```
