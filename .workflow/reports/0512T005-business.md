```md
执行线程：
- 业务线程-python

任务ID：
- 0512T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查 default-off 是否保持、add-side submit/re-add path 是否有覆盖、blocked reduce-side 是否为 0、Stage 6J 是否仅作为 replay-model regression gate、是否无 live micro test。

files：
- examples/binance_tick_mm/strategy_core.py
- examples/binance_tick_mm/live_tick_mm.py
- examples/binance_tick_mm/backtest_tick_mm.py
- examples/binance_tick_mm/audit_schema.py
- examples/binance_tick_mm/stage6j_replay.py
- examples/binance_tick_mm/test_backtest_tick_mm.py
- examples/binance_tick_mm/test_stage6j_replay.py
- local_live_analysis/stage6j_cross_sample_0512T005/
- .workflow/tasks/0512T005.md
- .workflow/reports/0512T005-business.md
- task_plan.md
- progress.md
- findings.md

action：
- 实现 default-off `add_side_toxic_timing_guard`，配置默认 `risk.add_side_toxic_timing_guard_enabled=false`，未默认开启规则。
- 在 `strategy_core.py` 增加共享 `AddSideToxicTimingResult` 和 `add_side_toxic_timing_guard_side_blocks(...)`，核心逻辑只实现一次。
- 在 `decide_actions(...)` 增加 submit-leg suppression：允许 stale same-side cancel，但 toxic window 内 suppress paired re-add submit；只 suppress add-side submit，不 suppress reduce-side submit。
- 在 `live_tick_mm.py` 与 `backtest_tick_mm.py` 接入同一 helper、同一 config、同一 per-side state 和同一 audit 字段。
- 在 `audit_schema.py` 增加 T002 合同要求的 add-side submit/re-add toxic timing audit 字段。
- 在 `stage6j_replay.py` 更新 candidate matrix：baseline、add-side guard、toxic timing 50/100/200ms、add-side guard + toxic timing 100ms、broad cooldown control。
- 在 `stage6j_replay.py` 增加 action-path coverage 汇总：eligible rows、blocked rows、blocked reduce-side、baseline action/planned diff、blocked submit overlap、submit removed。
- 增加 focused unit tests：default-off、add-side block、reduce-side allowed、Stage 6J config injection、coverage scan、baseline diff scan。
- 未启动 live，未连接交易所，未修改 AWS 状态。

verify：
- `python3 -m py_compile examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/stage6j_replay.py examples/binance_tick_mm/audit_schema.py` -> exit 0。
- `python3 -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py examples/binance_tick_mm/test_stage6j_replay.py -q` -> `131 passed in 0.99s`。
- `python3 examples/binance_tick_mm/stage6j_replay.py --local-root local_live_analysis --out-dir local_live_analysis/stage6j_cross_sample_0512T005 --run-id 5-11-night-active --run-id 5-10-day-control-1h-06 --run-id 5-9-noon --run-id 5-9-small` -> exit 0，decision `diagnostic_only_no_promotion`，samples `4`，candidates `7`，hard failures `0`。
- 用已有 Stage 6J audit CSV 重新扫描 baseline action-path diff 并刷新 `stage6j_replay_summary.csv/json`、`STAGE6J_B_REPLAY_SUMMARY.md`、`stage6j_replay_decision.json` -> rows `28`，hard failures `0`，blocked reduce-side total `0`，submit removed total `168`。
- `git diff --check -- examples/binance_tick_mm/...` -> exit 0。

done：
- default-off 保持：
  - `risk.add_side_toxic_timing_guard_enabled` 默认读取为 `False`。
  - Stage 6J baseline 和 non-toxic candidates 不启用新 guard。
  - 本任务没有 live、没有 default enable、没有 promotion。

- 新增 helper/config/audit：
  - helper/result：`AddSideToxicTimingResult`、`add_side_toxic_timing_guard_side_blocks(...)`。
  - config：`add_side_toxic_timing_guard_enabled/window_ms/min_target_move_ticks/latency_threshold_ms/pending_cancel_enabled/post_cancel_fill_enabled/target_move_enabled/block_mode`。
  - audit：`add_side_submit_eligible_*`、`add_side_submit_blocked_*`、`add_side_submit_block_reason_*`、`add_side_submit_reduce_side_allowed_*`、`target_move_since_last_quote_or_cancel_*`、`last_cancel_request_age_ms_*`、`last_cancel_fill_age_ms_*`、`toxic_timing_guard_until_ts_*`。

- 旧 trigger 结论：
  - 旧 `target_deterioration` working-quote-only trigger 没有作为主 trigger。
  - 新 guard 先计算 add-side submit eligibility，再基于 same-side cancel request / cancel-fill / target move / latency signal suppress submit leg。
  - Stage 6J candidate 中旧 adverse timing candidates 不再作为 T005 主矩阵。

- Action-path coverage summary：

| candidate | blocked add-side submit | blocked reduce-side | baseline action/planned diff | blocked submit overlap | submit removed |
|---|---:|---:|---:|---:|---:|
| `baseline_inflight_only` | 0 | 0 | 0 | 0 | 0 |
| `add_side_toxic_timing_50ms` | 594 | 0 | 199 | 56 | 56 |
| `add_side_toxic_timing_100ms` | 594 | 0 | 199 | 56 | 56 |
| `add_side_toxic_timing_200ms` | 594 | 0 | 199 | 56 | 56 |
| `add_side_guard_plus_toxic_timing_100ms` | 663 | 0 | 296 | 0 | 0 |

- Per-sample action-path coverage for `add_side_toxic_timing_100ms`:

| run | eligible buy | eligible sell | blocked add-side submit | blocked reduce-side | baseline action/planned diff | blocked submit overlap | submit removed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `5-11-night-active` | 133 | 366 | 299 | 0 | 0 | 0 | 0 |
| `5-10-day-control-1h-06` | 6 | 490 | 6 | 0 | 0 | 0 | 0 |
| `5-9-noon` | 5 | 51 | 46 | 0 | 17 | 2 | 2 |
| `5-9-small` | 654 | 7 | 243 | 0 | 182 | 54 | 54 |

- blocked reduce-side submit rows：
  - Stage 6J 28 rows total blocked reduce-side = `0`。
  - Unit tests also cover toxic timing blocking add-side while allowing reduce-side submit.

- Replay-model regression summary：

| candidate | pnl sum | max abs notional max | cancel-fill count | inventory worsening no readd | same-side worsening | decision |
|---|---:|---:|---:|---:|---:|---|
| `baseline_inflight_only` | -0.795400 | 243.750150 | 4 | 0 | 3 | baseline |
| `add_side_guard_only` | -0.445100 | 242.439450 | 2 | 1 | 0 | improves replay same-side worsening; not live proof |
| `add_side_toxic_timing_50ms` | -0.986100 | 243.750150 | 4 | 0 | 3 | action coverage yes; no replay risk improvement |
| `add_side_toxic_timing_100ms` | -0.986100 | 243.750150 | 4 | 0 | 3 | action coverage yes; no replay risk improvement |
| `add_side_toxic_timing_200ms` | -0.986100 | 243.750150 | 4 | 0 | 3 | action coverage yes; no replay risk improvement |
| `add_side_guard_plus_toxic_timing_100ms` | -0.445100 | 242.439450 | 2 | 1 | 0 | same replay result as add-side guard |
| `broad_add_side_cooldown_200ms_control` | -0.302900 | 243.750150 | 3 | 0 | 2 | control only |

- Evidence split:
  - action-path coverage：新 guard 确实产生 add-side submit blocked rows，且在 `5-9-noon` / `5-9-small` 有 baseline submit overlap 和 submit removed；blocked reduce-side = `0`。
  - replay-model regression：Stage 6J 完成 4 样本 x 7 candidate，hard failures `0`，但 pure toxic timing candidates 没有改善 replay cancel-fill / same-side worsening。
  - live-derived source-path proof：仍缺失。本任务没有 post-rule live 样本，也不能把 Stage 6J source-path 指标单独当成 live adverse-selection / inventory-reducing cancel race 改善证明。

- 结论：
  - T005 完成了 default-off 实现、共享 helper、live/backtest 接入、audit 字段、unit tests、Stage 6J replay matrix 和 action-path coverage reporting。
  - T005 没有完成 live-derived source-path proof，也不应进入 live promotion。
  - Replay 决策保持 `diagnostic_only_no_promotion`。
  - 后续默认只允许 QA；不允许 live micro test。

blockers：
- 无执行阻塞。
- 晋级阻塞：缺少 live-derived source-path proof；pure toxic timing candidates 在 Stage 6J replay 中有 action-path coverage，但没有 replay risk improvement，不能授权 live micro test。

commit：
- 34c954e

提交信息：
- Add default-off add-side toxic timing guard
```
