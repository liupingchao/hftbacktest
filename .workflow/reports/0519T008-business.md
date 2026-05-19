```md
执行线程：
- 业务线程-python

任务ID：
- 0519T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0519T008.md`
- `.workflow/reports/0519T008-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `examples/binance_tick_mm/quote_adjustment_replay.py`
- `examples/binance_tick_mm/test_quote_adjustment_replay.py`
- `local_live_analysis/5-13-day-control-30min/stage9b_quote_adjustment_replay_0519T008/`

action：
- 将 `0519T008` 状态切到执行中并完成 Step 9B default-off quote-adjustment offline replay runner implementation。
- 新增 `quote_adjustment_replay.py`，实现 default-off / offline-only 诊断 runner。
- 新增 `test_quote_adjustment_replay.py`，覆盖 candidate family、required artifacts、T006 字段缺失时的 classification。
- 在 `5-13-day-control-30min` 上运行 runner，生成 Stage 9B artifacts 和 diagnostic classification。
- 未修改 live/backtest production quote behavior，未跑 live，未 default-on，未 sample expansion，未 promotion。

candidate matrix：
- `baseline_control`
- `fair_reservation_shift`
- `inventory_reservation_shift`
- `spread_widening`
- `size_reduction_or_add_side_suppression`
- `stale_latency_no_fresh_add`
- `min_move_quote_age_churn_guard`
- `post_only_safety_interaction`

generated artifacts：
- `run_manifest.json`
- `candidate_matrix.csv`
- `candidate_matrix.json`
- `candidate_summary.json`
- `candidate_metrics.csv`
- `fill_quality_by_candidate.csv`
- `inventory_cycle_metrics.csv`
- `api_churn_metrics.csv`
- `post_only_safety_metrics.csv`
- `action_path_coverage.csv`
- `audit_field_coverage.csv`
- `candidate_decision_samples.csv`
- `acceptance_decision.md`

classification：
- result: `needs_more_instrumentation`
- reason:
  - existing `5-13-day-control-30min` audit predates T006
  - all 15 T006 quote-update fields are missing in the sample audit
  - runner used proxy fields to validate mechanics and metrics

key metrics：
- decision rows: `47499`
- submit orders: `2516`
- candidate families: `8`
- missing T006 fields: `15`
- maker acceptance input: passed
- baseline fill rate: `0.02106518282988871`
- baseline fill-after-cancel rate: `0.006359300476947536`
- baseline fill-by-5000ms rate: `0.015910898965791568`
- non-baseline candidates produced diagnostic decision coverage except `stale_latency_no_fresh_add`, which had `0` trigger rows on the current proxy definition.
- output metrics include explicit proxy columns for gross/net PnL, fee, spread capture, cancel-to-fill delay, fill horizons, markouts, inventory recovery, API/churn, quote age, and post-only pre/post checks.

interpretation：
- T008 validates runner, metric, artifact, and classification mechanics.
- Current output is not candidate performance proof because the accepted sample lacks T006 quote-update fields.
- Next useful direction is either:
  - collect/replay a current-format sample with T006 fields before candidate claims, or
  - explicitly open a narrow proxy-only diagnostic task if total controller accepts that weaker evidence.

verify：
- `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py`
- `python examples/binance_tick_mm/quote_adjustment_replay.py --help`
- `python examples/binance_tick_mm/quote_adjustment_replay.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage9b_quote_adjustment_replay_0519T008`
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Step 9B runner implementation 已完成。
- Required artifacts 已生成。
- Classification 已写入 `candidate_summary.json` 和 `acceptance_decision.md`。
- 明确 no live, no default-on, no promotion, no sample expansion, no production behavior change。

blockers：
- 无执行 blocker。
- 证据 blocker：当前样本缺少 T006 字段，classification 为 `needs_more_instrumentation`。

commit：
- 待提交

提交信息：
- 待提交
```
