```md
执行线程：
- 测试线程

任务ID：
- 0521T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0521T002.md`
- `.workflow/reports/0521T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `local_live_analysis/5-19-day-control-30min/**`
- `local_live_analysis/5-19-night-active-30min-a/**`
- `local_live_analysis/5-19-night-active-30min-b/**`
- `local_live_analysis/5-19-night-active-30min-c/**`
- `local_live_analysis/5-21-day-control-60min/**`
- `local_live_analysis/stage9c_multi_sample_validation_0521T002/**`

action：
- 基于现有 current-format no-rule/default-off 样本完成 Step 9C candidate x scenario bucket multi-sample determination。
- 先补齐 `5-21-day-control-60min` 的缺失派生链，再复跑 `quote_adjustment_replay.py` 的 multi-sample validator。
- 输出 accepted-set / clean-only sensitivity、candidate stability summary、bucket verdicts 和 validation report。
- 不做 live，不做 default-on，不做样本扩张，不做策略修改。

verify：
- `python examples/binance_tick_mm/binance_top5_provenance.py build-sidecars --input-gz local_live_analysis/5-21-day-control-60min/raw_market_data/btcusdt_20260521.gz --out-dir local_live_analysis/5-21-day-control-60min/t009_fixed_sidecar --sample-id 5-21-day-control-60min --symbol BTCUSDT --tick-size 0.1 --buffer-size 8000000`
- `python examples/binance_tick_mm/binance_top5_provenance.py join-decisions --audit-csv local_live_analysis/5-21-day-control-60min/audit_live_5-21-day-control-60min.csv --top5-csv local_live_analysis/5-21-day-control-60min/t009_fixed_sidecar/top5_sidecar.csv --out-csv local_live_analysis/5-21-day-control-60min/t009_fixed_sidecar/joined_decisions.csv --max-age-ms 250`
- `python examples/binance_tick_mm/execution_outcome_labels.py --run-dir local_live_analysis/5-21-day-control-60min`
- `python examples/binance_tick_mm/quote_anchor_safety.py --run-dir local_live_analysis/5-21-day-control-60min --output-dir local_live_analysis/5-21-day-control-60min/stage5c_quote_anchor_safety_0518T004`
- `python examples/binance_tick_mm/execution_outcome_calibration.py --run-dir local_live_analysis/5-21-day-control-60min`
- `python examples/binance_tick_mm/quote_adjustment_replay.py --multi-sample-run-dir local_live_analysis/5-19-day-control-30min --multi-sample-run-dir local_live_analysis/5-19-night-active-30min-a --multi-sample-run-dir local_live_analysis/5-19-night-active-30min-b --multi-sample-run-dir local_live_analysis/5-19-night-active-30min-c --multi-sample-run-dir local_live_analysis/5-21-day-control-60min --output-dir local_live_analysis/stage9c_multi_sample_validation_0521T002 --task-id 0521T002 --caveated-sample-id 5-19-night-active-30min-a
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- `5-21-day-control-60min` 已补齐为可判定样本的派生链，并纳入 0521T002 多样本判定。
- accepted-set / clean-only sensitivity 都达到 Step 9C research-comparison mass，但仍没有 `ready_for_tiny_live_design` candidate。
- `spread_widening_stale_latency` 保持 `keep_for_research`，但仍是 guard-suppressed。
- 其他候选的 reject / keep_for_research verdict 已写入 validation report 和 stability summary。
- 回报必须带 `commit id` 和 `提交信息`。

blockers：
- 无

commit：
- 67c3473

提交信息：
- docs(workflow): complete 0521T002 multi-sample validation
```
