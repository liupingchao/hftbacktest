```md
执行线程：
- 业务线程-python

任务ID：
- 0525T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0525T001.md`
- `.workflow/reports/0525T001-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `examples/binance_tick_mm/candidate_bucket_refinement.py`
- `examples/binance_tick_mm/test_quote_adjustment_replay.py`
- `local_live_analysis/stage9d_candidate_bucket_refinement_0525T001/**`

action：
- 新增 read-only Step 9D fine-bucket refinement runner。
- 对 `0521T002` 五个 current-format 样本运行细 bucket 拆分。
- 只分析三个 `keep_for_research` candidate：
  - `min_move_quote_age_churn_guard`
  - `inventory_reservation_shift_band`
  - `size_reduction_or_add_side_suppression_pressure`
- 输出 sample-local candidate vs baseline 的 fine bucket metrics、stability summary、recommendations 和 sample gap recommendations。
- 明确将 outcome-defined / market-view-quality bucket 标为 diagnostic-only，不作为 parameter-sweep seed。
- 未做 live、default-on、guard relaxation、策略行为修改、参数搜索或 promotion。

verify：
- `python examples/binance_tick_mm/candidate_bucket_refinement.py --help`
- `python examples/binance_tick_mm/quote_adjustment_replay.py --help`
- `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py`
- `python examples/binance_tick_mm/candidate_bucket_refinement.py --run-dir local_live_analysis/5-19-day-control-30min --run-dir local_live_analysis/5-19-night-active-30min-a --run-dir local_live_analysis/5-19-night-active-30min-b --run-dir local_live_analysis/5-19-night-active-30min-c --run-dir local_live_analysis/5-21-day-control-60min --output-dir local_live_analysis/stage9d_candidate_bucket_refinement_0525T001 --caveated-sample-id 5-19-night-active-30min-a`
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- 输出目录：`local_live_analysis/stage9d_candidate_bucket_refinement_0525T001/`
- 已生成：
  - `fine_bucket_metrics.csv`
  - `fine_bucket_stability_summary.csv`
  - `fine_bucket_stability_summary.json`
  - `candidate_bucket_recommendations.md`
  - `sample_gap_recommendations.csv`
  - `run_manifest.json`
- `min_move_quote_age_churn_guard` 是当前最适合作为后续参数搜索 seed 的候选：
  - stable promising buckets `20`
  - parameter-sweep seed buckets `8`
  - reject buckets `0`
  - seed 集中在 API/churn、young quote churn、large inventory skew、stale latency、one-tick spread、medium volatility 等可由决策时字段识别的 regime。
- `inventory_reservation_shift_band` 不应作为近期参数搜索主线：
  - reject buckets `17`
  - parameter-sweep seed buckets `0`
  - inventory bucket 中存在稳定但增量不强的 bucket，aggregate 正向不能直接解释成稳定有效的 inventory/reservation rule。
- `size_reduction_or_add_side_suppression_pressure` 只能作为窄 seed：
  - parameter-sweep seed buckets `2`
  - seed 为 `spread_quote_distance=one_tick_tight` 与 `volatility_markout_dispersion=volatility_high`
  - 同时存在 reject buckets `12`，尤其 young-quote churn、fill-after-cancel、stale-latency 等 bucket 有较差 worst-sample markout，不能全局启用。
- 仍不满足 `ready_for_tiny_live_design`：
  - 0521T002 accepted fills 为 `340`
  - global tiny-live filled-order gap 仍为 `160`
- 结果支持进入后续 multi-sample parameter sweep design，但搜索空间应先围绕 `min_move_quote_age_churn_guard` 和 `size_reduction` 的两个窄 seed，不应盲扫全部参数。

blockers：
- 无

commit：
- 见总控最终回报

提交信息：
- docs/workflow + Step 9D fine-bucket runner implementation
```
