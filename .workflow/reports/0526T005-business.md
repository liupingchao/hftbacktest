执行线程：
- 业务线程-python

任务ID：
- 0526T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0526T005.md`
- `.workflow/reports/0526T005-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `examples/binance_tick_mm/maker_edge_triage.py`
- `examples/binance_tick_mm/test_quote_adjustment_replay.py`
- `local_live_analysis/stage9h_maker_edge_triage_0526T005/**`

action：
- Created and executed a low-cost read-only maker edge family triage runner:
  - `examples/binance_tick_mm/maker_edge_triage.py`
- Added focused test coverage in:
  - `examples/binance_tick_mm/test_quote_adjustment_replay.py`
- Evaluated five families over the seven current-format samples:
  - `fair_price`
  - `reservation`
  - `inventory`
  - `quote_distance`
  - `size_side`
- Used Stage 5 execution labels as the low-cost evidence layer:
  - fill rate
  - 5s side-adjusted markout
  - spread capture
  - fill-after-cancel
  - inventory-increasing / inventory-reducing fill rates
- Treated caveated samples as sensitivity only:
  - `5-19-night-active-30min-a`
  - `5-26-active-minmove-control-60min-a`
- Did not run parameter search, change strategy behavior, collect live data, default-enable anything, or make promotion claims.

outputs：
- `local_live_analysis/stage9h_maker_edge_triage_0526T005/run_manifest.json`
- `local_live_analysis/stage9h_maker_edge_triage_0526T005/family_bucket_metrics.csv`
- `local_live_analysis/stage9h_maker_edge_triage_0526T005/family_triage_summary.csv`
- `local_live_analysis/stage9h_maker_edge_triage_0526T005/family_triage_summary.json`
- `local_live_analysis/stage9h_maker_edge_triage_0526T005/maker_edge_triage_recommendations.md`

result：
- `top_family`: `inventory`
- `top_verdict`: `strong_next_candidate`
- verdict counts:
  - `strong_next_candidate`: `5`
- ranking:
  1. `inventory`: `strong_next_candidate`, score `7.0`, clean fills `365`
  2. `quote_distance`: `strong_next_candidate`, score `7.0`, clean fills `365`
  3. `size_side`: `strong_next_candidate`, score `7.0`, clean fills `365`
  4. `fair_price`: `strong_next_candidate`, score `6.0`, clean fills `365`
  5. `reservation`: `strong_next_candidate`, score `6.0`, clean fills `365`

key observations：
- This triage is qualitatively different from `0526T004`: it did find broad clean-only separation signals across all five maker-edge families.
- Inventory, quote-distance, and size-side have the strongest immediate separation:
  - inventory clean 5s markout range about `92.44` ticks
  - quote-distance clean 5s markout range about `303.60` ticks
  - size-side clean 5s markout range about `283.60` ticks
- Fair-price and reservation are still promising but appear highly similar in the current Stage 5 label view:
  - both have clean 5s markout range about `220.67` ticks
  - both have similar spread-capture range about `29.53` ticks
- Current evidence supports a focused maker-edge design task that combines:
  - inventory state
  - quote distance
  - size / side selection
  - fair / reservation signal design
- It does not support splitting into five independent implementation tracks yet.

interpretation：
- Best next direction is not more `min_move_quote_age_churn_guard` experiments.
- Best next direction is a focused maker-edge design task around inventory-aware quote placement:
  - use fair/reservation as pricing signal context
  - use inventory state to decide skew / side preference
  - use quote-distance buckets to decide placement frontier
  - use size-side logic to reduce toxic add-side fills and preserve recovery-side fills
- This is still design/research direction only; it does not authorize live or default-on behavior.

verify：
- `python examples/binance_tick_mm/maker_edge_triage.py --help`
- `python -m pytest examples/binance_tick_mm/test_quote_adjustment_replay.py -q`
- `python examples/binance_tick_mm/maker_edge_triage.py`
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- Ranked all five families.
- Identified inventory / quote-distance / size-side as strongest immediate follow-up axes.
- Kept fair-price / reservation as promising context, not as separate immediate tracks.
- Explicitly did not authorize tiny-live, default-on, live promotion, parameter search, or production strategy changes.

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
