```md
执行线程：
- 业务线程-python

任务ID：
- 0531T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0531T002.md`
- `.workflow/reports/0531T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/stage9n_clean_fill_refinement_0531T002/axis_fill_rate_summary.csv`
- `local_live_analysis/stage9n_clean_fill_refinement_0531T002/candidate_regime_triage.csv`
- `local_live_analysis/stage9n_clean_fill_refinement_0531T002/collection_time_estimate.csv`
- `local_live_analysis/stage9n_clean_fill_refinement_0531T002/fill_flow_decomposition.csv`
- `local_live_analysis/stage9n_clean_fill_refinement_0531T002/run_manifest.json`
- `local_live_analysis/stage9n_clean_fill_refinement_0531T002/stage9n_recommendation.md`
- `local_live_analysis/stage9n_clean_fill_refinement_0531T002/target_gap_viability.csv`

action：
- 只读分析了 `0530T002` / `0529T002` / `0529T005` / `0530T002` 的既有 artifacts。
- 重建了 Stage 9K before/after fill-flow，量化了 `+108` aggregate clean-only fills 的去向。
- 对 Stage 9L top gap 和几个对照 regime 做了 clean-only viability / quality triage。
- 生成了 task-scoped 9N 输出目录与推荐文档。
- 更新了 workflow task/status 记录，未改策略行为、未采集新数据、未碰 Hyperliquid 或 connector/core/schema。

verify：
- 读取 `.workflow/workflow-kit/workflow-manual.md`
- 读取 `.workflow/workflow-kit/task-dispatch-template.md`
- 读取 `.workflow/workflow-kit/thread-report-template.md`
- 读取 `.workflow/workflow-kit/qa-acceptance-template.md`
- 读取 `docs/thread-playbook.md`
- 读取 `task_plan.md` / `progress.md` / `findings.md`
- 读取 `.workflow/reports/0530T002-business.md`
- 读取 `.workflow/reports/0530T002-qa.md`
- 读取 `local_live_analysis/stage9m_targeted_clean_fill_0530T002/**`
- 读取 `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/**`
- 读取 `local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002/**`
- `python -m json.tool local_live_analysis/stage9n_clean_fill_refinement_0531T002/run_manifest.json`
- `git diff --check`

done：
- final_classification: `short_collection_to_cross_minimum_only`
- top gap: `2605 -> 2901` rows, `34 -> 36` fills
- top gap remaining to clean minimum: `4` fills
- observed increment from the 120min sample: `2` fills / `120min`
- estimated follow-up window: `4h-6h` only to cross the clean minimum
- `+20` interpretive target: not worth pursuing on this line
- alternative regimes: higher fill-rate, but materially worse markout / spread-capture quality than the top gap
- generated artifacts:
  - `run_manifest.json`
  - `fill_flow_decomposition.csv`
  - `axis_fill_rate_summary.csv`
  - `target_gap_viability.csv`
  - `candidate_regime_triage.csv`
  - `collection_time_estimate.csv`
  - `stage9n_recommendation.md`

blockers：
- 无

commit：
- `37e783d`

提交信息：
- `Complete Stage 9N clean-fill viability analysis`
```
