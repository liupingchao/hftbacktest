```md
执行线程：
- 业务线程-python

任务ID：
- 0529T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0529T005.md`
- `.workflow/reports/0529T005-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `examples/binance_tick_mm/fill_quality_rejection_decomposition.py`
- `examples/binance_tick_mm/test_fill_quality_rejection_decomposition.py`
- `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/run_manifest.json`
- `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/rejection_reason_decomposition.csv`
- `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/churn_gate_sensitivity.csv`
- `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/coarsened_trigger_bucket_metrics.csv`
- `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/coarsened_shape_candidates.csv`
- `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/sample_gap_by_regime.csv`
- `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/stage9l_recommendation.md`

action：
- Implemented the read-only Stage 9L rejection decomposition / bucket coarsening runner at `examples/binance_tick_mm/fill_quality_rejection_decomposition.py`.
- Added focused tests covering Stage 9L verdict behavior, churn warning sensitivity, rejection component decomposition, and row-level `sample_id` coverage for coarsened buckets.
- The runner consumes `local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002/run_manifest.json`, reconstructs observed submit rows from existing Stage 5 labels, Stage 5 fill markouts, Stage 5C safety diagnostics, and live audit fields, and preserves sample/caveated status for clean-only vs accepted-set analysis.
- Generated Stage 9L artifacts under `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/`.
- Updated workflow task/status/findings files. Dashboard regeneration was intentionally skipped to avoid unrelated workflow/dashboard churn.

verify：
- `python examples/binance_tick_mm/fill_quality_rejection_decomposition.py --help` -> passed.
- `python -m pytest examples/binance_tick_mm/test_fill_quality_rejection_decomposition.py -q` -> passed, `4 passed`.
- `python -m py_compile examples/binance_tick_mm/fill_quality_rejection_decomposition.py` -> passed.
- `python examples/binance_tick_mm/fill_quality_rejection_decomposition.py --input-dir local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002 --output-dir local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005` -> passed.
- `python -m json.tool local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/run_manifest.json` -> passed.
- `git diff --check` -> passed.
- `python3 .workflow/build_dashboard.py` -> skipped; current task did not need dashboard output and skipping avoids unrelated workflow display churn.

done：
- Final classification: `needs_targeted_clean_fills`.
- Shape A / Shape B candidate rows after coarsening: `0`.
- Coarsened clean-only bucket counts:
  - `ready_for_policy_design`: `0`
  - `needs_more_clean_fills`: `310`
  - `reject_quality_negative`: `662`
  - `not_decisionable`: `0`
- Churn sensitivity results:
  - `stage9k_original_hard_gate`: `0` ready, `68` needs-more, `246` reject, `0` shape candidates
  - `recent_reject_throttle_warning`: `0` ready, `197` needs-more, `117` reject, `0` shape candidates
  - `fast_cancel_cancel_readd_warning`: `0` ready, `154` needs-more, `160` reject, `0` shape candidates
  - `non_true_reject_churn_warning`: `0` ready, `283` needs-more, `31` reject, `0` shape candidates
- Churn-warning sensitivity alone does not create ready candidates.
- Main rejection components include:
  - `recent_reject_or_throttle`: `134` buckets, `31123` rows, `881` fills
  - `high_fill_after_cancel_sensitivity`: `59` buckets, `23716` rows, `436` fills
  - `weak_spread_capture`: `49` buckets, `7146` rows, `502` fills
  - `adverse_5s_markout`: `5` buckets, `5510` rows, `274` fills
  - `reject_throttle_drop`: `26` buckets, `267` rows, `17` fills
- Top targeted clean-fill gap:
  - `churn_warning_coarsened / large_skew_or_low_score / add_side / step_back_gt1 / edge_non_adverse / market_view_usable / post_only_clean / warning_churn_context`
  - `2605` rows, `34` fills, `7` samples, `6` fill samples, `6` additional clean fills needed for the Stage 9L minimum fill threshold.
- The output confirms Stage 9L recomputes coarsened sample coverage from row-level `sample_id`, not by summing Stage 9K aggregate bucket coverage.
- This task did not change strategy behavior, live behavior, fill/cancel replay semantics, parameters, guards, default-on behavior, tiny-live status, or promotion state.
- This task does not authorize policy design yet; the next action should be QA, then controller decision between targeted clean-fill collection and further read-only evidence refinement.

blockers：
- 无

commit：
- fc73c0e

提交信息：
- Add Stage 9L rejection decomposition runner
```
