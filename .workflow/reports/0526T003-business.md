执行线程：
- 业务线程-python

任务ID：
- 0526T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0526T003.md`
- `.workflow/reports/0526T003-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`

action：
- 将 `0526T003` 从未派发的 design-only task 正式推进为已执行、待验收任务。
- 固定了 narrow `min_move_quote_age_churn_guard` parameter-sweep design contract：
  - candidate scope only: `min_move_quote_age_churn_guard`
  - seed A: `inventory_state=large_skew_or_low_score`
  - seed B: `latency_stale_age=stale_latency_medium`
  - required slices: A only, B only, A/B intersection, with A/B union only as context aggregation
- 固定了样本策略：
  - seven current-format samples as the design input universe
  - two caveated samples are sensitivity / broader evidence only
  - strict-clean samples remain the primary decision set
- 固定了 sweep grid：
  - `min_move_ticks`: `1`, `2`, `3`
  - `min_quote_age_ms`: `250`, `500`, `1000`, `2000`
  - `churn_window_ms`: `1000`, `3000`, `5000`
  - `max_readd_count_in_window`: `1`, `2`, `3`
  - `stale_latency_guard_ms`: `50`, `100`, `200`, only inside `latency_stale_age=stale_latency_medium`
  - expected scale: maximum `432` unique parameter sets and maximum `756` seed-slice evaluations
- 固定了 compute / parallelization design：
  - future implementation should use current local `amdserver` resources: `32` CPU cores and about `60G` memory
  - parallel units are `sample_id`, seed slice, and parameter chunks
  - default chunk size should be roughly `25` to `50` parameter sets per worker shard
  - future runner should produce shard outputs and deterministic reducer outputs
- 固定了 decision-time-visible input boundary:
  - allowed: quote age, join age, anchor age, stale/latency bucket, min-move distance, recent cancel/re-add churn state, inventory score/skew state, spread and post-only safety state
  - disallowed: future fill, future markout, same-sample PnL feedback, fill-after-cancel outcome, and audit overlay fields unavailable at live decision time
- 固定了 strategy-shape contract:
  - only inside seed regimes, suppress or delay quote update when move is below threshold, quote age is too young, churn pressure is active, and replacement does not materially improve post-only safety or inventory risk
  - outside seed regimes, behavior remains unchanged
  - bypasses for post-only safety improvement or inventory-risk improvement must be counted separately
- 固定了 evaluation and stability metrics:
  - decision rows, submit orders, submit reduction, cancel/re-add reduction, fast-cancel churn delta
  - filled orders, fill rate delta, 5s side-adjusted markout delta, spread capture delta
  - fill-after-cancel delta, inventory-increasing / reducing fill-rate deltas
  - post-only risk, missing/stale anchor exposure, reject/throttle/drop exposure
  - diagnostic-only net PnL proxy delta
  - clean-only median/worst/win-rate, all-sample sensitivity, caveated-sample influence, fill and coverage sufficiency
- 固定了 verdict taxonomy:
  - `sweep_seed_promising`
  - `stable_but_low_fill`
  - `churn_only_no_execution_benefit`
  - `too_conservative_fill_loss`
  - `caveated_only`
  - `reject`
  - `not_decisionable`

design details：
- Full-grid scale:
  - Inventory-only slice uses the four global parameters: `3 * 4 * 3 * 3 = 108` combinations.
  - Stale-latency slice adds seed-local `stale_latency_guard_ms`: `108 * 3 = 324` combinations.
  - A/B intersection also uses the stale-latency-aware grid: `324` combinations.
  - Maximum seed-slice evaluations: `108 + 324 + 324 = 756`.
  - Maximum unique parameter sets: `108 + 324 = 432`; the A/B intersection reuses stale-latency-aware parameter sets.
- Fallback grid if strict-clean seed fill mass is insufficient:
  - `min_move_ticks`: `1`, `2`
  - `min_quote_age_ms`: `500`, `1000`
  - `churn_window_ms`: `1000`, `3000`
  - `max_readd_count_in_window`: `1`, `2`
  - `stale_latency_guard_ms`: `100`, `200` only for stale-latency seed
  - Fallback scale: `16` inventory-only combinations, `32` stale-latency combinations, `32` intersection evaluations, about `80` seed-slice evaluations.
- Recommended future output artifacts:
  - `parameter_grid.csv`
  - `parameter_grid.json`
  - `sweep_shards/*.csv`
  - `sweep_metrics.csv`
  - `sweep_stability_summary.csv`
  - `sweep_stability_summary.json`
  - `seed_slice_coverage.csv`
  - `candidate_recommendations.md`
  - `run_manifest.json`
- Recommended future implementation mode:
  - `--smoke` runs a tiny grid on one clean sample and one caveated sample if present.
  - full mode uses the seven current-format samples, but reports clean-only primary results first.
  - all reducers sort by `sample_id`, `seed_slice`, `param_hash`, and metric name to make reruns deterministic.

verdict rules：
- `sweep_seed_promising`:
  - strict-clean sample count and seed-slice coverage are sufficient;
  - churn metrics improve in most clean samples;
  - 5s side-adjusted markout or adverse-fill proxies improve or do not materially degrade;
  - fill loss is bounded and not the sole explanation for improvement;
  - worst clean sample does not violate safety gates;
  - caveated samples do not reverse the clean-only conclusion.
- `stable_but_low_fill`:
  - churn / safety effects are directionally stable;
  - fill mass inside the seed slice is below the promotion-quality threshold;
  - next step should be more targeted strict-clean sampling or low-fill-aware implementation, not tiny live.
- `churn_only_no_execution_benefit`:
  - cancel/re-add or fast-cancel churn improves;
  - fill quality, markout, spread capture, or inventory outcome does not improve enough to justify parameter search as a maker edge.
- `too_conservative_fill_loss`:
  - submit or fill reduction is large;
  - markout / spread capture gains are mainly explained by avoiding fills rather than avoiding toxic fills;
  - inventory-reducing fills degrade or opportunity cost rises materially.
- `caveated_only`:
  - the effect appears only when caveated samples are included;
  - strict-clean samples are neutral, weak, or contradictory.
- `reject`:
  - clean-only metrics are worse, safety gates degrade, or the parameter set increases adverse fill / post-only / stale-anchor risk.
- `not_decisionable`:
  - coverage, fills, or required audit fields are insufficient to classify the parameter set.

future implementation boundary：
- A later implementation task may build a read-only runner for this contract.
- The implementation task should not enable the rule by default, change live behavior, collect live samples, or classify anything as `ready_for_tiny_live_design` unless separately authorized and Step 9C hard gates are met.
- The implementation task should treat net PnL proxy as diagnostic only, not as a promotion metric.
- The implementation task should include smoke verification and deterministic shard/reducer validation before full 7-sample execution.

verify：
- `python3 .workflow/build_dashboard.py`
- `git diff --check`

done：
- `0526T003` now supports creating a later read-only parameter-sweep implementation task.
- It does not authorize tiny-live, default-on, live promotion, guard relaxation, or `ready_for_tiny_live_design`.
- More strict-clean active sample collection remains a valid alternative if the controller requires the `500` strict-clean fill gate before running any sweep implementation.

blockers：
- 无

commit：
- 6772699

提交信息：
- docs(workflow): execute 0526T003 sweep design
