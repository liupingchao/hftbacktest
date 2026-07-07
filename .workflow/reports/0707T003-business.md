# 线程回报

执行线程：
- 测试线程-local-analysis

任务ID：
- 0707T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0707T003.md`
- `local_live_analysis/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/`
- `local_live_analysis/cross_exchange_t010_controlled_live_evidence_0706T010_20260706T110202Z/`
- `local_live_analysis/cross_exchange_t010_anti_drift_distribution_0707T003/`

action：
- Created the formal Task C dispatch file for `T010-ANTI-DRIFT-TOUCH-STABILITY-LIVE-DISTRIBUTION-DIAGNOSIS`.
- Read accepted `0706T008` public no-submit artifact and `0706T010` controlled live artifact.
- Generated read-only distribution diagnostics under `local_live_analysis/cross_exchange_t010_anti_drift_distribution_0707T003/`.
- Produced:
  - `funnel_summary.csv`
  - `touch_stability_quantiles.csv`
  - `reason_taxonomy.csv`
  - `cooccurring_reasons.csv`
  - `touch_stability_retention_curve.csv`
  - `anti_drift_distribution_manifest.json`
- Did not change thresholds, code, quote envelope, size, or live authorization.

verify：
- JSON parse for source manifests and generated manifest
  - passed
- CSV parse for source matrices and generated CSVs
  - passed
- Generated artifact sanity checks
  - passed
- `git diff --check`
  - passed

done：
- Source artifact paths:
  - `local_live_analysis/cross_exchange_t010_long_window_nosubmit_0706T008_20260706T102343Z/public_shadow_live_1800s/`
  - `local_live_analysis/cross_exchange_t010_controlled_live_evidence_0706T010_20260706T110202Z/event_driven_edge_gate_live/`
- Funnel summary:
  - `0706T008`: candidates `2480`, fresh-touch evidence pass `2170`, fresh-touch allowed `125`, anti-drift pass/block `7/118`, fair-mid source pass/block `1/6`, edge pass/block `1/6`, would-submit `1`.
  - `0706T010`: candidates `1872`, fresh-touch evidence pass `1630`, fresh-touch allowed `112`, anti-drift pass/block `5/107`, trigger count `1`, post-open-orders public-state pass/block `0/5`, live submissions `0`.
- Touch-stability anti-drift eval quantiles:
  - `0706T008`: count `125`, p50 `0`, p95 `302.2`, max `748`.
  - `0706T010`: count `112`, p50 `0`, p95 `333.4`, max `616`.
- Retention curve at current `250ms` threshold:
  - `0706T008`: `10/125` retained by touch-stability-only proxy, actual anti-drift pass `7`.
  - `0706T010`: `7/112` retained by touch-stability-only proxy, actual anti-drift pass `5`.
- Dominant reason taxonomy:
  - candidate skip dominated by `missing_same_side_strict_through_support` in both windows.
  - anti-drift block dominated by `touch_stability_below_minimum`.
  - `0706T008` edge blocks were `fair_mid_source_stale`.
- Recommendation:
  - `separately_authorized_controlled_live_evidence_after_a_b_repairs`
  - Rationale: accepted windows already show nonzero candidate/fresh-touch/anti-drift/edge pass paths; the specific A/B mechanism blockers have been repaired; current evidence is not enough to justify threshold changes before one repaired controlled live evidence attempt.

blockers：
- Full `0625T010` remains blocked.
- Any next controlled live evidence task must be separately created with explicit envelope and authorization.

commit：
- b512aef

提交信息：
- Diagnose T010 anti-drift distribution
