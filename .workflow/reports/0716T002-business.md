# 线程回报

执行线程：
- 业务线程-python/offline-analysis

任务ID：
- 0716T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0716T002.md`
- `.workflow/reports/0716T002-business.md`
- `local_live_analysis/cross_exchange_quote_fill_analysis_0716T002/`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 使用 0716T001 corrected fill attribution 对 0715T001 做 offline quote/fill analysis。
- Join 信息：
  - submitted order intents
  - corrected fill attribution
  - post-only reject status
  - quote placement
  - quote aging guard
  - resting interval public-flow/depth/depletion
  - public stream coverage
- 输出：
  - `attempt_level_quote_fill_analysis.csv`
  - `public_flow_fill_quality_summary.csv`
  - `quote_fill_analysis_manifest.json`
  - `route_decision.json`
  - `boundary_manifest.json`
  - `validation_report.md`
  - `sha256_manifest.csv`

verify：
- Generated artifact parse/semantic assertions:
  - submitted attempts = `5`
  - corrected fill intents = `2`
  - post-only rejects = `3`
  - resting fills = `2`
- `git diff --check`
  - passed

done：
- 0715T001 corrected evidence does not support low fill probability.
- Submitted attempts: `5`.
- Post-only rejects: `3`.
- Corrected filled resting attempts: `2`.
- Strict-trade-through filled attempts: `1`.
- Fill-state counts:
  - `corrected_fill_observed_liquidity_unknown=2`
  - `post_only_reject_not_fill_sample=3`
- Route signal counts:
  - `filled_without_full_public_depletion_explanation=1`
  - `filled_with_adverse_public_flow_quote_policy_risk=1`
  - `post_only_reject_quote_too_aggressive_or_exchange_drift=3`
- Final route:
  - `route_to_quote_policy_design_prework_and_liquidity_role_evidence_repair`
- Interpretation:
  - 0-tick touch placement did get fills in the corrected 0715T001 evidence.
  - One filled resting attempt has strong strict trade-through/adverse public flow.
  - The other fill is not fully explained by captured public depletion.
  - Maker/taker role remains unknown, so fee/PnL calibration remains unsupported.

blockers：
- No offline analysis blocker.
- Remaining evidence gap: maker/taker role and exchange-native fill lifecycle attribution are still unsupported for 0715T001.

commit：
- 8dcbe7c

提交信息：
- Analyze corrected 0715T001 quote fill evidence
