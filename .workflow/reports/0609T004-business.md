# 0609T004 Business Report

执行线程：
- 业务线程-research

任务ID：
- 0609T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0609T004.md`
- `.workflow/reports/0609T003-qa.md`
- `.workflow/reports/0609T004-business.md`
- `docs/qa-acceptance-report.md`
- `docs/basis_positive_clean_context_case_design.md`
- `local_live_analysis/basis_positive_clean_case_design_0609T004/`

action：
- 先补齐 `0609T003` QA gate，并将最新 QA 结果复制到 `docs/qa-acceptance-report.md`。
- 在 `0609T003` QA 通过、final recommendation 为 `candidate_for_read_only_case_design` 后，执行 `0609T004` read-only case-design contract。
- 读取 T003 manifest、raw-vs-filtered summary、tail-risk summary、sample/horizon/conditioning stability、research labels、next-step recommendation 和 execution gap register。
- 生成 `basis_positive_clean_context` read-only design contract、field taxonomy、label taxonomy、acceptance gates、reject conditions 和 execution gap map。

design outputs：
- `docs/basis_positive_clean_context_case_design.md`
- `local_live_analysis/basis_positive_clean_case_design_0609T004/case_design_manifest.json`
- `local_live_analysis/basis_positive_clean_case_design_0609T004/case_field_contract.csv`
- `local_live_analysis/basis_positive_clean_case_design_0609T004/case_label_contract.csv`
- `local_live_analysis/basis_positive_clean_case_design_0609T004/case_acceptance_gate.md`
- `local_live_analysis/basis_positive_clean_case_design_0609T004/case_reject_conditions.csv`
- `local_live_analysis/basis_positive_clean_case_design_0609T004/execution_gap_to_future_evidence_map.md`

field taxonomy：
- `decision_time_visible_context`: `context_basis_mid_ticks`, `basis_magnitude_bucket`, `hl_top5_imbalance_bucket`, `hl_microprice_minus_mid_bucket`, `input_binance_mid_move_ticks_from_prev`.
- `diagnostic_context`: `spread_bucket`, `join_age_bucket`, `visible_movement_bucket`.
- `future_label_for_research_only`: `horizon_ms`, `effective_future_row_delta_count`, `hyperliquid_future_mid_move_ticks`.
- `execution_gap_reference`: fill probability, queue/queue-ahead, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, and real order lifecycle gaps.

label taxonomy：
- All labels are marked `read_only_design_label`.
- Labels: `basis_positive_clean_context`, `basis_positive_raw_context`, `basis_positive_tail_risk_context`.
- All labels forbid row-level case-library entries, shadow decisions, executable trading instructions, order side, quote price/size, leverage, stop/take-profit, private/order endpoint behavior, strategy implementation, live/default-on/tiny-live, parameter search, deployment recommendation, and promotion.

acceptance gates：
- T003 and T004 QA must pass before any later read-only case-library design discussion.
- Clean context must remain represented across at least `7` samples.
- Clean context max sample row share must remain below `0.40`.
- Clean p95 wrong-way loss improvement versus raw must remain positive.
- No sample/horizon/conditioning negative-mean reversal may be present.
- Execution-layer gaps remain unproven unless separately dispatched and QA-accepted.

reject conditions：
- Reject any proposal that attempts executable trigger, row-level case-library entries, shadow decision generation, private/account/order endpoint use, order lifecycle logic, strategy implementation, live/default-on/tiny-live, parameter search, deployment, promotion, or future-label-as-input misuse.
- Reject or downgrade the direction if the T003 evidence gates fail: sample count below `7`, max sample row share at or above `0.40`, non-positive p95 wrong-way improvement, or sample/horizon/conditioning negative-mean reversal.

execution gap map：
- Preserves T003 boundary that current evidence is public observation-layer only.
- Does not prove fill probability, queue position, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, or real order lifecycle.
- Any future attempt to address those gaps requires a separate dispatch and QA.

final recommendation：
- `case_design_contract_ready_for_qa`
- This is design/read-only only. It does not authorize strategy implementation, private/account/order endpoints, order lifecycle, case-library implementation, shadow decision generation, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.

verify：
- `python -m json.tool local_live_analysis/basis_positive_filtered_context_viability_0609T003/filtered_context_viability_manifest.json` -> passed.
- `python -m json.tool local_live_analysis/basis_positive_clean_case_design_0609T004/case_design_manifest.json` -> passed.
- T004 CSV parse for `case_field_contract.csv`, `case_label_contract.csv`, and `case_reject_conditions.csv` -> passed.
- Required markdown artifact existence and boundary-language check -> passed.
- Boundary text check found only prohibition/scope/boundary statements and artifact paths for private/order/strategy/live/default-on/tiny-live/case-library/shadow/promotion/executable terms.
- `git diff --check` -> passed.

done：
- `0609T003` QA gate is complete.
- `0609T004` read-only case-design artifacts and business report are complete and ready for QA.

blockers：
- 无

commit：
- 7e43bed

提交信息：
- 0609T004 basis positive clean case design
