# 0609T003 Business Report

执行线程：
- 业务线程-research

任务ID：
- 0609T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0609T003.md`
- `.workflow/reports/0609T003-business.md`
- `examples/hyperliquid/basis_positive_filtered_context_viability.py`
- `examples/hyperliquid/test_basis_positive_filtered_context_viability.py`
- `local_live_analysis/basis_positive_filtered_context_viability_0609T003/`

action：
- 在 `0609T002` QA 通过后，解除 `0609T003` 前置阻塞并执行 read-only filtered context viability assessment。
- 新增 T003 专用 read-only runner `basis_positive_filtered_context_viability.py`，只读取 local public/canonical artifacts，不采集新数据、不远端执行、不使用 private/order/strategy/live 路径。
- 新增 focused tests，覆盖 required artifacts、T002 prerequisite、diagnostic-only synthetic input refusal、research labels forbidden-use boundary 和 execution gap register。
- 正式运行 T003 runner，输入为 T002 7-sample canonical event-mode aggregate、T002/T001 decomposition manifests 和 T006 robustness manifest。
- 产出 required task-scoped artifacts under `local_live_analysis/basis_positive_filtered_context_viability_0609T003/`。

filtering definitions：
- Raw context: `context_basis_mid_ticks > 0` after excluding missing/non-numeric basis rows.
- Tail-risk mask: `basis_magnitude_bucket=basis_positive_small` OR `hl_top5_imbalance_bucket=hl_top5_imbalance_negative_small` OR `hl_microprice_minus_mid_bucket=hl_microprice_minus_mid_negative_small`.
- Clean context: raw basis-positive context AND NOT tail-risk mask.
- Research labels are read-only context labels only: `basis_positive_raw_context`, `basis_positive_clean_context`, `basis_positive_tail_risk_context`.

raw vs filtered metrics：
- Raw basis-positive context: `5726` rows / `7` samples, hit rate `0.96462347`, mean future move `33.71201537` ticks, wrong-way count `101`, wrong-way rate `0.01763884`, p95 wrong-way loss `120`, max wrong-way loss `180`, max sample row share `0.28763535`.
- Clean filtered context: `3545` rows / `7` samples, hit rate `0.98180948`, mean future move `43.39492243` ticks, wrong-way count `38`, wrong-way rate `0.01071932`, p95 wrong-way loss `86`, max wrong-way loss `120`, max sample row share `0.31480959`.
- Rejected tail-risk context: `2181` rows / `7` samples, hit rate `0.91775457`, mean future move `17.97340669` ticks, wrong-way count `63`, wrong-way rate `0.02888583`, p95 wrong-way loss `147`, max wrong-way loss `180`, max sample row share `0.24346630`.
- Clean context p95 wrong-way loss improves by `34` ticks versus raw basis-positive context.

rejected-tail metrics：
- `basis_positive_small`: `1911` rows / `7` samples, wrong-way count `51`, wrong-way rate `0.02668760`, p95 wrong-way loss `160`.
- `hl_top5_imbalance_negative_small`: `501` rows / `7` samples, wrong-way count `35`, wrong-way rate `0.06986028`, p95 wrong-way loss `156`.
- `hl_microprice_minus_mid_negative_small`: `490` rows / `7` samples, wrong-way count `33`, wrong-way rate `0.06734694`, p95 wrong-way loss `158`.
- Combined tail-risk mask: `2181` rows / `7` samples, wrong-way count `63`, wrong-way rate `0.02888583`, p95 wrong-way loss `147`.

sample / horizon / conditioning stability：
- Clean context has `7` samples represented and max sample row share `0.31480959`, below the `0.40` gate.
- Per-sample clean context mean future move has no negative-mean sample.
- Horizon clean context mean future move has no negative-mean horizon; `100/250/500/1000/5000/10000ms` all remain positive by mean future move.
- Conditioning clean context checks found no negative-mean clean bucket across spread, join-age, visible movement, Binance momentum, and Hyperliquid book-state conditioning.
- Caveat: long horizons `5000/10000ms` retain larger wrong-way p95 losses; this is observation-layer tail caveat and not execution proof.

research labels：
- `research_context_labels.csv` marks all labels `read_only_context_only=True`.
- Labels explicitly forbid order side, quote price, size, leverage, stop/take-profit, case-library trigger, shadow decision, or executable trading instruction.

execution evidence gap register：
- `execution_evidence_gap_register.md` explicitly states unproven execution-layer items: fill probability, queue/queue-ahead, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, and real order lifecycle.
- Current evidence is public observation-layer only and does not prove maker execution viability.

final recommendation：
- `candidate_for_read_only_case_design`
- Reason: clean filtered basis-positive context improves wrong-way p95 loss and shows no sample/horizon reversal.
- This recommendation is read-only public observation-layer research only. It does not authorize strategy implementation, private/order endpoint use, order lifecycle, case-library implementation, shadow decision generation, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.

verify：
- `python -m json.tool local_live_analysis/basis_positive_targeted_public_collection_0609T002/local_processing_manifest.json` -> passed.
- `python -m json.tool local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T002/basis_positive_wrong_way_manifest.json` -> passed.
- `python -m json.tool local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_wrong_way_manifest.json` -> passed.
- `python -m json.tool local_live_analysis/canonical_basis_positive_robustness_0608T006/basis_positive_robustness_manifest.json` -> passed.
- `python examples/hyperliquid/basis_positive_filtered_context_viability.py --help` -> passed.
- `python -m py_compile examples/hyperliquid/basis_positive_filtered_context_viability.py` -> passed.
- `python -m pytest examples/hyperliquid/test_basis_positive_filtered_context_viability.py -q` -> `5 passed in 0.03s`.
- Formal run: `python examples/hyperliquid/basis_positive_filtered_context_viability.py --input-dir local_live_analysis/basis_positive_targeted_public_collection_0609T002/event_mode_canonical_pricing_signal_0609T002 --t002-collection-dir local_live_analysis/basis_positive_targeted_public_collection_0609T002 --t002-decomposition-dir local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T002 --t001-decomposition-dir local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001 --t006-dir local_live_analysis/canonical_basis_positive_robustness_0608T006 --output-dir local_live_analysis/basis_positive_filtered_context_viability_0609T003` -> final recommendation `candidate_for_read_only_case_design`.
- Output parse: `json_count 1 json_bad []`; `csv_count 6 csv_bad []`.
- Metric assertion check: clean context sample count `7`, max sample row share `0.31480959`, p95 wrong-way improvement `34`, no sample/horizon/conditioning negative-mean clean buckets.
- Boundary text check found only prohibition/scope/boundary statements and artifact paths for private/order/strategy/live/default-on/tiny-live/case-library/shadow/promotion/parameter-search terms.
- `git diff --check` -> passed.

done：
- `0609T003` execution is complete and ready for QA.
- Required artifacts exist:
  - `filtered_context_viability_manifest.json`
  - `raw_vs_filtered_basis_positive_summary.csv`
  - `tail_risk_reject_subset_summary.csv`
  - `per_sample_filtered_context_stability.csv`
  - `horizon_filtered_context_stability.csv`
  - `conditioning_filtered_context_summary.csv`
  - `research_context_labels.csv`
  - `execution_evidence_gap_register.md`
  - `filtered_context_next_step_recommendation.md`

blockers：
- 无

commit：
- 6eeabd3

提交信息：
- 0609T003 filtered context viability execution
