# 0608T006 Business Report

执行线程：
- 业务线程-research

任务ID：
- 0608T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0608T006.md`
- `.workflow/reports/0608T006-business.md`
- `examples/hyperliquid/canonical_basis_positive_robustness.py`
- `examples/hyperliquid/test_canonical_basis_positive_robustness.py`
- `local_live_analysis/canonical_basis_positive_robustness_0608T006/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Implemented a read-only `context_basis_mid_ticks > 0` robustness runner outside the Regime 011 shell.
- Reused the accepted canonical event-mode loader/source-lock path and refused diagnostic-only synthetic inputs as formal evidence.
- Required `0608T005` prerequisite manifest with `final_contract_decision=upgrade_to_context_only_supported`.
- Required `basis_mid_dislocation` contract state `allow/context_only_supported`.
- Resolved row-level inputs from `multi_sample_manifest.json` `samples[].pricing_signal_rows`.
- Generated task-scoped artifacts under `local_live_analysis/canonical_basis_positive_robustness_0608T006/`.

method：
- Primary pattern: `context_basis_mid_ticks > 0`.
- Primary horizon: `1000ms`.
- Diagnostic persistence horizons: `5000ms` and `10000ms`.
- `100/250ms` are parsed only as watch/alias context and are not counted as independent support.
- Base rows are canonical event-mode rows with primary joined/context quality and decision-time basis-positive context.
- Stratification covers spread, join age, deterministic volatility context, and Hyperliquid top5/microprice sign buckets.
- Collinearity check reports Pearson/Spearman correlation and sign overlap against Hyperliquid top5 imbalance and microprice-minus-mid.
- Cost/tail proxy uses fixed assumptions: `2` fee ticks, `2` slippage ticks, `5` latency-decay ticks.

input source-lock：
- Formal canonical input: `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/`.
- Loader/source-lock foundation: `examples/hyperliquid/canonical_event_mode_evidence.py`.
- Canonical sample count: `3`.
- Diagnostic rejection count: `0`.
- Row-level input policy: `multi_sample_manifest.samples[].pricing_signal_rows`.

T005 prerequisite check：
- T005 basis visibility manifest was accepted only when `task_id=0608T005`, `schema_version=canonical_basis_context_visibility_v1`, and `final_contract_decision=upgrade_to_context_only_supported`.
- Data contract check requires `basis_mid_dislocation=allow/context_only_supported`.

non-Regime-011 scope proof：
- Manifest `scope_policy`: `not_limited_to_regime_011`.
- Primary row count is `2425`, versus the prior Regime 011 local pattern size of `57`; this task is not limited to `regime_011_1000_spread_10_20_ticks`.

stratified robustness results：
- Overall: `2425` rows, `3` samples, hit rate `0.94600939`, mean future move `45.67216495` ticks, max sample row share `0.67917526`.
- By sample: quiet_a `1647` rows / mean `16.63934426`; quiet_b `444` rows / mean `99.42567568`; quiet_c `334` rows / mean `117.38023952`.
- By spread: `spread_10_20_ticks` has `2312` rows / mean `41.80795848`; `spread_gt_20_ticks` has `113` rows / mean `124.73451327`.
- Join-age coverage is only `join_age_0_50ms`, with `2425` rows.
- Volatility bucket fell back to `volatility_unbucketed`, with `2425` rows.
- Hyperliquid book-state buckets remain positive in both positive and negative sign buckets, but the positive HL book-state bucket dominates row mass.

HL book-state collinearity / proxy diagnosis：
- `context_hyperliquid_top5_imbalance`: Pearson `0.12536864`, Spearman `0.3248402`, classification `not_explained_solely_by_hl_book_state`.
- `context_hyperliquid_microprice_minus_mid_ticks`: Pearson `0.10341893`, Spearman `0.32296389`, classification `not_explained_solely_by_hl_book_state`.

cost / tail proxy：
- Gross edge: `45.67216495` ticks.
- Net edge proxy: `36.67216495` ticks.
- Wrong-way rate: `0.02845361`.
- p95 wrong-way loss: `138` ticks.
- Max wrong-way loss: `180` ticks.
- Classification: `cost_tail_reject`.

horizon persistence：
- `1000ms`: `2425` rows, mean `45.67216495`, direction `positive`.
- `5000ms`: `2423` rows, mean `78.24391251`, direction `positive`.
- `10000ms`: `2422` rows, mean `86.25928984`, direction `positive`.
- No `1000/5000/10000ms` direction reversal was observed.

final recommendation：
- `needs_more_samples`
- Reason: `basis-positive evidence is sample-concentrated`.
- Supporting caveat: although the gross/net edge proxy is large and horizon persistence is positive, max sample row share is `0.67917526`, join-age/volatility coverage is narrow, and the fixed conservative tail proxy rejects on p95 wrong-way loss.

artifacts：
- `basis_positive_robustness_manifest.json`
- `basis_positive_overall_summary.csv`
- `basis_positive_by_sample.csv`
- `basis_positive_by_spread.csv`
- `basis_positive_by_join_age.csv`
- `basis_positive_by_volatility.csv`
- `basis_positive_by_hl_book_state.csv`
- `basis_positive_collinearity.csv`
- `basis_positive_cost_tail.csv`
- `basis_positive_horizon_persistence.csv`
- `basis_positive_recommendation.md`

verify：
- `python examples/hyperliquid/canonical_basis_positive_robustness.py --help` -> passed
- `python -m py_compile examples/hyperliquid/canonical_basis_positive_robustness.py examples/hyperliquid/test_canonical_basis_positive_robustness.py` -> passed
- `python -m pytest examples/hyperliquid/test_canonical_basis_positive_robustness.py` -> `6 passed`
- `python examples/hyperliquid/canonical_basis_positive_robustness.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --basis-visibility-dir local_live_analysis/canonical_basis_context_visibility_0608T005 --data-contract-dir local_live_analysis/binance_led_hyperliquid_data_contract_0601T004 --output-dir local_live_analysis/canonical_basis_positive_robustness_0608T006` -> passed, final recommendation `needs_more_samples`
- `python -m json.tool local_live_analysis/canonical_basis_positive_robustness_0608T006/basis_positive_robustness_manifest.json` -> passed
- Boundary text check over the report and generated recommendation found only explicit prohibitions / no-authorization language.
- `git diff --check` -> passed

done：
- T006 is complete as read-only basis-positive robustness diagnosis and should enter normal QA.
- This result does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decisions, executable trading instructions, quote/order behavior, deployment recommendation, or promotion.

blockers：
- 无

commit：
- 提交后由最终回报给出

提交信息：
- 0608T006 basis positive robustness
