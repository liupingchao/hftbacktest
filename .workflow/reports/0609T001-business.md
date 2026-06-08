# 0609T001 Business Report

执行线程：
- 业务线程-research

任务ID：
- 0609T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0609T001.md`
- `.workflow/reports/0609T001-business.md`
- `examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py`
- `examples/hyperliquid/test_canonical_basis_positive_wrong_way_decomposition.py`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

method：
- Consumed accepted local canonical event-mode artifacts through the `0604T004/0604T005` canonical loader/source-lock path.
- Validated T006 prerequisite manifest before analysis: `task_id=0608T006`, `schema_version=canonical_basis_positive_robustness_v1`, `final_recommendation=needs_more_samples`, `scope_policy=not_limited_to_regime_011`, and `t005_final_contract_decision=upgrade_to_context_only_supported`.
- Loaded row-level pricing signal files from `multi_sample_manifest.json` `samples[].pricing_signal_rows`.
- Compared `context_basis_mid_ticks > 0` against `context_basis_mid_ticks <= 0`, excluding missing or non-numeric basis.
- Defined wrong-way tail at `1000ms` as `basis > 0` and `hyperliquid_future_mid_move_ticks < 0`.

source-lock：
- Input: `local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- T006 prerequisite: `local_live_analysis/canonical_basis_positive_robustness_0608T006`
- Canonical sample count: `3`
- Diagnostic rejection count: `0`

basis-positive economic hypothesis：
- The read-only idea is Binance lead / Hyperliquid lag basis context: when Binance mid is above Hyperliquid mid, Hyperliquid may later move toward Binance.
- This is context evidence only. It is not execution-PnL proof and does not address fill probability, spread capture, post-only constraints, inventory, latency decay, or realized maker execution.

baseline comparison：
- `basis > 0`: `2425` rows, `3` samples, hit rate `0.94600939`, mean future move `45.67216495` ticks, wrong-way count `69`, wrong-way rate `0.02845361`, p95 wrong-way loss `138` ticks, max wrong-way loss `180` ticks.
- `basis <= 0`: `7564` rows, `3` samples, hit rate `0.33853760`, mean future move `-16.94407721` ticks.

magnitude monotonicity：
- Small positive basis: `845` rows, mean future move `18.47337278` ticks, wrong-way count `42`, p95 wrong-way loss `169` ticks.
- Medium positive basis: `778` rows, mean future move `29.41516710` ticks, wrong-way count `20`, p95 wrong-way loss `120` ticks.
- Large positive basis: `802` rows, mean future move `90.09975062` ticks, wrong-way count `7`, p95 wrong-way loss `82` ticks.
- Mean future move strengthens monotonically by positive-basis magnitude, while wrong-way tail is heavier in the small positive-basis bucket.

wrong-way tail decomposition：
- Wrong-way rows: `69`.
- Strongest visible tail hypotheses:
  - `basis_magnitude_bucket=basis_positive_small`: `42` wrong-way rows, wrong-way share `0.60869565`, classification `promising_visible_filter`.
  - `hl_top5_imbalance_bucket=hl_top5_imbalance_negative_small`: `31` wrong-way rows, wrong-way share `0.44927536`, classification `promising_visible_filter`.
  - `hl_microprice_minus_mid_bucket=hl_microprice_minus_mid_negative_small`: `30` wrong-way rows, wrong-way share `0.43478261`, classification `promising_visible_filter`.
- Sample/time concentration remains material: `cross_exchange_public_sample_xemm_0603_quiet_a_event` contributes `60/69` wrong-way rows and is classified `needs_more_samples` rather than an accepted filter.

controlled-effect checks：
- Binance momentum control uses only `input_binance_mid_move_ticks_from_prev`; missing values are assigned to an unavailable bucket and no future label is substituted.
- Basis-positive retains nontrivial controlled effect in Binance momentum buckets.
- Basis-positive retains nontrivial controlled effect in Hyperliquid book-state buckets.
- Current evidence is not classified as merely a redundant Binance momentum or Hyperliquid book-state proxy.

tail filter feasibility：
- `basis_positive_small`, negative Hyperliquid top5 imbalance, and negative Hyperliquid microprice-minus-mid are plausible decision-time-visible tail hypotheses.
- Join age, spread, and zero Binance momentum are not accepted as filterable tail explanations because they cover too much of the basis-positive population or do not concentrate tail enough.
- Sample/time concentration remains a sample-design problem, not a production filter conclusion.

targeted sample design：
- Later collection should separately target active/high-vol, normal, quiet, wide-spread, basis-large, and HL-book-conflict regimes.
- Evidence gates for a later task: reduce max sample row share below `0.40`, reach at least `5` independent samples, include at least `2` active/high-vol, `2` normal, and `1` quiet windows, and require proposed filter concentration to repeat across at least `3` samples.
- Any collection is a separate future task requiring its own dispatch and QA. T001 did not collect data.

final recommendation：
- `targeted_collection_ready`
- Meaning: a later separately scoped targeted collection task can now be designed. This does not authorize collection inside T001.

artifact paths：
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_wrong_way_manifest.json`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_vs_nonpositive_baseline.csv`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_magnitude_bins.csv`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_wrong_way_rows.csv`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_tail_state_decomposition.csv`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_controlled_effects.csv`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_tail_filter_feasibility.csv`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_targeted_collection_plan.md`
- `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_next_step_recommendation.md`

verify：
- `python examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py --help` passed.
- `python -m py_compile examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py examples/hyperliquid/test_canonical_basis_positive_wrong_way_decomposition.py` passed.
- `python -m pytest examples/hyperliquid/test_canonical_basis_positive_wrong_way_decomposition.py` passed: `7 passed`.
- `python examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --t006-dir local_live_analysis/canonical_basis_positive_robustness_0608T006 --output-dir local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001` passed.
- `python -m json.tool local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/basis_positive_wrong_way_manifest.json` passed.
- Boundary text check passed for no new collection and no executable/private/order/live/case-library/shadow/promotion authorization.
- `git diff --check` passed.

blockers/caveats：
- No blockers.
- Evidence remains only read-only public/canonical event-mode context evidence. It is not execution-PnL proof.
- Final recommendation does not authorize strategy implementation, executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, case-library trigger, shadow decision, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.

done：
- Implemented T001 runner/tests, generated official artifacts, and updated task/tracking state to `待验收`.

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
