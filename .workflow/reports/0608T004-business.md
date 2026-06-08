```md
执行线程：
- 业务线程-research

任务ID：
- 0608T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0608T004.md`
- `.workflow/reports/0608T004-business.md`
- `examples/hyperliquid/canonical_feature_conditioned_signal_validity.py`
- `examples/hyperliquid/test_canonical_feature_conditioned_signal_validity.py`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_validity_manifest.json`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_pattern_validity_summary.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/per_sample_pattern_stability.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_redundancy_collinearity.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/decision_visibility_caveat_audit.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/cost_tail_validity.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_conditioned_validity_report.md`

action：
- Implemented `examples/hyperliquid/canonical_feature_conditioned_signal_validity.py`, a read-only validity diagnosis runner for fixed T003-observed feature-conditioned patterns under `regime_011_1000_spread_10_20_ticks`.
- The runner consumes `0604T003` canonical event-mode artifacts through the accepted source-lock guard path and refuses diagnostic-only synthetic formal input.
- The runner validates T002 prerequisite `reject_not_maker_executable` and T003 prerequisite `reject_directional_edge_unstable`.
- The runner resolves row-level canonical pricing rows from `multi_sample_manifest.json` `samples[].pricing_signal_rows`.
- Added focused tests covering canonical input acceptance, diagnostic-only refusal, T002/T003 prerequisite validation, decision-visibility caveats, redundancy/collinearity, cost/tail reject path, and final taxonomy generation.
- Generated task-scoped artifacts under `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/`.

method：
- Assessment is limited to existing local canonical public artifacts and the target regime context: `horizon_ms=1000`, `primary_usable`, `fresh_0_50ms`, `spread_10_20_ticks`.
- Fixed feature scope is exactly the T004 list: `context_basis_mid_ticks`, `context_hyperliquid_top5_imbalance`, `context_hyperliquid_microprice_minus_mid_ticks`, `input_binance_top5_imbalance`, and `input_binance_microprice_minus_mid_ticks`.
- Optional diagnostic comparison is limited to `input_binance_mid_move_ticks_from_prev` and `input_binance_top5_bid_qty`.
- Evaluated sign-positive, sign-negative, and absolute top-quartile buckets without broad feature search or parameter sweep.
- For each pattern, computed row/sample mass, per-sample stability, 1000ms direction consistency, 5000/10000ms diagnostic persistence, 100/250ms watch-only context, fixed cost proxy, wrong-way/tail proxy, decision visibility, and redundancy/collinearity.

input source-lock：
- Formal canonical input: `local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Candidate input: `local_live_analysis/canonical_regime_synthesis_0604T009`
- T002 maker executability input: `local_live_analysis/canonical_maker_executability_0608T002`
- T003 directional momentum input: `local_live_analysis/canonical_directional_momentum_viability_0608T003`
- Data contract input: `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004`
- Guard result: `canonical_sample_count=3`, `diagnostic_rejection_count=0`

T002/T003 prerequisite checks：
- T002 final recommendation: `reject_not_maker_executable`
- T003 final recommendation: `reject_directional_edge_unstable`
- T004 used both only as accepted read-only public-data proxy evidence; it did not authorize maker case-library, directional case-library, strategy implementation, shadow decisions, live, or promotion.

fixed feature validity results：
- Final recommendation: `watch_needs_contract_visibility_clarification`
- Valid supported pattern count: `0`
- Valid watch pattern count: `0`
- Invalid pattern count: `21`
- Observed primary row count: `242`
- Per-sample primary row counts: `39 / 99 / 104`

decision-visibility caveats：
- `context_basis_mid_ticks` maps to `basis_mid_dislocation` and is `diagnostic_only / diagnostic_context` under `0601T004`; its strong sign-positive bucket is therefore `invalid_not_decision_visible`, not a case-design candidate.
- Hyperliquid top5 imbalance and Hyperliquid microprice-minus-mid are decision-time visible context fields, but they are perfectly redundant in the target rows (`pearson_corr=1`), so their strong buckets are classified `invalid_redundant_or_leakage_risk`.
- Binance top5 imbalance and Binance microprice-minus-mid are decision-time-visible primary allowlist fields, but are highly redundant in the target rows (`pearson_corr=0.99476462`), so their strong sign buckets are also `invalid_redundant_or_leakage_risk`.
- Optional `input_binance_top5_bid_qty` top-quartile diagnostic is visible and not highly redundant, but remains `invalid_tail_or_cost_reject` because tail-risk proxy is rejected.

cost/tail results：
- Cost assumptions are fixed, conservative, and not optimized: fee `2` ticks, slippage `2` ticks, latency decay `5` ticks.
- Strong basis sign-positive bucket has net edge proxy `88.54385965` ticks and `tail_risk_acceptable_proxy`, but fails visibility because basis is contract-caveated diagnostic context only.
- Strong Hyperliquid context and Binance imbalance/microprice buckets can show positive net edge proxy, but fail redundancy and/or tail-risk checks.
- Optional bid-qty top-quartile has net edge proxy `44.68852459` ticks but `tail_risk_reject`.

final recommendation：
- `watch_needs_contract_visibility_clarification`
- This means the only non-tail/non-redundancy-looking strong pattern depends on contract-caveated basis context; it does not mean Regime 011 has a candidate for read-only case design.
- No pattern met the strict acceptance line for `candidate_for_read_only_case_design_discussion`.

artifact paths：
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_validity_manifest.json`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_pattern_validity_summary.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/per_sample_pattern_stability.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_redundancy_collinearity.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/decision_visibility_caveat_audit.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/cost_tail_validity.csv`
- `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_conditioned_validity_report.md`

verify：
- `python examples/hyperliquid/canonical_feature_conditioned_signal_validity.py --help` -> passed
- `python -m py_compile examples/hyperliquid/canonical_feature_conditioned_signal_validity.py examples/hyperliquid/test_canonical_feature_conditioned_signal_validity.py` -> passed
- `python -m pytest examples/hyperliquid/test_canonical_feature_conditioned_signal_validity.py` -> `5 passed`
- `python examples/hyperliquid/canonical_feature_conditioned_signal_validity.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --candidate-dir local_live_analysis/canonical_regime_synthesis_0604T009 --maker-executability-dir local_live_analysis/canonical_maker_executability_0608T002 --directional-momentum-dir local_live_analysis/canonical_directional_momentum_viability_0608T003 --signal-ranking-dir local_live_analysis/canonical_signal_quality_ranking_0604T006 --horizon-regime-dir local_live_analysis/canonical_horizon_regime_diagnostics_0604T007 --data-contract-dir local_live_analysis/binance_led_hyperliquid_data_contract_0601T004 --output-dir local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004` -> passed, final recommendation `watch_needs_contract_visibility_clarification`
- `python -m json.tool local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004/feature_validity_manifest.json` -> passed
- Manual boundary check on `feature_conditioned_validity_report.md` -> contains read-only public-data proxy caveat and explicit prohibitions; no executable trading instruction, order side, quote price, size, private/order endpoint, shadow decision, live/default-on/tiny-live, or promotion claim.
- `git diff --check` -> passed

done：
- Implemented and verified the read-only feature-conditioned signal validity runner and tests.
- Produced all required task-scoped artifacts.
- Assessed only `regime_011_1000_spread_10_20_ticks` and the fixed T004 feature list.
- Final task result is `watch_needs_contract_visibility_clarification` with `0` supported/watch-valid patterns and `21` invalid patterns.
- This result does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decisions, or promotion.

blockers：
- 无

commit：
- 6409f6b

提交信息：
- Add feature-conditioned signal validity diagnosis
```
