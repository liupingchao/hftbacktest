```md
执行线程：
- 业务线程-research

任务ID：
- 0608T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0608T005.md`
- `.workflow/reports/0608T005-business.md`
- `examples/hyperliquid/canonical_basis_context_visibility.py`
- `examples/hyperliquid/test_canonical_basis_context_visibility.py`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_visibility_manifest.json`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_lineage_audit.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_positive_pattern_by_sample.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_timestamp_join_audit.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_horizon_persistence.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_contract_decision.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_context_visibility_report.md`

action：
- Created `0608T005` as a narrow read-only basis-context visibility / lineage diagnosis task.
- Implemented `examples/hyperliquid/canonical_basis_context_visibility.py`.
- Added focused tests in `examples/hyperliquid/test_canonical_basis_context_visibility.py`.
- Generated task-scoped artifacts under `local_live_analysis/canonical_basis_context_visibility_0608T005/`.

method：
- Consumed `0604T003` canonical event-mode artifacts through the accepted source-lock guard path.
- Validated T004 prerequisite: `context_basis_mid_ticks > 0` was `invalid_not_decision_visible`, stable across all 3 samples, and tail-acceptable.
- Reconstructed only `regime_011_1000_spread_10_20_ticks` rows at `horizon_ms=1000` with `context_basis_mid_ticks > 0`.
- Audited basis lineage against source join rows by checking `(binance_mid_px - hyperliquid_mid_px) / 0.1 == source basis_mid_ticks == context_basis_mid_ticks`.
- Audited timestamp/as-of cleanliness using `binance_local_ts`, `hyperliquid_decision_ts`, `binance_source_age_ms`, row quality fields, future joins, and missing joins.
- Checked sample concentration and 1000/5000/10000ms persistence, with 100/250ms treated only as watch/alias context.

input source-lock：
- Formal canonical input: `local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- T004 input: `local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004`
- Data contract input: `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004`
- Guard result: `canonical_sample_count=3`, `diagnostic_rejection_count=0`

T004 prerequisite check：
- T004 basis-positive row was present exactly once.
- T004 basis-positive `signal_validity=invalid_not_decision_visible`.
- T004 basis-positive `stability_classification=stable_all_samples`.
- T004 basis-positive `tail_risk_classification=tail_risk_acceptable_proxy`.

basis lineage：
- Formula: `(binance_mid_px - hyperliquid_mid_px) / 0.1`.
- `basis_lineage_audit.csv` rows: `57`.
- Lineage status: `57 / 57` rows are `lineage_confirmed_decision_time_formula`.
- Max formula absolute error: `0.0`.

timestamp/as-of diagnosis：
- `basis_timestamp_join_audit.csv` rows: `57`.
- Timestamp status: `57 / 57` rows are `timestamp_clean_asof`.
- Future input joins: `0`.
- Missing input joins: `0`.
- Rows remain primary usable with fresh Hyperliquid join-age context.

sample distribution：
- Total basis-positive rows: `57`.
- Sample rows:
  - `cross_exchange_public_sample_xemm_0603_quiet_a_event`: `22`, row share `0.38596491`, hit rate `0.90909091`, mean future move `61.81818182` ticks.
  - `cross_exchange_public_sample_xemm_0603_quiet_b_event`: `19`, row share `0.33333333`, hit rate `0.94736842`, mean future move `100` ticks.
  - `cross_exchange_public_sample_xemm_0603_quiet_c_event`: `16`, row share `0.28070175`, hit rate `1`, mean future move `143.75` ticks.
- Max sample row share: `0.38596491`, so the pattern is not single-sample dominated.

persistence results：
- `100ms`: watch/alias context only, hit rate `0.92982456`, mean `57.45614035` ticks.
- `250ms`: watch/alias context only, hit rate `0.92982456`, mean `57.45614035` ticks.
- `1000ms`: primary, hit rate `0.94736842`, mean `97.54385965` ticks.
- `5000ms`: diagnostic persistence, hit rate `0.84210526`, mean `134.03508772` ticks.
- `10000ms`: diagnostic persistence, hit rate `0.71929825`, mean `122.63157895` ticks.
- The 1000/5000/10000ms direction does not reverse.

contract decision：
- Final contract decision: `upgrade_to_context_only_supported`.
- Reason: basis is as-of clean and formula-derived; retain execution-PnL caveat but remove decision-visibility blocker.
- This does not upgrade basis into executable strategy PnL, private execution proof, case-library, shadow decision, or live/promotion readiness.

artifact paths：
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_visibility_manifest.json`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_lineage_audit.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_positive_pattern_by_sample.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_timestamp_join_audit.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_horizon_persistence.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_contract_decision.csv`
- `local_live_analysis/canonical_basis_context_visibility_0608T005/basis_context_visibility_report.md`

verify：
- `python examples/hyperliquid/canonical_basis_context_visibility.py --help` -> passed
- `python -m py_compile examples/hyperliquid/canonical_basis_context_visibility.py examples/hyperliquid/test_canonical_basis_context_visibility.py` -> passed
- `python -m pytest examples/hyperliquid/test_canonical_basis_context_visibility.py` -> `5 passed`
- `python examples/hyperliquid/canonical_basis_context_visibility.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --feature-validity-dir local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004 --data-contract-dir local_live_analysis/binance_led_hyperliquid_data_contract_0601T004 --output-dir local_live_analysis/canonical_basis_context_visibility_0608T005` -> passed, final contract decision `upgrade_to_context_only_supported`
- `python -m json.tool local_live_analysis/canonical_basis_context_visibility_0608T005/basis_visibility_manifest.json` -> passed
- Manual boundary check on `basis_context_visibility_report.md` -> contains read-only public-data proxy caveat and explicit prohibitions; no executable trading instruction, order side, quote price, size, private/order endpoint, shadow decision, live/default-on/tiny-live, or promotion claim.
- `git diff --check` -> passed

done：
- Implemented and verified the read-only basis context visibility / lineage runner and tests.
- Produced all required task-scoped artifacts.
- Assessed only `regime_011_1000_spread_10_20_ticks` and `context_basis_mid_ticks > 0`.
- Final task result is `upgrade_to_context_only_supported`, scoped only to contract visibility: basis may be treated as decision-time context in later read-only discussion while retaining execution-PnL caveat.
- This result does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decisions, or promotion.

blockers：
- 无

commit：
- 待提交

提交信息：
- Add basis context visibility diagnosis
```
