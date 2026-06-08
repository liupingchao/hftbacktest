```md
执行线程：
- 业务线程-research

任务ID：
- 0608T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0608T003.md`
- `.workflow/reports/0608T003-business.md`
- `examples/hyperliquid/canonical_directional_momentum_viability.py`
- `examples/hyperliquid/test_canonical_directional_momentum_viability.py`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/directional_momentum_manifest.json`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/base_regime_directionality.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/feature_directionality_summary.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/cost_latency_adjusted_edge.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/tail_risk_summary.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/directional_candidate_watch_reject.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/directional_momentum_report.md`

action：
- Implemented `examples/hyperliquid/canonical_directional_momentum_viability.py`, a read-only public-data proxy runner for `regime_011_1000_spread_10_20_ticks`.
- The runner consumes `0604T003` canonical event-mode artifacts through the accepted source-lock guard path and refuses diagnostic-only synthetic formal input.
- The runner validates T002 maker executability prerequisite: `final_recommendation=reject_not_maker_executable`.
- The runner resolves row-level canonical pricing rows from `multi_sample_manifest.json` `samples[].pricing_signal_rows`; it does not require a root-level aggregate `pricing_signal_rows.csv`.
- Added focused tests covering canonical input acceptance, diagnostic-only input refusal, T002 prerequisite validation, feature-conditioned directionality, cost-adjusted taxonomy, and a weak-edge reject path.
- Generated task-scoped artifacts under `local_live_analysis/canonical_directional_momentum_viability_0608T003/`.

method：
- Assessment is limited to existing local canonical public artifacts and the T009/T002 target regime context: `horizon_ms=1000`, `primary_usable`, `fresh_0_50ms`, `spread_10_20_ticks`.
- Primary horizon is `1000ms`; `5000ms` and `10000ms` are diagnostic; `100/250ms` are parsed only as watch context and not used as independent canonical support.
- Directionality is evaluated from signed future mid move at the regime context.
- Cost proxy uses fixed, non-optimized assumptions recorded in the manifest: `fee_proxy_ticks=2`, `slippage_proxy_ticks=2`, `latency_decay_proxy_ticks=5`.
- Tail risk uses wrong-way rate and wrong-way loss magnitude as public-data proxy only.

input source-lock：
- Formal canonical input: `local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Candidate input: `local_live_analysis/canonical_regime_synthesis_0604T009`
- T002 maker executability input: `local_live_analysis/canonical_maker_executability_0608T002`
- Signal ranking context: `local_live_analysis/canonical_signal_quality_ranking_0604T006`
- Horizon/regime context: `local_live_analysis/canonical_horizon_regime_diagnostics_0604T007`
- Guard result: `canonical_sample_count=3`, `diagnostic_rejection_count=0`

T002 prerequisite check：
- T002 manifest task id: `0608T002`
- T002 assessed candidate: `regime_011_1000_spread_10_20_ticks`
- T002 final recommendation: `reject_not_maker_executable`
- T003 used T002 only as accepted read-only public-data proxy evidence; it did not authorize maker case-library, strategy implementation, live, or promotion.

directionality results：
- Final recommendation: `reject_directional_edge_unstable`
- Directionality classification: `directional_signal_reject`
- Base primary rows: `242`
- Sample count: `3`
- Positive / negative future counts: `124 / 118`
- Direction hit rate: `0.51239669`
- Mean signed future move ticks: `-0.92975207`
- Mean absolute future move ticks: `72.08677686`
- Per-sample signed edge ticks:
  - `xemm_0603_quiet_a_event`: `10.51282051`
  - `xemm_0603_quiet_b_event`: `-9.04040404`
  - `xemm_0603_quiet_c_event`: `2.5`
- Stability classification: `sample_unstable_reject`

cost / latency proxy assumptions and result：
- Gross directional edge proxy ticks: `0.92975207`
- Fee proxy ticks: `2`
- Slippage proxy ticks: `2`
- Latency decay proxy ticks: `5`
- Net directional edge proxy ticks: `-8.07024793`
- Cost-adjusted viability: `net_edge_negative_proxy`
- Cost assumptions are fixed manifest fields and were not optimized to pass.

tail-risk results：
- Wrong-way rate: `0.48760331`
- Mean wrong-way loss ticks: `74.87288136`
- p95 wrong-way loss ticks: `205.75`
- Max wrong-way loss ticks: `275`
- Tail risk classification: `tail_risk_reject`

final recommendation：
- `reject_directional_edge_unstable`
- Reason: at the accepted canonical 1000ms regime context, the observed public-data directionality is not stable enough for directional case-library follow-up. The primary row mass exists, but direction hit rate is near random, per-sample signed edge flips sign, conservative net edge proxy is negative, and tail-risk proxy is rejected.

artifact paths：
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/directional_momentum_manifest.json`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/base_regime_directionality.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/feature_directionality_summary.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/cost_latency_adjusted_edge.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/tail_risk_summary.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/directional_candidate_watch_reject.csv`
- `local_live_analysis/canonical_directional_momentum_viability_0608T003/directional_momentum_report.md`

verify：
- `python examples/hyperliquid/canonical_directional_momentum_viability.py --help` -> passed
- `python -m py_compile examples/hyperliquid/canonical_directional_momentum_viability.py examples/hyperliquid/test_canonical_directional_momentum_viability.py` -> passed
- `python -m pytest examples/hyperliquid/test_canonical_directional_momentum_viability.py` -> `5 passed`
- `python -m json.tool local_live_analysis/event_mode_canonical_pricing_signal_0604T003/multi_sample_manifest.json` -> passed
- `python examples/hyperliquid/canonical_directional_momentum_viability.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --candidate-dir local_live_analysis/canonical_regime_synthesis_0604T009 --maker-executability-dir local_live_analysis/canonical_maker_executability_0608T002 --signal-ranking-dir local_live_analysis/canonical_signal_quality_ranking_0604T006 --horizon-regime-dir local_live_analysis/canonical_horizon_regime_diagnostics_0604T007 --output-dir local_live_analysis/canonical_directional_momentum_viability_0608T003` -> passed, final recommendation `reject_directional_edge_unstable`
- `python -m json.tool local_live_analysis/canonical_directional_momentum_viability_0608T003/directional_momentum_manifest.json` -> passed
- Manual boundary check on `directional_momentum_report.md` -> contains read-only public-data proxy caveat and explicit prohibitions; no executable trading instruction, order side, quote price, size, private/order endpoint, live/default-on/tiny-live, or promotion claim.
- `git diff --check` -> passed

done：
- Implemented and verified the read-only directional momentum viability runner and tests.
- Produced all required task-scoped artifacts.
- Assessed only `regime_011_1000_spread_10_20_ticks`.
- Final task result is `reject_directional_edge_unstable`.
- This result does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decisions, or promotion.

blockers：
- 无

commit：
- e1dc1b9

提交信息：
- Add directional momentum viability assessment
```
