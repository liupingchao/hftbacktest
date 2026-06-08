```md
执行线程：
- 业务线程-research

任务ID：
- 0608T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0608T002.md`
- `.workflow/reports/0608T002-business.md`
- `examples/hyperliquid/canonical_maker_executability.py`
- `examples/hyperliquid/test_canonical_maker_executability.py`
- `local_live_analysis/canonical_maker_executability_0608T002/maker_executability_manifest.json`
- `local_live_analysis/canonical_maker_executability_0608T002/regime_executability_summary.csv`
- `local_live_analysis/canonical_maker_executability_0608T002/fill_opportunity_proxy.csv`
- `local_live_analysis/canonical_maker_executability_0608T002/spread_capture_adverse_selection.csv`
- `local_live_analysis/canonical_maker_executability_0608T002/quote_churn_post_only_latency.csv`
- `local_live_analysis/canonical_maker_executability_0608T002/inventory_exposure_what_if.csv`
- `local_live_analysis/canonical_maker_executability_0608T002/maker_executability_report.md`

action：
- Implemented `examples/hyperliquid/canonical_maker_executability.py`, a read-only public-data proxy runner for the single accepted T009 candidate `regime_011_1000_spread_10_20_ticks`.
- The runner consumes `0604T003` canonical event-mode artifacts through the accepted `canonical_event_mode_evidence` source-lock guard and refuses diagnostic-only synthetic inputs.
- The runner validates the T009 candidate manifest, T006 signal-ranking manifest, and T007 horizon/regime manifest before assessment.
- Added focused tests covering canonical input acceptance, diagnostic-only synthetic refusal, candidate-id filtering / ignored extra candidates, final recommendation taxonomy generation, and a watch path for row-count mismatch.
- Generated task-scoped artifacts under `local_live_analysis/canonical_maker_executability_0608T002/`.

method：
- Assessment is limited to existing local canonical public artifacts and the T009 candidate context: `horizon_ms=1000`, `primary_usable`, `fresh_0_50ms`, `spread_10_20_ticks`.
- Row-level public proxy rows are read from each canonical sample's `pricing_signal_rows.csv` and matched to the candidate context.
- Fill opportunity, spread capture, adverse selection, quote churn, post-only/latency, and inventory are reported as public-data proxies only.
- Real fill probability, real post-only reject rate, private order lifecycle, and inventory execution behavior are not claimed.

input source-lock：
- Formal canonical input: `local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Candidate input: `local_live_analysis/canonical_regime_synthesis_0604T009`
- Signal ranking context: `local_live_analysis/canonical_signal_quality_ranking_0604T006`
- Horizon/regime context: `local_live_analysis/canonical_horizon_regime_diagnostics_0604T007`
- Guard result: `canonical_sample_count=3`, `diagnostic_rejection_count=0`

per-dimension results：
- Fill opportunity proxy: `fillable_proxy_supported`
  - Public proxy rows `242 / 242`
  - Per-sample trigger counts: `39 / 99 / 104`
  - Sample concentration `0.42975207`
  - Trigger rate proxy `1.17328054` per minute
- Spread capture proxy: `spread_capture_negative`
  - Decision spread `20` ticks, half-spread `10` ticks
  - Adverse selection proxy `72.08677686` ticks
  - Net spread-capture proxy `-62.08677686` ticks
- Adverse selection proxy: `adverse_selection_reject`
- Quote churn proxy: `manageable_churn`
- Post-only / latency proxy: `reject_post_only_risk` / `latency_reject`
  - Fast-move-after-decision proxy `0.80991736`
  - Binance source age p50/p95/p99 `13.384776 / 25.3490692 / 34.90576522` ms
- Inventory what-if: `flat_only_watch`
  - Long/short/reduce/add side cannot be resolved without outputting maker side or using private/order lifecycle evidence.

final recommendation：
- `reject_not_maker_executable`
- Reason: public proxy has non-zero, multi-sample fill opportunity, but adverse-selection and post-only/latency proxy risk clearly consume the available half-spread under the current read-only public-data assessment.

verify：
- `python examples/hyperliquid/canonical_maker_executability.py --help` -> passed
- `python -m py_compile examples/hyperliquid/canonical_maker_executability.py examples/hyperliquid/test_canonical_maker_executability.py` -> passed
- `python -m pytest examples/hyperliquid/test_canonical_maker_executability.py` -> `5 passed`
- `python examples/hyperliquid/canonical_maker_executability.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --candidate-dir local_live_analysis/canonical_regime_synthesis_0604T009 --signal-ranking-dir local_live_analysis/canonical_signal_quality_ranking_0604T006 --horizon-regime-dir local_live_analysis/canonical_horizon_regime_diagnostics_0604T007 --output-dir local_live_analysis/canonical_maker_executability_0608T002` -> passed, final recommendation `reject_not_maker_executable`
- `python -m json.tool local_live_analysis/canonical_maker_executability_0608T002/maker_executability_manifest.json` -> passed
- Manual boundary check on `maker_executability_report.md` -> contains only proxy caveats and explicit prohibitions; no maker side, quote price, size, cancel rule, strategy action, private/order endpoint, live/default-on/tiny-live, or promotion claim.
- `git diff --check` -> passed

done：
- Implemented and verified the read-only maker executability assessment runner and tests.
- Produced all required task-scoped artifacts.
- Assessed only `regime_011_1000_spread_10_20_ticks`.
- Final task result is `reject_not_maker_executable`.
- This result does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decisions, or promotion.

blockers：
- 无

commit：
- 0196149

提交信息：
- Add canonical maker executability assessment
```
