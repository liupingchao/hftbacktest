执行线程：
- 业务线程-research

任务ID：
- 0604T009

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T009.md`
- `.workflow/reports/0604T009-business.md`
- `examples/hyperliquid/canonical_regime_synthesis.py`
- `examples/hyperliquid/test_canonical_regime_synthesis.py`
- `local_live_analysis/canonical_regime_synthesis_0604T009/candidate_regime_definitions.csv`
- `local_live_analysis/canonical_regime_synthesis_0604T009/candidate_regime_evidence_summary.csv`
- `local_live_analysis/canonical_regime_synthesis_0604T009/candidate_regime_watch_reject_list.csv`
- `local_live_analysis/canonical_regime_synthesis_0604T009/canonical_regime_synthesis_manifest.json`
- `local_live_analysis/canonical_regime_synthesis_0604T009/canonical_regime_synthesis_report.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Implemented `examples/hyperliquid/canonical_regime_synthesis.py`, a read-only synthesis runner over accepted canonical event-mode artifacts.
- The runner consumes `0604T003` canonical evidence through the `0604T004/0604T005` loader/source-lock guard path and refuses diagnostic-only synthetic inputs.
- It combines the T008-refreshed `0604T006` signal ranking with `0604T007` horizon/regime diagnostics.
- It locks promoted candidate definitions to the strict allowlist:
  - primary anchor: `binance_mid_move_ticks_from_prev`
  - secondary context only: `binance_top5_imbalance`, `binance_top5_bid_qty`, `binance_microprice_minus_mid_ticks`
  - allowed context fields: `hyperliquid_context_quality`, `hyperliquid_join_age_bucket`, `hyperliquid_spread_bucket`
- Added focused tests covering canonical-input refusal, short-horizon watch behavior, direction disagreement rejection, secondary-feature context-only handling, artifact output, and boundary flags.
- Generated task-scoped artifacts under `local_live_analysis/canonical_regime_synthesis_0604T009/`.

verify：
- `python examples/hyperliquid/canonical_regime_synthesis.py --help` passed.
- `python -m py_compile examples/hyperliquid/canonical_regime_synthesis.py examples/hyperliquid/test_canonical_regime_synthesis.py` passed.
- `python -m pytest examples/hyperliquid/test_canonical_regime_synthesis.py` passed: `6 passed`.
- `python -m pytest examples/hyperliquid/test_canonical_event_mode_evidence.py examples/hyperliquid/test_canonical_signal_quality_ranking.py examples/hyperliquid/test_canonical_horizon_regime_diagnostics.py examples/hyperliquid/test_canonical_regime_synthesis.py` passed: `28 passed`.
- `python examples/hyperliquid/canonical_regime_synthesis.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --signal-ranking-dir local_live_analysis/canonical_signal_quality_ranking_0604T006 --horizon-regime-dir local_live_analysis/canonical_horizon_regime_diagnostics_0604T007 --output-dir local_live_analysis/canonical_regime_synthesis_0604T009` passed: `canonical_sample_count=3`, `candidate_count=1`.
- `python -m json.tool local_live_analysis/canonical_regime_synthesis_0604T009/canonical_regime_synthesis_manifest.json` passed.
- Scripted candidate-boundary check passed: `checked_candidates 1`, classification counts `candidate_for_milestone3_executability=1`, `watch_needs_more_samples=6`, `reject_unstable_direction=9`, `reject_concentrated_or_aliased=2`.

done：
- Synthesis produced one read-only candidate regime eligible only for later Milestone 3 executability assessment:
  - `regime_011_1000_spread_10_20_ticks`
  - primary anchor `binance_mid_move_ticks_from_prev`
  - horizon `1000ms`
  - context `hyperliquid_context_quality=primary_usable`, `hyperliquid_join_age_bucket=fresh_0_50ms`, `hyperliquid_spread_bucket=spread_10_20_ticks`
  - row count `242`, sample count `3`, effective future-row-delta support `3`, classification `candidate_for_milestone3_executability`
- Other synthesized rows were not promoted: `6` watch rows, `9` unstable-direction rejects, and `2` concentrated/aliased rejects.
- This is only read-only regime synthesis / candidate definition. It does not authorize maker side, quote behavior, order behavior, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, or promotion.

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
