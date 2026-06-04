执行线程：
- 业务线程-research

任务ID：
- 0604T008

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T008.md`
- `.workflow/reports/0604T008-business.md`
- `examples/hyperliquid/canonical_signal_quality_ranking.py`
- `examples/hyperliquid/test_canonical_signal_quality_ranking.py`
- `local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_ranking_report.md`
- `local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_ranking_manifest.json`

action：
- Fixed the exact `0604T006` QA failure: `signal_quality_ranking_report.md` could describe `binance_top5_imbalance` as kept while the ranking CSV bucket was `watch_regime_dependent`.
- Replaced the static `Controller Interpretation Check` text in `canonical_signal_quality_ranking.py` with interpretation lines generated from `ranking_rows`, so each feature's report line mechanically includes its actual `final_bucket`.
- Added a focused regression test that verifies every report interpretation line matches the generated `final_bucket`, and specifically prevents the old `binance_top5_imbalance ... is kept` wording when the feature is watch-labeled.
- Regenerated `local_live_analysis/canonical_signal_quality_ranking_0604T006/` with the existing runner.

verify：
- `python examples/hyperliquid/canonical_signal_quality_ranking.py --help` passed.
- `python -m py_compile examples/hyperliquid/canonical_signal_quality_ranking.py examples/hyperliquid/test_canonical_signal_quality_ranking.py` passed.
- `python -m pytest examples/hyperliquid/test_canonical_signal_quality_ranking.py` passed: `7 passed`.
- `python examples/hyperliquid/canonical_signal_quality_ranking.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --output-dir local_live_analysis/canonical_signal_quality_ranking_0604T006` passed: `canonical_sample_count=3`, `ranked_feature_count=4`.
- `python -m json.tool local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_ranking_manifest.json` passed.
- Scripted report/CSV consistency check passed: `checked_features 4`, `missing []`.
- `git diff --check` passed.

done：
- The refreshed report now states:
  - `binance_mid_move_ticks_from_prev` is `keep_for_read_only_research`
  - `binance_top5_imbalance` is `watch_regime_dependent`
  - `binance_top5_bid_qty` is `watch_regime_dependent`
  - `binance_microprice_minus_mid_ticks` is `watch_regime_dependent`
- Ranking CSV semantics did not change; only report generation/report artifact and manifest regeneration metadata changed.
- This task only fixes T006 report consistency. It does not authorize scoring changes, allowlist changes, source-lock guard changes, regime selection, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.

blockers：
- 无

commit：
- c78cec3

提交信息：
- 0604T008 repair signal ranking report consistency
