```md
执行线程：
- 业务线程-research

任务ID：
- 0604T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T006.md`
- `.workflow/reports/0604T006-business.md`
- `examples/hyperliquid/canonical_signal_quality_ranking.py`
- `examples/hyperliquid/test_canonical_signal_quality_ranking.py`
- `local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_ranking.csv`
- `local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_reject_watch_list.csv`
- `local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_ranking_manifest.json`
- `local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_ranking_report.md`
- `progress.md`
- `task_plan.md`
- `findings.md`

action：
- Implemented a read-only canonical signal quality ranking runner over `0604T003` canonical event-mode artifacts through the `0604T004` loader/foundation.
- Locked ranking to the four `0601T004` primary Binance lead allowlist features:
  - `binance_top5_imbalance`
  - `binance_microprice_minus_mid_ticks`
  - `binance_mid_move_ticks_from_prev`
  - `binance_top5_bid_qty`
- Ranking method includes direction consistency, effect size, mean absolute correlation, usable primary row/canonical sample count, independent future-row-delta count, sample concentration, and 100/250ms versus 500ms+/1000ms+ reliance.
- Added focused tests for score/order behavior, keep/watch/reject classification, single-sample concentration penalty, short-horizon reliance penalty, and diagnostic-only synthetic input refusal.

ranking summary：
- `binance_mid_move_ticks_from_prev`: rank `1`, bucket `keep_for_read_only_research`, score `0.83656621`; matches controller interpretation as the most stable global signal candidate.
- `binance_top5_imbalance`: rank `2`, bucket `keep_for_read_only_research`, score `0.80557777`; matches controller interpretation as a strong book-pressure candidate.
- `binance_top5_bid_qty`: rank `3`, bucket `watch_regime_dependent`, score `0.70077258`; matches controller interpretation as useful liquidity/context rather than a simple global directional signal.
- `binance_microprice_minus_mid_ticks`: rank `4`, bucket `watch_regime_dependent`, score `0.5858533`; matches controller interpretation as regime-dependent.
- Reject bucket count: `0`.

verify：
- `python examples/hyperliquid/canonical_signal_quality_ranking.py --help`：通过。
- `python -m py_compile examples/hyperliquid/canonical_signal_quality_ranking.py examples/hyperliquid/test_canonical_signal_quality_ranking.py`：通过。
- `python -m pytest examples/hyperliquid/test_canonical_signal_quality_ranking.py`：通过，`4 passed`。
- `python examples/hyperliquid/canonical_signal_quality_ranking.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --output-dir local_live_analysis/canonical_signal_quality_ranking_0604T006`：通过，生成 `4` 个 ranked features。
- `python -m json.tool local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_ranking_manifest.json`：通过。
- `git diff --check`：通过。

done：
- Required artifacts exist under `local_live_analysis/canonical_signal_quality_ranking_0604T006/`.
- Manifest records `canonical_sample_count=3`, `diagnostic_rejection_count=0`, four primary allowlist features, bucket counts `keep=2 / watch=2 / reject=0`, and boundary flags for no collection/private/order/strategy/live/parameter/default-on/tiny-live/promotion.
- This is only read-only signal quality ranking. It does not authorize regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, or promotion.

blockers：
- 无

commit：
- 7ec5980

提交信息：
- Add canonical signal quality ranking
```
