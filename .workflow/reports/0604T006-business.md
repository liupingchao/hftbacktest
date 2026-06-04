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
- `local_live_analysis/canonical_signal_quality_ranking_0604T006/`
- `progress.md`
- `task_plan.md`
- `findings.md`

action：
- Implemented a read-only canonical signal quality ranking runner that consumes `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/` through the `0604T005` canonical source-lock guard / `0604T004` loader path.
- Limited ranking to the four `0601T004` primary allowlist features:
  - `binance_top5_imbalance`
  - `binance_microprice_minus_mid_ticks`
  - `binance_mid_move_ticks_from_prev`
  - `binance_top5_bid_qty`
- Ranking dimensions include `1000ms+` direction consistency, `1000ms+` stable-row ratio, effect size, mean absolute correlation, independent future-row-delta support, short-horizon reliance, and sample-effect concentration.
- Generated task-scoped artifacts:
  - `signal_quality_ranking.csv`
  - `signal_quality_reject_watch_list.csv`
  - `signal_quality_ranking_manifest.json`
  - `signal_quality_ranking_report.md`
- Added focused tests for ranking order, watch classification, concentration penalty, short-horizon-only rejection, diagnostic-only input refusal, and artifact generation.

verify：
- `python examples/hyperliquid/canonical_signal_quality_ranking.py --help` passed.
- `python -m py_compile examples/hyperliquid/canonical_signal_quality_ranking.py examples/hyperliquid/test_canonical_signal_quality_ranking.py` passed.
- `python -m pytest examples/hyperliquid/test_canonical_signal_quality_ranking.py` passed: `6 passed`.
- `python examples/hyperliquid/canonical_signal_quality_ranking.py --input-dir local_live_analysis/event_mode_canonical_pricing_signal_0604T003 --output-dir local_live_analysis/canonical_signal_quality_ranking_0604T006` passed: `canonical_sample_count=3`, `ranked_feature_count=4`.
- `python -m json.tool local_live_analysis/canonical_signal_quality_ranking_0604T006/signal_quality_ranking_manifest.json` passed.
- `git diff --check` passed.

done：
- Ranking result:
  - Rank 1: `binance_mid_move_ticks_from_prev`, `keep_for_read_only_research`, score `0.8047634`.
  - Rank 2: `binance_top5_imbalance`, `watch_regime_dependent`, score `0.7184303`.
  - Rank 3: `binance_top5_bid_qty`, `watch_regime_dependent`, score `0.64361754`.
  - Rank 4: `binance_microprice_minus_mid_ticks`, `watch_regime_dependent`, score `0.5473174`.
- Interpretation:
  - `binance_mid_move_ticks_from_prev` remains the strongest global candidate using `1000ms+` canonical evidence.
  - `binance_top5_imbalance` remains a strong book-pressure candidate, but the current strict ranking labels it watch because of sample concentration and short-horizon reliance caveats.
  - `binance_top5_bid_qty` remains useful liquidity/context watch evidence.
  - `binance_microprice_minus_mid_ticks` remains regime-dependent watch evidence.
- This is read-only signal quality ranking only. It does not authorize regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, or promotion.

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
