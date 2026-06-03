# 线程回报

执行线程：
- 业务线程-python

任务ID：
- 0601T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0601T003.md`
- `.workflow/reports/0601T003-business.md`
- `examples/hyperliquid/cross_exchange_lead_lag_analysis.py`
- `examples/hyperliquid/test_cross_exchange_lead_lag_analysis.py`
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/**`

action：
- 将 `0601T003` 从草案改为正式任务，并按用户确认的分析口径执行。
- 新增 `examples/hyperliquid/cross_exchange_lead_lag_analysis.py`，实现只读 lead-lag stability analyzer。
- 新增 focused tests，覆盖 horizon outcome no-future construction、z-score normalization、Binance volatility regime bucket、verdict thresholding、真实 `0601T002` artifact 生成。
- 生成任务产物到 `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/`。
- QA 前复查时补强 analyzer 可审计性：
  - `_verdict()` 改为使用实际传入的 `min_bucket_rows`，避免非默认阈值复现时仍使用全局 `100`。
  - `lead_lag_horizon_summary.csv`、`feature_effect_by_regime.csv`、`basis_response_summary.csv`、`venue_state_conditioning_summary.csv` 增加 `effective_future_age_ms_min/mean/max`，显式暴露 `at or after target horizon` 在 500ms Hyperliquid decision grid 下的实际未来行间隔。
  - 默认运行 verdict counts 未变化。

input：
- 唯一实证输入：`local_live_analysis/cross_exchange_lead_lag_join_0601T002/**`
- 主输入 CSV：`cross_exchange_joined_features.csv`
- 质量输入：`join_quality_summary.json`
- basis 输入：`basis_dislocation_summary.csv`
- manifest 输入：`sample_manifest.json`, `run_manifest.json`

analysis policy：
- Primary analysis 仅使用 `joined_row_quality=primary_usable` 且无 future/missing Binance join 的 rows。
- Input rows：`3599`
- Primary rows：`3596`
- Excluded rows：`3`
- Horizons：`100, 250, 500, 1000, 5000, 10000 ms`
- Lead features：所有 Binance top5-derived numeric features：
  - `binance_top5_imbalance`
  - `binance_top5_microprice_px`
  - `binance_microprice_minus_mid_ticks`
  - `binance_top5_bid_qty`
  - `binance_top5_ask_qty`
  - `binance_top5_total_qty`
  - `binance_mid_move_ticks_from_prev`
  - `binance_rolling_abs_mid_move_ticks_5`
  - `binance_rolling_rv_ticks_20`
- Feature policy：primary rows 上拟合 z-score 后分析。
- Lag outcomes：
  - `hyperliquid_mid_move_ticks`
  - `hyperliquid_spread_change_ticks`
  - `hyperliquid_top5_imbalance_change`
  - `hyperliquid_microprice_minus_mid_change_ticks`
  - `basis_mid_response_ticks`
  - `basis_microprice_response_ticks`
- Regime bucket：基于 `binance_rolling_rv_ticks_20` 的 Binance volatility buckets：`zero`, `positive_low`, `positive_high`。
- Minimum rows：`100`。
- Verdict thresholds：
  - `abs(pearson_corr) >= 0.05`
  - tick outcome high-vs-low z-score effect `>= 0.25` ticks
  - unitless outcome high-vs-low z-score effect `>= 0.01`
  - 至少 2 个同方向 passing horizons 为 `stable_enough_for_pricing_research`

generated artifacts：
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/run_manifest.json`
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/analysis_quality_summary.json`
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/lead_lag_horizon_summary.csv`
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/feature_effect_by_regime.csv`
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/basis_response_summary.csv`
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/venue_state_conditioning_summary.csv`
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/lead_lag_feature_verdicts.csv`
- `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/lead_lag_recommendation.md`

row counts：
- `lead_lag_horizon_summary.csv`：`324`
- `feature_effect_by_regime.csv`：`1188`
- `basis_response_summary.csv`：`108`
- `venue_state_conditioning_summary.csv`：`216`
- `lead_lag_feature_verdicts.csv`：`54`
- Horizon observations：`129246`

verdicts：
- `stable_enough_for_pricing_research`：`18`
- `watch_only`：`6`
- `unstable`：`30`
- `insufficient_samples`：`0`

key interpretation：
- 本任务发现 `18` 个 feature/outcome pair 达到 read-only pricing-research follow-up 阈值。
- 该结论只表示“可进入后续只读 pricing-signal runner design 研究”，不表示 strategy-ready、signal-ready、default-on-ready、tiny-live-ready 或 promotion-ready。
- `lead_lag_recommendation.md` 明确记录该结果不是策略信号、不是参数搜索、不是 promotion artifact。
- Horizon outcome 使用 `future_decision_ts >= hyperliquid_decision_ts + horizon_ms` 的第一行；产物现已报告 effective future-age，因此 `100/250/500ms` 在约 500ms decision grid 下可能共享同一未来行这一事实可被 QA 和后续任务直接审计。

boundary flags：
- no private keys：true
- no private account endpoints：true
- no order endpoints：true
- no order lifecycle：true
- no strategy process / live trading bot：true
- no parameter search：true
- no default-on：true
- no tiny-live：true
- no promotion：true
- Binance / Hyperliquid trade pressure policy：`disabled_unverified_side_semantics`

verify：
- `python examples/hyperliquid/cross_exchange_lead_lag_analysis.py --help`
  - 通过。
- `python -m pytest examples/hyperliquid/test_cross_exchange_lead_lag_analysis.py -q`
  - 通过：`4 passed`。
- `python examples/hyperliquid/cross_exchange_lead_lag_analysis.py --input-dir local_live_analysis/cross_exchange_lead_lag_join_0601T002 --output-dir local_live_analysis/cross_exchange_lead_lag_analysis_0601T003`
  - 通过：`primary_row_count=3596`, `excluded_row_count=3`, `stable_enough_for_pricing_research=18`, `watch_only=6`, `unstable=30`。
- `python -m json.tool local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/run_manifest.json`
  - 通过。
- `python -m json.tool local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/analysis_quality_summary.json`
  - 通过。
- `python -m py_compile examples/hyperliquid/cross_exchange_lead_lag_analysis.py examples/hyperliquid/test_cross_exchange_lead_lag_analysis.py`
  - 通过。
- `git diff --check`
  - 通过。

done：
- `0601T003` 已生成 read-only lead-lag stability evidence，可供后续总控决定是否设计 read-only pricing-signal runner。
- 输出不包含策略实现、参数搜索、live/default-on/tiny-live/promotion 结论。
- QA 前复查补丁已完成；默认 verdict 结论不变，仅修复非默认最小样本阈值复现行为并补充 future-age 审计字段。

blockers：
- 无。

commit：
- `1ae940f`
- `ff50c59`

提交信息：
- `Add cross-exchange lead lag analysis`
- `Harden lead lag analysis auditability`
