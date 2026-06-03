```md
执行线程：
- 业务线程-research

任务ID：
- 0604T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T001.md`
- `.workflow/reports/0604T001-business.md`
- `examples/hyperliquid/hyperliquid_raw_alignment.py`
- `examples/hyperliquid/cross_exchange_lead_lag_analysis.py`
- `examples/hyperliquid/binance_led_pricing_signal_runner.py`
- `examples/hyperliquid/test_hyperliquid_raw_alignment.py`
- `examples/hyperliquid/test_cross_exchange_lead_lag_analysis.py`
- `examples/hyperliquid/test_binance_led_pricing_signal_runner.py`
- `local_live_analysis/horizon_alias_repair_0604T001/**`

action：
- 新增 Hyperliquid event-driven decision rows 模式：`hyperliquid_raw_alignment.py --decision-mode event` 使用真实 `l2Book` rows 作为 decision/outcome rows，不再强制固定 500ms synthetic grid。
- 保留默认 synthetic mode 以兼容旧 artifacts；event mode 在 metrics/run manifest 中记录 `decision_mode=event`、`decision_row_count`、`event_decision_count`，并在 event mode 下将 `synthetic_decision_count` 置为 `0`。
- 修复 `cross_exchange_lead_lag_analysis.py` verdict 逻辑：
  - 每个 horizon row 输出 `effective_future_row_delta_min/mean/max`。
  - `stable_enough_for_pricing_research` 现在要求通过阈值的 horizons 至少覆盖两个独立 future-row delta。
  - nominal horizons 指向同一 future row 时只能进入 `watch_only`，不再被当作多个独立证据。
- 扩展 `binance_led_pricing_signal_runner.py`：
  - `pricing_signal_rows.csv` 输出 `effective_future_row_delta`。
  - `horizon_label_summary.csv` 输出 future-row-delta min/mean/max。
  - run manifest timestamp policy 记录 `future_row_delta_reported=true`。
- 新增 focused tests 覆盖：
  - event-mode alignment 从真实 l2Book rows 生成 decision rows。
  - aliased nominal horizons 不再被判成 stable。
  - pricing-signal rows / horizon summary 输出 future-row-delta diagnostics。
- 未重新采集数据；task-scoped rerun 只使用既有 `xemm_0603_quiet_a` local public raw/artifacts。

task-scoped rerun：
- 输出目录：`local_live_analysis/horizon_alias_repair_0604T001/`
- event-mode Hyperliquid alignment:
  - `decision_mode=event`
  - `decision_row_count=3332`
  - `event_decision_count=3332`
  - `synthetic_decision_count=0`
  - `decision_join_coverage=1.0`
  - `future_join_count=0`
  - `missing_join_count=0`
  - `sample_classification=passes_pricing_research_market_view`
- event-mode cross-exchange join:
  - `joined_feature_rows=3332`
  - `primary_usable_row_count=3332`
  - `future_join_count=0`
  - `missing_binance_join_count=0`
- event-mode lead-lag analyzer:
  - `primary_row_count=3332`
  - verdict counts: `17 stable_enough_for_pricing_research / 3 watch_only / 34 unstable`
  - old synthetic `xemm_0603_quiet_a` had `19 stable / 1 watch / 34 unstable`; de-aliasing demoted nominal-horizon-only evidence.
- event-mode pricing-signal runner:
  - `primary_rows=3332`
  - `pricing_signal_rows=19956`
  - `recommendation=needs_more_public_samples`

horizon alias findings：
- Event mode removes the artificial fixed 500ms decision grid, but public `l2Book` cadence still limits short-horizon independence.
- In event-mode `xemm_0603_quiet_a`, `100/250/500ms` all point to the same future row for `2393/3330` comparable source rows, not `100%` as in synthetic mode.
- Event-mode future-row delta distribution:
  - `100ms`: mostly delta `1`, mean effective age `541.721ms`
  - `250ms`: mostly delta `1`, mean effective age `557.644ms`
  - `500ms`: delta `1/2/3`, mean effective age `705.653ms`
  - `1000ms`: mostly delta `2/3`, mean effective age `1170.181ms`
  - `5000ms`: mostly delta `10`, mean effective age `5365.108ms`
  - `10000ms`: mostly delta `19`, mean effective age `10272.366ms`
- Interpretation: event mode fixes the artifact-level bug, but raw public Hyperliquid l2Book cadence still means 100/250ms book-outcome research often has limited independent evidence.

verify：
- `python -m py_compile examples/hyperliquid/hyperliquid_raw_alignment.py examples/hyperliquid/cross_exchange_lead_lag_analysis.py examples/hyperliquid/binance_led_pricing_signal_runner.py` passed.
- `python -m pytest examples/hyperliquid/test_hyperliquid_raw_alignment.py examples/hyperliquid/test_cross_exchange_lead_lag_analysis.py examples/hyperliquid/test_binance_led_pricing_signal_runner.py` passed: `13 passed`.
- `python examples/hyperliquid/hyperliquid_raw_alignment.py --help` passed and exposes `--decision-mode {synthetic,event}`.
- `python examples/hyperliquid/cross_exchange_lead_lag_analysis.py --help` passed.
- `python examples/hyperliquid/binance_led_pricing_signal_runner.py --help` passed.
- Task-scoped event-mode rerun commands passed:
  - `python examples/hyperliquid/hyperliquid_raw_alignment.py ... --decision-mode event`
  - `python examples/hyperliquid/cross_exchange_lead_lag_join.py ...`
  - `python examples/hyperliquid/cross_exchange_lead_lag_analysis.py ...`
  - `python examples/hyperliquid/binance_led_pricing_signal_runner.py ...`
- JSON checks passed:
  - `python -m json.tool local_live_analysis/horizon_alias_repair_0604T001/cross_exchange_public_sample_xemm_0603_quiet_a_event/hyperliquid_public_sample/alignment/metrics.json`
  - `python -m json.tool local_live_analysis/horizon_alias_repair_0604T001/cross_exchange_public_sample_xemm_0603_quiet_a_event/hyperliquid_public_sample/alignment/run_manifest.json`
  - `python -m json.tool local_live_analysis/horizon_alias_repair_0604T001/cross_exchange_lead_lag_join_xemm_0603_quiet_a_event/run_manifest.json`
  - `python -m json.tool local_live_analysis/horizon_alias_repair_0604T001/cross_exchange_lead_lag_join_xemm_0603_quiet_a_event/join_quality_summary.json`
  - `python -m json.tool local_live_analysis/horizon_alias_repair_0604T001/cross_exchange_lead_lag_analysis_xemm_0603_quiet_a_event/run_manifest.json`
  - `python -m json.tool local_live_analysis/horizon_alias_repair_0604T001/cross_exchange_lead_lag_analysis_xemm_0603_quiet_a_event/analysis_quality_summary.json`
  - `python -m json.tool local_live_analysis/horizon_alias_repair_0604T001/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_a_event/run_manifest.json`
- `git diff --check` passed.

done：
- 第一层修复完成：lead-lag verdict 不再把 aliased nominal horizons 当作多个独立稳定证据。
- 第二层修复完成：Hyperliquid raw alignment 支持 event-driven decision rows mode。
- Existing synthetic mode remains available for backward compatibility.
- 本任务未重新采集数据，未接 private/order endpoints，未执行 order lifecycle，未修改 strategy，未做 parameter search，未 default-on/tiny-live/promotion。

blockers：
- 无。
- Caveat：event mode 不能创造 raw feed 中不存在的 100/250ms book updates；若 public l2Book cadence 本身约 500ms，短 horizon book outcome 仍可能大量共享 future-row delta。

commit：
- 待提交

提交信息：
- 待提交
```
