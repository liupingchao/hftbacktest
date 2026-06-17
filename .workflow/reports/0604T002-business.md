```md
执行线程：
- 测试线程-research

任务ID：
- 0604T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T002.md`
- `.workflow/reports/0604T002-business.md`
- `local_live_analysis/event_horizon_comparison_0604T002/**`

action：
- 使用 `0604T001` 修改后的 event-driven Hyperliquid alignment / de-aliased horizon logic，对三个既有样本做本地只读重跑：
  - `xemm_0603_quiet_a`
  - `xemm_0603_quiet_b`
  - `xemm_0603_quiet_c`
- 本轮没有重新采集数据，没有改代码，没有接 private/order endpoints，没有执行 order lifecycle，没有修改 strategy，没有 parameter search / default-on / tiny-live / promotion。
- 每个样本均完成：
  - Hyperliquid event-mode alignment
  - cross-exchange join
  - lead-lag analysis
  - pricing-signal runner
- 对比输出：
  - `local_live_analysis/event_horizon_comparison_0604T002/event_comparison_summary.csv`
  - `local_live_analysis/event_horizon_comparison_0604T002/event_horizon_alias_summary.csv`
  - `local_live_analysis/event_horizon_comparison_0604T002/event_lead_lag_feature_verdicts_all_samples.csv`

data / join quality：
- `xemm_0603_quiet_a`
  - overlap `1800.0401s`
  - Binance bookTicker/depth/trade: `508154 / 67112 / 83027`
  - Hyperliquid l2Book/trade events: `3332 / 10863`
  - event decision rows `3332`
  - join rows / primary rows: `3332 / 3332`
  - future/missing Binance joins: `0 / 0`
  - Binance source age p99 `27.5668ms`
- `xemm_0603_quiet_b`
  - overlap `1800.0844s`
  - Binance bookTicker/depth/trade: `1710706 / 67964 / 213073`
  - Hyperliquid l2Book/trade events: `3330 / 21987`
  - event decision rows `3330`
  - join rows / primary rows: `3330 / 3330`
  - future/missing Binance joins: `0 / 0`
  - Binance source age p99 `27.1816ms`
- `xemm_0603_quiet_c`
  - overlap `1800.0704s`
  - Binance bookTicker/depth/trade: `1753489 / 68045 / 287991`
  - Hyperliquid l2Book/trade events: `3334 / 29525`
  - event decision rows `3334`
  - join rows / primary rows: `3334 / 3334`
  - future/missing Binance joins: `0 / 0`
  - Binance source age p99 `30.4933ms`

vol / regime comparison：
- `xemm_0603_quiet_a`
  - Binance mid-move mean abs `0.6227` ticks; p95 `0.0`; p99 `24.0`
  - Binance RV20 mean `2.0147`; p95 `12.8727`
  - Basis mid abs mean `45.9173` ticks
  - Hyperliquid future mid 500ms mean abs `10.8739` ticks; p95 `70.0`
  - Hyperliquid future mid 1000ms mean abs `18.5446` ticks
  - Interpretation: true quiet / low-vol baseline.
- `xemm_0603_quiet_b`
  - Binance mid-move mean abs `2.3045` ticks; p95 `17.0`; p99 `57.0`
  - Binance RV20 mean `6.8713`; p95 `21.1208`
  - Basis mid abs mean `119.1518` ticks
  - Hyperliquid future mid 500ms mean abs `27.4932` ticks; p95 `130.0`
  - Hyperliquid future mid 1000ms mean abs `44.0174` ticks
  - Interpretation: medium / active vol, not pure quiet repeat.
- `xemm_0603_quiet_c`
  - Binance mid-move mean abs `2.4883` ticks; p95 `21.0`; p99 `55.0`
  - Binance RV20 mean `7.2827`; p95 `21.7365`
  - Basis mid abs mean `156.9063` ticks
  - Hyperliquid future mid 500ms mean abs `32.0012` ticks; p95 `140.0`
  - Hyperliquid future mid 1000ms mean abs `49.5213` ticks
  - Interpretation: highest activity / vol among the three samples.

horizon alias / effective horizon：
- Event mode removes the synthetic fixed 500ms grid, but public `l2Book` cadence still creates substantial short-horizon aliasing.
- Same future row for nominal `100/250/500ms`:
  - `xemm_0603_quiet_a`: `2393/3330 = 0.7186`
  - `xemm_0603_quiet_b`: `2055/3329 = 0.6173`
  - `xemm_0603_quiet_c`: `1886/3333 = 0.5659`
- Effective future age mean by sample:
  - `100ms`: `541.721ms / 550.136ms / 574.120ms`
  - `250ms`: `557.644ms / 578.617ms / 588.114ms`
  - `500ms`: `705.653ms / 776.752ms / 813.225ms`
  - `1000ms`: `1170.181ms / 1245.818ms / 1298.154ms`
  - `5000ms`: `5365.108ms / 5336.719ms / 5339.598ms`
  - `10000ms`: `10272.366ms / 10274.671ms / 10290.125ms`
- Interpretation:
  - Event mode fixes artifact-level 100% aliasing.
  - `100/250ms` still often map to the next observed l2Book update, so they remain weakly independent for book-outcome research.
  - `500ms` becomes more separable in higher-activity samples but still often shares next-row evidence.
  - `1000ms+` horizons have more meaningful independent future-row deltas.

signal comparison：
- `xemm_0603_quiet_a`
  - lead-lag verdicts: `17 stable / 3 watch / 34 unstable`
  - pricing rows `19956`
  - recommendation `needs_more_public_samples`
- `xemm_0603_quiet_b`
  - lead-lag verdicts: `26 stable / 9 watch / 19 unstable`
  - pricing rows `19945`
  - recommendation `keep_for_read_only_research`
- `xemm_0603_quiet_c`
  - lead-lag verdicts: `19 stable / 13 watch / 22 unstable`
  - pricing rows `19969`
  - recommendation `keep_for_read_only_research`

done：
- 三个样本 event-mode 本地重跑和比较已完成。
- `quiet_a` 是低波动 baseline；`quiet_b` / `quiet_c` 都更活跃，其中 `quiet_c` 的 trade activity、basis dislocation 和 HL future move 最大。
- 从 signal 稳定性看，`quiet_b` 最强，`quiet_c` 次之且 watch bucket 较多，`quiet_a` 仍偏样本不足。
- 三个样本均不支持把 read-only pricing research 解释为 strategy-ready / live-ready / promotion-ready。

verify：
- 三个样本均运行：
  - `python examples/hyperliquid/hyperliquid_raw_alignment.py --decision-mode event ...`
  - `python examples/hyperliquid/cross_exchange_lead_lag_join.py ...`
  - `python examples/hyperliquid/cross_exchange_lead_lag_analysis.py ...`
  - `python examples/hyperliquid/binance_led_pricing_signal_runner.py ...`
- 三个样本 generated JSON manifests 全部通过 `python -m json.tool`。
- Focused comparison script completed and wrote all three comparison CSVs.
- `git diff --check` passed.

blockers：
- 无。

commit：
- 无

提交信息：
- 无
```
