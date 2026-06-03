# 线程回报

执行线程：
- 业务线程-research

任务ID：
- 0601T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0601T005.md`
- `.workflow/reports/0601T005-business.md`
- `examples/hyperliquid/binance_led_pricing_signal_runner.py`
- `examples/hyperliquid/test_binance_led_pricing_signal_runner.py`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/run_manifest.json`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/pricing_signal_rows.csv`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/pricing_signal_feature_quality.csv`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/horizon_label_summary.csv`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/feature_stability_by_regime.csv`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/venue_state_conditioning_summary.csv`
- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/pricing_signal_recommendation.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 正式派发并执行 `0601T005` read-only pricing-signal runner implementation。
- 新增 `examples/hyperliquid/binance_led_pricing_signal_runner.py`，只读取本地已验收 `0601T002/0601T003/0601T004` 产物。
- 新增 focused tests，覆盖 future-label 构造、primary allowlist 加载、默认样本 artifact 生成、boundary/recommendation 约束。
- 生成 task-scoped artifacts 到 `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/`。

source evidence：
- `0601T002` QA 已通过：joined features 使用 `binance_local_ts <= hyperliquid_decision_ts`，`future_join_count=0`，`missing_binance_join_count=0`，`primary_usable_row_count=3596`。
- `0601T003` QA 已通过：lead-lag analysis verdict counts 为 `18 stable / 6 watch / 30 unstable`，并要求输出 nominal horizon 与 effective future age。
- `0601T004` QA 已通过：primary allowlist 仅为 `binance_top5_imbalance`、`binance_microprice_minus_mid_ticks`、`binance_mid_move_ticks_from_prev`、`binance_top5_bid_qty`，trade pressure 保持 disabled。

generated artifacts：
- `run_manifest.json`
- `pricing_signal_rows.csv`
- `pricing_signal_feature_quality.csv`
- `horizon_label_summary.csv`
- `feature_stability_by_regime.csv`
- `venue_state_conditioning_summary.csv`
- `pricing_signal_recommendation.md`

key results：
- input rows：`3599`
- primary rows：`3596`
- excluded rows：`3`
- pricing signal rows：`21541`
- feature quality rows：`4`
- horizon label summary rows：`30`
- feature stability by regime rows：`540`
- venue-state conditioning rows：`54`
- recommendation：`keep_for_read_only_research`

feature allowlist enforcement：
- `pricing_signal_feature_quality.csv` 只包含 4 个 `0601T004` primary allowlist features。
- `pricing_signal_rows.csv` 中 decision-time inputs 使用 `input_<feature>` / `input_<feature>_z` 字段，future labels 使用独立 `hyperliquid_future_*` / `basis_future_*` 字段。
- `binance_mid_move_ticks_from_prev` 有 `1` 个 primary row 缺失，已在 feature quality 中显式报告。
- Binance / Hyperliquid trade pressure 均保持 `disabled_unverified_side_semantics`。

timestamp / no-future policy：
- Join clock 保持 `local_controller_capture_ts_ns`。
- Primary row policy 保持 `joined_row_quality=primary_usable` 且无 future/missing Binance join。
- Future label construction 使用 first Hyperliquid row where `future_decision_ts >= decision_ts + horizon_ms`。
- `horizon_label_summary.csv` 对每个 horizon/label 报告 `effective_future_age_ms_min/mean/max`。
- 100/250/500ms nominal horizons 当前仍可映射到同一约 `500ms` effective future row；runner 显式保留该审计信息。

boundary flags：
- no private keys：true
- no private account endpoints：true
- no order endpoints：true
- no order lifecycle：true
- no strategy implementation：true
- no live trading bot：true
- no parameter search：true
- no default-on：true
- no tiny-live：true
- no promotion：true
- no schema / connector / core API change：true

verify：
- `python examples/hyperliquid/binance_led_pricing_signal_runner.py --help`
  - 通过。
- `python -m py_compile examples/hyperliquid/binance_led_pricing_signal_runner.py examples/hyperliquid/test_binance_led_pricing_signal_runner.py`
  - 通过。
- `python -m pytest examples/hyperliquid/test_binance_led_pricing_signal_runner.py`
  - 通过，`3 passed`。
- `python examples/hyperliquid/binance_led_pricing_signal_runner.py --output-dir local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005`
  - 通过，输出 `pricing_signal_rows=21541`，`primary_rows=3596`，`recommendation=keep_for_read_only_research`。
- `python -m json.tool local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/run_manifest.json`
  - 通过。
- Required artifact existence checks for all seven outputs
  - 通过。
- `git diff --check`
  - 通过。

done：
- `0601T005` 已实现并生成 read-only pricing-signal runner artifacts。
- 输出仅支持继续 read-only research；不授权 strategy implementation、private/order endpoints、order lifecycle、live/default-on/tiny-live、parameter search 或 promotion。
- 业务线程结果可进入 QA 验收。

blockers：
- 无。

commit：
- 待提交

提交信息：
- 待提交
