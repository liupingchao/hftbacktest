# 线程回报

执行线程：
- 业务线程-research

任务ID：
- 0601T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0601T004.md`
- `.workflow/reports/0601T004-business.md`
- `docs/binance_led_hyperliquid_maker_data_input_contract.md`
- `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/run_manifest.json`
- `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/feature_decision_table.csv`
- `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/data_input_schema.md`
- `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/next_runner_contract.md`

action：
- 将 `0601T004` 从草案改为正式派发任务，并在 `0601T003` QA 通过后执行。
- 编写 `docs/binance_led_hyperliquid_maker_data_input_contract.md`，明确 Binance lead input 与 Hyperliquid lag venue-state/context input 的职责边界。
- 生成 task-scoped artifacts：
  - `run_manifest.json`
  - `feature_decision_table.csv`
  - `data_input_schema.md`
  - `next_runner_contract.md`
- 明确后续任务只能是 read-only pricing-signal runner，不是 strategy implementation、parameter search、live/default-on/tiny-live/promotion。

source evidence：
- `0601T002` QA 已通过：local-observation-time as-of join，`future_join_count=0`，`missing_binance_join_count=0`，`primary_usable_row_count=3596`，trade pressure disabled。
- `0601T003` QA 已通过：lead-lag analyzer primary rows `3596`，excluded rows `3`，horizons `100/250/500/1000/5000/10000ms`，verdict counts `18 stable / 6 watch / 30 unstable / 0 insufficient`。
- `0601T003` QA 接受 effective future-age auditability；后续 runner 必须同时报告 nominal horizon 和 actual future-row age。
- `0531T001` QA 已通过：Hyperliquid public market-view / pricing feature consumer 保持 trade pressure disabled，因为 public trade side semantics 未验证。

feature decisions：
- Primary allowlist：
  - `binance_top5_imbalance`
  - `binance_microprice_minus_mid_ticks`
  - `binance_mid_move_ticks_from_prev`
  - `binance_top5_bid_qty`
- Diagnostic/context only：
  - `binance_top5_microprice_px`
  - `binance_rolling_abs_mid_move_ticks_5`
  - `binance_rolling_rv_ticks_20`
  - `binance_top5_ask_qty`
  - `binance_top5_total_qty`
  - Hyperliquid venue-state and basis/dislocation context
- Disabled：
  - Binance trade pressure
  - Hyperliquid trade pressure
  - Any private/order/fill/lifecycle/future-label input

timestamp / no-future policy：
- Join clock remains `local_controller_capture_ts_ns`.
- Binance lead input must satisfy `binance_local_ts <= hyperliquid_decision_ts`.
- Future label construction must use first Hyperliquid row where `future_decision_ts >= decision_ts + horizon_ms`.
- Later outputs must report `effective_future_age_ms` because 100/250/500ms nominal horizons may map to the same future row on the current Hyperliquid decision grid.

next-runner boundary：
- Allowed later task: read-only pricing-signal runner.
- Allowed outputs: candidate signal rows, horizon label summaries, feature stability by regime, venue-state conditioning summaries, and read-only recommendation.
- Allowed recommendations only: `keep_for_read_only_research`, `needs_more_public_samples`, `reject_for_runner_design`.
- Forbidden conclusions: strategy-ready, signal-ready, tiny-live-ready, default-on-ready, promotion-ready.

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
- `python -m json.tool local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/run_manifest.json`
  - 通过。
- `test -s docs/binance_led_hyperliquid_maker_data_input_contract.md`
  - 通过。
- `test -s local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/feature_decision_table.csv`
  - 通过。
- `test -s local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/data_input_schema.md`
  - 通过。
- `test -s local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/next_runner_contract.md`
  - 通过。
- `git diff --check`
  - 通过。

done：
- `0601T004` 已生成 read-only data input / next-runner contract，可供后续总控创建 read-only pricing-signal runner implementation task。
- 本任务未实现 runner、未采集新样本、未连接网络、未修改策略、未触碰 private/order/live/parameter/default-on/tiny-live/promotion。

blockers：
- 无。

commit：
- 待提交

提交信息：
- 待提交
