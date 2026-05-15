```md
执行线程：
- 业务线程-python

任务ID：
- 0515T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0515T003.md`
- `.workflow/reports/0515T003-business.md`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 在 `audit_replay` 的 replay lifecycle 路径里实施了 narrow repair，范围只限 replay-side lifecycle，不改策略行为：
  1. 增加 live terminal constraint loader，从 live audit 读取每个 order 的 terminal state、cancel request ts、cancel ack ts、terminal ts。
  2. 在 replay lifecycle event 消费时，对 `fill` / `partial_fill` / `order_update` 做 live terminal gate：
     - 如果 live 在当前 decision 前已经 terminal 为 `canceled` / `expired` / `rejected`，则 replay 不再保留后续 fill，而是改写成对应 terminal lifecycle event。
  3. 增加 forced terminal fallback：
     - 对于 replay 这拍没有再发 lifecycle event、但 live 已经 terminal 且当前 live working state 里也不存在的订单，主动注入 synthetic terminal lifecycle event，避免把原先的 `canceled -> filled` 仅仅挪成 `canceled -> open_or_missing`。
- 为上述 repair 补充 focused tests，覆盖：
  - live terminal constraint loader
  - replay fill after live cancel 的改写
  - pre-terminal fill 不应被误杀
  - forced cancel-ack injection
  - live visible order 仍在时不应强制 terminal
- 用相同样本 `5-13-day-control-30min` 重跑：
  - `backtest_audit_replay`
  - Stage 6B calibration 到 `stage6d_replay_lifecycle_repair_0515T003/stage6_execution_calibration/`
  - Stage 6C mismatch diagnosis 到 `stage6d_replay_lifecycle_repair_0515T003/stage6c_replay_lifecycle_mismatch/`

verify：
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py`
- `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
- `python examples/binance_tick_mm/backtest_tick_mm.py --help`
- `python examples/binance_tick_mm/execution_outcome_calibration.py --help`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --help`
- `python examples/binance_tick_mm/backtest_tick_mm.py --config local_live_analysis/5-13-day-control-30min/config_backtest_audit_replay.toml --manifest local_live_analysis/5-13-day-control-30min/out/live_raw/btcusdt/manifest_2026-05-13_to_2026-05-13.json --window full_day --slice-ts-local-start 1778661013548339763 --slice-ts-local-end 1778662825090642602`
- `python examples/binance_tick_mm/execution_outcome_calibration.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6d_replay_lifecycle_repair_0515T003/stage6_execution_calibration`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6d_replay_lifecycle_repair_0515T003/stage6c_replay_lifecycle_mismatch`

done：
- repair 前后核心指标对比：
  - replay filled orders：`172 -> 53`
  - replay fill-after-cancel orders：`133 -> 14`
  - replay-only fill rows：`120 -> 1`
  - live-cancel / replay-filled rows：`120 -> 1`
  - terminal-state diff rows：`230 -> 2`
  - final-state gaps：
    - `canceled`: `0.09062 -> 0.0`
    - `filled`: `0.04730 -> 0.0`
    - `open_or_missing`: `0.04332 -> 0.0`
- hot spot 改善也很明显：
  - `step_back_gt1`：
    - replay_only_fill_rate：`0.05693 -> 0.0`
    - terminal_state_diff_rate：`0.11134 -> 0.0`
    - fill_after_cancel_gap：`0.05642 -> 0.00050`
    - time_to_fill_gap_mean_ms：`28198.95 -> 21.27`
  - latency `q5`：
    - replay_only_fill_rate：`0.05347 -> 0.0`
    - terminal_state_diff_rate：`0.07921 -> 0.0`
    - fill_after_cancel_gap：`0.05149 -> 0.00198`
    - time_to_fill_gap_mean_ms：`28059.16 -> 99.87`
  - latency `q2`：
    - replay_only_fill_rate：`0.04970 -> 0.0`
    - terminal_state_diff_rate：`0.10736 -> 0.00199`
    - fill_after_cancel_gap：`0.04771 -> 0.00199`
    - time_to_fill_gap_mean_ms：`25917.52 -> 3997.55`
- 当前 residual mismatch 还剩 2 个 matched submits：
  1. `live_filled_replay_canceled` 1 条
  2. `live_canceled_replay_filled` 1 条
- Stage 6B 新 decision state 变成了 `requires_more_current_format_samples`，说明这轮 repair 已经把原本的 core lifecycle mismatch 从结构性大偏差收敛到了 residual case 级别。
- 本任务未改策略、未补样本、未启动 live。

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
