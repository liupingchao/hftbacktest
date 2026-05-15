```md
执行线程：
- 业务线程-python

任务ID：
- 0515T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0515T005.md`
- `.workflow/reports/0515T005-business.md`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 仅针对 `cancel_race_window_too_short` residual class 实施了 very narrow replay-side repair。
- 新增 live short cancel-race fill constraint loader：
  - 只读取 live audit 中满足以下条件的 fills：
    - `fill_after_cancel_request = true`
    - `fill_ts_local - cancel_request_ts <= 25ms`
- 在 `audit_replay` lifecycle 消费时，只对这一种情况做最小改写：
  - 当 replay 当前准备发出 `cancel_ack`
  - 且 live 已经明确记录为 short cancel-race fill
  - 则把该条 replay `cancel_ack` 收窄改写成 `fill`
- 未修改 generalized touch fill / queue / trade-consumption 逻辑。
- 未处理 `28940|sell` / `4948` 这类 `residual_replay_fill_trigger_uncertain`。
- 同样本回归重跑：
  - `backtest_audit_replay`
  - Stage 6B calibration
  - Stage 6C mismatch diagnosis
  - Stage 6E residual diagnosis

verify：
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py`
- `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
- `python examples/binance_tick_mm/backtest_tick_mm.py --config local_live_analysis/5-13-day-control-30min/config_backtest_audit_replay.toml --manifest local_live_analysis/5-13-day-control-30min/out/live_raw/btcusdt/manifest_2026-05-13_to_2026-05-13.json --window full_day --slice-ts-local-start 1778661013548339763 --slice-ts-local-end 1778662825090642602`
- `python examples/binance_tick_mm/execution_outcome_calibration.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6f_cancel_race_residual_repair_0515T005/stage6_execution_calibration`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6f_cancel_race_residual_repair_0515T005/stage6c_replay_lifecycle_mismatch`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6f_cancel_race_residual_repair_0515T005/stage6e_residual_replay_fill_diagnosis --residual-only`

done：
- 目标 residual `572` 已消失：
  - `live_filled_replay_canceled` residual: `1 -> 0`
  - Stage 6E residual case count: `2 -> 1`
  - `cancel_race_window_too_short_rows: 1 -> 0`
- `4948` 保持未处理，符合任务边界：
  - remaining residual case: `28940|sell`
  - class: `residual_replay_fill_trigger_uncertain`
- aggregate 指标没有明显回退，仍在对齐阈值内：
  - replay filled orders: `53 -> 54`
  - replay fill-after-cancel orders: `14 -> 15`
  - final-state gaps:
    - `canceled`: `0.0 -> 0.000397`
    - `filled`: `0.0 -> 0.000397`
    - `open_or_missing`: `0.0 -> 0.0`
  - terminal-state diff rows: `2 -> 1`
- 这轮变化的本质是：
  - 用一个非常窄的 live-confirmed short cancel-race fill 映射，消掉了 `572`
  - 没有引入 broad touch/queue 修复
  - `4948` 仍保留为后续独立问题

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
