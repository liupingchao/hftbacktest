```md
执行线程：
- 业务线程-python

任务ID：
- 0515T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0515T004.md`
- `.workflow/reports/0515T004-business.md`
- `examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py`
- `examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`

action：
- 在现有 `replay_lifecycle_mismatch_diagnosis.py` 上增加 `0515T004` residual-only diagnosis 模式，保持只读：
  - 默认仍保留 `0515T001` mismatch runner
  - 新增 `--residual-only`，产出 residual-case diagnosis
- residual diagnosis 只分析 `0515T003` 修复后剩余的 2 个 matched submit residual cases：
  1. `live_filled_replay_canceled`
  2. `live_canceled_replay_filled`
- 为每个 residual case 增加：
  - live / replay event timeline
  - supportive raw trade counting
  - T009 top5 sidecar as-of submit / replay fill / live terminal context
  - residual trigger classification
- 分类规则保持保守：
  - `cancel_race_window_too_short`
  - `touch_fill_assumption_too_optimistic`
  - `queue_exposure_proxy_bias_possible`
  - `residual_replay_fill_trigger_uncertain`
- 不修改 replay fill model / lifecycle state machine / strategy / queue logic。

verify：
- `python -m pytest examples/binance_tick_mm/test_replay_lifecycle_mismatch_diagnosis.py`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --help`
- `python examples/binance_tick_mm/replay_lifecycle_mismatch_diagnosis.py --run-dir local_live_analysis/5-13-day-control-30min --output-dir local_live_analysis/5-13-day-control-30min/stage6e_residual_replay_fill_diagnosis_0515T004 --residual-only`

done：
- 真实样本输出目录：
  - `local_live_analysis/5-13-day-control-30min/stage6e_residual_replay_fill_diagnosis_0515T004/`
- 核心结论：
  1. `3879|buy` / order `572`
     - 归类：`cancel_race_window_too_short`
     - 证据：
       - live `cancel_request -> fill` 延迟约 `9.43ms`
       - live fill 前 `10ms` 内已有 `41` 笔 supportive trades
       - replay 走成 `cancel_ack`，没有复现这笔 short cancel-race fill
     - 结论：
       - 这是一个明确的 short cancel-race miss

  2. `28940|sell` / order `4948`
     - 归类：`residual_replay_fill_trigger_uncertain`
     - 证据：
       - replay fill 发生在 live cancel request 前约 `262.28ms`
       - 这说明它不是“cancel 后仍 fill”的问题
       - 但在当前 raw trade 证据里，replay fill 前 `10/25/50ms` supportive trades 计数都为 `0`
       - 因此不能把它直接判成 `touch_fill_assumption_too_optimistic`
     - 结论：
       - 当前更像一个“replay fill trigger 证据不足的孤例”
       - 还不能安全地下结论说是 touch optimism，或 queue exposure approximation bias

- 总体判断：
  - 两个 residual cases 并不是同一机制
  - `572` 已经足够支撑一个很窄的 cancel-race residual repair
  - `4948` 还不足以直接开 repair，因为缺少更强的 case-level trigger proof
  - 因此现在不建议直接进入统一的 `0515T005` repair；如果继续，应把后续任务收窄成：
    - 只修 `cancel_race_window_too_short` 这类 residual
    - 或先继续做更细的 replay fill trigger 证据采集，再决定是否动 `4948` 这类孤例

- 本任务未修改 replay、未改策略、未补样本、未启动 live。

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
