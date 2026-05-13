```md
执行线程：
- 业务线程-python

任务ID：
- 0511T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 正常验收。重点检查实现是否严格遵守 0511T001 设计合同：default-off、shared helper、add-side only / reduce-side allowed、禁止未来数据、只做离线 replay、不授权 live micro test。

files：
- .workflow/tasks/0511T002.md
- examples/binance_tick_mm/strategy_core.py
- examples/binance_tick_mm/live_tick_mm.py
- examples/binance_tick_mm/backtest_tick_mm.py
- examples/binance_tick_mm/audit_schema.py
- examples/binance_tick_mm/stage6j_replay.py
- examples/binance_tick_mm/test_backtest_tick_mm.py
- examples/binance_tick_mm/test_stage6j_replay.py
- local_live_analysis/stage6j_cross_sample_0511T002/*
- /tmp/0511T002-stage6j-single/*
- findings.md
- progress.md

action：
- 按 0511T001 合同实现 default-off adverse timing guard。
- 在 `strategy_core.py` 新增 `GuardBlockResult` 和 shared helper `adverse_timing_guard_side_blocks(...)`。
- 在 `decide_actions(...)` 中加入 `adverse_timing_guard_block_buy/sell`，保持 add-side only block；reduce-side 继续允许。
- 在 live/backtest 共享路径中读取同一组 risk 配置，并调用同一个 `adverse_timing_guard_side_blocks(...)` helper。
- 新增 audit schema 和 audit row 字段：`adverse_timing_guard_*` active/reason/until/target_move_ticks。
- 扩展 Stage 6J replay 候选到 7 个：baseline、add-side guard、adverse timing 50/100/200ms、add-side guard + adverse timing 100ms、broad cooldown control。
- 增加 unit tests 覆盖 disabled 默认行为、target deterioration、pending cancel、cooldown expiry、add-side blocked、reduce-side allowed、Stage 6J config 注入。
- 未做 `SignalSnapshot` / `GuardState` / `DecisionContext` / `DecisionResult` 大重构。
- 未启动 live，未连接交易所，未改 AWS 状态。

verify：
- `python3 -m py_compile examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/stage6j_replay.py examples/binance_tick_mm/audit_schema.py` -> exit 0。
- `python3 -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py examples/binance_tick_mm/test_stage6j_replay.py` -> exit 0，`126 passed`。
- `python3 examples/binance_tick_mm/stage6j_replay.py --local-root local_live_analysis --run-id 5-10-day-control-1h-06 --out-dir /tmp/0511T002-stage6j-single` -> exit 0，decision `diagnostic_only_no_promotion`，single-sample run `1`，candidates `7`，hard failures `0`。
- `python3 examples/binance_tick_mm/stage6j_replay.py --local-root local_live_analysis --out-dir local_live_analysis/stage6j_cross_sample_0511T002 --run-id 5-10-day-control-1h-06 --run-id 5-9-small --run-id 5-9-noon --run-id 5-8-stage3-15m-livetest-v4` -> exit 0，decision `diagnostic_only_no_promotion`，samples `4`，candidates `7`，hard failures `0`。
- 单样本和跨样本 replay 均显示 overlays 为 `off/off/off`，lag gate 为 `True`。

done：
- 默认行为保持 default-off：`risk.adverse_timing_guard_enabled` 默认 `false`，baseline 和 add-side guard 不启用 adverse timing。
- 新增配置项：
  - `risk.adverse_timing_guard_enabled`
  - `risk.adverse_timing_guard_target_deterioration_enabled`
  - `risk.adverse_timing_guard_pending_cancel_enabled`
  - `risk.adverse_timing_guard_post_cancel_fill_enabled`
  - `risk.adverse_timing_guard_cooldown_ms`
  - `risk.adverse_timing_guard_min_target_move_ticks`
  - `risk.adverse_timing_guard_block_mode`
- 新增 audit 字段：
  - `adverse_timing_guard_buy_active`
  - `adverse_timing_guard_sell_active`
  - `adverse_timing_guard_buy_reason`
  - `adverse_timing_guard_sell_reason`
  - `adverse_timing_guard_buy_until_ts`
  - `adverse_timing_guard_sell_until_ts`
  - `adverse_timing_guard_target_move_ticks_buy`
  - `adverse_timing_guard_target_move_ticks_sell`
- 跨样本 Stage 6J 结果：
  - samples：`4`
  - candidates：`7`
  - hard failures：`0`
  - decision：`diagnostic_only_no_promotion`
- 候选汇总：
  - `baseline_inflight_only`: pnl sum `-0.864100`，cancel-fill `3`，same-side worsening `3`
  - `add_side_guard_only`: pnl sum `-0.232900`，cancel-fill `0`，same-side worsening `0`
  - `adverse_timing_target_deterioration_50ms`: pnl sum `-0.864100`，cancel-fill `3`，same-side worsening `3`
  - `adverse_timing_target_deterioration_100ms`: pnl sum `-0.864100`，cancel-fill `3`，same-side worsening `3`
  - `adverse_timing_target_deterioration_200ms`: pnl sum `-0.864100`，cancel-fill `3`，same-side worsening `3`
  - `add_side_guard_plus_adverse_timing_100ms`: pnl sum `-0.232900`，cancel-fill `0`，same-side worsening `0`
  - `broad_add_side_cooldown_200ms_control`: pnl sum `-0.053500`，cancel-fill `1`，same-side worsening `1`
- 结论：本次实现满足 default-off 和 replay 可执行目标，但 target deterioration adverse timing 单独候选在当前 4 样本上等同 baseline，没有显示 source-path 改善。add-side guard 仍是当前有效窄组件；add-side guard + adverse timing 与 add-side guard-only 结果相同。
- 是否允许 live micro test：不允许。当前仍是 `diagnostic_only_no_promotion`，没有证据支持 live promotion 或默认开启规则。

blockers：
- 无。本任务实现和离线验证已完成。
- 注意：adverse timing target-deterioration 候选没有改善目标 source-path；后续若继续 adverse-selection timing，应重新设计触发信号或增加更敏感的诊断 replay，而不是上线当前候选。

commit：
- 无

提交信息：
- 无
```
