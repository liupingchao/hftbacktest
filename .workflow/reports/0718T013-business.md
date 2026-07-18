# 线程回报

执行线程：
- 业务线程-persistent-kill-switch

任务ID：
- 0718T013

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/test_hyperliquid_kill_switch.py`
- related executor/watcher/fill-window regression tests

action：
- 实现独立用户级 control state 目录与原子 JSON 状态写入，支持 `armed`、`triggered`、`reset`、成功 flatten 到期和失败永久 halted 语义。
- 实现 `KillSwitchConfig`、`HaltState`、`KillSwitchEvidence`、`check_halt_state`、`initialize_control_state`、`reset_halt_state` 和 `execute_kill_switch`。
- kill-switch 顺序固定为：持久化 trigger、撤销 owned refs、验证 owned open-order proof、读取真实 `szi`、按 `abs(szi)` 调用 reduce-only `market_close`、复读仓位、持久化脱敏证据。
- 增加 fcntl 互斥，串行化 concurrent trigger/reset/init；cancel 与 market-close 的非异常错误响应也 fail-closed。
- watcher、fill-window 和最终 `run_order_once` 提交边界复查同一 halt state；max-loss rejection 先执行 kill-switch，再阻止订单。
- 缺失/损坏状态、失败或 pending 状态、活跃 halt、缺失 control directory 均禁止新报价；显式初始化/reset 才能恢复 clean state。
- toxicity/stale/orchestrator trigger reason 已作为可调用的版本化 fail-closed contract；按方案约定，自动 toxicity observe-only 到后续独立任务再启用。

verify：
- T013 focused kill-switch tests: `23 passed`
- executor + watcher + fill-window/fill-attribution related regression: `139 passed`
- `python -m py_compile` modified production/test files: pass
- executor/watcher `--help`: pass
- `git diff --check`: pass
- 未执行 live、credential、private endpoint、order、cancel、network、remote 或 service action。

done：
- durable halt state 与跨 run persistence 已闭环。
- flat、long、short、actual abs(szi)、cancel/close exception、non-exception error response、residual position、corrupted state、missing state、expiry/reset、serial/concurrent idempotency 均有 offline coverage。
- implementation commit：`955cf9e`

blockers：
- 无

commit：
- `955cf9e`

提交信息：
- `Implement durable Hyperliquid kill switch`
