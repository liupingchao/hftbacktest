# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0718T014

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
- `examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py`
- `examples/hyperliquid/test_hyperliquid_kill_switch.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `.workflow/tasks/0718T014.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增 `ProjectedExposure`、`projected_exposure()` 和 `validate_runtime_envelope()`。
- 分别计算最坏 long/short exposure；working、cancel-pending 和 unknown/inflight leaves 在 exchange confirmation 前持续计入。
- proposed quotes 先按方向合计，再检查 position、position notional、aggregate notional、单订单和 submission cap。
- inventory-reducing side 在达到同向仓位上限时仍可通过；可能穿越到 opposite-side 的结果按最坏仓位计入。
- task-specific cap 使用不大于 global default 的严格值。
- repo operational submission upper bound 保留为 `30` 以兼容 anti-drift；formal tiny-live `TinyLiveConfig` 默认使用更严格的 `2`。
- `run_order_once` 要求 live caller 显式提供 `ProjectedExposure` 和 `submissions_used`，且 max-loss 先于普通 envelope rejection 执行 kill-switch。
- canary、event-driven watcher 和 fill-window 在提交前读取真实 `user_state/open_orders`；existing open-order 价格作为 notional valuation bound，无法分类的订单状态 fail-closed。
- fast event-driven 路径不再把 position proof 延迟到下单后；slow fee pullback 仍保持下单后。
- 纯 inventory-reducing quote 不因“当前持仓 notional + 减仓单 notional”的简单相加而被错误阻断。
- 更新 executor config snapshot 和 workflow 状态；未修改策略报价、价格栈或 live authorization。

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py -q`
  - `31 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_kill_switch.py -q`
  - `26 passed`
- 相关 executor、kill-switch、watcher、fill-loop、fill-attribution 回归：
  - `158 passed in 19.92s`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py`
  - pass
- `git diff --check`
  - pass
- 未执行 live、credential、private endpoint、order、cancel、network、remote 或 service action。

done：
- T014 的 typed aggregate exposure 与 runtime envelope implementation 已完成并提交 QA。
- 新增 focused coverage：最坏双向暴露、cancel-pending/inflight、聚合多层订单、reduce-side、严格 caps、submission budget、真实 snapshot 解析、existing high-price valuation、提交前阻断及 max-loss 优先级。

blockers：
- 无

commit：
- `b48d7c3`
- `4550726`

提交信息：
- `Implement aggregate exposure runtime envelope`
- `Wire runtime exposure snapshot into live callers`
