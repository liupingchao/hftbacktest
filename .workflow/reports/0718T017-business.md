# 线程回报

执行线程：
- 总控 auto-loop / 业务实现线程

任务ID：
- 0718T017

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/test_hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py`

action：
- 新增 strategy-owned、exchange-reconciled 的单层双边订单管理器，状态覆盖 desired、submit_inflight、resting、partial_fill、cancel_requested、cancel_confirmed、filled、rejected 和 unknown。
- 使用 `(symbol, side, canonical_price_key)` logical key；每侧最多一个 active quote，same-key hold/dedupe，confirmed cancel 后 same-price re-add 使用新的 generation/cloid。
- 新增固定长度、SDK-compatible 的 ownership cloid；按 task/run prefix 隔离，恢复时忽略 foreign order，并对重复 owned logical key fail-closed。
- 启动、重连和周期 reconciliation 读取 open orders/user state；submit 歧义先按 oid/cloid 查询，cancel 未确认和 unknown/inflight leaves 继续计入 working exposure。
- partial fill 更新 leaves、filled quantity、position evidence；恢复时优先使用 exchange `remainingSz`。
- 实现最小 price move、quote age、submit-inflight/cancel-pending guard、post-only reject cooldown、每侧 cancel/re-add rate limit；emergency cancel 不绕过 runtime exposure cap。
- 不把 `run_order_once` 改造成持续报价循环；本任务只提供 manager 和 mock/offline 验证，不接 watcher、不启动 live。

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_maker_order_manager.py examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py -q`
  - `43 passed`
- 相关 pricing/shadow/replay/price-math regression：
  - `30 passed`
- `python examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py --help`
  - pass
- `python -m py_compile examples/hyperliquid/hyperliquid_maker_order_manager.py examples/hyperliquid/test_hyperliquid_maker_order_manager.py examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py`
  - pass
- `git diff --check`
  - pass
- 明确确认未执行 live、credential、private、order、cancel、network、remote 或 service action。

done：
- Principal Alignment Task 6 的订单生命周期、ownership、exchange reconciliation、anti-churn 和 aggregate exposure preservation 已实现并通过 focused/regression 验证，提交等待 QA 记录。

blockers：
- 无

commit：
- `a3aed72`

提交信息：
- `Implement exchange-reconciled maker order manager`
