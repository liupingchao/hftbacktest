# 线程回报

执行线程：
- 业务线程-price-normalization

任务ID：
- 0718T012

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_price_math.py`
- `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/test_cross_exchange_price_math.py`
- `examples/hyperliquid/test_cross_exchange_shared_signal_kernel.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `docs/cross_exchange_price_taxonomy_contract.md`

action：
- 新增 Decimal-based authoritative price math，支持 buy floor、sell ceil、nearest、precision validation 和 post-only non-crossing clamp。
- executor 的旧 `round_hyperliquid_perp_price` 保留为 shared helper wrapper，并在 order intent validation 增加最终价格合法性检查。
- shared kernel、fill-window 的 quote 输出统一经过 shared post-only helper。
- 增加价格 taxonomy contract 与 precision、idempotence、invalid input、kernel/executor/fill-window 接线测试。
- 未修改 live 参数、quote policy、订单边界、risk envelope、endpoint gate 或 strategy semantics。

verify：
- T012 focused tests -> `55 passed`
- T012 + T011/Phase 1-4 related regression -> `143 passed`
- `python -m py_compile` 相关生产/测试文件 -> pass
- shared kernel 和 remote orchestrator `--help` -> pass
- `git diff --check` -> pass
- 未执行 live、credential、private、order、cancel、network、remote 或 service 动作。

done：
- 价格输出统一满足最多五位有效数字与 `max(0, 6 - szDecimals)` 小数位限制。
- normalize 对 buy/sell/nearest 方向可复现且幂等。
- post-only buy 严格低于 ask，sell 严格高于 bid；crossed/invalid BBO fail-closed。
- 非法价格和非法 precision 在 executor/kernel 路径阻断。

blockers：
- 无

commit：
- `9252c4b`

提交信息：
- `Implement authoritative Hyperliquid price normalization`
