# 线程回报

执行线程：
- 业务线程-python

任务ID：
- 0604T013

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T013.md`
- `.workflow/reports/0604T013-business.md`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 在 `live_tick_mm.py` 中新增 `SHUTDOWN_CANCEL_ACK_TIMEOUT_NS`、`ShutdownCancelResult` 和 `cancel_working_orders_for_shutdown(...)`，把 shutdown cancel 从旧的 fire-and-forget 路径改为逐单 `hbt.cancel(..., False)` 后显式 `hbt.wait_order_response(..., timeout_ns)`。
- `run_live()` 的 `finally` shutdown 路径已接入该 helper，并记录 attempts/sent/ack_waits/failed 以及逐单错误。
- primary buy/sell 仅在 `cancellable=true` 时 cancel；extra orders 仅在 `extra.cancellable=true` 且 `extra.req != "cancel"` 时 cancel。
- 单个 cancel 或 wait 异常会记录到对应 `ShutdownCancelResult.error`，不会阻止后续订单继续尝试。
- 为测试环境增加 `ROIVectorMarketDepthLiveBot` lazy import guard：模块导入和 helper 测试不再依赖 live-extension build；真正 `run_live()` 缺少 live bot 时仍 fail fast。
- 补充 `0604T012` case matrix 的 focused regression tests，证明正常 shutdown cancel 不再只发送请求就关闭。

0604T012 case matrix 修复后验证：
- `buy_only_cancellable`：通过；事件顺序为 `cancel(wait=False)` 后 `wait_order_response(timeout_ns)`。
- `sell_only_cancellable`：通过；事件顺序为 `cancel(wait=False)` 后 `wait_order_response(timeout_ns)`。
- `buy_sell_cancellable`：通过；两个 primary 订单都逐单 wait ACK。
- `extra_buy_exists`：通过；extra buy 会 cancel 并 wait ACK。
- `mixed_buy_sell_extras`：通过；primary buy/sell 和 buy/sell extras 都逐单 wait ACK。
- `primary_not_cancellable`：通过；不可撤 primary 不发 cancel。
- `extra_not_cancellable_still_cancelled`：已修复为不可撤 extra 不发 cancel；`req == "cancel"` 的 extra 也不重复发 cancel。
- `cancel_false_success_no_wait`：已修复；测试强制要求 `wait_order_response(...)` 出现在事件记录中。
- `cancel_false_exception_still_position_close`：已修复；一个订单 cancel 异常不会阻止剩余订单 cancel/wait。
- `close_after_pending_cancel_no_wait_api`：已修复；测试证明 `wait_order_response(...)` 发生在 `close()` 之前。

verify：
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "shutdown or cancel"`：`30 passed, 137 deselected`
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "live_shutdown or shutdown_cancel"`：`10 passed, 157 deselected`
- `python -m pytest examples/binance_tick_mm/test_*.py`：`282 passed`
- `git diff --check`：通过，无输出
- `python -m pytest examples/binance_tick_mm`：失败于已知 collection 环境问题；`run_env_test.py` 导入 `hftbacktest.data.utils.tardis` 时 numba cache 报错 `cannot cache function '_convert_depth': no locator available for file '/home/molly/anaconda3/lib/python3.13/site-packages/hftbacktest/data/utils/tardis.py'`。本任务以 `test_*.py` 作为可执行 package-level verification。

done：
- shutdown cancel 不再依赖 `close()` 作为唯一 ack drain 机制；`close()` 仍保持调用，但在 helper 完成逐单 cancel/wait 后执行。
- 本轮未做 REST final open-orders check；按 T013 范围，它保留为后续可选 live 安全增强，而不是本任务缺陷。
- 未修改 connector、py binding、Rust live bot、Rust `close()`、生产配置、audit schema 或 normal trading loop cancel 语义。

blockers：
- 无。

commit：
- `61f06d8fe2fa988bc4c1b131fc4a3566dfd21802`

提交信息：
- `fix live shutdown cancel ack wait`
