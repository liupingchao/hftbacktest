# 线程回报

执行线程：
- 业务线程-python

任务ID：
- 0604T016

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T016.md`
- `.workflow/reports/0604T016-business.md`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 修复 shutdown cancel helper 的 wait-result 语义，把 raw `wait_order_response()` 返回值保存为 `wait_result_raw`，并单独分类为 `wait_outcome` / `order_response_received`。
- 新增独立 `terminal_confirmed` / `terminal_confirmation_source` / `final_order_status` 维度；terminal proof 只来自 wait 后本地 `hbt.orders(asset_no)` 重新收集的 active working-order 状态。
- 保留旧 `wait_result` 只读 property 兼容现有调用，但不再把它作为 confirmed ack 语义。
- 固定 `wait_outcome` 枚举：`order_response_received`、`ok_unknown_or_timeout`、`wait_error`、`not_requested`。
- 固定 `terminal_confirmation_source` 枚举：`local_orders`、`rest_open_orders`、`none`；本任务没有实现 REST reconciliation，也没有产生 `rest_open_orders` source。
- 修改 shutdown summary log，移除误导性的 `ack_waits` 口径，改为分开输出 `wait_requests`、`order_responses`、`terminal_confirmed`、`unknown_or_timeout`、`failed`。
- 扩展 focused tests，覆盖 wait code 与 final local-state proof 的独立组合，以及 enum/source regression。

required tests 实际结果：
- `wait_result=3 + local order still active`：`order_response_received=True`、`terminal_confirmed=False`、`terminal_confirmation_source=local_orders`、`final_order_status=local_active_working_order`。
- `wait_result=0 + local order terminal/absent`：`order_response_received=False`、`wait_outcome=ok_unknown_or_timeout`、`terminal_confirmed=True`。
- `wait_result=3 + local order terminal/absent`：`order_response_received=True`、`terminal_confirmed=True`，两者来源不同。
- `wait_result=0 + no final status proof`：`order_response_received=False`、`terminal_confirmed=False`、`terminal_confirmation_source=none`。
- `wait_result=3 + no final status proof`：`order_response_received=True`、`terminal_confirmed=False`、`terminal_confirmation_source=none`。
- wait exception：`wait_outcome=wait_error`、`order_response_received=False`；后续订单仍继续尝试 shutdown cancel。
- enum regression：测试断言所有 `wait_outcome` 属于固定集合，所有 `terminal_confirmation_source` 属于固定集合，且本任务没有使用 `rest_open_orders`。
- summary/log counter：测试按 `wait_requests`、`order_responses`、`terminal_confirmed`、`unknown_or_timeout` 分开统计。

verify：
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "live_shutdown or shutdown_cancel or wait_order_response or terminal_confirmed or order_response_received"`：`13 passed, 170 deselected`
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_*.py`：`298 passed`
- `git diff --check`：通过，无输出

done：
- `order_response_received` 与 `terminal_confirmed` 已拆成独立字段和独立语义。
- `wait_result == 3` 只设置 order-response 维度，不会单独设置 terminal proof。
- `wait_result == 0` 分类为 `ok_unknown_or_timeout`，不会被当成 confirmed ack。
- terminal confirmation 只由本地 final working-order proof 设置；本任务未做 REST final open-orders reconciliation。
- 未修改 py binding、Rust live bot/backtest、connector、production config、audit schema、normal loop cancel semantics。
- 未启动 live / tiny-live / default-on，未做 promotion。

blockers：
- 无。

commit：
- 待提交

提交信息：
- 待提交
