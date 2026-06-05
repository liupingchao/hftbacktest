# 线程回报

执行线程：
- 业务线程-python

任务ID：
- 0605T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0605T001.md`
- `.workflow/reports/0605T001-business.md`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 新建 `0605T001` 窄修复任务，专门处理 `0604T016` QA 指出的 `PARTIALLY_FILLED` active local order 被误判为 terminal confirmed 的问题。
- 修改 `_confirm_shutdown_terminal_state_from_local_orders(...)`，不再复用只收集 `NEW` 订单的 `collect_working_orders(...)` 作为 final proof。
- 新增本地 shutdown order snapshot 遍历和状态分类：
  - `NEW` / `PARTIALLY_FILLED`：active，`terminal_confirmed=False`。
  - `EXPIRED` / `FILLED` / `CANCELED` / `REJECTED`：terminal，`terminal_confirmed=True`。
  - unknown / status unavailable：不确认 terminal，`terminal_confirmed=False`。
  - local snapshot 找不到该 `order_id`：保留 local absent proof，`terminal_confirmed=True`。
- 保持 `order_response_received` 与 `terminal_confirmed` 独立，没有改动 T016 已通过的 wait-result 分类、summary counter、REST 边界或 shutdown cancel request 行为。
- 补 focused regression：`wait_result=3 + local PARTIALLY_FILLED same order_id` 必须 `order_response_received=True` 且 `terminal_confirmed=False`。
- 补本地 terminal status regression：`FILLED` / `CANCELED` / `REJECTED` 可独立确认 terminal，即使 `wait_result=0` 仍不表示 order response。

verify：
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "live_shutdown or shutdown_cancel or wait_order_response or terminal_confirmed or order_response_received or partially_filled"`：`14 passed, 173 deselected`
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_*.py`：`302 passed`
- `git diff --check`：通过，无输出

done：
- 已修复 T016 QA 缺陷：`PARTIALLY_FILLED` 本地订单不再被过滤成 absent/terminal。
- 已保留本地 absent proof、terminal status proof 和 no-proof failure semantics。
- 未接入 REST final open-orders reconciliation。
- 未修改 py binding、Rust live bot/backtest、connector、production config、normal trading loop cancel、audit schema、strategy quote/submit/cancel 行为。
- 未启动 live / tiny-live / default-on，未做 promotion。

blockers：
- 无。

commit：
- `4d2624c`

提交信息：
- `fix shutdown terminal proof status handling`
