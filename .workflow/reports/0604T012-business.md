```md
执行线程：
- 业务线程-research

任务ID：
- 0604T012

状态：
- 待验收

是否进行QA验收：
- 否

QA说明：
- 当前任务结果暂不进入QA验收，待总控确认后再决定是否派发QA验收。

files：
- `.workflow/tasks/0604T012.md`
- `.workflow/reports/0604T012-business.md`
- 只读检查：
  - `examples/binance_tick_mm/live_tick_mm.py`
  - `examples/binance_tick_mm/strategy_core.py`
  - `py-hftbacktest/hftbacktest/binding.py`
  - `py-hftbacktest/src/live.rs`
  - `hftbacktest/src/live/bot.rs`
  - `connector/src/binancefutures/mod.rs`
  - `connector/src/binancespot/mod.rs`

action：
- 已按 shutdown cancel case matrix 复现 live shutdown 不等待 cancel acknowledgement 的问题。
- 已用一次性 fake/stub Python 片段模拟 `live_tick_mm.py` shutdown finally 片段；未落盘复现脚本，未修改 shutdown 逻辑，未新增/修改测试。
- 已定位共同根因、分支根因、live safety 风险和 audit/alignment 尾部污染路径。

结论：
- review 属实。
- 当前 shutdown 路径会对最终本地 working orders 发出 cancel request，但调用参数固定为 `wait=False`，没有等待 cancel response。
- 底层 live bot `close()` 当前为空实现，不会兜底 drain pending cancel ack。
- 因此进程退出时可能存在“本地已经结束，但交易所仍有活跃挂单或 cancel ack 尚未落地”的窗口。

关键代码事实：
- shutdown finally 中 buy/sell/extra cancel 均使用 `hbt.cancel(..., False)`：
  - `examples/binance_tick_mm/live_tick_mm.py:1019`
  - `examples/binance_tick_mm/live_tick_mm.py:1022`
  - `examples/binance_tick_mm/live_tick_mm.py:1025`
- shutdown 之后直接读取 position 并调用 `hbt.close()`，没有 `wait_order_response()` 或 `wait_next_feed(include_order_resp=True)` drain：
  - `examples/binance_tick_mm/live_tick_mm.py:1031-1041`
- Python binding 明确：`wait=True` 才等待 cancel response：
  - `py-hftbacktest/hftbacktest/binding.py:1807-1821`
  - `py-hftbacktest/hftbacktest/binding.py:1835-1851`
- Rust live bot `cancel(..., wait)` 先发送 `LiveRequest::Order`；只有 `wait` 为 true 时才调用 `wait_order_response(...)`，否则直接 `Ok` 返回：
  - `hftbacktest/src/live/bot.rs:677-712`
- Rust live bot `close()` 当前只返回 `Ok(())`，没有 pending cancel ack / lifecycle drain：
  - `hftbacktest/src/live/bot.rs:763-765`
- normal loop cancel 也使用 `wait=False`：
  - `examples/binance_tick_mm/live_tick_mm.py:718`
  - 但 normal loop 后续继续 `wait_next_feed(True, ...)` 处理 order responses；shutdown 路径发完 cancel 后 close/return，风险集中在退出语义。
- connector cancel 是异步 `tokio::spawn` 后通过 channel 发布 order/error event；shutdown close 不等待这些事件被 live bot 处理：
  - `connector/src/binancefutures/mod.rs:427-475`
  - `connector/src/binancespot/mod.rs:313-360`

case matrix：

| case | 初始形态 | cancel calls | wait API calls | close calls | 判定 |
|---|---|---:|---:|---:|---|
| buy_only_cancellable | 1 buy, cancellable=true | `(0,101,False)` | 0 | 1 | 漏等 ack |
| sell_only_cancellable | 1 sell, cancellable=true | `(0,201,False)` | 0 | 1 | 漏等 ack |
| buy_sell_cancellable | buy+sell, both cancellable=true | 2, all wait=False | 0 | 1 | 漏等 ack |
| extra_buy_exists | primary buy + extra buy | 2, all wait=False | 0 | 1 | 漏等 ack |
| mixed_buy_sell_extras | buy/sell + buy/sell extras | 4, all wait=False | 0 | 1 | 漏等 ack |
| primary_not_cancellable | buy/sell cancellable=false | 0 | 0 | 1 | 不适用：未发 cancel |
| extra_not_cancellable_still_cancelled | primary buy cancellable=true + extra buy cancellable=false | 2, all wait=False | 0 | 1 | 分支问题：extra 未检查 cancellable 且漏等 ack |
| cancel_false_success_no_wait | single cancel succeeds | 1, wait=False | 0 | 1 | 漏等 ack |
| cancel_false_exception_still_position_close | first cancel raises | first cancel wait=False, later cancel skipped | 0 | 1 | 异常分支：catch 后继续 position/close，未完成剩余撤单 |
| close_after_pending_cancel_no_wait_api | buy+sell pending cancel | 2, all wait=False | 0 | 1 | close 不 drain pending ack |

一次性复现输出摘要：
- 所有已发 cancel 的正常 shutdown case 中，cancel 调用第三个参数均为 `False`。
- fake hbt 的 `wait_order_response_calls` 与 `wait_next_feed_calls` 均为空。
- 每个正常 case 都继续调用 `position()` 和 `close()`。
- `extra_not_cancellable_still_cancelled` 暴露额外边界：shutdown 对 `working_final.extra_ids` 直接 cancel，没有像 buy/sell 一样检查 `extra.cancellable`。
- `cancel_false_exception_still_position_close` 显示 cancel try 块一旦抛异常，后续同一 try 内剩余 cancel 会跳过，但 position/close 仍继续执行。

共同根本原因：
- shutdown cancel 被实现成 fire-and-forget request：`hbt.cancel(..., False)`。
- shutdown 没有在 cancel 后调用 `wait_order_response()`、`wait_next_feed(include_order_resp=True)`、REST final open-orders reconciliation、timeout/retry 或 lifecycle drain。
- live bot `close()` 没有关闭前 pending order response drain 职责。
- connector cancel response 是异步事件，shutdown 立即 close/return 时没有保证这些事件被处理到本地 order lifecycle/audit。

分支根因：
- buy/sell primary orders：只有 `cancellable=true` 才发 cancel，但发出后不等待 ack。
- extra orders：shutdown 只遍历 `extra_ids`，不检查 `ExtraOrder.cancellable` 或 `req`；可能对不可撤/已 cancel-pending extra 再发 cancel。
- cancel 异常：同一个 try 覆盖全部 shutdown cancels，某个 cancel 抛异常后剩余 orders 不会继续尝试 cancel；但外层仍继续 position/close。
- close 层：`close()` 为空实现，不兜底等待 pending cancel/order response。

live safety 影响：
- 进程退出后，交易所上可能仍残留活跃挂单或 cancel request 尚未确认。
- 如果 cancel REST call/connector task 仍在飞行中，进程结束会让本地无法确认最终状态。
- 如果 cancel 失败或未处理到本地，策略/用户可能误以为 shutdown 已完成撤单。

audit/alignment 影响：
- audit row/lifecycle event 在 event loop 内产生；shutdown cancel 发生在 final flush 之后，当前路径没有把 shutdown cancel ack / final REST open-order reconciliation 写进 audit tail。
- replay/alignment 看到的尾部 open-order / lifecycle 偏差可能来自 shutdown 语义，而不是策略 quote/cancel 行为本身。
- 尾部缺少 cancel_ack 会污染对 live terminal lifecycle、fill-after-cancel、open-order drift 的解释。

测试覆盖缺口：
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "shutdown or cancel"` 通过：`20 passed, 137 deselected`。
- 这些测试覆盖已有 cancel/guard/lifecycle 逻辑，但没有专门覆盖 live shutdown finally 是否等待 cancel ack。
- 当前没有 fake live hbt shutdown 单测验证：
  - cancel 调用应使用 `wait=True` 或显式等待。
  - 多订单 shutdown 应逐个等待/处理 ack。
  - extra order 是否检查 cancellable/req。
  - cancel 异常是否继续尝试剩余订单。
  - close 是否需要 drain 或 shutdown 层自行 drain。

建议后续修复任务边界：
- 只修复 `examples/binance_tick_mm/live_tick_mm.py` shutdown cancel/close 语义和 focused tests。
- 建议先不改 connector 或 Rust live bot `close()`，除非后续设计明确要求 close 具备全局 drain 职责。
- 修复方向建议：
  - shutdown cancel 对每个 cancellable order 使用 `wait=True`，或发送后显式 `wait_order_response(asset, order_id, timeout)`。
  - 为 shutdown 添加 bounded timeout，不允许无限卡住。
  - extra orders 应使用完整 `ExtraOrder` 对象，检查 `extra.cancellable` 和 `extra.req != "cancel"`。
  - cancel 某单失败不应阻止后续订单尝试 shutdown cancel。
  - shutdown 末尾可做 REST final open-orders check，并把 final result 写入 audit/lifecycle 或 shutdown report。
- 回归测试建议覆盖本报告 case matrix，证明：
  - buy/sell/extra 正常 case 会等待 ack。
  - 不可撤订单不会误发 cancel。
  - cancel 异常时仍尝试剩余订单。
  - close 不再被当作 cancel ack drain 的唯一机制。

verify：
- 只读代码检查完成。
- 一次性 fake/stub Python 复现片段完成，覆盖 10 个 shutdown cancel case。
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "shutdown or cancel"`
  - 结果：`20 passed, 137 deselected`

done：
- 已确认 review 属实。
- 已复现所有主要 shutdown cancel case。
- 已定位共同根因：shutdown 使用 `wait=False` 且没有显式 drain，底层 `close()` 不兜底。
- 已记录分支问题：extra 不检查 cancellable；cancel 异常会跳过剩余 cancel。
- 已给出后续修复任务边界。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
```
