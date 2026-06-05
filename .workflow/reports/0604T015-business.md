# 线程回报

执行线程：
- 业务线程-python

任务ID：
- 0604T015

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0604T015.md`
- `.workflow/reports/0604T015-business.md`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 新增 focused 诊断测试，只证明当前 `0604T013` 后 shutdown cancel wait 仍不能证明 cancel acknowledgement，没有修改 production 行为。
- 只读确认 `examples/binance_tick_mm/live_tick_mm.py` 中 `cancel_working_orders_for_shutdown(...)` 当前逐单调用 `hbt.cancel(..., False)` 后 `hbt.wait_order_response(...)`，但 `ShutdownCancelResult` 只记录 `wait_result` 原始 int 和 `error`。
- 只读确认 shutdown summary 的 `ack_waits` 由 `wait_requested` 计数得到，因此它表示 wait attempts，不是 confirmed cancel ack。
- 只读确认 Python live binding 返回码：`ElapseResult::Ok -> 0`，`ElapseResult::OrderResponse -> 3`。
- 只读确认 Rust live bot 非 batch 路径收到指定订单响应会返回 `OrderResponse`，timeout 返回 `Ok`，batch 路径收到响应后在 `BatchEnd` 折叠为 `Ok`。

稳定复现矩阵：

| case | 复现方式 | 实际结果 | 结论 |
|---|---|---|---|
| `live_wait_response_non_batch_order_response_returns_3` | 只读源码 + diagnostic test 断言 live binding / live bot 路径 | `OrderResponse -> 3` | `3` 明确是 order response，不是 cancel-state proof |
| `live_wait_response_timeout_returns_0` | 只读源码 + diagnostic test 断言 timeout path | timeout 返回 `Ok -> 0` | `0` 不能表示 confirmed ack |
| `live_wait_response_batch_received_response_returns_0` | 只读源码 + diagnostic test 断言 batch `wait_resp_received` path | batch 内收到响应后 `BatchEnd` 返回 `Ok -> 0` | `0` 同时覆盖 timeout 和 batch received-response |
| `shutdown_helper_records_raw_zero_as_no_error` | fake hbt 返回 `wait_result=0` | helper 记录 `wait_result=0`、`error=""`，无 timeout/ack 字段 | 当前 helper 无法分类 `0` |
| `shutdown_helper_records_three_but_not_cancel_ack_state` | fake hbt 返回 `wait_result=3` | helper 记录 `wait_result=3`、`error=""`，未查订单状态 | `3` 只证明有 order response，不证明 canceled/inactive |
| `shutdown_summary_ack_waits_is_wait_requested_not_ack_confirmed` | fake hbt 分别返回 `0` 与 `3` 后复算 summary 口径 | `waited=2`，但明确 response 只有 `1` | `ack_waits` 名称/口径会误导为 ack count |
| `existing_shutdown_tests_fake_zero_masks_semantics` | 检查并扩展现有 fake tests | 旧 fake 默认返回 `0` 且旧测试把 `0` 当无错误成功 | 旧测试只能证明调用了 wait，不能证明收到 ack |
| `order_response_status_not_checked` | fake hbt 暴露 `orders/open_orders/order_status` 并记录调用 | `status_checks == []` | helper 没有 local/final status proof |
| `audit_tail_status_not_proven` | 只读 helper / shutdown summary | helper 只返回内存结果并写 log summary，无 audit tail reconciliation | final live/audit state proof 缺失 |

根因分层：
- API 语义层：Python binding 中 `0` 是 `ElapseResult::Ok`，不是 confirmed ack；`3` 是 `ElapseResult::OrderResponse`，仍不是 canceled-state proof。
- live bot batch 层：batch mode 已收到目标 order response 后，`BatchEnd` 返回 `Ok`，导致 Python 侧同样是 `0`。
- shutdown helper 层：`ShutdownCancelResult` 只有 `wait_result` 原始值，没有 `timeout_or_ok`、`order_response_received`、`cancel_ack_confirmed`、`final_order_status` 等分类字段。
- logging/reporting 层：`ack_waits` 由 `wait_requested` 计数而来，实际含义是发起等待次数，不是 ack 确认次数。
- tests 层：原 fake `wait_order_response()` 返回 `0`，旧测试断言 `wait_result == 0` 且 `error == ""`，会掩盖真实 binding 语义。
- audit/live-safety 层：shutdown 没有 final local/REST open-order reconciliation，也没有 audit tail 证明订单最终 canceled/inactive。

结论：
- review 属实，但需要更精确地表述：非 batch live 路径中指定订单响应映射为 `3`；`0` 是 `Ok`，可来自 timeout，也可来自 batch 收到响应后的折叠。因此 `0` 不是 confirmed ack。
- 即使 `wait_result == 3`，当前 helper 也只能证明收到该 `order_id` 的 order response，不能证明响应语义是 cancel ack，也不能证明订单最终 canceled/inactive。
- 当前 shutdown 已经从“不等”推进到“有 bounded wait”，但仍没有推进到“可证明 cancel ack / final exchange state”。

后续修复建议边界：
- 将 `wait_order_response()` raw int 分类为 `order_response_received`、`timeout_or_no_response`、`unknown_ok_batch_or_timeout` 等明确状态。
- 不要把 `ack_waits` 作为 confirmed ack 口径；改名或增加单独 `confirmed_responses` / `confirmed_cancel_acks` / `unknown_waits`。
- 对 `wait_result == 3` 仍需复查 local order status 或引入 final REST open-orders reconciliation，才能形成 canceled/inactive proof。
- 需要补 audit/log 尾部证据，区分 shutdown cancel request、order response、final no-open-order proof。
- 需要修正 fake test semantics，让 `0` 不再被默认为 “ack success”。

verify：
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "live_shutdown or shutdown_cancel or wait_order_response"`：`12 passed, 167 deselected`
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_*.py`：`294 passed`
- `cargo test -p hftbacktest live::bot`：`2 passed, 22 filtered out`
- `git diff --check`：通过，无输出

done：
- 已稳定复现所有任务要求的主要问题。
- 已定位根因到 binding 返回码、live bot batch/timeout 返回路径、shutdown helper raw-result 保存、summary 统计口径、测试 fake 语义和 final state proof 缺口。
- 未修复 `live_tick_mm.py` shutdown helper。
- 未修改 py binding、Rust live bot/backtest、connector、生产配置、audit schema、normal loop cancel 语义。
- 未启动 live / tiny-live / default-on，未做 promotion。

blockers：
- 无。

commit：
- `0e8995352af2f3b48f98ca4fd8a85e4ccc2b0a2f`

提交信息：
- `diagnose shutdown cancel wait ambiguity`
