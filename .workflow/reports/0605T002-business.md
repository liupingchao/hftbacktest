# 线程回报

执行线程：
- 业务线程-python

任务ID：
- 0605T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0605T002.md`
- `.workflow/reports/0605T002-business.md`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 新增 focused 诊断测试，只复现和定位 `0604T016` / `0605T001` 后剩余的 shutdown proof level 缺口，没有修改 `live_tick_mm.py` 生产行为。
- 只读确认当前 `local_absent_from_local_orders` 来自 `hbt.orders(asset_no)` 本地 snapshot；本地 snapshot 找不到同一 `order_id` 时会设置 `terminal_confirmed=True` / `terminal_confirmation_source=local_orders`。
- 用 fake/stub 稳定复现：本地 `orders()` 不含订单，但 fake `open_orders()` 仍含同一 `order_id` 时，当前 shutdown helper 不查询 `open_orders()`，仍返回 local absent terminal proof。
- 只读确认 `ShutdownCancelResult` 没有 exchange-side final proof 字段，例如 `exchange_reconciliation_checked`、`exchange_open_order_absent`、`exchange_confirmation_source`、`final_proof_level`。
- 只读确认 shutdown summary 当前只输出 attempts / sent / wait_requests / order_responses / terminal_confirmed / unknown_or_timeout / failed，没有 exchange-open-orders reconciliation counter。
- 只读确认 graceful shutdown block 没有写入 final no-open-order / exchange reconciliation audit tail record。

稳定复现矩阵：

| case | 复现方式 | 实际结果 | 结论 |
|---|---|---|---|
| `local_absent_sets_terminal_confirmed_without_exchange_check` | `test_shutdown_wait_three_with_local_terminal_proof_can_confirm_both_dimensions` / 新增 local-absent tests | 本地 snapshot absent 时 `terminal_confirmed=True`、`final_order_status=local_absent_from_local_orders` | 当前 local absent 是 terminal proof，但 proof source 是 local only |
| `local_absent_can_disagree_with_exchange_open_orders_stub` | `test_shutdown_local_absent_terminal_confirmed_is_local_only_without_exchange_check` | fake `exchange_open_order_ids={101}`，但调用记录只有 `("orders", 0)`，没有 `open_orders` | 当前 helper 不会发现本地 absent 与 exchange still-open 不一致 |
| `shutdown_result_has_no_exchange_final_proof_field` | `test_shutdown_result_has_no_exchange_final_proof_fields` | result 没有 `exchange_reconciliation_checked` / `exchange_open_order_absent` / `exchange_confirmation_source` / `final_proof_level` | 结果对象不能表达 exchange-side final proof |
| `shutdown_summary_has_no_exchange_reconciliation_counter` | `test_shutdown_summary_has_no_exchange_reconciliation_counter` | summary 有 wait/order/terminal/unknown/failed 计数，没有 exchange/open_orders/reconciliation | 日志口径没有 exchange final proof |
| `audit_tail_final_state_not_written` | `test_shutdown_audit_tail_final_exchange_state_not_written` | graceful shutdown block 没有 audit / reconciliation / final no-open-order 写入 | audit tail 缺少 final exchange-state proof |
| `rest_reconciliation_boundary_not_implemented` | 只读 grep / summary block 检查 | normal loop 有 REST open-orders diagnostics，但 shutdown helper 未接入；本任务未实现 REST | 属于后续修复候选，不是当前任务修复项 |

根因分层：
- local proof 层：`local_absent_from_local_orders` 只说明 wait 后的本地 `hbt.orders(asset_no)` snapshot 找不到同一 `order_id`。
- exchange proof 层：shutdown cancel proof 当前没有独立 REST/open-orders reconciliation，因此不能证明交易所侧 open orders 已清空。
- logging/reporting 层：shutdown summary 没有 exchange final proof counter，不能区分 local-only terminal proof 和 exchange-reconciled proof。
- audit 层：shutdown 阶段没有 final no-open-order / reconciliation tail record，后续无法从 audit artifact 独立验明最终交易所状态。
- tests 层：新增 focused fake 覆盖了 local absent 与 exchange still-open stub 不一致场景。

结论：
- 风险 1 / 3 属实，但更精确表述是：`0604T016` / `0605T001` 已修复 wait-result 和 local terminal-state 语义 bug；当前剩余的是更高 proof level 缺口，即 local absent proof 不等价于 exchange-side no-open-order proof。
- 这不是 `0605T001` 的回归；`0605T001` 对 `NEW` / `PARTIALLY_FILLED` active local state 的修复仍然成立。
- 如果近期要恢复 live/tiny-live，建议单独派发修复任务补 exchange reconciliation / audit tail proof；如果不恢复 live，该风险不是立即阻塞策略研究的紧急问题。

后续修复建议边界：
- 增加独立 exchange proof 字段，例如 `exchange_reconciliation_checked`、`exchange_open_order_absent`、`exchange_confirmation_source`、`final_proof_level`。
- 将 `local_absent_from_local_orders` 明确标为 local-only proof；是否仍设置 `terminal_confirmed=True` 由总控在后续修复任务中决定。
- 安全接入 REST open-orders reconciliation 时必须定义 timeout、权限缺失、REST 失败、仍存在 open order、symbol mismatch 的状态枚举。
- 增加 shutdown audit tail record，记录 cancel request、wait outcome、local proof、exchange reconciliation proof 和 final proof level。
- 修复任务仍应禁止策略 quote/submit/cancel 行为改变、参数搜索、live/default-on/tiny-live 和 promotion。

verify：
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "shutdown_cancel or terminal_confirmed or exchange_reconciliation or local_absent or open_orders"`：`11 passed, 180 deselected`
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_*.py`：`306 passed`
- `git diff --check`：通过，无输出

done：
- 已完成风险 1 / 3 的稳定复现和定位。
- 已区分当前已修 bug 与更高 proof level 缺口。
- 未修复 `live_tick_mm.py` shutdown helper。
- 未接入 REST/open-orders reconciliation。
- 未修改 py binding、Rust live bot/backtest、connector、production config、normal trading loop cancel、audit schema、strategy quote/submit/cancel 行为。
- 未启动 live / tiny-live / default-on，未做 promotion。

blockers：
- 无。

commit：
- `272b49e`

提交信息：
- `diagnose shutdown exchange reconciliation gap`
