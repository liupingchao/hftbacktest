执行线程：
- 业务线程-python

任务ID：
- 0605T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0605T003.md`
- `.workflow/reports/0605T003-business.md`
- `examples/binance_tick_mm/live_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`

action：
- 采用 final proof 语义方案 B：保留本地 terminal 判断，但通过 `final_proof_level` 明确区分 `local_only`、`exchange_reconciled`、`exchange_still_open`、`exchange_absent_only` 和 `none`。
- 扩展 `ShutdownCancelResult`，新增 `exchange_reconciliation_checked`、`exchange_open_order_absent`、`exchange_confirmation_source`、`exchange_reconciliation_status`、`exchange_open_orders` 和 `final_proof_level`。
- 在 shutdown cancel helper 中增加 safe exchange open-orders reconciliation：仅在传入现有 `rest_client` 和 `symbol` 且 `open_orders()` 可用时调用；无 client、无 symbol、无方法、调用失败或返回类型异常时只记录状态，不伪造 exchange confirmed。
- `run_live` shutdown 路径复用已有 REST client / symbol / tick size，summary logging 分离输出 wait、order response、terminal、本地/交易所 reconciliation、final proof level 和 failed 计数。
- 增加 shutdown final proof audit tail：追加 `shutdown_cancel_final_proof` lifecycle 行，在现有 audit schema 内记录 local proof、exchange reconciliation result 和 final proof level；不新增 audit schema 字段。
- 保持 `wait_result == 3` 只表达 `order_response_received=True`，不单独推导 `terminal_confirmed=True`。

verify：
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -k "shutdown_cancel or terminal_confirmed or exchange_reconciliation or local_absent or open_orders or final_proof_level"` -> `15 passed, 181 deselected`
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_*.py` -> `311 passed`
- `git diff --check` -> passed

done：
- `local_absent` 与 `exchange_absent` 已独立表达；`local_absent_from_local_orders` 保留为 local terminal proof，但 `final_proof_level=local_only`，只有同时 exchange absent 时才为 `exchange_reconciled`。
- exchange still-open 时输出 `final_proof_level=exchange_still_open`，不会给出高等级 final proof。
- exchange reconciliation 失败时输出明确 `open_orders_error:*` 状态，不伪造 `exchange_open_order_absent=True`。
- focused tests 覆盖 local absent / exchange absent、local absent / exchange still-open、exchange failure、local active、local terminal + exchange absent、summary counters 和 audit tail proof。
- 未修改 py binding、Rust live bot/backtest、connector 核心实现、production config、normal trading loop cancel 语义、strategy quote/submit/cancel 行为。
- 未启动 live / tiny-live / default-on，未做参数搜索、回放实验扩展或 promotion。

blockers：
- 无

commit：
- f6f6b20

提交信息：
- Harden shutdown final proof reconciliation
