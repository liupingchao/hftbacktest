执行线程：
- 业务线程-python

任务ID：
- 0605T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0605T004.md`
- `.workflow/reports/0605T004-business.md`
- `examples/binance_tick_mm/shutdown_proof_dry_run.py`
- `examples/binance_tick_mm/test_shutdown_proof_dry_run.py`
- `local_live_analysis/shutdown_proof_dry_run_0605T004/`

action：
- 新增本地 no-order dry-run runner `shutdown_proof_dry_run.py`，只使用 fake HBT / fake REST 对象调用 `cancel_working_orders_for_shutdown()` 和 `_append_shutdown_final_proof_audit_tail()`。
- 生成 6 个 dry-run 场景：`local_absent_exchange_absent`、`local_absent_exchange_still_open`、`exchange_check_failed`、`local_active_exchange_absent`、`local_terminal_exchange_absent`、`no_rest_client`。
- 输出 task-scoped artifacts：`run_manifest.json`、`shutdown_proof_summary.csv`、`shutdown_final_proof_audit_tail.csv`、`shutdown_proof_dry_run_report.md`。
- 新增 focused tests 覆盖 artifact 生成、scenario proof levels、`wait_result == 3` 与 terminal 独立性、audit tail final proof level。

verify：
- `/home/molly/anaconda3/envs/hftbacktest/bin/python examples/binance_tick_mm/shutdown_proof_dry_run.py --output-dir local_live_analysis/shutdown_proof_dry_run_0605T004` -> `all_scenarios_passed=true`, proof counts `{"exchange_absent_only": 1, "exchange_reconciled": 2, "exchange_still_open": 1, "local_only": 2}`
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_shutdown_proof_dry_run.py` -> `4 passed`
- `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm/test_*.py` -> `315 passed`
- `git diff --check` -> passed

done：
- dry-run artifacts 已写入 `local_live_analysis/shutdown_proof_dry_run_0605T004/`。
- manifest 明确 `network_used=false`、`live_started=false`、`real_orders_used=false`、`real_cancel_used=false`、`real_rest_used=false`。
- summary CSV 明确每个场景的 local proof、exchange reconciliation result、final proof level 和 pass 状态。
- audit tail CSV 记录 `shutdown_cancel_final_proof` 行，并在现有 schema 内包含 `final_proof_level`、exchange status、local status 和 exchange absent 标记。
- 本任务未连接交易所、未下单、未启动 live、未调用真实 REST private/order endpoint。
- 未修改 py binding、Rust live bot/backtest、connector 核心实现、production config、normal trading loop cancel 语义、strategy quote/submit/cancel 行为。
- 未做参数搜索、回放实验扩展或 promotion。

blockers：
- 无

commit：
- 4f7a19d

提交信息：
- Add shutdown proof dry-run validation
