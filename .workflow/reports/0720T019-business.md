执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T019

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_maker_order_manager.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0720T019.md`
- `.workflow/reports/0720T019-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 将 final open-orders reconciliation 与 position snapshot reconciliation 解耦。
- 当 final open-orders 有效但 user-state 无效时，仍以 exchange rows 重建 owned orders 和 working exposure，并以 `fail_closed` 标记 position snapshot 缺失。
- 让 final exchange row 在 supplied final snapshot 中覆盖本地 `cancel_requested` 展示状态为 `resting/partial`，同时保留 cancel request 时间和诊断字段。
- 增加 partial final snapshot、cancel-pending reappearance 和 position unavailable 回归测试。
- 未改变 pricing、signal、quote、risk cap、activation、live endpoint 或任何策略行为。

verify：
- T019 focused manager/watcher/fill/acceptance tests：`365 passed in 15.96s`。
- Full `python -m pytest -q -p no:cacheprovider examples/hyperliquid`：`759 passed in 36.15s`。
- `python -m py_compile`、acceptance CLI `--help`、`git diff --check`：通过。
- T016 exact replay：离线边界保持，无 network/private/order/cancel/remote 调用；decision `43/43` pass，lifecycle `49 pass / 12 fail`，历史 attempt 2 blocker 保持。
- T016 输入目录 path+bytes aggregate hash 为 `a5443f8f9cb777509ffde21f9caed3837a9ef58f31b25cd82acef2d6a3991398`，回放未改写输入。
- T019 implementation `git show --check`：通过。

done：
- 有效 final open-orders 不再因 position snapshot 失败而被丢弃。
- final operator order/exposure facts 与有效 final open-orders 保持一致。
- position snapshot 失败仍显式阻断，且不伪造零仓位。
- exchange-present order 不再以 `cancel_requested` 隐藏，cancel audit metadata 保留。
- T017/T018 已接受行为和 T016 历史阻断结果在回归与回放中保持。

blockers：
- 业务实现无已知阻塞；独立 QA 尚未完成，QA 通过前不启动新 bounded live、Task 8 或 adaptive/multi-level activation。

commit：
- `b0115d7cfe09f4612f68581481215f79026f0bfc`

提交信息：
- `Decouple final open-order snapshot reconciliation`
