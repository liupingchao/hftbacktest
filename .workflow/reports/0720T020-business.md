执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T020

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
- `.workflow/tasks/0720T020.md`
- `.workflow/reports/0720T020-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- `reconcile_supplied_snapshot()` 现在对所有 position semantic parsing failure 返回 `position_snapshot_status=fail_closed`、canonical reason 和 redacted error，同时保留已完成的 final open-order reconciliation。
- Finalizer 依据 manager reconciliation 返回值，而不是 `dict(...)` conversion 结果，决定 `final_position_snapshot_unavailable` blocker。
- `live_status.json` 显式输出 position snapshot status/reason；非 `pass` 时 operator-facing flat/exposure/risk `position_btc` 为 unknown。
- Manager 内部 position 数值未清零或伪造，继续保留用于保守风险计算；valid `assetPositions=[]` 仍是 verified zero。
- 新增 invalid root/object/list-row、valid-empty、risk-snapshot conflict、known working orders 和 partial final snapshot 对抗回归。
- 未改变 pricing、signal、quote、order、cancel、risk cap、activation 或 endpoint 行为。

verify：
- Focused manager/watcher/fill-attribution/acceptance：`377 passed in 16.24s`。
- Full `python -m pytest -q -p no:cacheprovider examples/hyperliquid`：`771 passed in 36.78s`。
- Modified modules/tests `py_compile`、acceptance CLI `--help`、`git diff --check`、implementation `git show --check`：通过。
- T016 exact replay：预期 exit `2`；decision `43/43`、lifecycle `49 pass / 12 fail`、provenance `112/112`、config `72/72`，历史 attempt 2 blocker保持。
- T016 replay boundary：offline-only，network/private/order/cancel/remote/new-live 均为 false。
- T016 输入 path+bytes aggregate hash 回放前后均为 `a5443f8f9cb777509ffde21f9caed3837a9ef58f31b25cd82acef2d6a3991398`。

done：
- Semantic final position result 现在控制 blocker 和 operator status。
- 无效 position snapshot 不再静默显示默认或旧数值为 verified。
- Valid empty positions 正确显示 verified `0.0`。
- Known final open orders、working exposure 和 cancel audit 语义保持。
- T017-T019 已接受行为及 T016 历史阻断保持。

blockers：
- 业务实现无已知阻塞；独立 QA 尚未完成，QA 通过前不启动新 bounded live、Task 8 或 adaptive/multi-level activation。

commit：
- `9a332c13e3a00d496e50c9ff73bc77b1962cec96`

提交信息：
- `Propagate final position snapshot validity`
