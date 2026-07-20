# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T029

状态：
- 待验收

更新时间：
- 2026-07-20

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/cross_exchange_online_estimators.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_cross_exchange_online_estimators.py`
- `.workflow/tasks/0720T029.md`
- `.workflow/reports/0720T029-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Manager hold 使用唯一 daemon public-event pump 调用底层 iterator；pump 只传递 event、exhaustion 或 exception，所有 `EventDrivenPublicState` mutation 继续由交易线程执行。
- 交易线程以 `min(hold_remaining, 0.25s)` 等待 pump 结果；底层 iterator 即使阻塞，也不能阻止交易线程在 hold deadline 后进入 owned-order cancel。
- Pump stop 后不会继续调用下一次 source read；内建 manager live source 的 websocket timeout 被限制为 `0.25s`，legacy/non-manager 路径保留配置值。
- Source exhaustion、disconnect、exception、无公共事件和 continuity failure 继续 fail closed；observer failure 后两侧 cancel 和终态证明仍执行。
- Estimator replay 从 contract 文件存在性和精确 producer CSV schema 判断 manager contract，而不再使用 row-list truthiness。
- Contract 存在时，无论是否有数据行，所有 persisted confirmed exposure 都从 replay input 排除，只允许从 event rows 加 interval contract 重建。
- Replay manifest 新增 contract presence、schema validity 和 quarantine-empty 字段；`snapshot_match` 与 CLI success 同时要求 snapshot hash、confirmed exposure、schema 和 quarantine 全部通过。
- Dynamic spread、fill feedback、inventory skew、multi-level 和 actual quote behavior activation 保持关闭。

verify：
- Exact hostile manager timing：`quote_hold_seconds=3`、单次 source `next()` 阻塞 `3.6s`；首个 cancel 不晚于 hold deadline 加 `0.25s`，`cancel_count=2`，final owned orders 为空，hold evidence 为 `fail_closed / manager_hold_no_public_event`。
- Built-in source integration：manager websocket timeout 为 `0.25s`；non-manager 配置 `0.9s` 原样保留。
- Replay hostile cases：
  - invalid interval 产生 quarantine `1`，`snapshot_match=false`，CLI exit `1`；
  - header-only valid contract 加 forged persisted confirmed exposure 时，persisted/rebuilt 为 `1/0`，forged row 不进入 replay，`snapshot_match=false`；
  - malformed contract schema 时 `confirmed_resting_contract_schema_valid=false` 且 replay fail closed；
  - contract absent 的 v1 replay 保持 `snapshot_match=true`。
- Focused hostile regression：`9 passed`。
- Full `python -m pytest -q examples/hyperliquid`：`982 passed in 48.93s`。
- `py_compile`、`git diff --check`、cached diff check：通过。
- Exact T026 acceptance exit `0`：provenance `112/112`、config `72/72`、decision `43/43`、lifecycle `71/71`、economics `6/6`。
- Exact T016/T022 acceptance 均保持预期 exit `2`：decision `43/43`、lifecycle `59 pass / 12 fail`；历史 terminal blocker 未升级。
- T026/T016/T022 estimator replay 均为 contract absent legacy path、snapshot hash exact、confirmed interval/exposure/quarantine `0/0/0`。
- 所有 historical replay boundary 均为 `offline_only=true`，network/private/order/cancel/remote/new-live 全部 false。
- 本任务未进行 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- T028 P1 已修复：manager cancel timing 不再依赖阻塞 public iterator 返回。
- T028 P2 已修复：present-empty、invalid schema 和 quarantine evidence 均不能得到 replay/CLI success。
- Existing confirmed exposure reconstruction、Task 12 acceptance、T026 zero exposure 和 T016/T022 historical blocker 均保持。
- T029 已具备独立 QA 验收条件；QA 通过前 Task 8 observe-only tiny-live 继续锁定。

blockers：
- 无实现阻塞；独立 QA 是当前流程节点。

commit：
- `15de951f3d9911d4a3e7070080a48be24e5f4ec8`

提交信息：
- `Bound manager hold and replay evidence`
