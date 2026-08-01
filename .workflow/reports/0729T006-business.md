# 线程回报

执行线程：
- 业务线程-python/public-collector

任务ID：
- 0729T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0729T006.md`
- `.workflow/reports/0729T006-business.md`
- `examples/hyperliquid/synchronized_public_collection.py`
- `examples/hyperliquid/test_synchronized_public_collection.py`

action：
- 新增有界 `snapshot_bridge_refresh_attempts`。
- snapshot 小于第一条可用 buffered diff `U` 时，保留当前 WebSocket
  reader、queue 和 retained buffer，重新获取 REST snapshot。
- 每次 snapshot refresh 都从完整 retained buffer 重新寻找
  `U <= lastUpdateId <= u` bridge。
- refresh 耗尽时抛出 `binance_snapshot_refresh_exhausted` 并 fail closed。
- manifest 新增：
  - `depth_snapshot_refresh_count`
  - `depth_snapshot_request_count`
  - `depth_snapshot_bridge_refresh_attempts`
  - 每次 refresh 的 snapshot ID、状态和 stale candidate
- CLI/orchestrator 新增对应参数并显式传递。

verify：
- py_compile passed。
- focused synchronized collector：`19 passed`。
- synchronized + Hyperliquid public sample：`22 passed`。
- 环境兼容目录回归：`1310 passed, 2 skipped`。
- CLI help 包含两个 snapshot bridge refresh 参数。
- git diff check passed。

done：
- 首次 snapshot 过旧、第二次 snapshot 在同一 WebSocket 连接上 bridge
  的回归通过；connection attempts `1`、reconnects `0`。
- refresh 耗尽路径保持 fail closed。
- 既有 continuity、overflow、reconnect、reader shutdown 和 deadline
  tail drain 回归保持通过。

blockers：
- 无。
- 本任务为 offline repair；新的 live 证据需要独立任务。

commit：
- 无

提交信息：
- 无
