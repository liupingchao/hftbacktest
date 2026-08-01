# 线程回报

执行线程：
- 业务线程-python/public-collector

任务ID：
- 0729T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0729T004.md`
- `.workflow/reports/0729T004-business.md`
- `examples/hyperliquid/synchronized_public_collection.py`
- `examples/hyperliquid/test_synchronized_public_collection.py`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Binance subscribe 后立即启动单一 WebSocket reader，并在 REST snapshot
  获取期间持续接收消息。
- 使用有界 queue 和有界 pre-bridge retained buffer；任一缓冲区溢出都
  fail closed。
- snapshot 有效后丢弃 `u < lastUpdateId` 的旧 depth，只接受第一条满足
  `U <= lastUpdateId <= u` 的 bridge event。
- snapshot 在 raw stream 中写到 bridge event 之前，再从 bridge event
  开始写入 diff-depth。
- bridge 后每条 depth 强制 `pu == previous u`；字段、数组或连续性不合法
  时 fail closed。
- 到达采集 deadline 后等待 reader 退出并 drain 已接收的队列尾部，避免
  成功样本在结束瞬间静默截断。
- 每次 WebSocket reconnect 都重新获取 snapshot、重新寻找 bridge，并
  重置该连接的连续性链。
- reconnect 只会在旧 reader 已确认退出后发生；reader 在 close+join 后
  仍存活时记录 shutdown event、禁止 reconnect 并 fail closed。
- manifest 新增 bootstrap、bridge、buffer peak/overflow、discarded
  depth、continuity gap、reader shutdown 和 `depth_replay_ready` 证据
  字段。
- CLI 新增 `--bootstrap-buffer-max-messages` 及 orchestrator 对应参数。
- 保持 Hyperliquid、symbol profile、public-only boundary、alignment 和
  策略行为不变。

verify：
- `python -m py_compile examples/hyperliquid/synchronized_public_collection.py examples/hyperliquid/test_synchronized_public_collection.py`
  passed。
- `python examples/hyperliquid/synchronized_public_collection.py collect-binance-public --help`
  passed，包含 `--bootstrap-buffer-max-messages`。
- `python examples/hyperliquid/synchronized_public_collection.py collect --help`
  passed，包含 `--binance-bootstrap-buffer-max-messages`。
- `python -m pytest examples/hyperliquid/test_synchronized_public_collection.py -q`
  passed：`18 passed`。
- 聚焦覆盖：
  - reader 在 snapshot 返回前已启动
  - valid bridge
  - missing bridge fail closed
  - `pu/u` continuity gap fail closed
  - reader queue overflow fail closed
  - retained pre-bridge buffer overflow fail closed
  - stale `u < lastUpdateId` depth is counted, discarded and absent from raw
  - blocked reader shutdown prevents reconnect and records fail-closed evidence
  - reconnect reacquires snapshot and rebridges
  - deadline shutdown drains the reader tail before success
- `python -m pytest examples/hyperliquid -q --ignore=examples/hyperliquid/test_hyperliquid_raw_alignment.py --ignore=examples/hyperliquid/test_hyperliquid_tiny_live_sdk_readiness.py`
  passed：`1309 passed, 2 skipped`。
- 未过滤目录命令受两个既有环境项影响：
  - `test_hyperliquid_raw_alignment.py` 在收集阶段失败，因为当前
    NumPy `2.4` 超过 Numba 支持的 `<=2.3`。
  - `test_hyperliquid_tiny_live_sdk_readiness.py` 当前环境返回
    `hyperliquid_official_sdk_readiness_blocked`，单测结果
    `1 failed, 1 passed`。
- `git diff --check` passed。

done：
- T003 的 Binance snapshot-to-depth 启动桥接缺陷已在采集器实现层修复。
- 新采集只有在 snapshot 有效、bridge 已建立、全链连续且正常跑到时限时
  才会标记 `depth_replay_ready=true`。
- 失败路径不会把缺失 bridge、缓冲区丢消息或连续性断裂的数据声明为
  replay-ready。
- 本任务仅完成离线实现和测试；没有 AWS/远端执行、没有交易所采集，也
  没有重新运行 2H/8H SKHYNIX 窗口。
- 因此本轮证明的是 collector contract 和回归覆盖，不是新的 exact
  Binance 或双交易所订单簿回放实测证据。

blockers：
- 本次代码修复无阻塞。
- 新的 exact replay 实测结论仍需另行授权并运行 canary/2H 数据采集。
- 当前本机全量目录原始命令仍受 NumPy/Numba 和 Hyperliquid official SDK
  readiness 两个环境项影响。

commit：
- 无

提交信息：
- 无
