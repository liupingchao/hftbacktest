# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T030

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
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0720T030.md`
- `.workflow/reports/0720T030-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 为 manager hold pump 增加单槽 read-request queue；每次底层 `next(source_iterator)` 必须消费一个交易线程显式发出的 demand token。
- Pump 投递一个 event/exhaustion/error 后不再自动预取，而是等待下一次 demand；交易线程只有在完整处理前一结果并确认 hold 仍有效后才发下一 token。
- Disconnect、source exhaustion、exception 和 deadline 均终止 demand；退出时主线程设置 stop、发送 stop token，并执行 `0.05s` bounded join。
- Iterator close 由 pump thread 在退出 `finally` 中执行；交易线程不调用潜在阻塞的 arbitrary `close()`，因此 cancel deadline 不受 source close 行为控制。
- Idle built-in generator 在 bounded stop acknowledgement 内关闭，触发 websocket `close()`；任意正在阻塞的 custom read 可暂时保留 daemon thread，但返回后只能 close/exit，不能 publish 或发起下一次 read。
- Hold evidence 新增 `pump_stop_acknowledged`、`pump_read_inflight_at_stop`、`pump_source_closed` 和 redacted close error。
- T029 已接受的 websocket `0.25s` bound、estimator replay、quote、risk、submission 和 activation behavior 均保持不变。

verify：
- Built-in disconnect hostile manager cycle：
  - 首次 fake `_connect_websocket` failure 产生 disconnect；
  - 完整 manager cycle 返回并额外等待 `0.35s` 后，connect call count 仍严格为 `1`；
  - `pump_stop_acknowledged=true`，无 `manager-hold-public-event-pump` thread；
  - 两侧 cancel 完成，final owned orders 为空。
- Built-in normal L2 idle-stop probe：
  - 单一 recv event 后交易线程不再授权第二次 read；
  - `recv_calls=1`；
  - generator/source close 被 pump acknowledgement 执行；
  - websocket `closed=true`。
- Exact blocking-source deadline：
  - `quote_hold_seconds=3`，一次 `next()` 阻塞 `3.6s`；
  - 首 cancel 不晚于 deadline 加 `0.25s`；
  - 两次 cancel、final owned orders 为空；
  - blocking pump 返回后只 close/exit，没有后续 read。
- Public-state mutation、source exhaustion、disconnect 和 observer exception regression 均通过。
- Focused pump/manager hostile regression：`6 passed`。
- Full `python -m pytest -q examples/hyperliquid`：`984 passed in 47.31s`。
- `py_compile`、`git diff --check`、cached diff check、implementation `git show --check`：通过。
- Final-source exact T026 acceptance exit `0`：provenance `112/112`、config `72/72`、decision `43/43`、lifecycle `71/71`、economics `6/6`。
- Final-source exact T016/T022 acceptance 均保持预期 exit `2`：decision `43/43`、lifecycle `59 pass / 12 fail`。
- Historical replay boundary 均为 `offline_only=true`，network/private/order/cancel/remote/new-live 全部 false。
- 本任务未进行 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- T029 QA 的 post-disconnect prefetch/reconnect P2 已使用 per-event demand/ack 消除。
- Idle pump/source shutdown 在 observer 返回前得到 bounded acknowledgement；built-in generator 和 websocket 同步关闭。
- In-flight arbitrary blocker 不再控制交易线程 cancel timing，且无法发起第二次 source read。
- T030 已具备独立 QA 验收条件；QA 通过前 Task 8 observe-only tiny-live 继续锁定。

blockers：
- 无实现阻塞；独立 QA 是当前流程节点。

commit：
- `ae8468c324750f502c7b137124db032c5efc81d4`

提交信息：
- `Acknowledge manager pump shutdown`
