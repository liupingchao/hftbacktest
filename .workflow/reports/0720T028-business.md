# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0720T028

状态：
- 待验收

更新时间：
- 2026-07-20

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_online_estimators.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_online_estimators.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`
- `.workflow/tasks/0720T028.md`
- `.workflow/reports/0720T028-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Manager quote hold 不再 blind sleep；注入的 hold observer 在不改变 quote、submit、cancel、risk 或 kill-switch 决策的前提下消费现有 Hyperliquid 公共流。
- Observer 异常、source exhaustion、无公共事件、disconnect/reconnect 或超过 cancel-check bound 时 fail closed，但 manager 仍立即进入 owned-order cancel 和终态证明。
- 对每个 exact resting response 建立 `submit_end_ms < public local receive < cancel_request_time_ms` 的保守 interval contract；rejected、missing/inverted bounds 和 continuity failure 不进入 confirmed exposure。
- 将 interval 内事件转换为固定 1s event-time buckets；coverage 严格裁剪在 private response/cancel epoch-ms 边界内，首尾 partial bucket 保留真实 duration。
- Arrival 只使用 directional at-or-through unique trades，并使用同一 resting interval 内、arrival sequence 之前的 L2 side depth；trade id 全局去重，book state 按 bucket 状态去重。
- 新增 confirmed interval、exposure quarantine、replay exposure/intensity/snapshot artifacts；replay 从 event rows 加 interval contract 重建，不信任持久化 confirmed exposure。
- Task 12 acceptance 从 raw order response、intent、cancel request 和 hold observation 独立重建 interval contract，再独立重建 exposure；producer helper 未被调用。
- Manager contract 存在时，任何 interval/public-event quarantine 都阻断 estimator evidence acceptance；dynamic spread 和 actual quote behavior 始终保持 disabled/unchanged。
- 无 interval contract 时保留 v1 行为；immutable T026 不补造 interval start 或 confirmed exposure。

verify：
- Focused regression：`310 passed in 22.52s`。
- Full `python -m pytest -q examples/hyperliquid`：`977 passed in 42.84s`。
- `py_compile`、`git diff --check`、cached diff check：通过。
- Hostile/edge coverage 包含 observer exception 后仍 cancel、source exhaustion、disconnect/reconnect、missing/inverted bounds、无事件、invalid/future/out-of-order event、cross-bucket duplicate trade id、same-timestamp no-lookahead、stale buffered event、private-bound clipping 和 persisted interval tamper。
- 真实 writer fixture 保持 `quote_hold_seconds=3`，在持续公共流下通过完整 Task 12 acceptance。
- Replay fixture 从 interval contract 重建 exposure、A/k、candidate 和 snapshot；篡改 persisted confirmed exposure 时 `snapshot_match=false`。
- Exact T026 replay exit `0`：provenance `112/112`、config `72/72`、decision `43/43`、lifecycle `71/71`、economics `6/6`；confirmed interval/exposure 均为空。
- Exact T016/T022 replay 均保持 exit `2`、decision `43/43`、lifecycle `59 pass / 12 fail`；历史 blocker 未升级。
- 三份 replay boundary 均为 `offline_only=true`，network/private/order/cancel/remote/new-live 全部 false。
- 本任务未进行 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- Manager resting hold 现在能够持续收集公共事件，并输出 response-to-cancel-request 的保守 confirmed exposure evidence。
- Producer replay 和 Task 12 acceptance 可从独立输入重建相同 interval、exposure、fit 和 observe-only candidate。
- Invalid 或不完整 exposure evidence 会 quarantine 并阻断后续 estimator gate。
- Dynamic spread、fill feedback、inventory skew 和 multi-level 均未激活。
- T026 保持零 confirmed exposure；T016/T022 保持历史阻塞。

blockers：
- 无实现阻塞；独立 QA 是当前流程节点。
- 后续 Task 8 observe-only tiny-live 必须在 QA 通过后单独派发，并继续保持 dynamic spread activation disabled。

commit：
- 0e930dff76892430d4a99baed7deb7190c010af5

提交信息：
- Capture confirmed resting exposure
