# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0717T001

状态：
- 已通过

更新时间：
- 2026-07-17 12:30 CST

验收线程：
- QA验收线程

验收对象：
- 业务线程-live-infra 0717T001

验收范围：
- 验收本轮是否修复 live evidence collection 对长公网 SSH 会话的硬依赖。
- 本 QA 只验收离线 infra contract，不验收 live execution、fill role evidence、fee/PnL、T004 public shadow 或策略有效性。

验收步骤：
1. Inspect task/report scope and confirm no strategy/live-envelope expansion.
2. Run focused orchestrator tests.
3. Compile the orchestrator and test module.
4. Check the CLI help path.
5. Run `git diff --check`.

实际结果：
- Added an SSM-friendly remote live orchestrator.
- The orchestrator records lock/status/heartbeat/window status/abort/complete/sha256 artifacts.
- The orchestrator attempts per-window read-only `open_orders()` proof by default.
- Offline tests cover complete and failed window paths without reading credentials or touching exchange endpoints.
- Documentation records SSM-first launch and recovery order.

验收结论：
- 已通过
- 结论说明：
  - The SSH disconnect risk is mitigated at the collection contract level: future live runs no longer need a long-lived SSH session as the evidence source of truth.

通过项：
1. Remote job status and heartbeat are persisted.
2. Failed windows write `abort_manifest.json`.
3. Completed runs write `run_complete.json` and `remote_sha256_manifest.txt`.
4. Offline tests prove complete and fail-closed behavior.
5. Scope does not change strategy or live envelope.

不通过项：
1. 无

缺陷清单：
1. 无

阻塞项：
- 无

建议总控下一步：
1. Use the SSM-first orchestrator for the next separately authorized live evidence run.
2. Consider a later task for S3 artifact upload and permanent systemd hardening.

提交信息：
- commit：911b9d3e483bb94b8871b908c2cc9402f9acaa25 / Add SSM-first live collection orchestrator
