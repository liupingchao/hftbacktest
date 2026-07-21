# 0721T034 Business Report

执行线程：
- 总控 auto-loop / 业务执行线程

任务ID：
- 0721T034

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/cross_exchange_t024_same_window_acceptance.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_attribution.py`
- `examples/hyperliquid/test_cross_exchange_t024_same_window_acceptance.py`

action：
- Producer audit 和 Task 12 独立 audit helper 不再以 exact protocol marker 自身决定是否运行 delayed-history validation。
- 任一 historical call/result 或任一 delayed timing/budget field 存在时，都要求 exact `delayed_one_call_history_v1` marker，并继续执行 T033 的 `4.0/0.5/5/5.0/1` exact value/type/timing contract。
- Missing、empty、legacy、wrong 或 boolean marker 在 delayed contract required 时均显式产生 `terminal_audit_history_protocol_invalid`，producer 与 independent helper 保持完全一致。
- Task 12 T034+ missing/wrong marker 端到端 fail closed。
- 无 historical call、无 delayed timing evidence 的 legacy direct-only v4 contract 仍按既有边界通过。
- Live executor、endpoint timing、quote、risk、size、submission、side-set 和所有 adaptive activation 未改变。

verify：
- Producer/audit focused regression：`189 passed`。
- Task 12 focused regression：`212 passed`。
- Combined four-file focused regression：`524 passed in 27.65s`。
- Full Hyperliquid regression：`1027 passed in 47.83s`。
- Python compile、`git diff --check` 通过。
- T031 estimator replay 保持 blocked evidence：exposure persisted/rebuilt `6/6`，censor `0/2`，quarantine `1/0`，`snapshot_match=false`。
- T031 acceptance 保持 blocked：provenance `112/112`、config `72/72`、decision `43/43`、lifecycle/evidence `55 pass / 23 fail`、economics `6/6`。
- T026 exact acceptance 保持 pass，lifecycle/evidence `78/78 pass`。
- T016/T022 exact acceptance 保持 blocked，lifecycle/evidence 均为 `66 pass / 12 fail`。
- 全程 offline；未触发 live、private/account、order、cancel、network、remote 或 service 操作。

done：
- T033 QA 的唯一 P1 已在 producer、independent helper 和 Task 12 三层 fail closed。
- 修复 source 已准备进入独立 QA；通过前不允许新 private read、live 或 adaptive/multi-level activation。

blockers：
- 独立 QA acceptance。

commit：
- fe8d725304ca8edf27b32de8f95322bce458154d

提交信息：
- Repair delayed history marker trigger
