# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程 / live-awsserver1

任务ID：
- 0721T037

状态：
- 待验收

更新时间：
- 2026-07-21 16:20:58 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0721T037.md`
- `.workflow/reports/0721T037-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/principal_alignment_estimator_observe_only_0721T037/`（本地 immutable evidence，按仓库规则忽略）

action：
- 使用 exact source `597a5e478f291eea2c4e7178bd3bbe187b034589` 构建隔离 Git archive。
- Local/remote source archive SHA-256 均为 `d7d9c743b95cffd84eb2b1c7a325f4dd748e7dab8c385afb99b776d87096564d`。
- Remote source marker exact：
  - `/home/admin/hftbacktest-cross-exchange-0721T037/source_commit.txt`
  - `597a5e478f291eea2c4e7178bd3bbe187b034589`
- 完成 exact no-start orchestrator preflight：
  - task/source/root/window/profile/duration/caps/manager/fast-L2 全部匹配；
  - dynamic spread、fill feedback、inventory skew、multi-level 和 actual quote behavior change 全部 false；
  - watcher、private、account、order、cancel endpoint 均未调用。
- 完成 pre-live read-only account/service proof：
  - account scope token 与 T031 exact 一致；
  - open orders `0`、BTC position `0.0`、kill-switch clear；
  - `xemm.service` inactive、T037 transient service inactive、conflicting process count `0`；
  - exact source marker、62-file runtime scope、executor import 和 SDK Python 全部 pass。
- 启动唯一一个 detached `principal-alignment-t037.service`：
  - Hyperliquid BTC lag、Binance public lead、post-only `Alo`；
  - one window、`1800s` max、`0.005 BTC/order`、`0.01 BTC` aggregate position、`1 USDC` loss、`2` submissions、`2` requotes、`3s` hold、`10s` wait；
  - fixed quote、single level、two-sided exchange-reconciled manager；
  - all adaptive activation off。
- Window 从 `2026-07-21T07:57:48Z` 运行到 `2026-07-21T08:14:24Z`，完成后没有启动第二个 window。
- 两侧真实 endpoint 结果：
  - buy `0.005 BTC @ 66238.0` 收到 exact post-only immediate-match rejection；
  - sell `0.005 BTC @ 66264.0` 收到 exact resting oid/cloid，随后 exact target-bound cancel success；
  - fills `0`，final owned/account open orders `0`，BTC position `0.0`。
- Terminal reconciliation：
  - submitted references `2/2`；
  - buy 由 exact submit rejection terminal；
  - sell 由 exact reference-bound cancel success terminal；
  - lifecycle/evidence acceptance `78/78 pass`。
- 本窗口未调用 historical endpoint：
  - direct/historical terminal audit attempt rows `0`；
  - historical fallback attempts `0`；
  - exact `4.0s/0.5s/5/1` delayed protocol fields存在且结构有效，但完整 live historical call path没有被本窗口实际触发。
- Manager observe-only evidence：
  - confirmed resting interval `1`，duration `3.055s`；
  - confirmed exposure `3` rows，positive duration合计 `2.142s`；
  - explicit leading left-censor `1` row，duration `438ms`；
  - quarantine `0` rows；
  - dynamic candidate `fallback_fixed`，authoritative half-spread保持 `0.5` ticks；
  - dynamic spread activation false，actual quote behavior unchanged。
- Pullback：
  - remote evidence file count `112`；
  - run terminal SHA-256 `107/107` pass；
  - complete transfer archive remote/local SHA-256 均为 `b6ab97cb3695bac12bbcbc75223bc82ff25c05a69ce9e929101504f7831568ae`。
- Post-live read-only account/service proof再次 pass：
  - same account scope、open orders `0`、BTC position `0.0`、kill-switch clear；
  - `xemm.service` inactive、T037 service inactive、conflicting process count `0`。

verify：
- Runtime source provenance：
  - exact source `597a5e478f291eea2c4e7178bd3bbe187b034589`；
  - source start verification `62/62 pass`；
  - source postrun verification `62/62 pass`；
  - missing/unexpected/mismatched files均为 `0`。
- Orchestrator：
  - child `rc=0`、reaped true；
  - no termination、no SIGKILL、no abort；
  - independent open-orders proof在 child exit 后执行并为 `0`；
  - terminal manifest `107/107 pass`。
- Estimator CLI replay exit `0`：
  - event rows `7499`；
  - intervals `1`；
  - persisted/rebuilt exposure `3/3`；
  - persisted/rebuilt censor `1/1`；
  - persisted/rebuilt quarantine `0/0`；
  - source/replay snapshot SHA-256 均为 `6a61b8767319c5785cf5ce611ca42474d99144b961912403afe76593030f2358`；
  - `snapshot_match=true`。
- Same-window acceptance exit `2`：
  - provenance `112/112 pass`；
  - config `72/72 pass`；
  - decision `39 pass / 4 fail`；
  - lifecycle/evidence `78/78 pass`；
  - economics `6/6 pass`；
  - optimism `6/6 pass`；
  - final recommendation `principal_task12_same_window_acceptance_blocked`。
- Four decision failures reduce to one raw evidence contradiction at event sequence `3053`：
  - immediate guard row is `fail_closed` with `outside_quality_a_b_queue_bands;missing_intent_limit_px;missing_or_nonpositive_intent_size;missing_quality_bucket`；
  - post-open-orders anti-drift row is `block` with `adverse_trade_pressure_with_recent_adverse_bbo`；
  - trigger/attempt `guard_status` chooses anti-drift as the primary status/reason；
  - attempt and anti-drift-submit `skip_reason` choose the immediate-guard reason；
  - independent reconstruction therefore emits:
    - `attempt_skip_reason_mismatch:18:3053`
    - `immediate_guard_trigger_join_mismatch:18:3053`
    - `trigger_anti_drift_guard_mismatch:3053`
- No credential value、raw account address、raw signature or unredacted order reference was persisted。
- No second live window、flatten、dynamic spread、fill feedback、inventory skew or multi-level activation occurred。

done：
- One and only one authorized fixed-quote tiny-live window is preserved locally。
- Execution/account safety、exact source、terminal reference proof、confirmed resting exposure、explicit censor/quarantine and deterministic estimator replay all pass。
- Same-window mechanism/evidence acceptance remains fail closed because producer artifacts do not define one consistent primary reason when immediate guard and anti-drift both fail on the same event。
- The complete delayed historical endpoint path was not exercised by this immutable window and remains unproven by new live evidence。
- No stable A/k、fill-rate、fees/rebates、PnL、maker viability、multi-level or promotion claim is supported。

blockers：
- Decision evidence primary-cause precedence is inconsistent for simultaneous immediate-guard and anti-drift failures。
- This immutable window contains no actual historical fallback call；the repaired full `4s` delayed-history live path remains unexercised。
- No second window is allowed in T037。
- Dynamic spread activation and Principal Task 9-12 remain locked pending independent QA and offline repair。

commit：
- source `597a5e478f291eea2c4e7178bd3bbe187b034589`

提交信息：
- source `Record T036 QA acceptance`
