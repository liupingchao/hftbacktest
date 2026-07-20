# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程 / live-awsserver1

任务ID：
- 0720T026

状态：
- 待验收

更新时间：
- 2026-07-20 19:12 CST

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0720T026.md`
- `.workflow/reports/0720T026-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/principal_alignment_single_level_two_sided_0720T026/`（本地证据，按仓库规则忽略）

action：
- 从 exact source `40dc56a3225df4afb0f2185873c91f17b578f550` 构建隔离远端运行树；local/remote archive SHA-256 均为 `b92b4b09dcac2b5d0d8fa615823c87d75d3cdd7c6220a1fd50dce2e91776be6e`。
- 完成 exact no-start orchestrator preflight 和只读账户/服务 preflight：
  - open orders `0`、BTC position `0.0`、kill-switch clear；
  - `xemm.service` inactive，无冲突交易进程；
  - source/import/runtime files `62/62` exact。
- 在 `awsserver1` 启动唯一一个 detached two-sided-manager window：
  - Binance public lead；
  - Hyperliquid BTC post-only `Alo`；
  - `1800s` maximum、`0.005 BTC/order`、`0.01 BTC position`、`1 USDC loss`、`2 submissions`；
  - dynamic spread、fill feedback、inventory skew、multi-level 和 actual quote behavior change 全部关闭。
- Window 从 `2026-07-20T11:00:23Z` 运行到 `2026-07-20T11:07:03Z`，完成两侧 endpoint 尝试和 manager terminal cycle；没有启动第二个 window。
- 实际两侧结果：
  - buy `0.005 BTC @ 64393.0` 到达真实 order endpoint，交易所明确返回 post-only immediate-match reject，响应 BBO 为 `64391@64392`；
  - sell `0.005 BTC @ 64419.0` 到达真实 order endpoint并 resting，随后获得 exact oid/cloid-bound cancel success；
  - submissions `2`、resting `1`、post-only reject `1`、fills `0`。
- Producer 将明确 rejected 的 buy cloid 仍纳入 cancel/history reference 集合，但没有把原始 exchange rejection 识别为 authoritative terminal，因而记录首个 blocker `fill_reconciliation_required_no_fill_unproven`。
- 回收完整远端 evidence root，验证 terminal SHA-256 `104/104`，并运行 exact same-window acceptance。

verify：
- Pre-live account/service proof：pass；source、account scope、kill-switch、open orders、position、service/process 全部满足任务边界。
- Orchestrator preflight：pass；one window、exact `1800s`、two-sided manager、fast L2、requote `2`，且 no-start execution boundary 全部为 false。
- Runtime source start/postrun verification：均为 `62/62` pass，source commit exact。
- Live decision evidence：
  - public evaluations `1163`；
  - trigger rows `14`；
  - anti-drift pass/block `20/4`；
  - edge pass/block `1/4`；
  - manager/submitted attempt identities `2/2`；
  - validation reasons `[]`。
- Child `rc=0`、已 reap、未请求 termination 或 SIGKILL；transient service `Result=success` 且最终 inactive。
- Independent after-child open-orders proof：`0`。
- Post-live independent account proof：pass；open orders `0`、BTC position `0.0`、kill-switch clear、无冲突服务/进程。
- Final live status：owned open orders `0`、BTC position `0.0`、estimated loss `0.0 USDC`。
- Writer health：healthy，successful writes `351`、failures `0`。
- Remote/local terminal checksums：`104/104` pass。
- Same-window acceptance exit `2`：
  - provenance `112 pass / 0 fail`；
  - config `72 pass / 0 fail`；
  - decision replay `41 pass / 2 fail`；
  - economics boundary `6 pass / 0 fail`；
  - lifecycle `46 pass / 15 fail`；
  - mechanism/evidence integrity `fail`；
  - final recommendation `principal_task12_same_window_acceptance_blocked`。
- 两个 decision failures 都来自 acceptance 的 exact-two-resting 假设：buy 原始响应是明确 `error/rejected`，不是 resting payload。
- 十五个 lifecycle failures 收敛到同一分类缺口：
  - acceptance 期望两侧均 resting；
  - producer/acceptance 未把 exact response-bound post-only rejection 计为 authoritative terminal；
  - cancel reconciliation 因而错误要求 rejected buy 具备 cancel/history 证明，仅报告 `1/2` references proven。
- Raw evidence 没有 unknown/ambiguous response：buy 是 reference-bound explicit reject，sell 是 reference-bound resting 加 cancel success。
- No credential value、raw signature、raw account identifier 或 unredacted order reference被持久化。
- No second live window、flatten、strategy relaxation 或 adaptive/multi-level activation发生。

done：
- One and only one authorized single-level two-sided bounded-live window is preserved locally。
- Buy/sell 两侧都到达真实 post-only endpoint；buy 获得明确 rejected terminal，sell 获得 resting 加 authoritative cancel terminal。
- Account terminal safety independently passes：no open orders、no BTC position、no observed loss。
- 当前 producer/acceptance 的 first exact mechanism stop condition已被原样记录，没有通过 account-wide absence 或乐观解释绕过。
- No stable PnL、fill-rate calibration、queue priority、maker viability、promotion、multi-level 或 final MVP claim is supported。

blockers：
- Current source/acceptance 没有把 exact exchange post-only rejection 纳入 reference-bound authoritative terminal contract，并错误要求两侧都 resting。
- Same-window mechanism/evidence integrity当前仍为 `fail`；Task 8 和 adaptive/multi-level activation继续锁定。
- 下一任务应为 offline-only rejected-terminal lifecycle repair；T026 raw evidence不得改写，也不得在 T026 内启动第二个 window。

commit：
- source `40dc56a3225df4afb0f2185873c91f17b578f550`

提交信息：
- source `Record T025 QA acceptance`
