# 业务线程执行回报

执行线程：
- 总控 auto-loop / 业务执行线程 / live-awsserver1

任务ID：
- 0720T031

状态：
- 待验收

更新时间：
- 2026-07-20

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0720T031.md`
- `.workflow/reports/0720T031-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/principal_alignment_estimator_observe_only_0720T031/`（本地证据，按仓库规则忽略）

action：
- 从 exact source `83238c104163813ed5c89a8f71ed8be561d17b1e` 构建隔离远端运行树；local/remote archive SHA-256 均为 `cae96401ea5a6705845b28e57c62daa07e890c9957f381dfc7ab9f3764229e4b`。
- 完成 exact no-start orchestrator preflight 和只读账户/服务 preflight：
  - exact task/source/profile/window/duration/caps/activation 全部匹配；
  - watcher/order/cancel/private/account endpoint 在 no-start preflight 中全部未调用；
  - open orders `0`、BTC position `0.0`、kill-switch clear；
  - `xemm.service` inactive、冲突交易进程 `0`；
  - runtime source files `62`，executor/estimator import 来自 exact isolated root。
- 在 `awsserver1` 启动唯一一个 detached fixed-quote two-sided-manager window：
  - Binance public lead；
  - Hyperliquid BTC post-only `Alo`；
  - `1800s` maximum、`0.005 BTC/order`、`0.01 BTC position`、`1 USDC loss`、`2 submissions`；
  - dynamic spread、fill feedback、inventory skew、multi-level 和 actual quote behavior change 全部关闭。
- Window 从 `2026-07-20T18:16:17Z` 运行到 `2026-07-20T18:17:44Z`，完成两侧真实 endpoint、manager hold、cancel/query 和 artifact seal；没有启动第二个 window。
- 两侧真实结果：
  - buy `0.005 BTC @ 65729.0` 获得 exchange resting oid/cloid；
  - sell `0.005 BTC @ 65745.0` 获得 exchange resting oid/cloid；
  - sell cancel 返回 exact `success`；
  - buy cancel 返回通用 error：`Order was never placed, already canceled, or filled`，不能区分已取消、已成交或其他终态；
  - buy 随后完成五轮 oid/cloid direct query 和一次 historical fallback，仍为 unknown；
  - fills `0`，最终账户 open orders `0`、BTC position `0.0`。
- Manager hold 记录两个 conservative confirmed resting intervals，并从公共事件重建 `6` 条 positive-duration exposure rows。
- Producer/replay 对 buy 首个 event-time partial bucket 持久化一条 `bucket_reference_book_missing` quarantine；独立 Task 12 重建跳过该无可用 reference book 的前缀 bucket，不生成对应 exposure 或 quarantine。
- 回收完整远端 evidence root，验证 terminal SHA-256 `106/106`，并运行 estimator CLI replay 和 exact same-window acceptance。

verify：
- Pre-live 与 post-live account/service proof 均为 pass：
  - account scope token exact；
  - open orders `0`；
  - BTC position `0.0`；
  - kill-switch clear；
  - `xemm.service` 和 T031 transient service 最终 inactive；
  - 无冲突交易进程。
- Orchestrator：
  - child `rc=0` 且已 reap；
  - no termination、no SIGKILL、no abort；
  - independent after-child open orders `0`；
  - runtime source start/postrun verification 均 pass；
  - remote terminal checksum `106/106`；
  - remote/local preflight、post-live proof、manifest 和 verification hashes exact。
- Live decision evidence：
  - candidate evaluations `310`；
  - trigger rows `4`；
  - edge pass/block `1/1`；
  - manager/submitted attempt identities `2/2`；
  - decision validation reasons `[]`；
  - both sides reached real post-only endpoint and resting state。
- Manager public observation：
  - public event count `20`；
  - reconnect/disconnect count `0/0`；
  - pump stop acknowledged；
  - source/websocket closed；
  - dynamic spread activation false；
  - actual quote behavior changed false。
- Estimator evidence：
  - confirmed interval rows `2`，both pass；
  - confirmed exposure rows `6`；
  - producer and CLI replay exposure exact-match；
  - source/replay snapshot SHA-256 both `ff80b097a50ac4e69276836c311792745a1bb60e130f68e455febe4f0cdbf5c6`；
  - buy A/k fit status pass：`A=9936.40915821`、`k=2.50532025`、`3` observations；
  - sell fit unavailable：`insufficient_distance_variation`；
  - dynamic candidate `fallback_fixed`，authoritative half-spread remains `0.5` ticks；
  - persisted/rebuilt exposure both `6`，but quarantine row count `1`，therefore replay exit `1` and `snapshot_match=false` by fail-closed policy。
- Same-window acceptance exit `2`：
  - provenance `112/112`；
  - config `72/72`；
  - decision `43/43`；
  - economics `6/6`；
  - lifecycle/evidence `57 pass / 14 fail`；
  - mechanism/evidence integrity `fail`；
  - final recommendation `principal_task12_same_window_acceptance_blocked`。
- The `14` acceptance failures reduce to two root blockers：
  - buy reference has no authoritative target-bound terminal evidence；generic cancel error plus current account absence remains insufficient；
  - persisted confirmed-resting quarantine is non-empty and differs from independent Task 12 prefix-bucket handling。
- No credential value、raw signature、raw account identifier or unredacted order reference was persisted。
- No second live window、flatten、strategy relaxation or adaptive/multi-level activation occurred。

done：
- One and only one authorized fixed-quote Task 8 observe-only live window is preserved locally。
- Both sides reached real resting state，and `2` confirmed intervals plus `6` exposure rows were collected without changing quote behavior。
- Account terminal safety independently passes：no open orders、no BTC position、no observed loss。
- The first exact execution/evidence stop conditions are preserved without using account-wide absence or producer/replay equality as a waiver。
- No stable A/k、fill-rate、PnL、maker viability、dynamic-spread activation、multi-level or promotion claim is supported。

blockers：
- Buy cancel result is generic and the bounded direct/history chain remained unknown，so one of two submitted references lacks authoritative terminal proof。
- Confirmed-resting exposure quarantine contains `bucket_reference_book_missing` for the leading partial bucket；estimator CLI and same-window acceptance therefore fail closed。
- Task 8 live observe-only evidence gate is not complete；dynamic spread activation and Principal Task 9-12 remain locked pending independent QA and an offline repair。

commit：
- source `83238c104163813ed5c89a8f71ed8be561d17b1e`

提交信息：
- source `Record T030 QA acceptance`
