# 0831T001 Plan Amendment Round 26

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

日期：
- 2026-09-01

是否进行QA验收：
- 否

目的：
- 关闭 implementation probe 发现的 witness ref ABSENT 命令与冻结状态
  推导不相容问题，确保 recovery witness 可以在正式 Apple Git 2.39.5
  runtime 上从缺席状态执行 CAS。

发现：
1. Revision 25 注册
   `git ... show-ref --hash --verify <witness-ref>`。
2. 在 Apple Git 2.39.5 上，缺失 ref 的真实结果为 exit `128`、空 stdout、
   非空 fatal stderr。
3. 冻结 `ref_state_derivation` 只把 exit `1`、空 stdout/stderr 解释为
   `ABSENT`，因此原命令会把全新 controller repo 永久分类为
   `UNREADABLE`，witness CAS 正常入口不可达。

amendment：
1. 仅把 `recovery_start_witness.ref_observe_command` 改为：
   `git ... rev-parse --verify --quiet <full-witness-ref>`。
2. 本机直接 probe 证明：
   - ref absent：exit `1`，stdout/stderr 均为空；
   - ref present：exit `0`，stdout 为唯一 40-hex OID 加 LF，stderr 为空。
3. witness object-type、blob-read、blob-write、CAS create、evidence schema、
   A12、quarantine、phase tables 和所有 aggregate 保持不变。

new authority：
- execution plan SHA256 / blob：
  `bdc934202cd9ee9e1743830121eec80f1cf3ab7e8bb4f3bbc1f8728c3619f7dc`
  / `87ffe9e70d050342b74b258e9ae5578f56319368`
- task SHA256 / blob：
  `19585d2501994535eca3d462b62860be76c54cc1196609ce9fd9a788255b138b`
  / `f9bcdef809a02635cacfe0da2b2d0aed142862f3`
- surface SHA256 / blob：
  `096b70b70ce723f30d2c719309d3431081041a195933c050d78157b6a4f91657`
  / `5db7f47abcb47935b2c28d93035086d21a47ea1b`
- fixture truth：保持不变。

boundary：
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ledger ref、witness ref 与 task tags：
  均未创建。
- 尚未通过的 implementation worktree edits 不属于本计划提交。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: make 0831T001 witness absence observable`
