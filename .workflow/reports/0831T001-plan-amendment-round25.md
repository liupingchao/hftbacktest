# 0831T001 Plan Amendment Round 25

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 关闭 plan review round 24 唯一 P1，恢复 A10 的 pre-artifact phase
  reachability，同时保留 Revision 24 已接受的 restart-row disjointness。

amendment：
1. phase resolver：
   - `BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE` phase predicate 只使用 durable
     Git/controller/consumption-receipt state；
   - terminal receipt/business report presence 明确不属于 phase input；
   - report-present、controller `ABSENT` 的 A10 状态重新获得唯一
     pre-terminal phase，不再 zero-match G05。
2. artifact/restart split：
   - ordered A01-A12 在 phase derivation 后执行；
   - pre-terminal blocker restart row 仍要求 terminal receipt 和 business
     report 同时 absent；
   - 任一存在且 A01-A12 命中时，只匹配
     `ARTIFACT_BLOCKER_POST_RECEIPT_NO_TERMINAL_COMMIT`。

unchanged accepted authority：
- mandatory witness observation 与 canonical command tuple。
- canonical quarantine six-key row grammar。
- exact `.abandoned.<sha256>.<ordinal>` ownership/exclusion。
- Darwin `renamex_np(RENAME_EXCL=0x00000004)` 与 no-unlink。
- action phases：`16`。
- Git preimage variants：`23`。
- mutation rows / aggregate：
  `736` /
  `4f28bc0e5f99e795600064219ceea2c5c192b2a5f8d80ae92b3822392f4504cd`。
- PRE_BLOCKER rows / legal / invalid / aggregate：
  `124416 / 18 / 124398` /
  `8f1b2d435c2291aca90320827479dcb3ab847a33fb7a1873794aa32b11e8ecba`。
- POST_CONTROLLER rows / legal / invalid / aggregate：
  `21384 / 11 / 21373` /
  `d9e4682bf43d4516b276eb46760fd020196503af5dfd2edc6e8b90b4336d4356`。

new authority：
- execution plan SHA256 / blob：
  `6b1e8040b21d86d5bc90ae17e81dbe9510527ac9fc5483a12707290f90255e0d`
  / `092cd17e4a7f824bdf2416dfbe9b62ee1846755f`
- task SHA256 / blob：
  `dba38d071f261d344d9d59c79e1cf5f8572b55362bb4e0ba5febbd3c86aef99d`
  / `37bd3dbe69db0d3af8bc2bee5a6858690ea82d61`
- surface SHA256 / blob：
  `da72d818a88f244581dde17866a5e42b8464df2265f1c10702f44d453a552fd1`
  / `a59e67dcca5cebf183e7dbe949d3ad5b532d24d1`
- fixture truth：保持不变。

boundary：
- date：2026-08-31。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ledger ref、witness ref 与 task tags：
  均未创建。
- Revision 24 已独立记录为 FAIL，只保留历史。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: restore 0831T001 artifact phase reachability`
