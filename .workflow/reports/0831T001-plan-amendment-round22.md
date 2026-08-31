# 0831T001 Plan Amendment Round 22

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 关闭 plan review round 21 的四个 P1，并移除 control publication 的
  pathname unlink race。

amendment：
1. A12 post-receipt restart：
   - `ARTIFACT_BLOCKER_POST_RECEIPT_NO_TERMINAL_COMMIT` 扩展至
     A09-A12；
   - valid terminal receipt 已存在时保留全部 bytes，禁止 render、
     terminal commit/tag 与 controller push。
2. QA totality：
   - `workflow_outcome_mode`：
     `TERMINAL | ARTIFACT_BLOCKED | CONTROLLER_BLOCKED`；
   - `recovery_evidence_state`：
     `NONE | START_ONLY | COMPLETE | WITNESS_MISMATCH`；
   - terminal result/transition 与 recovery start/observation 各自记录
     `ABSENT | VALID | INVALID`，valid/invalid 均记录 raw SHA256。
3. mechanical witness state：
   - ref/type/blob command exit/stdout/stderr 按固定顺序映射 witness state；
   - target/temp 按 `lstat -> open no-follow -> fstat identity -> read`
     映射 state/hash/error stage+errno；
   - quarantine inventory/state 另有固定目录扫描与 content-addressed
     suffix 验证函数。
4. atomic publication：
   - frozen Darwin primitive：
     `renamex_np(source, target, RENAME_EXCL)`，
     `RENAME_EXCL=0x00000004`；
   - successful rename 后 final FD bytes 与 `st_dev/st_ino` 必须等于
     retained pre-rename temporary FD；
   - 不执行 pathname unlink。
5. content-addressed quarantine：
   - abandoned regular temporary 原子移动到
     `<target>.abandoned.<temporary_sha256>`；
   - reopen 后验证 inode、bytes 与 path suffix，永久保留；
   - noncanonical/multiple/existing quarantine 或 identity mismatch
     fail closed。

unchanged accepted authority：
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
  `f6f196c18b100dd1452f98006fe2c3df72ff88d5f7964b6ced9e29b6d595dfe3`
  / `c69d12ba59d611d324e24679ca19ea68d4fe1b53`
- task SHA256 / blob：
  `61da7f86bb47c7f28e770ef7b252dd1426d4ce9736a68c35a1f96048663e89d5`
  / `bda4ac161c39f7d8ddc042ecf659b61fbbe0a3b1`
- surface SHA256 / blob：
  `8b7160d57739e3bef23d70c91d75c870109b3ed108c883543697fb6d8998b847`
  / `265cb49534754c2a12bb19e5675f19bea87cf896`
- fixture truth：保持不变。

boundary：
- date：2026-08-31。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ledger ref、witness ref 与 task tags：
  均未创建。
- Revision 21 已独立记录为 FAIL，只保留历史。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: close 0831T001 publication and QA totality`
