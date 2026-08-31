# 0831T001 Plan Review Round 23

执行线程：
- 独立 plan review 线程

任务ID：
- 0831T001

状态：
- 未通过

reviewed commit：
- `94576ad1f97ef2ccac0efd7a2d06e95f61265a63`

severity counts：
- P0：0
- P1：4
- P2：0
- P3：0

findings：
1. P1：`NOT_OBSERVED` 不是 exact witness observation function 可机械推导
   的状态。QA 可以把同一个 absent witness 解释为 `NOT_OBSERVED` 或
   `ABSENT`，导致 `NOT_STARTED` 与 `ABNORMAL` 的选择依赖实现策略。
2. P1：quarantine inventory row 虽包含 `path_state`、
   `observed_sha256` 和 `observation_error`，但没有冻结 path-state 值域
   及字段间约束；不同实现可以对同一路径生成不同 canonical row。
3. P1：正式 quarantine 路径已扩展为
   `.abandoned.<sha256>.<ordinal>`，但 recovery ownership 和
   `initial_committed_paths_json` exclusion 仍使用旧的
   `.abandoned.<sha256>` 形式。
4. P1：A10 的 `BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE` 只要求 terminal
   receipt absent；当 business report 已存在且 receipt absent 时，它与
   `ARTIFACT_BLOCKER_POST_RECEIPT_NO_TERMINAL_COMMIT` 同时匹配。

accepted checks：
- exact commit 工作树干净。
- Darwin `renamex_np(..., RENAME_EXCL=0x00000004)` 与 final FD
  bytes/inode binding：接受。
- pathname unlink 禁止：接受。
- direct Git blob recovery witness CAS-from-ABSENT：接受。
- recovery partition 公式在给定输入状态时完整且互斥：接受。
- QA common file-state domain 覆盖 regular valid/invalid、
  nonregular 和 observation error：接受。
- content-addressed quarantine inventory 与 per-SHA contiguous ordinal
  方向：接受。
- artifact post-receipt row 扩展至 A01-A12：方向接受，但 A10 overlap
  仍须关闭。
- ordinals 为唯一 `0..22`，共 23 variants。
- 736 mutation rows aggregate：
  `4f28bc0e5f99e795600064219ceea2c5c192b2a5f8d80ae92b3822392f4504cd`。
- 124416 PRE_BLOCKER rows、18 legal、124398 invalid，aggregate：
  `8f1b2d435c2291aca90320827479dcb3ab847a33fb7a1873794aa32b11e8ecba`。
- POST_CONTROLLER 21384 / 11 / 21373，aggregate：
  `d9e4682bf43d4516b276eb46760fd020196503af5dfd2edc6e8b90b4336d4356`。
- surface/plan/task/truth SHA256 与 Git blob：匹配。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ref 与 tag：未创建。

结论：
- `FAIL`
- Revision 23 不得解锁 implementation。
- 下一版必须固定 witness 的强制观测语义、冻结 quarantine row
  canonical schema、更新所有 ordinal path exclusion，并消除 A10 restart
  overlap。

提交信息：
- `review: reject 0831T001 plan amendment round 23`
