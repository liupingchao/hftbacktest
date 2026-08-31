# 0831T001 Plan Amendment Round 24

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 关闭 plan review round 23 的四个 P1，不扩展 implementation 或 formal
  scope。

amendment：
1. mandatory witness observation：
   - 删除不可机械区分的 `NOT_OBSERVED`；
   - 每次 terminal 或 blocker QA handoff 都执行 exact witness
     ref/type/blob observation；
   - canonical command tuple 写入
     `recovery_witness_observation_json`；
   - normal no-recovery 唯一推导为 witness/start/observation 全部
     `ABSENT`。
2. canonical quarantine row：
   - exact keys：
     `name_hex, observed_sha256, observation_error, ordinal, path_state,
     sha256_suffix`；
   - exact path-state domain：
     `NONCANONICAL_NAME | VALID_REGULAR | INVALID_REGULAR | SYMLINK |
     DIRECTORY | FIFO | OTHER_NONREGULAR | OBSERVATION_ERROR`；
   - 每个 state 的 SHA/suffix/ordinal/error 交叉绑定已冻结；
   - `VALID_INVENTORY` 只允许 hash/suffix 相等且 per-SHA ordinal 连续的
     `VALID_REGULAR` rows。
3. ordinal path exclusions：
   - generic reconciliation ownership 排除
     `recovery_start.json.abandoned.<sha256>.<ordinal>`；
   - `initial_committed_paths_json` 排除每个
     `.abandoned.<sha256>.<ordinal>` non-authority evidence path。
4. restart disjointness：
   - `BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE` 的 restart row 和 phase
     predicate 都要求 terminal receipt 与 business report 同时 absent；
   - 任一存在且 A01-A12 命中时，只由
     `ARTIFACT_BLOCKER_POST_RECEIPT_NO_TERMINAL_COMMIT` 处理。

unchanged accepted authority：
- Darwin `renamex_np(RENAME_EXCL=0x00000004)` commit primitive。
- pathname unlink：禁止。
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
  `92431c6c20f8bc744ddfa8bc61881dc8d722810aed40a7e5b30410dae742d3cf`
  / `3cba11839d712c3428e300c08121b0130b20b88e`
- task SHA256 / blob：
  `2aca596a0f5168b07cfbed5678fd77140b5bf37666b39d2cd1d1a177eff2c5eb`
  / `7523390e636864eefeb084512b4c7f4bb97120e9`
- surface SHA256 / blob：
  `c4639ed4b01d1d310224d88734f96bf01f3110437f28743e9040d43510d790c0`
  / `f257bddb44fad8c6efeeb9023b24be16028cd01d`
- fixture truth：保持不变。

boundary：
- date：2026-08-31。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ledger ref、witness ref 与 task tags：
  均未创建。
- Revision 23 已独立记录为 FAIL，只保留历史。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: close 0831T001 witness and restart determinism`
