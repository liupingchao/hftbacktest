# 0831T001 Plan Amendment Round 23

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 关闭 plan review round 22 的四个 P1，同时保留已接受的 atomic
  no-replace publication 与 no-unlink boundary。

amendment：
1. recovery evidence partition：
   - ordered states：
     `NOT_STARTED -> WITNESS_ONLY -> START_ONLY -> COMPLETE -> ABNORMAL`；
   - 只依赖 witness/start/observation states 与 bytes；
   - `ABNORMAL` 是前四者补集，workflow outcome 不参与推导。
2. QA file-kind evidence：
   - common state：
     `ABSENT | VALID_REGULAR | INVALID_REGULAR | SYMLINK | DIRECTORY |
     FIFO | OTHER_NONREGULAR | OBSERVATION_ERROR`；
   - SHA256 只属于 regular states；
   - observation error 单独记录 stage/errno。
3. repeatable quarantine inventory：
   - path：
     `<target>.abandoned.<temporary_sha256>.<ordinal>`；
   - 每个 SHA 的 ordinal 必须连续 `0..count-1`；
   - repeated rebuild crash 追加 next ordinal，不覆盖或删除已有 evidence；
   - inventory JSON 冻结 raw basename hex、parsed rows、kind/hash/error。
4. post-receipt artifact restart：
   - terminal receipt 或 report 已存在且 terminal commit 未形成时，
     `ARTIFACT_BLOCKER_POST_RECEIPT_NO_TERMINAL_COMMIT` 覆盖所有
     A01-A12 first match。

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
  `7b4616b853969a396e6a48d4960f1b9d8fb65bcd897d4636b6bcd6904b0830e4`
  / `cdc0f9b4f96da0d35abe05b23ddd167029bdcdf5`
- task SHA256 / blob：
  `960427e330d4ef7ca26a3fddc6a18cc7f4503b4f2e6c8ceda370f7e02d415a15`
  / `f6308bdd77541ce833b23f84ed8ad621fe9c07e6`
- surface SHA256 / blob：
  `e7696f64a040b7ead3a32ff21fce0a52003863361264f5d76fb4259de0154dda`
  / `d4fd7f8d729a98eafe88dc04fb06cbbcf67543b1`
- fixture truth：保持不变。

boundary：
- date：2026-08-31。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ledger ref、witness ref 与 task tags：
  均未创建。
- Revision 22 已独立记录为 FAIL，只保留历史。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: partition 0831T001 recovery and quarantine state`
