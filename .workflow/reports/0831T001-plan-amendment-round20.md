# 0831T001 Plan Amendment Round 20

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 关闭 plan review round 19 的四个 boundary-completeness P1，不改变已
  接受的 witness architecture、phase derivation 或 aggregate domains。

amendment：
1. 新增 artifact rule：
   `A12_RECOVERY_WITNESS_MISMATCH`。
   witness ref/local target/temporary matrix 的所有异常都进入
   `ARTIFACT_STATE_CORRUPTION`、classification `NONE`，并复用既有 blocker
   restart 与 QA 语义。
2. recovery-start temporary crash rows：
   - exact regular temporary：no-follow reopen、fstat/read/fsync，同字节
     hard-link、parent fsync、unlink、再次 fsync；
   - mismatched/truncated regular：unlink 后重建；
   - nonregular/symlink：A12，禁止删除或修复；
   - generic temporary reconciliation 明确排除 recovery-start temporary，
     由 witness matrix 独占处理。
3. `variant_ordinal_schema.maximum` 从 `21` 修正为 `22`；当前 prose 统一
   为 23 variants。
4. QA evidence 新增：
   - `recovery_witness_ref`
   - `recovery_witness_blob_oid`
   recovery path 必须验证 ref 指向 exact recovery-start blob；normal path
   两字段与两个 recovery SHA 字段均为 `NONE`。

unchanged accepted authority：
- action phases：`16`。
- Git preimage variants：`23`。
- mutation rows / aggregate：
  `736` /
  `4f28bc0e5f99e795600064219ceea2c5c192b2a5f8d80ae92b3822392f4504cd`。
- PRE_BLOCKER rows / legal / invalid / aggregate：
  `124416 / 18 / 124398` /
  `8f1b2d435c2291aca90320827479dcb3ab847a33fb7a1873794aa32b11e8ecba`。
- POST_CONTROLLER authority：不变。

new authority：
- execution plan SHA256 / blob：
  `bf3162e9d4e9c9b5f0f9dfb8c63bc1bce84141a9b639ff18a7cc47a9545b2403`
  / `e8048ac0e376ee19811496f2616065fa88d6e328`
- task SHA256 / blob：
  `f8ea271a1947d340a3cbf37d3a6274c72d9ad489c02b3dd871d2b04b883dd98f`
  / `3f1f1a6bdcb3dca33e4a558b50818388ef654b1b`
- surface SHA256 / blob：
  `fe28103f5aaa265eb59598075e609bcf0953dc50ae93a810d1909fb5c3ee6aa6`
  / `693d401ea40caf6b14d9df3c76cf68e9fea32716`
- fixture truth：保持不变。

boundary：
- date：2026-08-31。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ledger ref、witness ref 与 task tags：
  均未创建。
- Revision 19 已独立记录为 FAIL，只保留历史。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: close 0831T001 witness workflow boundaries`
