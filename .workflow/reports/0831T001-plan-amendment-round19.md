# 0831T001 Plan Amendment Round 19

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 撤回被 round 18 plan review 拒绝的 inode seal，使用已有 controller
  bare repo CAS trust root 建立 recovery-start 外部 witness，并冻结 raw
  durable observation 到唯一 PRE_BLOCKER action phase 的机械映射。

amendment：
1. external recovery witness：
   - ref：
     `refs/tags/skhynix-trade-led-depth-follower-q0-recovery-start-v1`；
   - 位于 frozen controller bare repo；
   - ref 直接指向 exact canonical `recovery_start.json` bytes 的 Git blob；
   - 使用 `update-ref ... <blob_oid> 000...000` CAS-from-ABSENT；
   - witness 验证完成前禁止创建本地 recovery-start temporary/target 或执行
     任何 recovery state mutation。
2. witness/local crash matrix：
   - ref absent 时所有本地 recovery-start path 必须 absent；
   - ref exact + target absent 时从 witness blob 重建；
   - ref exact + target exact regular 时 no-follow descriptor verify；
   - ref absent + local path present、ref invalid、target mismatch/nonregular
     均为 integrity corruption。
3. durable control publication：
   - target/temporary 使用 `O_NOFOLLOW|O_CLOEXEC`；
   - 通过 `fstat` 与 descriptor read/write/fsync 绑定同一 inode；
   - 禁止 pathname `exists/is_file/read_bytes` 作为 authority；
   - 禁止 resolve final component。
4. phase derivation：
   - 新增 16 条有序、互斥、穷尽的 durable-observation row；
   - receipt union/cardinality/schema/tuple/source/tracked-copy 在 phase 前验证；
   - zero/multiple phase matches 直接为 G05；
   - 无 durable receipt 时，只有 canonical controller token 等于 new SHA
     才选择 PUSH_UNRECEIPTED；其他 token/observation failure 选择 PRE_PUSH。
5. 新增
   `NORMAL_CONSUMPTION_UNTRACKED_RECEIPT_COMMITTED`，表示 untracked control
   receipt 已 durable、tracked copy 尚 absent。

machine updates：
- action phases：`16`。
- Git preimage variants：`23`，ordinals `0..22`。
- mutation rows：`736`。
- mutation aggregate：
  `4f28bc0e5f99e795600064219ceea2c5c192b2a5f8d80ae92b3822392f4504cd`。
- PRE_BLOCKER rows：`124416`。
- legal / invalid：`18 / 124398`。
- PRE_BLOCKER aggregate：
  `8f1b2d435c2291aca90320827479dcb3ab847a33fb7a1873794aa32b11e8ecba`。
- POST_CONTROLLER table：保持 `21384 / 11 / 21373` 与原 aggregate 不变。

new authority：
- execution plan SHA256 / blob：
  `7d1d2751ab176a012a70f5504e49c89880dcbdf817b3717fddd0746cf69e035b`
  / `7553c5df647e0f6b4d9ef10c0cee9075d84e2886`
- task SHA256 / blob：
  `2c24aecabd222e923af03d6a4270daa09a6b683ad4239fcb85b2b53779f052e2`
  / `03048ebc45c3d8add1860f97745074b47a13195f`
- surface SHA256 / blob：
  `54801b632b63fbfe7d091101bf4282d3c3276c61f87991fc697fcf19b0be11b4`
  / `612867e8b22e54a4ddd89e1f34d4b9ed5cd6af1b`
- fixture truth：保持不变。

boundary：
- current date：2026-08-31。
- historical cache / future outcome：未访问。
- implementation tag、claim、controller、formal root、receipt、ledger ref、
  recovery witness ref 和 task tags：均未创建。
- Revision 18 已独立记录为 FAIL，只保留历史。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: amend 0831T001 recovery witness and phase derivation`
