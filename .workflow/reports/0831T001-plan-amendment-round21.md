# 0831T001 Plan Amendment Round 21

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 关闭 plan review round 20 的三个 witness-boundary P1，不改变已接受的
  23-variant、PRE_BLOCKER 或 POST_CONTROLLER authority。

amendment：
1. A12 mutation boundary：
   - blocker publication 和 deterministic local claim consumption 是 witness
     failure 后仅有的允许写入；
   - controller mutation、recovery-start/observation publication、
     terminalization 和 baseline mutation 明确禁止；
   - blocker target 提交后只执行 workflow blocker restart rows。
2. A12 evidence totality：
   - `artifact_state_corruption.json` 新增
     `recovery_witness_evidence_json`；
   - exact ref/object/blob commands 的 exit code、stdout/stderr SHA256，
     observed OID/object type，以及 no-follow target/temp state/SHA256
     一次冻结；
   - 非 A12 blocker 必须将该字段设为 `NONE`。
3. QA evidence totality：
   - exactly one mode：`NORMAL`、`RECOVERY`、`WITNESS_BLOCKED`；
   - witness-blocked mode 记录 exact blocker SHA256 与诊断证据，recovery
     hashes 为 `NONE`；已有 valid terminal receipt 仅按 immutable evidence
     记录 SHA256，classification 仍为 `NONE`。
4. publication final binding：
   - successful hard-link 后，final no-follow FD 必须验证 exact bytes 且
     `st_dev/st_ino` 与仍打开的 temporary FD 一致；
   - `EEXIST` 重新进入 existing-target row，mismatch/nonregular 时不删除
     或重写。

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
  `e4ac9db758975070f87ea66ef44bbdb5516b0a248a520215c6e8fcb3e277be08`
  / `900aa747b0df59a673502a3afef5d98a29f98efa`
- task SHA256 / blob：
  `be036540b9762742858e1fd5b9ff5afaa5791598cb0fe31b541c283d991e9201`
  / `606326ddbed9f86c3fe2fb66ad924916825a26e0`
- surface SHA256 / blob：
  `2e470e647e85bc60249a6661cadf451c95735fbd655a693ddf5a3aeef84eb52e`
  / `eeb3e7345e9ae42ac169bf96b5ffe9d41dc8d559`
- fixture truth：保持不变。

boundary：
- date：2026-08-31。
- historical cache / future outcome：未访问。
- formal、claim、controller、receipt、ledger ref、witness ref 与 task tags：
  均未创建。
- Revision 20 已独立记录为 FAIL，只保留历史。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: totalize 0831T001 witness blocker evidence`
