# 0831T001 Implementation Readiness Round 6

执行线程：
- 独立 implementation readiness 验收线程

任务ID：
- 0831T001

状态：
- 未通过

日期：
- 2026-08-31

是否进行QA验收：
- 否

reviewed commit：
- `62b0438c9b2a27bd776cb254546419df93ca53b3`

severity counts：
- P0：0
- P1：3
- P2：0
- P3：0

findings：
1. P1：PRE_BLOCKER 顺序错误，controller proof-stage 可被无效 tracked
   receipt 提前。`_recovery_controller_proof_stage()` 仅凭 tracked
   consumption receipt 是普通文件就进入
   `CONSUMPTION_RECEIPT_COMMITTED`，未验证 receipt 内容；普通 recovery
   路径又在 G02-G07 之前计算 controller/A-rule 分类。对抗 probe 证明
   `not-json-and-not-a-receipt` 可推进 proof-stage，且 A01 与 Git drift
   同时存在时只执行 A01-A11，不执行 Git verifier。冻结 contract 要求
   PRE_BLOCKER 首先按 G01-G07 顺序评价。
2. P1：已提交 `recovery_start.json` 的 immutable identity 仍可重新自洽
   伪造。当前验证器接受合法值域内的 `initial_controller_sha`、
   `crash_boundary` 和当前 path set 子集，只要攻击者同时重算
   `recovery_id`。因此不能证明 original crash boundary、initial
   controller SHA 与 initial committed-control path set 从首次提交后未变。
3. P1：通用 durable control publication 接受 symlink。
   `publish_control_no_replace()` 使用会跟随 symlink 的
   `exists()/is_file()/read_bytes()`，部分调用还提前 `.resolve()`。对抗
   probe 证明 committed target symlink 到外部同字节文件会被接受，
   `recovery_start.json` symlink 也可通过验证，违反 no-replace regular
   target 的冻结 publication 语义。

closure review：
- Round 4 controller temporary：fresh exact observation binding 已实现，
  但 proof-stage derivation 受 finding 1 影响，未完整关闭。
- Round 4 recovery identity：未关闭，见 finding 2。
- Round 4 existing A-blocker restart：已有 blocker 路径先执行 G02-G07；
  普通 PRE_BLOCKER 路径仍有 finding 1。
- Round 4 `MALFORMED_OUTPUT` exit code：已关闭。
- Round 5 ASCII path ordering：已关闭。

verify：
- exact detached commit：
  `62b0438c9b2a27bd776cb254546419df93ca53b3`。
- authority SHA256/blob：全部匹配 Revision 17。
- focused pytest：`107 passed`。
- expanded accepted predecessor regression：`410 passed`。
- primary、fresh detached 与 independent fresh readiness 均完成 `57`
  feature calls、`37` projection files。
- projection 逐字节相同；独立 ASCII path-order tree SHA256：
  `05db53ca596a99d845bd7b4e5a6f500c3cc6a5355958eb5ebaa9fae726cb57b7`。
- worktree clean；formal root、claims、receipts、controller repo 与三个
  task tags 全部不存在。
- historical cache / future market outcome：未访问。

结论：
- `FAIL`
- Formal、arming、controller creation 和 task tags 继续锁定。
- 下一候选需增加三类对抗测试：合法字段改写并重算 recovery ID、
  control target symlink、A/G 同时失效并伴随无效 tracked receipt。

提交信息：
- `review: audit 0831T001 Q0 implementation round 6`
