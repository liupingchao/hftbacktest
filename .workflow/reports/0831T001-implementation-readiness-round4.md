# 0831T001 Implementation Readiness Round 4

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
- `b895c29ae3b997238665a1e01bcd4e74901a2543`

severity counts：
- P0：0
- P1：3
- P2：1
- P3：0

findings：
1. P1：controller blocker temporary 未绑定当前 durable proof stage，也未
   独立证明 observation。`_allowed_controller_expected_sets` 接受当前历史
   中任一可能阶段的集合，而非当前状态的唯一 expected set；stdout/stderr
   只验证为 64-hex。对抗测试确认旧阶段 expected-set 和任意 observation
   hash 可被接受。
2. P1：已提交的 `recovery_start.json` 未验证 immutable identity。文件
   存在时直接返回 JSON，没有验证字段集合、schema、claim/attempt-lock
   hash、initial paths/controller SHA 或重算 `recovery_id`。对抗测试可
   接受 `schema_version=999` 且 recovery ID 错误的记录。
3. P1：已有 A-rule artifact blocker restart 未优先重放完整 G02-G07。
   A-blocker 分支会重算 artifact payload，但 terminal-history 与
   post-receipt 分支未调用 production Git phase verifier；commit identity、
   tracked state 或 tag integrity 漂移后仍可能被分类为合法 artifact
   restart row。
4. P2：observation failure value domain 不精确。Surface 要求
   `MALFORMED_OUTPUT` 的 exit code 必须为 `0`，实现接受任意整数；对抗
   测试中的 exit `7` 被接受。

round3 closure：
- control `.publishing` crash states：部分关闭。通用 hard-link 恢复通过，
  controller temporary 的当前 proof-stage/observation 独立有效性未关闭。
- original recovery identity/order：部分关闭。runtime lock 排除与
  attempt-lock publication 顺序已修复，committed recovery identity 验证
  未关闭。
- post-attempt G01 blocker publication：已关闭。
- 14-row executable durable predicates：未关闭。A-blocker restart 未完整
  重放 G02-G07。
- blocker schema/preserve semantics：部分关闭。baseline hash 与合法
  receipt/report preserve 已改善，proof-stage 与 value-domain 仍有缺口。
- QF13 clean/mutated projection equality：已关闭。runner 与 independent
  verifier 均执行 production anchor/model-input equality check。

verify：
- authority SHA256/blob：全部匹配 Revision 17。
- independent focused pytest：`97 passed in 88.93s`。
- independent targeted restart/QF13 tests：
  `14 passed, 83 deselected`。
- business-thread focused pytest：`97 passed in 91.30s`。
- expanded accepted predecessor regression：
  `410 passed, 1 skipped in 95.43s`。
- primary 与 fresh detached readiness：各 `57` feature calls、`37`
  projection files，逐字节相同。
- readiness tree SHA256：
  `763e9a0d1dbbb49f7baefd5e3414ffd355ddd95bc5ba190c344e74beb50e34f3`。
- ruff、format、py_compile、CLI help、`git diff --check`、`git fsck`：
  通过。
- 主 worktree 与 detached readiness worktree：干净并指向 exact commit。
- claim、controller、formal root、receipt 与 task tags：全部不存在。
- historical cache / future market outcome：未访问。

结论：
- `FAIL`
- Formal、arming、controller creation 和 task tags 继续锁定。
- 按用户指令，本轮审计完成后暂停；不在本轮修复、诊断扩展或重新提交
  implementation candidate。

提交信息：
- `review: audit 0831T001 Q0 implementation round 4`
