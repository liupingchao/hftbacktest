# 0831T001 Implementation Readiness Round 3

执行线程：
- 独立 implementation readiness 验收线程

任务ID：
- 0831T001

状态：
- 未通过

是否进行QA验收：
- 否

reviewed commit：
- `3daeb4d880cf392bfeb48903a39569bf00f4dbc9`

severity counts：
- P0：0
- P1：5
- P2：1
- P3：0

findings：
1. P1：control publication 在 crash-before-hard-link 后无法重启。完整
   `<target>.publishing` 存在而 target 不存在时，`O_EXCL` 重建会稳定失败，
   影响 attempt lock、recovery start、blocker 和 process receipts。
2. P1：20-state recovery 的 original crash boundary 与 recovery identity
   不具备 restart invariance。runtime lock 文件被计入 initial committed
   path set；`after_attempt_root_before_lock` 在 attempt-lock 与
   recovery-start 之间再次崩溃会漂移到下一 boundary；A01-A11 artifact
   blocker 也可能在 recovery-start 之后才被识别。
3. P1：G01 已默认开启，但 post-attempt action phase 遇到 controller
   divergence/observation failure 时只抛异常，未立即固化 controller
   blocker receipt；异常 observation 可能在 recovery 前丢失。
4. P1：14-row workflow-blocker restart 仍不是完整 executable state
   machine。artifact blocker 分支未验证 receipt/report、原始
   `first_invalid_rule` 和完整 G02-G07 durable predicates；现有测试主要是
   source/call-graph coverage。
5. P1：blocker receipt schema 与 preserve semantics 未闭合。controller
   expected set 未绑定当前 durable proof stage且允许空集合；ABSENT
   divergence 与冻结 40-hex schema 冲突；artifact blocker 会把存在的
   terminal receipt 一律标成 INVALID，并漏掉 present baseline hashes。
6. P2：QF13 已真实执行 production `CausalView.read`，但 formal runner
   和 independent verifier 尚未重放并记录 mutated/unmutated
   anchor/model-input bytes 相同的冻结 evidence。

round2 closure：
- G01 + 15 action phases：未关闭；phase 可达，post-attempt blocker
  publication 未闭合。
- 20 crash boundaries/recovery：未关闭；classifier 存在，restart
  identity 不稳定。
- 14-row blocker restart：未关闭；row 名称已接入，durable predicates
  未完整执行。
- QF12/QF13 production hostile boundary：核心执行已关闭；QF13 保留
  一个 P2 evidence gap。

verify：
- focused pytest：`83 passed in 86.57s`。
- 扩大后的 accepted regressions：`389 passed, 1 skipped in 94.51s`。
- ruff check / format check：通过。
- py_compile：4 个文件通过。
- runner/verifier `--help`：通过。
- `git diff --check`、`git fsck`：通过。
- worktree 保持干净；未创建 tag、claim、controller、formal root 或
  receipt，未访问 historical cache 或 future outcome。

结论：
- `FAIL`
- formal 必须继续锁定。

提交信息：
- `review: audit 0831T001 Q0 implementation round 3`
