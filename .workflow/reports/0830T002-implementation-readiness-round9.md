# 0830T002 Independent Implementation Readiness Review Round 9

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `ac60b83e77a3dcc60d8e8ed092bbc9613489c3c7`
- Round 8 remediation commit:
  `bf76c090bb816ecf962ca136a19da7ec1e5b35eb`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- annotated tag object:
  `2443d1a7e5c46287ebb007d4c3ec58487f1b216e`
- tag peel:
  `ac60b83e77a3dcc60d8e8ed092bbc9613489c3c7`
- frozen Revision 18 plan SHA256:
  `c690cfb13f11d34bc08a19ecf4e44d987316b54d8099b02efc384de45c33538e`

更新时间：
- 2026-08-30 CST

审查限制：
- 未打开或读取 source-cache-root 下 29 个正式 cache。
- 未运行 formal attempt。
- 未读取 future outcomes。
- 未修改 frozen idea、plan、task、armed claim、runner、verifier 或 tests。
- 本轮只新增本报告。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/1/0/0**
- 29-cache formal execution lock **不得释放**。

Round 8 的 projection-root symlink 缺陷、四种 manifest serialization 的完整
V09 first-fail，以及 root-symlink V07 closure 已关闭。唯一剩余 finding 是
新增的 “synchronized SEALED difference” 测试没有同步更新 comparison
evidence；它测试的是 evidence mismatch，不是 frozen plan 指定的
inherited-lineage rejection。

## Finding

### P1-1 Synchronized SEALED hostile test 未同步 execution evidence

Frozen plan lines 2039-2042 要求：

```text
synchronized producer-canonical SEALED_15 or FINAL_17 A/B or A/P
new inequality outside inherited RAW rows and deterministic manifest row
must fail even when comparison evidence is updated consistently
```

新增测试
`test_complete_package_synchronized_sealed_difference_first_fails_v09`
（tests lines 2975-3000）：

1. 修改 P root 的 `reports/A_minus1_summary.json`；
2. 重算 P root 的 `run_manifest.json`；
3. 直接调用完整 `verify_terminal()` 并断言 first-fail V09。

它没有重算并同步三路 `contracts/execution_evidence.json` 中的
`sealed_a_p` ComparisonRow。独立逐 check 复现显示该测试的实际内部 failure
reason 是：

```text
VerificationError execution_comparison:sealed_a_p
```

因此它只证明 physical SEALED mutation 与旧 evidence 不一致时 V09 fail，并未
证明攻击者同步更新 comparison evidence 后仍因 inherited-row lineage fail。

本轮使用同一完整 synthetic package，额外完成真正同步的临时 probe：

1. 修改 P SEALED dynamic artifact；
2. 重新计算 raw/sealed A/B 与 A/P comparisons；
3. 将新 comparison evidence 同步写入 A/B/P；
4. 重算三路 self-excluding manifests；
5. 执行 production V09。

Observed：

```text
VerificationError sealed_comparison_lineage
first failure = V09_COMPARISON_CLOSURE
```

这证明 runner/verifier 的 lineage 实现是正确的，但 frozen hostile minimum
要求持久化对应 production package test。当前 test 名称和实际攻击强度不一致，
不能用 `151 passed` 代替该证据。

Required remediation：

1. 在测试中重算 physical mutation 后的 RAW/SEALED comparisons；
2. 同步更新三路 `execution_evidence.json`，保持 evidence bytes 相等；
3. 重算三路 manifests，使 V08 self-excluding closure 仍通过；
4. 通过完整 `verify_terminal()` 断言 first-fail
   `V09_COMPARISON_CLOSURE`；
5. 额外直接调用 production `check_v09()` 或 helper，断言内部原因是
   `sealed_comparison_lineage` 或 `final_comparison_lineage`，而不是
   `execution_comparison:*`。

## Round 8 Disposition

1. **Producer projection root closure：已闭合。**
   - `require_exact_projection()` 在枚举前要求 root 存在、不是 symlink 且
     `lstat()` 为 directory（runner lines 1912-1916）；
   - RAW closure 仍位于 comparison、gate 和 classification 前
     （runner lines 3159-3178）；
   - missing、extra、artifact symlink、directory、FIFO 和 root symlink
     tests 均存在。
2. **Verifier projection root closure：已闭合。**
   - `validate_final_root()` 独立要求 root 为 non-symlink lstat directory
     （verifier lines 1900-1917）；
   - Build A 与 B root symlink 临时 probes first-fail
     `V07_FINAL17_SCHEMAS_AND_PATHS`；
   - tracked complete-package root-symlink test first-fails V07
     （tests lines 3003-3012）。
   - Build P root symlink first-fails V06，因为 poison attestation 的 resolved
     output-root identity 更早失配；它仍 fail closed，且符合 sequential
     verifier order。
3. **Manifest serialization V09 closure：已闭合。**
   - compact、key-order、indent、trailing-newline 四种 mutation 均进入完整
     `verify_terminal()`；
   - 每种均 exact first-fail `V09_COMPARISON_CLOSURE`
     （tests lines 2942-2972）。
4. **Production A/B 与 A/P negative packages：保持闭合。**
   - 两路均由 production `seal_roots()` 生成完整 17-path package；
   - RAW/SEALED/FINAL lineage 与 derived manifest difference exact；
   - 完整 terminal verifier PASS。
5. **Synchronized new SEALED/FINAL inequality：实现闭合，测试未闭合。**
   - 真同步临时 probe 被 production lineage check 拒绝；
   - tracked test 未同步 execution evidence。

## Passed Checks

1. Worktree 在审查开始、测试结束和写报告前均 clean；branch/HEAD exact。
2. Implementation tag 是 annotated tag，并精确 peel 到 reviewed HEAD。
3. Idea SHA256：
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
4. Plan SHA256：
   `c690cfb13f11d34bc08a19ecf4e44d987316b54d8099b02efc384de45c33538e`。
5. Task SHA256/blob：
   `c8b94ac242e82b4d598764a26b70c343e81b9d6b68111c7137ea01d08e221784`
   / `f0ea6f2da624f48362b0c44a9432852da309ca19`。
6. Armed claim SHA256/blob：
   `3b2db82d32c4f00d8af5ee5c6156dbf2bb0958ea9d1b74d0f27c14de76927684`
   / `784dbcc2756fdfc6a8d624f6c2fdc8898f65800c`。
7. Runner SHA256/blob：
   `486a66778f12c35e5440a1d39524a073ed9dc3cacfe82a04c411b265f345c9e6`
   / `894b9c1f76a50232ef7a5d9edf7437e853f7d993`。
8. Verifier SHA256/blob：
   `da8c7a4e8f333a595832558b30fbeb5c6b02f25eb007753280bc26b9d0afaf9c`
   / `15211f7361640ff3e0501ba7ea953be572218ef5`。
9. Tests SHA256/blob：
   `4792766815f96f27630ca822463176ea8349651d51a69bd8c7eedc3055891456`
   / `e9f46309ab3a5136103bf5c702ac2ffef0a9e6df`。
10. Claim 的 idea/plan/task/runner/verifier/tests SHA、exact formal argv、
    repo/source/attempt roots、implementation tag 和 controller identities
    均独立复算一致；production `verify_armed_claim()` PASS。
11. Remediation commit `bf76c090` 只修改 runner/verifier/tests；
    `dd012991` 只更新 task identities；`ac60b83e` 只重建 armed claim。
12. Git config：
    `core.fsync=all`、`core.fsyncMethod=fsync`、
    `core.logAllRefUpdates=always`；origin fetch/push URL exact。
13. Controller ref `refs/heads/codex/0830T002-controller-ledger`
    `git ls-remote --heads` exit 0、stdout empty。
14. Claimed path、terminal receipt、formal attempt root、terminal verifier
    result、consumption tag 和 terminal tag 均不存在。
15. Production `verify_no_historical_attempt()` PASS；未发现 reachable、
    reflog 或 unreachable consumption/terminal attempt。
16. Frozen documents、baseline authority 和 successor callable binding
    preflight PASS。
17. Focused suite：
    `151 passed in 47.09s`。
18. Inherited suite：
    `66 passed, 1 skipped in 0.63s`。
19. Ruff、Ruff format check、py_compile、runner/verifier `--help` 和
    `git diff --check` 全部通过。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/1/0/0**
- Formal attempt: **NOT AUTHORIZED**
- Armed claim: **UNCONSUMED**
- Data execution lock: **CLOSED**
