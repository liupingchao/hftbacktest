# 0830T002 Independent Implementation Readiness Review Round 8

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `d86aa51f230e511d9c4728889000d6c0c518ed78`
- Round 7 remediation commit:
  `ed8414191780513ac1062719b7d59dde00823d11`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- annotated tag object:
  `742bc8e2f8429365bc47f75ddf481bcce8e57a30`
- tag peel:
  `d86aa51f230e511d9c4728889000d6c0c518ed78`
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
- **P0/P1/P2/P3 = 0/2/0/0**
- 29-cache formal execution lock **不得释放**。

Round 7 的 expected artifact-path 修复、production-generated A/B 与 A/P
negative package、canonical serializer 和 V09 主路径均可执行。但是
projection root 本身仍未使用 non-following symlink rejection，且 A/B root
symlink 可以通过完整 terminal verifier。Revision 18 新增的 synchronized
lineage 和全部 manifest serialization V09 hostile evidence 也没有完整持久化。

## Findings

### P1-1 Projection root 自身 symlink 仍穿透 producer，A/B 可通过完整 verifier

Frozen plan 要求 A/B/P 的 exact `RAW_11` missing、extra 或 non-regular path
在 scientific comparison 和 classification 前 fail closed
（plan lines 805-846、2019-2024）。用户本轮要求 expected path/component
采用 non-following regular-file 与 symlink rejection。

Runner remediation 的 `require_exact_projection()`：

- 枚举 `entries = list(root.rglob("*"))`；
- 拒绝 `entries` 中的 symlink；
- 对 expected artifact 使用 `lstat()` 和 `S_ISREG`
  （runner lines 1912-1928）。

这正确关闭了 expected artifact 和枚举到的中间 component，但 `root` 本身不在
`rglob("*")` 的结果中，也没有 `root.is_symlink()` 或
`S_ISDIR(root.lstat().st_mode)` 检查。若 `canonical_a`、`canonical_b` 或
`poison_p` 根目录本身被替换为指向完整树的 symlink，producer 会沿该 root
遍历并接受 exact path set。

本轮不接触正式 cache 的 production helper 复现：

```text
root_is_symlink = true
require_exact_projection(root_symlink, RAW_11, ...)
observed = ACCEPTED_ROOT_SYMLINK
```

Verifier 的 `validate_final_root()` 同样只检查 `root.rglob("*")` 中的条目
（verifier lines 1899-1912），没有检查 root 自身。完整 synthetic terminal
package 将 Build A root 移到外部目录并在原路径建立 directory symlink 后：

```text
root_is_symlink = true
exit_code = 0
status = PASS
first_failure = null
V00-V12 = all PASS
```

Build P root 会因 poison attestation 的 resolved path 偶然在 V06 失败，但
A/B 没有对应绑定，因此该结果不能构成通用 root closure。

Required remediation：

1. producer 在枚举任何 artifact 前，使用 `lstat()` 要求 root 自身是
   non-symlink directory，并对 root 到每个 expected artifact 的全部组件做
   non-following 检查；
2. verifier 对 A/B/P root 自身和全部 descendants 做同样独立检查，不依赖
   poison attestation 的 Build P 特例；
3. 增加 A、B、P root-symlink production helper 与完整
   `verify_terminal()` hostile tests；A/B/P 必须在注册的 phase fail closed。

### P1-2 Frozen V09/lineage hostile minimum 仍未完整持久化

Round 7 已新增并通过：

- missing/extra/expected-path symlink/FIFO/directory helper tests
  （tests lines 923-955）；
- production `seal_roots()` 生成的 A/B 与 A/P RAW negative package，并通过
  完整 verifier（tests lines 2648-2684）；
- `QUOTE_ALL`、CRLF 和 alternate quoted header；
- compact、key-order、indent 和 trailing-newline manifest guard；
- compact manifest 的完整 package V09 first-fail。

仍缺少 frozen plan lines 2035-2045 要求的完整持久化 evidence：

1. key-order、indent 和 trailing-newline 只直接调用 runner
   `comparison()` 并断言内部 `AuditError`
   （tests lines 2895-2927）；只有 compact JSON 进入完整
   `verify_terminal()` 并断言 `V09_COMPARISON_CLOSURE`
   （tests lines 2930-2941）。
2. “同步更新 comparison evidence 后新增 SEALED/FINAL inequality 仍拒绝”只用
   手工构造的 `ComparisonRow` 字典调用
   `require_projection_lineage()`（tests lines 2968-2992）。它没有构造完整
   A/B 或 A/P package、同步更新 `execution_evidence.json` 和 manifests，再走
   production V09。
3. root 自身 symlink 未出现在 missing/extra/non-regular 参数集，因而 P1-1
   未被 145 个 focused cases 捕获。

本轮临时完整包 probes 证明实现的 V09 核心逻辑可以拒绝这些 mutation：

```text
compact          -> V09_COMPARISON_CLOSURE
key_order        -> V09_COMPARISON_CLOSURE
indent           -> V09_COMPARISON_CLOSURE
trailing_newline -> V09_COMPARISON_CLOSURE
synchronized new SEALED difference with updated evidence
                 -> V09_COMPARISON_CLOSURE
```

因此本 finding 是 frozen hostile evidence closure 缺口，而不是上述 V09
计算逻辑本身失败。但 one-shot formal contract 明确把这些项目列为 minimum；
测试未持久化前不能释放 execution lock。

## Round 7 Disposition

1. **Expected artifact missing/extra/symlink/FIFO/directory：部分闭合。**
   - 五类 mutation 均被 production helper 拒绝；
   - `seal_roots()` 在 `raw_ab/raw_ap`、gate 和 classification 前调用 closure；
   - projection root 自身 symlink 仍未闭合。
2. **Canonical CSV/JSON serializer：实现闭合。**
   - `QUOTE_ALL`、CRLF、alternate quoting、compact、key-order、indent 和
     trailing newline 均 fail closed；
   - 完整包手工复验确认四种 manifest serialization mutation 都 first-fail
     V09。
3. **Production A/B 与 A/P negative packages：已闭合。**
   - 两路均由 production `seal_roots()` 生成；
   - RAW difference 1、SEALED difference 1、FINAL difference 2；
   - FINAL 只增加 inherited RAW path 与 derived `run_manifest.json`；
   - 两路完整 terminal verifier 均 PASS。
4. **Lineage 新增差异拒绝：实现闭合，测试合同未闭合。**
   - runner/verifier helper 均拒绝新增 SEALED difference；
   - 完整包临时 probe first-fail V09；
   - 对应 production V09 hostile test 尚未持久化。

## Passed Checks

1. Worktree 在审查开始、测试结束和写报告前均 clean；branch/HEAD exact。
2. Implementation tag 是 annotated tag，并精确 peel 到 reviewed HEAD。
3. Idea SHA256：
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
4. Plan SHA256：
   `c690cfb13f11d34bc08a19ecf4e44d987316b54d8099b02efc384de45c33538e`。
5. Task SHA256/blob：
   `1d578dedb14b5da23f1cf38b54ad86cc995969f8cc1c4315415a56011904585b`
   / `b3553a12f88e64f2513f1fffc98d664640745dd8`。
6. Armed claim SHA256/blob：
   `70e574fd174f796cb8c22e7208e3fdef9890a1cb706565711a9cfe818387e4bd`
   / `3f62b82620c623f2482bdd6800b0f8ce1a2e4b02`。
7. Runner SHA256/blob：
   `ede54aedfb64e15bee6eb124aeaa4f2f999d5123ba41876b6b3da098774b8e2f`
   / `a204b47346de6ee216f115e89ec81a4a48892e43`。
8. Verifier SHA256/blob：
   `c70759d07df80bd33dc6765605f3e9c634755fd23f5e4b860bd11be027da184b`
   / `18a49c3921c402d5193f4cd5a632df85c92e9fc1`。
9. Tests SHA256/blob：
   `a9345e0d0f4e7ef1488114907abfc3d752f48c7840be46ff3d23747b9f467953`
   / `1f8f224e2bc8bdfcd4ccba0bee429cfa041c1743`。
10. Claim 的 idea/plan/task/runner/verifier/tests SHA、exact formal argv、
    repo/source/attempt roots、implementation tag 和 controller identities
    均独立复算一致；production `verify_armed_claim()` PASS。
11. Remediation commit `ed841419` 只修改 runner/verifier/tests；
    `96e65d18` 只更新 task identities；`d86aa51f` 只重建 armed claim。
12. Git config：
    `core.fsync=all`、`core.fsyncMethod=fsync`、
    `core.logAllRefUpdates=always`；origin fetch/push URL exact。
13. Controller ref `refs/heads/codex/0830T002-controller-ledger`
    `git ls-remote --heads` exit 0、stdout empty。
14. Claimed path、terminal receipt、formal attempt root、terminal verifier
    result、consumption tag 和 terminal tag 均不存在。
15. Production `verify_no_historical_attempt()` PASS；未发现 reachable、
    reflog 或 unreachable consumption/terminal attempt。
16. Frozen document、baseline authority 和 successor callable binding
    preflight PASS。
17. Focused suite：
    `145 passed in 40.97s`。
18. Inherited suite：
    `66 passed, 1 skipped in 0.54s`。
19. Ruff、Ruff format check、py_compile、runner/verifier `--help` 和
    `git diff --check` 全部通过。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/2/0/0**
- Formal attempt: **NOT AUTHORIZED**
- Armed claim: **UNCONSUMED**
- Data execution lock: **CLOSED**
