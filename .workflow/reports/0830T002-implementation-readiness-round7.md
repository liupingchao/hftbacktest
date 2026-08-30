# 0830T002 Independent Implementation Readiness Review Round 7

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `0cab333cc7c927b2773efb8596cce219693ea83e`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- annotated tag object:
  `cd1812768d7e9eb3417b175a5fd9e1711c7e9f9e`
- tag peel:
  `0cab333cc7c927b2773efb8596cce219693ea83e`
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

Revision 18 的核心 comparison 设计已经进入 runner/verifier：canonical
CSV/JSON physical serialization、RAW/SEALED/FINAL path domains、A/B 与 A/P
分离、manifest-derived FINAL difference 以及 V09 lineage 重算均存在且基线
测试通过。但是 RAW pre-classification closure 仍会接受 expected path 上的
symlink；同时 Revision 18 新增的 hostile-test minimum 只实现了一部分。

## Findings

### P1-1 RAW exact-path closure 接受 symlink，错误推迟到分类后的 V07

Frozen plan 要求 A/B/P 在 scientific comparison 前包含 exact `RAW_11`，
missing、extra 或 non-regular RAW path 必须形成 pre-classification
execution-integrity failure，而不是形成科学分类
（plan lines 805-846、2019-2024）。

Runner 的 `require_exact_projection()` 只收集
`root.rglob("*")` 中 `Path.is_file()` 为真的路径
（runner lines 1912-1916）。`Path.is_file()` 跟随 symlink，因此指向普通文件
的 expected-path symlink 会被视为合法 RAW artifact。三个 RAW roots 在
gate/classification 前只调用该 helper（runner lines 3143-3163），没有
`is_symlink()` 或 `lstat/S_ISREG` 检查。

Verifier 最终会在 `validate_final_root()` 拒绝任何 symlink
（verifier lines 1892-1905），但该检查属于 V07，发生在 producer 已生成
gate、classification 和四个 SEALED dynamic artifacts 以后。这违反了
Revision 18 对 non-regular RAW path 的 phase ownership。

本轮不接触正式 cache 的最小复现：

```text
1. 临时创建完整 RAW_11 path set；
2. 将其中一个 expected path 建为指向普通文件的 symlink；
3. 调用 production require_exact_projection(root, RAW_11, ...);
4. observed: ACCEPTED_SYMLINK。
```

Required remediation：

1. producer 的 RAW/SEALED/FINAL projection closure 对每个 expected artifact
   使用不跟随 symlink 的 regular-file 判定，并拒绝任意 symlink component；
2. RAW non-regular failure 必须发生在 `raw_ab/raw_ap`、gate 和 classification
   形成之前；
3. 增加 production helper/`seal_roots()` hostile test，证明 expected-path
   symlink、FIFO、directory、missing 和 extra RAW file 均在分类前 fail closed。

### P1-2 Revision 18 frozen hostile-test minimum 未完整持久化

Plan lines 2019-2045 新冻结的 minimum 要求：

- RAW missing/extra/non-regular pre-classification failure；
- producer-canonical A/B RAW mutation 形成 A-1-0 negative classification，
  且完成 inherited-lineage package；
- producer-canonical A/P RAW mutation形成 A-1-1 negative classification，
  且完成 inherited-lineage package；
- CSV `QUOTE_ALL`、CRLF、alternate escaping 全部 fail closed；
- manifest compact JSON、alternate key order、whitespace、trailing-newline
  mutation 形成 post-classification package failure，并在完整包 first-fail
  `V09_COMPARISON_CLOSURE`；
- synchronized SEALED/FINAL mutation 不能靠同步更新 evidence 绕过；
- nonzero RAW difference 必须在真实 package 中精确继承到 SEALED，并只增加
  derived manifest FINAL difference。

当前 focused tests：

- 没有直接调用 `require_exact_projection()`，也没有 RAW symlink/FIFO/
  directory/extra-path phase test；
- `test_poison_comparison_rejects_noncanonical_slice_csv()` 只覆盖
  `QUOTE_ALL`（tests lines 2737-2761），未覆盖 CRLF/alternate escaping；
- `test_poison_comparison_rejects_noncanonical_manifest_json()` 只覆盖 compact
  JSON（tests lines 2764-2784），未覆盖 alternate key order、whitespace 或
  trailing-newline，并未通过完整 `verify_terminal()` 断言 V09；
- negative lineage 只用手工 `ComparisonRow` 字典调用
  `require_projection_lineage()`（tests lines 2787-2835），没有由 production
  `seal_roots()` 生成 A/B、A/P negative classification 和完整 17-path package；
- 完整包的 V09 mutation 只篡改已持久化
  `execution_evidence.raw_a_b.difference_count`
  （tests lines 2515-2526），没有覆盖 Revision 18 的 canonical serializer、
  synchronized lineage 或 derived-manifest hostile cases。

这不是计数型 test-gap。P1-1 正是该缺口未被 132 个 focused cases 捕获的实际
合同错误。Frozen hostile minimum 完整落地前，formal one-shot attempt 不能
消费 armed claim。

## Passed Checks

1. Worktree 在审查开始、测试结束和写报告前均 clean；HEAD/branch 精确。
2. Implementation tag 是 annotated tag，并精确 peel 到 reviewed HEAD。
3. Idea SHA256：
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
4. Plan SHA256：
   `c690cfb13f11d34bc08a19ecf4e44d987316b54d8099b02efc384de45c33538e`。
5. Task SHA256/blob：
   `f742d3e63aa16ccf8c1326fe374495fc555b28261629fa6313023afd3d54ffc5`
   / `27c78954913c72c81b171c7c5b81d3a129931b49`。
6. Armed claim SHA256/blob：
   `41b692b5641669a2916a28fb5749283a8c5028897168aed320f302da64284597`
   / `80c3b436aef5194c7c2e3b8d243dbafd368e548f`。
7. Runner SHA256/blob：
   `16001a30da8e5a3c92fc343ebfc6b3ef8ebdf81becac099f66e6a2194e22b6e9`
   / `fa437067c079f97acc15f74e990b36aa91ce7dbf`。
8. Verifier SHA256/blob：
   `d1336ad5668e18fbd4aef8e960aeec51643bb710500aba5a361455c83491accd`
   / `d7cfe7f421536f9138e7942acbe20862935aa324`。
9. Tests SHA256/blob：
   `229a4976638551d7afefda1a8bff1f2838e238503583a05e291093b86265236e`
   / `d42df08cdce543a12107e53cb7ef82a7c53cb61c`。
10. Claim 的 idea/plan/task/runner/verifier/tests SHA、exact formal argv、
    repo/source/attempt roots、implementation tag 和 controller identities
    均独立复算一致；production `verify_armed_claim()` PASS。
11. Revision 18 code commit `0a003d07` 只修改 runner/verifier/tests；
    后续 `ebed0472` 只冻结 task identities，`0cab333c` 只重建 armed claim。
12. Producer/verifier 对 canonical CSV/JSON normalized hash preimage、
    RAW/SEALED/FINAL comparison 和 lineage 公式保持同义；合法 physical
    A/P slice/manifest difference 的 synthetic terminal baseline通过 V00-V12。
13. Runner 在三个阶段检查 exact expected file path set；verifier V09 独立
    重算 RAW/SEALED/FINAL comparisons，并检查 inherited rows、四个新增 SEALED
    rows、execution evidence 和 derived manifest difference。
14. Git config：
    `core.fsync=all`、`core.fsyncMethod=fsync`、
    `core.logAllRefUpdates=always`；origin fetch/push URL exact。
15. Controller ref `refs/heads/codex/0830T002-controller-ledger`
    `git ls-remote --heads` exit 0、stdout empty。
16. Claimed path、terminal receipt、formal attempt root、terminal verifier
    result、consumption tag 和 terminal tag 均不存在。
17. Production `verify_no_historical_attempt()` PASS；未发现 reachable、
    reflog 或 unreachable consumption/terminal attempt。
18. Baseline/frozen document and successor callable authority preflight PASS。
19. Focused suite：
    `132 passed in 33.58s`。
20. Inherited suite：
    `66 passed, 1 skipped in 0.52s`。
21. Ruff、Ruff format check、py_compile、runner/verifier `--help` 和
    `git diff --check` 全部通过。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/2/0/0**
- Formal attempt: **NOT AUTHORIZED**
- Armed claim: **UNCONSUMED**
- Data execution lock: **CLOSED**
