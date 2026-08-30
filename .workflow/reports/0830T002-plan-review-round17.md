# 0830T002 Hostile Plan Review Round 17

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `d4a8fd7af1f0c5fbf425efdc241a049dd73743d2`
- idea SHA256
  `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`
- Revision 17 plan SHA256
  `47b3ecc6a654e2a2453bf9d6d1bf1471e4496f998ac15fecd5cda69a18d27718`

审查边界：
- 未打开或读取 29 个正式 cache。
- 未读取 future outcomes。
- 未运行 formal attempt 或 A0。
- 未修改 idea、plan、task、claim、runner、verifier 或 tests。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate HEAD、branch、idea SHA 和 Revision 17 plan SHA 精确匹配。
- 审查开始时 working tree 干净。
- `d4a8fd7a` 相对 Round 16 review 节点只修改 execution plan。
- 当前 task、claim、runner、verifier 仍绑定旧 plan/implementation identity；
  Revision 17 尚未成为可消费 execution identity。

## Severity Summary

- P0: 0
- P1: 1
- P2: 0
- P3: 0

## Round 16 Closure

Round 16 的唯一 finding 已闭合：

1. Canonical A/B ownership 已明确限定为 A/B RAW_11。
2. A/B missing、extra、byte hostile mutations 已限定为 RAW_11。
3. A/P missing、extra hostile mutations已限定为 RAW_11。
4. A-1-0、A-1-1、first-failure ownership 和 sequential gates 都只消费
   RAW_11 comparison。
5. SEALED_15、FINAL_17 继续唯一归属 post-classification package closure，
   不改写已形成的科学分类。

最终 hostile review 发现一个新的 nested-projection closure 缺口：后两层
projection 包含 RAW_11，因此不能在保留合法 RAW negative evidence 的同时
要求其总 `difference_count` 为零。

## Findings

### P1-1 SEALED_15/FINAL_17 zero rule 使 comparison-based negative package 不可完成

Plan 的 projection 是严格嵌套的：

```text
SEALED_15 = RAW_11 + 4 dynamic files
FINAL_17 = SEALED_15 + execution_evidence.json + run_manifest.json
```

对应 plan lines 727-750。

Revision 17 同时要求：

1. A/B RAW_11 mismatch 可形成 A-1-0；
2. A/P RAW_11 mismatch 可形成 A-1-1，并保留为 negative evidence，且不阻止
   final package creation（plan lines 883-889）；
3. SEALED_15 和 FINAL_17 的所有 A/B、A/P `difference_count` 必须为零，
   否则终止 package closure（plan lines 840-845）。

这三条不能同时成立。任何合法 RAW mismatch 都仍然是 SEALED_15 的同一路径
mismatch，因此：

```text
raw_difference_count > 0
=> sealed_difference_count >= raw_difference_count > 0
```

FINAL_17 继续继承该 mismatch；其 self-excluding manifest 还会忠实记录不同
artifact SHA，因此 manifest comparison 可能再增加一个 derived difference。

本轮使用临时 synthetic files 验证嵌套关系，未接触正式 cache：

```text
raw_difference_count    = 1
sealed_difference_count = 1
final_difference_count  = 2
```

因此当前合同会把本应记录为
`Aminus1_authority_or_source_failed` 或
`Aminus1_outcome_boundary_violated` 的合法科学负面结果，在 step 8/11 再次
转换为 package failure。这样 comparison-based negative classification
无法形成 accepted terminal package，与 one-shot “记录矛盾结果、不修复”
原则冲突。

Required remediation：

不要要求 nested projection 的总 difference count 为零。应冻结 row-level
lineage，例如：

1. SEALED_15 comparison 的 RAW_11 rows 必须与已冻结 RAW comparison rows
   exact identical；
2. SEALED_15 新增的四个 dynamic paths 必须 A/B/P equal；
3. FINAL_17 必须 exact 继承 SEALED comparison rows；
4. `execution_evidence.json` 等新增非-manifest paths 必须 equal；
5. `run_manifest.json` comparison 必须由各 root 的真实 artifact rows和注册
   normalization独立重算，允许其反映已经被 RAW gate 接受的 inherited
   difference；
6. V09 验证 comparison recomputation、row lineage 和“无新增未授权
   differences”，而不是 blanket `difference_count == 0`。

同时 hostile minimum 应增加两类成对测试：

- A/B RAW_11 或 A/P RAW_11 非零时，正确形成对应科学负面分类并完成 package；
- RAW comparison 固定后，SEALED/FINAL 新增或同步篡改任何非继承 difference
  时，post-classification package closure fail。

## Passed Contract Checks

除 P1-1 外，未发现独立缺陷：

1. Round 16 三处 broad ownership 均已精确限定 RAW_11。
2. Slice canonicality 是 pre-classification execution failure。
3. Manifest canonicality 是 post-classification package failure，不改写
   scientific classification。
4. Producer interruption由 `INTERRUPTED_TERMINAL` 唯一表达；不生成
   synthetic result。
5. Terminal verifier public code 仍只有 V00-V12，comparison closure
   first-fails V09。
6. A-1-1 只拥有 producer-canonical A/P RAW_11 missing/extra/unequal rows。
7. Canonical raw-byte enforcement、唯一 64-byte SHA replacement、physical
   size、manifest derivation与证据保留未回退。
8. Gate precedence、NOT_EVALUATED、primary/sensitivity non-rescue、17-output
   schemas、one-shot claim、no-repair 与 post-Build-A lock 未发现新冲突。
9. Future outcomes、A0、live/private/order execution仍被禁止。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/1/0/0
```

Revision 17 不可冻结。29-cache formal execution lock 必须保持关闭；不得更新
task/claim/code execution identity，也不得消费 armed claim。将 SEALED_15 /
FINAL_17 closure 从 blanket zero 改为 exact inherited-row lineage 后，需要
新的独立 plan review。
