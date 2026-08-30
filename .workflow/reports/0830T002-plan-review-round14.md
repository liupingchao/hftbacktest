# 0830T002 Hostile Plan Review Round 14

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `c0192fd53653fdc8464c18043aa426c3e1c52549`
- idea SHA256
  `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`
- Revision 14 plan SHA256
  `eb5c05b5285e08995768a1fc9c1422ba24dddcc6210b05aed46496f30a71cc90`

审查边界：
- 未打开或读取 29 个正式 cache。
- 未读取 future outcomes。
- 未运行 formal attempt 或 A0。
- 未修改 idea、plan、task、claim、runner、verifier 或 tests。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate HEAD、branch、idea SHA 和 Revision 14 plan SHA 精确匹配。
- 审查开始时 working tree 干净。
- `c0192fd5` 相对 Round 13 review 节点只修改 execution plan。
- 当前 task、claim、runner、verifier 仍绑定旧 plan/implementation identity；
  Revision 14 尚未成为可消费 execution identity。

## Severity Summary

- P0: 0
- P1: 1
- P2: 0
- P3: 0

## Round 13 P1 Disposition

Revision 14 已闭合 Round 13 的核心 byte-projection 缺口：

1. `slice_invariance.csv` 先以注册的 producer CSV serializer 重建未修改
   bytes，并要求与物理文件逐字节相等，再替换
   `slice_source_sha256`。
2. `run_manifest.json` 先以注册的 pretty/sorted/trailing-newline JSON
   serializer 重建未修改 bytes，并要求逐字节相等，再替换唯一 slice
   artifact SHA。
3. CSV `QUOTE_ALL`、CRLF、非 canonical escaping，以及 JSON compact、
   whitespace、key order 和 trailing-newline 变化均被注册为
   pre-normalization failures。
4. 两个允许替换的 SHA 均为 64-byte ASCII；合法输入替换为另一 64-byte
   值后，normalized serialization 与原物理文件尺寸相同。
5. 其他 path、field、value、row、count、order、size 和 manifest identity
   不进入 normalization exception。

本轮临时 synthetic serializer probe 未接触正式 cache，结果为：

```text
canonical CSV A/P accepted = true/true
QUOTE_ALL accepted = false
CRLF accepted = false
normalized CSV A/P equal = true
normalized CSV size preserved = true/true

canonical JSON A/P accepted = true/true
compact/key-order/trailing-newline accepted = false/false/false
normalized manifest A/P equal = true
normalized manifest size preserved = true/true
```

但是 failure-channel ownership 仍有一处会改变正式科学结果的全文矛盾，
因此 Round 13 P1 尚不能判定完全关闭。

## Findings

### P1-1 Non-canonical A/P bytes 的 terminal failure 与 A-1-1 ownership 冲突

Revision 14 对两个 normalized path 明确规定：

- canonical reserialization 不等于原文件时“fail before comparison”
  （plan lines 771-803）；
- non-canonical CSV/JSON serialization “fail before normalization”
  （plan lines 805-814）。

按现有 `Comparison` / `ComparisonRow` schema，没有 canonicality failure
字段或 fallback row。直接 fail before comparison 意味着：

```text
没有 raw_a_p Comparison
没有 raw_a_p_difference_count
不能生成 A-1-1 gate/classification
formal sequence 在科学结果形成前终止
```

但全文同时要求：

- A/P differences 只属于 A-1-1，并保留为 negative outcome-boundary
  evidence，且不阻止 final package creation（plan lines 852-855）；
- A/P RAW_11 mismatch 的 first-failure owner 是 A-1-1
  （plan lines 1734-1747）；
- A/P normalized fields 之外的 byte mutations 必须产生 A-1-1 negative
  evidence（plan lines 1953-1956）。

同一 `QUOTE_ALL`、CRLF、JSON whitespace 或 trailing-newline mutation
既是“normalized fields 之外的 byte mutation”，又被新增条款要求在
comparison 前失败（plan lines 1963-1967）。因此实现和 hostile test 可以
合法地产生两种互斥结果：

1. 抛出 execution failure，不产生科学 classification；
2. 构造 unequal `ComparisonRow`，产生
   `Aminus1_outcome_boundary_violated` 并继续 final package。

这不是措辞层面的轻微问题。它决定是否存在正式科学负面结果、是否创建
FINAL_17 package，以及 one-shot attempt 的终态，因此必须唯一化。

Required remediation：

选择并冻结一种语义：

1. **Terminal fail 路径**：明确 canonical serialization defect 是
   pre-classification execution-integrity failure；删除或限定 lines
   1953-1956 的 A-1-1 要求，使其只覆盖 producer-canonical、可形成
   `ComparisonRow` 的 non-registered value/path differences，并注册 exact
   terminal failure code/state。
2. **A-1-1 路径**：禁止抛出 comparison 前异常；注册 canonicality mismatch
   如何生成 deterministic unequal `ComparisonRow` 和
   `raw_a_p_difference_count`，随后按 sequential gates 形成 A-1-1 negative
   package。

当前 schema 更自然地支持方案 1，但计划必须明确选择；不能把两种行为留给
runner/verifier 或测试实现决定。

## Passed Contract Checks

除 P1-1 外，未发现独立缺陷：

1. A/B 仍是三个 projection 的全路径 exact file-byte comparison。
2. A/P semantic exception 仍只覆盖两个注册 path。
3. CSV 原始 bytes 在归一化前受 exact field order、DictWriter、
   `extrasaction="raise"`、LF、minimal quoting 和 ASCII 约束。
4. JSON 原始 bytes 在归一化前受 indent、sorted keys、ASCII 和唯一 trailing
   newline 约束。
5. slice 仅替换每行 `slice_source_sha256`；manifest 仅替换唯一 slice
   artifact `sha256`，且后者从同 root normalized slice SHA 派生。
6. 合法 64-byte 替换保持尺寸；其他尺寸变化无法通过 canonical raw-byte
   equality 和 normalized hash。
7. malformed header、missing/duplicate manifest slice row、missing/extra
   path 和 non-registered semantic mutation 均 fail closed。
8. physical slice/manifest、same-build WorkRow、SLICE FeatureCall、
   work-manifest、consumer output 和 terminal self-exclusion evidence 保留。
9. 未发现新的 normalization path、跨 root 派生、self-reference 或
   A-1-0/A-1-1 交叉所有权。
10. post-Build-A no-repair、single-use claim、future-outcome/A0 prohibition
    和 execution lock 未被 Revision 14 放宽。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/1/0/0
```

Revision 14 不可冻结。29-cache formal execution lock 必须保持关闭；不得更新
task/claim/code execution identity，也不得消费 armed claim。唯一化
canonicality failure 的 terminal/A-1-1 ownership 后，需要新的独立 plan
review。
