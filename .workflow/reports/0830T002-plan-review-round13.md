# 0830T002 Hostile Plan Review Round 13

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `889dcecb1a47bbeeb698ff267edb7dcfc84d71e7`
- idea SHA256
  `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`
- Revision 13 plan SHA256
  `eb61c4b80ad31801387caaff0550ae22eac89c7bea3567c1180edc8d5e5a4cab`

审查边界：
- 未打开或读取 29 个正式 cache。
- 未读取 future outcomes。
- 未运行 formal attempt 或 A0。
- 未修改 idea、plan、task、claim、runner、verifier 或 tests。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate HEAD、branch、idea SHA 和 Revision 13 plan SHA 精确匹配。
- 审查开始时 working tree 干净。
- `889dcecb` 相对上一 readiness 节点只修改 execution plan。
- 当前 task、claim、runner、verifier 仍绑定 Revision 12 plan SHA；因此
  Revision 13 尚未成为可消费 execution identity。

## Severity Summary

- P0: 0
- P1: 1
- P2: 0
- P3: 0

## Round 6 P1 Disposition

Revision 13 已经正确注册以下此前缺失的规范元素：

- A/B 三个 domain 全路径 exact file-byte SHA；
- A/P 仅两个 path 使用 poison-normalized semantic SHA；
- slice 与 manifest 的 canonical JSON hash preimage；
- manifest slice artifact SHA 从同 root normalized slice SHA 派生；
- A/B mismatch 归 A-1-0、A/P mismatch 归 A-1-1；
- A/B/P 物理 slice、manifest、WorkRow、FeatureCall 和 consumer-output
  evidence 保留；
- producer/verifier 使用同一 comparison projection。

但“唯一允许 path/field”仍未闭合。当前 preimage 会额外吞掉 normalized
path 的物理 serialization 与 size 差异，和计划自己的排他声明及 hostile
minimum 冲突。因此 readiness Round 6 的 P1 只算部分关闭。

## Findings

### P1-1 Semantic preimage 归一化了注册字段之外的 byte/size 差异

计划声称：

- slice preimage 保留全部 row/order/value，仅替换
  `slice_source_sha256`（plan lines 771-781）；
- manifest preimage 保留全部 key/value/list order/artifact row，仅替换
  slice artifact `sha256`（plan lines 783-793）；
- 不归一化任何其他 path、field、value、row、count、order、size 或 manifest
  identity，任何 non-registered mutation fail closed
  （plan lines 795-799）；
- A/P 两个注册字段以外的 byte mutation 必须形成 A-1-1 negative evidence
  （plan lines 1938-1947）。

但是注册算法先解析 CSV/JSON，再对解析对象做 canonical JSON：

1. `slice_invariance.csv` 的 quoting、escape 形式及其他等价 CSV
   serialization 不进入 hash preimage；
2. `run_manifest.json` 的 whitespace、indent、object-key serialization
   order 和 trailing newline 不进入 hash preimage；
3. 这些变化可改变物理 file SHA 和 `size_bytes`，却仍得到相同 normalized
   `ComparisonRow` SHA。

现有 runner/verifier 精确实现了这个更宽的语义：

- runner 解析 CSV row objects 后归零 source SHA，再 `canonical_sha(rows)`
  （runner lines 1854-1864）；
- runner 解析 manifest object 后替换 artifact SHA，再
  `canonical_sha(payload)`（runner lines 1865-1887）；
- verifier 镜像同一逻辑（verifier lines 1526-1560）；
- typed CSV 校验只要求 LF、无 CR、parsed header/schema/value 合法，不要求
  producer-canonical CSV bytes（verifier lines 771-821）。

本轮仅用临时 synthetic evidence 做了两个不接触正式 cache 的反例：

```text
slice:
  A = normal CSV serialization
  P = QUOTE_ALL serialization
  parsed rows equal except registered slice_source_sha256
  physical SHA unequal, size 916 != 1060
  typed/schema validation PASS
  normalized difference_count = 0

manifest:
  A = indented/sorted JSON
  P = compact JSON
  semantic object equal except registered slice artifact SHA
  physical SHA unequal, size 201 != 154
  normalized difference_count = 0
```

这会让未注册的物理 byte/size mutation 在 A-1-1 gate 中被当作零差异，而
不是计划 lines 1938-1947 要求的 negative evidence。物理 SHA 虽然仍可在
WorkRow、manifest inventory 和 root closure 中保留，但这些证据只证明两份
物理文件是什么，不会把该额外差异重新计入
`raw_a_p_difference_count`。因此它们不能修复 gate 语义。

Required remediation：

1. 对两个 normalized path 注册并强制 producer-canonical physical
   serialization；比较前先验证原文件等于其未归一化对象的 canonical
   serializer bytes，然后只替换注册字段并以同一 serializer 重建 preimage；
   或
2. 明确承认整个 CSV/JSON serialization 都属于允许归一化的 domain，删除
   “only fields / no other size / outside-fields byte mutation”声明，并重新定义
   相应 hostile expectation。

按当前 precision-first、fail-closed 目标，建议采用方案 1，并增加至少两个
production-path hostile mutations：

- schema/value 不变但把 P slice 改为 `QUOTE_ALL`；
- JSON object/value 不变但改变 P manifest whitespace/key serialization。

两者必须经真实 comparison/verifier 路径产生 A-1-1 negative evidence或
更早 terminal fail，且后续 scientific gates 保持 `NOT_EVALUATED`。

## Passed Contract Checks

除 P1-1 外，未发现独立缺陷：

1. comparison domain 唯一：A/B exact，A/P 仅注册两个 semantic paths。
2. normalized slice hash 与 manifest slice artifact hash 的同-root 派生关系
   明确，未形成跨 root 或循环依赖。
3. missing/extra path、malformed header、missing/duplicate manifest slice row
   都有 fail-closed 语义。
4. A/B mismatch 只属于 A-1-0；A/P mismatch 只属于 A-1-1，classification
   precedence 未回退或交叉所有权。
5. physical slice、manifest、same-build WorkRow、SLICE FeatureCall、
   work-manifest、consumer output 和 terminal self-exclusion evidence 均保留。
6. `RAW_11`、`SEALED_15`、`FINAL_17` 的 comparison mode 表述已替换旧的
   全域 exact-byte 文字，未发现另一处 Revision 12 式直接矛盾。
7. post-Build-A no-repair、one-shot claim、future-outcome/A0 prohibition 和
   execution lock 未被 Revision 13 放宽。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/1/0/0
```

Revision 13 不可冻结。29-cache formal execution lock 必须保持关闭；不得更新
task/claim/code execution identity，也不得消费 armed claim。修订 canonical
serialization/hostile contract 后，需要新的独立 plan review。
