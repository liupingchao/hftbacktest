# 0830T002 Hostile Plan Review Round 15

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `956b2b4fb835ba1706c7b042f9bbacc3791ff785`
- idea SHA256
  `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`
- Revision 15 plan SHA256
  `f5f6c5fbbd98a55fc804a7067d62a393eb1e890d8844d5706dbec60a6a8c2711`

审查边界：
- 未打开或读取 29 个正式 cache。
- 未读取 future outcomes。
- 未运行 formal attempt 或 A0。
- 未修改 idea、plan、task、claim、runner、verifier 或 tests。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate HEAD、branch、idea SHA 和 Revision 15 plan SHA 精确匹配。
- 审查开始时 working tree 干净。
- `956b2b4f` 相对 Round 14 review 节点只修改 execution plan。
- 当前 task、claim、runner、verifier 仍绑定旧 plan/implementation identity；
  Revision 15 尚未成为可消费 execution identity。

## Severity Summary

- P0: 0
- P1: 2
- P2: 0
- P3: 0

## Round 14 P1 Disposition

Revision 15 已正确完成以下收口：

1. non-canonical serialization 不再属于 A-1-1 scientific evidence；
2. producer-canonical A/P path/hash inequality 才能形成 comparison evidence；
3. hostile minimum 已把 canonical value mutation 与 non-canonical byte
   serialization 分开；
4. 两个 intended low-level codes 已明确写入计划；
5. CSV/JSON producer-canonical byte check、唯一 SHA replacement 和 fixed-size
   约束未回退。

但是新增语义没有按 comparison phase 分层，并且两个 low-level code 与
terminal verifier 的冻结 code schema 冲突。Round 14 P1 因此仍未完整闭合。

## Findings

### P1-1 `run_manifest.json` 不可能是 pre-classification failure

冻结 build sequence 的顺序是：

```text
step 6  compare RAW_11
step 7  derive gate/classification payload
step 10 write run_manifest.json
step 11 compare FINAL_17
```

对应 plan lines 699-725。也就是说，`run_manifest.json` 的 canonicality
只能在 gate/classification 已形成之后检查。

Revision 15 却把 slice 与 manifest 两种 canonicality defect 一并定义为：

- pre-classification execution-integrity failure；
- 在 `ComparisonRow`、`raw_a_p_difference_count`、gate payload 或 scientific
  classification 形成前终止；
- no scientific classification synthesized
  （plan lines 816-830）。

这对 RAW_11 中的 `slice_invariance.csv` 可以成立，但对仅存在于 FINAL_17
的 `run_manifest.json` 不可能成立。全文其他规范也明确：

- A-1-1 条件只消费 A/P `RAW_11` difference
  （plan lines 1816-1825）；
- `run_manifest.json` 属于 terminal-verifier-only domain，scientific
  classification unchanged（plan lines 1752-1794）。

同一 producer-canonical manifest inequality 也存在相同问题。Plan lines
868-873 把 producer-canonical A/P differences 概括为 A-1-1，但 FINAL_17
manifest comparison 发生在 A-1-1 classification 之后，无法在不重写已形成
结果的前提下归入 A-1-1。

Required remediation：

按 phase 冻结唯一语义：

1. RAW_11 `slice_invariance.csv` canonicality failure：
   pre-classification terminal execution failure，无科学 classification。
2. FINAL_17 `run_manifest.json` canonicality failure或 producer-canonical
   inequality：
   post-classification terminal package failure，已形成的 scientific
   classification 不变，不属于 A-1-1。
3. A-1-1 明确只拥有 producer-canonical A/P **RAW_11** missing/extra/unequal
   comparison rows。

对应 hostile bullets 必须分别断言 pre-classification runner failure 与
post-classification verifier/package failure，不能继续共享同一 ownership
描述。

### P1-2 两个 low-level code 与 verifier 唯一 failure-code vocabulary 冲突

Plan lines 816-823 把以下字符串声明为 exact runner/verifier failure codes：

```text
poison_slice_comparison_noncanonical
poison_manifest_comparison_noncanonical
```

但 terminal verifier schema 冻结：

- `first_failure_code` 等于首个 FAIL row 的 `check_id`；
- check IDs 只能是 `V00` 至 `V12`；
- comparison closure 的 check ID 是 `V09_COMPARISON_CLOSURE`；
- “check_id itself is the failure code; no alternate code vocabulary exists”
  （plan lines 1447-1501）。

因此 verifier 遇到 non-canonical comparison 时无法同时满足：

```text
first_failure_code = poison_*_comparison_noncanonical
first_failure_code = V09_COMPARISON_CLOSURE
```

当前 verifier 语义也印证该冲突：每个 check 内的任意 exception 都被映射成
当前 `check_id`；V09 内部 detail 不进入 result schema。

Required remediation：

冻结两个 namespace 的唯一关系，建议：

1. runner pre-classification exception code 可以保留
   `poison_slice_comparison_noncanonical`；
2. terminal verifier 对 manifest canonicality 的唯一公开 failure code 保持
   `V09_COMPARISON_CLOSURE`；
3. 若需要保留 verifier low-level detail，必须显式新增并冻结独立
   `detail_code` schema、允许值和 hostile assertions；否则删除
   “exact runner/verifier failure codes”的 verifier 部分。

不能在仍声明“no alternate code vocabulary”的同时要求 verifier 暴露第二套
failure code。

## Passed Contract Checks

除上述两项外，未发现独立缺陷：

1. producer-canonical CSV/JSON raw-byte equality、quoting/CRLF/escaping、
   whitespace/key-order/trailing-newline fail-closed 语义保持完整。
2. slice 只替换每行 `slice_source_sha256`；manifest 只替换唯一 slice
   artifact SHA，且从同 root normalized slice SHA 派生。
3. 合法 64-byte substitution 保持物理尺寸；其他 path/field/value/row/count/
   order/size 不被 normalization 隐藏。
4. A/B comparison 仍是 RAW_11、SEALED_15、FINAL_17 的全路径 exact byte SHA。
5. producer-canonical A/P RAW_11 missing/extra/unequal rows可以由
   `ComparisonRow` 和 `raw_a_p_difference_count` 唯一表达。
6. sequential gate precedence、A-1-0/A-1-1 separation 和 later
   `NOT_EVALUATED` 规则未回退。
7. physical slice/manifest、WorkRow、FeatureCall、work manifest 和 consumer
   identity evidence 保留。
8. post-Build-A no-repair、single-use claim、future-outcome/A0 prohibition
   和 execution lock 未被 Revision 15 放宽。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/2/0/0
```

Revision 15 不可冻结。29-cache formal execution lock 必须保持关闭；不得更新
task/claim/code execution identity，也不得消费 armed claim。按 comparison
phase 拆分 manifest ownership，并唯一化 runner/verifier failure-code
namespace 后，需要新的独立 plan review。
