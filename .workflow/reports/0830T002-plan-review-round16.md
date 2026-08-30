# 0830T002 Hostile Plan Review Round 16

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `4e5de39b09fe730a8e3b1c9fe15009846fd3381d`
- idea SHA256
  `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`
- Revision 16 plan SHA256
  `32993c016ac79234d2f80840f3c04510ef8d2c448ab61f118512e8bfb3a9df65`

审查边界：
- 未打开或读取 29 个正式 cache。
- 未读取 future outcomes。
- 未运行 formal attempt 或 A0。
- 未修改 idea、plan、task、claim、runner、verifier 或 tests。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate HEAD、branch、idea SHA 和 Revision 16 plan SHA 精确匹配。
- 审查开始时 working tree 干净。
- `4e5de39b` 相对 Round 15 review 节点只修改 execution plan。
- 当前 task、claim、runner、verifier 仍绑定旧 plan/implementation identity；
  Revision 16 尚未成为可消费 execution identity。

## Severity Summary

- P0: 0
- P1: 1
- P2: 0
- P3: 0

## Round 15 Closure

Round 15 的两项 finding 在 Revision 16 主合同中均已闭合：

1. `slice_invariance.csv` canonicality 在 RAW_11 comparison 中检查，失败时
   gate/classification 尚未形成。
2. `run_manifest.json` canonicality 在 FINAL_17 comparison 中检查，失败是
   post-classification package failure，已形成的科学分类不改写。
3. A-1-1 明确只拥有 A/P RAW_11 missing/extra/producer-canonical unequal
   rows。
4. SEALED_15 与 FINAL_17 的 A/B、A/P nonzero comparison 均为
   post-classification package failure。
5. producer 在 `attempt-result.json` 和 terminal receipt 前停止时，由
   precommitted claim 表达 `INTERRUPTED_TERMINAL`。
6. terminal verifier 的公开 failure code 仍只有 V00-V12；完成包的
   comparison closure defect first-fails `V09_COMPARISON_CLOSURE`。
7. runner guard detail 不再被声明为第二套 verifier public code。

但是全文仍有三处旧的、未限定 RAW_11 的 broad ownership 语句，与上述
phase-specific 主合同冲突。

## Findings

### P1-1 三处 broad A/B、A/P ownership 仍覆盖 SEALED_15/FINAL_17

Revision 16 的 phase-specific 规则明确：

- A-1-1 只拥有 A/P RAW_11
  （plan lines 840-845）；
- SEALED_15、FINAL_17 的任意 A/B 或 A/P nonzero comparison 都是
  post-classification package failure，不改写科学分类
  （plan lines 842-845）；
- first-failure ownership 和 sequential gate 也只列 A/B RAW_11 与 A/P
  RAW_11（plan lines 1767-1837）。

但以下旧语句仍未限定 projection：

1. `Canonical A/B differences belong only to A-1-0`
   （plan line 883）。
2. `A/B missing, extra and byte mutations fail A-1-0`
   （plan line 1986）。
3. `A/P missing and extra paths produce A-1-1 negative evidence`
   （plan line 1987）。

这些语句按字面同时覆盖 SEALED_15 与 FINAL_17。例如删除
`FINAL_17/run_manifest.json`：

```text
phase-specific rule:
  post-classification package failure
  scientific classification unchanged

hostile broad rule:
  A/P missing path -> A-1-1 negative classification
```

同理，SEALED_15/FINAL_17 的 A/B byte mutation 可以被解释为 A-1-0，而主
合同要求其只终止 package closure。结果会决定是否重写 scientific
classification，属于正式行为冲突。

Required remediation：

将三处全部限定为 RAW_11：

```text
Canonical A/B RAW_11 differences belong only to A-1-0.
A/B RAW_11 missing, extra and byte mutations fail A-1-0.
A/P RAW_11 missing and extra paths produce A-1-1 negative evidence.
```

并保留当前 lines 2002-2004 对 SEALED_15/FINAL_17 synchronized inequality
的 post-classification package-failure hostile test。这样每个 projection
才只有一个 owner。

## Passed Contract Checks

除 P1-1 外，未发现独立缺陷：

1. RAW slice canonicality failure 的时点早于
   `raw_a_p_difference_count`、gate payload 和 classification。
2. Manifest canonicality failure 的时点晚于 classification，且不改写其
   bytes。
3. Completed mutated package 的 comparison closure 由 verifier
   `V09_COMPARISON_CLOSURE` 公开报告。
4. Producer 中断不生成 synthetic result，由 consumption claim 和缺失
   terminal closure 唯一表达 `INTERRUPTED_TERMINAL`。
5. verifier 的 `first_failure_code`、check rows 和 `V00`-`V12` 单一
   vocabulary 未引入 detail-code 冲突。
6. producer-canonical comparison、canonical raw-byte enforcement、唯一
   SHA replacement 和 fixed-size semantics 未回退。
7. CSV quoting/CRLF/escaping 与 JSON whitespace/key-order/trailing-newline
   hostile cases的 phase ownership明确。
8. SEALED_15/FINAL_17 nonzero comparison 必须终止 package closure，不能靠
   同步修改 comparison evidence 被接受。
9. physical evidence、manifest self-exclusion、no-repair、single-use claim、
   future-outcome/A0 prohibition 和 execution lock 均未放宽。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/1/0/0
```

Revision 16 不可冻结。29-cache formal execution lock 必须保持关闭；不得更新
task/claim/code execution identity，也不得消费 armed claim。将三处 broad
ownership 语句限定到 RAW_11 后，需要新的独立 plan review。
