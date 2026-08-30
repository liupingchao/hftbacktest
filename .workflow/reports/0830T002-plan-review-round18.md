# 0830T002 Hostile Plan Review Round 18

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `626dac4053769506fef5bcd1f62329482250797d`
- idea SHA256
  `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`
- Revision 18 plan SHA256
  `c690cfb13f11d34bc08a19ecf4e44d987316b54d8099b02efc384de45c33538e`

审查边界：
- 未打开或读取 29 个正式 cache。
- 未读取 future outcomes。
- 未运行 formal attempt 或 A0。
- 未修改 idea、plan、task、claim、runner、verifier 或 tests。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate HEAD、branch、idea SHA 和 Revision 18 plan SHA 精确匹配。
- 审查开始时 working tree 干净。
- `626dac40` 相对 Round 17 review 节点只修改 execution plan。
- 当前 task、claim、runner、verifier 仍绑定旧 plan/implementation identity；
  它们必须在后续 implementation freeze 中更新，不能直接消费现有 claim。

## Severity Summary

- P0: 0
- P1: 0
- P2: 0
- P3: 0

## Round 17 Closure

Round 17 的唯一 finding 已闭合：

1. SEALED_15 不再要求 blanket zero；其 RAW_11 rows 必须逐行继承 frozen RAW
   comparison，四个新增 dynamic paths 必须 equal。
2. SEALED difference path set 和 count 必须与 RAW 完全相同。
3. FINAL_17 必须逐行继承 SEALED，`execution_evidence.json` 必须 equal。
4. `run_manifest.json` 仅作为 inherited RAW difference 的确定性派生：
   RAW 为零时 manifest equal，RAW 非零时 manifest unequal。
5. FINAL difference set 和 count 已冻结为：

```text
final_difference_paths =
  raw_difference_paths
  union {"run_manifest.json"} iff raw_difference_count > 0

final_difference_count =
  raw_difference_count + int(raw_difference_count > 0)
```

6. RAW scientific negative 不再被 SEALED/FINAL 转换成 package failure；
   新增或篡改的非继承差异仍会 fail package closure。

## Hostile Review Result

### RAW exact closure and scientific ownership

- A/B/P 在科学比较前必须具有 exact RAW_11 regular-file path set。
- RAW missing、extra、non-regular path 是 pre-classification execution
  failure，不进入 A-1-0/A-1-1。
- Producer-canonical A/B RAW byte difference 唯一进入 A-1-0。
- Producer-canonical A/P RAW semantic difference 唯一进入 A-1-1。
- Slice canonicality、registered normalization、fixed-width SHA replacement
  和 same-build physical identity binding 没有回退。

### SEALED inherited lineage

- 每个 SEALED RAW row 必须与对应 frozen RAW row exact identical。
- `outcome_access_ledger.json`、`gate_contract.json`、
  `A_minus1_summary.json`、`classification.json` 四个新增路径必须相等。
- SEALED path set/count 公式允许 A-1-0/A-1-1 negative classification
  继续形成完整 package，同时拒绝任何 post-gate 新差异。

### FINAL manifest derivation

- FINAL 继承全部 SEALED rows，`execution_evidence.json` 必须相等。
- 每个 root 的 physical manifest 先独立通过 self-excluding 16-artifact
  closure。
- A/P manifest projection只归一化已注册的 slice artifact SHA；A/B 保持
  exact bytes。
- 任意 inherited RAW difference 必然改变对应 manifest artifact identity；
  零 RAW difference 时 normalized manifests相等。
- Manifest 只贡献一个 path-level difference，因此集合与计数公式完整且不
  重复计算底层 artifact rows。

本轮使用临时 synthetic A/P slice/manifest evidence 验证公式，未接触正式
cache：

```text
source-identity-only:
  RAW/SEALED/FINAL = 0/0/0

one non-source RAW difference:
  RAW/SEALED/FINAL = 1/1/2
  FINAL differences =
    support/slice_invariance.csv
    run_manifest.json
```

### Failure lifecycle and verifier

- Slice non-canonical serialization 在 classification 前终止。
- Manifest canonicality 与 SEALED/FINAL lineage defect 是
  post-classification package failure，不改写科学分类。
- Producer 在 result/receipt 前停止由 `INTERRUPTED_TERMINAL` 表达。
- Completed package comparison defect 的 verifier public first failure 是
  `V09_COMPARISON_CLOSURE`；公开 code namespace 仍只有 V00-V12。
- V09 的职责是独立重算 comparison、row lineage 和 deterministic manifest
  derivation，不是要求所有 nested difference count 为零。

### Full-contract regression

未发现 Revision 18 引入以下方面的回归：

- detector causal order、trigger/veto/persistence/thinning/confirmation；
- fixed epoch、slice/reset identity与 conservation；
- source/poison authority、outcome boundary和 consumed-field isolation；
- exact 17 outputs、schemas、manifest self-exclusion与 sibling closure；
- sequential gates、classification precedence与 `NOT_EVALUATED`；
- primary/sensitivity non-rescue；
- single-use claim、Git transition、post-Build-A no-repair；
- future outcomes、A0和 live/private/order prohibition。

## Findings

无。

## Freeze Decision

```text
PASS
P0/P1/P2/P3 = 0/0/0/0
```

Revision 18 可冻结，plan-review lock 已释放。29-cache formal execution lock
仍不得立即释放；必须先将 exact Revision 18 plan SHA 写入 task，更新并冻结
runner/verifier/tests 与 armed claim identities，重建 implementation tag，
完成 focused/inherited tests 和新的独立 implementation readiness review。
