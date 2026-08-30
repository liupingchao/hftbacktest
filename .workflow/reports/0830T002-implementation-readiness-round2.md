# 0830T002 Independent Implementation Readiness Review Round 2

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `6d1e6dda537245e4beda5b22f97ac6f3466fe3a2`
- remediation commit:
  `a86e2eb1769b1127a5ee3e75f24fc776c1f51b37`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- tag peel:
  `6d1e6dda537245e4beda5b22f97ac6f3466fe3a2`

更新时间：
- 2026-08-30 20:24 CST

审查限制：
- 未打开 29 个正式 cache。
- 未运行 formal attempt。
- 未读取 future outcomes。
- 未修改 frozen idea、plan、task、armed claim、runner、verifier 或 tests。
- 本轮只新增本报告。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/3/0/0**
- 29-cache formal execution lock **不得释放**。

Round 1 的 historical attempt、claim transition、identity binding 和真实
DETECTOR child-side FD enforcement 已得到实质修复。但是 production verifier
仍同时存在合法 formal output 必然被拒绝的 false-reject，以及不符合 frozen
identity contract 的 output 可被接受的 false-PASS。当前 88 个 focused cases
没有执行到这些生产路径，因此不能授权 single-use formal attempt。

## Findings

### P1-1 V07 会拒绝 frozen contract 明确要求的 `direction=-1`

Frozen plan 要求 `epoch_variant_counters.csv` 为每个
`(research_date,capture_id,epoch_id,variant,direction)` 输出一行并包含零计数，
方向域包含双方向；排序也按 direction numeric
（plan lines 897-911）。Runner 将 `DIRECTIONS` 冻结为 `(-1, 1)`，并为每个
epoch/variant 枚举两个方向写 counter rows
（runner lines 161、954-957、1077-1109）。

Verifier 的 `typed_csv_rows()` 把除少数特殊字段外的所有整数交给默认
`minimum=0` 的 `parse_csv_int()`；`direction` 没有负数例外
（verifier lines 755-759、769-815）。直接 synthetic 调用
`parse_csv_int("-1", "direction")` 返回
`VerificationError: csv_int_range:direction`。

因此任何包含正式负方向 counter row 的合法 17-output root 都会在 V07
失败。由于 counter ledger 必须枚举负方向，这不是边缘样本，而是 formal
execution 的确定性 terminal-verifier blocker。

### P1-2 Candidate/cluster identity 仍未被 independent verifier 复算

Frozen plan 规定 `candidate_id` 必须是以下 preimage 的 canonical SHA256：

```text
(capture_id,variant,epoch_id,segment_id,direction,
 candidate_ts_ns,candidate_event_seq)
```

并规定 cluster key 为 `(capture_id,epoch_id)`
（plan lines 918-937、fixed epoch contract lines 1456-1458）。Runner 确实按
该 preimage 生成 candidate ID，并将 cluster 写为
`capture_id:epoch_id`（runner lines 834-850、1026-1065）。

Verifier 只检查 `candidate_id` 是任意 64 位 lowercase hex，
`dependence_cluster_id` 是任意非空 ASCII；trigger semantics 只检查排序和
enum/pair domains，没有从 row fields 复算 identity
（verifier lines 802-815、885-913）。本轮构造的 synthetic trigger row 使用
全零 candidate SHA 和 `WRONG_CLUSTER`，`typed_csv_rows()` 仍返回 ACCEPTED；
同一 row 的正确 candidate SHA 应为
`7641cd153f2d827dc780b2ba2c7330fd1c76af2a41c658479c4fe4e085c55aa0`。

这允许 A/B/P 三路一致地篡改 candidate/cluster identity 后仍通过 typed
closure 和 exact-comparison 检查，直接影响 cluster support 与浓度语义。
Round 1 P1-3 所要求的完整 17-output semantic closure 因此尚未关闭。

### P1-3 Frozen V00-V12 hostile minimum 仍未由 production checks 覆盖

Frozen plan 要求对 all 17 typed schemas、candidate/slice identity、claim/lock、
RawOpenEvent/IPC/FD、poison、gate precedence、terminal closure 和 V00-V12
逐项进行 hostile mutation（plan lines 1860-1926）。

新增的参数化 `test_verifier_v00_v12_mutation_short_circuit` 将整个
`CHECK_FUNCTIONS` tuple 替换成 no-op/failure lambdas；它只证明 13-row
短路状态编码正确，没有调用 production `check_v01` 至 `check_v12`
（tests lines 1353-1388）。当前测试文件对 production checks 的唯一直接
调用是 V00 tag-alias case；`rg` 未发现 V01-V12 的直接调用
（tests lines 1298-1311）。

因此 `88 passed` 没有检测到 P1-1 的合法负方向 false-reject，也没有检测到
P1-2 的 candidate/cluster false-PASS。Round 1 P1-5 的核心要求仍未闭合。

## Round 1 Remediation Disposition

1. **P1-1 historical blob/tree + exact rename delta：闭合。**
   Runner 现在扫描 refs/reflogs、unreachable/dangling commit/tree/blob，并
   对 consumption parent、same claim blob、唯一 delete/add delta 和 commit
   message 做 exact verification；synthetic hostile cases覆盖 dangling
   blob/tree 与 extra delta。
2. **P1-2 verifier tag/blob/claim/lock binding：闭合。**
   V00-V03 已绑定 frozen tag names、implementation tree identities、
   claimed bytes、formal argv/roots、SHA identities、lock schema及 remote
   transition。
3. **P1-3 17 outputs/typed/path closure：部分闭合，仍失败。**
   WorkRow、RawOpenEvent、FeatureCall、poison、gate/classification 的大部分
   exact checks 已补齐，但 P1-1/P1-2 所述方向域和 identity 语义仍不正确。
4. **P1-4 DETECTOR child-side FD enforcement：闭合。**
   实际 detector worker 执行 child-side FD enumeration/closure/domain
   enforcement，计数来自子进程结果；regular FD 与 control-pipe hostile cases
   已覆盖。
5. **P1-5 hostile tests：未闭合。**
   测试数量增加到 88，但生产 V01-V12 mutation coverage 仍缺失。

## Passed Checks

1. Frozen idea SHA256：
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
2. Frozen plan SHA256：
   `8171a7bed7b216527fed468dbcff8f31ab1f48ec871bb2eee9b0ae7ad123f79c`。
3. HEAD 与 annotated implementation tag 均 peel 到
   `6d1e6dda537245e4beda5b22f97ac6f3466fe3a2`。
4. Idea、plan、task、runner、verifier、tests、armed claim 的 working
   bytes、Git blob 和 implementation-tag tree blob 全部一致。
5. Armed claim 中 task/runner/verifier/tests SHA、formal argv、repo/source/
   attempt roots 与 controller identities 精确匹配当前冻结状态。
6. Git config 为 `core.fsync=all`、`core.fsyncMethod=fsync`、
   `core.logAllRefUpdates=always`；origin fetch/push URL 精确匹配。
7. Controller ref 查询 exit 0，stdout/stderr 均为空，remote ref 当前不存在。
8. Baseline authority verifier PASS；successor frozen docs、8 callable
   bindings 与只读 historical-attempt scan PASS。
9. Focused suite：`88 passed in 9.09s`。
10. Inherited suite：`53 passed, 1 skipped in 0.44s`。
11. Ruff、Ruff format check、py_compile、runner/verifier `--help` 全部通过。
12. 本轮结束前 worktree 除本报告外无新增或修改。

## Required Remediation

1. 为 `direction` 注册 exact signed domain `{-1,1}`，并增加包含负方向的
   production V07 valid-root test。
2. 在 verifier 中从 trigger row exact fields 复算 `candidate_id`，并验证
   dependence cluster 的 `(capture_id,epoch_id)` identity/partition；补充
   candidate、cluster 与 retained-candidate cross-file mutations。
3. 用可复用的 synthetic terminal fixture 直接运行 production V00-V12，
   按 frozen hostile minimum 持久化每个 check 的正例和 fail-closed mutation；
   不能以 monkeypatched `CHECK_FUNCTIONS` short-circuit test 代替。
4. 修复后重新冻结 verifier/tests SHA/blob、task、armed claim 与
   implementation tag，再执行下一轮 independent readiness review。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/3/0/0**
- Formal attempt: **NOT AUTHORIZED**
- Data execution lock: **CLOSED**
