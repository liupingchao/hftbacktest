# 0830T002 Independent Implementation Readiness Review Round 4

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `fbd6084f0e564123f94bc6550354e42b0c119f69`
- remediation implementation commit:
  `76ef890ed4d7d9dbe7684e8633594a255abd8798`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- tag peel:
  `fbd6084f0e564123f94bc6550354e42b0c119f69`

更新时间：
- 2026-08-30 21:01 CST

审查限制：
- 未打开 29 个正式 cache。
- 未运行 formal attempt。
- 未读取 future outcomes。
- 未修改 frozen idea、plan、task、armed claim、runner、verifier 或 tests。
- 本轮只新增本报告。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/2/0/0**
- 29-cache formal execution lock **不得释放**。

Round 3 的 ledger-derived A-1-3 缺陷已经闭合：verifier 现在从
`epoch_variant_counters.csv` 和 `trigger_ledger.csv` 独立重算
`support_by_date.csv`、`variant_summary.csv`，并将 primary A-1-3 actuals
绑定到重算结果。三路同步改写派生统计、gate 和 classification 会在 V07
失败。

但“完整合法 synthetic terminal package + production V00-V12 deep
mutation”仍未闭合。新增 baseline 是当前 verifier 可接受的包，不是 frozen
plan 意义上的合法 evidence package；它利用 verifier 尚未绑定 A-1-2
integrity actuals 的缺口，使空 slice/work evidence 被报告为通过支持门槛。
同时 integrated mutation harness 没有通过真实 `verify_terminal()` 验证
first-failure/NOT_EVALUATED 结果编码，也没有进入 WorkRow/SLICE
FeatureCall/SLICE RawOpenEvent 路径。

## Findings

### P1-1 Synthetic PASS baseline 暴露 A-1-2 evidence-derived gate false-PASS

Runner 对 A-1-2 support actuals 的正式来源是：

```text
slice_rows
  -> slice_mismatch_count
  -> cross_segment_compared_checkpoint_count
  -> represented_slice_date_count
  -> compared_support_checkpoint_count

all build comparable_keys
  -> distinct_comparable_epoch_count
```

具体实现位于 runner lines 3897-3941。Frozen plan 要求 represented slice
dates >=4、distinct comparable epochs >=30、positive compared support
（plan lines 1743-1751、1818-1830）。

新增 synthetic package 却构造：

- `work_manifest.rows=[]`、`per_build_slice_count=0`
  （tests lines 1785-1789）；
- `slice_invariance.csv=[]`、`epoch_support.csv=[]`
  （tests lines 1904-1912）；
- 没有任何 SLICE FeatureCall 或 SLICE_MATERIALIZER RawOpenEvent；
- 同时直接自报
  `represented_slice_date_count=4`、
  `distinct_comparable_epoch_count=30`、
  `compared_support_checkpoint_count=1`
  （tests lines 1937-1950）。

Verifier 只检查 summary integrity 的 exact keys、非负整数类型
（verifier lines 1688-1706），再根据 artifact 自带 actual 判断 gate
predicate/precedence（verifier lines 1320-1500）。它没有从
`slice_invariance.csv`、WorkRow 或 comparable support evidence 重算这些
A-1-2 actuals。

因此 `test_complete_synthetic_terminal_package_passes_v00_v12` 能让上述
不一致 package 在 production V00-V12 全部 PASS（tests lines 2289-2298）。
这不是单纯的 fixture 简化：相同 mutation 能把正式应为
`Aminus1_detector_integrity_failed` 的空 slice evidence 伪装成 A-1-2 PASS，
并让 classification 继续进入 A-1-3，属于 terminal verifier 的 scientific
false-PASS。

在 execution lock 释放前，verifier 必须至少独立绑定：

```text
slice_mismatch_count
cross_segment_compared_checkpoint_count
represented_slice_date_count
compared_support_checkpoint_count
```

到 typed `slice_invariance.csv`。`distinct_comparable_epoch_count` 必须绑定到
一个 verifier 可独立重算的 exact comparable-epoch identity ledger；如果
现有 17 outputs 不足以重建该 union，应在 formal execution 前解决 authority
缺口，而不能信任 summary 自报值。

### P1-2 Integrated V00-V12 mutation tests 仍未证明 frozen terminal-result contract

合法 baseline 调用了真实 `verify_terminal()`，这是进展。但 12 个 mutation
case 使用 `run_checks_through()` 手工逐个调用 production check，并只断言目标
函数抛出任意 `VerificationError`（tests lines 2128-2134、2310-2321）。
它们没有调用真实 `verify_terminal()`，因此没有验证 frozen plan 要求的：

```text
first_failure_code = exact Vxx code
目标前 rows = PASS
目标 row = FAIL
目标后 rows = NOT_EVALUATED
始终 exact 13 rows
exit code = 2
```

这些要求位于 plan lines 1356-1393。现有 exact 13-row negative test
（tests lines 2733 onward）仍通过 monkeypatch 替换全部
`CHECK_FUNCTIONS`，不能证明 production checks 的 mutation 结果编码。

此外所谓 deep V05 mutation 只改了一个 receiver EOF flag
（tests lines 2245-2249）。由于 baseline 的 WorkRow 数为零，它没有覆盖：

- WorkRow path/size/SHA/tree mutation；
- SLICE FeatureCall authority/input hash；
- HASHER/SLICE_MATERIALIZER/LOADER phase matrix；
- SLICE RawOpenEvent drop/reorder/caller/path mutation。

这些是 frozen hostile minimum 的明确项目
（plan lines 1870-1874、1900-1901、1912-1921）。现有较小的 unit tests
不能替代“完整合法 package 上 production V05 深层 mutation”的注册要求，
因为本轮 P1-1 正是跨 sibling、gate 和 terminal closure 才能暴露的缺陷。

修复应构造至少一组真实非空 slice/work evidence，使 A-1-2 actuals由该 evidence
重算成立；随后每个 V00-V12 mutation 均调用 production
`verify_terminal()`，断言 exact failure code、13-row statuses 和 exit 2。

## Round 3 P1 Disposition

1. **Ledger-derived summaries/gates/classification：A-1-3 部分闭合。**
   `recompute_scientific_tables()` 与 runner 的 counter/trigger aggregation
   字段、variant cluster/date/share 公式一致
   （verifier lines 1053-1151；runner lines 1483-1588）。
   `validate_scientific_derivations()` exact 比较两个派生表并绑定三个 primary
   A-1-3 actuals（verifier lines 1154-1188）。Gate truth、precedence 和
   classification 随后由 verifier 重算，三路同步 scientific mutation test
   正确失败。

   但 full-readiness 复核发现同一 gate derivation 问题仍存在于 A-1-2，
   因而不能释放 formal lock。

2. **完整 synthetic package production V00-V12 deep mutation：未闭合。**
   Production PASS baseline 已建立，但 evidence 并不合法完整；mutation
   harness 未验证真实 terminal-result first-failure/NOT_EVALUATED，且没有
   nonempty WorkRow/slice/raw-open 深层覆盖。

## Passed Checks

1. Frozen idea SHA256：
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
2. Frozen plan SHA256：
   `8171a7bed7b216527fed468dbcff8f31ab1f48ec871bb2eee9b0ae7ad123f79c`。
3. HEAD 与 annotated implementation tag 均 peel 到
   `fbd6084f0e564123f94bc6550354e42b0c119f69`。
4. Runner 保持冻结：
   SHA256
   `5fdfe09a42c85f9d7a135d50bca3dda680f33b429a6869ac8e0586f9663967cf`，
   blob `165b076f53df73818de56abfabb55c4c5b472bef`。
5. Verifier SHA256/blob：
   `6f6df6219b13430f308414e3a13fe384c12db47ac704a7eab98118c0df37f1a5`
   / `e582faba6da724e2678e442a957528812dae144a`。
6. Tests SHA256/blob：
   `0fd2aeafc091440e2d0a36543040a2ad87bda147fd2908bed446c5d82633e3f0`
   / `09b68a72473bc93567dc6cb5119d584082ad4144`。
7. Armed claim 的 task/runner/verifier/tests SHA、formal argv、repo/source/
   attempt roots、implementation tag 和 controller identities 与当前冻结状态
   精确一致。
8. Git config 为 `core.fsync=all`、`core.fsyncMethod=fsync`、
   `core.logAllRefUpdates=always`；origin fetch/push URL 精确匹配。
9. Controller ref 查询 exit 0 且为空；claimed path、formal attempt root、
   terminal verifier result、consumption tag 和 terminal tag 均不存在。
10. Baseline authority binding 返回 exact 8 callables；
    `verify_no_historical_attempt()` PASS。
11. Focused suite：`119 passed in 24.39s`。
12. Inherited suite：`53 passed, 1 skipped in 0.45s`。
13. Ruff、Ruff format check、py_compile、git diff check、runner/verifier
    `--help` 全部通过。

## Required Remediation

1. 将 A-1-2 integrity actuals 绑定到 verifier 可独立重算的 typed evidence，
   尤其是 slice date、comparable epoch 和 compared support；增加同步改写
   integrity/gates/classification 的 fail-closed test。
2. Synthetic terminal fixture 必须包含 nonempty WorkRow、SLICE FeatureCall、
   SLICE_MATERIALIZER RawOpenEvent 和 coherent slice-invariance rows，不能以
   空 evidence 配合手填 passing integrity。
3. V00-V12 integrated mutations 必须走真实 `verify_terminal()`，逐项断言
   exact `first_failure_code`、13-row PASS/FAIL/NOT_EVALUATED 序列和 exit 2。
4. 在完整 package 上补 WorkRow、SLICE FeatureCall、RawOpenEvent 及
   candidate/cluster scientific sibling mutation；较小 unit tests可以保留，
   但不能作为 integrated closure 的替代。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/2/0/0**
- Formal attempt: **NOT AUTHORIZED**
- Data execution lock: **CLOSED**
