# 0830T002 Independent Implementation Readiness Review Round 3

执行线程：
- 独立 implementation readiness review

任务ID：
- 0830T002

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch: `codex/fixed-epoch-relaxed-mstate-successor`
- reviewed HEAD:
  `615df73ff9fa106521c0b7d53d768b0a61c93e60`
- remediation commit:
  `0b7f6823d3d7242703a93f9f4356ee8a2d59001e`
- implementation tag:
  `skhynix-fixed-epoch-leader-trigger-a-minus1-implementation-v1`
- tag peel:
  `615df73ff9fa106521c0b7d53d768b0a61c93e60`

更新时间：
- 2026-08-30 20:40 CST

审查限制：
- 未打开 29 个正式 cache。
- 未运行 formal attempt。
- 未读取 future outcomes。
- 未修改 frozen idea、plan、task、armed claim、runner、verifier、tests 或
  Round 2 report。
- 本轮只新增本报告。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/2/0/0**
- 29-cache formal execution lock **不得释放**。

Round 2 的 signed direction 和 candidate/cluster identity 缺陷已经关闭。
Production `check_v01` 至 `check_v12` 也都有直接调用，但新增测试只证明各
check 的一个早期失败分支，没有建立完整合法 terminal package 的正例。

全量复核同时发现一个新的 scientific false-PASS：V07 没有从
epoch counter/trigger ledgers 独立重算 `support_by_date`、
`variant_summary` 和 gate actuals。三路一致地改写这些派生统计及
classification 后，当前 verifier 只验证它们彼此一致和阈值计算正确，不能
证明它们仍由底层 observed ledgers 推导。

## Findings

### P1-1 V07 未将派生 support/gate/classification 绑定到底层 ledgers

Runner 的冻结语义明确规定：

- `support_by_date.csv` 的各 count 是 epoch counter rows 按
  `(research_date,variant,direction)` 聚合；
- confirmed cluster/support0/1/2 来自 confirmed trigger rows；
- `variant_summary.csv` 的 counts、distinct clusters、represented dates 和
  maximum date share 由 counter/trigger rows 重算；
- A-1-3 三个 actuals 来自 primary `variant_summary` row
  （runner lines 1483-1587、1925-1977）。

Verifier 当前只对 `support_by_date.csv` 检查排序和 direction/variant domain，
对 `variant_summary.csv` 检查三行 variant order、primary flag 和 share
sentinel（verifier lines 931-958）。新加的 `validate_candidate_links()` 只
绑定 trigger candidate 与 epoch counter 的 retained ID/count
（verifier lines 989-1018）。

Gate verifier 随后只根据 artifact 自带的 `actual` 重新判断 threshold 与
precedence（verifier lines 1150-1303）；`validate_final_root()` 只要求
summary 中的 `variant_rows/gates` 等于对应 CSV/JSON，并要求 classification
三份文本彼此相等（verifier lines 1513-1559）。没有任何路径重新执行 frozen
aggregation 或将以下字段交叉绑定：

```text
epoch_variant_counters + trigger_ledger
  -> support_by_date
  -> variant_summary
  -> gate actuals
  -> classification
```

因此，一个三路一致 mutation 可以同步改写 `support_by_date.csv`、
`variant_summary.csv`、`gate_contract.json`、summary 和 classification，
再重建 manifests/comparison/receipt hashes；V07/V08/V09/V10/V11 仍可能接受
与 observed ledgers 不一致的科学分类。这直接影响 A-1-3 是否 PASS，属于
single-use formal attempt 的 P1 false-PASS。

### P1-2 Round 2 注册的 production V01-V12 hostile closure 仍不完整

Round 2 明确要求使用可复用 synthetic terminal fixture：

```text
完整合法 package 通过 production V00-V12
每个 check 的 registered artifact mutation fail closed
```

（Round 2 report lines 150-159）。

当前新增测试确实直接调用 production V01-V12，但多数只构造能触发首个浅层
错误的残缺 context：

- V03 只改 claim formal argv；
- V04 只放一个 unexpected child；
- V05 只改 work-manifest schema version；
- V06 只改 poison task ID；
- V07 使用三个空目录，只触发 `final17_path_set`；
- V08 只改 manifest artifact count；
- V10/V11 只改 task ID；
- V12 使用空 roots，只触发 unexpected child
  （tests lines 1450-1789）。

现有 synthetic 17-path seal test 不调用 verifier
（tests line 716），production V07 test 也没有让一个完整合法 root 走过
`validate_final_root()`。原有 monkeypatched `CHECK_FUNCTIONS` short-circuit
test 仍只验证 13-row 状态编码（tests line 1832）。

这组测试没有建立“先证明合法完整 package PASS，再逐层 mutation”的因果
基线，无法覆盖 V05 instrumentation 深层路径、V06 poison cache rows、V07
cross-file aggregation、V08-V11 完整 sibling closure 或 V12 root drift。
P1-1 正是该缺口未被 104 个 passing cases 检出的实例，因此 Round 2 的第三个
P1 只能判定为部分闭合。

## Round 2 P1 Disposition

1. **合法 `direction=-1`：闭合。**
   `direction` 使用 signed parser，随后在 counter/trigger/support tables 中
   由 semantic domain 精确限制为 `{-1,1}`
   （verifier lines 755-819、868-946）。Synthetic negative-direction row
   已通过 parser/semantic checks（tests lines 1346-1404）。
2. **Canonical candidate/cluster identity：闭合。**
   Verifier 从七元组重算 candidate SHA，验证 cluster
   `capture_id:epoch_id`，并将 counter retained ID/count 与 trigger row
   交叉绑定（verifier lines 887-929、989-1018）。Candidate、cluster 和
   retained-link mutations 均有直接 negative tests
   （tests lines 1407-1437）。
3. **真实 production V01-V12 mutation coverage：部分闭合，仍失败。**
   所有 check functions 已被直接调用，但缺少完整合法 fixture、production
   V07 valid-root test 和注册 hostile minimum 的深层 mutations。

## Passed Checks

1. Frozen idea SHA256：
   `a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997`。
2. Frozen plan SHA256：
   `8171a7bed7b216527fed468dbcff8f31ab1f48ec871bb2eee9b0ae7ad123f79c`。
3. HEAD 与 annotated implementation tag 均 peel 到
   `615df73ff9fa106521c0b7d53d768b0a61c93e60`。
4. Idea、plan、task、runner、verifier、tests、armed claim 的 working
   bytes、Git blob 和 implementation-tag tree blob 全部一致。
5. Armed claim 中 task/runner/verifier/tests SHA、formal argv、repo/source/
   attempt roots 与 controller identities 精确匹配当前冻结状态。
6. Runner SHA/blob 未变：
   `5fdfe09a42c85f9d7a135d50bca3dda680f33b429a6869ac8e0586f9663967cf`
   / `165b076f53df73818de56abfabb55c4c5b472bef`。
7. Git config 为 `core.fsync=all`、`core.fsyncMethod=fsync`、
   `core.logAllRefUpdates=always`；origin fetch/push URL 精确匹配。
8. Controller ref 查询 exit 0，stdout/stderr 均为空，remote ref 当前不存在。
9. Consumption/terminal tags、claimed path、formal root 和 terminal verifier
   result 均不存在；formal state 为 `UNCONSUMED`。
10. Baseline authority verifier PASS；successor frozen docs、8 callable
    bindings 与只读 historical-attempt scan PASS。
11. Focused suite：`104 passed in 9.73s`。
12. Inherited suite：`53 passed, 1 skipped in 0.47s`。
13. Ruff、Ruff format check、py_compile、runner/verifier `--help` 全部通过。

## Required Remediation

1. 在 verifier 中独立重算并 exact 比较：

```text
epoch_variant_counters + trigger_ledger
  -> support_by_date
  -> variant_summary
  -> primary A-1-3 actuals
```

   同时将 A-1-0/A-1-1/A-1-2 actuals 绑定到 inventory、comparison、
   poison、slice 和 instrumentation evidence 的 authoritative values。
2. 建立一个不读正式 cache 的完整 synthetic terminal package fixture，使
   production V00-V12 全部 PASS；随后对 frozen hostile minimum 的每个
   authority/sibling/schema/identity surface 做单点 mutation，并断言准确的
   first failure/NOT_EVALUATED。
3. 至少增加以下 integrated production mutations：
   support-by-date count、variant cluster/date/share、gate actual、
   classification、candidate/cluster、WorkRow/FeatureCall/RawOpenEvent、
   poison cache row、manifest/comparison/result/receipt 和 post-seal drift。
4. 修复后重新冻结 verifier/tests SHA/blob、task、armed claim 与
   implementation tag，再执行下一轮 independent readiness review。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/2/0/0**
- Formal attempt: **NOT AUTHORIZED**
- Data execution lock: **CLOSED**
