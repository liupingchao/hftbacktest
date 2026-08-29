# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0829T003

状态：
- 已通过

更新时间：
- 2026-08-29 23:12 CST（星期六）

验收线程：
- QA验收线程

验收对象：
- SKHYNIX Fixed Causal Epoch M-State V2 A-1 第三轮最终 QA
- branch `codex/fixed-causal-epoch-mstate-a-minus1`
- HEAD `9cd9be5171936f2b28a469a293cc1012c56cb394`
- hostile-test remediation `b583d02b12a2464a9d58863c3b761fca6f37e8d3`
- frozen plan SHA256
  `682ea69016c472d5ae3adc255f974d05ef72d7d78e90e11b976b52589a501aba`

验收范围：
- 关闭上一轮唯一 P2：冻结计划第 12 节 hostile-test minimum 的持久回归覆盖。
- 复验正式 Build A、Build B、poison Build P 的 25 项 evidence closure、
  exact determinism、outcome poison、null、slice、gates、classification
  与 A0 lock。
- QA 未运行 29-cache，未读取 future outcomes，未修改 plan、runner、
  tests、task、execution report 或正式研究产物。
- 本轮只覆盖本报告与 `docs/qa-acceptance-report.md`。

严重度：
- P0: 0
- P1: 0
- P2: 0
- P3: 0

验收步骤：
1. 核对 HEAD、remediation ancestry、冻结计划 SHA 和工作区初始状态。
2. 将冻结计划 `:1035-1100` 的 hostile minimum 逐项映射到 current、
   predecessor tests 和正式 evidence invariants。
3. 运行 current focused suite、current+predecessor suite、Ruff 和隔离
   py_compile。
4. 独立枚举 A/B/P 的 25 个 non-cache artifacts，复算 manifest
   size/SHA256 和三路逐字节 equality。
5. 独立重建 gates、numeric integrity、classification，并复核 candidate、
   null、slice、outcome ledger、attestation 和 A0/future locks。

实际结果：
- Frozen plan SHA256 精确匹配任务冻结值；`b583d02b` 是当前 HEAD 的直接
  祖先。
- Current focused suite 为 `48 passed`；current+predecessor suite 为
  `68 passed`；Ruff 与隔离 py_compile 均通过。
- 上一轮缺失的 epoch disposition mutations、core reset 同/反向隔离、
  typed/hash support mutation、raw/structural occupancy 与 numeric gate
  precedence、A-1-5/A-1-6 后续 gate 语义、manifest self-exclusion 和
  exact-25 mutation 均已形成可执行的 fail-closed 回归。
- Structural `occupied > eligible` 现在与 raw occupancy 对称地计入 numeric
  integrity violation；对应 mutation 在 A-1-4 精确失败，后续 gates
  保持 `NOT_EVALUATED`。
- 冻结 hostile minimum 的其余边界、thinning、delete-only、null、
  exposure、source/action/memory、poison 和 build closure 条目由 current
  与 predecessor suites 以及正式 evidence invariants 共同覆盖，未发现
  未闭合条目。
- A/B/P 各有精确 25 个唯一 non-cache artifacts；manifest 各列出除自身
  外的 24 项，所有 size/SHA256 独立复算正确。
- A/B、A/P、B/P 的 25 项 artifact byte-difference 均为 `0`。
- 三路 `outcome_access_ledger.json` SHA256 均为
  `5072c60ba234042bcf981ae1fd17ca199adbdd2e33500bff395925f79b847140`；
  均为 `stage=final`、`executed=true`、
  `poisoned_unconsumed_fields_change_output=false`、
  `final_difference_count=0`。
- Poison attestation SHA256 为
  `3eaca61d3769f2dee6c50aea45a95109fc622b2ce557277451b504c8aca8942b`；
  记录 29 caches、15 unconsumed fields、435 changed instances、
  0 consumed mismatch。
- `candidate_ledger.csv` 有 2219 行，四个 frozen epoch fields 均无空值。
- Evaluation null 有 597 行，10s/30s/60s 各有完整 199 replicates。
- `slice_invariance.csv` 有 186 行，identity/support mismatch 与
  cross-segment checkpoint violation 均为 `0`。
- 三路 summary 的 stored gates 与当前 runner 重建结果完全相同，
  `numeric_integrity_violations=0`：A-1-0 至 A-1-4 PASS，A-1-5 FAIL，
  A-1-6/A-1-7 为 `NOT_EVALUATED`。
- 三路 classification 均为
  `Aminus1_structural_support_not_estimable`；confirmatory A0、
  exploratory A0 execution 与 future-target access 均保持 `false`。

Previous Finding Closure：
- P2-1 已关闭。冻结计划第 12 节要求的缺失 hostile surfaces 已由
  `b583d02b` 的 mutation tests 和 structural occupancy 对称 fail-closed
  修复持久化；本轮未发现新的 P0-P3 finding。

验收结论：
- 已通过
- 结论说明：
  - 实现、测试和三路 evidence 流程符合冻结合同；QA 通过只表示执行与
    证据正确，不改变负面科学分类。

通过项：
1. Frozen hostile-test minimum 已闭合为持久回归。
2. A/B/P 25-artifact closure、exact determinism 与 poison boundary 通过。
3. Fixed epoch/reset/slice、199 null、sequential gates 与 classification
   可独立复算。
4. `Aminus1_structural_support_not_estimable` 和 A0/future locks 保持不变。

不通过项：
1. 无

缺陷清单：
1. 无

阻塞项：
- 无

建议总控下一步：
1. 可将 0829T003 流程状态推进为 `已通过`。
2. 保持 A0 与 future outcome lock；若继续研究，应注册新的独立假设，
   不得把本次 QA 通过解释为当前 M-state 获得科学支持。

提交信息：
- commit：无
