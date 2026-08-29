# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0829T003

状态：
- 未通过

更新时间：
- 2026-08-29 23:00 CST（星期六）

验收线程：
- QA验收线程

验收对象：
- SKHYNIX Fixed Causal Epoch M-State V2 A-1 remediation
- branch `codex/fixed-causal-epoch-mstate-a-minus1`
- HEAD `6c5ff601c69bf9e05b4c2c5c84bc1ad2956ab90a`
- implementation remediation `cfc04b4921635eb46fffe08f7684679990f31ec9`
- final ledger remediation `5b6c2ecde058af4d235f9d12dd2288fb20bb6c63`

验收范围：
- 重新从最新 HEAD 验收冻结计划、runner/tests、canonical A、fresh
  canonical B、poison P、poison attestation、25 项 evidence closure、
  candidate schema/value、outcome boundary、199 null、sequential gates、
  classification 与 A0 lock。
- QA 未运行 29-cache，未读取 future outcomes，未修改 plan、runner、
  tests、task、执行报告或研究产物。

严重度：
- P0: 0
- P1: 0
- P2: 1
- P3: 0

验收步骤：
1. 核对最新 HEAD、remediation ancestry、任务状态和 frozen plan SHA256。
2. 独立枚举 A/B/P 的 non-cache paths，复算每份 manifest 的 size/SHA256
   及三路逐字节 equality。
3. 独立解析并重算 candidate CSV exact header、epoch arithmetic、
   candidate identity、cluster identity 和空值计数。
4. 从 canonical cache arrays 独立重建每个 XOR poison hash，核对
   attestation、三路 outcome ledger 和 A-1-1 evidence。
5. 复算 199 null、slice evidence、gate precedence、classification 与
   A0/future outcome locks。
6. 运行 current focused tests、predecessor tests 和 Ruff，并逐条对照
   frozen hostile-test minimum。

实际结果：
- frozen plan SHA256 精确为
  `682ea69016c472d5ae3adc255f974d05ef72d7d78e90e11b976b52589a501aba`。
- A/B/P 各有精确 25 个唯一 non-cache artifacts；manifest 各列出除自身
  外的 24 项，所有 size/SHA256 可独立复算，A/B 与 A/P 的 25 项
  byte-difference 均为 `0`。
- 三路 `outcome_access_ledger.json` SHA256 均为
  `5072c60ba234042bcf981ae1fd17ca199adbdd2e33500bff395925f79b847140`。
- 三路 ledger 均为 `stage=final`、`executed=true`、
  `poisoned_unconsumed_fields_change_output=false`、
  `preseal/pending/final_difference_count=0`，且 poison evidence 与
  summary 完全一致。
- attestation SHA256 精确为
  `3eaca61d3769f2dee6c50aea45a95109fc622b2ce557277451b504c8aca8942b`。
  独立核对 29 caches、15 unconsumed fields、435 field instances；
  435 个 source hashes 和按 byte XOR `0xff` 重建的 poison hashes 全部
  匹配，`consumed_field_mismatch_count=0`。
- `candidate_ledger.csv` exact header 与冻结 schema 一致，共 2219 行；
  四个 epoch fields 空值均为 `0`。所有行通过 epoch/core arithmetic、
  half-open core、direction、cluster、candidate-id 和唯一性复算。
- evaluation null CSV 有 597 行，10s/30s/60s 各为完整 replicate
  `0..198`；selection/evaluation 四个 bank-duration fingerprint count
  均为 199，stream overlap 和 invariant mismatch 均为 `0`。
- 186 个 slice rows 全部 `identity_exact=True`、
  `support_identity_exact=True`、`cross_segment_checkpoint_count=0`；
  覆盖 9 日期和 1730 distinct comparable epochs。
- summary、classification 和 gate contract 完全一致：
  A-1-0 至 A-1-4 PASS，A-1-5 FAIL，A-1-6/A-1-7 为
  `NOT_EVALUATED` 且每个 condition 均保持
  `passed=null,actual=null`。
- classification 为 `Aminus1_structural_support_not_estimable`；
  confirmatory/exploratory A0 与 future-target authority 均为 false。
- current suite 为 `30 passed`；连同 predecessor suite 为
  `50 passed`；Ruff 为 `All checks passed`。

## Previous Finding Closure

### P1-1 Candidate Ledger Epoch Fields

- 已关闭。
- `candidate_diagnostics()` 现在填充四个 epoch fields，并在 production
  write 前调用 `validate_candidate_ledger_rows()`。
- 新增正向 schema/value 测试和缺失 epoch value 的 fail-closed 测试。
- 三路正式 CSV 均通过本轮独立逐行复算。

### P1-2 Full Outcome Poison Pipeline

- 已关闭。
- poison P 运行完整 29-cache pipeline，finalizer 比较 A/B/P 全部 25 项
  artifacts；attestation 可由 canonical arrays 独立重建。
- commit `5b6c2ecd` 修复 dynamic seal 未重写 outcome ledger 的问题；
  三路 ledger 现在均为 final 且 SHA 相同。
- 新增 poison field、attestation mutation、triad artifact mutation 和
  pending/final ledger payload tests。

## Finding

### P2-1 Frozen Hostile-Test Minimum 仍未完整落成持久回归

- 冻结计划
  `docs/skhynix_binance_precision_first_fixed_causal_epoch_mstate_v2_a_minus1_audit_plan_20260829.md:1035-1100`
  明确把整组 hostile cases 定义为 `At minimum`，不是可选建议。
- 本轮新增测试已经覆盖上一轮直接暴露的 candidate、poison、attestation、
  triad comparison 和 final ledger 缺陷，但 current suite 仍只有 30 个
  tests。
- 仍没有对应的 durable mutation tests 覆盖至少以下冻结 surfaces：
  missing/irregular/duplicate/off-grid/empty-intermediate epoch 与 disposition
  precedence；core 内 reset 的同向/反向隔离；slice support hash/typed
  ordering mutation；raw/structural occupancy hash、subset 和 spoofed share；
  A-1-2/A-1-3/A-1-4 numeric/`NOT_EVALUATED` precedence；manifest
  self-exclusion/exact-25 path mutation。
- 本轮 QA 的只读脚本能够证明当前 artifacts 正确，但不能替代这些
  fail-closed mutation 的持久回归保护。因此上一报告的 P2-1 只部分
  关闭，尚不满足 frozen test contract。

验收结论：
- 未通过
- 结论说明：
  - 当前三路执行、outcome evidence、负面科学分类和 A0 lock 均可复现且
    自洽；剩余失败仅为冻结 hostile-test minimum 未完整闭合。

通过项：
1. 上一轮 P1-1 candidate schema/value 缺陷已关闭。
2. 上一轮 P1-2 full outcome poison pipeline 缺陷已关闭。
3. 最新 outcome ledger pending 缺陷已关闭，三路 final ledger 完全一致。
4. A/B/P 25-artifact closure、attestation、199 null、slice、gates、
   classification 和 A0 lock 全部通过独立复算。
5. Current 30 tests、current+predecessor 50 tests 和 Ruff 全部通过。

不通过项：
1. Frozen hostile-test minimum 尚未形成完整持久 mutation coverage。

缺陷清单：
1. P2-1：补齐计划 `:1035-1100` 中尚未覆盖的 durable hostile tests。

阻塞项：
- 无外部阻塞。

建议总控下一步：
1. 仅补测试，不改变冻结计划、阈值、runner 科学语义或正式结果。
2. 优先用参数化测试覆盖 epoch disposition、reset/slice、occupancy/numeric
   gate precedence 和 manifest mutation。
3. 测试补齐后重新运行 current+predecessor suites、Ruff，并做一次只读
   QA；无需因纯测试补充重新解释当前负面科学分类。
4. 在 QA 通过前继续保持 A0、future outcome 与 live/private/order lock。

提交信息：
- commit：无
