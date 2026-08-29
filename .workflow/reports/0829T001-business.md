# 业务线程执行报告

任务ID：
- 0829T001

标题：
- PRECISION_FIRST_FLOW_COHERENCE_V2 A-1 structural false-fire audit

日期：
- 2026-08-29

状态：
- 待验收

合同：
- Revision 8 independent plan review：
  `P0/P1/P2/P3 = 0/0/0/0`
- Frozen plan SHA256：
  `6e6e1af47dbf0c982ee83654c60f054aac6d81452b1246d65a39123a7d384593`
- Frozen implementation commit：
  `734bf9d5`

执行范围：
- 29 个 frozen historical caches；
- 9 个 historically reused post-selection dates；
- 30s selection bank，199 replicates；
- 10s/30s/60s evaluation banks，各 199 replicates；
- 27 个 precision filters；
- 未读取 future return、markout、fill、fee、PnL 或 model loss。

Build 证据：
- Canonical Build A：
  `local_live_analysis/skhynix_precision_first_flow_coherence_a_minus1_0829T001/`
- Fresh Build B：
  `local_live_analysis/skhynix_precision_first_flow_coherence_a_minus1_0829T001_build_b/`
- Final non-cache artifact count：18。
- Final Build A/B difference count：0。
- Manifest closure violations：0。

执行 remediation：
- 初始 Build A 的 187 个 slice rows 中有 12 个 support-count mismatch。
- 原因是 actual count 未限制目标 segment；所有 candidate identities
  原本已 exact。
- 修复 segment mask 后，slice identity/support mismatch 均为 0。
- 同时将合法的 `NOT_ESTIMABLE` 从 JSON Infinity 改为 null，避免违反
  non-finite fail-closed contract。
- Null counts、selection、thresholds 和 detector semantics 未改变。

结果：
- Common base candidates：350。
- Candidate ledger SHA256：
  `b47e8998353a1059c88b7c25b67a58c29193fed152985463964632d35d09f5dc`
- 所有九个 folds 由 null-only selection 选择 `F000`。
- 所有 27 个 filters 的 observed admitted candidate/cluster count 为 0。
- `F000` cancellation：
  - `ABSTAIN=279`
  - `not_novel=34`
  - `conflict=22`
  - `coherence_lost=8`
  - `opposite_coherence=5`
  - `margin=2`

Primary 30s：
- Comparison-supported exposure：0.411655556 hours。
- Observed clusters：0。
- Null p95 clusters：0。
- Burden ratio：NOT_ESTIMABLE。
- Count-tail p：1.0。
- Represented dates：0。

Null admissibility：
- 每个 bank distinct fingerprints：199。
- Cross-bank stream identity overlap：0。
- Conservation/invariant mismatches：0。
- Minimum date pair count：6。
- Maximum date p95 joint distance：0.582116629。

Gate 结果：
- A-1-0 Authority：通过。
- A-1-1 Zero Outcome：通过。
- A-1-2 Tri-State And Abstention：通过。
- A-1-3 Null Admissibility：通过。
- A-1-4 Selection Integrity：通过。
- A-1-5 Structural Support Estimability：未通过。
- 后续 gates 不得用于救援 A-1-5。

唯一 classification：
- `Aminus1_structural_support_not_estimable`

科学解释：
- 该 detector 没有证明低 structural false-fire burden。
- 更准确的结论是：在严格 precision-first 路径支持、novelty 和
  persistence 语义下，没有信号存活，因此 false-fire burden 与经济
  precision 都不可估计。
- 主要瓶颈是路径 support，而不是 margin；这与“接受漏检”不同，因为
  当前漏检已退化为零可评估触发。

权限：
- `future_target_access_authorized = false`
- `exploratory_a0_execution_authorized = false`
- `confirmatory_a0_authorized = false`
- live trading 未授权

验证：
- Focused pytest：`34 passed`
- Ruff：通过
- py_compile：通过
- git diff check：通过
- Build A/B full non-cache SHA：零差异

QA Round 1 remediation：
- 代码 commit：`e56bb5a5`。
- 数值损坏现在由动态 integrity audit 捕捉；negative/non-finite
  raw、primary、sensitivity hostile cases 都首先分类到 A-1-4。
- Build A/B 各自恰好包含 frozen Required Outputs 的 21 个
  non-cache artifacts，无缺失、无额外文件。
- 新增 350-row `candidate_ledger.csv`、243-row
  `filter_support_by_date.csv` 和 243-row
  `tri_state_support_by_date.csv`。
- preseal、pending、final Build A/B difference count 均持久化为 0。
- Manifest 包含 20 个 manifest-external artifacts，hash closure
  mismatch 为 0。
- Candidate count、candidate SHA、F000 cancellation、fold selection、
  observed/null counts 和最终 classification 均未改变。

当前状态：
- 待验收
- 等待独立 QA Round 2。
