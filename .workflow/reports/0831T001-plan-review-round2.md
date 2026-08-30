# 0831T001 Independent Hostile Plan Review Round 2

执行线程：
- 独立 hostile plan reviewer

任务ID：
- 0831T001

状态：
- FAIL

更新时间：
- 2026-08-31 CST

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol`
- branch:
  `codex/leader-trigger-transition-hazard-protocol`
- reviewed commit:
  `10643c21fac1b86986e5bbbe5b5ef688c6955d83`
- frozen parent commit:
  `bbcb1bba1d48dc00f59bbbc21f502edbd6df56fb`
- frozen parent protocol SHA256:
  `8e51ce3c278a4da5e94d30e5443b89dd03c43e28bed673165dbb6514a81f8b62`
- frozen parent protocol Git blob:
  `06d24e91589cc1e112e084ebba0c347c54ce5ca3`
- Revision 2 plan SHA256:
  `85bfba7f2d1cbd9ac51aaff30861d71d3e6eccc7ba763f01afafd407f04116c1`
- Revision 2 plan Git blob:
  `f7742203d8acd99c3e7dfbff433090620b97486a`
- Revision 2 task SHA256:
  `dca9085967c9c0ec1908dbc18a910beea6ef0279d5d15d40bac2ba2466d9700a`
- Revision 2 task Git blob:
  `e1bd2cbe29b854ea8030f382c3ddf2993518f1ff`
- fixture-truth SHA256:
  `f032af2f19888cb2198d1fa376222dbc58e1cf964956e2277140562540a8053a`
- fixture-truth Git blob:
  `8e17ce239a98095d96f41af61eb1633ce1c123b7`
- Round 1 report SHA256:
  `08e81c89fcb1c599d0fd8d0ec85db20e73a3d34014d86d332e3aece462f444c2`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未运行 formal、A-1a、A-1b、A0、live/private/order 操作。
- 未修改 parent、plan、task、fixture truth、runner、verifier 或 tests。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/8/1/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 2 有实质改进，但仍不是唯一可执行、production-isomorphic、
independently verifiable 的 Q0 合同。按规则，任一 P0-P2 均阻断
implementation。

## Round 1 Closure Matrix

| Round 1 finding | Round 2 status | 说明 |
|---|---|---|
| P1-1 fixture self-proof | `PARTIAL` | tracked truth 已加入，但 slice/outcome/model-input truth 仍不完整，见 P1-3 |
| P1-2 A-1a causal no-read | `CLOSED_AT_PLAN_LEVEL` | distinct stage entry、CausalView 和 future-read failure 已注册；implementation readiness 仍需验证真实 enforcement |
| P1-3 FeatureBundle/source schema | `NOT_CLOSED` | schema 与 accepted production cache 不同构，且 ratios/base eligibility 未定义，见 P1-1/P1-2 |
| P1-4 slice non-vacuity | `PARTIAL` | exact start/common epoch/floors 已有；expected full/slice semantic identity 仍不完整，见 P1-3 |
| P1-5 A/B/P physical binding | `PARTIAL` | roots/ledger columns 已有；hash preimage 与 runtime access authority 未闭合，见 P1-4 |
| P1-6 package/verifier/negative exactness | `NOT_CLOSED` | JSON schemas、mutation recipes 和 gate precedence 仍有缺口，见 P1-5/P1-6 |
| P1-7 one-shot/fresh-worktree provenance | `NOT_CLOSED` | pre-consumption sequencing 已澄清；readiness equality 不可满足且 claim/controller 协议不完整，见 P1-7/P1-8 |
| P2-1 overbroad production claim | `CLOSED` | Q0 scope 已缩窄到 shared core，stage-specific matching/model qualification 明确排除 |
| P2-2 parent identity | `CLOSED` | parent commit/SHA/blob 精确匹配且 current bytes 无 drift |
| P2-3 reset fixture | `CLOSED_AT_FIXTURE_LEVEL` | QF14/QF15 注册 reset/cross-segment expectations；其 slice oracle 完整性仍由 P1-3 阻断 |

## Findings

### P1-1 Raw schema 与 accepted production cache 不同构且内部矛盾

证据：
- Revision 2 plan lines 197-223 允许 consumed allowlist 之外的 extra raw
  fields 存在，只禁止读取其值。
- lines 278-309 又规定全部 18 字段 row-aligned、`tick_size` 为
  `float64[n]`、只允许一个 extra `qualification_poison`，且禁止任何 scalar
  metadata。
- accepted fixed-epoch source lines 122-155 明确把 `tick_size` 和
  `cache_schema_version`、segment endpoints、quality/reset/gap counters 等
  放在 `METADATA_CACHE_FIELDS`，`tick_size` 不属于 row-aligned fields。

影响：
- 同一个 `build_features` 若严格执行 Q0 schema，会拒绝后续科学阶段实际
  accepted-cache schema；若允许 accepted metadata，又违反 Q0 的 exact
  schema。
- Q0 可以在 synthetic path PASS，却没有资格证明 production historical
  cache path。

必须闭合：
- 使用与 accepted production cache exact-isomorphic 的 row-aligned/metadata
  schema；或注册一个明确、同样由 Q0 调用并验证的 production adapter。
- 消除“extra fields may exist”与“only qualification_poison may be extra”的
  冲突。

### P1-2 Ratios、rolling availability 与 base eligibility 没有冻结公式

证据：
- plan lines 225-232 只绑定 `source_preflight`、`channel_actions`、
  `channel_memories`、`epoch_support_ledger`，没有绑定 predecessor
  `build_features` 或 `base_masks`。
- lines 339-374 只列 `ratios_100`、`ratios_500` 和 `base_eligible` 的输出
  shape/sentinel，没有定义 trade/depletion/OFI ratio 分子分母、checkpoint
  count、segment handling 或 `base_eligible` 布尔公式。
- accepted `channel_actions` 直接消费 externally supplied
  `base_eligible` 和两组 ratios（accepted source lines 718-735）。
- fixture default `activity=1` 给出 500ms rolling sum `25`；accepted
  predecessor `base_masks` 的 activity threshold 是 `44`
  （accepted source `skhynix_flow_coherence_a_minus1_audit.py`
  lines 442-467）。复用该路径会使注册 anchors 不 eligible；不复用则当前
  plan 未说明替代公式。

影响：
- 多个不等价 feature/action paths 都满足文档表面描述。
- fixture expected anchors 不能从 plan + frozen authorities 唯一推出。

必须闭合：
- 冻结 exact rolling-sum/ratio formulas、window checkpoint count、boundary
  availability、base-eligibility formula 和 source/epoch admission order。
- 对新 callable 冻结 path/blob/AST identity requirements。

### P1-3 Fixture oracle 仍允许 full/slice 同源错误自证

证据：
- fixture truth QF07/QF08（lines 403-526）只冻结 anchor ID、common epochs
  和 floors，没有冻结 expected structural cause/event/censor、suppression
  rows 或 `expected_identity_sha256`。
- QF15（lines 846-914）同样没有冻结 full/slice expected semantic tuple/hash。
- QF13（lines 702-776）只冻结 mismatch/violation count 为零，没有冻结
  exact model-input names、canonical values 和 per-field expected access
  rows。
- plan lines 543-565 要求 full/slice outcome、censor、suppression identity，
  但这些 truth values 不在独立 oracle 中。

影响：
- full 和 slice 调用同一个错误 outcome/suppression implementation 时可得到
  equality，仍然 PASS。
- `expected_identity_sha256` 可由 producer/verifier 共同从 observed rows
  构造，而不是由 frozen truth 唯一决定。

必须闭合：
- 为 QF07/QF08/QF15 冻结 exact full/slice semantic rows及其 canonical
  preimage/hash。
- 为 QF13 冻结 exact model-input/access-row oracle，而不是只冻结零 mismatch
  aggregate。

### P1-4 A/B/P hash 与 runtime-access evidence 仍由 producer 自述

证据：
- plan lines 615-640 注册
  `canonical_array_sha256`、`feature_output_sha256`、
  `consumer_input_sha256`，但没有冻结任何一个 hash 的 canonical preimage、
  field order、dtype/shape encoding 或 NaN normalization。
- `field_accesses.csv` 记录 producer 声称的 read rows；plan 没有规定
  production loader proxy/FD boundary、child receipt、process exit/transcript
  或 verifier 如何证明 ledger 来自真实调用而不是事后填写。
- `sender_process_id` 是 producer evidence，不能独立证明 B/P child 实际消费
  对应 input。

影响：
- 不同实现可以对同一数组产生不同“canonical” hashes。
- copied A package 加伪造 B/P ledger 仍缺少不可伪造的 runtime consumer
  closure。

必须闭合：
- 冻结所有 hash preimages 和 typed normalization。
- 冻结真实 loader/access instrumentation、per-call child evidence、
  expected call cardinality和 verifier recomputation rules。

### P1-5 54-file package 只有 CSV schema，JSON 与 negative mutation 不 exact

证据：
- plan lines 645-815 列出 54 paths 和 CSV headers，但没有给
  `authority_binding.json`、`feature_contract.json`、`state_contract.json`、
  `qualification_summary.json`、`formal_identity.json`、
  `fixture_truth_binding.json`、`abp_comparison.json` 或
  `fixture_source_evidence.json` 的 exact schemas/value formulas。
- lines 693-694 禁止额外 directory artifact，却没有枚举 package root 下
  必需且允许的 exact directory set。
- fixture truth lines 916-968 与 plan lines 817-860 只给 negative probe ID
  和 expected error，没有冻结 mutation target path、operation、replacement
  bytes、single-defect condition 或 QF12 failure-injection checkpoint。

影响：
- 54-file path count 相同的两个语义不同 package 都可能被称为 canonical。
- first-fail probes 无法由两套独立实现唯一重放。

必须闭合：
- 冻结所有 JSON schemas、field domains/value derivations、allowed directory
  set 和 raw/sealed comparison projections。
- 为每个 hostile probe 冻结 exact mutation recipe 和 clean baseline
  preconditions。

### P1-6 Missing-artifact first error 与 gate precedence 不可同时成立

证据：
- fixture truth `error_precedence` lines 36-56 和 plan gate order
  lines 1082-1118 把 `AB_IDENTITY` / `AP_IDENTITY` 放在
  `PACKAGE_PATH_SET` 之前。
- negative contract plan lines 821-826 要求 missing/extra artifact 首先报
  `PACKAGE_PATH_SET_MISSING` / `PACKAGE_PATH_SET_EXTRA`。

影响：
- 若从 A 或 B structural projection 删除 required artifact，A/B raw/sealed
  identity 在 path-set gate 前已不可计算或不相等。
- 若 identity gate忽略 missing path以等到后续 path-set gate，则
  `A raw/sealed package bytes == B` 的 gate 不再是 exact/full projection。
- 因而至少 `QF11_MISSING_ARTIFACT` 的 registered first error 在当前顺序下
  不可唯一实现。

必须闭合：
- 将 exact root/path/kind closure 置于任何 A/B/P byte comparison 前；或
  明确注册不依赖文件存在的 comparison semantics，并证明与 equality claim
  一致。

### P1-7 Fresh-worktree “package byte comparison” 按当前合同不可能

证据：
- plan lines 892-908 要求 detached readiness package 与 primary workspace
  readiness package 做 byte comparison。
- `feature_calls.csv` schema lines 747-754 包含 `sender_process_id`。
- lines 632-633、814-815 明确 PID 与 exact roots 保留在 evidence/formal
  identity surfaces，并进入 54-file terminal package。
- 两次 readiness 的 process IDs、worktree path 和 output roots必然不同。

影响：
- 若“package bytes”指完整 54-file package，PASS 条件不可满足。
- 若只比较 structural projection，当前文档没有冻结该 projection、允许差异
  paths 或 comparison manifest。

必须闭合：
- 明确 readiness comparison domain；对环境相关 evidence 使用独立 physical
  closure，并冻结 normalized comparison preimage及唯一允许差异。

### P1-8 Claim/controller/receipt one-shot protocol 仍不足以机器执行

证据：
- plan lines 910-990 只列 claim/tag/root/ref 名称和概括性步骤。
- 没有冻结 armed/claimed claim schema、canonical bytes/hash、attempt-lock
  payload、consumption/terminal receipt schema、Git fsync config、exact commit
  tree、annotated tag messages或 controller old/new observations。
- “local controller ref moves ... exactly once” 没有 exact command、no-replace
  semantics、push receipt、retry policy或 terminal second transition。
- crash 在 armed rename、lock、consumption commit、controller update 任意两步
  之间时的唯一 terminal interpretation未定义。

影响：
- 无法独立判断 claim 是否已消费、是否允许重试、哪个 commit 是唯一
  consumption/terminal authority。
- same-task replacement/rerun 仍不能 fail closed。

必须闭合：
- 冻结 claim/lock/receipt/tag/controller schemas、exact command/transitions、
  fsync/no-replace/retry rules、crash-state classification 和 verifier authority。

### P2-1 AGENTS controller route 与当前 formal task 冲突

证据：
- `AGENTS.md` lines 5-27 仍声明 2026-08-20 controller route，且
  Research Package Trust Kernel 是 next mandatory prerequisite。
- current `task_plan.md` lines 3-18 与 task lines 20-21 则声明 0831T001 是
  当前唯一 formal task。

影响：
- 业务线程同时收到两个不同“下一 mandatory task” authority。
- 即使 Q0 plan 修复，workflow preflight 仍不能唯一判断任务是否获授权。

必须闭合：
- 更新 repo-level controller route，或添加明确 supersession/precedence
  statement，使 0831T001 与 AGENTS 一致。

## Positive Checks

以下 Round 2 修复成立：
- frozen parent commit/SHA/blob 精确匹配，master status 为
  `REGISTERED_MASTER_PROTOCOL`，working bytes 无 drift。
- fixture truth JSON 语法有效，15 fixture order、patch bounds 与 anchor-ID
  internal fields一致。
- scope 已正确缩窄，不再声称完整资格验证 matching/model/statistics。
- A_MINUS1A/A_MINUS1B entry point、CausalView/AvailabilityView/OutcomeView
  语义已分离。
- QF07/QF08 使用 exact epoch-boundary slice，common epoch/floor 非空。
- QF14/QF15 增加 reset/cross-segment fixture。
- fresh detached-worktree 已明确放在 claim consumption 前。
- 54-file arithmetic及 terminal-manifest 53-file preimage count算术一致。

这些通过项不足以覆盖上述 P1/P2。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/8/1/0**
- Plan freeze: **NOT AUTHORIZED**
- Implementation: **NOT AUTHORIZED**
- Formal Q0 execution: **NOT AUTHORIZED**
- historical-cache access: `NONE`
- outcome access: `NONE`
