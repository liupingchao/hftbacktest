# 0831T001 Independent Hostile Plan Review

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
  `5c0945c21cea806ff56d1e7c8ae03fb7b88c5a71`
- master protocol SHA256:
  `80173328a0a0225a0e81406e3fa978fb4a3dc26d73f00e02e5905b94ead3915d`
- master protocol Git blob:
  `a571a1f1458abff7a1a563c5ce4ffd3155926c7a`
- Q0 execution plan SHA256:
  `27f7db534f70cce5e229e3126ec2afe2a28eaa8be02623f465587c7c3a6b6201`
- Q0 execution plan Git blob:
  `64f94bde610c0949cce67b8d7541892c4be1fd40`
- task SHA256:
  `a4b071910d16405bf496b481b3a1ded4f57f9004e1e4dcfd1e6e0136ef60f9c8`
- task Git blob:
  `d7d0476cb9b163ad12482892575a45f1c9b27af2`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未运行 formal、A-1a、A-1b、A0、live/private/order 操作。
- 未修改 plan、task、master protocol、runner、verifier 或 tests。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/7/3/0**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

当前 plan/task 不能冻结。按任务规则，任一 P0-P2 均阻断 implementation。

## Findings

### P1-1 Fixture truth 仍可由同一实现自证

证据：
- Q0 plan lines 95-106 把 fixture generation、negative probes 和 expectation
  reconstruction 分别交给 runner/verifier，但没有冻结独立 oracle。
- Q0 plan lines 229-251 只给出自然语言矩阵；没有精确 fixture payload、
  时间戳、segment、anchor/cause/censor identity、生成配置或静态 truth
  artifact 的 path/SHA/schema。

影响：
- runner、verifier 和 tests 可以共享同一个错误语义并全部 PASS。
- “static table before execution” 不能阻止实现从 observed output 反向生成
  更具体的期望值。

必须闭合：
- 冻结独立、机器可读的 fixture truth authority，包含精确输入生成参数、
  预期 identity/timestamp/cause/censor/counter 和 SHA。
- verifier 必须从该 authority 独立复算，且有 hostile mutation 证明修改
  truth 或 observed output 均 fail closed。

### P1-2 A-1a 的 future-structural no-read 边界不可执行

证据：
- master protocol lines 186-207 禁止 A-1a 读取 post-anchor follower
  direction/cause。
- Q0 plan lines 193-211 只定义一个 `analyze_features(bundle)`，同一分析路径
  随后观察 post-anchor structural outcomes。
- QF13（lines 248-251）只要求 mutation 后 H0/H1 inputs 不变；这证明不了
  future structural values 没有被读取。

影响：
- 输出不变不等于访问边界成立。A-1a 可以读取 future structural state 后
  丢弃结果，仍通过当前 QF13。
- Q0 因而不能授权 outcome-blind A-1a production path。

必须闭合：
- 冻结 A-1a/A-1b 的阶段入口、进程或显式 mode，并规定各阶段可访问字段与
  时间域。
- 用 typed per-call access ledger/poison 证明 A-1a 未读取 post-anchor
  structural direction/cause；同时冻结 mutation domain 和 exact first
  failure。

### P1-3 新 FeatureBundle/source authority 没有唯一 schema

证据：
- Q0 plan lines 120-146 列出字段名，但未冻结 dtype、rank、length、scalar
  versus row-aligned 分类、finite/domain constraints 或 canonical NPZ
  serialization。
- lines 174-191 只称返回 typed `FeatureBundle`，没有字段级 schema 或
  unavailable sentinel。
- slice materializer lines 257-267 依赖“row-aligned/scalar”分类，但该分类
  未注册。
- accepted `source_preflight` 只验证六个 flow contribution
  （accepted source lines 676-701），并不覆盖新加入的 OBI、spread、depth、
  midpoint、timestamp、segment 和 tick size 合法性。

影响：
- 多个不等价实现都符合文字描述；slice、poison、schema error ownership 和
  causal feature validity 无法独立复验。

必须闭合：
- 冻结原始 fixture/cache schema、FeatureBundle schema、字段分类、数值域、
  timestamp/segment规则、sentinel 和 gate/error precedence。

### P1-4 Slice fixtures 可能空比较，且未绑定 accepted epoch disposition

证据：
- QF07/QF08（Q0 plan lines 242-243、253-280）没有冻结 nominal/actual
  slice start、segment、full/slice comparable epoch IDs、预期 support tuple
  或非空 identity set。
- accepted `epoch_support_ledger` 将 slice 内首个不完整 epoch 判为
  `partial_capture_start`，且仅 exact 3000-point single-segment epoch eligible
  （accepted source lines 950-1001）。

影响：
- 实现可以比较不同 epoch universe，或以空 common support 得到 mismatch=0。
- “slice begins before anchor” 并不足以保证该 anchor 所在 epoch 在 slice
  中仍 eligible。

必须闭合：
- 为 QF07/QF08 冻结 exact starts、expected epoch dispositions、common
  support tuple/hash、anchor/cause identities，以及 non-vacuous row/anchor
  floors。

### P1-5 A/B/P equality 没有证明 B/P 实际消费各自输入

证据：
- Q0 plan lines 282-311 要求 A/B/P package equality，并把 fixture cache
  hashes 放到 separate evidence layer。
- package paths（lines 313-343）没有 build-specific input manifest、
  cache-to-call ledger、process/root identity 或 poison attestation schema。

影响：
- B/P 可以复制 A package，再在独立 evidence JSON 中声明不同 cache SHA，
  当前合同仍可能 PASS。
- A/P package equality 与物理 poison 输入差异之间没有 typed、可追溯的
  consumer binding。

必须闭合：
- 冻结 A/B/P 各自 root、input inventory、cache SHA、production call
  inputs/outputs、consumed-field ledger 和 poison mutation attestation。
- verifier 必须从物理输入和 typed ledgers 独立派生 equality，拒绝 copy-only
  package。

### P1-6 Package/verifier/negative-boundary 合同不是 exact

证据：
- Q0 plan lines 313-348 只列路径，没有固定 CSV/JSON schemas、typed
  sentinels、exact row order、manifest preimage/self-exclusion、path count 或
  regular-file/symlink policy。
- negative codes lines 350-371 使用 “At minimum”，没有完整 mutation ->
  first-code 表、gate precedence、later `NOT_EVALUATED` 规则、verifier CLI/
  exit/result schema。
- QF11 只写 missing/extra/reordered，未覆盖 root/path symlink、FIFO、
  directory、non-canonical JSON/CSV 和 synchronized lineage mutation。

影响：
- runner 与 verifier 可对同一坏包给出不同但都“合理”的行为；generic later
  failure 仍可能伪装成 exact-boundary PASS。

必须闭合：
- 冻结 exact artifact set 和每项 schema、canonical bytes、manifest
  lineage、verifier result/exit codes、first-fail precedence 与完整 hostile
  mutation minimum。

### P1-7 One-shot、fresh-worktree 与 provenance 时序冲突

证据：
- formal/verifier argv（Q0 plan lines 373-390）仍含
  `<fresh temporary root>` / `<formal package root>` 占位符。
- lines 392-399 要求 formal 后复制 baseline，再从 implementation commit
  fresh detached-worktree “regeneration”；acceptance line 417 又把该第二次
  regeneration 设为 PASS 条件。
- failure semantics lines 430-445 同时声明 formal 只能一次、失败不得修复或
  rerun。
- master protocol lines 1299-1314、1340-1350 要求 exact roots、input
  manifests、implementation/consumption/terminal identities、frozen claim
  和 one-shot sequence；Q0 plan/task 未定义 claim、attempt root、tags、
  receipt 或 controller ref。

影响：
- 无法确定 fresh-worktree regeneration 是 formal 前 readiness、formal 后
  第二次运行，还是 verifier-only replay。
- 无法机器证明 claim 何时 consumed、哪个 execution 唯一、失败后是否发生
  replacement/rerun。

必须闭合：
- 冻结 exact argv/cwd/roots、implementation/claim/consumption/terminal
  identities和 no-replace attempt lock。
- 明确 fresh-worktree 检查发生在 claim consumption 前还是只验证已发布
  bytes；不得把 post-formal regeneration 留成隐含第二次 formal。

### P2-1 “exact later production path” 的 claim 过宽

证据：
- Q0 plan lines 18-19、72-86 声称资格验证 later A-1a/A-1b exact software
  path。
- 但列出的 production callables 不包含 A-1a control-pool/matching、fold
  support/bin realization，也不包含 A-1b H0/H1 fit、permutation或 bootstrap；
  master protocol lines 1128-1193 明确这些是后续核心产物与分类路径。

影响：
- 即使 Q0 PASS，也只能证明当前 structural feature/anchor/slice/package
  core，而不能证明完整 A-1a/A-1b path 已通过 integration qualification。

必须闭合：
- 缩窄 Q0 claim，或把被授权阶段的真实 production entry points 与 synthetic
  fixtures 纳入资格验证。

### P2-2 Parent master protocol identity 未由 plan/task 固定

证据：
- master protocol lines 8-12 仍标记 `MASTER_PROTOCOL_DRAFT`。
- Q0 plan lines 13-14 和 task lines 23-27 只引用 protocol ID，没有冻结其
  SHA256/Git blob。
- master protocol lines 1301-1305 要求 formal stage 冻结 protocol/document
  SHA256 与 Git blob。

影响：
- implementation/readiness 期间 master scientific semantics 可漂移，而
  task identity 仍看似不变。

必须闭合：
- 在 revised plan/task 中绑定 reviewed master SHA256/blob/commit，并规定
  drift fail closed。

### P2-3 Reset contract 缺少独立 hostile fixture

证据：
- Q0 plan lines 190-191 要求 rolling windows 在 segment boundary reset。
- accepted/master requirement includes exact slice/reset invariance
  （master protocol line 1334）。
- QF06 只验证 post-anchor boundary censor；QF07/QF08 只声明 slice identity，
  没有独立验证 pre-anchor rolling/channel-memory reset、same/opposite
  direction isolation 或 cross-segment carry=0。

影响：
- segment-boundary censor 正确并不能证明 feature and memory reset 正确；
  stale pre-boundary state仍可能污染 anchor。

必须闭合：
- 增加 registered reset fixtures、exact expected arrays/counters，以及
  cross-segment carry hostile mutations。

## Accepted Authority Check

只读核对确认：
- accepted authority commit:
  `f06eb5cb012cb62b2a778ad90d433c4083f9ba14`
- annotated tag `skhynix-fixed-epoch-suppression-v1` 精确 peel 到该 commit。
- frozen source SHA256:
  `dfa8af1f4b8410370ec7ccd0bea30b63840ebe484d74446c8cbe2564918ac070`
- reused callable normalized AST SHA256:
  - `source_preflight`:
    `9089e2f348965c6601d23315be625f333558b688dda4318b034fff38d80d57a6`
  - `channel_actions`:
    `0dc86f04ff18fe490f6f7c2f6332acadc8d441c68deb798e4829bfae5d0277ab`
  - `channel_memories`:
    `5507a492a9984d0aec1f36ef16a9977ff2321af4c86e245fa3a0ddfe8c7e4df1`
  - `epoch_support_ledger`:
    `d76a80ef31b3f229eb23099f1f3b06cf6ab2e2a3f101a07c37c2286436f5b453`

这些 authority identity 本身可恢复；本轮 FAIL 来自新 Q0 plan/task 尚未把
它们组织成唯一、不可自证、可独立验证的一次性执行合同。

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/7/3/0**
- Plan freeze: **NOT AUTHORIZED**
- Implementation: **NOT AUTHORIZED**
- Formal Q0 execution: **NOT AUTHORIZED**
- historical-cache access: `NONE`
- outcome access: `NONE`
