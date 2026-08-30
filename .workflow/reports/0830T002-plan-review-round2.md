# 0830T002 Hostile Plan Review Round 2

日期：
- 2026-08-30 CST（星期日）

审查角色：
- 独立 hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `4a8b6655d9db3866ed84c7718343204e167cdf54`
- idea SHA256
  `597f0f7da889fa8f9d8c38f2750a94224a3cfae4df4cb99eb8f262dca693cd40`
- plan SHA256
  `3c6b0d73603b5c231e63263bc1bc974d3a0b3913adb54ce6505ce1112758cbbc`
- task SHA256
  `56a9718558dc38e25bd0adcf3e5ba065a9625b11569d46ed918d885321c52ae8`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

## Severity Summary

- P0: 0
- P1: 8
- P2: 3
- P3: 0

## Round 1 Closure Matrix

| Round 1 finding | Round 2 status | 结论 |
|---|---|---|
| P1-1 executed primitive authority | PARTIAL | source/action/memory/epoch 已改为 direct-call，但 slice/full feature builder 仍未绑定唯一 callable authority |
| P1-2 checkpoint order/TTL | CLOSED | checkpoint 顺序、prestate、post-action veto 和 `age <= 100ms` 已冻结 |
| P1-3 confirmation completion | PARTIAL | completion 与双时间戳已冻结，但 core-close 非法窗口没有唯一 disposition/cancel atom |
| P1-4 slice state identity | PARTIAL | epoch/counter/retained/status/support identities 已补齐，但 feature builder 与 mismatch encoding 仍不唯一 |
| P1-5 poison/A-B-P stages | PARTIAL | roots、29/15/435、三阶段已注册，但 final comparison/gate ownership 与动态证据闭包矛盾 |
| P1-6 output schemas | PARTIAL | CSV 大幅补齐；JSON、sibling receipts 和 stage-specific schemas 仍不是 exact schema |
| P1-7 gates/classification | PARTIAL | sequential gate 框架已补齐，但 A-1-0/A-1-1 precedence 冲突，A-1-3 内部 short-circuit 未冻结 |
| P1-8 one-shot immutability | PARTIAL | formal CLI 与 pre-cache lock 已注册，但 claim 不耐删除/崩溃，不能证明 attempt 不可替换 |
| P2-1 vocabulary | PARTIAL | 分层守恒已补齐，但 idea 顶层顺序与 normative checkpoint 顺序冲突 |
| P2-2 hostile minimum | CLOSED | 已注册 hostile-test minimum |

## Findings

### P1-1 Normative detector order remains contradictory

位置：
- idea `:44-48`
- idea `:127-139`
- execution plan `:99-110`

问题：
- Idea 的顶层定义写成
  `fixed-epoch admission and thinning -> explicit-opposition veto`。
- 同一 idea 的 checkpoint-exact normative 顺序却是
  `epoch/core -> anchor veto -> thinning`。
- Plan 又声明 idea Revision 2 的 veto、thinning 等定义均为 normative。
- 两种实现会产生不同的 retained trigger：先 thinning 时，最早但被 veto
  的 onset 可以占用 key；先 veto 时，它不会阻止下一笔 admitted onset。

必须修复：
- 全文只保留唯一顺序：
  `raw onset -> epoch/core admission -> anchor veto -> thinning ->
  retained-only confirmation`。
- hostile test 必须证明被 veto 的早期 onset 不占用 thinning key。

### P1-2 Slice/full feature builder has no frozen callable authority

位置：
- execution plan `:58-72`
- execution plan `:184-185`
- baseline manifest `baselines/skhynix_fixed_epoch_suppression_v1/baseline_manifest.json:15-24`
- authority runner
  `examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py:298-349`
- authority runner
  `examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py:2110-2114`

问题：
- Plan 的 direct-call list 不包含 `build_features`，却要求 slice 使用
  “authority feature builder”。
- Suppression baseline 冻结的九个 callable AST 也不包含 feature builder。
- Accepted runner 实际通过一个更早 predecessor 的
  `predecessor.build_features` 间接构建 full/slice features。
- 因而 successor 可以选择不同的 window、边界或 raw-to-ratio builder，
  同时仍声称满足当前 direct-call list；slice identity 的输入不唯一。

必须修复：
- 冻结 feature builder 的 exact path、commit、blob、file SHA、callable AST
  和 direct invocation。
- full A/B/P 与每个 sliced raw cache 必须调用同一个绑定 callable。
- 将 feature-builder mutation/direct-call hostile case 加入 minimum。

### P1-3 Core-close failure has no unique state or conservation disposition

位置：
- idea `:197-214`
- idea `:247-290`
- execution plan `:148-173`

问题：
- Raw onset 只要 `t` 位于 `[core_open,core_close)` 就进入 anchor veto。
- Confirmation 又要求 `t+200ms <= core_close`。
- 对于 `t` 在 core 内但 `t+200ms > core_close` 的 onset，合同没有说明它是：
  - `epoch_core_omitted`；
  - `veto_admitted` 后被取消；
  - 还是 formal integrity failure。
- 四个 cancellation booleans 没有 `confirmation_outside_core`，而
  `insufficient_confirmation_history` 也没有被定义为包含该情况。
- 不同选择会改变 raw/veto/admitted/retained conservation、thinning key
  占用和 confirmed support。

必须修复：
- 在 thinning 前注册 exact confirmation-edge admission，或增加唯一
  cancellation atom。
- 明确该对象进入哪一条守恒边，并冻结 equality 与 `+20ms` hostile case
  的 expected disposition/status/reason。

### P1-4 Formal attempt claim is not durable or non-replaceable

位置：
- execution plan `:281-304`
- execution plan `:319-325`
- execution plan `:800-821`

问题：
- 合同先创建 attempt root，再写 lock；两步之间崩溃会留下没有 receipt
  的 root，且没有冻结 recovery/terminal-state protocol。
- 只要求 fsync JSON file，没有要求 attempt namespace、root 和 lock
  directory entry 的 parent-directory fsync。
- 没有冻结 `O_EXCL`/atomic no-replace primitive、lexical no-symlink path
  检查或同目录 temporary-file protocol。
- Attempt root 位于可删除的 ignored `local_live_analysis`。删除失败或中断
  的 root 后，当前“root absent”规则会允许重新执行；没有 root 外的 durable
  bootstrap claim 可证明这是 replacement attempt。
- `attempt-lock.json` 与 `attempt-result.json` 也没有 exact status/phase/error
  schema，无法唯一审计 interrupted/failed/completed 状态。

必须修复：
- 在 root 创建前，于不可替换的 canonical attempts namespace 建立并 fsync
  sibling bootstrap claim。
- 冻结 no-symlink、no-replace、file fsync、parent fsync、terminal receipt
  和 interrupted recovery 语义。
- 任何删除、缺失、替换或 identity drift 都必须 fail closed，不能恢复为
  “首次 attempt”。

### P1-5 A-1-0 precedence makes the registered A-1-1 failure unreachable

位置：
- execution plan `:679-710`

问题：
- A-1-0 先要求 `A/B/P final path/SHA difference count zero`。
- A-1-1 随后才要求 poison identities 与 `final A/P difference zero`。
- 若 unconsumed poison 真影响任何 output，A/P 必然不同，A-1-0 会先失败为
  `Aminus1_authority_or_source_failed`；注册的
  `Aminus1_outcome_boundary_violated` 无法成为 first-failure classification。
- 这也把 deterministic A/B failure 与 poison A/P failure 混在同一 condition。

必须修复：
- A-1-0 只拥有 source/authority 与 canonical A/B determinism。
- A-1-1 独占 poison attestation、consumed mismatch、forbidden access 与
  A/P comparison。
- 明确 A/P mismatch 时 formal package 如何保留负面分类，而不是因要求
  final A/B/P byte equality 而无法形成结果。

### P1-6 Three-stage comparison evidence is self-referential and not executable

位置：
- execution plan `:340-374`
- execution plan `:388-413`
- execution plan `:609-612`

问题：
- `execution_evidence.json` 要包含 A/B/P path sets、per-file comparison rows
  和 difference counts，但它自身也属于 exact 17 paths。
- 若 comparison row 包含自身 SHA，则存在不可解自引用；若排除自身，合同没有
  注册 exclusion/projection。
- Final rewrite order先写 `execution_evidence.json`，随后还会改写 outcome
  ledger、gates、summary、classification 和 manifest。此前写出的 evidence
  不可能同时包含这些最终字节的 SHA，除非再次改写，违反已冻结顺序。
- `run_manifest` 只明确 self-exclusion，没有定义 execution evidence、
  outcome ledger、gate 与 summary 的非自引用 comparison domain。

必须修复：
- 为 preseal/pending/final 分别冻结 exact comparison projection 和排除项。
- 定义哪一个 sibling terminal receipt 对最终 17-path tree 做外部 SHA
  closure，并给出无自引用的计算顺序。
- 为每一阶段冻结 exact path set、comparison-row schema 和 expected count。

### P1-7 The 17-output package still lacks exact JSON and sibling schemas

位置：
- execution plan `:376-386`
- execution plan `:587-657`

问题：
- 七个 JSON 合同仅用 “Contains” 描述，没有 exact ordered logical schema、
  nested keys、types、enums、stage-dependent null sentinels 或 unknown-key
  rejection。
- `attempt-lock.json`、`attempt-result.json` 和
  `poison-attestation.json` 没有 exact schemas。
- `attempt-result.json` 被赋予 sibling SHA-binding 和 terminal proof 权限，
  但没有规定 path rows、tree hash、status、phase、error、exit code 或
  missing artifact 的表示。
- 因而多个字节级不同、审计能力不同的实现都满足文字描述，无法达到用户要求的
  “17 outputs/schemas 足以唯一实现”。

必须修复：
- 对每个 JSON/sibling artifact 冻结 exact keys、nested schemas、types、
  enums、sentinels、sort 和 unknown/missing-key rejection。
- 明确 stage-specific ledger/evidence 值，以及 terminal result 如何绑定
  lock、attestation 和三个 final trees。

### P1-8 Task verification contract still says 14 artifacts

位置：
- task `.workflow/tasks/0830T002.md:71-82`
- execution plan `:388-413`

问题：
- Task 的 review history 声称 Revision 2 已补齐 exact 17-output schemas，
  但 verify 仍要求 `Exact A/B/P 14-artifact closure`。
- Task 与 plan 对正式 evidence namespace 的 cardinality 不一致，QA 与
  implementation 无法同时满足两份冻结合同。

必须修复：
- Task 改为 exact 17-path closure、16-row self-excluding manifest，并同时
  注册三个 required sibling artifacts 的验收边界。

### P2-1 `mismatch_reason` simultaneously means first failure and `multiple`

位置：
- execution plan `:555-585`

问题：
- Contract 说 `mismatch_reason` 是按给定顺序的 first failed exact flag，
  但允许值又包含 `multiple`。
- 当两个或以上 exact flags 同时失败时，无法确定输出 first failure 还是
  `multiple`。

必须修复：
- 二选一：永远输出 first failure；或先计算 failure count，超过一个时固定
  输出 `multiple`。同时冻结 precedence。

### P2-2 A-1-3 condition-row short-circuit is not frozen

位置：
- execution plan `:631-647`
- execution plan `:659-672`
- execution plan `:734-765`

问题：
- Later gates 的 `NOT_EVALUATED` 已定义，但同一个 A-1-3 gate 内三个顺序
  condition 的 short-circuit 未定义。
- 当 cluster count 为零或 `<30` 时，maximum share 为 null；合同没有说明
  date/share condition rows 的 `status/passed/actual` 是 evaluated false、
  NOT_EVALUATED，还是 null comparison failure。
- 分类文本虽然可判为 not estimable，gate artifact 的 exact bytes 仍不唯一。

必须修复：
- 冻结 A-1-3 内部 sequential condition status 与 null-share row 的 exact
  sentinels，并增加 zero/count/date/concentration hostile cases。

### P2-3 `epoch_support.csv` claims exact authority fields but specifies a subset

位置：
- execution plan `:473-485`
- authority runner
  `examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py:1005-1041`

问题：
- Plan 说使用 authority epoch-ledger “exact ordered fields”，但列出的 CSV
  到 `grid_exact` 即结束。
- Authority row 还含 raw onset、edge omission、retained/suppressed、
  candidate ID 和 dependence cluster fields。
- Successor 可以合理地写 authority 全字段，也可以只写 plan 列表；两者
  manifest 和 schema gate 均不同。

必须修复：
- 明确 `epoch_support.csv` 是 exact projection，并列出 projection authority；
  或列出全部 authority fields。禁止使用“exact authority fields”指代子集。

## Closure Assessment

Revision 2 已真实关闭：
- checkpoint-exact action/memory/veto 顺序与 TTL boundary；
- prestate 不含 `t`；
- confirmation 在 window close 完成，并分离 first-additional/window-close；
- primary/sensitivity non-rescue；
- retained/status/counter/support slice identities；
- poison 29/15/435 expectations；
- CSV serialization、numeric basics、hostile-test minimum。

但以下 load-bearing contract 仍未闭合：
- detector 顺序唯一性；
- feature input authority；
- core-close state conservation；
- durable one-shot claim；
- A-1-0/A-1-1 ownership；
- non-self-referential A/B/P finalization；
- exact JSON/sibling schemas；
- task/plan artifact cardinality identity。

## Review Decision

结论：
- **FAIL / 不可冻结**

计数：
- **P0/P1/P2/P3 = 0/8/3/0**

执行锁：
- 29-cache execution lock **保持关闭**。
- Future outcomes、A0、live/private/order lock **保持关闭**。

释放条件：
- 修订 idea、plan、task 后进行新的独立 hostile review。
- 只有新一轮达到 `0/0/0/0`，才允许冻结 SHA、实现 successor 或启动 formal
  one-shot attempt。
