# 0830T002 Hostile Plan Review Round 1

日期：
- 2026-08-30 17:27 CST（星期日）

审查角色：
- 独立 hostile scientific-contract reviewer

候选对象：
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `ad50bcf90d2dfacc2d0da0734d54bd758f2673ce`
- idea SHA256
  `a9c73fc75404dec738deace25bf3c8ddf3c3c2c70c35605a41c176eeb10d4ba5`
- plan SHA256
  `a70b984c056b741d32919e69e4aa1b6e952875463bce4e61c1fbe10f28d86e7f`

审查边界：
- 未运行 29-cache。
- 未读取 future outcomes。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 仅审查 pre-execution contract 是否可以唯一、因果、fail-closed 地实现。

基础核对：
- Candidate commit、branch 和两份文档 SHA256 均精确匹配。
- 初始 working tree 干净。
- Fixed-epoch baseline verifier 以
  `--check-working-tree --require-tags` 返回 `PASS`。
- Suppression authority 六个冻结文件与 `f06eb5cb` 保持一致。
- Research-kit tag peel 到 `45afe244`，suppression tag peel 到 `f06eb5cb`。
- Future outcome、A0、live/private/order 和 29-cache execution lock 均未释放。

## Severity Summary

- P0: 0
- P1: 8
- P2: 2
- P3: 0

## Findings

### P1-1 Authority binding does not bind the executed primitive path

位置：
- execution plan `:56-63`
- execution plan `:104-105`
- execution plan `:140-141`
- idea `:104-110`

问题：
- Plan 只说 successor runner `may call` frozen source/action/memory/epoch
  helpers。验证 authority file/blob/AST 存在，并不能证明正式 detector
  实际调用这些 helpers；实现仍可重写 event mask、action、memory、TTL
  或 epoch semantics。
- `new explicit evidence update` 没有在 successor contract 中冻结为
  `trade_total>0`、`bid_depletion+ask_depletion>0`、`ofi_abs>0`，也没有
  要求 checkpoint-exact masks 等于 authority 输出。
- 因而两个实现都可声称绑定 authority，却产生不同 trigger set。

必须修复：
- 把所有继承 primitive 改为 direct-call authority，或冻结 successor
  wrapper AST 并证明其输出逐 checkpoint 等于 authority callable。
- 明确 event masks、action enum、raw preflight、TTL memory 和 epoch ledger
  的唯一 callable/source/blob/AST authority。
- 任何 source preflight violation 必须在 action、memory、trigger 前失败。

### P1-2 Same-checkpoint causal ordering and TTL boundary are not frozen

位置：
- idea `:104-114`
- execution plan `:129-141`
- execution plan `:153-159`

问题：
- 合同没有规定 checkpoint `t` 内的唯一顺序：source preflight、new
  action、neutral overwrite、memory clear/refresh/expiry、prestate
  evaluation、anchor-time veto、support count 和 thinning。
- 这会直接改变以下边界结果：
  - secondary 在 `t` 收到 `NEW_NEUTRAL`，是否先清除旧 opposite memory；
  - secondary 在 `t` 收到 `NEW_-d`，veto 读取更新前还是更新后 memory；
  - memory age 精确等于 `100ms` 时是否仍 fresh；
  - segment reset 与同 timestamp action 谁优先。
- Plan 只给出参数，没有冻结这些因果操作的先后关系。

必须修复：
- 注册 checkpoint-exact state transition order。
- 明确 prestate 是 `t-120ms ... t-20ms` 六点且不含 `t`。
- 明确 veto 使用 action/reset/expiry 处理后的 `t` 时刻 memory，并冻结
  TTL `age <= 100ms` 或其他唯一边界。
- 增加 same-timestamp neutral/opposite/expiry/reset hostile cases。

### P1-3 Confirmation completion time and retained-trigger semantics are ambiguous

位置：
- idea `:116-134`
- idea `:150-152`
- execution plan `:153-159`
- execution plan `:186-190`

问题：
- Confirmation 要求整个 `(t,t+200ms]` 内无 opposite update，所以在窗口
  关闭前不能因较早的 additional leader update 而完成确认。
- Ledger 同时要求单一 `confirmation_event_seq` 和 `confirmation_ts_ns`，
  但没有说明它们表示：
  - 第一笔 additional `NEW_d`；
  - 最后一笔 additional `NEW_d`；
  - 或窗口关闭 checkpoint。
- 也没有明确 confirmation window 可否越过 `[15s,45s)` core close。
- 这些选择会改变 causal confirmation timestamp、cancel status 和
  exposure/ledger identity。

必须修复：
- 冻结 candidate 只能在 `t+200ms` 完成判定。
- 分开记录 `first_additional_same_update_*` 与
  `confirmation_window_close_*`，或定义现有字段的唯一语义。
- 明确 core 只约束 trigger time，还是也约束全部 confirmation checkpoints。
- 保持已注册的“先 veto、再 thinning、仅确认 retained trigger、失败不允许
  later replacement”顺序，并为边界写 hostile tests。

### P1-4 Slice identity does not cover the state that thinning actually freezes

位置：
- execution plan `:209-246`
- execution plan `:153-159`
- idea `:150-152`

问题：
- Slice 只比较 confirmed candidate identity 和 action/memory support。
- 因为 thinning 在 confirmation 前发生，full analysis 与 slice analysis
  可以保留不同的 earliest trigger，但两者都失败 confirmation；此时
  confirmed identity 均为空，slice gate 仍会通过。
- 当前 tuple 也不比较 retained trigger status、confirmation timestamp、
  support count、cancel reason、veto/suppressed counters 或 epoch
  disposition。
- 这不能证明 reset/slice 对实际 suppression state machine 不变。

必须修复：
- 增加 exact retained-trigger identity：
  `(capture,variant,epoch,direction,candidate_ts,event_seq,cluster)`;
- 逐 comparable epoch 比较 confirmation status/timestamps、cancel reason、
  same-direction support、opposite counts、veto/suppressed counts；
- 比较 exact epoch disposition identity 和对应 hashes；
- 所有 identity tuple 必须冻结 typed sorting、canonical serialization、
  count 和 SHA256。

### P1-5 Poison authority and A/B/P stage closure are not executable

位置：
- execution plan `:248-269`
- execution plan `:298-314`
- execution plan `:271-292`

问题：
- `Build B: fresh canonical source caches` 没有说明是同一 29 个 immutable
  cache bytes 加 fresh output root，还是重新生成一套 caches。
- Build P 没有冻结：
  - poisoned cache/output/attestation exact roots；
  - unconsumed field set 等于 allowed-minus-consumed；
  - dtype/shape preservation；
  - 每个 source/poison value hash；
  - 29 caches、15 fields、435 changed instances、0 consumed mismatch；
  - attestation schema/path/SHA authority。
- Poison attestation 不在 14 项 Required Outputs 中，也没有注册独立 required
  external artifact。
- `preseal outputs` 的 exact path namespace、pending stage、comparison
  receipt 和 finalization order 未定义；动态 summary/ledger/manifest 在何时
  出现也未定义。
- 因此 final equality 可以由不同 staging 实现得到，且无法独立证明完整
  poison pipeline 和 preseal/pending/final closure。

必须修复：
- 定义 A/B 使用完全相同 canonical cache bytes，仅 output roots 不同。
- 注册 exact A/B/P/poison-cache/attestation roots 和 immutable source
  inventory identity。
- 冻结 preseal、pending、final 三阶段 path sets、SHA comparison rows、
  difference counts 和 finalizer transitions。
- 将 poison attestation 纳入 Required Outputs，或冻结为 required sibling
  artifact，并要求 A/B/P ledger 引用同一 attestation SHA。

### P1-6 Required Outputs are filenames, not exact evidence schemas

位置：
- execution plan `:163-207`
- execution plan `:271-292`
- execution plan `:298-363`

问题：
- 除 trigger ledger 的字段名外，其余 13 个 artifacts 没有 exact schema。
- Trigger ledger 也没有冻结 field types、sentinels、row sort、
  `confirmation_status` values、`candidate_id` formula 或 canonical JSON
  encoding。
- 没有定义以下 load-bearing evidence 的输出位置和 row grain：
  - anchor-time veto 与 same-key suppression counters；
  - per-variant/per-direction action and trigger conservation；
  - gate condition rows和 `NOT_EVALUATED` sentinels；
  - primary/sensitivity aggregates；
  - source/poison/determinism evidence；
  - exact support/slice identity hashes。
- 实现无法据此生成唯一的 14-output package，QA 也无法独立重建 gates。

必须修复：
- 为 14 项逐一冻结 ordered fields、types、row grain、sort keys、empty/null
  sentinels、canonical serialization 和 cross-artifact conservation。
- Manifest 必须校验 exact 14-path namespace、13 self-excluding rows、size 和
  SHA256，拒绝 missing/extra/duplicate artifacts。

### P1-7 Gate and classification precedence are incomplete

位置：
- execution plan `:294-363`

问题：
- 虽然写了 sequential gates，但没有冻结 later gate
  `NOT_EVALUATED` representation。
- 没有定义 negative、NaN、inf、wrong type、zero denominator 和 exact-zero
  语义。`maximum single-date cluster share` 在 zero clusters 时尤其不唯一。
- Raw source invalid contribution 被放入 A-1-2，而 A-1-0 名为 Authority and
  Source；source preflight defect 的唯一 first-failure classification 不清楚。
- A-1-2 没有说明任何一个 sensitivity 的 detector-integrity violation 是否
  使全任务失败。
- A-1-3 的三个不同失败原因都映射为
  `structural_support_not_estimable`；例如有大量 clusters 但单日占比过高，
  是 concentration failure，不是数值不可估计。

必须修复：
- 冻结逐 gate condition schema 和 first-failure precedence。
- Later gates 必须保留完整 condition rows，但使用
  `status=NOT_EVALUATED, passed=null, actual=null`。
- Source preflight failure 唯一归 A-1-0；post-preflight derived detector
  defects归 A-1-2。
- 冻结 zero/nonfinite/type arithmetic，并区分 support-count failure、
  date-coverage failure 和 concentration failure 的准确分类或统一但准确的
  registered-negative claim。
- 明确 sensitivities 总是报告但绝不进入 A-1-3，任何 variant 的执行完整性
  缺陷仍在 A-1-2 fail closed。

### P1-8 Post-Build-A immutability is policy-only and cannot be audited

位置：
- execution plan `:368-410`
- task `.workflow/tasks/0830T002.md:65-85`

问题：
- Plan 未冻结 exact Build A/B/P/finalize commands、output roots、attempt ID
  或 clean-worktree identity。
- 没有在第一次 Build A cache access 前原子写入 immutable attempt/lock
  receipt，记录 idea/plan/task/runner/tests commit、blob、SHA、CLI 和 roots。
- A-1-0 只绑定 baseline authority，没有要求 final artifacts 绑定 successor
  implementation commit 和 runner/tests bytes。
- 因而执行者可以在 Build A 后修改 successor code/tests，再声称 final
  package仍符合 baseline authority；Git 历史或聊天规则无法证明没有 repair。

必须修复：
- 注册唯一 formal CLI 和 one-shot attempt root。
- 在任何 cache read 前原子创建 no-replace lock receipt，并拒绝已有 attempt、
  非空 roots、dirty worktree 或 identity mismatch。
- Finalizer 和 gate contract 必须绑定同一 successor commit、idea/plan/task/
  runner/tests blob/SHA/AST identities。
- Build A 一旦开始，无论成功、失败或中断都不得重新使用新 attempt 来替代。

### P2-1 Trigger/veto/suppression ledger vocabulary is internally inconsistent

位置：
- idea `:104-114`
- execution plan `:153-159`
- execution plan `:194-207`

问题：
- Idea 把“无 anchor-time opposition”写成 trigger 成立条件；plan 又要求
  `anchor-time vetoed triggers` counters。
- 因而 veto 前对象究竟叫 raw leader onset、trigger attempt 还是 trigger
  candidate 不明确。
- `multiple` cancel reason 也没有冻结组成原因的 canonical order，而
  same-key suppressed rows完全不落 ledger，只有未定义 schema 的 counters。

必须修复：
- 明确分层命名和守恒：
  `raw leader onset -> veto-admitted trigger -> retained/suppressed ->
  confirmed/cancelled`。
- 为每层冻结 exact per-date/variant/direction counts、互斥/重叠关系和
  conservation equations。
- 为 `multiple` 冻结 ordered reason atoms，或改为独立 boolean columns。

### P2-2 No frozen hostile-test minimum exists

位置：
- execution plan `:368-379`

问题：
- Pre-execution lock 只要求 `focused hostile tests pass`，但没有注册任何
  minimum cases。
- 当前关键歧义都可能在没有对应 mutation test 的情况下被实现并“通过测试”。

必须修复：
- 至少冻结以下 hostile matrix：
  - exact threshold、TTL、prestate、core-open/core-close 和 confirmation
    endpoint boundaries；
  - same-timestamp neutral/opposite/reset/expiry ordering；
  - earliest failed retained trigger suppresses later confirmed-capable trigger；
  - slice retained/status/counter/hash mutations；
  - source invalid与 unconsumed poison mutations；
  - A/B/P missing/extra/byte mutation、attestation mutation；
  - 14 schemas、typed sentinels、sort/hash和 manifest self-exclusion；
  - zero/nonfinite/`NOT_EVALUATED` gate precedence；
  - sensitivity cannot rescue primary；
  - post-Build-A identity/attempt/output mutation fail closed。

## Review Decision

结论：
- **FAIL / 不可冻结**

计数：
- **P0/P1/P2/P3 = 0/8/2/0**

执行锁：
- 29-cache execution lock **保持关闭**。
- Future outcomes、A0、live/private/order lock **保持关闭**。

释放条件：
- 修订 idea/plan/task 后进行新的独立 hostile review。
- 只有新一轮达到 `0/0/0/0`，才允许冻结 plan SHA、implementation 和 formal
  execution command。
