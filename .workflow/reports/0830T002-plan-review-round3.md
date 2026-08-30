# 0830T002 Hostile Plan Review Round 3

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `5930d40b5d9c5bddd6ecb7ba918b139b9d605ee4`
- idea SHA256
  `83aa707ecc18e7de87cc0606fab4da522f8592c9ebeaa26f6db10f17c096b386`
- plan SHA256
  `14daa22721b221267cdae3add0645acef1225e37a1e2e469ee513ad58b5f0706`
- task SHA256
  `e06baa1c2a5f735c818d5871beafa2f106b7dae6511030418e8192d923950b76`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Registered `build_features` authority commit、blob、file SHA 和 callable
  AST SHA 均独立复算匹配。
- `45544ecc` 是当前 HEAD 的 ancestor。
- Revision 3 未修改 frozen authority runner/tests。

## Severity Summary

- P0: 0
- P1: 6
- P2: 3
- P3: 0

## Round 2 Closure Matrix

| Round 2 finding | Round 3 status | 结论 |
|---|---|---|
| P1-1 detector order contradiction | CLOSED | idea/plan 现唯一规定 core -> confirmation edge -> veto -> thinning -> confirmation |
| P1-2 `build_features` callable authority | PARTIAL | commit/blob/file/AST 已精确冻结，但 formal evidence 尚未绑定每次调用的输入和返回值使用 |
| P1-3 core-close disposition | CLOSED | `confirmation_edge_omitted` 已进入唯一守恒边，equality 与 `+20ms` 已冻结 |
| P1-4 one-shot durability | PARTIAL | hard-link/fsync 顺序已补齐，但 tracked claim 可无痕恢复，terminal evidence 也未 no-replace seal |
| P1-5 A-1-0/A-1-1 ownership | CLOSED | A/B `RAW_11` 独占 A-1-0，A/P `RAW_11` 独占 A-1-1 |
| P1-6 self-referential closure | CLOSED | RAW_11/SEALED_15/EVIDENCED_16/FINAL_17 projection 可无自引用实现 |
| P1-7 exact JSON/sibling schemas | PARTIAL | object keys 已冻结，但若干 list/domain/value authority 与 post-gate schema ownership 仍不唯一 |
| P1-8 task 14/17 mismatch | CLOSED | task 已改为 17 paths、16-row manifest 和 exact siblings |
| P2-1 mismatch precedence | CLOSED | 永远输出第一个失败项，`multiple` 已删除 |
| P2-2 A-1-3 short circuit | CLOSED | count/date/share 内部 NOT_EVALUATED 语义已精确冻结 |
| P2-3 epoch projection | CLOSED | 已明确为 successor exact projection，不再声称输出 authority 全字段 |

## Findings

### P1-1 Tracked armed claim can be restored without an auditable trace

位置：
- idea `:335-346`
- execution plan `:320-353`
- execution plan `:1078-1103`

问题：
- Formal runner 将 tracked `armed` hard-link 为 untracked `claimed`，再删除
  `armed`。这能阻止普通重复调用，但不能证明只发生过一次。
- 操作者可执行等价于：
  - 从 implementation tag 恢复 `0830T002.armed.json`；
  - 删除 untracked `0830T002.claimed.json`；
  - 删除 ignored attempt root。
- 以上操作可使 HEAD、working tree、armed/claimed/root 三个 preflight 条件
  精确恢复到首次执行前状态；`git restore` 不产生 commit 或可由 QA 强制读取
  的审计记录。
- Plan 仅声明 restoring/deleting “outside protocol”，但没有 append-only
  consumption receipt、consumption commit/tag 或外部 durable ledger。因此
  “interrupted or deleted attempt cannot silently appear first”这一 claim
  不成立。

必须修复：
- 在任何 cache read 前，将 claim consumption 写入不可回滚的审计 authority，
  例如由 controller 创建并推送/封存 consumption commit/tag，或使用任务外的
  append-only receipt。
- Preflight 必须验证该 authority 中不存在既往 consumption，而不只检查当前
  working-tree 文件状态。
- 增加 restore-armed + delete-claimed + delete-attempt-root hostile case。

### P1-2 Final evidence and `attempt-result` are not durably no-replace sealed

位置：
- execution plan `:369-385`
- execution plan `:819-864`
- execution plan `:1096-1103`

问题：
- Claim 与 `attempt-lock.json` 注册了 hard-link no-replace publish 和 parent
  directory fsync。
- A/B/P files、renamed poison attestation、dynamic evidence、manifests 和
  `attempt-result.json` 没有同等 publication contract。
- Step 12 只写 “publish and fsync”，没有冻结：
  - same-directory temporary；
  - hard-link/O_EXCL no-replace publish；
  - parent-directory fsync；
  - final tree directories 在 terminal receipt 前全部 fsync。
- Formal result 位于 ignored mutable tree。操作者可一致地重写 A/B/P 和
  `attempt-result.json`；tracked claimed state只证明 attempt 被消费，不绑定
  原始 terminal result SHA。
- Power loss 也可能留下 `status=COMPLETED` 的 receipt，但部分 FINAL_17
  directory entries 尚未 durable。

必须修复：
- 对全部 FINAL_17、attestation 和 terminal result 冻结 atomic write、
  file fsync、no-replace publish 和 parent fsync 顺序。
- 在 attempt root 外建立不可修改的 terminal tree/result SHA receipt。
- `attempt-result` 一旦存在，任何 replacement、unlink 或 tree drift 必须可由
  verifier 独立检测。

### P1-3 `build_features` evidence does not bind call inputs or returned-value use

位置：
- execution plan `:74-98`
- execution plan `:367-378`
- execution plan `:651-719`
- execution plan `:1058-1061`

问题：
- Registered commit/blob/file/AST values本身正确。
- Formal output只记录每个 callable 的一个 `direct_call_count`。它不记录：
  - build label A/B/P/slice；
  - resolved input cache path；
  - canonical或poison input SHA；
  - returned feature identity；
  - detector是否实际消费该返回值。
- 因此错误实现可调用 exact callable 以满足 count，同时让 Build P 使用
  canonical path、让 slice 使用 full features，或丢弃返回值后使用另一套
  features。
- Expected count 还是全局公式，但 17 outputs 不包含
  `total artificial slice build count` 的独立 ledger；QA 无法从正式证据重算
  该 count。
- Build B 被要求 fresh subprocess，Build P 没有 fresh-process 或
  no-cross-build-state 条件，进一步扩大 false A/P equality 风险。

必须修复：
- 增加 exact feature-call ledger，至少绑定
  `(build_label,capture/slice,input_path,input_sha,output_feature_sha)`。
- 冻结每个 build/cache/slice 的 expected call identity，而不只冻结总数。
- Hostile tests 必须证明 wrong-root、ignored-return、canonical-path-for-P、
  full-feature-for-slice 和 cross-build memoization 均 fail closed。

### P1-4 `forbidden_access_count` has no measurement authority

位置：
- execution plan `:121-123`
- execution plan `:740-749`
- execution plan `:911-924`
- execution plan `:998-1004`

问题：
- Plan 声明 permitted consumed fields exact，并在 outcome ledger 中输出
  `forbidden_access_count` 和三个 accessed booleans。
- 但没有注册 instrumented loader、field-access ledger、restricted feature
  interface、AST ban 或 runtime guard来产生这些值。
- Poison A/P equality只能证明 unconsumed value没有改变 RAW_11；它不能证明
  forbidden field从未被读取。读取后不使用、只用于分支外诊断、或读取后得到
  相同输出，均可通过 poison comparison。
- 因而 runner 可以自报 `forbidden_access_count=0`，A-1-1 仍 PASS。

必须修复：
- 除 poison authority 外，正式 detector只能获得 exact consumed-field view；
  所有 raw field access必须经可计数、fail-closed 的 authority loader。
- 冻结 access ledger 的 row schema、allowed caller、field、cache和count
  conservation。
- 增加“读取 poison field但丢弃结果” hostile case，要求 A-1-1 FAIL。

### P1-5 A-1-2 schema ownership conflicts with the finalization order

位置：
- execution plan `:369-385`
- execution plan `:689-699`
- execution plan `:866-889`
- execution plan `:926-946`
- execution plan `:1006-1018`
- execution plan `:1065-1076`

问题：
- A-1-2包含 `schema_violation_count=0`，并声明 malformed SHA 是 A-1-2。
- Hostile minimum又要求验证全部17项 JSON/CSV/manifest schema。
- 但 gate/summary/classification 在 Step 7 已经生成；execution evidence、
  manifests 和 attempt result 直到 Steps 9-12 才存在。
- 如果后续 artifact 有 schema、sorting、manifest 或 hash violation，当前
  gate不能记录它；重写 gate又会改变 SEALED_15、execution evidence、
  manifests 和 FINAL_17，形成新的 finalization cycle。
- 当前合同没有区分：
  - 可在 gate 前计算的 scientific/raw schema defects；
  - gate 后只能导致 terminal execution failure 的 package defects。

必须修复：
- 精确限定 A-1-2 `schema_violation_count` 的 path/domain ownership，例如只
  覆盖 gate前的 RAW_11 和内存 scientific payload。
- execution evidence、manifest、sibling/result schema failure 应由外部
  terminal verifier 独占，并不得改写 scientific classification。
- 为每个17-path和三个 sibling逐项注册 first-failure owner。

### P1-6 Exact JSON schemas still leave load-bearing list domains unspecified

位置：
- execution plan `:648-719`
- execution plan `:772-781`
- execution plan `:796-864`
- execution plan `:866-871`

问题：
- Exact object keys已冻结，但以下值域/顺序仍未注册：
  - `authority_binding.tracked_files` 的 exact path set/count/order；
  - `authority_binding.callables` 的 exact callable set/order，以及除
    `build_features` 外各 callable 的 expected direct call count；
  - 四个 `Comparison.domain` 的 exact strings；
  - `RootRow.label` 的 exact A/B/P enum和 `root_rows` order；
  - `classification.gate_statuses` 是否严格对应 gate order；
  - `attempt-result` 所称 “all sibling artifacts” 是否明确排除自身。
- 多个字节级不同、证据覆盖不同的实现仍可满足当前 object schema，导致
  manifest/tree SHA 与 QA重算目标不唯一。

必须修复：
- 列出 exact tracked-file和 callable rows、count、sort keys与 expected
  call counts。
- 冻结 comparison domain strings、root labels/order、gate-status projection
  和 sibling closure domain。
- 对所有 list增加 duplicate rejection和 exact cardinality。

### P2-1 Comparable-epoch ceiling arithmetic is not integer-exact

位置：
- execution plan `:219-225`

问题：
- `ceil((actual_start_ts_ns + 122s)/60s)` 没有禁止 binary-float arithmetic。
- Nanosecond timestamps远超双精度整数精确范围；在 epoch boundary 附近，
  float `ceil` 与 integer ceiling division可能给出不同 epoch ID。

必须修复：
- 冻结为：
  `(actual_start_ts_ns + 122_000_000_000 + EPOCH_NS - 1) // EPOCH_NS`。
- 增加 exact-boundary、`-20ms`、equality 和 `+20ms` hostile cases。

### P2-2 Time strings and terminal row order are not canonical

位置：
- execution plan `:705-708`
- execution plan `:810-817`
- execution plan `:851-871`

问题：
- `started_at_utc` 与 `finished_at_utc` 仅定义为 nonempty ASCII string，
  没有冻结 RFC3339格式、UTC `Z`、fraction precision或时区偏移规则。
- `root_rows` 没有明确按 A/B/P label还是 path排序。
- 这不会改变 detector结果，但会造成 sibling bytes和QA parser实现不唯一。

必须修复：
- 注册 exact UTC timestamp format和precision。
- 注册 `root_rows` exact labels、order和count=3。

### P2-3 Empty maximum-memory-age sentinel remains unspecified

位置：
- execution plan `:477-489`
- execution plan `:502-513`

问题：
- Common CSV规则只说 N/A identity使用空字符串。
- `channel_action_by_date.maximum_memory_age_ms` 在一个 date/channel 没有
  established memory时，可能合理输出 empty、`-1` 或 `0`。
- 三者会产生不同 CSV、RAW_11 SHA 和 schema interpretation。

必须修复：
- 冻结无 memory时的唯一 sentinel，并明确该字段是 non-negative int、
  `-1` sentinel或 nullable/empty三者之一。

## Closed Contract Areas

本轮确认已闭合：
- 唯一 detector 顺序；
- confirmation-edge omission与守恒；
- registered `build_features` commit/blob/file/AST identity；
- A/B 与 A/P gate ownership；
- RAW_11、SEALED_15、EVIDENCED_16、FINAL_17 无自引用 projection；
- task的17-path identity；
- mismatch first-failure precedence；
- A-1-3 sequential short circuit；
- epoch-support exact successor projection；
- primary/sensitivity non-rescue。

## Review Decision

结论：
- **FAIL / 不可冻结**

计数：
- **P0/P1/P2/P3 = 0/6/3/0**

执行锁：
- 29-cache execution lock **保持关闭**。
- Future outcomes、A0、live/private/order lock **保持关闭**。

释放条件：
- 修订 idea、plan、task 后进行新的独立 hostile review。
- 只有新一轮达到 `0/0/0/0`，才允许冻结 SHA、实现 successor 或消费 tracked
  armed claim。
