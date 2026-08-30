# 0830T002 Hostile Plan Review Round 4

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `fe9a08c6ba01c5b07529d9ff903b81f40f8a4ae6`
- idea SHA256
  `605ec18a002bb7aa7d213b1143dc0cbb2585b35e82b25a33c3b11bbd0734abf8`
- plan SHA256
  `effe37cf75d23bef18bcf47585099c2b823f13754bde481170489b27d2ec9d1e`
- task SHA256
  `62b76bb4a0f70d539cfd711f7c54315d9e443900b270a0ce926b0f6a798e06de`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Revision 4 只修改 docs/task/workflow记录，未修改 frozen authority。
- `git diff --check 5930d40b..fe9a08c6` 通过。

## Severity Summary

- P0: 0
- P1: 5
- P2: 2
- P3: 0

## Round 3 Closure Matrix

| Round 3 finding | Round 4 status | 结论 |
|---|---|---|
| P1-1 reversible armed claim | PARTIAL | consumption commit/tag使普通 restore/delete 不再无痕，但本地 refs/commits仍可删除，Git object durability未冻结 |
| P1-2 terminal no-replace seal | CLOSED | FINAL_17、attestation、result已注册 write-once、file fsync、hard-link no-replace和parent fsync |
| P1-3 feature-call input/output evidence | PARTIAL | per-call schema和consumer hash已补齐，但 formal work inputs未进入 exact attempt/terminal closure |
| P1-4 forbidden access authority | PARTIAL | `np.load` proxy已补齐，但 forbidden attempt无法按当前成功schema形成A-1-1结果，且proxy可被其他raw readers绕过 |
| P1-5 post-gate schema ownership | CLOSED | scientific gates与terminal-verifier ownership已明确分离 |
| P1-6 list/domain schemas | CLOSED | tracked files、callables、domains、roots、gate statuses及duplicate rules已冻结 |
| P2-1 integer ceiling | CLOSED | 已改为整数 ceiling division并注册边界测试 |
| P2-2 timestamp/root order | CLOSED | UTC microsecond-Z与A/B/P顺序已冻结 |
| P2-3 maximum-memory sentinel | CLOSED | 无memory唯一为 `-1` |

## Findings

### P1-1 Consumption and terminal Git authority is still locally erasable and incompletely fsynced

位置：
- idea `:335-346`
- execution plan `:357-429`
- execution plan `:462-464`
- execution plan `:1337-1342`

问题：
- Consumption commit/tag与terminal commit/tag解决了普通
  restore-armed/delete-claimed/root绕过。
- 但 preflight只检查 consumption/terminal tags当前不存在。操作者仍可：
  - 删除两个local tags；
  - 将branch/HEAD reset回implementation tag；
  - 删除ignored attempt root；
  - 重新执行。
- Plan未要求检查 reflog、unreachable consumption/terminal commits，或将refs
  推送到append-only/protected authority。删除local refs后，正式preflight
  无法区分首次执行与已被擦除的执行。
- Consumption阶段只要求 fsync Git ref/log directories。Git commit/tag还会
  创建 tree、commit和annotated-tag objects，并更新HEAD/index；这些object和
  directory的durability未冻结。若ref/tag在cache read前看似成功但objects未
  durable，掉电后authority可能消失。
- Terminal commit/tag也没有单独注册Git object/ref/log/index fsync顺序。

必须修复：
- 在cache read前将consumption ref推送/写入不可删除的controller authority，
  或要求preflight扫描并拒绝既往reflog/unreachable transition identities。
- 冻结 Git durability configuration/protocol，例如 exact `core.fsync` /
  `core.fsyncMethod`，并验证 commit/tree/tag objects、HEAD/ref、index及相关
  parent directories均durable。
- Hostile tests增加delete-tags + reset-to-implementation + delete-root，以及
  commit/tag object-loss模拟。

### P1-2 Formal `work/` feature inputs are outside the exact attempt and terminal closure

位置：
- execution plan `:125-140`
- execution plan `:431-441`
- execution plan `:491-499`
- execution plan `:748-755`
- execution plan `:969-978`

问题：
- Full和slice inputs被注册在：
  `attempt_root/work/{A|B|P}/{cache_name}/`。
- “Exact attempt children”却没有 `work/`。它只列出lock、A/B roots、
  poison cache/root、attestation和result。
- FINAL_17、attempt-result和tracked terminal receipt也不包含work tree
  manifest或tree SHA。
- FeatureCall只自报每个input path/SHA。若work files未保留，independent
  verifier无法重算；若保留，它们又成为未注册的extra attempt child。
- 这使wrong-root、canonical-for-P、slice input mutation和call-ledger
  fabrication无法由terminal evidence独立复验。

必须修复：
- 二选一并唯一冻结：
  - 将 `work/` 纳入 exact attempt children，并发布 exact path/count/SHA
    manifest与terminal tree closure；
  - 或不materialize work files，直接绑定canonical/poison/sliced authority
    路径，并将所有slice bytes作为正式sealed evidence。
- Independent verifier必须逐FeatureCall重算input SHA并证明path authority。
- 明确work evidence是否永久保留；claim后不得清理未封存inputs。

### P1-3 Forbidden/discarded reads cannot produce the registered A-1-1 package

位置：
- execution plan `:110-123`
- execution plan `:748-761`
- execution plan `:1030-1059`
- execution plan `:1166-1181`
- execution plan `:1320-1325`

问题：
- Proxy遇到unconsumed field时“raises before returning any value”。
- Exact successful ledger同时要求：
  - 每个FeatureCall有valid output和consumer SHA；
  - forbidden count为0；
  - 每call恰好12个CONSUMED_VALUE FieldAccess rows；
  - consumer use count为1。
- Forbidden read一旦raise，builder没有feature output，detector没有consumer
  input，RAW_11也无法完整生成；当前schemas无法表示failed call或forbidden
  attempted-access row。
- 因而该事件实际只能导致interrupted formal attempt，而不能形成计划所注册的
  `Aminus1_outcome_boundary_violated` A-1-1 negative classification。
- 同样，discarded return、alternate feature object或second consumer被写成
  “fails A-1-1”，但没有failure-row schema和继续finalize协议。

必须修复：
- 明确 instrumentation defect属于：
  - 可形成A-1-1正式负面结果；或
  - terminal execution failure。
- 若属于A-1-1，增加 failed FeatureCall/FieldAccess schema、exception code、
  attempted field及缺失 output/consumer的typed sentinels，并规定如何在不伪造
  RAW scientific outputs的情况下finalize。
- 若属于terminal failure，从A-1-1 conditions移除，并由verifier/attempt
  result独占。

### P1-4 `numpy.load` proxy and entry hash do not fully enforce the raw/feature boundary

位置：
- execution plan `:100-123`
- execution plan `:748-761`
- execution plan `:1050-1059`
- execution plan `:1317-1325`

问题：
- “successor runner has zero `np.load` call sites”不能禁止通过
  `open`、`zipfile`、`numpy.lib.npyio.NpzFile`或其他loader读取raw NPZ。
- Proxy只在bound builder调用期间替换 `numpy.load`，无法观察proxy范围外的
  raw reads。
- `feature_output_sha256 == consumer_input_sha256`只在detector entry验证。
  Detector收到mutable numpy arrays后仍可原地修改；entry hash和
  `consumer_use_count=1`不会发现post-entry mutation。
- Independent verifier没有formal runtime trace可证明proxy是唯一raw-value
  access path。

必须修复：
- 对build subprocess建立完整raw-open allowlist/audit hook，禁止除hash reader、
  bound builder proxy和poison authority外的NPZ/raw readers。
- 静态检查至少覆盖 `open`、`Path.open`、zipfile/NpzFile及动态import aliases。
- Builder return进入detector前将arrays设为read-only，并在detector exit再次
  hash；任何mutation fail closed。
- Hostile tests增加alternate loader、alias import、post-entry array mutation
  和proxy uninstall/early restore。

### P1-5 Independent verifier lacks a unique executable and evidence contract

位置：
- execution plan `:389-398`
- execution plan `:426-429`
- execution plan `:969-978`
- execution plan `:1130-1144`
- task `.workflow/tasks/0830T002.md:53-95`

问题：
- Plan只注册verifier文件路径和高层职责，没有冻结：
  - exact verifier CLI/argv/cwd；
  - required attempt root、repo root和tag arguments；
  - success/failure exit codes；
  - stdout/stderr或report artifact schema；
  - verifier是否允许读取 `work/`、poison caches或source caches；
  - exact检查顺序与first failure code。
- Verifier不是17 outputs、attempt siblings或tracked terminal receipt的一部分；
  也没有独立verifier result artifact供QA绑定。
- 因此实现可提供多个行为不同的verifier，均声称“independently recomputes”；
  task中的“独立最终QA”无法调用唯一命令或审计exact result。

必须修复：
- 注册唯一read-only verifier command、arguments、cwd、exit-code table和
  machine-readable result schema。
- 明确verifier禁止读取future outcomes/A0，是否读取source/work/poison NPZ，
  以及每类evidence的authority。
- Verifier result应由QA保存并绑定verifier SHA/blob、terminal tag/head、
  checked roots和所有condition rows。
- Hostile minimum加入wrong CLI/root/tag、missing work evidence、post-seal
  mutation和verifier-self-mutation。

### P2-1 Plan references the wrong normative idea revision

位置：
- idea `:8`
- execution plan `:167-179`

问题：
- Current idea明确为 `Revision: 4`。
- Plan仍写 “idea document's Revision 3 definitions are normative”。
- Revision 4 detector语义目前未改变，因此不是即时scientific divergence，
  但冻结合同引用错误，后续implementation/QA可能绑定revision label或current
  idea SHA两种不同authority。

必须修复：
- 改为 Revision 4，并明确normative authority同时由current idea SHA约束。

### P2-2 Feature-call indexing and full-input filename are not exact

位置：
- execution plan `:125-140`
- execution plan `:748-755`
- execution plan `:1030-1035`

问题：
- `call_index:int`没有冻结从0还是1开始、必须连续还是仅唯一。
- Slice filename已冻结为 `slice_{ordinal:06d}.npz`，但FULL input在
  `.../{cache_name}/` 下的exact filename未给出。
- `analyzed_unit_count`也没有单独给出算术定义，必须从FeatureCall自身推断，
  容易形成count自证。

必须修复：
- 冻结global call index为exact contiguous range及起点。
- 冻结FULL input exact path/filename。
- 将 `analyzed_unit_count` 定义为可从independent work/input manifest重算的
  exact整数和组成项。

## Closed Contract Areas

本轮确认已闭合：
- 唯一detector顺序与confirmation-edge conservation；
- FINAL_17、poison attestation和attempt-result write-once no-replace/fsync；
- RAW/SEALED/FINAL无自引用comparison；
- A/B与A/P gate ownership；
- post-gate terminal-verifier ownership；
- exact tracked-file/callable/domain/root/gate-status lists；
- exact UTC timestamp、`-1` memory sentinel和integer ceiling；
- 17-path task identity与primary/sensitivity non-rescue。

## Review Decision

结论：
- **FAIL / 不可冻结**

计数：
- **P0/P1/P2/P3 = 0/5/2/0**

执行锁：
- 29-cache execution lock **保持关闭**。
- Future outcomes、A0、live/private/order lock **保持关闭**。

释放条件：
- 修订 idea、plan、task 后进行新的独立 hostile review。
- 只有新一轮达到 `0/0/0/0`，才允许冻结 SHA、实现 successor 或消费 formal
  attempt claim。
