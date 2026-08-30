# 0830T002 Hostile Plan Review Round 5

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `1cfe8bcefd2f204c8a77908309b2e88bbdbf33a2`
- idea SHA256
  `cc6322f90b03f32e22e5b52c372263716ae67787d7676491b1fe6b28e352ffa7`
- plan SHA256
  `049c856bb3857d5272906567e78d8bd28d7d77f4aad9314340ae2ef7251f5bf5`
- task SHA256
  `23cc6f515deb437c84bf124698fc8db7af8ccb56e9c157582f6d2b6cb9dfecb8`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Revision 5 只修改 docs/task/workflow 记录，未修改 frozen authority。
- `git diff --check fe9a08c6..1cfe8bce` 通过。

## Severity Summary

- P0: 0
- P1: 3
- P2: 2
- P3: 0

## Round 4 Closure Matrix

| Round 4 finding | Round 5 status | 结论 |
|---|---|---|
| P1-1 locally erasable/incompletely fsynced Git authority | PARTIAL | `core.fsync=all`、reflog/unreachable scan和二次 `git fsck` 已增加；但完整 reflog/object prune 后仍可无痕重开 formal attempt |
| P1-2 formal `work/` inputs outside closure | PARTIAL | `work/`、manifest、terminal receipt closure已补齐；`work_tree_sha256` 的 exact projection 仍不唯一 |
| P1-3 forbidden/discarded reads cannot form registered package | PARTIAL | 已改为 terminal-only；但 `outcome_access_ledger` 的文件级 A-1-1 ownership 与其内部 instrumentation rows 的 terminal-only ownership 冲突 |
| P1-4 raw/feature boundary incomplete | PARTIAL | detector read-only/exit hash已补齐；LOADER 阶段仍没有完整 raw-open allowlist |
| P1-5 verifier executable/evidence contract absent | PARTIAL | 唯一 CLI、cwd、exit codes、result path和13项顺序已补齐；每项 exact result value contract仍未冻结 |
| P2-1 wrong normative revision | CLOSED | idea/plan均为 Revision 5，且绑定 task-frozen idea SHA |
| P2-2 call index/FULL input ambiguity | CLOSED | 已冻结 global zero-based contiguous index、FULL authority paths和 analyzed-unit arithmetic |

## Findings

### P1-1 Reflog/unreachable authority is not durable against complete local Git cleanup

位置：
- execution plan `:468-497`
- execution plan `:1471-1472`

问题：
- Revision 5 要求扫描 refs、reflogs 和
  `git fsck --full --unreachable --no-reflogs`，能阻止“仅删 tag、reset
  branch、删 attempt root”的普通擦除。
- 但这些证据仍全部位于同一个本地 Git object store：
  - reflog 的 reachable/unreachable entries可以被显式 expire；
  - unreachable commit/tree/tag/blob可以随后被 prune；
  - 当 refs、reflogs和相关unreachable objects被一致删除后，
    `git fsck --full`不会报告一个已不存在且不再被引用的对象。
- `core.fsync=all`保证已写对象的掉电持久性，不保证对象永不被后续合法
  Git maintenance删除。
- 因而“consumption authority cannot be erased”和“formal attempt不可替换”
  的绝对 claim 仍不成立。当前 preflight只能发现尚未被完整清理的历史。
- Hostile minimum中的“simulated Git object/ref loss fails `git fsck`”也只对
  dangling ref或部分对象丢失成立；ref、reflog和对象同时清除时可能通过。

必须修复：
- 将 consumption transition写入 formal worktree之外的 append-only /
  protected controller authority，并在cache read前取得其 durable receipt；
  或明确缩小 threat model，不再声称可抵抗本地历史完整擦除。
- 若仍依赖本地 reflog/unreachable，至少冻结
  `gc.reflogExpire=never`、`gc.reflogExpireUnreachable=never`、
  `gc.pruneExpire=never`，并注册显式 expire/prune hostile case；但这仍不能
  替代外部不可删除authority对主动操作者的保护。

### P1-2 Instrumentation defects have conflicting A-1-1 and terminal-only ownership

位置：
- execution plan `:930-940`
- execution plan `:1247-1265`
- execution plan `:1301-1307`
- execution plan `:1453-1454`

问题：
- `feature_calls`、`field_accesses`、`forbidden_access_count` 都是
  `outcome_access_ledger.json` 的正式字段。
- First-failure table同时规定：
  - `outcome_access_ledger.json semantic/schema defects` 属于 A-1-1；
  - `feature-call/field-access/boundary defects` 只属于 terminal verifier，
    scientific classification unchanged。
- 因此同一个缺陷，例如：
  - `feature_calls` 缺行或错序；
  - `field_accesses` 出现第13行；
  - `forbidden_access_count > 0`；
  - feature-call字段类型错误；
  既可被实现解释为 ledger semantic/schema defect并生成
  `Aminus1_outcome_boundary_violated`，也可被解释为 terminal execution
  failure且不生成科学分类。
- A-1-1条件列表虽然删除了三项instrumentation gate rows，但文件级ownership
  仍覆盖这些字段，所以删除并未形成唯一 precedence。

必须修复：
- 将 A-1-1 对 `outcome_access_ledger.json` 的ownership细化到exact字段，
  明确只覆盖future/outcome flags、poison identity和A/P comparison，显式排除
  `FeatureCall`、`FieldAccess`、access-count和consumer-boundary字段；或把
  instrumentation evidence拆为独立 terminal-only artifact。
- 冻结 runtime-detected 与 verifier-detected instrumentation defect 的同一
  terminal result，并增加一个同时触发ledger schema defect与boundary defect
  的precedence hostile test。

### P1-3 Raw-open enforcement starts too late and does not close the LOADER phase

位置：
- execution plan `:105-117`
- execution plan `:126-142`
- execution plan `:1453-1454`

问题：
- `numpy.load` proxy只约束bound `build_features`调用。
- Runtime audit hook只在 `DETECTOR` 阶段拒绝 `.npz/raw open`；静态AST拒绝
  也只覆盖successor detector callables。
- 在 `LOADER` 阶段，worker/wrapper仍持有exact raw path，但合同没有定义：
  - whole-worker audit hook及允许的exact caller/stack；
  - 对 `open`、`io.open`、`zipfile`、`NpzFile`、`mmap`、already-open
    descriptor或preloaded raw bytes的拒绝；
  - loader退出时所有raw handle/object/buffer均已关闭且没有跨阶段引用。
- 因而wrapper可在proxy之外读取并暂存raw值，再把路径引用删除后进入
  DETECTOR；detector audit hook、read-only feature arrays和exit hash都不会
  发现这条旁路。
- “exact raw path is available only to bound build_features + load proxy”
  当前是声明，不是唯一可执行的enforcement contract。

必须修复：
- 在worker启动时安装不可卸载的whole-process raw-open audit hook：
  LOADER仅允许exact bound builder/proxy路径，DETECTOR拒绝全部registered raw
  paths；冻结事件类型、caller authority、计数ledger和first-failure code。
- 对整个successor worker/orchestrator做raw-loader AST/import closure，而不只
  检查detector callables。
- 在LOADER -> DETECTOR边界证明所有raw handles、NPZ objects、file
  descriptors和raw-byte buffers为零；更强的实现是让detector在只接收sealed
  feature payload的独立subprocess中运行。

### P2-1 `work_tree_sha256` has two incompatible canonical projections

位置：
- execution plan `:889-892`
- execution plan `:1055-1058`
- execution plan `:1114-1116`
- execution plan `:1200-1205`

问题：
- `work-manifest.json.rows` 的元素是 `WorkRow`，包含
  `build_label/cache_name/slice_ordinal/path/size_bytes/sha256`。
- 通用tree-hash条款却规定tree hash是排序后的 `ManifestRow` 数组；
  `ManifestRow`只包含 `path/size_bytes/sha256`。
- 因此 `work_tree_sha256` 可以合法地被实现为：
  - 完整 `WorkRow` 数组的hash；或
  - 从 `WorkRow` 投影出的 `ManifestRow` 数组hash。
- 两者bytes不同，verifier与runner可各自选择不同解释。

必须修复：
- 单独冻结 `work_tree_sha256` 的exact canonical preimage：
  明确使用完整 `WorkRow` 或明确列出 `ManifestRow` projection、排序键、
  JSON参数与重复拒绝规则。

### P2-2 The 13-row verifier result is structurally named but not value-exact

位置：
- execution plan `:894-896`
- execution plan `:1073-1104`
- execution plan `:1481-1482`

问题：
- Revision 5冻结了唯一CLI、result path、exit codes、13个check IDs和顺序。
- 但 `VerifierCheck` 的四个字段全部是generic `str`，未冻结：
  - evaluated row `status` 的exact domain；
  - 每个check的exact `required` string；
  - PASS/FAIL时 `actual` 的canonical encoding；
  - `first_failure_code` 必须等于首个FAIL `check_id` 的显式等式；
  - PASS时 `first_failure_code=null`、13行全PASS的显式约束。
- 文档要求later rows“retain their exact required string”，但这些exact
  strings并不存在于合同中，所以多个不同result bytes都可声称合规。

必须修复：
- 注册13行完整表格：
  `check_id/order/required/actual encoding/status domain/failure code`。
- 冻结全局等式：PASS exit 0 iff 13 rows PASS and
  `first_failure_code=null`; FAIL exit 2 iff首个FAIL row与
  `first_failure_code`一致，后续全部NOT_EVALUATED。

## Closed Contract Areas

本轮确认已闭合或没有发现新缺陷：
- idea/plan Revision 5与task-frozen idea SHA authority；
- FULL A/B/P input路径、zero-based contiguous call index和
  `analyzed_unit_count`算术；
- retained slice no-replace/fsync与`work/` exact attempt child；
- `work-manifest`进入attempt-result和tracked terminal receipt closure；
- detector feature arrays read-only与entry/exit feature hash要求；
- instrumentation defect不再被要求伪造缺失scientific outputs；
- 唯一formal runner命令；
- 唯一terminal verifier CLI、cwd、result path和0/2/64 exit code；
- verifier可读根、禁止future/A0和read-only/no-replace边界；
- exact 17 root outputs、16-row self-excluding manifests和A/B/P root顺序；
- Revision 4的integer ceiling、UTC microsecond-Z和`-1` sentinel修复；
- scientific gate顺序、primary/sensitivity non-rescue和post-Build-A
  no-repair lock没有发现新的语义漂移。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/3/2/0
```

Revision 5 不可冻结。29-cache、future outcomes和A0 execution lock必须继续
关闭。只有上述 findings 全部闭合并经下一轮独立 review 达到
`0/0/0/0`，才可进入implementation/data execution。
