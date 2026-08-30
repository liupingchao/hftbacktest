# 0830T002 Hostile Plan Review Round 6

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `8c8558c95d0902775548163542f783dd0e58c540`
- idea SHA256
  `f66d77546551bb16343f9517c3a2701c1117d8be105814de0e76ff99e8e1c027`
- plan SHA256
  `eb53557cb5f0f1f64b12946b15bc0d427a3f7c49e4023277b2d0436e4cb0e215`
- task SHA256
  `a44f00bb6ec39a236186c1e98ae6c9eb29cbb76669de1d7c789c648693145fe9`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Revision 6 只修改 docs/task/workflow 记录，未修改 frozen authority。
- `git diff --check 1cfe8bce..8c8558c9` 通过。

## Severity Summary

- P0: 1
- P1: 1
- P2: 2
- P3: 0

## Round 5 Closure Matrix

| Round 5 finding | Round 6 status | 结论 |
|---|---|---|
| P1-1 erasable local Git authority | PARTIAL | controller ledger已移出worktree并注册receive hook；但同一OS principal仍可直接改写bare repo，且新增terminal SHA自引用使completion不可执行 |
| P1-2 instrumentation/A-1-1 ownership conflict | CLOSED | feature/access/raw-open evidence已拆为terminal-only sibling，A-1-1仅保留future/poison/A-P字段 |
| P1-3 LOADER raw-open enforcement incomplete | PARTIAL | HASHER/SLICE/LOADER/DETECTOR进程边界、audit hook和close-fds已注册；formal event domains和边界 hostile evidence仍不exact |
| P2-1 ambiguous `work_tree_sha256` preimage | CLOSED | 已明确hash完整WorkRow数组、排序键、JSON参数和duplicate rejection |
| P2-2 verifier 13-row values/equalities incomplete | CLOSED | required/status/actual、first-failure code和exit 0/2 iff等式已冻结 |

## Findings

### P0-1 Terminal commit SHA is recursively embedded in its own tree

位置：
- execution plan `:607-615`
- execution plan `:1130-1140`
- execution plan `:1161-1172`
- execution plan `:1242-1245`

问题：
- Step 13先发布 `attempt-result.json`，其schema要求
  `controller_terminal_head:sha1`。
- Step 14随后创建terminal receipt；receipt又要求：
  - `terminal_head:sha1`；
  - `controller_terminal_head:sha1`；
  - `attempt_result_sha256`。
- Terminal commit的唯一tree delta正是这个receipt，因此terminal commit
  SHA取决于receipt bytes。
- 当前形成不可解的递归：

```text
terminal_head H
  -> attempt-result contains H
  -> receipt contains sha256(attempt-result(H)) and H
  -> terminal tree contains receipt(H)
  -> terminal commit SHA must equal H
```

- Git commit SHA不能在普通一次性构造中预先写入其自身tree。即使删除receipt中
  直接的 `terminal_head`，`attempt-result.controller_terminal_head ->`
  `attempt_result_sha256 -> receipt -> terminal commit` 仍保留同一固定点循环。
- 因此formal sequence无法生成同时满足schema、terminal commit和controller
  ledger equality的结果。这不是实现选择问题，而是合同不可执行。

必须修复：
- 从所有terminal commit之前发布且被terminal tree间接闭包的artifact中删除
  `terminal_head` / `controller_terminal_head`。
- Terminal receipt只绑定implementation/consumption heads和pre-terminal
  artifacts。
- Terminal commit创建并push后，由controller ref和post-terminal verifier
  result记录terminal head；如需独立controller receipt，必须是commit之外的
  post-push evidence，且不得回写已sealed attempt-result/receipt。
- 增加明确的self-reference graph hostile test，证明任何pre-terminal
  artifact都不含terminal commit SHA。

### P1-1 The local bare controller is not protected from its writer principal

位置：
- execution plan `:457-490`
- execution plan `:542-568`
- execution plan `:1073-1081`

问题：
- Bare repo虽然位于formal worktree之外，但仍在：

```text
/Users/liu/Documents/.codex-research-controller/0830T002.git
```

- Plan没有注册不同OS owner、只允许receive-pack的credential、filesystem
  ACL、sandbox或其他write isolation。
- `receive.denyDeletes`、`receive.denyNonFastForwards`和`pre-receive`只约束
  通过 `git receive-pack` 的push。拥有bare repo filesystem write权限的同一
  formal runner/operator仍可：
  - 直接执行 `git --git-dir=<controller> update-ref`；
  - 改写或删除hook/config；
  - 删除refs/reflogs/objects或整个bare repo；
  - 重建一个相同路径的新controller。
- 因而该authority尚未满足Round 5要求的append-only/protected controller。
  它只是第二个同权限本地Git目录。

必须修复：
- Controller必须由不同OS principal或独立服务拥有；formal process仅持有
  push-only transport权限，不能直接写controller Git目录、hook、config、
  refs或objects。
- 冻结transport、controller identity、owner/mode/ACL和forced receive-pack
  command，并让pre-cache preflight取得controller签发的durable consumption
  acknowledgement。
- Hostile tests必须覆盖direct `update-ref`、hook/config mutation、controller
  directory replacement和source-principal filesystem write，并全部fail
  closed。

### P2-1 Controller installed-state and transition identity are not exact

位置：
- execution plan `:468-490`
- execution plan `:546-564`
- execution plan `:1073-1081`

问题：
- `controller_config_sha256` 未定义canonical preimage：
  - 是完整bare `.git/config`文件bytes；
  - 还是四个注册key的排序projection；
  - 是否包含`core.bare`、repository format、filemode等初始化字段。
- `controller_hook_sha256`绑定了SHA，但没有冻结installed hook的exact
  controller-relative path、mode/executable bit、size和installed-byte
  identity。
- Hook要求consumption commit的parent是“implementation tag”。Git commit
  parent实际是commit OID，不是annotated-tag object；合同没有明确比较
  `implementation_tag^{commit}`。
- Pre-cache步骤没有逐项写明controller必须满足：
  - exact bare identity/config/hook installed bytes；
  - exact ledger ref absent；
  - zero unregistered refs；
  - hook可执行且本次push确实由该installed hook裁决。
- 因此不同controller初始化与parent比较实现仍可能同时声称合规。

必须修复：
- 冻结controller config canonical JSON projection或完整file-byte schema，
  并明确所有required/forbidden config keys。
- 注册installed hook path、file SHA、size、mode和executable requirement。
- 将parent规则写为exact peeled implementation commit SHA。
- 在formal argv/preflight和verifier中注册controller installed-state的exact
  检查顺序与first-failure ownership。

### P2-2 Instrumentation sibling lacks exact event domains and new hostile minimum

位置：
- execution plan `:144-181`
- execution plan `:919-923`
- execution plan `:1151-1157`
- execution plan `:1301-1330`
- execution plan `:1593-1599`

问题：
- Revision 6已定义 `RawOpenEvent` 结构，但未冻结：
  - `event_index` 是global还是per-process、起点和连续性；
  - `phase` exact enum；
  - `event_type` exact enum与Python audit event映射；
  - `operation` canonical vocabulary；
  - `caller_path/caller_name` 是immediate caller还是authority-root caller；
  - event rows的exact排序与duplicate规则。
- `instrumentation-evidence.status` 的exact success value未定义。
- 三个violation counts虽在成功语义中应为零，但未进入exact list-domain
  equalities；“zero raw handles/NPZ/mmap/raw buffers cross boundary”也没有
  对应typed count/hash/IPC-envelope evidence。
- Frozen hostile minimum仍主要保留Revision 5的alternate-loader/proxy测试，
  未显式覆盖Revision 6新增边界：
  - inherited FD注入；
  - DETECTOR raw path/env/cwd reconstruction；
  - LOADER IPC extra field/raw-byte smuggling；
  - RawOpenEvent drop/reorder/caller spoof；
  - HASHER/SLICE caller-authority mutation；
  - instrumentation sibling missing/extra/mutation和no-replace publication。
- 因此subprocess架构方向合理，但terminal verifier尚不能按唯一合同审计
  whole-process boundary evidence。

必须修复：
- 冻结RawOpenEvent所有enum、index scope、排序、caller定义和成功等式。
- 注册exact IPC envelope，只允许canonical feature arrays与access ledger，
  并发布per-process inherited/open FD count和payload schema/hash evidence。
- 将上述controller/process/instrumentation mutation加入Frozen Hostile-Test
  Minimum。

## Closed Contract Areas

本轮确认已闭合或没有发现新缺陷：
- idea/plan Revision 6与task-frozen idea SHA authority；
- `outcome_access_ledger`不再包含FeatureCall/FieldAccess/instrumentation
  fields，A-1-1 ownership冲突已消除；
- 独立terminal-only `instrumentation-evidence.json` sibling及其
  attempt-result/terminal receipt SHA binding；
- HASHER、SLICE_MATERIALIZER、LOADER、DETECTOR职责拆分；
- LOADER在input open前安装audit hook，DETECTOR不接收registered raw path；
- `close_fds=True`、read-only features、entry/exit feature hash和
  process-exit boundary方向；
- whole-successor AST/import closure与dynamic import/eval/exec禁令；
- FULL/SLICED path authority、zero-based call index和analyzed-unit arithmetic；
- 完整WorkRow hash preimage、排序与duplicate rejection；
- verifier唯一CLI、cwd、result path和0/2/64 exit codes；
- verifier 13-row required/status/actual encoding与global iff equalities；
- scientific gate precedence、primary/sensitivity non-rescue和post-Build-A
  no-repair lock未发现新的科学语义漂移。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 1/1/2/0
```

Revision 6 不可冻结。P0意味着当前formal completion在结构上不可生成。
29-cache、future outcomes和A0 execution lock必须继续关闭。只有上述findings
全部闭合并经下一轮独立review达到`0/0/0/0`，才可进入implementation/data
execution。
