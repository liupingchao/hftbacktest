# 0830T002 Hostile Plan Review Round 7

日期：
- 2026-08-30 CST（星期日）

审查角色：
- independent hostile scientific-contract reviewer

候选对象：
- worktree
  `/Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate`
- branch `codex/fixed-epoch-relaxed-mstate-successor`
- commit `e9b8c521a234db305a0c2f11641ad0207c6f8b5a`
- idea SHA256
  `ba5574ea2bd64b8b79f06f4678e7be54e51459de99badc37cdce8c59d481f78b`
- plan SHA256
  `ce3e979dbf824323e728e1f278c3f188ef2e61f6dcf0621c6bfec2cdaba03793`
- task SHA256
  `9b87dae431b267ea6f7820f1125acd39e13039409b89c3754206e0f8b0516a61`

审查边界：
- 未读取或运行 29-cache。
- 未读取 future outcomes。
- 未运行 A0。
- 未修改 idea、plan、task、runner、tests 或研究产物。
- 本报告是本轮唯一新增文件。

静态身份核对：
- Candidate commit、branch、idea SHA 和 plan SHA 精确匹配。
- 初始 working tree 干净。
- Revision 7 只修改 docs/task/workflow 记录，未修改 frozen authority。
- `git diff --check 8c8558c9..e9b8c521` 通过。
- Local `origin` fetch/push URL均为计划注册的
  `git@github.com:liupingchao/hftbacktest.git`。

## Severity Summary

- P0: 0
- P1: 2
- P2: 2
- P3: 0

## Round 6 Closure Matrix

| Round 6 finding | Round 7 status | 结论 |
|---|---|---|
| P0-1 terminal SHA recursive self-reference | CLOSED | terminal/result/receipt均不再包含terminal commit SHA；terminal head只在post-terminal verifier result中出现 |
| P1-1 local bare controller writable by source principal | PARTIAL | ledger已移到external GitHub origin；但普通remote write credential仍可force/delete/追加第三次fast-forward，超出当前只排除admin的threat-model文字 |
| P2-1 controller installed state/transition identity not exact | PARTIAL | exact remote name/URL/ref已注册并进入claim；first-push state、exact refspec和terminal verifier remote observation仍未唯一冻结 |
| P2-2 RawOpenEvent/IPC/FD domains incomplete | PARTIAL | event enums、排序、violation counts和hostile cases已增加；actual IPC payload未进入envelope closure，部分derived字段仍无exact arithmetic |

## Findings

### P1-1 External origin ledger is mutable by ordinary write credentials inside the stated threat model

位置：
- execution plan `:457-482`
- execution plan `:555-557`
- execution plan `:607-609`
- execution plan `:1645-1647`

问题：
- Revision 7 明确排除的是“deliberately uses GitHub
  repository-administration credentials”的remote-admin actor。
- 但注册ledger只是普通GitHub branch：

```text
refs/heads/codex/0830T002-controller-ledger
```

- Plan未注册GitHub branch protection/ruleset、restricted push actor、
  deletion/force-push prohibition或terminal后write lock。
- 对未保护branch，持有普通repository write credential的actor通常即可：
  - force-push另一条history；
  - 删除ledger branch；
  - 在terminal commit后普通fast-forward第三个commit。
- 第三次fast-forward甚至不需要force或repository-admin credential。
- 因此当前ledger不能兑现“protects against ordinary protocol deviation”的
  claim，也未把所有可改写actor纳入outside-threat-model条款。
- Formal runner在push后立即执行`ls-remote`只能证明当时ref相等，不能阻止
  随后的普通writer变更。

必须修复：
- 二选一并唯一冻结：
  - 配置并验证external ruleset/protected branch，使formal credential只能
    完成注册的consumption/terminal transitions，terminal后禁止更新与删除；
  - 将threat model明确缩小为：任何持有remote write credential并偏离exact
    two-push protocol的actor均在模型外，且该行为使study invalid。
- Hostile tests必须区分admin、ordinary writer、force/delete和第三次
  fast-forward，而不是只测试local object-store deletion。

### P1-2 IPC envelope does not bind the actual serialized feature payload

位置：
- execution plan `:144-181`
- execution plan `:898-905`
- execution plan `:1145-1154`
- execution plan `:1348-1357`

问题：
- 注册的IPC envelope只包含：

```text
schema_version
call_index
feature_output_sha256
field_access_sha256
feature_key_count
```

- 它没有包含或闭包：
  - actual serialized array payload SHA；
  - payload byte length；
  - per-array dtype/shape/offset/length table；
  - exact frame count与EOF/no-extra-frame证明。
- `feature_output_sha256 == consumer_input_sha256`证明detector使用的feature
  values与loader输出语义相同，但不能证明IPC channel只传输了这些values。
- Loader可以发送正确feature payload外加额外raw bytes/extra frame；detector
  忽略额外内容后，三段feature hash和metadata envelope仍可全部相等。
- `raw_buffer_cross_boundary_count=0`与
  `ipc_envelope_violation_count=0`目前是runtime自报count，terminal verifier
  无法从envelope bytes独立重算“没有额外payload”。
- 因而Revision 6要求的“zero raw-byte buffers crossing a process boundary”
  尚未由formal evidence闭合。

必须修复：
- 冻结唯一IPC framing和canonical payload：
  exact header、array order、dtype、shape、offset、length、payload size、
  payload SHA、single-frame/EOF规则及禁止extra keys/frames。
- Loader与detector分别独立计算同一payload SHA/size，写入FeatureCall和
  instrumentation sibling；两端及envelope必须exact相等。
- Hostile case应实际追加raw bytes、second frame和unused field，证明即使
  detector feature hash不变也会terminal fail。

### P2-1 External remote/ref transition protocol is not yet uniquely executable

位置：
- execution plan `:457-472`
- execution plan `:534-557`
- execution plan `:1069-1087`
- execution plan `:1171-1198`

问题：
- Remote name、URL和ledger ref已固定，claim也携带三者，这是实质进展。
- 但formal contract仍未冻结：
  - exact `git push` argv/refspec；
  - exact `git ls-remote` argv和单行输出解析；
  - consumption前ledger ref必须absent，还是允许already-equal；
  - fetch URL和push URL必须各自exact且不得存在additional push URL；
  - first push、terminal push的expected old/new SHA tuples；
  - terminal verifier是否必须在线重查external ref，以及观察值写入哪个
    machine-readable field。
- 当前verifier result只有local `terminal_head`，没有
  `controller_remote/controller_url/controller_ref/observed_remote_head`。
  V02的generic `"true"`无法保留external observation identity。
- 因此不同实现可以在pre-existing equal ref、多push URL或不同refspec下都
  声称“remote ref exactly verified”。

必须修复：
- 冻结两次push和三次remote observation的exact commands、expected
  old/new tuple、absence/equality rules及failure precedence。
- Verifier result增加exact external ledger identity和observed head，或增加
  独立no-replace external-ledger verification sibling。

### P2-2 RawOpenEvent and IPC derived values still lack exact arithmetic/mapping

位置：
- execution plan `:914-918`
- execution plan `:1327-1357`
- execution plan `:1660-1663`

问题：
- IPC envelope引用 `field_access_sha256`，但合同没有定义：
  - preimage是本call的12个FieldAccess rows还是global ledger；
  - row order、JSON projection和serialization；
  - duplicate/foreign-call rejection。
- `feature_key_count` 也没有定义为：
  - loader output dictionary key count；
  - canonical feature-hash row count；
  - detector received key count；
  以及三者必须相等的exact等式。
- RawOpenEvent缺少phase-operation authority matrix，例如：
  - HASHER只能 `READ_INPUT`；
  - LOADER只能 `READ_INPUT`；
  - SLICE_MATERIALIZER的input/output path分别如何绑定。
- 当前Python runtime中，`os.open()`触发的audit event name是`open`；plan同时
  注册`open`和`os.open`，但没有冻结native event到canonical event_type的
  normalization rule。
- 因而相同runtime trace仍可能生成不同但表面合规的instrumentation bytes。

必须修复：
- 冻结per-call FieldAccess canonical hash、feature-key count来源和cross-end
  equality。
- 冻结每个phase允许的operation/path/caller matrix。
- 明确audit native event normalization，尤其`os.open -> open`还是基于
  authority stack规范化为`os.open`。

## Closed Contract Areas

本轮确认已闭合或没有发现新缺陷：
- idea/plan Revision 7与task-frozen idea SHA authority；
- pre-terminal attempt-result和terminal receipt均不含terminal commit/ref
  SHA，原P0 fixed-point cycle已消除；
- terminal head只在commit完成后的verifier result中出现；
- controller authority已移出local worktree和local Git object store；
- exact controller remote name、URL和ledger ref已写入plan/claim；
- remote-admin tampering已明确标为study invalidation而非supported recovery；
- outcome/instrumentation ownership继续保持无冲突；
- instrumentation sibling新增七个zero violation counts及loader/detector
  process-count equalities；
- RawOpenEvent已有global zero-based index、phase/event/operation enums、
  canonical sorting、caller-root语义和duplicate rejection；
- FeatureCall已有loader/consumer/detector-exit三段feature hash equality；
- inherited FD、path reconstruction、IPC extra field、event mutation和
  instrumentation sibling hostile cases已进入minimum；
- WorkRow hash、verifier 13-row equality、scientific gate precedence和
  primary/sensitivity non-rescue未发现新的回归。

## Freeze Decision

```text
FAIL
P0/P1/P2/P3 = 0/2/2/0
```

Revision 7 不可冻结。29-cache、future outcomes和A0 execution lock必须继续
关闭。只有上述findings全部闭合并经下一轮独立review达到`0/0/0/0`，才可进入
implementation/data execution。
