# QA 验收结果

执行线程：
- 独立计划审查线程

任务ID：
- 0831T001

状态：
- 未通过

更新时间：
- 2026-08-31 12:30 CST

验收线程：
- 0831T001 Q0 Revision 13 独立 plan reviewer

验收对象：
- reviewed commit:
  `db7866127540bebce9b7dc8401cdd9af27a3a833`
- commit message:
  `workflow: harden 0831T001 Q0 contract revision 13`
- frozen execution plan、task 与 surface contract
- Round 12 唯一 P1 的 Revision 13 closure

验收范围：
- raw `--no-renames` cached/worktree/untracked observation。
- staged/unstaged armed-to-claimed transition。
- `diff.renames=true/false` invariance。
- path/mode/blob/staging mutation fail-closed 性质。
- Revision 13 新增 P0-P3 风险。

验收限制：
- historical-cache access: `NONE`
- future-outcome access: `NONE`
- implementation access/review: `NONE`
- formal attempt root access: `NONE`
- claim/receipt/controller-ref access: `NONE`
- 未运行 formal、live、private、order 或交易操作。
- Git observation 验证仅使用系统临时目录的合成仓库。
- 未修改冻结 execution plan、task 或 surface contract。

## Severity

- **P0/P1/P2/P3 = 0/2/0/0**

## Round 12 P1 判定

- **NOT CLOSED**
- Round 12 carry-forward:
  `P0/P1/P2/P3=0/1/0/0`
- `--no-renames` 已关闭 `R100` rename collapse，且
  `diff.renames=true/false` observation byte-identical。
- 但 Round 12 还要求 path/mode/blob/staging mutation 全部 fail closed；
  staged mode mutation 仍有反例。

### P1-1 Round 12 P1 remains open: staged mode and exact bytes are not independently bound

Revision 13 的 `--no-renames` 命令成功阻止 `D+A` 被折叠为 `R100`，但
cached/worktree raw diff 仍受其他 Git 语义影响。

隔离仓库反例：

```text
core.filemode=false
claimed mode mutation: 0644 -> 0755
cached observation unchanged: true
worktree observation unchanged: true
worktree raw output: empty
```

因此一个真实 staged mode mutation 仍保持
`EXACT_CONSUMPTION_INDEX_STAGED` 的 canonical observation，G05 fail
open。`core.autocrlf=true` 下 LF 到 CRLF 的精确字节变化也可被 clean
normalization 隐藏，说明 staged worktree 实体字节没有被独立 hash 绑定。

## 新增 P0-P3

- New severity:
  `P0/P1/P2/P3=0/1/0/0`
- 当前已完成审查范围内没有新增 P0、P2 或 P3。

### P1-2 New: raw parser framing and all-state canonical preimages are incomplete

冻结 parser 声称每个 NUL record 同时包含 metadata 与 path。实际
`git diff --raw -z` 对每个非 rename row 编码为：

```text
metadata NUL path NUL
```

staged armed-to-claimed 因而产生四个非空 NUL fields，而不是两个各自含
metadata+path 的 record。契约未冻结 metadata/path pairing grammar、exact
regex、final-NUL 与 odd-field rejection。

此外，`state_derivation` 引用 exact
`tracked_transition_state_semantics preimage`，但该对象只有 prose。
只有两个 consumption state 新增了 row 描述；其余五个 legal dirty state
没有 machine-readable six-field preimage，receipt/report/baseline 路径也
没有冻结 exact mode。故 “every legal state” mutation probe 不可独立复现。

## Passing Evidence

1. `db786612` 的 HEAD、parent 与 message identity 通过。
2. plan/task/surface SHA256 与 Git blob identity 通过。
3. surface contract strict duplicate-key JSON parse 通过。
4. `git diff --check` 与 `git fsck` 通过。
5. staged raw output 在 `diff.renames=true/false` 下均为独立
   `D armed + A claimed`，字节完全一致。
6. unstaged raw/untracked observation 在两种 rename config 下完全一致。
7. 按 Git 实际 framing 构造 intended parser 时，两种 legal consumption
   state 均可精确分类。
8. 默认配置下 path/blob/staging 及 mode mutation 均触发 G05。
9. 116,640-row pre-blocker table 精确复现：
   `17 legal / 116,623 invalid`，aggregate
   `c55f8ccabea22cd4e386f5cd2923b5cfd40eb2e028f1fc9b4b0385475ad0bd2e`。
10. 21,384-row post-controller table 精确复现：
    `11 legal / 21,373 invalid`，aggregate
    `d9e4682bf43d4516b276eb46760fd020196503af5dfd2edc6e8b90b4336d4356`。

## 验收结论

- **未通过**
- **P0/P1/P2/P3 = 0/2/0/0**
- Round 12 carry-forward: `0/1/0/0`
- new findings: `0/1/0/0`
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

结论说明：
- Revision 13 关闭了 Round 12 的 rename collapse，但没有关闭完整
  observation-to-classification contract。
- staged mode mutation 仍有可执行 fail-open 反例。
- raw parser 与所有 legal state 的 canonical preimage 尚未冻结到唯一可实现。
- 只有 `0/0/0/0` 才可 PASS。

通过项：
1. raw `--no-renames` 分离 armed deletion 与 claimed addition。
2. `diff.renames=true/false` canonical observation invariant。
3. 两张既有状态机聚合保持精确。

不通过项：
1. staged mode/exact-byte mutation 不能保证 G05 fail closed。
2. raw NUL parser framing 不正确。
3. all-state canonical preimage/mutation matrix 不完整。

缺陷清单：
1. P1-1：staged worktree mode 与 exact bytes 未被独立绑定。
2. P1-2：raw parser framing 与 all-state preimages 不可执行-total。

阻塞项：
- Revision 13 不得解锁 implementation 或 formal execution。

建议总控下一步：
1. 冻结所有 present worktree path 的 no-follow lstat 与 exact-byte hash
   观察，包括 staged paths。
2. 冻结真实 `metadata NUL path NUL` parser grammar。
3. 为每个 legal action phase/branch 冻结 machine-readable canonical rows、
   exact mode/blob source 与 mutation aggregate。
4. 完成修订后发起新的独立 plan review。

详细报告：
- `.workflow/reports/0831T001-plan-review-round13.md`

提交信息：
- commit：由本轮审查提交承载，不在报告内自引用。
