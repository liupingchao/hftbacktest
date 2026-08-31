# 0831T001 Independent Hostile Plan Review Round 14

执行线程：
- 独立 hostile plan reviewer

任务ID：
- 0831T001

状态：
- FAIL

更新时间：
- 2026-08-31 13:09 CST

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol`
- branch:
  `codex/leader-trigger-transition-hazard-protocol`
- reviewed commit:
  `7fae56ff22e98c1f0922776ece5afdc285fa8ceb`
- reviewed commit parent:
  `768799dde44120b6baf58f7729d68d7dd2463be7`
- reviewed commit message:
  `workflow: harden 0831T001 Q0 contract revision 14`
- predecessor review:
  `.workflow/reports/0831T001-plan-review-round13.md`
- Revision 14 plan SHA256 / Git blob:
  `2450b7ca7d074ffacb3c19d9a218bc2af7ac2c2a459a0b23dcba580e779a141f`
  / `bd4bff23c8df15405fb5c0019bdaf90833396a50`
- surface-contract SHA256 / Git blob:
  `23b5b1bf67b0aab09d33757eba89ea047789fd826138af00e91414484e1a4e26`
  / `97495dba843fd848aa5f52bce6a5e364a816d613`
- task SHA256 / Git blob:
  `881f08c08e7e7390639a24128ba682e19e5c3c7f97e87a23e1e1c3de234f564a`
  / `80bf3d9a94694d544e8434899f6c6bfe91796932`

访问边界：
- historical-cache access: `NONE`
- future-outcome access: `NONE`
- implementation access/review: `NONE`
- formal attempt root access: `NONE`
- claim/receipt/controller artifact access: `NONE`
- 未读取任何 implementation、cache、outcome、formal、claim、receipt 或
  controller artifact。
- 未运行 formal Q0、A-1a、A-1b、A0、live、private、order 或交易操作。
- Git parser 与 mutation 复验仅在系统临时目录的合成仓库中执行。
- 未修改冻结 task、execution plan 或 surface contract。

## Round 13 Closure

### Raw empty/nonempty metadata-NUL-path-NUL parser: PASS

- 真实 `git diff --raw -z --no-renames` staged rename 输出被独立解析为
  两个 `metadata NUL path NUL` records，状态为 `D,A`。
- empty bytes 被接受为零 records。
- lone NUL、缺 final NUL、odd field count、duplicate path 与 rename
  status 均 fail closed。
- 冻结 metadata regex、one-letter status、ASCII normalized path 与
  duplicate rejection 足以关闭 Round 13 parser framing 缺口。

### HEAD -> complete index -> complete physical inventory: PASS

- 冻结 derivation 从完整 HEAD tree 应用 cached preimage，得到完整
  expected stage-0 index，再与 `ls-files --stage/-v` 比较。
- expected physical map 从完整 expected index 应用 exact
  worktree/untracked preimage，并逐路径执行 no-follow kind/mode、
  SHA256 与 unfiltered Git blob 验证。
- 合成 staged transition 在 `core.filemode=false` 下将 claimed path 从
  `0644` 改为 `0755` 时，Git worktree raw 仍为空，但 physical mode
  mismatch 被捕捉。
- 合成 `core.autocrlf=true` transition 将 LF 改为 CRLF 时，filtered blob
  仍等于 index blob，但 exact blob 与 SHA256 均 mismatch。
- 因此 Round 13 staged mode/exact-byte fail-open 反例已关闭。

### Eight states, 22 variants and PASS60/FAIL3 expansion: PASS

- `tracked_transition_preimages` 恰有 8 个 legal states。
- `action_phase_preimage_variants` 恰有 22 个唯一
  `(action_phase, terminal_branch)` rows，覆盖 15 个 action phases。
- 每个 variant 引用已注册 state，无 missing state reference。
- `package_layout.package_files` 为 57 个唯一路径；
  `terminal_common` 为 3 个唯一且不重叠路径。
- PASS terminal delta 展开为 `57 + 3 = 60` paths；FAIL 展开为 3 paths。

## P0-P3 Finding

- **P0/P1/P2/P3 = 0/1/0/0**
- 未发现新 P0、P2 或 P3。

### P1-1 704-row aggregate encoding is not executable-total

冻结内容给出：

```text
22 variants * 4 mutation kinds * 8 repository configurations = 704 rows
aggregate = 8b28971875e83b64fe10a185e15a4a6871004b435c84387fa8a8403b68ecc06c
```

计数可复现，且目标 aggregate 可在以下未明文冻结的编码下复现：

```text
repository_config =
  {
    "diff_renames": <bool>,
    "core_filemode": <bool>,
    "core_autocrlf": <bool>
  }
variant_ordinal = zero-based integer
row order = variant, mutation kind, repository config
```

独立结果：

```text
snake_case keys + zero-based ordinal =
  8b28971875e83b64fe10a185e15a4a6871004b435c84387fa8a8403b68ecc06c

snake_case keys + one-based ordinal =
  3158320b1904b73f6b23c2a1831cbb5a7679aa8c684998401b39290765b6eed3

dotted keys + zero-based ordinal =
  6997e59ebeeeedca181eb89a0b4a3a031655a8ae13d699e07a436cbf93bd685c
```

但 surface contract：

- 将 configuration dimensions 命名为 dotted
  `diff.renames/core.filemode/core.autocrlf`；
- 只把 `repository_config` 注册为一个 row field，没有冻结其 JSON
  object schema、key names 或 value representation；
- 没有冻结 `variant_ordinal` 是 zero-based 还是 one-based，也没有冻结
  PATH probe 中 two-digit ordinal 的基数；
- 没有把上述实际产生目标哈希的 snake_case/zero-based preimage 写入
  machine-readable authority。

影响：

- 两个遵循现有文字的独立实现可生成不同 canonical rows、不同 PATH
  mutation 名称和不同 aggregate。
- 目标 SHA256 只能作为未知 preimage 的 commitment，不能替代唯一可执行
  derivation。
- Round 13 要求的 all-state mutation matrix 仍未完全关闭，因此不能以
  `0/0/0/0` 解锁 implementation。

最小关闭条件：

1. 冻结 `repository_config` 的 exact JSON schema、exact key names 与
   boolean encoding。
2. 冻结 `variant_ordinal` 的基数、类型与 PATH probe 的 two-digit
   formatting。
3. 冻结完整 row nesting/order rule，并重新发布由该明文 preimage
   机械导出的 704-row aggregate。

## Strict JSON And Hash References

```text
strict duplicate-key JSON parse = PASS
reviewed HEAD / parent / message identity = PASS
plan / task / surface SHA256 and Git blob identity = PASS
task -> surface SHA256/blob references = PASS
plan -> surface SHA256/blob references = PASS
git diff --check = PASS
git fsck --no-dangling --no-progress = PASS
```

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/1/0/0**
- Round 13 physical mode/exact-byte finding: `CLOSED`
- Round 13 raw parser finding: `CLOSED`
- Round 13 all-state canonical mutation derivation: `NOT CLOSED`
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 14 correctly closes the executable Git parser and physical inventory
defects. It does not freeze the exact canonical row representation that
produces the registered 704-row hash. Under the explicit rule that only
`0/0/0/0` may pass, this review must fail.

## Review Conclusion

执行线程：
- 独立 hostile plan reviewer

任务ID：
- 0831T001

状态：
- 未通过

是否进行QA验收：
- 否

QA说明：
- 当前为 formal 前独立 plan review；只有
  `P0/P1/P2/P3=0/0/0/0` 才可解锁 implementation readiness。

files：
- `.workflow/reports/0831T001-plan-review-round14.md`
- `docs/qa-acceptance-report.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 独立审查 Revision 14 的冻结 task、execution plan 与 surface contract。
- 在临时合成仓库复验 raw parser、filemode/autocrlf physical inventory。
- 机械核对 8 states、22 variants、PASS60/FAIL3 与 704-row aggregate。
- 复验 strict JSON、identity、hash references 与新 P0-P3。

verify：
- 见本报告各节。

done：
- Verdict: `FAIL`
- Counts: `P0/P1/P2/P3=0/1/0/0`
- implementation/formal locks remain closed.

blockers：
- 704-row canonical preimage 的 repository-config schema 与 ordinal
  encoding 未冻结。

commit：
- 由本轮审查提交承载，不在报告内自引用。

提交信息：
- `review: audit 0831T001 Q0 plan round 14`
