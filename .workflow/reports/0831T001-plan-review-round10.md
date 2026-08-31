# 0831T001 Independent Hostile Plan Review Round 10

执行线程：
- 独立 hostile plan reviewer

任务ID：
- 0831T001

状态：
- FAIL

更新时间：
- 2026-08-31 11:38 CST

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol`
- branch:
  `codex/leader-trigger-transition-hazard-protocol`
- reviewed commit:
  `0b6be7f3e134b37113773d9244dbc3480c2b3d23`
- reviewed commit parent:
  `2c02683336a078955a8d7fe2750e5770b5b3d301`
- reviewed commit message:
  `workflow: harden 0831T001 Q0 contract revision 10`
- frozen parent commit:
  `2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19`
- frozen parent protocol SHA256 / Git blob:
  `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40`
  / `69c5cdf51b7fdf07d55170ed58bc791ff37bd0af`
- Revision 10 plan SHA256 / Git blob:
  `b5936513192b58d0e261c5ba815ef573801954c0657d044274d211254aa10f15`
  / `1a8e225c11aa33a19e3cdbe62a4ef4e63147ebc2`
- fixture-truth SHA256 / Git blob:
  `c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529`
  / `ea66f4ff2e7cddf9302215d62c3268299682add7`
- surface-contract SHA256 / Git blob:
  `8934957274fa083a9a44fec7f1f5d721a7a4609e822ff85eafa491dca32411e2`
  / `e05b171b6861f6696f6bd022013cc348084911d5`
- task SHA256 / Git blob:
  `b1714b670d523e5d8229d0a29b1cc4d83ac60f991d009031908ce3b3237c6a29`
  / `5b15ad99a870fbe88a9c09f54a58522c9621d718`
- predecessor review commit:
  `2c02683336a078955a8d7fe2750e5770b5b3d301`
- predecessor review:
  `.workflow/reports/0831T001-plan-review-round9.md`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未读取任何 `local_live_analysis*`、formal 29-cache source root、historical
  cache 或 future outcome artifact。
- 未运行 formal Q0、A-1a、A-1b、A0、live、private、order 或交易操作。
- 未读取或调用 formal attempt root、controller bare repo、private
  credentials 或 exchange endpoint。
- 仅审查冻结 plan、task、contracts、workflow evidence、accepted tracked
  authority Git objects，并在系统临时目录运行隔离 Git/FD/flock 模拟。
- Q0 implementation、claim、receipt、tag 和 baseline 均不存在。
- 除本报告外未修改任何文件。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/1/0/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 10 已关闭合法 `Popen` receipt 域、artifact presence
cross-product 和 49-row FAIL rendering freeze。Controller pre/post-root
phase 也已分离，并新增十行 blocker restart 表。

但 controller expected-SHA authority 仍把已由 durable consumption receipt
证明写入过的 ref 的后续 `ABSENT` 状态视为合法；restart 表同时没有为错误
tag target、额外 HEAD 等本地 Git 偏差定义唯一 fail-closed 行。因此 Round 9
的 P1-2 仍未 executable-total。只有 `0/0/0/0` 才可 PASS。

## Round 9 Closure Matrix

| Round 9 finding | Revision 10 status | 独立证据 |
|---|---|---|
| P1-1 terminal resolver / legal `Popen` / invalid artifact order | `CLOSED` | producer/verifier `POPEN_ERROR` 均派生 `exit_code=NONE`；四个 terminal receipt hashes 精确复算；A01-A11 有序，A01-A05 的 64 行表精确为 7 legal / 57 invalid，aggregate hash 匹配；invalid outcome 唯一进入 `ARTIFACT_STATE_CORRUPTION` |
| P1-2 controller phase, observation and restart totality | `NOT_CLOSED` | pre/post-attempt-root observation outcomes已完整；但 receipt-sensitive expected SHA 与 invalid local Git/tag 状态仍不 total；见 P1-1 |
| P2-1 all legal FAIL report bytes | `CLOSED` | 49 行全部独立渲染，size range `1472..1537`，49 个 SHA256 全部唯一，canonical aggregate 精确匹配 |

## Findings

### P1-1 Controller expected-ref and blocker restart authority are still not total

Evidence:
- Every PASS and FAIL terminal commit must include:
  `.workflow/attempt-receipts/0831T001.consumption-push.json`.
- That tracked file is an exact copy of one durable
  `PUSH_CALL | REF_OBSERVATION` union member. Both legal members prove that the
  controller ref reached the exact consumption SHA.
- Nevertheless
  `expected_sha_sets_by_local_state.terminal_commit_without_terminal_push_receipt`
  is:

```text
<consumption_sha>
<terminal_sha>
ABSENT
```

- The coarser
  `consumption_commit_without_terminal_commit` state also allows `ABSENT`
  regardless of whether the tracked consumption receipt already exists.
- Therefore deletion or disappearance of a ref after durable consumption
  proof is accepted by `POST_ATTEMPT_ROOT.ABSENT_OR_EXPECTED_SHA` as a legal
  state rather than routed to `CONTROLLER_REF_DIVERGENCE`.
- The recovery policy can consequently rebuild the controller sequence from
  `ABSENT`, erasing the fact that an already-proven external ledger ref
  disappeared.
- The ten restart rows claim that exactly one row is selected from durable
  local state, but their terminal rows do not freeze all Git identity:
  - `BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE` requires the consumption tag target
    but does not require `HEAD` to equal the consumption commit.
  - `BLOCKER_POST_TERMINAL_LOCAL_COMPLETE` requires the terminal tag target
    but does not require `HEAD` to equal the terminal commit.
  - no row handles an existing consumption or terminal tag that peels to the
    wrong commit;
  - no row routes an extra or otherwise unexpected local HEAD to one
    deterministic corruption outcome.
- A wrong tag target can therefore leave no restart row, while an extra HEAD
  can satisfy a loosely defined "local complete" row and proceed to QA.

Machine counterexample:

```text
terminal common delta contains tracked consumption receipt = true
tracked receipt union proves remote consumption SHA = true
terminal-commit expected SHA set contains ABSENT = true

BLOCKER_PRE_TERMINAL_LOCAL_COMPLETE requires exact HEAD = false
BLOCKER_POST_TERMINAL_LOCAL_COMPLETE requires exact HEAD = false
invalid/wrong tag-target restart row exists = false
restart authority claims exactly one row = true
```

Impact:
- controller-ledger disappearance after durable write proof is not preserved
  as a workflow-integrity blocker;
- recovery may normalize a missing external ref instead of recording the
  loss;
- malformed local Git/tag state can either strand recovery outside all ten
  rows or be falsely treated as locally complete;
- the claimed controller blocker totality and one-shot provenance are not
  executable from all durable restart states.

Minimum executable closure:
1. Split expected SHA sets by both local commit state and durable receipt
   state. Before a successful/observed consumption receipt, `ABSENT` may be
   legal; after that receipt exists, only `<consumption_sha>` is legal until a
   terminal transition is proven.
2. After a terminal push receipt/observation exists, permit only
   `<terminal_sha>`. Treat later `ABSENT` as
   `CONTROLLER_REF_DIVERGENCE` or a separately frozen ledger-loss blocker.
3. Make every restart row bind exact `HEAD`, parent, commit message, tree
   delta, index/worktree state, tag presence and peeled tag target.
4. Add ordered invalid local-Git/tag predicates. Route wrong targets,
   unexpected commits, duplicate/conflicting tags and impossible
   receipt/ref combinations to one deterministic blocker without push or
   artifact rewrite.
5. Machine-enumerate the receipt/ref/HEAD/tag restart domain and freeze its
   legal-row count, invalid-row count and canonical aggregate SHA256.

## Machine Verification

Commands and isolated checks executed:

```text
git status --short --branch
git rev-parse / git show / git diff-tree
git hash-object / shasum -a 256
git diff --check 65590c52..0b6be7f3
git fsck --no-dangling --no-progress

strict duplicate-key JSON parse:
  .workflow/contracts/0831T001-fixture-truth-v1.json
  .workflow/contracts/0831T001-q0-surface-contract-v1.json

accepted authority source SHA/blob checks
six normalized callable AST SHA256 checks
four canonical terminal receipt derivation/hash checks
producer/verifier POPEN_ERROR tuple compatibility checks
ordered A01-A11 identity check
64-row artifact-presence derivation and aggregate check
terminal error-domain and hostile-error coverage
PASS/FAIL report example rendering
49-row exhaustive FAIL rendering and aggregate check
crash_states / crash_recovery_matrix key equality
controller phase/outcome/SHA-set/restart-row structural checks

temporary Git repositories:
  implementation -> arming -> consumption -> FAIL terminal
  implementation -> arming -> consumption -> PASS terminal
  exact parent chains, tags, staged deltas and controller refs

fixed FD 198/199 subprocess simulation:
  ACK
  MALFORMED_ACK
  EXTRA_ACK
  EOF_BEFORE_ACK
  TIMEOUT
  POPEN_ERROR
  inherited flock before and after child exit
```

Key passing results:

```text
STRICT_JSON = PASS
reviewed HEAD/parent identity = PASS
master/truth/surface SHA256 and Git blob identity = PASS
accepted authority source SHA/blob identity = PASS
six callable AST identities = PASS

terminal receipt hashes:
  NORMAL_PASS
    bc778f948ee54def6a964de44ea73745aac6e2a59cb528f6ee2f454fa24131a8
  RECOVERY_FAIL_PRE_PRODUCER
    a3edd59a990182b7dabc19e196911a270bafd8f404de59be22d38922800429f9
  PRODUCER_POPEN_FAIL_PRE_VERIFIER
    4ab52ad4ca9cc5c74f4bf1fd2c233e0310892fd56dbdb4b29a59b81d403d240f
  VERIFIER_POPEN_FAIL_CLASSIFIED
    d674cdf16b53e322c3ecabf1e1e803ec35bd88bd7168890973504226ed1847c1

producer POPEN_ERROR -> formal_exit_code NONE = PASS
verifier POPEN_ERROR -> terminal_verifier_exit_code NONE = PASS

artifact ordered rules = A01..A11
artifact presence rows = 64
legal / invalid = 7 / 57
artifact aggregate =
  f902794088e1b7649ca78fbc7053856fa51126d45e9d0347e061a73c65dd251b

FAIL report rows = 49
size range = 1472..1537
unique report SHA256 = 49
FAIL aggregate =
  73f050f783e7659a955c8a6650f28550665abb358b06db9c4ab9e15111bcdc48
PASS report = 1368 bytes, expected SHA256 exact
registered FAIL report = 1503 bytes, expected SHA256 exact

crash/matrix key sets = 20 / 20 equal
unknown crash stage-profile tokens = 0
truth/hostile errors missing from terminal domain = 0
fixture count = 15
feature calls = 57
controller precheck phases = 2
expected-SHA local-state rows = 4
blocker restart rows = 10 unique

Git FAIL:
  terminal staged paths = 3
  terminal delta paths = 3
  parent chain exact
  controller ref = terminal commit
  final tree clean

Git PASS:
  terminal staged paths = 60
  terminal delta paths = 60
  parent chain exact
  controller ref = terminal commit
  final tree clean

FD handoff:
  ACK -> ACKED
  malformed -> MALFORMED_ACK
  extra -> EXTRA_ACK
  EOF -> EOF_BEFORE_ACK
  timeout -> TIMEOUT
  Popen exception releases every parent/fixed descriptor
  flock blocked before child exit in every launched case
  flock acquirable after child exit/error cleanup
```

## Identity And Boundary Result

- reviewed commit identity: `PASS`
- reviewed parent chronology: `PASS`
- frozen parent identity: `PASS`
- plan SHA256/blob identity: `PASS`
- task SHA256/blob identity: `PASS`
- fixture-truth SHA256/blob identity: `PASS`
- surface-contract SHA256/blob identity: `PASS`
- accepted authority source SHA/blob identities: `PASS`
- six accepted callable AST identities: `PASS`
- worktree clean at review start: `PASS`
- Q0 implementation files: `ABSENT`
- 0831T001 tracked claim/receipt/tag: `ABSENT`
- formal attempt root: `NOT ACCESSED / NOT INSPECTED`
- historical cache access: `NONE`
- outcome access: `NONE`
- formal/live/private/order execution: `NONE`

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
- 当前为 formal 前独立 plan review；只有本轮
  `P0/P1/P2/P3=0/0/0/0` 才可解锁 implementation readiness。

files：
- `.workflow/reports/0831T001-plan-review-round10.md`

action：
- 独立审查冻结 Revision 10 plan/task/contracts/workflow evidence。
- 复算 Round 9 P1-1、P1-2、P2-1，并主动检查 strict JSON、identities、
  receipt hashes、Popen、artifact table、controller、report、crash、Git 和
  FD 边界。

verify：
- 见本报告 `Machine Verification`。

done：
- Verdict: `FAIL`
- Counts: `P0/P1/P2/P3=0/1/0/0`
- Round 9 P1-1: `CLOSED`
- Round 9 P1-2: `NOT_CLOSED`
- Round 9 P2-1: `CLOSED`
- implementation 与 formal execution 继续锁定。

blockers：
- controller expected-ref authority 和 blocker restart Git-state authority
  尚非 executable-total。

commit：
- 待本报告提交后填写于线程回报。

提交信息：
- `review: audit 0831T001 Q0 plan round 10`
