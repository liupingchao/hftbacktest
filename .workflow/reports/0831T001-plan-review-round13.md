# 0831T001 Independent Hostile Plan Review Round 13

执行线程：
- 独立 hostile plan reviewer

任务ID：
- 0831T001

状态：
- FAIL

更新时间：
- 2026-08-31 12:30 CST

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol`
- branch:
  `codex/leader-trigger-transition-hazard-protocol`
- reviewed commit:
  `db7866127540bebce9b7dc8401cdd9af27a3a833`
- reviewed commit parent:
  `318a191ffae76e7f38e1afaaad13d51fb46e63ab`
- reviewed commit message:
  `workflow: harden 0831T001 Q0 contract revision 13`
- predecessor review commit:
  `318a191ffae76e7f38e1afaaad13d51fb46e63ab`
- predecessor review:
  `.workflow/reports/0831T001-plan-review-round12.md`
- Revision 13 plan SHA256 / Git blob:
  `e28c1475dcf732fe086be7c585ac694e65dff3378d7615451dfb03375456ff16`
  / `0d242682ef3579e9e6a2a6518192168f59f23461`
- surface-contract SHA256 / Git blob:
  `767f360e73859d98371c768cded65d06df676753af0672f20a7ce4ff41ef78dd`
  / `f1f2d91d574429a35df6c105da501f2f73514c3a`
- task SHA256 / Git blob:
  `ed73ca52f9b1e4752fde294ac412b421e7a8e073acf65bd63d508c3580499e54`
  / `1e28e8ff6164dc244a99d6186716c4110c0e3d00`

访问边界：
- historical-cache access: `NONE`
- future-outcome access: `NONE`
- implementation access/review: `NONE`
- formal attempt root access: `NONE`
- claim/receipt/controller-ref access: `NONE`
- 未读取任何 `examples/hyperliquid/` implementation 文件。
- 未读取任何 `local_live_analysis*`、formal 29-cache source root、
  historical cache 或 future outcome artifact。
- 未运行 formal Q0、A-1a、A-1b、A0、live、private、order 或交易操作。
- 未读取或调用 formal attempt root、claim、receipt、controller bare repo
  或 controller ref。
- 仅审查目标 commit、冻结 plan/task/surface contract、Round 12 review 和
  workflow 状态文档。
- Git observation 与 mutation 验证仅在系统临时目录的合成仓库中执行。

## Round 12 P1 Determination

结论：
- **NOT CLOSED**
- Round 12 carry-forward severity:
  `P0/P1/P2/P3=0/1/0/0`
- Revision 13 closes rename collapse and proves byte-identical observations
  under `diff.renames=true/false`.
- It does not close the full Round 12 requirement that every path, mode, blob
  and staging-partition mutation fail closed.

### P1-1 Round 12 P1 remains open: staged mode and exact-byte mutations can remain legal

Evidence:
- The frozen cached/worktree commands scope `commit.gpgSign` and `core.fsync`,
  but do not bind `core.filemode`, content normalization, attributes or an
  independent lstat/hash observation for paths already present in the index.
- The plan requires every one-at-a-time path, mode, blob and staging-partition
  mutation to select `G05_INDEX_OR_TRACKED_WORKTREE_DIRTY`.
- In an isolated exact staged armed-to-claimed transition with
  `core.filemode=false`, changing the claimed worktree file from mode `0644`
  to `0755` left both frozen observations byte-identical to the legal state:

```text
cached_identical = true
worktree_identical = true
worktree_raw_bytes = empty
lstat mode = 0755
```

- The same class of omission exists for exact physical bytes under Git clean
  normalization. With `core.autocrlf=true`, changing the staged claimed
  worktree file from LF to CRLF produced an empty worktree raw diff while:

```text
index blob       = c52c573e54ce124152f93aec39ab2042100eef4b
exact-byte blob  = 588531188564b1160ce329cbe979cf5d91a9b8e6
filtered blob    = c52c573e54ce124152f93aec39ab2042100eef4b
```

- Therefore explicit `--no-renames` closes rename similarity only. It does
  not independently prove the actual mode or exact worktree bytes of a
  staged path.

Impact:
- `BLOCKER_CONSUMPTION_INDEX_STAGED` can retain the exact legal cached rows
  and an empty worktree row array after a real mode mutation.
- G05 fails open for a mutation category that Revision 13 explicitly claims
  must fail closed.
- The same observation weakness applies to terminal staged paths unless every
  present path is independently rebound to expected mode and bytes.

Minimum executable closure:
1. Independently lstat without following symlinks and hash exact bytes for
   every expected present worktree path, including paths already staged.
2. Bind those observations to explicit expected Git mode and expected blob
   sources, rather than relying on `git diff` cleanliness.
3. Freeze and test the observation under at least
   `core.filemode=true/false`, normalization enabled/disabled and the existing
   `diff.renames=true/false` matrix.

## New P0-P3 Findings

- New severity:
  `P0/P1/P2/P3=0/1/0/0`
- No new P0, P2 or P3 finding was identified in the completed review scope.

### P1-2 New: canonical raw parser and all-state preimages are not executable-total

Evidence:
- The surface contract states:

```text
with --no-renames every NUL record has exactly one path and fields
old_mode,new_mode,old_blob,new_blob,status,path
```

- Actual `git diff --raw -z` output is not one such NUL field. Each logical
  non-rename row is encoded as:

```text
metadata NUL path NUL
```

- The exact staged armed-to-claimed transition therefore produced four
  nonempty NUL fields:

```text
:100644 000000 <claim_blob> <zero_blob> D
.workflow/attempt-claims/0831T001.armed.json
:000000 100644 <zero_blob> <claim_blob> A
.workflow/attempt-claims/0831T001.claimed.json
```

- The unstaged deletion produced two nonempty NUL fields, not one record
  containing both metadata and path.
- The contract does not freeze the required metadata/path pairing grammar,
  exact metadata regex, final-NUL rule or odd-field rejection.
- `state_derivation` requires byte comparison with an exact
  `tracked_transition_state_semantics preimage`, but that object contains
  prose strings rather than row arrays.
- Only the two consumption states have added cached/worktree/untracked row
  descriptions. The other five legal dirty transition states have no
  machine-readable six-field row preimages, and the surface contract
  registers no exact file modes for their receipt/report/baseline paths.
- `mutation_probes` nevertheless requires path/mode/blob/staging probes for
  every legal transition state, without freezing the probe rows or aggregate.

Impact:
- A literal parser rejects every legal nonempty raw observation because no
  NUL-delimited field contains both metadata and path.
- An implementation that infers Git's pairing grammar may accept the
  consumption states, but different implementations can still construct
  different terminal preimages and legal modes.
- The all-state mutation guarantee is not independently reproducible from
  the frozen surface.

Minimum executable closure:
1. Freeze raw grammar as repeated `metadata NUL path NUL` pairs, including an
   exact metadata regex, one-letter status rule, final NUL, ASCII path rule
   and rejection of odd, duplicate or extra fields.
2. Freeze a machine-readable canonical observation preimage for every legal
   action phase and PASS/FAIL branch, with all six row fields and the
   independent source of every expected mode/blob.
3. Freeze the one-field mutation matrix and its expected G05 result, then
   publish a deterministic row count and aggregate.

## Round 12 Closure Matrix

| Round 12 requirement | Revision 13 status | Independent evidence |
|---|---|---|
| disable rename detection for staged armed-to-claimed | `CLOSED` | raw cached output is separate `D armed` and `A claimed` |
| prove unstaged armed deletion plus claimed addition | `CLOSED WITH INTENDED PARSER` | raw worktree has `D armed`; untracked lstat/hash has `A claimed` |
| rename configuration invariant | `CLOSED` | `diff.renames=true/false` produced byte-identical raw and canonical observations |
| exact classification under ordinary repository config | `CLOSED WITH INTENDED PARSER` | both legal consumption states matched their intended six-field rows |
| path/blob/staging mutation fail closed | `CLOSED UNDER TESTED DEFAULT CONFIG` | each mutation selected INVALID or the other legal partition, causing G05 for the selected phase |
| mode mutation fail closed | `NOT CLOSED` | `core.filemode=false` hid staged `0644 -> 0755` |
| exact raw parser | `NOT CLOSED` | frozen NUL-record statement contradicts actual metadata/path pair encoding |
| every legal state has an exact mutation preimage | `NOT CLOSED` | only two consumption preimages exist; terminal modes and canonical rows are absent |
| stale clean-index G05 sentence | `CLOSED` | plan now names exact cached/worktree/untracked rows |

## Machine Verification

Commands and isolated checks executed:

```text
git status --short --branch
git rev-parse / git show / git rev-list
git hash-object / shasum -a 256
git diff --check db786612^ db786612
git fsck --no-dangling --no-progress

strict duplicate-key JSON parse:
  .workflow/contracts/0831T001-q0-surface-contract-v1.json

temporary Git repositories:
  unstaged armed deletion + untracked byte-identical claimed addition
  staged D armed + A claimed transition
  diff.renames=true / diff.renames=false raw-byte comparison
  path / mode / blob / staging-partition mutations
  core.filemode=false staged mode mutation
  core.autocrlf=true exact-byte mutation

ordered G01-G07 cross-product reproduction:
  116,640-row pre-blocker table
  21,384-row post-controller table
```

Key results:

```text
STRICT_DUPLICATE_KEY_JSON = PASS
reviewed HEAD / parent / message identity = PASS
plan / task / surface SHA256 and Git blob identity = PASS
git diff --check = PASS
git fsck = PASS

unstaged rename-invariant = true
unstaged exact classification = true
staged rename-invariant = true
staged exact classification = true

default mutation matrix:
  unstaged path  -> INVALID -> G05
  unstaged mode  -> INVALID -> G05
  unstaged blob  -> INVALID -> G05
  unstaged stage -> STAGED  -> G05 for unstaged phase
  staged path    -> INVALID -> G05
  staged mode    -> INVALID -> G05
  staged blob    -> INVALID -> G05
  staged unstage -> UNSTAGED -> G05 for staged phase

core.filemode=false:
  staged mode mutation leaves cached/worktree observations unchanged

core.autocrlf=true:
  exact worktree bytes differ while cached/worktree observations remain
  unchanged

pre-blocker rows = 116,640
pre-blocker legal / invalid = 17 / 116,623
pre-blocker aggregate =
  c55f8ccabea22cd4e386f5cd2923b5cfd40eb2e028f1fc9b4b0385475ad0bd2e

post-controller rows = 21,384
post-controller legal / invalid = 11 / 21,373
post-controller aggregate =
  d9e4682bf43d4516b276eb46760fd020196503af5dfd2edc6e8b90b4336d4356
```

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/2/0/0**
- Round 12 carry-forward: `0/1/0/0`
- new findings: `0/1/0/0`
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 13 closes Round 12's rename-similarity defect at the Git command
level. It does not yet provide a complete fail-closed observation contract:
staged mode mutations can be invisible under a legal repository
configuration, exact worktree bytes are not independently rebound, the raw
NUL parser contradicts actual output framing, and most legal transition
preimages remain prose-only. Only `0/0/0/0` may PASS.

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
- `.workflow/reports/0831T001-plan-review-round13.md`
- `docs/qa-acceptance-report.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 独立审查 Revision 13 的冻结 plan/task/surface contract。
- 复验 Round 12 唯一 P1 的 raw `--no-renames` staged/unstaged
  armed-to-claimed observation。
- 在临时 Git 仓库验证 rename configuration invariance、exact
  classification 和 path/mode/blob/staging mutation fail-closed 性质。
- 主动复验 strict JSON、identity 和两张状态机聚合。

verify：
- 见本报告 `Machine Verification`。

done：
- Verdict: `FAIL`
- Counts: `P0/P1/P2/P3=0/2/0/0`
- raw delete/add separation: `PASS`
- rename-config invariance: `PASS`
- ordinary-config consumption classification: `PASS`
- universal mode/blob fail-closed observation: `FAIL`
- executable raw parser and all-state preimages: `FAIL`
- implementation 与 formal execution 继续锁定。

blockers：
- Staged worktree mode/exact-byte evidence is not independently bound.
- Raw NUL framing and every legal state's canonical preimage are not frozen
  in executable form.

commit：
- 待本报告提交后填写于线程回报。

提交信息：
- `review: audit 0831T001 Q0 plan round 13`
