# 0831T001 Independent Hostile Plan Review Round 9

执行线程：
- 独立 hostile plan reviewer

任务ID：
- 0831T001

状态：
- FAIL

更新时间：
- 2026-08-31 CST

审查对象：
- worktree:
  `/Users/liu/Documents/hftbacktest-0831-leader-trigger-transition-hazard-protocol`
- branch:
  `codex/leader-trigger-transition-hazard-protocol`
- reviewed commit:
  `65590c524801ecd0e9af4c547985eda526dbf8ed`
- reviewed commit parent:
  `7a0c05bc529f3d113cc38d70b2f87b6e85869700`
- reviewed commit message:
  `workflow: harden 0831T001 Q0 contract revision 9`
- frozen parent commit:
  `2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19`
- frozen parent protocol SHA256 / Git blob:
  `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40`
  / `69c5cdf51b7fdf07d55170ed58bc791ff37bd0af`
- Revision 9 plan SHA256 / Git blob:
  `fa5be4d973be0c870a3ebb19e5bc8711fcdefc026e3c5ec443760fea95b7d0f3`
  / `9b3ac140b6ba676a2839c307e2458915bbedb0a6`
- fixture-truth SHA256 / Git blob:
  `c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529`
  / `ea66f4ff2e7cddf9302215d62c3268299682add7`
- surface-contract SHA256 / Git blob:
  `07b41192fcf46b36bff29d7bc9e663dcacd360f5f644d2b8dbd60075ac5eed40`
  / `6c48931fd54df49bc05d3009ea56699980f5a234`
- task SHA256 / Git blob:
  `e1c60017db921aef662da0bff5e1b6606661c128a64d861096caa886ff5dfccd`
  / `dd4de1cb44b61a39be8533cc8a3a8acd444339aa`
- predecessor review commit:
  `7a0c05bc529f3d113cc38d70b2f87b6e85869700`
- predecessor review:
  `.workflow/reports/0831T001-plan-review-round8.md`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未读取任何 `local_live_analysis` historical cache、29-cache source root
  或 future outcome artifact。
- 未运行 formal Q0、A-1a、A-1b、A0、live、private、order 或交易操作。
- 仅审查 plan、task、contracts、workflow evidence，并运行隔离临时 Git、
  fixed-FD/flock、canonical bytes/hash、schema 和 cross-product 校验。
- Q0 implementation 文件仍不存在。
- 除本报告外未修改任何文件。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/2/1/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 9 已正确建立 producer-error precedence，且 FAIL 文案不再声称
缺失的 producer/verifier 阶段已执行。但 resolver 的合法 `Popen` 失败
分支仍无法生成符合自身合同的 terminal receipt；controller divergence
也没有覆盖 arming 后、attempt-root 前以及 observation/restart 的完整状态。
只有 `0/0/0/0` 才可 PASS。

## Round 8 Closure Matrix

| Round 8 finding | Revision 9 status | 证据 |
|---|---|---|
| process/crash first-error precedence 与 controller divergence workflow outcome | `PARTIALLY_CLOSED` | 10 个正常 chronological case 中 producer error precedence 已闭合，verifier exit 0 不能覆盖 producer failure；但 `Popen` receipt/terminal receipt 域冲突，非法 artifact cross-product 未注册，controller blocker 的 pre-root 与 restart 语义仍不 total；见 P1-1/P1-2 |
| early FAIL report truthful | `SEMANTICALLY_CLOSED / BYTE-MATRIX_PARTIAL` | FAIL template 只陈述 committed/missing stages，五个 early profile 均无虚假执行陈述；示例 bytes/hash 可复算，但 49 个允许的 profile/error rendering 仅冻结 1 个 derivation；见 P2-1 |

## Findings

### P1-1 The terminal-state resolver is not executable-total

Evidence:
- `producer_exit_tuple_table` registers a legal producer `POPEN_ERROR` receipt:
  `formal_producer_exit.json` exists while `exit_code=NONE`.
- `verifier_exit_tuple_table` registers the same legal shape for verifier
  `POPEN_ERROR`.
- `terminalization_branches.FAIL` instead requires `formal_exit_code` and
  `terminal_verifier_exit_code` to be base-10 integers whenever the
  corresponding exit receipt exists, reserving `NONE` only for an absent
  receipt.
- Therefore the registered
  `FORMAL_PRODUCER_LAUNCH_ERROR` and
  `TERMINAL_VERIFIER_LAUNCH_ERROR` resolver cases cannot produce a terminal
  receipt satisfying the branch contract.
- The 10 prose cases enumerate chronological states only. They do not freeze
  invalid artifact-order predicates such as verifier invocation without
  producer exit, verifier exit without verifier invocation, or terminal
  receipt/baseline without their predecessors. The precedence list would
  otherwise return an earlier absence case and silently ignore the later
  committed artifact.
- `result_totality` says invalid tuples "fail closed", but provides no exact
  error/blocker, profile, receipt policy or machine table for that outcome.

Machine evidence:

```text
producer POPEN_ERROR:
  exit receipt exists
  exit_code = NONE
FAIL terminal receipt rule:
  receipt exists -> base-10 integer

verifier POPEN_ERROR:
  exit receipt exists
  exit_code = NONE
FAIL terminal receipt rule:
  receipt exists -> base-10 integer

artifact-presence boolean cross-product = 64
chronologically invalid combinations under predecessor constraints = 56
explicit invalid-artifact table = absent
```

Impact:
- two registered launch-error paths have no legal terminal receipt;
- corrupt or partially published later artifacts can be misclassified as an
  ordinary earlier interruption instead of receiving one deterministic
  integrity outcome;
- normal and recovery implementations cannot reconstruct one total terminal
  tuple from the frozen contract.

Minimum executable closure:
1. Make terminal receipt exit fields derive from the legal process tuple:
   `NONE` for `POPEN_ERROR`, integer for `STARTED`, independently of receipt
   existence.
2. Replace the prose-only 10-case claim with a machine-readable ordered table
   over terminal receipt, baseline, producer invocation/exit tuple and
   verifier invocation/exit/result tuple.
3. Register exact invalid-order predicates and one deterministic
   terminal-error or workflow-blocker outcome, including profile and
   publication policy.
4. Retain the now-correct rule that verifier exit 0 can yield PASS only when
   producer exit is successful and baseline publication completes.

### P1-2 Controller-ref divergence is not a total restartable blocker state

Evidence:
- Controller preparation verifies an absent ref before arming. A ref can still
  become unexpected after the arming commit and before the formal outer
  driver's first preflight.
- That preflight must verify the ref before creating the attempt root and
  "without writing state".
- `before_attempt_root` is registered as
  `UNCONSUMED_RETRY_ALLOWED`, while
  `CONTROLLER_REF_DIVERGENCE` requires the claim to become consumed and the
  canonical observation to be published below
  `<attempt_root>/control/controller_ref_divergence.json`.
- No rule selects between those contradictory outcomes or authorizes creation
  of the attempt root solely for blocker publication.
- The blocker observation only admits `ls-remote` exit code `0` and one
  unexpected 40-hex SHA. Nonzero observation exit, malformed output, or
  unverifiable repository state has no exact consumed-state outcome.
- After blocker publication, the pre-terminal branch may still perform local
  claim rename/consumption commit/tag, and the post-terminal branch may perform
  local terminal commit/tag. There are no matching crash-state/matrix rows for
  interruption between those actions; `recovery_start` and
  `recovery_observation` are expressly forbidden after divergence.
- The control-publication row closes only the hard-link boundary of the
  blocker JSON, not the subsequent local Git transitions.

Impact:
- the same arming-to-attempt-root state can be retryable, permanently consumed,
  or unable to publish its required evidence depending on which prose clause
  is followed;
- a crash after blocker publication can leave claim/commit/tag state without a
  registered restart transition;
- controller observation failure can strand a consumed one-shot attempt
  outside both Q0 terminalization and the blocker workflow.

Minimum executable closure:
1. Freeze separate controller precheck phases. Either keep pre-attempt-root
   divergence retryable with no blocker/consumption claim, or create the
   no-replace attempt root first and route every post-arming divergence through
   the blocker contract; do not permit both.
2. Add an exact observation outcome table for absent, expected SHA, unexpected
   SHA, command failure and malformed output.
3. Add blocker-specific restart rows for observation -> claim rename ->
   consumption commit -> consumption tag and, post-receipt, terminal commit ->
   terminal tag.
4. Freeze expected SHA set, local commit/tag state, permitted next action and
   QA evidence for every blocker row.

### P2-1 Exact FAIL report derivations do not cover every allowed rendering

Evidence:
- The revised FAIL template is stage-truthful for all five early profiles.
- Placeholder occurrence/source checks pass, and the registered example
  recomputes exactly:
  `1503 bytes`,
  `53bd4a77cf7029117707e01b926cebb8a3c078635019538a9aa607a0debb885c`.
- The allowed resolver/profile/error combinations produce 49 distinct
  permitted FAIL renderings in the current domain.
- `execution_report.derivation_examples.FAIL` freezes only the
  `FAIL_PRODUCER_INTERRUPTED / FORMAL_PRODUCER_INTERRUPTED` rendering.
- For example, the legal pre-producer rendering is instead:
  `1537 bytes`,
  `17305de4942f54c75f837dd0e17abb0405dd7752b427c252ac514e545cebc2ec`,
  but that derivation is not registered.

Impact:
- the renderer is deterministic, but the advertised exhaustive exact
  size/hash evidence does not cover most legal terminal reports;
- a profile-specific substitution regression can escape the single example.

Minimum executable closure:
- register a machine-readable allowed `(stage_profile, first_error)` matrix
  and expected `size_bytes/sha256` for every permitted FAIL rendering, or
  freeze an equivalent canonical derivation table that mechanically proves
  the same set without prose.

## Machine Verification

Commands and isolated checks executed:

```text
git status --short --branch
git rev-parse HEAD
git show / git ls-tree / git hash-object / shasum for plan, task and contracts
strict duplicate-key JSON parse for truth and surface
crash_states versus crash_recovery_matrix key equality
terminal stage-profile partition and error-domain coverage
canonical terminal receipt derivation/hash checks
PASS/FAIL report placeholder, UTF-8/LF, size and SHA256 checks
all allowed FAIL profile/error rendering enumeration
artifact-presence cross-product and predecessor-order audit
accepted-authority Git blob/SHA and six callable AST hash checks

temporary Git repositories:
  implementation -> arming -> consumption -> terminal graph
  exact FAIL tracked delta and controller pushes
  exact PASS 57-file baseline delta and controller pushes

fixed FD 198/199 subprocess simulation:
  ACK
  MALFORMED_ACK
  EXTRA_ACK
  EOF_BEFORE_ACK
  TIMEOUT
  inherited flock blocked until child exit

git diff --check 7a0c05bc..65590c52
git fsck --no-dangling --no-progress
```

Key passing results:

```text
STRICT_JSON = PASS
crash/matrix key sets equal = true
unknown stage profiles = 0
terminal error duplicates = 0
truth/hostile errors missing from terminal domain = 0

NORMAL_PASS receipt SHA256 exact
RECOVERY_FAIL_PRE_PRODUCER receipt SHA256 exact
PASS report = 1368 bytes, expected SHA256 exact
registered FAIL report = 1503 bytes, expected SHA256 exact

Git FAIL graph:
  terminal staged paths = 3
  parent chain exact
  controller ref = terminal commit
  final tree clean

Git PASS graph:
  package files = 57
  terminal staged paths = 60
  parent chain exact
  controller ref = terminal commit
  final tree clean

FD handoff:
  ACK -> ACKED
  malformed -> MALFORMED_ACK
  extra -> EXTRA_ACK
  EOF -> EOF_BEFORE_ACK
  timeout -> TIMEOUT
  flock remained blocked before child exit in all cases
```

## Identity And Boundary Result

- reviewed commit identity: `PASS`
- frozen parent identity: `PASS`
- plan SHA256/blob identity: `PASS`
- task SHA256/blob identity: `PASS`
- fixture-truth SHA256/blob identity: `PASS`
- surface-contract SHA256/blob identity: `PASS`
- accepted authority source SHA/blob identities: `PASS`
- six accepted callable AST identities: `PASS`
- worktree clean at review start: `PASS`
- Q0 implementation files: `ABSENT`
- 0831T001 claim/receipt/tag/formal attempt root: `ABSENT`
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
- `.workflow/reports/0831T001-plan-review-round9.md`

action：
- 审查 Revision 9 plan/task/contracts/workflow evidence。
- 复验 Round 8 两个 P1，并主动检查 terminal resolver、
  controller blocker、exact report、Git、FD 和 recovery 边界。

verify：
- 见本报告 `Machine Verification`。

done：
- Verdict: `FAIL`
- Counts: `P0/P1/P2/P3=0/2/1/0`
- implementation 与 formal execution 继续锁定。

blockers：
- terminal-state resolver 尚非 executable-total。
- controller-ref divergence 尚非 total restartable workflow blocker。
- FAIL report exhaustive derivation matrix 尚未冻结。

commit：
- 待本报告提交后填写于线程回报。

提交信息：
- `review: audit 0831T001 Q0 plan round 9`
