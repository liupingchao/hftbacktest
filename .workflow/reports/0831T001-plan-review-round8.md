# 0831T001 Independent Hostile Plan Review Round 8

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
  `8952619e653cc8fdb5974659a8b394a4f1195d98`
- reviewed commit parent:
  `07ccc0bc37ae965f9d05613f6adb36a4d03c432c`
- reviewed commit message:
  `workflow: harden 0831T001 Q0 contract revision 8`
- frozen parent commit:
  `2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19`
- frozen parent protocol SHA256 / Git blob:
  `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40`
  / `69c5cdf51b7fdf07d55170ed58bc791ff37bd0af`
- Revision 8 plan SHA256 / Git blob:
  `fc086a4974a4d549109eb1a8df1dc20aba69b3a6d47587d85be3f54f59c3bb09`
  / `6ee60b6c78962fffd470439a0f94ee7c60736188`
- fixture-truth SHA256 / Git blob:
  `c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529`
  / `ea66f4ff2e7cddf9302215d62c3268299682add7`
- surface-contract SHA256 / Git blob:
  `e422fc3b709f7477704be970b880a3b40b223fe833eed332ec2f172f6e02f3f5`
  / `2eb04ae4a4eb321d2328dac006369df77b1904eb`
- task SHA256 / Git blob:
  `188e8784314917fa005442c1acc7d0a1ebeaadf992a32949f0f402878f67658a`
  / `57f638b6d7278dc9ed850bfe6a81337c39ec1651`
- predecessor review commit:
  `07ccc0bc37ae965f9d05613f6adb36a4d03c432c`
- predecessor review:
  `.workflow/reports/0831T001-plan-review-round7.md`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未读取任何 `local_live_analysis` historical cache、29-cache source root
  或 future outcome artifact。
- 未运行 formal Q0、A-1a、A-1b、A0、live、private、order 或交易操作。
- 仅运行 plan/task/contracts/workflow evidence 的只读检查，以及隔离临时
  Git repository、固定 FD/ACK/flock、canonical bytes/hash 和 schema
  机器验证。
- 未读取 implementation source，因为 Revision 8 implementation 文件尚不存在。
- 除本报告外未修改任何文件。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/2/0/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 8 的正常 Git graph、固定 FD handoff、early FAIL stage profiles、
immutable late recovery 和 exact report bytes 均有可执行进展，但 process/
crash first-error 合同仍不 total，且 exact FAIL 报告会在已注册的 early
failure 中产生错误执行陈述。只有 `0/0/0/0` 才可 PASS。

## Round 7 Closure Matrix

| Round 7 finding | Revision 8 status | 证据 |
|---|---|---|
| exact Git arming/index/consumption/terminal graph | `CLOSED` | exact stage argv 可创建 arming、consumption 和 terminal commits；FAIL 暂存 3 paths，PASS 暂存 60 paths；parent chain、100644 claim mode 和 armed/claimed blob identity 均通过 |
| fixed child FD/ACK/lifecycle | `CLOSED` | FD `198/199`、`pass_fds=(198,199)`、ACK `0x41 + EOF`、5000ms deadline、SIGTERM/SIGKILL/waitpid 和 child-held flock 生命周期已冻结；五种 synthetic handoff 分支通过 |
| process exit tuple、terminal error totality、early FAIL profiles | `NOT_CLOSED` | early FAIL profiles 本身闭合，但 crash matrix 会覆盖已经持久化的 producer error，且 `CONTROLLER_REF_DIVERGENCE` 无合法 terminal error；见 P1-1 |
| immutable terminal receipt、recovery_start/observation、control publication crash rows | `CLOSED` | terminal receipt 已移除 recovery hash；recovery start/observation 与 QA ownership 分离；新增 consumption/terminal observation、verifier result 和 recovery start crash rows |
| PASS/FAIL exact report bytes/hash | `PARTIALLY_CLOSED` | PASS/FAIL bytes、size 和 SHA256 可复算，FAIL baseline 已移除；但 FAIL 模板对 early-fail 执行事实作出错误陈述；见 P1-2 |

## Findings

### P1-1 Process/crash first-error authority is still contradictory and incomplete

Evidence:
- `terminalization_branches.FAIL.first_error_precedence` requires producer
  launch, handoff and nonzero-exit errors to precede later orchestration or
  verifier errors.
- `crash_recovery_matrix.after_producer_exit_before_verifier_invocation`
  nevertheless fixes
  `FORMAL_ORCHESTRATOR_INTERRUPTED_PRE_VERIFIER`.
- `after_verifier_invocation_before_exit_receipt` fixes
  `TERMINAL_VERIFIER_INTERRUPTED`, and `after_terminal_verifier_fail` resolves
  only `FROM_VERIFIER_EXIT_RECEIPT`.
- Those three boundaries can all coexist with an already committed producer
  exit receipt containing `FORMAL_PRODUCER_LAUNCH_ERROR`,
  `FORMAL_PRODUCER_HANDOFF_ERROR` or `FORMAL_PRODUCER_EXIT_NONZERO`.
- `recovery_observation_value_domains.first_error` requires the exact crash
  matrix value, so recovery cannot legally apply the separately frozen
  precedence.
- `recovery.remote_state_rules.other` names
  `CONTROLLER_REF_DIVERGENCE`, but that code is absent from
  `terminal_error_codes`; FAIL receipts allow only a code from that array.

Machine evidence:

```text
after_producer_exit_before_verifier_invocation
  matrix = FORMAL_ORCHESTRATOR_INTERRUPTED_PRE_VERIFIER
  prior durable producer errors =
    FORMAL_PRODUCER_LAUNCH_ERROR
    FORMAL_PRODUCER_HANDOFF_ERROR
    FORMAL_PRODUCER_EXIT_NONZERO

after_verifier_invocation_before_exit_receipt
  matrix = TERMINAL_VERIFIER_INTERRUPTED

after_terminal_verifier_fail
  matrix = FROM_VERIFIER_EXIT_RECEIPT

CONTROLLER_REF_DIVERGENCE registered_terminal = false
```

Impact:
- the same durable process state admits incompatible `first_error` values
  depending on whether the implementation follows precedence or crash matrix;
- controller divergence after attempt consumption has no legal terminal
  receipt value;
- the independent terminal verifier cannot reconstruct one total,
  deterministic terminal receipt for every registered state.

Required closure:
- freeze one executable cross-product/resolver over durable producer tuple,
  verifier tuple and crash boundary, with producer-error precedence preserved;
- make every crash-matrix `FROM_*` resolution and stage profile explicit;
- add `CONTROLLER_REF_DIVERGENCE` to the legal terminal domain with exact
  stage/receipt semantics, or explicitly remove that state from terminal
  recovery and define its workflow terminal outcome consistently.

### P1-2 The exact FAIL report can make a false formal-execution claim

Evidence:
- `FAIL_PRE_PRODUCER` is a legal FAIL profile with only
  `F00_PREFLIGHT` and `F01_CONSUMPTION_CLOSED` completed; no producer or
  verifier invocation exists.
- The single exact FAIL template nevertheless always states:
  `Executed the frozen synthetic QF01-QF15 Q0 pipeline exactly once.`
- Its `verify` section also always lists
  `formal producer and independent terminal verifier`.
- The byte derivation itself is exact and reproduces
  `size=1368`,
  `sha256=c5196a95e25b45ea394efd3682167d457dd1987ec8256108a8caf18e8d377056`;
  therefore this is a semantic evidence defect, not a whitespace ambiguity.

Impact:
- a valid pre-producer, launch, handoff or interrupted terminal branch can
  commit a byte-perfect report that falsely claims formal work occurred;
- the workflow evidence would disagree with the terminal receipt and stage
  profile it is supposed to summarize.

Required closure:
- use exact stage-profile-specific FAIL templates, or replace the execution
  claim and verify list with wording derived only from committed stages;
- freeze new placeholder sources and derivation hashes for every permitted
  rendering.

## Machine Verification

Commands and isolated checks executed:

```text
git status --short --branch
git rev-parse HEAD
git show -s --format=... 8952619e
git ls-tree 8952619e <plan/task/truth/surface paths>
git hash-object <plan/task/truth/surface paths>
shasum -a 256 <plan/task/truth/surface paths>

strict duplicate-key JSON parse for truth and surface
truth/hostile expected-error subset check against terminal_error_codes
terminal stage completed/missing partition check
crash matrix profile/error reference check
canonical terminal receipt derivation/hash check
canonical PASS/FAIL report rendering, size and SHA256 check
control-publication crash-row inventory check

temporary Git repository execution of exact:
  implementation tag
  arming_stage + arming_commit
  consumption_stage + consumption_commit + consumption tag
  terminal_stage_common
  terminal_stage_PASS
  terminal commit + terminal tag

fixed FD 198/199 subprocess simulation:
  ACK
  MALFORMED_ACK
  EXTRA_ACK
  EOF_BEFORE_ACK
  TIMEOUT
  inherited flock blocked-until-child-exit

accepted-authority Git blob/SHA and six callable AST hash checks
git diff --check 07ccc0bc..8952619e
git fsck --no-dangling --no-progress
```

Key passing results:

```text
STRICT_JSON = PASS
truth errors missing from terminal domain = []
hostile errors missing from terminal domain = []
terminal error duplicates = 0
stage profile overlap/unknown = 0/0
PASS report = 1368 bytes, expected SHA256 exact
FAIL report = 1368 bytes, expected SHA256 exact
NORMAL_PASS terminal receipt SHA256 exact
RECOVERY_FAIL_PRE_PRODUCER terminal receipt SHA256 exact

Git FAIL graph:
  arming add = 1
  consumption rename = R100
  terminal paths = 3
  parent chain exact
  final tree clean

Git PASS graph:
  arming add = 1
  consumption rename = R100
  terminal paths = 60
  parent chain exact
  final tree clean

FD handoff:
  ACK -> ACKED, flock blocked until child exit
  malformed -> MALFORMED_ACK
  extra -> EXTRA_ACK
  EOF -> EOF_BEFORE_ACK
  timeout -> TIMEOUT
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

## Verdict

- verdict: **FAIL**
- severity counts: **P0/P1/P2/P3 = 0/2/0/0**
- Round 8 is not authorized for implementation or formal execution.
- A new reviewed revision is required; no implementation or data execution
  may begin from commit `8952619e`.
