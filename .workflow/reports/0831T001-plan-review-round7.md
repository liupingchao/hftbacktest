# 0831T001 Independent Hostile Plan Review Round 7

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
  `497f9648a6b3c6f36b0e2e8cbcea95d4f249fd53`
- frozen parent commit:
  `2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19`
- frozen parent protocol SHA256 / Git blob:
  `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40`
  / `69c5cdf51b7fdf07d55170ed58bc791ff37bd0af`
- Revision 7 plan SHA256 / Git blob:
  `d58648fe66ab8ee1fbf6fc446a7e00ca752a78e96f848708b628a6a4edb7a5e8`
  / `f8fc964e6fc0b61bda2abc5131ddd443bb02eebb`
- fixture-truth SHA256 / Git blob:
  `c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529`
  / `ea66f4ff2e7cddf9302215d62c3268299682add7`
- surface-contract SHA256 / Git blob:
  `a096435f5ec52108a5aa003a99bcdf9e8c652eb2134cbe4cc0f78f7ed69073d0`
  / `df2dd1a25913debecc9415e9cd6969878d5534d1`
- task SHA256 / Git blob:
  `600836857a8545f8b7c835084da576a831945523fbfc663b214ecef74e317e17`
  / `2ee4152722cb97390ba32de78d95936e9fe56f49`
- predecessor review commit:
  `3994325d8b594e658f4631915701d672648e4a04`
- predecessor review:
  `.workflow/reports/0831T001-plan-review-round6.md`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未读取任何 `local_live_analysis` historical cache 或 outcome artifact。
- 未运行 formal Q0、A-1a、A-1b、A0、live/private/order 操作。
- 仅运行 deterministic synthetic fixture、temporary-file publication、
  inherited-flock、temporary Git repository CAS、schema 和 package
  arithmetic 检查。
- 未修改 master、plan、task、contracts、production source、runner、
  verifier 或 tests。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/4/1/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 7 closes the ordered two-authority binding, corrects QF13, makes the
command-scoped Git configuration executable, and demonstrates a viable
parent-held flock pattern. It does not yet define an executable Git index/
arming history, a complete child FD handoff, or a total crash-to-terminal
mapping. Only `0/0/0/0` may PASS.

## Round 6 Closure Matrix

| Round 6 finding | Revision 7 status | 说明 |
|---|---|---|
| orphan-child runtime-lock race | `PARTIALLY_CLOSED` | parent-held inherited flock blocks recovery in a synthetic test, but child-visible FD identity, ACK EOF/timeout and descriptor lifetime are not frozen; see P1-2 |
| direct O_EXCL partial control JSON | `PARTIALLY_CLOSED` | temporary + fsync + hard-link no-replace works synthetically, but late recovery and observation-artifact crash states remain inconsistent; see P1-4 |
| invocation/recovery/terminal/report bytes | `NOT_CLOSED` | stage requirements contradict early-failure profiles, verifier exit 3 has no terminal code, and report bytes are not canonical; see P1-3/P1-4/P2-1 |
| two accepted production authorities | `CLOSED` | ordered `FEATURE_AUTHORITY`, `FIXED_EPOCH_AUTHORITY` rows bind path, commit, blob, SHA and six exact callable AST hashes |
| Git durability configuration | `CLOSED_AS_CONFIGURATION` | Apple Git 2.39.5 accepts and reports command-scoped `commit.gpgSign=false`, `core.fsync=all`, `core.fsyncMethod=fsync`; the separate formal index/arming command graph is still blocked by P1-1 |

## Findings

### P1-1 The exact Git command graph cannot create either registered commit

Evidence:
- surface lines 690-707 require a tracked armed-to-claimed rename followed by
  the consumption commit, then a terminal commit with three or sixty exact
  tracked paths.
- `one_shot.git_commands` contains commit/tag/push argv arrays but no
  `git add`, `git mv`, `git rm`, index writer, arming commit or implementation
  tag argv.
- the armed claim binds `implementation_commit`; therefore it cannot be
  contained in that same implementation commit. A separate tracked arming
  commit is required before the consumption commit, but no message, delta,
  parent, command or tag chronology for that commit is frozen.
- a synthetic clean repository with a committed `armed.json`, followed by
  filesystem rename to `claimed.json`, produced:

```text
exact consumption_commit argv exit = 1
status = D armed.json / ?? claimed.json
commit count unchanged
```

Impact:
- the exact `consumption_commit` command has no staged rename and cannot
  create the commit required by the absent-ref lease push.
- the terminal commit likewise has no registered way to stage its PASS/FAIL
  path set.
- implementations may invent different arming/index histories, so the
  claimed provenance and terminal delta are not one executable contract.

Required closure:
- freeze the arming commit, parent, exact tracked delta, message and
  implementation-tag target.
- add exact argv arrays for every index mutation, including PASS/FAIL-specific
  terminal staging, using the same command-scoped Git configuration.
- require exact clean-index preconditions and exact staged path/mode checks
  before each commit.

### P1-2 The inherited-lock idea works, but the registered child handoff is not executable as frozen

Evidence:
- surface line 1030 names symbolic
  `pass_fds=(runtime_lock_fd,handoff_ack_write_fd)`.
- the exact producer/verifier argv arrays at lines 1003-1022 contain no FD
  numbers, fixed FD slots or another child-visible binding for those two
  descriptors. No environment binding is registered.
- the contract does not state when the parent closes its own ACK write end.
  Without that close, `EOF_BEFORE_ACK` cannot be observed because the parent
  itself keeps a pipe writer alive.
- no ACK deadline, malformed-byte rule, extra-byte rule, child termination
  rule or requirement to retain the runtime descriptor through the child's
  final attempt-state write is frozen.
- a synthetic test passed only after explicitly giving the child both numeric
  FDs: ACK was `A`, recovery remained blocked after the parent closed its lock
  FD, and the lock became acquirable only after child exit.

Impact:
- the Round 6 pre-acquisition gap can be closed in principle, but two
  incompatible implementations can still satisfy the prose while differing
  on how the child finds FDs and when EOF/timeout is declared.
- a child that cannot identify the descriptors, or a parent retaining the ACK
  writer, can hang the one-shot state machine without a legal exit receipt.

Required closure:
- freeze fixed inherited FD numbers or exact
  `--runtime-lock-fd/--handoff-ack-fd` child argv fields.
- freeze parent read/write-end close order, exact one-byte ACK validation,
  deadline/kill/waitpid behavior and malformed/EOF classification.
- require the child to retain the runtime lock FD until all child-owned writes
  are complete and process exit closes it.

### P1-3 The terminal state machine is not total and its stage authorities contradict each other

Evidence:
- `formal_stage_requirements.FAIL` at surface lines 723-732 requires
  `F00..F06`.
- the registered `FAIL_PRE_PRODUCER`, `FAIL_PRODUCER_INTERRUPTED`,
  `FAIL_PRE_VERIFIER` and `FAIL_VERIFIER_INTERRUPTED` profiles at lines
  744-760 intentionally omit one or more of `F02..F06`.
- the plan freezes verifier exit code `3` for invocation/internal error, but
  the terminal FAIL domain at lines 969-982 has no
  `TERMINAL_VERIFIER_INTERNAL_ERROR` or verifier-exit-nonzero code.
  It maps every verifier exit to a registered verifier first error even when
  no canonical verifier result exists.
- exit-receipt domains list values independently but do not freeze legal
  tuples. For example, they do not require
  `POPEN_ERROR -> NOT_APPLICABLE/NONE/NONE`, nor bind verifier exit
  `0/2/3` to PASS/registered-failure/internal-error result states.

Impact:
- early interruption receipts violate the machine-level FAIL requirements.
- a child that ACKs and exits `3` has a committed exit receipt but no legal,
  unique terminal `first_error`.
- the terminal verifier cannot reconstruct one authoritative receipt from the
  same observed process state.

Required closure:
- remove the singular full-stage FAIL requirement or replace it with the
  exact allowed profile set.
- enumerate every orchestration and verifier error code as a machine array.
- freeze complete cross-field tables for launch status, handoff status, exit
  code, result presence, manifest presence, first error and stage profile.

### P1-4 Recovery after an already published terminal receipt cannot satisfy the receipt contract

Evidence:
- a normal terminal receipt freezes
  `recovery_observation_sha256=NONE` at surface lines 636-669.
- crash states at lines 393-397 and 846-869 explicitly require recovery after
  terminal receipt publication, terminal commit, tag or push.
- every recovery path requires one recovery observation, while PASS/FAIL
  domains at lines 959-980 require a 64-hex recovery hash on recovery.
- `control_publication` makes the existing terminal receipt immutable: an
  existing target is accepted only if its bytes equal the already derived
  bytes.

Executable contradiction:

```text
normal verifier result
-> publish terminal receipt with recovery_observation_sha256 = NONE
-> crash before terminal commit
-> recovery_observation.json is now required
-> terminal receipt must now contain its SHA256
-> existing terminal receipt cannot be changed
```

Additional incompleteness:
- `control_publication_crash_states` has no rows for
  `consumption_push_observation.json`,
  `terminal_push_observation.json` or `terminal_verifier_result.json`.
- rebuilding `recovery_observation.json` from current committed state is not
  byte-idempotent if a prior recovery advanced consumption/controller state
  and then crashed before linking the observation target; the original
  `crash_boundary` may no longer be derivable.

Impact:
- several registered process-crash windows have neither a legal immutable
  terminal receipt nor a legal successor receipt.
- late controller recovery cannot be represented without rewriting authority
  or violating the recovery-hash domain.

Required closure:
- define a late-controller-recovery rule in which the immutable terminal
  receipt remains valid, or defer/fork terminal receipt authority so recovery
  evidence can be bound without rewriting it.
- publish an immutable recovery-start marker before recovery changes durable
  state, or define recovery identity only from a stable normalized state key.
- add exact before/after-link rules for every observation/result control
  artifact.

### P2-1 The business execution report is not byte-unique

Evidence:
- surface lines 1159-1224 freeze encoding, line ending, section order and
  values, but not a complete byte template, canonical Markdown renderer or
  expected PASS/FAIL hash.
- multiple whitespace, heading, colon and list renderings satisfy the current
  field/order contract while producing different tracked bytes.
- the `files` list always contains
  `baselines/skhynix_trade_led_depth_follower_q0_v1 (PASS only)`, while
  `derived_values` branches only the blockers list and the FAIL terminal delta
  forbids every baseline path.

Impact:
- recovery cannot independently reconstruct the one exact report bytes
  required by the terminal commit.
- a FAIL report can carry a baseline files entry despite the exact FAIL delta
  having no baseline.

Required closure:
- freeze complete canonical PASS and FAIL report byte templates or an exact
  renderer plus expected derivation hashes.
- make the files list branch-specific and bind every placeholder source.

## Hostile Cross-Checks

Passed checks:
- reviewed commit was exact and the worktree was clean at review start.
- parent, plan, task, truth and surface SHA256/Git blob identities match.
- strict duplicate-key JSON parsing passes both machine authorities.
- both accepted source blobs and all six callable AST hashes match.
- ordered two-authority binding is complete and production paths are exact.
- accepted production primitives realize all registered anchor/event/reset
  geometry; QF05 contradiction precedence, QF06 boundary censor and QF09
  dual-follower tie are non-vacuous.
- QF13 all 14 model values match, including fractional `90.2s` time-of-day,
  `log(81)` background run and exact causal ranges.
- QF14/QF15 reset state and zero cross-segment carry match.
- QF07/QF08/QF12/QF15 retained full/slice anchors are nonempty and exact.
- package arithmetic is exact: 57 unique files, 56 terminal-manifest preimage
  files, 17 directories and 37 readiness files.
- the 14 truth/surface hostile probes have the same order and first errors.
- both terminal receipt example hashes reproduce exactly.
- temporary hard-link publication passes before-link/after-link checks.
- inherited parent-held flock and absent/exact-old lease-CAS patterns pass
  when their missing runtime bindings are supplied explicitly.
- command-scoped Git config returns `false / all / fsync` on the reviewed
  Apple Git runtime.

Failed checks:
- executable arming/index/commit command graph.
- exact child FD/ACK handoff and bounded EOF semantics.
- total verifier-exit/stage-profile/terminal-error mapping.
- late recovery after immutable terminal receipt publication.
- byte-unique business report reconstruction.

## Verification Performed

- read AGENTS, workflow-kit manual/templates, task_plan/progress/findings,
  task, frozen parent, Revision 7 plan, truth/surface contracts, accepted
  production sources and Round 6 report.
- verified current/frozen SHA256, Git blobs and normalized callable AST hashes.
- ran strict JSON, fixture/probe order, package/readiness arithmetic and
  terminal-receipt derivation checks.
- reconstructed deterministic fixtures in temporary directories and called
  accepted `build_features`, `base_masks`, `source_preflight`,
  `channel_actions`, `channel_memories` and `epoch_support_ledger`.
- exercised temporary hard-link publication, inherited flock handoff,
  command-scoped Git config and absent/exact-old lease-CAS.
- reproduced the unstaged-rename failure using the exact registered
  consumption commit argv.
- did not execute formal code or access historical/outcome data.

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/4/1/0**
- Plan freeze: **NOT AUTHORIZED**
- Implementation: **NOT AUTHORIZED**
- Formal Q0 execution: **NOT AUTHORIZED**
- historical-cache access: `NONE`
- outcome access: `NONE`
