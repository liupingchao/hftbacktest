# 0831T001 Independent Hostile Plan Review Round 6

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
  `6b05ff0cdcd16678ded45f83ae035f3f30a0ffda`
- frozen parent commit:
  `2dcd1d95b7c6ff24cb5991e8dc1d3d97b2666b19`
- frozen parent protocol SHA256 / Git blob:
  `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40`
  / `69c5cdf51b7fdf07d55170ed58bc791ff37bd0af`
- Revision 6 plan SHA256 / Git blob:
  `f801eb187a8561e30c12d776f1c5c051100fc5301641abf5948fabfe566560f9`
  / `6209742339d686f2b8a4cff81f093f14d259a1a9`
- fixture-truth SHA256 / Git blob:
  `c9e1c5dba760309add5e0debfdfff6be3387e8978b1e5506b6d1fff9df87f529`
  / `ea66f4ff2e7cddf9302215d62c3268299682add7`
- surface-contract SHA256 / Git blob:
  `62a7a3d751ec46bc702b4242f85fdea425eb9c5c4879c233c4dfdf5a0d4d8851`
  / `9f3ab2c13f0f8ea03cffc65dcd7e0abb9cf5f25b`
- task SHA256 / Git blob:
  `f67c7b7041ca2d3ad4c5d1180df0680a5a86ba520712066d80e9dcc6f837a6f4`
  / `100842136e67ca94f492a237846532a1dc2b0710`
- predecessor review:
  `.workflow/reports/0831T001-plan-review-round5.md`

访问边界：
- historical-cache access: `NONE`
- outcome access: `NONE`
- 未读取任何 `local_live_analysis` historical cache 或 outcome artifact。
- 未运行 formal Q0、A-1a、A-1b、A0、live/private/order 操作。
- 仅运行 deterministic synthetic primitive、temporary-file crash、
  temporary bare-repository CAS、schema 和 package arithmetic 检查。
- 未修改 master、plan、task、contracts、production source、runner、
  verifier 或 tests。

## Decision

- **FAIL**
- **P0/P1/P2/P3 = 0/4/1/0**
- plan freeze: **NOT AUTHORIZED**
- implementation lock: **CLOSED**
- formal execution lock: **CLOSED**

Revision 6 正确修复了 QF13 oracle，并冻结了 outer、producer、verifier、
recovery 四组 argv、`ABSENT`/`NONE` 与两种 push receipt；但 child runtime
lock 仍有可执行竞态，O_EXCL JSON 直接写入不能覆盖 crash-during-write，
terminal/recovery receipt bytes 与 accepted-authority package binding 仍不唯一。
只有 `0/0/0/0` 才可 PASS。

## Round 5 Closure Matrix

| Round 5 finding | Revision 6 status | 说明 |
|---|---|---|
| QF13 H1 oracle/access range | `CLOSED` | accepted primitives exact produce `log(81)`；14 个 model inputs 全部匹配；memory scan exact 为 `4499..4509` |
| crash-state terminalization | `PARTIALLY_CLOSED` | pre-producer/pre-verifier branches、REF_OBSERVATION 和 baseline recovery 已注册，但 orphan-child 与 partial-control-file crash 仍不能闭合；见 P1-1/P1-2 |
| producer argv and absent-ref bytes | `PARTIALLY_CLOSED` | 四组 argv 与 `ABSENT`/`NONE` 已精确冻结；其余 invocation/recovery/stage/report value domains 仍不唯一；见 P1-3 |

## Findings

### P1-1 Child runtime lock has a pre-acquisition race and cannot prove an orphan child is gone

Evidence:
- surface lines 581-582 require writing the invocation claim before `Popen`;
  the child itself then acquires its runtime flock.
- surface lines 554 and 586 allow recovery to classify an invocation without
  an exit receipt after recovery can acquire and release that same lock.
- there is an uncovered interval:

```text
invocation claim fsynced
-> Popen succeeds
-> parent dies
-> child exists but has not yet acquired its runtime flock
-> recovery acquires/releases the apparently free lock and terminalizes
-> child later acquires the lock and writes attempt state
```

- a synthetic process check reproduced exactly this ordering:
  `recovery_lock_acquired_while_child_alive=true`, followed by
  `child_later_acquired_and_wrote=true`.

Impact:
- recovery can emit `FORMAL_PRODUCER_INTERRUPTED` or
  `TERMINAL_VERIFIER_INTERRUPTED` while the corresponding child is still
  capable of executing.
- the at-most-once and terminal immutability claims are false under a normal
  scheduler delay between `Popen` and child-side `flock`.
- later child writes can mutate package/result state after a terminal FAIL
  commit has been derived.

Required closure:
- the outer driver must acquire the child runtime lock before `Popen` and pass
  the already locked descriptor to the child, or use an equally strong
  parent-held handoff protocol.
- freeze exact FD inheritance, `close_fds`/`pass_fds`, handoff acknowledgement
  and recovery acquisition semantics.
- recovery must not classify interruption until ownership transferred to the
  child and the inherited lock has subsequently become acquirable.

### P1-2 O_EXCL direct JSON writes can leave permanent partial control artifacts

Evidence:
- surface lines 466, 510, 581-582 and 604 define receipts, invocation claims
  and exit receipts as direct `O_CREAT|O_EXCL` writes followed by fsync.
- `O_EXCL` makes pathname creation exclusive; it does not atomically publish
  complete file contents.
- a crash after `open(O_EXCL)` and before the full write/fsync can leave a
  zero-byte or prefix-only file. A synthetic check left a 22-byte invalid JSON
  path, and a second O_EXCL open correctly failed with `FileExistsError`.
- the consumption union requires REF_OBSERVATION only when the normal
  PUSH_CALL receipt is absent. A partial normal receipt path is present but is
  not a valid PUSH_CALL member, so neither branch is legal.
- the same unresolved state exists for attempt lock, invocation claim, process
  exit receipt, recovery observation, tracked receipt copy and terminal result
  receipt.

Impact:
- multiple registered crash windows can end with neither a valid PASS nor a
  valid FAIL artifact sequence.
- recovery restart is not idempotent when a target pathname exists with
  incomplete bytes.
- the claim may remain consumed while the required receipt union and terminal
  commit are impossible to construct.

Required closure:
- publish every content-bearing control artifact through a sibling temporary
  file, complete write, file fsync, atomic no-replace rename and parent fsync.
- separately freeze how an abandoned temporary file is detected and handled.
- if O_EXCL pathname creation itself is the claim, use a distinct committed
  payload/marker protocol so an empty claim still has an executable terminal
  interpretation.
- add explicit crash states for interruption during each control-artifact
  publication, not only before/after the high-level transition.

### P1-3 Terminal, invocation and recovery receipt value domains are not byte-unique

Evidence:
- surface lines 555-559 define invocation fields but do not freeze exact
  `child_kind`, `invocation_ordinal` or their types/domains.
- surface lines 590-600 define recovery observation fields but do not freeze
  domains for `crash_boundary`, invocation/exit states or `recovery_mode`.
- PUSH_CALL and REF_OBSERVATION freeze SHA relationships, but
  `transition_kind`, command representation, schema-value types and all
  non-SHA enums are not completely enumerated.
- terminal receipt fields require `completed_stages_json` and
  `missing_stages_json`; lines 498-506 refer to stage IDs, while
  `transition_order` contains prose descriptions and defines no stage IDs.
- PASS says `missing_stages = NONE`, while the tracked field is
  `missing_stages_json`; the contract does not choose between the JSON bytes
  `[]`, the JSON string `"NONE"` or another token.
- both branches require an execution report, but no exact report path, field
  schema, canonical serializer or terminal-commit tracked path set is frozen.

Impact:
- different implementations can produce different valid-looking receipt,
  recovery and terminal commit bytes from the same durable state.
- an independent verifier cannot reconstruct the authoritative terminal
  receipt or prove the exact terminal commit delta.
- crash recovery cannot be exact-byte idempotent because its supposedly
  idempotent observation bytes are under-specified.

Required closure:
- freeze exact value/type domains for every invocation, PUSH_CALL,
  REF_OBSERVATION, recovery and terminal field.
- register stable stage IDs and exact completed/missing arrays for every crash
  state.
- freeze the execution-report path/schema/serializer and the exact allowed
  tracked delta for PASS and FAIL terminal commits.
- include canonical normal-path and recovery-path receipt examples as machine
  authorities or derivation tests.

### P1-4 authority_binding.json cannot exactly represent both accepted production authorities

Evidence:
- plan lines 296-344 require path, commit, Git blob, file SHA and callable AST
  identities for two distinct source authorities and six reused callables.
- the fixed-epoch source path is not stated in its frozen identity block.
- surface lines 187-203 provide only singular
  `accepted_authority_commit`, `accepted_authority_path`,
  `accepted_authority_sha256` and `callable_ast_sha256` fields.
- there is no `accepted_authority_blob` field even though the plan requires
  both source Git blobs to be package evidence.
- no nested object/array schema defines how the two commits, two paths, two
  blobs, two file hashes and six AST hashes are encoded under those singular
  fields.

Impact:
- the producer and verifier must invent an authority-binding shape.
- the exact accepted fixed-epoch path/blob can be omitted while the package
  still satisfies the registered top-level field set.
- A/B/P structural package bytes are not uniquely derivable from the frozen
  contract.

Required closure:
- replace the singular fields with an exact machine schema such as an ordered
  `accepted_authorities` array.
- each row must freeze authority ID, path, commit, Git blob, file SHA256 and an
  exact callable-name-to-AST-SHA map.
- explicitly register
  `examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py` as the
  fixed-epoch source path and refresh package schemas/identities.

### P2-1 Frozen Git durability configuration has no executable setup or matching preflight

Evidence:
- surface lines 363-366 freeze:

```text
commit.gpgSign=false
core.fsync=committed,loose-object,batch,reference
core.fsyncMethod=fsync
```

- the exact controller preparation commands set only
  `receive.denyNonFastForwards` and `transfer.fsckObjects`.
- the exact commit/push commands do not use command-scoped `git -c` values,
  and no pre-arm working-repository config transition is registered.
- the reviewed worktree currently has local `core.fsync=all`,
  `core.fsyncMethod=fsync` and no explicit local `commit.gpgSign` entry.

Impact:
- enforcing the frozen `git_config` blocks the current formal preflight.
- ignoring the mismatch means the implementation is not executing the exact
  registered Git runtime contract.

Required closure:
- either register a retryable pre-arm setup and exact verification for the
  working repository, or place the frozen values directly in every exact Git
  argv with command-scoped `-c`.
- define whether a stronger value such as `core.fsync=all` is accepted or is
  an exact mismatch; do not leave this to implementation choice.

## Hostile Cross-Checks

Passed checks:
- reviewed commit was exact and the worktree was clean at review start.
- parent, truth, surface, task and both accepted source SHA256/Git blob
  identities match.
- all six accepted callable AST hashes match.
- strict duplicate-key JSON parsing passes both authorities.
- all 15 fixtures produce registered anchor counts and identities under the
  accepted production primitives.
- QF13 all 14 model values match exactly; its background scan reads
  `4499..4509` and produces `4.394449154672439`.
- QF14/QF15 exact reset state is `[3,5,5]`, `[-1,0,0]`, `[0,80,80]`,
  followed by `[9,9,9]`, with zero cross-segment carry.
- QF07/QF08/QF12/QF15 full/slice epochs and retained outcome identities are
  nonempty and exact.
- QF09 has simultaneous same-direction depletion and OFI actions at the
  registered checkpoint, supporting the frozen tie detail.
- truth and surface freeze the same 14 hostile probes in the same order.
- package arithmetic is exact: 57 unique files, 56 terminal-manifest preimage
  files, 17 directories and 37 readiness files.
- the 17 row fields partition exactly into 12 accepted feature reads plus five
  staged-extension reads.
- prose and machine contract contain byte-identical token arrays for outer,
  producer, verifier and recovery argv.
- temporary bare-repository tests passed both absent-ref and exact-old
  lease-CAS transitions.
- PASS baseline roots are on one filesystem; Darwin provides exclusive
  directory-rename primitives, so atomic baseline publication is feasible.

Failed checks:
- child runtime-lock orphan exclusion.
- crash-safe atomic publication of control JSON and tracked receipt copies.
- exact invocation/recovery/terminal value domains, stage IDs and report
  closure.
- exact two-authority package binding.
- executable Git durability configuration.

## Verification Performed

- read AGENTS, workflow-kit manual/templates, task_plan/progress/findings,
  task, frozen parent, Revision 6 plan, both machine contracts, accepted
  production sources and Round 5 report.
- verified current/frozen file SHA256, Git blobs and callable AST hashes.
- ran strict JSON duplicate-key, fixture-order, hostile-order,
  semantic-preimage and package/readiness arithmetic checks.
- reconstructed synthetic fixture arrays and executed accepted
  `build_features`, strict `base_masks`, `source_preflight`,
  `channel_actions`, `channel_memories` and `epoch_support_ledger`.
- independently checked QF13 model values/access geometry, all anchor counts,
  reset state, structural event geometry and registered full/slice cases.
- compared all four prose argv commands with their machine arrays.
- exercised exact controller lease-CAS commands in a temporary bare
  repository.
- reproduced the child-lock race and O_EXCL partial-file dead state using
  temporary synthetic processes/files only.
- checked baseline filesystem and exclusive-rename capability.
- did not execute formal code or access historical/outcome data.

## Final

- Result: **FAIL**
- Severity: **P0/P1/P2/P3 = 0/4/1/0**
- Plan freeze: **NOT AUTHORIZED**
- Implementation: **NOT AUTHORIZED**
- Formal Q0 execution: **NOT AUTHORIZED**
- historical-cache access: `NONE`
- outcome access: `NONE`
