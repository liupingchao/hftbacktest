# 0823T002 V4 Round 4 Independent Review Submission

Reviewer actor:
- codex-independent-reviewer-0823T002-v4-round4-4df43f92

Candidate:
- commit=4df43f92e5a814f48cc0ab805f5205e9d23062e9
- tree_oid=ef959a59b5abf28a2a003072415cdb56bd81a20a
- candidate_receipt_commit=c522cb35d13b7a9581f5c733f27233c2c7d7cfc3
- candidate_receipt_sha256=facf84e08f39361dea8ed7e9b3a175366c8f9e7d3c1aca2e72781b3e89581869

## Finding

### P1: Hostile mutations can pass on the right code at the wrong target

On macOS, default `TemporaryDirectory()` paths are under `/var`, which is a
symlink. The new path-component guard therefore rejects the ambient temporary
root before at least five formal-attempt mutations reach their constructed
targets:

- `mutate_formal_partial_hard_stop_recovery`
- `mutate_formal_torn_receipt`
- `mutate_formal_attempts_root_escape`
- `mutate_formal_bootstrap_receipt_cross_binding`
- `mutate_formal_attempts_root_symlink`

Current and frozen hostile replay compare only the error code. They can report
`88 + 88` and `fail_open_count=0` even though those mutations did not execute
their declared rejection targets.

Hostile temporary roots must use a resolved non-symlink parent. Mutation
evidence must also bind the intended rejection location or target, and the
production suite must include a controlled parent-chain symlink case.

## Confirmed

- The Round 3 root and parent-chain symlink escape is functionally rejected
  before external evidence is written.
- The 96-artifact assertion executes while review is pending and passes on a
  simulated accepted path.
- Candidate, receipt, chronology, scope, Round 3 rejection history, plan,
  matrix, runtime and actor identities are correct.
- The matrix is `65 surfaces / 88 mutations / 96 artifacts`.
- The existing 42-file package and all research identities are unchanged.
- Core tests pass `125`; package tests pass `58`.
- No outcome, diagnostic, H0B0 or formal-build command was run.

## Disposition

- final_severity=P0/P1/P2/P3=0/1/0/0
- disposition=REJECTED
- formal_build=NO-GO
