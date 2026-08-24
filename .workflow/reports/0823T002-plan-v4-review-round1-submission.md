# 0823T002 V4 Independent Review Round 1 Submission

- reviewer_actor_id=01a03142-480e-7553-a301-3fc23a718979
- candidate_commit=612bda21826fc430e2d21d12f96e3097f6f66c85
- candidate_receipt_commit=72e44ce5db79641d5bb6eae0d472ba18b2c65417
- candidate_receipt_sha256=8240669ad4434b1bbdc6ed7e63422cd719a0c757db33021bd293e9911b9b4111
- final_severity=P0/P1/P2/P3=0/3/2/1
- disposition=REJECTED

## Findings

### P1: Hard-stop partial evidence cannot enter recovery

Recovery validates `completed_entry_identities` against the current disk
before checking whether the recorded PID is dead. If a process is killed
after creating or extending Build A/B/package evidence but before the next
receipt phase update, the added evidence is treated as drift. Both `inspect`
and `mark-interrupted` then fail with
`H0B_FORMAL_ATTEMPT_STATE_MISMATCH`. Hidden package staging is also outside
the inventory.

### P1: Attempt receipt updates are not crash-atomic

`write_json()` directly overwrites `attempt_receipt.json`. A hard interruption
can leave truncated JSON. Creation of the attempt root is also not followed by
an fsync of the parent attempts directory, so the new directory entry is not
durably established before receipt publication.

### P1: Review provenance can be rewritten after reviewer introduction

The validator checks only the commit that first introduced the review path and
its author. It does not compare current review/submission bytes with the Git
blobs in that introduction commit. A later controller commit can therefore
rewrite the attestation, update mutable task pins and continue borrowing the
original reviewer introduction commit. It also does not require the candidate
receipt commit to be an ancestor of the review commit.

### P2: Formal attempts root can escape the frozen namespace

The CLI exposes arbitrary `--attempts-root`, and the runner does not require
the resolved path to equal the canonical `FORMAL_ATTEMPTS_ROOT`.

### P2: Negative coverage omits the new failure modes

The declared 74 mutations execute and reject with the correct codes, but the
matrix has no independently counted mutation for partial hard-stop recovery,
torn receipt, post-introduction review rewrite, receipt-before-review
chronology or attempts-root escape. Existing recovery tests cover only an
empty attempt with a dead PID.

### P3: Git tree object IDs are named SHA256

`candidate_tree_sha256` and `execution_authority_tree_sha256` contain 40-byte
hex Git object IDs, not SHA-256 digests. The binding works in the current
repository but the field names are inaccurate.

## Verified

- Candidate receipt validation passed.
- Candidate contains no candidate receipt, final review or final submission.
- Receipt commit is the candidate's direct child.
- Existing package remains `42 files`, `zero_write=true`, bound to
  `71adbfa678ff3646982160d220f5c223e0f7e59f`.
- R/C/E/composite, primary seal, classification and Stage 4 identities are
  unchanged.
- Focused tests passed `154`; all 74 declared mutation entrypoints rejected
  with their exact expected error codes.
- Surface Matrix is `65 surfaces / 74 mutations / 87 artifacts`.
- Ruff, compileall and scoped diff checks passed.
- No outcome or `build-formal` command was run.

## Result

Formal build is `NO-GO`. The candidate must be remediated and independently
reviewed again before QA Round 2 handoff.
