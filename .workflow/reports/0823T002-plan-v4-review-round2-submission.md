# 0823T002 V4 Round 2 Independent Review Submission

Reviewer actor:
- codex-independent-reviewer-0823T002-v4-round2-32c5ef62

Candidate:
- commit=32c5ef62d44d720e122aa7007b929fc288b66405
- tree_oid=66ac1acb2f41089f50feed2ddc6b49af823bb998
- candidate_receipt_commit=b0522750dd3008e3f339b53619a1492fec8b3edf
- candidate_receipt_sha256=7f4fb1467dfb663c3e25046b42bce1b9d785e753a048750b6a3d6860758a566f

## Findings

### P1: Frozen hostile execution lacks a Git object store

The production frozen loop copies files into a normal temporary directory,
while two mutations use Git helpers rooted at that directory. Current helper
and CLI execution pass 83/83, but frozen CLI execution passes only 81/83:

- `mutate_complete_package_portability` returns
  `H0B_EXECUTION_AUTHORITY_MISMATCH` instead of `H0B_BUILD_MISMATCH`.
- `mutate_review_receipt_chronology` returns
  `H0B_EXECUTION_AUTHORITY_MISMATCH` instead of
  `H0B_REVIEW_PROVENANCE_MISMATCH`.

Providing the same frozen tree with a read-only Git object store restores
83/83. The production `hostile_preflight` would therefore record
`fail_open_count=2`, which blocks Gate 0.

### P1: First creation of the attempts root is not durably published

The implementation creates the canonical attempts root and later fsyncs that
new directory, but it does not fsync the parent `.workflow/reports` directory
entry. A hard interruption can lose the entire attempt namespace even though
the bootstrap and attempt receipts were otherwise fsynced.

### P2: Concurrent duplicate attempt callers can cross-wire bootstrap state

The bootstrap staging existence check, staging write, root creation and
bootstrap publication do not form an atomic ownership claim. A deterministic
interleaving can produce:

```text
attempt_receipt.dispatch = caller A
attempt_bootstrap.dispatch = caller B
```

The current receipt validator accepts that mixed state. The protocol needs an
atomic no-replace claim and exact bootstrap/receipt cross-binding of
`attempt_id`, `controller_pid`, `dispatch` and `paths`.

## Confirmed

- All 42 package files equal the formal `71adbfa6` Git blobs.
- Package verification succeeds without a mutable worktree latency manifest.
- Immutable execution authority and mutable workflow transition are separate.
- Candidate/receipt chronology and receipt-only commit scope are correct.
- Current runtime equals the candidate runtime.
- The actor claim is correctly limited to repository-visible workflow
  separation, not cryptographic personhood.
- Focused tests pass: `173 passed`.
- Ruff, compileall and diff checks pass.
- No outcome, diagnostic or formal-build command was run.
- Research R/C/E/composite, primary seal and Stage 4 are unchanged.

## Disposition

- final_severity=P0/P1/P2/P3=0/2/1/0
- disposition=REJECTED
- formal_build=NO-GO
