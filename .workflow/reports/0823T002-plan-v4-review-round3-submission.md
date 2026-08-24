# 0823T002 V4 Round 3 Independent Review Submission

Reviewer actor:
- codex-independent-reviewer-0823T002-v4-round3-43c97bc7

Candidate:
- commit=43c97bc70ba77df9dec864d81f4bfed5a81c1ac4
- tree_oid=9672ff574f9ffebc253ea0db5a53618cbd273ec9
- candidate_receipt_commit=9a6f64accf96a57d479a4b11ef11413eede2b841
- candidate_receipt_sha256=4c9bef2f7fcbd55e8e4ac1d3544a510665deb7599619aa382a807fd0944f9254

## Findings

### P2: Canonical attempts namespace can escape through a symlink

`require_canonical_formal_attempts_root()` resolves both the supplied path and
the canonical constant before comparing them. If the canonical attempts
namespace itself is a symlink to an external directory, both resolved values
remain equal. `durably_ensure_directory()` then follows that symlink, allowing
attempt bootstrap and receipt evidence to be written outside the repository.

The current `mutate_formal_attempts_root_escape` case covers a directly
selected external path, but not a symlink placed at the canonical namespace.
The namespace and its existing parent chain must reject symlinks, with a
production hostile mutation for this exact case.

### P2: Accepted-review regression still expects 79 artifacts

The frozen Surface Matrix has 65 surfaces, 87 negative mutations and 93
artifact assignments. The accepted-review branch in
`test_skhynix_stage_h0b_package.py` still asserts `artifact_count == 79`.
Pending review returns before that assertion, so the current suite hides a
deterministic post-acceptance failure.

## Confirmed

- All three Round 2 findings are closed.
- Production hostile replay passes `87 current + 87 frozen` mutations with
  `fail_open_count=0`.
- First namespace creation fsyncs each newly created parent entry.
- Duplicate attempt IDs use a no-replace claim, and bootstrap/receipt bind
  PID, dispatch, paths and `outcome_rerun`.
- Candidate, receipt, chronology, scope, plan, matrix, runtime and actor
  separation identities are correct.
- The task human-readable table matches the canonical matrix.
- The existing 42-file package and all research identities are unchanged.
- Core tests pass `124`; package tests pass `57`.
- No outcome, diagnostic, H0B0 or formal-build command was run.

## Disposition

- final_severity=P0/P1/P2/P3=0/0/2/0
- disposition=REJECTED
- formal_build=NO-GO
