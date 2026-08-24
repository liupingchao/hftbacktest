# SKHYNIX Stage H0-B Execution Authority And Recovery Plan V4

Date: 2026-08-24

Status: candidate governance authority. It becomes effective only after a
candidate revision is frozen, a controller candidate receipt is issued, and a
distinct independent reviewer accepts the exact candidate through the
attestation chain defined below.

Task ID: `0823T002`.

## 1. Scope

This V4 closes the post-handoff review at
`P0/P1/P2/P3=0/2/1/0`. The first V4 candidate was independently rejected
and preserved at commit `4413d36f`; its primary review reported
`P0/P1/P2/P3=0/3/2/1`, and a second read-only audit reported
`P0/P1/P2/P3=0/5/2/0`. Round 2 incorporates both reviews. It changes only
execution governance, failure recovery and review provenance. It does not
change or rerun:

- the primary or diagnostic research inputs;
- the H0-B formulas, seeds, folds, thresholds or classification precedence;
- `6600ms` as the sole primary latency;
- `850ms` as diagnostic-only and unable to rescue the primary;
- the admitted primary results, classification, primary seal or Stage 4
  aggregate;
- the admitted R/C/E/composite identities from formal commit
  `71adbfa678ff3646982160d220f5c223e0f7e59f`.

The existing 42-file package remains the formal research package. V4 is a
control-plane compatibility and recovery layer around that immutable
evidence.

## 2. Immutable Execution Authority

The package execution authority is the exact Git object:

```text
commit =
  71adbfa678ff3646982160d220f5c223e0f7e59f
tree_oid =
  4c15ab4f613178e1e9468db600885f091244aadc
task_path =
  .workflow/tasks/0823T002.md
task_sha256 =
  84333cf0b4915ef413b709f0a43db88fb9796d87b97ecfd24decea6ea32f1661
surface_matrix_sha256 =
  a523b91162c1783cff3e8ddbb70a4b91ad903b4757efa2ba7df8169cd8fb18df
runtime_source_tree_sha256 =
  ce52d3050ece7947db1185df672e089f819ff06df946b2b44e9e86c6afd5dd66
```

Package admission must compare all 42 packaged files byte-for-byte with the
corresponding package Git blobs at that commit. The accepted-input task SHA
must equal both the packaged task SHA and the pinned execution-authority task
SHA.

The accepted latency measurement manifest was not tracked by the formal
commit. Its authority is therefore the frozen accepted-input row only:

```text
bytes = 944
sha256 =
  8ac3b362e8d64cbd81232eaf7ed5856bada63ece20408e0d0b3fb5f84c562afd
```

Package verification must not read the current worktree copy as an oracle.

The current mutable task is not an execution-authority oracle for this
package. A normal workflow status edit therefore cannot invalidate the
package.

The authority commit must be an ancestor of the current candidate and handoff
revisions. The exact commit and tree identities are task-pinned and validated
without using worktree bytes.

## 3. Mutable Workflow State

The current `.workflow/tasks/0823T002.md` remains the human and workflow-kit
status surface. Its bytes may change for reviewed control-plane pins and the
normal:

```text
执行中 -> 待验收 -> 已通过 | 未通过 | 阻塞
```

state lifecycle.

The current task is validated as a control authority. It must never replace
the immutable execution authority of an already produced package.

While status is `执行中`:

```text
workflow_transition_receipt_sha256 = PENDING_HANDOFF
transition receipt path must not exist
```

While status is `待验收`:

```text
workflow_transition_receipt_sha256 = exact lowercase SHA256
transition receipt path must be a regular non-symlink file
```

Any other combination fails closed.

## 4. Workflow Transition Receipt

The handoff receipt uses canonical JSON:

```text
schema_version =
  skhynix_stage_h0b_workflow_transition_receipt_v1
```

Its exact keys are:

```text
schema_version
task_id
from_status
to_status
execution_authority_commit
execution_authority_tree_oid
execution_authority_task_sha256
formal_evidence_commit
package_manifest_sha256
research_data_identity
runtime_contract_identity
publication_envelope_identity
composite_package_identity
control_candidate_receipt_sha256
control_review_attestation_sha256
controller_actor_id
outcome_rerun
```

Required values include:

```text
from_status = 执行中
to_status = 待验收
formal_evidence_commit = execution_authority_commit
outcome_rerun = false
```

All package identities are recomputed from the existing package. The
transition receipt is outside the package and does not enter R/C/E. It
expresses workflow state only.

## 5. Versioned Formal Attempts

Every future formal execution uses a caller-supplied attempt ID and an
immutable versioned root:

```text
.workflow/reports/0823T002-formal-attempts/<attempt_id>/
```

Before the root is created, the controller atomically writes and fsyncs a
sibling bootstrap receipt. The root contains:

```text
attempt_receipt.json
attempt_bootstrap.json
build-a/
build-b/
package/
build-receipt.json
```

The attempt ID is restricted to lowercase ASCII letters, digits and hyphens.
The attempt root and bootstrap staging path must not exist before start. The
bootstrap is moved into the root and a durable `attempt_receipt.json` is
written before any Build A/B root or package staging path is created.

The receipt records:

```text
schema_version
task_id
attempt_id
status
phase
controller_pid
dispatch
paths
completed_entry_identities
error
outcome_rerun
```

Allowed statuses are:

```text
running
failed
interrupted
completed
```

Allowed phases are:

```text
initialized
gate0_validated
build_a_completed
build_b_completed
primary_sealed
stage4_completed
package_admitted
receipt_written
```

Every receipt update uses a same-directory temporary file, file fsync,
`os.replace()` and parent-directory fsync. Evidence files are never deleted
or overwritten. Every existing file and directory under the attempt root,
including hidden package staging, enters the exact evidence inventory.

The attempts root is fixed at the path above. CLI callers cannot override it.
The `h0b0`, `outcome`, `diagnostic-permit` and `diagnostic` entrypoints require
an active canonical attempt context and reject standalone use.

## 6. Failure And Recovery

Any caught exception after attempt creation writes:

```text
status = failed
error = {code, location, detail}
```

The partially produced versioned attempt root remains intact. A new attempt
uses a new attempt ID and does not require cleanup or a new hard-coded retire
command.

An uncatchable process termination may leave `status=running`, or may stop
after the bootstrap is durable but before the root receipt exists. Recovery
may mark either state `interrupted` only when the recorded PID is no longer
alive. A running receipt may differ from current disk while work is in
progress; recovery inventories and seals the observed tree. After status
becomes `failed`, `interrupted` or `completed`, any identity drift fails
closed.

Recovery is metadata-only. It does not reopen outcome inputs, resume a partial
research computation, delete files, move evidence into canonical paths or
overwrite an existing attempt. A fresh attempt is required for any new formal
research execution.

Historical `retire-*` commands remain readable for their already frozen
archives but are not part of the future formal execution lifecycle.

## 7. Atomic Package Staging

`assemble_package()` retains same-filesystem staging and atomic
`os.replace(staging, final)`.

If the deterministic staging path already exists, assembly fails:

```text
PUBLICATION_STAGING_EXISTS
```

It must not call `shutil.rmtree()` or otherwise remove the staging tree.
Versioned attempt roots prevent an unrelated attempt from sharing that path.

## 8. Candidate Revision Receipt

Before independent review, the controller freezes one Git candidate commit.
The candidate commit must not contain either the candidate receipt or the
review attestation.

The controller then issues:

```text
.workflow/reports/0823T002-v4-candidate-receipt-round2.json
```

with schema:

```text
skhynix_stage_h0b_v4_candidate_receipt_v2
```

and exact keys:

```text
schema_version
task_id
candidate_commit
candidate_tree_oid
candidate_parent_commit
controller_actor_id
execution_authority_commit
plan_path
plan_sha256
surface_matrix_path
surface_matrix_sha256
runtime_source_tree_sha256
review_path
review_submission_path
review_absent_in_candidate
review_submission_absent_in_candidate
candidate_receipt_absent_in_candidate
outcome_accessed
```

The task pins both the candidate receipt SHA256 and the exact Git commit that
introduced it. The validator reloads every candidate object and the receipt
blob from Git and recomputes all identities. `review_absent_in_candidate`,
`review_submission_absent_in_candidate`,
`candidate_receipt_absent_in_candidate` and `outcome_accessed=false` are
mandatory.

## 9. Independent Review Attestation

The independent review file uses machine-readable fields:

```text
schema_version=skhynix_stage_h0b_v4_independent_review_v1
task_id=0823T002
reviewer_role=independent_read_only
reviewer_actor_id=<workflow-issued reviewer actor ID>
controller_actor_id=<controller actor ID>
candidate_commit=<exact candidate commit>
candidate_tree_oid=<exact candidate tree object ID>
candidate_receipt_sha256=<exact candidate receipt>
reviewed_plan_sha256=<candidate V4 plan>
reviewed_surface_matrix_sha256=<candidate matrix>
reviewed_runtime_source_tree_sha256=<candidate runtime tree>
review_submission_sha256=<SHA256 of the independent reviewer submission>
final_severity=P0/P1/P2/P3=0/0/0/0
disposition=ACCEPTED
```

The independent reviewer submission is preserved separately at:

```text
.workflow/reports/0823T002-plan-v4-review-submission.md
```

The task pins its exact SHA, and `review_submission_sha256` in the attestation
must equal those bytes.

The reviewer actor ID must differ from the controller actor ID. The exact
chronology is:

```text
candidate commit
  -> candidate-receipt introduction commit
  -> review/submission introduction commit
  -> current HEAD
```

The review and submission must be introduced together. Their current bytes
must equal their exact Git blobs at that introduction commit. The candidate
receipt current bytes must likewise equal its introduction blob. Neither
review nor candidate receipt may exist in the candidate commit.

The candidate-receipt commit may change only the receipt path. The review
commit may change only the review and submission paths. The current
runtime-source tree must still equal the candidate runtime-source tree.

This proves repository-visible revision separation, exact candidate binding,
immutable attestation bytes and distinct workflow actor identifiers. It does
not claim cryptographic personhood: that would require an external signing
service. QA must evaluate the repository-visible control claim as stated, not
reinterpret it as proof of human identity.

## 10. Gate 0 Composition

Current Gate 0 has two independent checks:

1. current V4 control authority, matrix, runtime and current/frozen negative
   evidence;
2. immutable formal execution authority and existing package/legacy hostile
   evidence compatibility.

The current matrix and runtime are not substituted for the package's formal
authority. Conversely, the old formal authority does not authorize future
execution under unreviewed current code.

## 11. Required Regression And Negative Coverage

Before handoff:

1. existing package verifies while the current task is `执行中`;
2. the same package verifies after a valid simulated `待验收` transition;
3. status `待验收` without a receipt fails;
4. a forged transition package identity fails;
5. any of the 42 packaged files differing from commit `71adbfa6` fails;
6. mutable worktree latency-manifest bytes cannot alter package admission;
7. a candidate commit containing its own review or receipt fails;
8. same controller/reviewer actor IDs fail;
9. candidate commit/tree/runtime drift fails;
10. candidate-receipt/review chronology drift fails;
11. post-introduction receipt, review or submission rewrites fail;
12. bootstrap and attempt receipt exist before any build root is created;
13. a caught failure writes a durable failed receipt and preserves partial
    evidence;
14. dead-PID partial and root-before-receipt crashes can be sealed
    `interrupted`;
15. a live-PID running attempt cannot be marked interrupted;
16. duplicate attempt IDs fail without changing the prior attempt;
17. torn receipts fail with the stable attempt-state error;
18. non-canonical attempt roots and standalone formal subcommands fail;
19. an occupied package staging path fails without deletion;
20. receipt/review commits containing implementation changes fail;
21. the current and frozen hostile suites reject every declared mutation with
    `fail_open_count=0`.

## 12. Handoff

No H0-B outcome command is run for this remediation.

After implementation:

1. freeze the candidate commit;
2. issue and commit the candidate receipt;
3. obtain an independent actor-bound review attestation;
4. pin the accepted attestation;
5. run focused tests, static checks, current/frozen hostile preflight, current
   Gate 0 and zero-write verification of the existing 42-file package;
6. issue the workflow transition receipt;
7. change only the mutable task/report state to `待验收`;
8. submit QA Round 2 against the immutable formal authority plus V4 control
   chain.

No research R/C/E/composite identity may change during these steps.
