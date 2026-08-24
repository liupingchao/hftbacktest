# SKHYNIX Stage H0-B Publication Portability Remediation Plan V3

Date: 2026-08-24

Status: candidate remediation authority; it becomes effective only after
independent review, exact task pinning and Surface Matrix pinning.

Task ID: `0823T002`.

The review authority is executable only when the pinned review file contains
the exact machine-readable contract:

```text
schema_version=skhynix_stage_h0b_v3_independent_review_v1
task_id=0823T002
reviewer_role=independent_read_only
reviewed_plan_sha256=<this exact V3 plan>
reviewed_surface_matrix_sha256=<current canonical matrix>
reviewed_runtime_source_tree_sha256=<task-pinned runtime tree>
final_severity=P0/P1/P2/P3=0/0/0/0
disposition=ACCEPTED
```

The task must independently state the same final severity. A syntactically
valid SHA of a rejected, pending or nonconforming review never authorizes
dispatch.

## 1. Scope And Authority

This V3 is a narrow remediation contract for the P1 found by independent QA
Round 1. It does not change:

- the V1 primary-analysis authority, SHA256
  `c1be0fdbd58f19c201c2faa7251621402486e6ebabf259af316b98bcf4c92b10`;
- the V2 Stage 4 diagnostic authority, SHA256
  `12b09677c0bcf2e921900f04e424ae7977e967ace70f391ab28c26fc4fb98a63`;
- the primary tuple, RQ1/RQ2/RQ3 formulas, seeds, gates or classification;
- `6600ms` as the unique primary or `850ms` as diagnostic-only/non-rescue;
- the primary-result payload, classification, primary-seal chronology/schema
  or Stage 4 aggregate chronology/projection.

The new Surface Matrix and runtime-source pins necessarily change the raw
bytes and SHA256 of `primary_result_seal.json`, both Stage 4 diagnostic permits
and both Stage 4 diagnostic receipts. Those contract-bearing wrappers are not
required to match QA Round 1. The sealed primary result bytes,
`primary_classification.json` and
`diagnostics/stage4_landmark_crosscheck.csv` must remain byte-identical.

V3 supersedes only the conflicting V2 statements that require raw,
root-specific runtime permits and ledgers to be copied byte-for-byte into the
portable package. Runtime authorization evidence and package publication
evidence are separate artifacts with separate schemas.

## 2. QA Round 1 Failure

Fresh-root QA reproduced:

- all primary research bytes and classification;
- the primary seal and Stage 4 diagnostic;
- research identity R and runtime-contract identity C;
- package path/type cardinality and zero-write admission.

It reproduced only `37/42` package files byte-for-byte. The two raw outcome
permits embedded absolute build roots and runtime PIDs; the two ledgers bound
those raw permit SHAs; the manifest therefore sealed a different E and
composite in every fresh work root.

This violates:

```text
kernel_package_admission_portable = true
fresh-root exact package parity
```

## 3. Runtime Authorization Evidence

Each isolated Build A/B root retains the exact runtime files:

```text
outcome_access_permit.json
outcome_access_ledger.json
```

The runtime permit schema remains:

```text
skhynix_stage_h0b_outcome_access_permit_v2
```

Its `status=admitted`, `fsynced=true`, absolute `resolved_build_root`,
positive real `runtime_pid`, runtime-source tree, semantic inventory,
preoutcome contract and support/input bindings remain mandatory. H0B1 and the
Stage 4 diagnostic validate these raw files before any authorized read.

The external formal build receipt uses:

```text
schema_version = skhynix_stage_h0b_build_receipt_v2
canonicalization = canonical JSON
unknown top-level or nested key policy = reject
```

Its exact top-level keys are:

```text
schema_version
task_id
build_a
build_b
semantic_source_inventory_sha256
expected_runtime_source_tree_sha256
build_envelopes_distinct
primary_result_seal
stage4_permit_build_a
stage4_permit_build_b
stage4_build_a
stage4_build_b
runtime_evidence_build_a
runtime_evidence_build_b
package
admission
aug07_event_rows_opened
network_private_order_cancel_live_access
```

Each `runtime_evidence_build_{a,b}` object has exactly:

```text
build_label
resolved_build_root
runtime_pid
build_envelope_sha256
runtime_outcome_access_permit_sha256
runtime_outcome_access_ledger_sha256
publication_outcome_access_permit_sha256
publication_outcome_access_ledger_sha256
publication_projection_contract_sha256
```

This receipt is outside the package and outside R/C/E. It preserves the exact
runtime forensic chain without making package identity root-dependent. The
Surface Matrix records
`.workflow/reports/0823T002-build-receipt.json` as required durable evidence
owned by `output_schema`.

Every nested receipt object is also exact. `primary_result_seal`, both Stage 4
permits and both Stage 4 receipts must equal their durable Build A/B files.
`package` must equal an independently recomputed production package summary,
and `admission` must equal a second zero-write `verify_package()` result.
Unknown, omitted or forged nested keys are rejected.

## 4. Publication Permit Projection

The package files retain their existing paths:

```text
outcome_access_permit_build_a.json
outcome_access_permit_build_b.json
```

They use a distinct schema and are not executable authorization permits:

```text
schema_version =
  skhynix_stage_h0b_outcome_access_permit_publication_v1
status = verified_runtime_projection
fsynced = true
```

Their exact top-level keys are:

```text
schema_version
task_id
build_label
status
fsynced
primary_plan_sha256
diagnostic_plan_sha256
diagnostic_review_sha256
surface_matrix_sha256
runtime_source_tree_sha256
preoutcome_contract_sha256
source_inventory_contract_sha256
semantic_source_inventory_sha256
publication_build_envelope
publication_build_envelope_sha256
support_replay_receipt_sha256
accepted_input_bindings_sha256
projection_contract_sha256
```

`publication_build_envelope` has exactly:

```text
build_label
build_root_role
process_role
runtime_source_tree_sha256
semantic_source_inventory_sha256
preoutcome_contract_sha256
```

Values are:

```text
build_root_role = build_a | build_b
process_role = H0B0
```

No absolute path, PID, hostname, inode, mtime or ctime is present.
`publication_build_envelope_sha256` is the canonical JSON SHA256 of this
object.

`projection_contract_sha256` is the canonical JSON SHA256 of:

```text
{
  "excluded_runtime_fields": ["resolved_build_root","runtime_pid"],
  "publication_schema_version":
    "skhynix_stage_h0b_outcome_access_permit_publication_v1",
  "runtime_schema_version":
    "skhynix_stage_h0b_outcome_access_permit_v2",
  "version": "h0b_outcome_permit_publication_projection_v1"
}
```

All other values are copied from a fully validated raw runtime permit.

## 5. Publication Ledger Projection

The package files:

```text
outcome_access_ledger_build_a.json
outcome_access_ledger_build_b.json
```

use:

```text
schema_version =
  skhynix_stage_h0b_outcome_access_ledger_publication_v1
```

The event key universe is unchanged. Only events in phases:

```text
permit
primary_outcome
```

may have the raw runtime permit SHA replaced by the corresponding publication
permit SHA. Diagnostic-permit and Stage 4 event SHAs are never rewritten.

Admission reconstructs the exact primary event list from
`preoutcome_source_inventory.csv`. For every admitted source row with role:

```text
r0_binance_bookticker
r0_hyperliquid_bbo
accepted_stage2_primary
accepted_stage3_primary
```

it requires exact ordered equality of:

```text
sequence
process_role
phase
relative_path
access_kind
bytes_read
permit_sha256
admitted
```

The access kind is:

```text
public_bbo_outcome_scan
  for r0_binance_bookticker and r0_hyperliquid_bbo

accepted_public_feature_metadata
  for accepted_stage2_primary and accepted_stage3_primary
```

Extra, missing, reordered, forged-path, forged-access-kind or byte-mismatched
events fail `H0B_OUTCOME_PERMIT_MISMATCH`.

## 6. Packaged Runtime Binding

Package admission reconstructs the original runtime-source inventory path
names from:

```text
runtime_source/skhynix_stage_h0b.py
runtime_source/skhynix_stage_h0b_contracts.py
runtime_tests/test_skhynix_stage_h0b.py
runtime_tests/test_skhynix_stage_h0b_package.py
```

It hashes their raw bytes under the original repository-relative path names.
The resulting runtime-source tree SHA256 must equal:

- the dispatch-pinned `expected_runtime_source_tree_sha256` in
  `.workflow/tasks/0823T002.md`;
- both publication permits;
- `preoutcome_contract.runtime_source_tree_sha256`;
- both Stage 4 diagnostic permits.

The task pin is frozen after implementation and before final V3 independent
review, hostile preflight or formal Build A. It is outside the package and is
the independent oracle that prevents a stale package from replacing all
internal runtime SHA fields self-consistently. A stale or arbitrarily
self-consistent runtime SHA fails
`H0B_OUTCOME_PERMIT_MISMATCH`.

The executable dispatch envelope also binds the raw task SHA256, exact V3 plan
path/SHA, exact V3 review path/SHA, Surface Matrix SHA and runtime-source tree
SHA. A hostile receipt from an older task or review cannot be reused even when
the generic task-validator summary is otherwise unchanged.

Direct `h0b0` and `outcome` entrypoints do not weaken this boundary. `h0b0`
must finish dispatch validation before creating a build root. `outcome` must
reconstruct and compare the current accepted-input bindings and preoutcome
contract, exact accepted H0-A support commitment and receipt, exact semantic
inventory bytes and the unopened two-event preoutcome ledger before any
outcome read. It must also rehash the current canonical Surface Matrix bytes
and require the matrix itself as an accepted-input binding. The
accepted-input bindings include the raw task SHA256, so a permit cannot
survive any task-text or matrix-byte change under an otherwise unchanged
review identity. Package admission repeats the same exact payload comparison
and requires the task SHA to equal both the controller task and packaged task
bytes.

## 7. Package And Identity Contract

The exact package remains:

```text
R17 / C10 / E14 / manifest1 = 42 regular files
5 exact directories
```

The four publication permit/ledger files remain in E. Their deterministic raw
bytes are included in E without Trust Kernel content normalization.

Fresh builds with different absolute roots and PIDs must reproduce:

```text
42/42 package file bytes
R
C
E
composite
h0b_manifest.json
```

Build A/B publication envelopes remain distinct by exact `build_label` and
`build_root_role`.

## 8. Required Negative And Regression Tests

Before formal rebuild:

1. two different root/PID runtime evidence pairs project to identical package
   permits and ledgers for the same build role;
2. Build A/B publication envelopes remain distinct;
3. runtime permit SHA in a diagnostic phase is rejected, not rewritten;
4. stale packaged runtime source is rejected;
5. forged primary path, access kind, byte count, event omission/addition and
   event reorder are rejected;
6. external receipt binds raw and publication permit/ledger SHAs;
7. two lightweight complete 42-file package builds from different root/PID
   inputs have byte parity and identical R/C/E/composite.
8. a package whose runtime files and all internal runtime SHA fields are
   replaced self-consistently still fails the dispatch-pinned runtime oracle.
9. hostile receipts from an older task SHA or V3 review SHA are rejected;
10. a self-consistent permit whose accepted-input bindings are forged is
    rejected by the direct outcome entrypoint;
11. an external receipt with an unknown or forged nested key is rejected;
12. failed dispatch creates no H0B0 build root;
13. two different root/PID Build A/B fixture pairs each pass the real
    `assemble_package()` and full `verify_package()`, then reproduce all
    `42/42` bytes and exact R/C/E/composite.
14. a pinned review with rejected/pending severity or non-accepted disposition
    is rejected before dispatch;
15. changing the current Surface Matrix bytes after H0B0 invalidates the
    direct outcome permit before any outcome read.

V3 items 4 through 8 each have a distinct Surface Matrix mutation ID and
current/frozen execution row:

```text
mutate_packaged_runtime_source
mutate_guarded_opener
mutate_external_receipt_binding
mutate_complete_package_portability
mutate_runtime_source_external_oracle
```

The complete hostile contract is `61 surfaces / 65 mutations`; reporting only
the first mutation per surface is forbidden.

Current/frozen hostile mutations, focused tests, Ruff, compileall and
`git diff --check` must pass before formal Build A.

The accepted Trust Kernel V1 source is immutable under `0823T002`. Its generic
`validate_research_package_task.py --negative-evidence` contract validates the
declared mutation IDs and observed error codes, but by itself does not prove
that the receipt belongs to the current H0-B dispatch, Surface Matrix or
runtime-source tree. It is therefore necessary but not sufficient for this
task.

The only valid H0-B negative-evidence admission entrypoint is:

```text
skhynix_stage_h0b.py gate0
```

This composed Gate 0 must first run the exact reviewed dispatch validation,
including the V3 review path/SHA and dispatch-pinned runtime-source tree. It
must then require the hostile receipt to bind the exact dispatch object,
current Surface Matrix SHA, current and frozen runtime-source SHA, the complete
ordered current/frozen 65-mutation contracts, exact mutation counts,
`fail_open_count=0` and all prohibited-access flags false. Finally it runs the
accepted generic Trust Kernel validator against that same receipt. Running the
generic validator alone must never satisfy H0-B Gate 0.

`build-formal` must call this composed Gate 0 before H0B0 and must independently
replay the hostile preflight to byte equality before any outcome phase starts.
Receipts from an older dispatch, matrix or runtime tree must fail even when
their mutation IDs and error codes remain syntactically valid.

The task's human-readable Surface Matrix is a generated canonical summary of
the JSON surface IDs, every mutation ID/error code, artifact counts and
identity layers. Dispatch compares the whole rendered table, not only the 61
surface IDs. While independent review is pending, both the review SHA and
`final_severity` remain explicitly pending.

## 9. Controlled Supersession Lifecycle

QA Round 1 occupies the canonical paths required by the formal rebuild:

```text
.workflow/reports/0823T002-build-a
.workflow/reports/0823T002-build-b
.workflow/reports/0823T002-build-receipt.json
local_live_analysis/skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002
```

No execution-time `rm`, `rm -rf` or ad hoc new output path is permitted. After
V3 independent acceptance, the controller must run:

```text
skhynix_stage_h0b.py retire-superseded
```

The command admits only the exact rejected Round 1 evidence:

```text
Build A tree SHA256 =
  980ce48e11fd278a8c73394816bb7c12a9387a049dbc2c788b3987da662456a2
Build B tree SHA256 =
  6f1f1b3fc5cc19b7d6d0e0da648828e5215b227b7f5f6836c38025e02262ddc2
build receipt SHA256 =
  0f1f828917fd90e34ff4aaf90e4704bb5e15c1b4aba0ecc3ecab0bc20f05881e
package tree SHA256 =
  61a24313c24679000496920045f018d561363d5538473b37ec73d43034590eb6
package manifest SHA256 =
  83513cfab3fe975cc3ef174c1f23993b0532927e182f0ba572171d0c06fec9b7
primary seal SHA256 =
  bdcc9925bed058a675e49bd2b0d3ce087e7bf41ac37beeb38873d7b76b2e8c5b
prior composite =
  a40c436510af3dce943cc20e44cb6fc017f80f1e0adaac94e1942c2f26656c37
```

It moves those exact entries with resumable `os.replace` operations into:

```text
.workflow/reports/0823T002-qa-round1-rejected-formal/
```

through a hidden staging directory, fsyncs the tree, writes an exact
`archive_receipt.json`, and atomically publishes the archive directory. It
never deletes an artifact. `build-formal` requires this archive receipt,
revalidates every archived identity, requires all four canonical paths to be
absent, and then rebuilds only at the original canonical paths.

## 10. Formal And QA Gates

The existing formal package and identities are superseded after this V3 is
independently reviewed and pinned.

The formal rerun must prove:

- primary result files, classification and Stage 4 aggregate research bytes
  unchanged from QA Round 1;
- new primary-seal, Stage 4 permit and Stage 4 receipt bytes differ only
  because reviewed matrix/runtime contract identities changed;
- a new C caused only by reviewed task/runtime/contract changes;
- portable E/composite and `42/42` package parity across formal Build A/B.

Independent QA Round 2 must use a fresh work root and different runtime
roots/PIDs. It may pass only after exact package bytes and all identities
match the new frozen formal package.
