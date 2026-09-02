# 0902T003 Project Controller Authorization R1

Date:
- 2026-09-02

Authorization ID:
- `0902T003-PROJECT-CONTROLLER-AUTHORIZATION-R1`

Actor:
- existing hftbacktest Project Controller under the legacy Workflow v1.0
  coordination authority

Decision:
- authorize one project-local Workflow Governance Kit V2.1.1 legacy-project
  adoption task and migration candidate/plan authoring only

## 1. Repository And Authority Separation

```text
Kit development and release repository =
  /Users/liu/Documents/workflow-proj

immutable Kit release directory =
  /Users/liu/Documents/vibe/workflow-gov-kit-V2.1.1

Kit-consuming project repository =
  git@github.com:liupingchao/hftbacktest.git

Project Controller authority =
  project-local hftbacktest legacy Workflow v1.0/C0 Controller

workflow-proj Project Controller authority =
  none
```

The Kit release repository may attest release bytes and lineage only. All
adoption, task, execution, acceptance, and closure authority for hftbacktest
originates inside hftbacktest.

## 2. Task And Worktree Identity

```text
task_id =
  0902T003

title =
  WORKFLOW_GOV_KIT_V2_1_1_LEGACY_PROJECT_ADOPTION

status_after_registration =
  待执行

active_phase =
  migration_candidate_plan_authoring

branch =
  codex/0902t003-workflow-v2-1-1-legacy-adoption

worktree =
  /Users/liu/Documents/hftbacktest-0902t003-workflow-v2-1-1-legacy-adoption

authoring_base_commit =
  9287e1f392ca8d28d7217de32dd414ea47acee79
```

The authoring base includes the accepted `0902T001` repair and the
project-local `0902T002` supersession. It creates no governance root.

## 3. Preserved Target Evidence

```text
0902T001_status =
  已通过

0902T001_implementation_commit =
  5eed10e59dcab91657bd332ca2b94ba3d2b7476b

0902T001_independent_QA_commit =
  f5f05dfb5833b0a2cc019476543a244d5e82fcd5

0902T001_QA =
  P0/P1/P2/P3 = 0/0/0/0

0902T001_accepted_target_commit =
  824e0431b96bda16515efb41544fd9e1feb78868

0902T002_project_correction_commit =
  d1e06fe5dce8400887d935f57c1710f758fce160

0902T002_project_coordination_commit =
  9287e1f392ca8d28d7217de32dd414ea47acee79

0902T002_status =
  作废
```

The argv repair remains target-project technical evidence. No `0831T001` or
`0902T002` mutable task, candidate, attempt, claim, arming, ref, tag, report,
or output identity may be reused by the migration.

## 4. Verified V2.1.1 Release Input

The immutable release directory was checked without modifying its bytes:

```text
release_name =
  workflow-gov-kit-V2.1.1

release_version =
  2.1.1

release_identity =
  346ae2d5232bc132de2189eb26219c6763480d05f74b298a85bdef2b3ca2509c

document_set_sha256 =
  47d672a0df3cfc496737c96c04080e76e761eb7bfe1151d44402815f5770a86d

release_manifest_file_sha256 =
  2789a3e4e2bdee93b7e1bf9cafcc08f2d590a2aa60ebecc5771c893de1330113

release_manifest_member_count =
  104
```

The migration candidate must independently reverify these values and the
release fileset before proposing any target mutation.

## 5. Migration Classification

At authorization time:

```text
legacy_project =
  true

legacy_coordination_model =
  Workflow_v1_0_C0

target_governance_root =
  .workflow/governance

target_governance_root_state =
  absent

migration_class =
  existing_legacy_c0_project_new_v2_1_1_g1_root_adoption

migration_class !=
  v2_1_0_to_v2_1_1_online_upgrade

migration_class !=
  empty_project_bootstrap

V2_2_dependency =
  none
```

V2.1.1 explicitly permits existing C0 projects to remain valid until
explicit migration and permits migration one task class at a time. Its
`gov init` command creates a new governance root. The target is not an
existing V2.1.0 governance root, so the V2.1.0 online-upgrade prohibition does
not apply.

Historical Markdown tasks, reports, statuses, dashboards, accepted artifacts,
failed attempts, and scientific evidence remain under their historical
contracts. They may be inventoried but must not be silently converted into
V2.1.1 transition-ledger authority or Accepted Registry entries.

## 6. Current Authorized Deliverable

The current actor may author one immutable revision-specific migration
candidate:

```text
candidate_id =
  0902T003-V2_1_1-LEGACY-ADOPTION-CANDIDATE-R1

candidate_path =
  .workflow/plans/0902T003/workflow-gov-kit-v2-1-1-legacy-project-adoption-r1.md
```

The candidate must define:

- exact release verification and source-path rules;
- exact target Git, worktree, filesystem, and `.workflow/governance`
  absence observations;
- a complete legacy evidence inventory and non-conversion policy;
- actor names and the V2.1.1 Project Governance Profile;
- the exact proposed `gov init` command without executing it;
- isolated materialization, atomicity, rollback, cleanup, and restart rules;
- observe-only rollout and one effect-free governance smoke task;
- independent plan review, preflight review, adoption acceptance, and
  post-adoption audit gates;
- proof that `0902T001` stays accepted and unchanged;
- proof that no business, scientific, private, live, or historical-outcome
  computation occurs during migration.

The migration candidate and its Review Request must use separate commits. The
candidate must receive independent `P0/P1/P2/P3 = 0/0/0/0` plan/readiness
acceptance before any preflight or materialization.

## 7. Required Chain

```text
project-local Controller Authorization
-> task registration
-> migration candidate/plan
-> independent plan/readiness review
-> effect-free release and target preflight
-> independent preflight acceptance
-> Project Controller adoption authorization
-> isolated V2.1.1 governance-root materialization
-> project-local adoption receipt and state audit
-> independent migration QA
-> Project Controller migration closure
-> effect-free governed smoke task
-> separate decision on successor Q0
```

No successor Q0 is authorized by migration registration or migration success.

## 8. Current Locks

```text
migration_plan_authoring =
  authorized

migration_plan_review =
  pending_candidate

effect_free_preflight =
  locked

gov_init =
  locked

governance_root_materialization =
  locked

adoption_cutover =
  locked

governed_smoke_task =
  locked

successor_Q0_registration =
  locked

formal_Q0_execution =
  locked

business_execution =
  false
```

This authorization creates no `.workflow/governance` path, canonical object,
profile, adoption object, transition, authority, capability, attempt, receipt,
registry entry, generated view, claim, baseline, output root, ref, or tag.

## 9. Stop Conditions

Stop and report `阻塞` if:

- the worktree does not descend from exact authoring base
  `9287e1f392ca8d28d7217de32dd414ea47acee79`;
- `.workflow/governance` already exists or appears during authoring;
- V2.1.1 release identity, manifest, fileset, or source path differs;
- migration would require modifying the immutable release directory;
- historical Markdown or accepted artifacts would need to become canonical
  authority without a new reviewed object;
- a V2.1.0 governance root is discovered;
- business, scientific, private, live, or historical-outcome execution would
  occur;
- `workflow-proj` authority would be required for a project-local decision;
- unrelated worktree changes cannot be isolated.

## 10. Handoff

```text
next_actor =
  0902T003_migration_candidate_plan_author

business_execution =
  false

governance_root_created =
  false

migration_executed =
  false
```
