# 0902T002 Project Controller Authority Correction 01

Date:
- 2026-09-02

Decision ID:
- `0902T002-PROJECT-CONTROLLER-AUTHORITY-CORRECTION-01`

Actor:
- existing hftbacktest Project Controller under the legacy Workflow v1.0
  coordination authority

Repository:
- `/Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery`

## 1. Correct Repository Roles

The Project Controller records the following authoritative repository
classification:

```text
workflow-proj =
  upstream Workflow Governance Kit development and release repository

hftbacktest =
  Kit-consuming project and sole owner of project task authority

workflow-proj != hftbacktest Project Controller repository
Kit Release Authority != Project Controller authority
```

`workflow-proj` may provide an immutable Kit release, Release Manifest,
release identity, verification instructions, and upstream design evidence.
It cannot register, arm, execute, accept, or close an hftbacktest project
task.

The hftbacktest Project Controller is the only actor that may authorize an
hftbacktest governance migration or successor project task.

## 2. Preserved Accepted Repair

Task `0902T001 / TARGET_PROJECT_ARGV_CONTRACT_REPAIR_V1` remains accepted and
closed without reinterpretation:

```text
implementation_commit =
  5eed10e59dcab91657bd332ca2b94ba3d2b7476b

independent_QA_commit =
  f5f05dfb5833b0a2cc019476543a244d5e82fcd5

independent_QA =
  P0/P1/P2/P3 = 0/0/0/0

accepted_target_commit =
  824e0431b96bda16515efb41544fd9e1feb78868

business_execution =
  false
```

The accepted argv contract remains:

```text
exec_argv = [python_executable, script_path, ...args]
program_argv = exec_argv[1:]
observed Python sys.argv == program_argv
observed complete command == [sys.executable, *sys.argv]
shell = false
```

This repair is target-project technical evidence. It is not Kit adoption,
governance-root migration, Q0 execution, or authority issued by
`workflow-proj`.

## 3. 0902T002 Disposition

The `0902T002` route incorrectly bound project authorization, arming, closure,
and active-task decisions to commits and reports in `workflow-proj`.

```text
root_cause =
  cross_repository_project_authority_source_misbinding

0902T002_status =
  作废

0902T002_route =
  superseded

R1_candidate =
  rejected_immutable

R2_candidate_file =
  absent

R2_plan_authoring_authority =
  revoked_before_creation

successor_Q0_registered =
  false

successor_Q0_executed =
  false
```

The frozen R1 candidate, Review Request, Addendum, Review Report, and all Git
commits remain immutable historical evidence. The `workflow-proj` Controller
Authorization and disposition records also remain historical planning
evidence, but they are non-authoritative for hftbacktest project execution.

No R2 plan may be created from the superseded route. No projection,
preflight, execution-ledger repository, arming, claim, attempt, receipt,
baseline, output, ref, tag, formal Q0, recovery, or business execution is
authorized by `0902T002`.

## 4. Correct Migration Classification

The target project already has a legacy Workflow v1.0/C0 human coordination
history, but it has no `.workflow/governance` root.

The verified immutable V2.1.1 release supports creation of a new G1
governance root while preserving existing C0 project history:

```text
target_project_state =
  existing_legacy_project_without_g1_governance_root

migration_class =
  legacy_c0_project_to_new_v2_1_1_g1_governance_root

migration_class != v2_1_0_online_upgrade
migration_class != empty_project_bootstrap
migration_class != v2_2_fresh_adoption_protocol
```

Existing Markdown tasks, reports, statuses, accepted artifacts, and failure
evidence remain historical C0 facts. They must not be converted into
canonical V2.1.1 transition authority merely because they exist.

## 5. Next Authorized Decision

The next Project Controller action is a separate, project-local authorization
for a new migration task:

```text
next_task_id =
  0902T003

next_task_title =
  WORKFLOW_GOV_KIT_V2_1_1_LEGACY_PROJECT_ADOPTION

next_scope =
  effect_free_migration_candidate_and_readiness_only
```

That separate authorization may permit migration plan authoring and
effect-free inventory design only. It must not execute `gov init`, create
`.workflow/governance`, change business or scientific logic, access private
or live resources, or execute a successor Q0.

## 6. Final State

```text
0902T001 = preserved_accepted
0902T002 = 作废
workflow_proj_project_authority = invalid
hftbacktest_project_controller_authority = retained
v2_1_1_migration_authorized = false_pending_separate_authorization
business_execution = false
```
