# 0902T002 Successor Q0 Recovery And Effect-Free Preflight Plan R1

Date:
- 2026-09-02

Revision:
- `R1`

Candidate ID:
- `0902T002-SUCCESSOR-Q0-CANDIDATE-R1`

Candidate path:
- `.workflow/plans/0902T002/successor-q0-recovery-and-effect-free-preflight-r1.md`

Candidate state at authoring:
- `FROZEN_PLAN_ONLY`

## 0. Decision And Current Boundary

This revision defines a new, task-scoped successor route for the failed
`0831T001` Q0 attempt. It does not rehabilitate, consume, recover or rerun any
`0831T001` mutable identity.

This revision authorizes no execution by itself. The current author stops
after:

1. freezing this candidate in a candidate-only commit;
2. updating mutable authoring status to `待验收`;
3. creating an independent Review Request.

The following remain false after this authoring handoff:

```text
projection_materialized = false
effect_free_preflight_executed = false
controller_arming_authorized = false
claim_created = false
attempt_created = false
baseline_created = false
output_root_created = false
controller_repository_created = false
controller_ref_created = false
task_tag_created = false
formal_Q0_executed = false
business_execution = false
locked_data_accessed = false
```

## 1. Authority

### 1.1 Target repository identity

```text
repository =
  git@github.com:liupingchao/hftbacktest.git

worktree =
  /Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery

branch =
  codex/0902t002-successor-q0-recovery

authoring parent HEAD =
  e2fdd395049e42b8e093f781b76af6718a06318f

accepted target base =
  824e0431b96bda16515efb41544fd9e1feb78868
```

The authoring parent and its upstream both resolved to the exact authoring
parent HEAD before this file was created. The accepted target base is an
ancestor of that HEAD.

### 1.2 Controller authority

```text
Controller Authorization ID =
  0902T002-CONTROLLER-AUTHORIZATION-R1

Controller Authorization path =
  /Users/liu/Documents/workflow-proj/.workflow/reports/0902T002-controller-authorization-r1.md

Controller Authorization commit =
  288cf86ea4db8abc47477cae5f4a955020786190

Controller Authorization SHA256 =
  5f3c1062b526884959240e28adb3684df08ab1f038526871f8b2e39e27d17334

Controller active-boundary commit =
  8894646cea97b4c1e980b65ff0018d5e265e202b
```

### 1.3 Accepted argv repair

```text
repair implementation =
  5eed10e59dcab91657bd332ca2b94ba3d2b7476b

independent QA =
  f5f05dfb5833b0a2cc019476543a244d5e82fcd5

QA severity =
  P0/P1/P2/P3 = 0/0/0/0

target final accepted state =
  824e0431b96bda16515efb41544fd9e1feb78868

Controller acceptance =
  b3cbdc3867cf8ea3d629707bea000498d410ec20
```

## 2. Immutable Source Inputs

The following paths and exact bytes are design or implementation source
material. They are not successor execution authority until a later
task-scoped projection is materialized, frozen and accepted.

| Input | Path | SHA256 |
| --- | --- | --- |
| Workflow Kit V2.1.1 | external immutable release | `346ae2d5232bc132de2189eb26219c6763480d05f74b298a85bdef2b3ca2509c` |
| master scientific protocol | `docs/skhynix_trade_led_depth_follower_transition_hazard_master_protocol_20260831.md` | `4ac0772ae4f2bdf29e6572e22092108de293ec05deeaa77679d606cf1e4c0d40` |
| Revision 26 source plan | `docs/skhynix_trade_led_depth_follower_q0_pipeline_qualification_execution_plan_20260831.md` | `bdc934202cd9ee9e1743830121eec80f1cf3ab7e8bb4f3bbc1f8728c3619f7dc` |
| scientific implementation | `examples/hyperliquid/skhynix_trade_led_depth_follower_transition_hazard.py` | `c0364ca4ee736ace322b330f2a24654cf7c7bb3e19138a798a5343cd6fc609af` |
| repaired runner source | `examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification.py` | `efdaca45419b1e87573be68e3ec0bd398a344d8cb39194cb3c95a514811c4dc6` |
| repaired verifier source | `examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_verifier.py` | `e680b2b8413cac900a36419aef662dbff700b7fa43af864af5d8adefb5e9eced` |
| repaired test source | `examples/hyperliquid/test_skhynix_trade_led_depth_follower_q0_pipeline_qualification.py` | `4a1b763f62ff00f7e667a40ca2cf263bcab4708a7cd2e28bff50819b33928601` |

The complete Revision 26 scientific, fixture, package, causal-access,
negative-boundary and one-shot semantics remain the source model except where
this revision explicitly replaces an `0831T001` operational identity with a
new `0902T002` identity.

No Tardis, market result, historical cache, future outcome, private data or
live data may influence the projection.

## 3. Why A Task-Scoped Projection Is Required

The accepted repaired runner source still contains hard-coded `0831T001`
task, truth, surface, claim, receipt, report, baseline, tag, controller,
worktree and attempt-root identities.

Therefore:

```text
accepted repaired runner source
  != directly executable 0902T002 successor authority
```

Running it unchanged would require reuse of forbidden `0831T001` identities
and would fail the new worktree/cwd bindings. This plan forbids that route.

The `successor candidate/plan` top-level node contains two subnodes:

```text
R1 plan freeze and independent review
-> separately authorized task-scoped projection materialization
```

This turn completes only the first subnode. Projection materialization:

- is not authorized by this file alone;
- must be separately enabled by Controller state after R1 is accepted;
- must complete before effect-free preflight;
- may change only task-scoped operational identity bindings;
- must not change scientific logic, schema shape, thresholds, fixtures'
  scientific meaning, package semantics or one-shot state-machine meaning.

If a Workflow Kit change, schema change or scientific-logic change is found
necessary, the route stops as `阻塞`.

## 4. Reserved Successor Identities

All values in this section are reserved by the plan but remain absent during
R1 authoring.

### 4.1 Candidate and projection

```text
candidate_id =
  0902T002-SUCCESSOR-Q0-CANDIDATE-R1

candidate_path =
  .workflow/plans/0902T002/successor-q0-recovery-and-effect-free-preflight-r1.md

projected truth path =
  .workflow/contracts/0902T002-fixture-truth-v1.json

projected surface path =
  .workflow/contracts/0902T002-q0-surface-contract-v1.json

projected runner path =
  examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_0902t002.py

projected verifier path =
  examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_0902t002_verifier.py

projected test path =
  examples/hyperliquid/test_skhynix_trade_led_depth_follower_q0_pipeline_qualification_0902t002.py

projection receipt ID =
  0902T002-PROJECTION-RECEIPT-R1

projection receipt path =
  .workflow/reports/0902T002-projection-receipt-r1.json

projection implementation commit message =
  feat: materialize 0902T002 Q0 authority projection
```

The projection commit and projected file SHA256 values are intentionally not
invented in this plan. They become authority only when the files exist in a
later candidate-only implementation commit and are recorded by exact full
commit and SHA256 in later Controller and preflight evidence.

### 4.2 Formal attempt and output

```text
formal attempt ID =
  0902T002-FORMAL-001

output identity =
  0902T002-Q0-OUTPUT-V1

formal cwd =
  /Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery

formal attempt root =
  /Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery/local_live_analysis/skhynix_trade_led_depth_follower_q0_0902T002_formal_v1

formal package root =
  /Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery/local_live_analysis/skhynix_trade_led_depth_follower_q0_0902T002_formal_v1/package

baseline identity =
  0902T002-Q0-BASELINE-V1

baseline path =
  baselines/skhynix_trade_led_depth_follower_q0_0902t002_v1
```

### 4.3 Claim, receipt and reports

```text
armed claim ID =
  0902T002-ARMED-CLAIM-R1

armed claim =
  .workflow/attempt-claims/0902T002.armed.json

claimed claim ID =
  0902T002-CLAIMED-CLAIM-R1

claimed claim =
  .workflow/attempt-claims/0902T002.claimed.json

consumption receipt =
  .workflow/attempt-receipts/0902T002.consumption-push.json

terminal receipt =
  .workflow/attempt-receipts/0902T002.terminal.json

effect-free preflight report =
  .workflow/reports/0902T002-effect-free-argv-preflight-r1.json

execution report ID =
  0902T002-EXECUTION-REPORT-R1

execution report =
  .workflow/reports/0902T002-execution.md

business report ID =
  0902T002-BUSINESS-REPORT-R1

business report =
  .workflow/reports/0902T002-business.md

independent QA report ID =
  0902T002-QA-REPORT-R1

independent QA report =
  .workflow/reports/0902T002-qa.md

Controller closure report ID =
  0902T002-CONTROLLER-CLOSURE-R1

Controller closure report =
  /Users/liu/Documents/workflow-proj/.workflow/reports/0902T002-controller-closure-r1.md
```

### 4.4 Controller and tags

```text
controller bare repository =
  /Users/liu/Documents/hftbacktest-0902t002-q0-controller.git

controller ref =
  refs/heads/codex/0902T002-controller-ledger

implementation tag =
  skhynix-trade-led-depth-follower-q0-0902t002-implementation-v1

consumption tag =
  skhynix-trade-led-depth-follower-q0-0902t002-consumed-v1

terminal tag =
  skhynix-trade-led-depth-follower-q0-0902t002-terminal-v1

recovery witness tag =
  skhynix-trade-led-depth-follower-q0-0902t002-recovery-start-v1

arming commit message =
  audit: arm 0902T002 Q0 attempt
```

### 4.5 Arming authority

```text
Controller projection/preflight authorization ID =
  0902T002-CONTROLLER-PROJECTION-PREFLIGHT-AUTHORIZATION-R1

Controller projection/preflight authorization path =
  /Users/liu/Documents/workflow-proj/.workflow/reports/0902T002-controller-projection-preflight-authorization-r1.md

Controller arming authorization ID =
  0902T002-CONTROLLER-ARMING-AUTHORIZATION-R1

Controller arming authorization path =
  /Users/liu/Documents/workflow-proj/.workflow/reports/0902T002-controller-arming-authorization-r1.md
```

The arming authorization commit and SHA256 do not exist at R1 authoring time.
They must be produced only after accepted effect-free preflight evidence.

### 4.6 Review evidence

```text
Review Request ID =
  0902T002-PLAN-REVIEW-REQUEST-R1

Review Request path =
  .workflow/reports/0902T002-plan-review-request-r1.md

Review Report ID =
  0902T002-PLAN-REVIEW-R1

Review Report path =
  .workflow/reports/0902T002-plan-review-r1.md
```

## 5. Freshness And Non-Reuse Rules

Before projection materialization and again before preflight, verify that
every Section 4 path, tag, ref, controller repository, attempt root and
baseline is absent.

The following are forbidden as successor authority:

- task ID `0831T001`;
- any `0831T001` candidate or attempt ID;
- `.workflow/attempt-claims/0831T001.*`;
- `.workflow/attempt-receipts/0831T001.*`;
- the old formal cwd, attempt root or baseline;
- the old controller bare repository or controller ref;
- any old implementation, consumption, terminal or recovery tag;
- any old execution, business, QA or closure report;
- the old armed claim bytes, even if copied to a new path;
- the old arming, consumption or terminal commit;
- any mutable output produced by the failed attempt.

The scientific protocol and source implementation bytes may be derived from
only through the exact Section 2 identities.

## 6. Projection Materialization Contract

Projection materialization is a future effect-free engineering operation. It
must be performed with structured JSON and AST-aware source handling. Blind
global string replacement is forbidden.

### 6.1 Allowed operational substitutions

The projection may replace only:

- task ID and task path;
- candidate plan path and SHA binding;
- truth and surface contract paths;
- runner, verifier and test paths;
- formal cwd, readiness roots, attempt root and package root;
- claim, receipt, report and baseline paths;
- controller repository and ref;
- implementation, consumption, terminal and recovery tag names;
- task-specific commit messages;
- hashes and canonical aggregates that are mechanically derived from those
  operational identity fields.

### 6.2 Frozen semantic equality

After removing the explicitly allowed operational identity fields from both
source and projection, canonical structured comparison must be equal for:

- fixture scientific values and event ordering;
- QF01-QF15 semantics;
- feature, base-mask and causal-access formulas;
- slice/reset and A/B/P semantics;
- package schemas and structural comparison;
- error precedence;
- terminal classification;
- at-most-once child and claim semantics;
- controller CAS and recovery semantics;
- hostile mutation meaning;
- all scientific thresholds and zero outcome-access rules.

The scientific implementation file remains byte-identical to its Section 2
SHA256.

### 6.3 Required projection evidence

The future projection commit must contain only the projected truth, surface,
runner, verifier and test files plus a revision-specific projection receipt.
That receipt must record:

- this R1 path, freeze commit and SHA256;
- all Section 2 source path/SHA256 pairs;
- every projected path/SHA256 pair;
- the exact allowed-substitution inventory;
- canonical semantic-equality results;
- source and projected test inventory;
- proof that no `0831T001` mutable path is selected by the projection;
- proof that all Section 4 reserved execution identities remain absent.

No claim, receipt, tag, ref, baseline, attempt root or controller repository
may be created by projection materialization.

## 7. Identity Dependency DAG

```text
Controller Authorization R1
  -> target registration commit
  -> R1 candidate freeze commit and file SHA256
  -> independent R1 plan/readiness acceptance
  -> Controller projection/preflight authorization
  -> task-scoped projection implementation commit and file SHA256 set
  -> static projection tests and readiness evidence
  -> effect-free argv preflight evidence
  -> Controller arming authorization
  -> annotated implementation tag
  -> armed claim bytes
  -> arming commit
  -> exact one-shot formal command
  -> claimed claim and consumption commit/tag/ref receipt
  -> producer/verifier terminal result
  -> terminal commit/tag/ref receipt
  -> execution and business reports
  -> independent QA
  -> Controller closure
```

No child may construct or authorize an ancestor. Mutable coordination files
do not grant authority to any node in this DAG.

## 8. Ownership Table

| Object | Sole writer | Required predecessor | Forbidden writer |
| --- | --- | --- | --- |
| R1 plan | plan author | Controller Authorization R1 | reviewer, executor |
| R1 Review Request | plan author | frozen R1 | reviewer |
| R1 Review Report | independent reviewer | Review Request and frozen R1 | plan author |
| projection files/receipt | future projection actor | accepted R1 and Controller projection authorization | current R1 author |
| preflight report | future preflight actor | accepted projection bytes | current R1 author, formal runner |
| arming authorization | Controller | accepted preflight | projection actor, formal runner |
| implementation tag | arming actor | arming authorization | current R1 author |
| armed claim | arming actor | exact implementation tag and arming authorization | formal child |
| claimed claim | formal outer driver | validated armed claim and no-replace attempt root | reviewer |
| consumption receipt | formal/recovery state machine | consumption transition | producer, verifier |
| terminal receipt | formal/recovery state machine | durable process state | reviewer |
| baseline | terminal PASS path only | valid PASS terminal package | FAIL/recovery blocker path |
| execution/business report | formal terminalizer | terminal receipt | reviewer |
| QA report | independent QA | terminal evidence | formal runner |
| Controller closure | Controller | independent QA | executor |

## 9. Effect-Free Argv Preflight

### 9.1 Bound runtime

```text
runtime path =
  /Users/liu/.local/conda/bin/python

runtime resolved path =
  /Users/liu/.local/conda/bin/python3.13

runtime SHA256 =
  333e66ec89afec4a6295f1afb6c50da4d9f5629ed5c3f831e9868bf8c5479b9f

shell =
  false
```

### 9.2 Argv contract

```text
exec_argv =
  [python_executable, script_path, ...args]

program_argv =
  exec_argv[1:]

observed Python sys.argv =
  program_argv

observed complete command =
  [sys.executable, *sys.argv]

observed Python sys.argv != exec_argv
shell = false
```

### 9.3 Preflight command construction

After projection freeze, substitute the exact projected runner SHA256 recorded
by the projection receipt into this token array:

```text
[
  "/Users/liu/.local/conda/bin/python",
  "examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_0902t002.py",
  "--qualify-argv-contract",
  "--expected-runtime",
  "/Users/liu/.local/conda/bin/python",
  "--expected-runtime-sha256",
  "333e66ec89afec4a6295f1afb6c50da4d9f5629ed5c3f831e9868bf8c5479b9f",
  "--expected-script-sha256",
  "<EXACT_PROJECTED_RUNNER_SHA256_FROM_PROJECTION_RECEIPT>",
  "--expected-cwd",
  "/Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery"
]
```

The placeholder is a construction rule, not authority. A Controller
projection/preflight authorization must bind one exact 64-lowercase-hex value
before execution. Any unresolved placeholder is an automatic refusal.

### 9.4 Preflight side-effect boundary

The later preflight:

- writes evidence only to captured stdout until a separate evidence actor
  persists the exact stdout bytes;
- creates no claim, receipt, tag, ref, baseline, attempt root, output root or
  controller repository;
- does not read truth, surface, cache, package, future outcome, private or
  live data;
- verifies accepted repair commit `5eed10e59dcab91657bd332ca2b94ba3d2b7476b`;
- verifies Workflow Kit V2.1.1 identity
  `346ae2d5232bc132de2189eb26219c6763480d05f74b298a85bdef2b3ca2509c`;
- verifies projection commit, runner path and runner SHA256;
- verifies runtime path, resolution, bytes, cwd, argv and `shell = false`;
- compares before/after Git status, refs, tags and reserved-path inventories.

Expected successful evidence includes:

```text
business_execution = false
effectful_outputs = false
successor_q0 = ABSENT
observed_sys_argv = program_argv
shell = false
```

Drift or observation failure exits fail closed and does not authorize arming.

## 10. Arming Gate

Controller arming authorization may be created only when:

1. independent R1 review is accepted at `P0/P1/P2/P3 = 0/0/0/0`;
2. projection implementation bytes are frozen and all projection checks pass;
3. effect-free preflight passes with no before/after mutation;
4. every reserved claim, receipt, attempt, baseline, controller, ref and tag
   remains absent;
5. target HEAD/upstream and all bound hashes match;
6. no locked data has been accessed.

Arming then performs this exact chronology:

```text
projection implementation commit
-> annotated implementation tag
-> atomically publish armed claim
-> stage only armed claim
-> arming commit, parent = projection implementation commit
-> formal outer driver starts with HEAD = arming commit
```

The armed claim must bind:

- task and attempt IDs;
- R1 path, freeze commit and SHA256;
- R1 accepted review identity;
- projection commit and projected file SHA256 values;
- master protocol and scientific implementation SHA256;
- runtime path, resolved path and SHA256;
- exact formal cwd and complete formal argv;
- exact claim, receipt, output, baseline, controller, ref and tag identities;
- Controller arming authorization commit/SHA256;
- zero historical-cache, future-outcome, private and live authorization.

## 11. Exact Formal And Recovery Commands

These commands are frozen as token arrays and are not executed by R1
authoring.

### 11.1 Outer formal command

```text
[
  "/Users/liu/.local/conda/bin/python",
  "examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_0902t002.py",
  "--formal",
  "--claim",
  ".workflow/attempt-claims/0902T002.armed.json",
  "--attempt-root",
  "/Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery/local_live_analysis/skhynix_trade_led_depth_follower_q0_0902T002_formal_v1"
]
```

### 11.2 Recovery command

```text
[
  "/Users/liu/.local/conda/bin/python",
  "examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_0902t002.py",
  "--recover",
  "--attempt-root",
  "/Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery/local_live_analysis/skhynix_trade_led_depth_follower_q0_0902T002_formal_v1"
]
```

### 11.3 Producer child command

```text
[
  "/Users/liu/.local/conda/bin/python",
  "examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_0902t002.py",
  "--formal-producer",
  "--attempt-root",
  "/Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery/local_live_analysis/skhynix_trade_led_depth_follower_q0_0902T002_formal_v1",
  "--package-root",
  "/Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery/local_live_analysis/skhynix_trade_led_depth_follower_q0_0902T002_formal_v1/package",
  "--truth",
  ".workflow/contracts/0902T002-fixture-truth-v1.json",
  "--surface-contract",
  ".workflow/contracts/0902T002-q0-surface-contract-v1.json",
  "--runtime-lock-fd",
  "198",
  "--handoff-ack-fd",
  "199"
]
```

### 11.4 Terminal verifier child command

```text
[
  "/Users/liu/.local/conda/bin/python",
  "examples/hyperliquid/skhynix_trade_led_depth_follower_q0_pipeline_qualification_0902t002_verifier.py",
  "--package-root",
  "/Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery/local_live_analysis/skhynix_trade_led_depth_follower_q0_0902T002_formal_v1/package",
  "--result",
  "/Users/liu/Documents/hftbacktest-0902t002-successor-q0-recovery/local_live_analysis/skhynix_trade_led_depth_follower_q0_0902T002_formal_v1/control/terminal_verifier_result.json",
  "--runtime-lock-fd",
  "198",
  "--handoff-ack-fd",
  "199"
]
```

Every command uses `shell = false`. No alternate formal command is permitted.

## 12. One-Shot State Machine

The formal state machine retains Revision 26 semantics with only the new
Section 4 identities.

```text
PRE_ARMED
-> ARMED_UNCONSUMED
-> ATTEMPT_ROOT_COMMITTED
-> CLAIM_CONSUMED
-> CONSUMPTION_LEDGER_DURABLE
-> PRODUCER_TERMINAL
-> VERIFIER_TERMINAL
-> Q0_TERMINAL
-> QA_REVIEWED
-> CONTROLLER_CLOSED
```

Rules:

1. Before no-replace attempt-root creation, infrastructure observation
   failure is retryable only after proving zero claim consumption and zero
   task-scoped state.
2. Successful no-replace attempt-root creation consumes the task's sole
   formal-attempt right.
3. The armed claim is atomically renamed to byte-identical claimed state and
   may never be recreated.
4. Producer and verifier invocation claims are each at-most-once.
5. Recovery may continue durable publication and terminalization but may not
   create a second producer or verifier invocation.
6. A valid terminal receipt is immutable.
7. PASS may publish the exact baseline; FAIL and blocker paths may not.
8. Controller divergence is workflow integrity evidence, not a rewritten Q0
   scientific result.
9. No post-formal regeneration is permitted.

## 13. Recovery Matrix

| Durable state | Allowed next action | Forbidden action |
| --- | --- | --- |
| no attempt root, armed claim intact | stop or retry only after Controller authorization remains valid | create claimed state |
| attempt root exists, armed claim present | exact recovery continues claim consumption | recreate attempt root |
| claimed exists, consumption commit/tag/ref incomplete | finish exact registered consumption transition | run producer twice |
| producer invocation exists, exit missing | classify producer interruption after lock acquisition | relaunch producer |
| producer exit durable, verifier invocation absent | create the sole verifier invocation | alter producer receipt |
| verifier invocation exists, exit missing | classify verifier interruption after lock acquisition | relaunch verifier |
| process receipts durable, terminal receipt absent | derive one terminal result from durable-state resolver | regenerate package |
| terminal receipt durable, local terminal Git incomplete | complete local commit/tag and eligible controller transition | rewrite receipt/report |
| unexpected controller ref before terminal | publish blocker, consume only unfinished local claim state | push or create terminal result |
| unexpected controller ref after terminal | preserve terminal bytes; complete only permitted local Git state | change classification |
| recovery witness already committed | continue from exact witness identity | create a second witness identity |

The projected surface must retain Revision 26's complete crash-state,
publication, quarantine, controller-observation and raw Git integrity tables
after operational identity projection.

## 14. Locked Data And Effect Boundary

Before valid Controller arming authorization and exact formal invocation:

- historical cache access is `NONE`;
- future outcome access is `NONE`;
- private and live data access is `NONE`;
- truth/surface content may be read only during later static projection and
  readiness checks, never as business evidence;
- no classification, prediction, market conclusion or strategy result may
  be produced.

Q0 remains deterministic software qualification on the frozen synthetic
fixture model. PASS cannot be described as market support or trading
readiness.

## 15. Error Precedence And Fail-Closed Rules

Before preflight:

1. authority commit/upstream mismatch;
2. source SHA mismatch;
3. reserved identity collision;
4. unauthorized working-tree mutation;
5. required Kit/schema/scientific change.

During preflight:

1. unresolved projected runner identity;
2. runtime path/resolution/bytes mismatch;
3. script path/bytes mismatch;
4. cwd mismatch;
5. exec/program/observed argv mismatch;
6. `shell != false`;
7. any before/after side effect.

During arming/formal/recovery, Revision 26's ordered artifact, process,
controller and Git integrity rules remain authoritative after task-scoped
projection. No later generic error may overwrite an earlier durable process
error or a committed terminal receipt.

Any unregistered state returns a non-success exit and grants no later
authority.

## 16. Static And Constructibility Checks For R1

The R1 author must perform only non-effectful checks:

- verify Section 1 Git and Controller identities;
- recompute every Section 2 SHA256;
- prove the accepted base is an ancestor of the authoring parent;
- prove all Section 4 reserved local paths are absent;
- prove the controller repository, ref and task tags are absent locally and
  remotely;
- verify no old `0831T001` mutable identity is selected by an exact formal
  command;
- verify the DAG has no authority cycle;
- verify every object in the recovery matrix has one owner;
- verify all commands are token arrays with `shell = false`;
- verify no unresolved token occurs in a formal or recovery command;
- allow the single preflight script-SHA construction token only under the
  Section 9.3 rule;
- run Markdown, whitespace and `git diff --check` checks;
- verify the candidate-only freeze commit contains only this path;
- verify Git-object bytes and working-tree bytes have the same SHA256.

These checks must not invoke runner modes, tests that execute scientific
logic, data readers or formal/recovery commands.

## 17. Independent Review Requirements

The independent reviewer is read-only and reviews the exact candidate tuple:

```text
(candidate ID, candidate path, full freeze commit, candidate SHA256)
```

The reviewer must independently:

1. reconstruct the authority chain;
2. verify all Section 2 bytes;
3. reconstruct the identity-dependency DAG;
4. reconstruct the ownership table;
5. reconstruct the recovery matrix;
6. prove the projection substage is necessary and does not silently authorize
   source changes in this R1 commit;
7. test whether allowed operational substitutions are sufficient without a
   Kit, schema or scientific-logic change;
8. verify every reserved identity is new and absent;
9. verify the argv contract and effect-free boundary;
10. verify the exact formal/recovery commands contain only `0902T002`
    mutable identities;
11. inspect old-identity non-reuse and locked-data prohibitions;
12. report `P0/P1/P2/P3` and one decision.

Acceptance requires:

```text
P0/P1/P2/P3 = 0/0/0/0
decision = accepted
projection_materialization_eligible = true
effect_free_preflight_eligible = false_pending_projection
formal_execution_eligible = false
```

Any finding preserves R1 unchanged. A correction requires R2 at a new path,
new freeze commit, new SHA256 and new Review Request.

## 18. Completion Criteria For This Authoring Turn

This authoring turn is complete only when:

- authority and source identities match;
- this file is the sole file in its freeze commit;
- candidate Git bytes and working-tree bytes have identical SHA256;
- mechanical and constructibility checks pass;
- mutable task coordination records `待验收`;
- a separate revision-specific Review Request is committed;
- no projection, preflight, claim, attempt, baseline, output, controller,
  ref, tag or business execution state exists.

The next actor is an independent R1 plan/readiness reviewer.

## 19. Stop Conditions

Stop as `阻塞` without projection, preflight or execution if:

- any authority, upstream or SHA256 identity mismatches;
- a reserved identity collides;
- an old mutable identity is needed;
- the projection requires a Workflow Kit or schema change;
- the projection requires a scientific-logic change;
- locked data must be opened to make the plan constructible;
- unrelated worktree changes cannot be isolated;
- the independent review is not exactly accepted at `0/0/0/0`.

## 20. R1 Author Declaration

R1 freezes a plan and identity construction model only. It does not claim
that projected implementation bytes, preflight evidence, arming authority or
formal state already exist.

The formal route remains locked until every predecessor in Section 7 exists
with its exact independently accepted identity.
