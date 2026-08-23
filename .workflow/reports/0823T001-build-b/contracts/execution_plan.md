# SKHYNIX H0-B Primary Tuple Supersession Plan

Date: 2026-08-23

Status: independent review draft

## 1. Authority And Position In The Chain

This plan defines the outcome-blind supersession required between the accepted
c6in Hyperliquid execution-latency measurement and Stage H0-B.

The exact chain is:

```text
accepted H0-A
-> accepted 0822T002 latency measurement
-> controller Route B decision
-> reviewed tuple-supersession plan
-> formal tuple-supersession task
-> independent QA
-> controller acceptance
-> H0-B conditional-risk audit
```

Accepted authorities:

- `0821T001 / SKHYNIX-STAGE-H0A-SUPPORT-ONLY`: `已通过`;
- `0822T002 /
  SKHYNIX-C6IN-HYPERLIQUID-EXECUTION-LATENCY-MEASUREMENT-REVISION-2`:
  `已通过`;
- controller decision:
  `latency_pre_h0b_decision=revise_primary_tuple_before_outcomes`;
- accepted replacement latency bucket:
  `6600ms`.

This plan authorizes no H0-B outcome access. H0-B remains locked until the
formal superseding tuple has passed independent QA and a later controller
closure has accepted its exact identity.

## 2. Purpose

The task has one purpose:

> Publish one immutable H0-B input tuple that preserves the accepted H0-A
> research definition while replacing the preregistered `100ms` latency
> assumption with the independently accepted `6600ms` execution-latency
> bucket, and pre-registering `850ms` as a named diagnostic-only
> normal-path bucket.

The supersession is not an H0-A repair. The accepted H0-A package and its
original tuple remain immutable historical evidence.

## 3. Non-Goals

The plan does not authorize:

- reading any Stage 4 outcome, adverse label, feature or view;
- reading Aug07 event rows;
- running H0-B, RQ1, RQ2 or RQ3;
- recomputing the `6600ms` latency recommendation;
- filtering or replacing the accepted `100/100` population used for the
  `6600ms` primary; the only permitted filtered view is the exact
  pre-registered 81-row normal-path diagnostic in section 7.1;
- substituting a normal-path or cancel-response statistic for the accepted
  cancel-effective statistic;
- changing target, venue, channel, distance, delta, selected horizon, side
  aggregation, grid, block size, session roles or identification contract;
- changing H0-A package bytes, identities or accepted registry state;
- network, credential, private endpoint, order, cancel, collection or live
  activity;
- strategy, GLFT, deployment or capital decisions.

## 4. Accepted Input Pins

### 4.1 Accepted H0-A

The formal task must pin:

```text
task_id = 0821T001
status = 已通过
accepted_at = 2026-08-21T17:07:45Z
package =
  local_live_analysis/
  skhynix_continuous_conditional_risk_v2_stage_h0a_support_only
primary_tuple_path =
  local_live_analysis/
  skhynix_continuous_conditional_risk_v2_stage_h0a_support_only/
  primary_tuple_freeze.json
primary_tuple_sha256 =
  e5d1b132248ff1a6933678c32a54e6b4147c1c6f47dab25103011ecbd7a68eca
R =
  7176c78c2b6bfadf11432e9f5a1c1eaf9627c8028a4a22257d67d5a5404dc8fd
C =
  4e8ccc7466f84d9eb2f71f681557424b59ca58249d67426325ecdbd661189636
E =
  8745458fcf4e0ab31f8ad3b2bc2d1d93704f18b13a776c515f14195b51213969
composite =
  2682c32eefac427eed1899a3d492b3fc7545520f72021723e8fef7a0d4d8d9d0
qa_report_sha256 =
  337cb9990adc84376e2083fa4076ba40f9f56c8709d687fd1487502f6992dcac
controller_closure_sha256 =
  9cc1b1d24d29cd2b55a8c1774a9d9e6e59338242c95c3af861460a0b8b07aded
```

The task must fail closed if any path, byte identity or accepted value differs.

### 4.2 Accepted Execution-Latency Measurement

The formal task must pin:

```text
task_id = 0822T002
status = 已通过
accepted_at = 2026-08-23T15:02:27Z
source_commit =
  0c0c5b1c232fce18b3ea5e9efa53a78da3ee503f
package =
  local_live_analysis/
  skhynix_c6in_hyperliquid_execution_latency_0822T002
primary_quantile = nearest_rank_p95
primary_interval =
  risk_decision_ready_to_authoritative_terminal_confirm
p95_cancel_effective_latency_us = 6561052
bucket_rule = max_100ms_then_round_up_50ms
recommended_gate_latency_ms = 6600
recommendation = revise_primary_tuple_before_outcomes
R =
  e8b118bfcf9cbad4c0d95d070084aa9a268f62c13140728c80e373966388eb55
C =
  20a5837162d63763ee42e3fc8ed7bef824316e102eb9325a15f83fe901b37ea9
E =
  103dbe0d2e02392b5e45d61bb106bbdf7d4235b982cea44f895c72aea98ff958
composite =
  7d851ab161ec02c621dffff63ef2f3e962a2aa0c7ed8b4382b285df9534bb0df
package_inventory_sha256 =
  1750474bdd04e1ff5b4beaddf1d93c3e79177060abd2cf7e3c6bacad0876af43
measurement_manifest_sha256 =
  8ac3b362e8d64cbd81232eaf7ed5856bada63ece20408e0d0b3fb5f84c562afd
qa_report_sha256 =
  f8f8f534013ebeb0fcb7d5b6c87efa6e655e23065d0ae516ae3399436471fe86
controller_closure_sha256 =
  96523141ad541f64ce952db84ac9f7ee82502e20fe13f83367bbb6cb9d114cf7
```

The accepted `100/100` population and its overall nearest-rank p95 are the
authority. A filtered normal-path statistic is diagnostic and cannot replace
the accepted bucket.

The `850ms` diagnostic additionally pins the sealed L0/L1 reconstruction
boundary:

```text
sealed_l0_attempt_ledger_sha256 =
  d5c8d3116effe2279ec5c0bacf8d40539c63563f4494471d53e774e10114bf0a
sealed_l0_lifecycle_events_sha256 =
  9fe581f4978ab8138787cf833ced437a02808fd44cacc263739c7920f55c1c6d
sealed_l0_collection_window_schedule_sha256 =
  18a6d0e3e5a12c650953b484a47ffa0cbebee57f4242c5f67d186508b4b90d8f
frozen_l1_contract_runtime_sha256 =
  98bc4823e46feda6c2228e124c53b312e901861c215f17739a9e99aafc38ede0
frozen_l1_entrypoint_sha256 =
  9df437c3693c24fbc22e42fdffa1e6294ea7e343040a17a3cfce3943b5921b63
frozen_l1_terminal_classifier_source_sha256 =
  5b209a6d2fc834b3eeb97da7bd4eb3ed988aef6aaf73a2e4988ace1786c99079
frozen_l1_executor_source_sha256 =
  d940cef5859410ca5caaa1a17018b2f7d90f113348d73d449e9870f315e40d5d
frozen_l1_price_math_source_sha256 =
  6f78e8b8b0c00f18d8186273b0fbebdacb1714ed963bbebb0253116dc7b80487
frozen_l1_runtime_source_inventory_canonical_sha256 =
  e29e0150e010d98dd5f196f596eaa6f288fe15243f95163f0fc7825ef338e8f7
formal_latency_by_attempt_sha256 =
  8ad174041046b3527eb65d969f976bcb5891f68b82568fd6a786a5d56285d4bd
business_l1_a_latency_by_attempt_sha256 =
  8ad174041046b3527eb65d969f976bcb5891f68b82568fd6a786a5d56285d4bd
business_l1_b_latency_by_attempt_sha256 =
  8ad174041046b3527eb65d969f976bcb5891f68b82568fd6a786a5d56285d4bd
```

The exact accepted runtime-source mapping is:

```json
{
  "examples/hyperliquid/cross_exchange_price_math.py": "6f78e8b8b0c00f18d8186273b0fbebdacb1714ed963bbebb0253116dc7b80487",
  "examples/hyperliquid/hyperliquid_maker_order_manager.py": "5b209a6d2fc834b3eeb97da7bd4eb3ed988aef6aaf73a2e4988ace1786c99079",
  "examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py": "d940cef5859410ca5caaa1a17018b2f7d90f113348d73d449e9870f315e40d5d",
  "examples/hyperliquid/skhynix_c6in_latency_contracts_v2.py": "98bc4823e46feda6c2228e124c53b312e901861c215f17739a9e99aafc38ede0",
  "examples/hyperliquid/skhynix_c6in_latency_v2.py": "9df437c3693c24fbc22e42fdffa1e6294ea7e343040a17a3cfce3943b5921b63"
}
```

Its canonical SHA is computed using:

```text
JSON sort_keys=true, separators=(",",":"), ensure_ascii=true, newline=true
```

Changing a path key, including replacing a full repository-relative path with
a basename, must fail with
`H0B_SUPERSESSION_L1_RUNTIME_DEPENDENCY_MISMATCH`.

The formal task and QA must use the frozen L1 summarizer over an isolated
three-file sealed root. Directly trusting only the formal derived CSV is
insufficient.

## 5. Immutable H0-A Fields

The superseding tuple must copy the following semantic fields exactly from
the accepted H0-A tuple:

```text
feature_set_id
selection_status
target
target_venue
target_channel
distance_definition
delta_ticks
horizon_ms
pre_h0b_requirements
side_aggregation
calendar_grid_ms
primary_block_seconds
formal_session_ids
diagnostic_session_ids
descriptive_horizons_ms
distance_sensitivity
single_side_results
support_projection_identity
horizon_selection_trace_sha256
input_inventory_sha256
code_contract_identity
accepted_dependency_identities
kernel_pin
latency_observation_review
boundary
```

The copied values retain their H0-A meaning. In particular:

- `horizon_ms=50`;
- `delta_ticks=0`;
- `side_aggregation=equal_weight_bid_ask_session_scores`;
- Jul30 and Aug04 remain formal;
- Aug03 remains diagnostic only;
- every interval/right-censor and support-replay prerequisite remains
  mandatory before H0-B outcome access.

Any other semantic change is out of scope and must fail closed.

The historical objects are preserved byte-semantically:

```text
canonicalization =
  JSON sort_keys=true, separators=(",",":"), ensure_ascii=true, newline=true
latency_observation_review_canonical_sha256 =
  871b501dc3676d92f0f6b3e11bd4dc2aad9246fcc06b0de301da165094536dd7
historical_h0a_boundary_canonical_sha256 =
  8350d5a2f110f9909c3bf689761dfb2ab890b9b001581cac3bf3c31597a8b680
```

`latency_observation_review` remains the historical H0-A public-cadence
record. It is not rewritten to claim that H0-A measured execution latency.
The accepted measurement is serialized separately in
`latency_measurement_binding`.

Within the superseding tuple, the copied `code_contract_identity` continues to
mean the accepted H0-A selection code identity. The new supersession package's
own C identity is recorded in `supersession_manifest.json`; it must not
overwrite or be confused with the H0-A field.

## 6. Allowed Supersession Delta

The only primary-tuple core change is:

```text
gate_latency_ms: 100 -> 6600
```

Two scenario-role changes are also explicitly declared before outcomes:

```text
100ms: accepted H0-A primary -> historical optimistic sensitivity
850ms: absent -> terminal-observability normal-path diagnostic only
```

The exact structural/provenance allowlist is:

1. `schema_version`:
   `skhynix_stage_h0a_primary_tuple_v1` ->
   `skhynix_stage_h0b_primary_tuple_supersession_v1`;
2. `task_id`: `0821T001` -> `0823T001`;
3. `stage_id`: `stage_h0a` -> `stage_h0b_input_contract`;
4. `frozen_date`: `2026-08-21` -> `2026-08-23`;
5. add `supersession_status=supersedes_latency_only`;
6. add exact `supersedes_h0a`;
7. add exact `latency_measurement_binding`;
8. add
   `controller_latency_decision=revise_primary_tuple_before_outcomes`;
9. `gate_latency_basis` changes from the H0-A preregistered-scenario label to
   the accepted `0822T002` measurement label;
10. add `latency_scenario_order_ms`;
11. add exact `latency_scenarios`;
12. `latency_sensitivity_ms` changes from `[25,50,250,500]` to
    `[25,50,100,250,500]`;
13. add `latency_diagnostic_ms=[850]`;
14. add `supersession_boundary`.

`latency_observation_review` and historical `boundary` are preserved exactly.
No deletion or wildcard structural change is allowed.

The tuple diff must separately report:

- one primary-core change;
- two declared scenario-role changes;
- the 14 exact structural/provenance changes above;
- zero undeclared changes.

## 7. Frozen Latency Scenario Roles

The complete ordered scenario set is:

| Latency | Role | Gate authority |
| ---: | --- | --- |
| `25ms` | legacy sensitivity | secondary only |
| `50ms` | legacy sensitivity | secondary only |
| `100ms` | historical optimistic sensitivity | secondary only |
| `250ms` | legacy sensitivity | secondary only |
| `500ms` | legacy sensitivity | secondary only |
| `850ms` | normal-path execution diagnostic | diagnostic only |
| `6600ms` | measurement-selected primary | only Gate H-C primary |

The tuple must serialize:

```text
gate_latency_ms = 6600
latency_sensitivity_ms = [25, 50, 100, 250, 500]
latency_diagnostic_ms = [850]
latency_scenario_order_ms = [25, 50, 100, 250, 500, 850, 6600]
```

Rules:

- `6600ms` is the only primary latency;
- `100ms` is explicitly optimistic and may not be called measured,
  production-realistic or primary;
- no secondary result may rescue, replace or supersede the `6600ms` primary
  classification;
- `850ms` is diagnostic only and may not become a Gate H-C pass/fail
  authority;
- all secondary scenarios must be reported when their denominators are
  identifiable; selective omission is forbidden;
- H0-B may not add, delete, reorder or relabel a latency scenario after
  outcome access.

### 7.1 The `850ms` Diagnostic

The diagnostic is frozen before H0-B outcome access from the accepted
`0822T002` primary-eligible rows:

```text
normal_path_definition =
  primary_latency_eligible == true && retry_path == normal
normal_path_attempt_count = 81
normal_path_statistic = nearest_rank_p95_cancel_effective_latency
normal_path_p95_us = 833510
normal_path_p95_ms = 833.510
diagnostic_bucket_rule = round_up_to_50ms
diagnostic_latency_ms = 850
diagnostic_role = terminal_observability_normal_path_diagnostic_only
```

The `850ms` value is measurement-informed, but it is not the accepted overall
primary statistic. It is included because the user explicitly froze it as
H0-B's diagnostic latency on 2026-08-23.

H0-B must use it only to describe whether a conclusion differs between the
normal path and the accepted overall execution path. It may not:

- replace the `6600ms` primary;
- rescue a failed `6600ms` Gate H-C result;
- be described as the independently accepted overall latency;
- authorize a strategy, deployment or latency rewrite;
- change after outcome access.

Independent review must verify the `81`-row filter, `833510us` nearest-rank
p95 and upward-50ms `850ms` bucket directly from sealed `0822T002` evidence.

## 8. Superseding Tuple Schema

The canonical JSON file is:

```text
local_live_analysis/
skhynix_h0b_primary_tuple_supersession_0823T001/
superseding_primary_tuple.json
```

Its exact top-level key order is:

```text
schema_version
task_id
stage_id
feature_set_id
frozen_date
selection_status
supersession_status
supersedes_h0a
latency_measurement_binding
controller_latency_decision
target
target_venue
target_channel
distance_definition
delta_ticks
horizon_ms
gate_latency_ms
gate_latency_basis
latency_observation_review
pre_h0b_requirements
side_aggregation
calendar_grid_ms
primary_block_seconds
formal_session_ids
diagnostic_session_ids
descriptive_horizons_ms
latency_scenario_order_ms
latency_scenarios
latency_sensitivity_ms
latency_diagnostic_ms
distance_sensitivity
single_side_results
support_projection_identity
horizon_selection_trace_sha256
input_inventory_sha256
code_contract_identity
accepted_dependency_identities
kernel_pin
boundary
supersession_boundary
```

Required fixed values include:

```text
schema_version =
  skhynix_stage_h0b_primary_tuple_supersession_v1
stage_id = stage_h0b_input_contract
feature_set_id = feature_set_h0
selection_status = selected
supersession_status = supersedes_latency_only
controller_latency_decision =
  revise_primary_tuple_before_outcomes
gate_latency_ms = 6600
gate_latency_basis =
  accepted_0822T002_risk_decision_ready_to_authoritative_terminal_confirm_
  nearest_rank_p95_upward_50ms_bucket
boundary =
  exact accepted H0-A historical boundary object
supersession_boundary.h0b_outcome_aggregate_opened = false
supersession_boundary.outcome_path_open_count = 0
supersession_boundary.network_accessed = false
supersession_boundary.private_endpoint_accessed = false
```

`supersedes_h0a` has this exact key order:

```text
task_id
accepted_at_utc
package_path
primary_tuple_path
primary_tuple_sha256
research_data_identity
code_contract_identity
evidence_identity
composite_identity
qa_report_sha256
controller_closure_sha256
```

`latency_measurement_binding` has this exact key order:

```text
task_id
accepted_at_utc
source_commit
package_path
primary_interval
primary_quantile
p95_cancel_effective_latency_us
bucket_rule
recommended_gate_latency_ms
recommendation
research_data_identity
code_contract_identity
evidence_identity
composite_identity
package_inventory_sha256
measurement_manifest_sha256
qa_report_sha256
controller_closure_sha256
sealed_l0_attempt_ledger_sha256
sealed_l0_lifecycle_events_sha256
sealed_l0_collection_window_schedule_sha256
frozen_l1_contract_runtime_sha256
frozen_l1_entrypoint_sha256
frozen_l1_terminal_classifier_source_sha256
frozen_l1_executor_source_sha256
frozen_l1_price_math_source_sha256
frozen_l1_runtime_source_inventory_canonical_sha256
latency_by_attempt_sha256
```

Every value is the corresponding exact value in section 4.

Each `latency_scenarios` row has this exact key order:

```text
latency_ms
role
authority
primary
diagnostic
can_rescue_primary
```

The exact rows are:

| latency_ms | role | authority | primary | diagnostic | can_rescue_primary |
| ---: | --- | --- | --- | --- | --- |
| `25` | `legacy_sensitivity` | `accepted_h0a_legacy_sensitivity` | `false` | `false` | `false` |
| `50` | `legacy_sensitivity` | `accepted_h0a_legacy_sensitivity` | `false` | `false` | `false` |
| `100` | `historical_optimistic_sensitivity` | `accepted_h0a_primary_reclassified_by_route_b` | `false` | `false` | `false` |
| `250` | `legacy_sensitivity` | `accepted_h0a_legacy_sensitivity` | `false` | `false` | `false` |
| `500` | `legacy_sensitivity` | `accepted_h0a_legacy_sensitivity` | `false` | `false` | `false` |
| `850` | `terminal_observability_normal_path_diagnostic_only` | `user_frozen_2026_08_23_from_accepted_0822T002_sealed_l1` | `false` | `true` | `false` |
| `6600` | `measurement_selected_primary` | `accepted_0822T002_controller_route_b` | `true` | `false` | `false` |

`supersession_boundary` has this exact key order:

```text
h0b_outcome_aggregate_opened
outcome_path_open_count
stage4_outcome_paths_opened
aug07_event_rows_opened
raw_market_rows_opened
network_accessed
private_endpoint_accessed
order_endpoint_accessed
cancel_endpoint_accessed
new_collection
live_action
frozen_l1_rebuild_performed
frozen_l1_network_blocked
```

All access/action booleans are `false`, both L1 proof booleans are `true`, and
`outcome_path_open_count=0`.

Nested objects and arrays must also have exact key order, type and value
contracts in the formal task and canonical Surface Matrix.

## 9. Exact Package

The formal task should publish:

```text
local_live_analysis/
skhynix_h0b_primary_tuple_supersession_0823T001/
```

with this exact tree:

```text
accepted_input_bindings.json
boundary_manifest.json
latency_scenario_roles.csv
normal_path_latency_diagnostic.json
superseding_primary_tuple.json
supersession_manifest.json
tuple_diff.json
contracts/task.md
contracts/surface_matrix.json
contracts/execution_plan.md
reports/tuple_supersession.md
runtime_source/skhynix_h0b_tuple_supersession.py
runtime_tests/test_skhynix_h0b_tuple_supersession.py
```

No symlink, special file, extra path or mutable external reference is allowed.

`normal_path_latency_diagnostic.json` has this exact key order:

```text
schema_version
task_id
source_task_id
source_latency_by_attempt_sha256
row_predicate
primary_eligible_population_count
normal_path_attempt_count
quantile
nearest_rank_formula
nearest_rank_index_one_based
normal_path_p95_us
bucket_rule
diagnostic_latency_ms
diagnostic_role
primary_latency_ms
primary_authority
can_rescue_primary
h0b_outcome_accessed
```

Its fixed values include:

```text
schema_version =
  skhynix_h0b_normal_path_latency_diagnostic_v1
task_id = 0823T001
source_task_id = 0822T002
source_latency_by_attempt_sha256 =
  8ad174041046b3527eb65d969f976bcb5891f68b82568fd6a786a5d56285d4bd
row_predicate =
  primary_latency_eligible=true && retry_path=="normal"
primary_eligible_population_count = 100
normal_path_attempt_count = 81
quantile = nearest_rank_p95
nearest_rank_formula = ceil(0.95*n)
nearest_rank_index_one_based = 77
normal_path_p95_us = 833510
bucket_rule = round_up_to_50ms
diagnostic_latency_ms = 850
diagnostic_role =
  terminal_observability_normal_path_diagnostic_only
primary_latency_ms = 6600
primary_authority = accepted_0822T002_controller_route_b
can_rescue_primary = false
h0b_outcome_accessed = false
```

The package must bind:

- exact accepted H0-A package and tuple identities;
- exact accepted latency package and QA/controller identities;
- exact scenario roles;
- exact tuple diff;
- exact code/contract identity;
- zero-outcome-access boundary;
- package inventory and reverse bindings.

The task is an accepted-Trust-Kernel research package with deterministic
Build A and Build B. Its output is metadata-only; its R layer contains no
outcome value.

## 10. Tuple Diff Contract

`tuple_diff.json` must mechanically compare the accepted H0-A tuple with the
superseding tuple.

It must report:

```text
primary_core_change_count = 1
primary_core_changes = [
  {
    "path": "gate_latency_ms",
    "old_value": 100,
    "new_value": 6600,
    "authority": "accepted_0822T002_controller_route_b"
  }
]
scenario_role_change_count = 2
scenario_role_changes = [
  {
    "latency_ms": 100,
    "old_role": "accepted_h0a_primary",
    "new_role": "historical_optimistic_sensitivity"
  },
  {
    "latency_ms": 850,
    "old_role": "absent",
    "new_role": "terminal_observability_normal_path_diagnostic_only"
  }
]
undeclared_semantic_change_count = 0
declared_structural_change_count = 14
undeclared_structural_change_count = 0
```

The exact `declared_structural_changes` array must contain the 14 path-level
changes in section 6, in order, with exact old/new values or an exact
`absent -> value` representation. It may not use a free-text wildcard such as
`schema changes` or `provenance metadata`.

The comparison must prove the exact immutable fields in section 5, not merely
compare a hand-picked subset that can omit an accidental change.

## 11. Source And Access Boundary

The formal task may read only:

- the accepted H0-A tuple and package manifests/contracts required to verify
  its identity;
- the accepted `0821T001` QA and controller closure;
- the accepted `0822T002` recommendation, manifest, package inventory, QA
  report and controller closure;
- the exact sealed `attempt_ledger.csv`, `lifecycle_events.csv` and
  `collection_window_schedule.csv` pinned in section 4.2;
- the complete accepted five-file frozen L1 runtime source inventory:
  - `examples/hyperliquid/skhynix_c6in_latency_v2.py`:
    `9df437c3693c24fbc22e42fdffa1e6294ea7e343040a17a3cfce3943b5921b63`;
  - `examples/hyperliquid/skhynix_c6in_latency_contracts_v2.py`:
    `98bc4823e46feda6c2228e124c53b312e901861c215f17739a9e99aafc38ede0`;
  - `examples/hyperliquid/hyperliquid_maker_order_manager.py`:
    `5b209a6d2fc834b3eeb97da7bd4eb3ed988aef6aaf73a2e4988ace1786c99079`;
  - `examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py`:
    `d940cef5859410ca5caaa1a17018b2f7d90f113348d73d449e9870f315e40d5d`;
  - `examples/hyperliquid/cross_exchange_price_math.py`:
    `6f78e8b8b0c00f18d8186273b0fbebdacb1714ed963bbebb0253116dc7b80487`;
- the canonical five-file mapping SHA256
  `e29e0150e010d98dd5f196f596eaa6f288fe15243f95163f0fc7825ef338e8f7`;
- the formal and business L1-A/L1-B `latency_by_attempt.csv` files only for
  byte-parity comparison after fresh reconstruction;
- the accepted Trust Kernel registry and v1 acceptance package;
- the formal task, execution plan and canonical Surface Matrix.

For `850ms`, the task must:

1. copy only the three allowed sealed L0 files into a fresh isolated root;
2. run the frozen no-network L1 summarizer into a fresh output root;
3. require the reconstructed `latency_by_attempt.csv` SHA256 to equal
   `8ad174041046b3527eb65d969f976bcb5891f68b82568fd6a786a5d56285d4bd`;
4. select rows using exactly
   `primary_latency_eligible=true && retry_path=="normal"`;
5. require count `81`, nearest-rank p95 rank `ceil(0.95*81)=77`,
   value `833510us` and upward-50ms bucket `850ms`.

It must not open:

- Stage 4 `outcomes/`, `features/` or `views/`;
- Aug07 event rows;
- raw Jul30/Aug03/Aug04 market rows;
- any H0-B output path;
- network or private/order sources.

The guarded opener must reject forbidden paths before content is read.

## 12. Required Surface Matrix

The formal task must define at least these surfaces:

| Surface | Required proof |
| --- | --- |
| `kernel_pin` | accepted Trust Kernel v1 pin unchanged |
| `accepted_h0a_binding` | H0-A package, tuple, QA and closure identities exact |
| `accepted_latency_binding` | 0822T002 package, measurement, QA and closure identities exact |
| `controller_route_b` | exact `revise_primary_tuple_before_outcomes` decision |
| `immutable_h0a_fields` | every section 5 semantic field copied exactly |
| `primary_latency_supersession` | only `100 -> 6600` primary-core delta |
| `latency_scenario_roles` | complete ordered set and unique primary |
| `normal_path_850ms_diagnostic` | exact five-file frozen import closure, 81-row filter, 833510us p95 and 850ms bucket |
| `optimistic_100ms_label` | 100ms cannot be primary or measured |
| `tuple_schema` | exact key/type/value/order contract |
| `tuple_diff` | exact `1 primary-core / 2 scenario-role / 14 structural / 0 undeclared` |
| `outcome_nonaccess` | forbidden path ledger remains zero |
| `deterministic_build` | Build A and Build B exact |
| `package_tree` | exact path/type universe |
| `layered_identity` | R/C/E/composite and reverse bindings exact |
| `atomic_publication` | no partial or overwritten final package |

Each surface requires:

- an exact authoritative source;
- an observed-at rule;
- a rebuild oracle;
- a stable error code;
- at least one executed negative mutation;
- an identity-layer assignment.

## 13. Minimum Stable Error Codes

The implementation must fail closed with stable codes including:

```text
H0B_SUPERSESSION_H0A_IDENTITY_MISMATCH
H0B_SUPERSESSION_LATENCY_IDENTITY_MISMATCH
H0B_SUPERSESSION_CONTROLLER_DECISION_MISMATCH
H0B_SUPERSESSION_IMMUTABLE_FIELD_CHANGED
H0B_SUPERSESSION_PRIMARY_LATENCY_MISMATCH
H0B_SUPERSESSION_SCENARIO_SET_MISMATCH
H0B_SUPERSESSION_PRIMARY_ROLE_NOT_UNIQUE
H0B_SUPERSESSION_850MS_DIAGNOSTIC_MISMATCH
H0B_SUPERSESSION_L1_RUNTIME_DEPENDENCY_MISMATCH
H0B_SUPERSESSION_100MS_LABEL_MISMATCH
H0B_SUPERSESSION_UNDECLARED_TUPLE_CHANGE
H0B_SUPERSESSION_UNDECLARED_STRUCTURAL_CHANGE
H0B_SUPERSESSION_FORBIDDEN_PATH_ACCESS
H0B_SUPERSESSION_OUTCOME_ACCESS_FORBIDDEN
H0B_SUPERSESSION_BUILD_MISMATCH
H0B_SUPERSESSION_PACKAGE_TREE_MISMATCH
H0B_SUPERSESSION_IDENTITY_BINDING_MISMATCH
```

Hostile preflight must execute current and frozen implementations and require
stable-code parity with fail-open count zero before the formal build.

## 14. Formal Task

Proposed task:

```text
task_id = 0823T001
title = SKHYNIX-H0B-PRIMARY-TUPLE-SUPERSESSION
thread = 业务线程-python/research
task_type = research_package
produces_research_package = true
status = 待执行
```

The formal task must be the only active task.

Business completion ends at `待验收` and must report:

- exact package path and tree;
- old and new tuple SHA256;
- primary-core, scenario-role, structural and undeclared diff counts;
- scenario roles and unique primary;
- H0-A and latency input pins;
- Build A/Build B identity;
- R/C/E/composite identities;
- hostile-preflight result;
- package admission result;
- zero forbidden/outcome/network/private access;
- independent QA entrypoint.

## 15. Independent Review Gate

Before dispatch, an independent reviewer must check:

1. the input identities against accepted closure records;
2. whether section 5 freezes every load-bearing H0-A semantic field;
3. whether `100 -> 6600` is the only primary-core change, the two
   scenario-role changes are complete and all 14 structural changes are
   exact;
4. whether the scenario set and labels satisfy the controller closure;
5. whether the `850ms` diagnostic is exactly reconstructed from the 81
   `retry_path=normal` rows and remains non-primary;
6. whether the historical cadence/boundary objects and exact 14-item
   structural allowlist close every schema/provenance escape;
7. whether the tuple schema can be independently reconstructed;
8. whether the source boundary prevents outcome leakage;
9. whether the Surface Matrix and stable-code set cover all load-bearing
   claims;
10. whether the package supplies an unambiguous H0-B pin.

Any P0, P1 or unresolved P2 finding blocks formal dispatch.

## 16. Independent QA Gate

QA uses a fresh work root and must:

1. validate the task and canonical Surface Matrix;
2. verify all accepted input pins byte-for-byte;
3. run every hostile mutation on current and frozen runtime;
   this includes one dependency-SHA mutation for each of the five frozen L1
   source files and one full-path mapping-key mutation;
4. independently rebuild the tuple and package;
5. verify Build A/Build B and business/QA reconstruction identity;
6. independently compare the old and new tuple;
7. prove one primary-core change, two scenario-role changes, 14 declared
   structural changes and zero undeclared semantic or structural changes;
8. verify the complete scenario set, unique `6600ms` primary, exact `850ms`
   diagnostic and optimistic `100ms` label;
9. verify the exact package tree and Trust Kernel admission;
10. verify zero outcome, Aug07, network, private, order and cancel access.

QA may end only at `已通过`, `未通过` or `阻塞`.

QA acceptance does not itself unlock H0-B. The controller must later accept
the exact tuple/package identity and record the H0-B pin.

## 17. Controller Closure And H0-B Unlock

After QA passes, controller closure must record:

- task `0823T001` status `已通过`;
- exact business and QA commits;
- superseding tuple SHA256;
- package R/C/E/composite identities;
- accepted H0-A and latency dependency pins;
- `gate_latency_ms=6600`;
- exact latency scenario roles;
- `100ms=historical_optimistic_sensitivity`;
- `850ms=terminal_observability_normal_path_diagnostic_only`;
- zero outcome access;
- `h0b_tuple_authority=accepted_superseding_tuple`;
- whether H0-B is now eligible for a separate reviewed plan/task.

H0-B may then be planned and dispatched only if its own pre-outcome contract:

- pins the accepted `0823T001` tuple and package identities;
- reconstructs accepted H0-A support commitments;
- freezes every interval-only and full-horizon right-censor branch;
- uses `6600ms` as Gate H-C primary;
- uses `850ms` only as a named diagnostic;
- treats all other latency scenarios as secondary;
- prevents any latency change after outcome access.

## 18. Completion Definition

Tuple supersession is complete only when:

- this plan passes independent review;
- formal task `0823T001` is created and dispatched;
- the metadata-only package is built deterministically;
- hostile preflight and package admission pass;
- business status reaches `待验收`;
- independent QA returns `已通过`;
- controller closure accepts the exact tuple/package identity.

Until then, H0-B remains locked.
