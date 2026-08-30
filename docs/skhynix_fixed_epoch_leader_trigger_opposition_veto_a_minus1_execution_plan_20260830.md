# SKHYNIX Fixed Epoch Leader Trigger With Opposition Veto A-1 Execution Plan

Date: 2026-08-30

Task ID: `0830T002`

Hypothesis ID:
`FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1`

Audit ID:
`FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1_A_MINUS1`

Revision: 2, pre-execution

## 1. Objective and Prediction

Execute one frozen, outcome-blind structural-support audit.

Unique primary:

```text
TRADE_LED
```

Non-rescue sensitivities:

```text
DEPLETION_LED
OFI_LED
```

Primary prediction:

```text
confirmed clusters >= 30
represented dates >= 4
maximum single-date share <= 0.50
```

## 2. Git and Authority

Research-kit start:

```text
tag: skhynix-fixed-epoch-research-kit-v1
commit: 45afe2446e27449bb8f3c8ffde7e95663b58e4fb
```

Suppression authority:

```text
tag: skhynix-fixed-epoch-suppression-v1
commit: f06eb5cb012cb62b2a778ad90d433c4083f9ba14
runner:
  examples/hyperliquid/skhynix_fixed_causal_epoch_mstate_a_minus1.py
```

The successor directly calls these authority functions:

```text
source_preflight
base_eligibility
channel_actions
channel_memories
epoch_support_ledger
materialize_poisoned_cache_set
verify_poison_attestation
```

The baseline manifest supplies exact authority file SHA256, Git blob OID and
callable AST SHA256. Formal tests must prove direct invocation; a local
reimplementation is forbidden.

`source_preflight` must complete for all rows before action, memory, trigger
or output construction. Its failure is uniquely A-1-0.

## 3. Source Authority

Canonical immutable cache root:

```text
/Users/liu/Documents/
hftbacktest-0829t003-fixed-causal-epoch-mstate-a-minus1/
local_live_analysis/
skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003/cache
```

The exact 29 names, row counts, schema versions, sizes and SHA256 values are
the baseline `support/source_cache_inventory.csv`.

Build A and Build B read the same immutable canonical bytes. "Fresh" means a
new empty output root and a new process invocation inside the one-shot
orchestrator; it does not mean regenerated caches.

Permitted consumed fields are exactly the accepted baseline consumed set.
Every allowed-minus-consumed field is poison authority and may not affect an
output.

## 4. Frozen Detector

The idea document's Revision 2 definitions are normative:

- checkpoint-exact causal order;
- raw onset;
- epoch/core omission;
- anchor-time veto;
- support counts;
- fixed-epoch thinning;
- explicit-evidence confirmation;
- cancellation booleans and primary reason.

Shared values:

```text
TTL = 100ms, age <= 100ms
prestate = [t-120ms,t-20ms], six checkpoints
confirmation = [t+20ms,t+200ms], ten checkpoints
margin = 0.00
fast threshold = +/-0.50 inclusive
medium threshold = +/-0.25 inclusive
```

Variants:

```text
TRADE_LED: leader index 0
DEPLETION_LED: leader index 1
OFI_LED: leader index 2
```

## 5. Fixed Epoch Contract

Direct-call authority `epoch_support_ledger`.

Frozen:

```text
epoch origin = 0
epoch width = 60s
checkpoint = 20ms
expected checkpoints = 3,000
core = [15s,45s)
thinning key = (capture_id,epoch_id,variant,direction)
tie-break = (candidate_ts_ns,candidate_event_seq)
cluster = capture_id:epoch_id
```

Confirmation must close inside the core. Core-close equality is allowed:

```text
candidate_ts_ns + 200ms <= core_close_ns
```

## 6. Conservation Contract

For each `(date,capture,epoch,variant,direction)`:

```text
raw_onset_count
  = epoch_core_omitted_count
  + anchor_vetoed_count
  + veto_admitted_count

veto_admitted_count
  = retained_count
  + same_key_suppressed_count

retained_count
  = confirmed_count
  + cancelled_count
```

Every count is a non-negative base-10 integer. Violation is A-1-2.

## 7. Slice/Reset Invariance

Nominal artificial starts:

```text
stride = 600s per segment
guard = 122s
```

Every slice is rebuilt from sliced raw cache via the authority feature
builder. No derived feature reuse is permitted.

Comparable epochs:

```text
epoch_id >= ceil((actual_start_ts_ns + 122s)/60s)
eligible in both full and slice
same artificial-start segment
```

For every qualifying start, compare exact typed, numerically sorted
identities.

Epoch disposition identity:

```text
(capture_id,epoch_id,disposition,segment_id_or_empty,
 segment_ids_json,segment_set_sha256,observed_checkpoint_count,
 duplicate_timestamp_count,off_grid_timestamp_count,
 missing_expected_timestamp_count,grid_exact)
```

Epoch/variant counter identity:

```text
(capture_id,epoch_id,variant,direction,
 raw_onset_count,epoch_core_omitted_count,anchor_vetoed_count,
 veto_admitted_count,retained_count,same_key_suppressed_count,
 confirmed_count,cancelled_count,retained_candidate_id_or_empty)
```

Retained trigger identity:

```text
(capture_id,variant,epoch_id,direction,candidate_ts_ns,
 candidate_event_seq,cluster_id)
```

Retained status identity adds:

```text
secondary_same_direction_count
secondary_opposite_count
additional_same_leader_update_count
opposite_update_count
first_additional_same_update_ts_ns_or_empty
first_additional_same_update_event_seq_or_empty
confirmation_window_close_ts_ns
confirmation_window_close_event_seq
confirmation_status
cancel_reason
four cancellation booleans
```

Support identity:

```text
(capture_id,epoch_id,checkpoint_ts_ns,channel_index,
 action_int,memory_int,memory_age_ms)
```

Canonical identity hashing:

```text
JSON sort_keys=true
separators=(",",":")
ensure_ascii=true
Python typed int/bool/str
SHA256
```

Each identity stores expected/actual count and SHA256. Required:

```text
all exact flags true
cross-segment checkpoint count = 0
represented slice dates >= 4
distinct comparable epochs >= 30
positive support checkpoint count
```

## 8. One-Shot Formal Attempt

The only formal command is:

```bash
python \
  examples/hyperliquid/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1.py \
  --formal-attempt \
  --repo-root /Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate \
  --source-cache-root \
    /Users/liu/Documents/hftbacktest-0829t003-fixed-causal-epoch-mstate-a-minus1/local_live_analysis/skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003/cache \
  --attempt-root \
    /Users/liu/Documents/hftbacktest-0830t002-fixed-epoch-relaxed-mstate/local_live_analysis/skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_0830T002_formal_v1
```

Exact attempt children:

```text
attempt-lock.json
canonical_a/
canonical_b/
poison_cache/
poison_p/
poison-attestation.json
attempt-result.json
```

Before any source cache opens, the orchestrator must:

1. require clean Git working tree;
2. require HEAD equals the task-frozen implementation commit;
3. require idea/plan/task/runner/tests SHA256 and Git blob identities;
4. require baseline verifier success;
5. atomically create the absent attempt root with no replacement;
6. atomically write and fsync `attempt-lock.json`;
7. record exact argv, cwd, PID, start time, roots and identities.

An existing attempt root or any non-empty output child fails closed. A failed
or interrupted attempt may not be replaced by another attempt in this task.

## 9. Build and Poison Sequence

The one-shot attempt performs:

1. Build A over canonical caches into empty `canonical_a`;
2. Build B in a fresh process over the identical canonical bytes into empty
   `canonical_b`;
3. materialize poison caches from canonical authority:
   - all allowed-minus-consumed fields;
   - same field names, dtypes and shapes;
   - every non-empty unconsumed field value changes;
   - every consumed field remains byte-identical;
4. write required sibling `poison-attestation.json`;
5. Build P over poison caches into empty `poison_p`, while source inventory
   authority remains canonical;
6. compare A/B/P preseal outputs;
7. write identical pending evidence to all three roots and compare;
8. write identical final evidence, outcome ledger, gates, classification and
   manifests to all roots and compare;
9. atomically write and fsync `attempt-result.json`.

Frozen poison expectations from the 29-cache authority:

```text
cache_count = 29
unconsumed_field_count = 15
nonempty field instances = 435
changed field instances = 435
consumed mismatch = 0
```

The sibling attestation SHA is referenced identically by A/B/P outcome
ledgers.

## 10. Determinism Stages

Each stage has exact equal non-cache path sets:

```text
preseal:
  stage=preseal
  preseal_difference_count=null
  pending_difference_count=null
  final_difference_count=null

pending:
  stage=pending
  preseal_difference_count=0
  pending_difference_count=null
  final_difference_count=null

final:
  stage=final
  preseal_difference_count=0
  pending_difference_count=0
  final_difference_count=0
```

The final 17-path output namespace is identical across A/B/P. Dynamic files
are rewritten in this order:

```text
execution_evidence.json
outcome_access_ledger.json
gate_contract.json
A_minus1_summary.json
classification.json
run_manifest.json last
```

## 11. Required Sibling Artifacts

Outside A/B/P but inside the attempt:

```text
attempt-lock.json
poison-attestation.json
attempt-result.json
```

They are required and SHA-bound by `attempt-result.json`.

## 12. Exact 17 Output Paths

Each A/B/P root contains:

```text
classification.json
contracts/authority_binding.json
contracts/detector_contract.json
contracts/execution_evidence.json
contracts/fixed_epoch_contract.json
contracts/gate_contract.json
contracts/outcome_access_ledger.json
reports/A_minus1_summary.json
run_manifest.json
support/channel_action_by_date.csv
support/epoch_support.csv
support/epoch_variant_counters.csv
support/slice_invariance.csv
support/source_cache_inventory.csv
support/support_by_date.csv
support/trigger_ledger.csv
support/variant_summary.csv
```

`run_manifest.json` excludes itself and lists exactly 16 unique entries with
path, size and SHA256. Missing, extra, duplicate or cache payload paths fail.

## 13. Common Serialization

JSON:

```text
UTF-8/ASCII-compatible
indent=2
sort_keys=true
ensure_ascii=true
one trailing newline
```

Canonical hash JSON:

```text
sort_keys=true
separators=(",",":")
ensure_ascii=true
```

CSV:

```text
UTF-8/ASCII-compatible
Unix newline
exact ordered header
True/False booleans
base-10 integers
finite decimal floats
empty string for N/A identity
```

No NaN or infinity may be emitted.

## 14. CSV Schemas

### source_cache_inventory.csv

Row grain: one canonical cache, sorted by `cache_name`.

```text
cache_name,size_bytes,row_count,cache_schema_version,cache_sha256,
paired_determinism_verified,cache_field_schema_verified
```

### channel_action_by_date.csv

Row grain: `(research_date,channel)`, sorted ASCII.

```text
research_date,channel,total_action_count,global_invalid_action_count,
new_invalid_action_count,new_pos_action_count,new_neg_action_count,
new_neutral_action_count,no_update_action_count,observed_new_evidence_count,
expiry_count,neutral_overwrite_count,unauthorized_ttl_refresh_count,
cross_segment_memory_carry_count,maximum_memory_age_ms,
action_partition_exact
```

### epoch_support.csv

Use the authority epoch-ledger exact ordered fields and typed ordering:

```text
research_date,capture_id,epoch_id,epoch_start_ns,epoch_end_ns,
core_open_ns,core_close_ns,segment_id,segment_count,segment_ids_json,
segment_set_sha256,disposition,observed_checkpoint_count,
unique_timestamp_count,duplicate_timestamp_count,
off_grid_timestamp_count,missing_expected_timestamp_count,grid_exact
```

`segment_id` is empty unless disposition is `eligible`.

### epoch_variant_counters.csv

Row grain: every enumerated
`(research_date,capture_id,epoch_id,variant,direction)`, including zeros.

```text
research_date,capture_id,epoch_id,variant,direction,
raw_onset_count,epoch_core_omitted_count,anchor_vetoed_count,
veto_admitted_count,retained_count,same_key_suppressed_count,
confirmed_count,cancelled_count,retained_candidate_id
```

Sort by date ASCII, capture ASCII, epoch numeric, variant registered order,
direction numeric.

### trigger_ledger.csv

Row grain: one retained trigger.

```text
research_date,capture_id,variant,epoch_id,epoch_start_ns,core_open_ns,
core_close_ns,segment_id,direction,candidate_id,candidate_ts_ns,
candidate_event_seq,dependence_cluster_id,leader_channel,
secondary_same_direction_count,secondary_opposite_count,leader_age_ms,
secondary_age_json,additional_same_leader_update_count,
opposite_update_count,first_additional_same_update_ts_ns,
first_additional_same_update_event_seq,confirmation_window_close_ts_ns,
confirmation_window_close_event_seq,confirmation_status,cancel_reason,
insufficient_confirmation_history,confirmation_segment_boundary,
explicit_opposite_update,no_additional_same_leader_update
```

`confirmation_status` is exactly `CONFIRMED` or `CANCELLED`.

Candidate ID is canonical SHA256 of:

```text
(capture_id,variant,epoch_id,segment_id,direction,
 candidate_ts_ns,candidate_event_seq)
```

Optional first-additional fields are empty together.

### support_by_date.csv

Row grain: `(research_date,variant,direction)`.

```text
research_date,variant,direction,raw_onset_count,epoch_core_omitted_count,
anchor_vetoed_count,veto_admitted_count,retained_count,
same_key_suppressed_count,confirmed_count,cancelled_count,
distinct_confirmed_cluster_count,support0_confirmed_count,
support1_confirmed_count,support2_confirmed_count
```

### variant_summary.csv

Row grain: one registered variant.

```text
variant,is_primary,raw_onset_count,veto_admitted_count,retained_count,
confirmed_count,cancelled_count,distinct_confirmed_cluster_count,
represented_date_count,maximum_single_date_cluster_share,
support_prediction_passed
```

If cluster count is zero, maximum share is empty and prediction is False.

### slice_invariance.csv

Row grain: one qualifying artificial start.

```text
research_date,capture_id,segment_id,nominal_start_ts_ns,
actual_start_ts_ns,comparison_floor_ns,first_comparable_epoch_id,
slice_source_sha256,comparable_epoch_count,
expected_epoch_disposition_count,actual_epoch_disposition_count,
expected_epoch_disposition_sha256,actual_epoch_disposition_sha256,
epoch_disposition_exact,expected_counter_count,actual_counter_count,
expected_counter_sha256,actual_counter_sha256,counter_exact,
expected_retained_count,actual_retained_count,expected_retained_sha256,
actual_retained_sha256,retained_exact,expected_status_count,
actual_status_count,expected_status_sha256,actual_status_sha256,
status_exact,expected_support_count,actual_support_count,
expected_support_sha256,actual_support_sha256,support_exact,
cross_segment_checkpoint_count,mismatch_reason
```

`mismatch_reason` is `none` or first failed exact flag in this order:

```text
epoch_disposition
counter
retained
status
support
cross_segment
multiple
```

## 15. JSON Contracts

### authority_binding.json

Contains:

- baseline IDs/tags/commits;
- successor HEAD/runner/tests/task/idea/plan paths, SHA256 and Git blobs;
- direct-call authority function AST hashes;
- source inventory SHA;
- attempt-lock SHA.

### detector_contract.json

Contains exact variants, channel indexes, thresholds, TTL boundary, prestate,
checkpoint order, veto, thinning, confirmation and cancellation semantics.

### fixed_epoch_contract.json

Byte-equivalent semantic values to the authority fixed-epoch contract plus
the successor variant dimension in the thinning key.

### execution_evidence.json

Contains exact stage, A/B/P path sets, per-file comparison rows, difference
counts, successor identity and sibling artifact SHA values.

### outcome_access_ledger.json

Contains:

```text
future_target_accessed=false
future_price_accessed=false
fill_fee_pnl_accessed=false
consumed_cache_fields=sorted exact set
poisoned_unconsumed_fields_change_output=false only at final
poison_attestation_sha256
cache_count=29
unconsumed_field_count=15
changed_field_instance_count=435
consumed_field_mismatch_count=0
```

### gate_contract.json

Contains all four gates and every condition row, even after earlier failure.
Condition fields:

```text
condition,status,passed,actual,required
```

Later gates:

```text
status=NOT_EVALUATED
passed=null
actual=null
required remains frozen
```

### A_minus1_summary.json

Contains task/hypothesis/audit IDs, exact plan/idea/successor identities,
variant summaries, integrity counters, sibling evidence, gates,
classification and all authorization flags.

### classification.json

Contains only registered classification, gates and authorization flags.

## 16. Numeric Semantics

All counts are non-negative integers, not booleans.

All shares are finite in `[0,1]`.

```text
cluster_count = 0:
  maximum_single_date_share = null in JSON / empty in CSV
  support prediction = false

cluster_count > 0:
  share = maximum per-date distinct clusters / total distinct clusters
```

Any negative, non-finite, wrong type, inconsistent share, failed conservation
or malformed SHA is A-1-2.

## 17. Sequential Gates and Classification

### A-1-0 Authority and Source

Conditions:

- baseline verifier;
- frozen idea/plan/task/successor identities;
- direct-call authority bindings;
- clean one-shot attempt receipt;
- exact 29-cache source closure;
- source preflight violations zero;
- A/B/P final path/SHA difference count zero.

Failure:

```text
Aminus1_authority_or_source_failed
```

### A-1-1 Outcome Boundary

Conditions:

- forbidden access count zero;
- poison 29/15/435 identities exact;
- consumed mismatch zero;
- final A/P difference zero.

Failure:

```text
Aminus1_outcome_boundary_violated
```

### A-1-2 Detector Integrity

Conditions:

- all channel/action/memory invariants;
- all three variants' conservation;
- all fixed-epoch and identity invariants;
- all slice exact flags;
- cross-segment checkpoint count zero;
- represented slice dates >=4;
- distinct comparable epochs >=30;
- positive compared support count;
- numeric/schema/manifest violations zero.

Any sensitivity integrity defect fails this gate.

Failure:

```text
Aminus1_detector_integrity_failed
```

### A-1-3 Primary Structural Support

Conditions in order:

```text
TRADE_LED cluster count >=30
TRADE_LED represented dates >=4
TRADE_LED maximum single-date share <=0.50
```

If count or date coverage fails:

```text
Aminus1_trade_led_structural_support_not_estimable
```

If count/date pass but concentration fails:

```text
Aminus1_trade_led_structure_date_concentrated
```

If all pass:

```text
Aminus1_trade_led_recurrent_structural_candidate
```

Sensitivities are absent from A-1-3.

Every gate after the first failed gate is retained as exact
`NOT_EVALUATED`.

## 18. Frozen Hostile-Test Minimum

At minimum:

- exact +/-0.50 and +/-0.25 inclusivity;
- TTL age 100ms fresh and 120ms stale;
- six-point prestate excludes `t`;
- core open included, core close excluded for trigger;
- confirmation close equality allowed and +20ms rejected;
- same-checkpoint neutral clears old opposite before veto;
- same-checkpoint new opposite is visible to veto;
- reset clears memory and invalidates epoch before trigger;
- raw/veto/admitted/retained/suppressed/confirmed conservation;
- earliest retained failure suppresses a later confirmable trigger;
- opposite directions and variants share epoch cluster;
- no later same-key replacement;
- first additional update and window-close timestamps differ correctly;
- confirmation waits until window close;
- independent cancellation booleans and reason precedence;
- slice epoch/counter/retained/status/support hash mutations fail;
- slice cannot reuse derived full features;
- source invalid fails before action;
- every unconsumed poison value changes and consumed values do not;
- A/B/P missing, extra and byte mutations fail;
- attestation mutation fails;
- all 17 schemas, typed sentinels, sorting and manifest self-exclusion;
- zero, negative, NaN, infinity and wrong-type gate mutations;
- later `NOT_EVALUATED` rows preserve required values;
- sensitivity cannot rescue primary;
- dirty worktree, wrong HEAD, wrong CLI/root, existing attempt root and
  successor identity mutations fail before cache read;
- interrupted/failed attempt cannot be replaced.

## 19. Pre-Execution Locks

Before formal execution:

```text
independent idea/plan review = 0/0/0/0
idea and plan SHA frozen in task
implementation commit and runner/tests SHA/blob frozen in task
independent readiness review confirms command/receipt/test contract
focused and inherited tests pass
```

Until then, 29-cache execution is locked.

## 20. Post-Build-A Rule

Atomic `attempt-lock.json` creation is the start of formal execution and
occurs before any cache read.

After it exists, only the already registered one-shot sequence may continue.
No repair, diagnosis, replacement attempt, code/plan/test change or
alternative execution is permitted.

If the result contradicts the prediction, record it and stop.
