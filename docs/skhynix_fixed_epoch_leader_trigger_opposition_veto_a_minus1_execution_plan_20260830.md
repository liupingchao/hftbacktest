# SKHYNIX Fixed Epoch Leader Trigger With Opposition Veto A-1 Execution Plan

Date: 2026-08-30

Task ID: `0830T002`

Hypothesis ID:
`FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1`

Audit ID:
`FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1_A_MINUS1`

## 1. Objective

Execute one frozen outcome-blind structural-support audit of the registered
leader-trigger idea.

Unique primary:

```text
TRADE_LED
```

Non-rescue sensitivities:

```text
DEPLETION_LED
OFI_LED
```

This task does not select a winning variant and does not inspect future
market responses.

## 2. Authority Binding

Research-kit start:

```text
skhynix-fixed-epoch-research-kit-v1
45afe2446e27449bb8f3c8ffde7e95663b58e4fb
```

Suppression authority:

```text
skhynix-fixed-epoch-suppression-v1
f06eb5cb012cb62b2a778ad90d433c4083f9ba14
```

Before execution, the baseline verifier must pass with:

```text
--check-working-tree --require-tags
```

The successor runner must bind the authority runner by exact file SHA256,
Git blob OID and callable AST identities. It may call the frozen:

- source preflight;
- base eligibility;
- channel action and memory construction;
- epoch disposition and eligibility;
- poison materialization and attestation helpers.

The six frozen authority files may not change.

## 3. Source Data

Use exactly the 29 source caches recorded by the baseline
`source_cache_inventory.csv`.

Canonical source root:

```text
/Users/liu/Documents/
hftbacktest-0829t003-fixed-causal-epoch-mstate-a-minus1/
local_live_analysis/
skhynix_fixed_causal_epoch_mstate_a_minus1_0829T003/cache
```

Each cache name, row count, schema version and SHA256 must equal the baseline
inventory before any feature construction.

## 4. Information Boundary

Permitted row-aligned values are exactly the fields already consumed by the
accepted baseline:

```text
ts_ns
event_seq
segment_id
ready
valid_book
activity
trade_total
trade_signed
bid_depletion
ask_depletion
ofi
ofi_abs
```

Derived `ratios_100` and `ratios_500` must come from the frozen baseline
feature builder.

Forbidden:

```text
midpoint
future returns
future direction labels
fills
fees
PnL
strategy outcomes
```

## 5. Frozen Detector

Channels:

```text
trade index     = 0
depletion index = 1
ofi index       = 2
```

Shared parameters:

```text
TTL = 100ms
prestate = 120ms = 6 checkpoints
confirmation = 200ms = 10 checkpoints
margin = 0.00
fast threshold = 0.50
medium threshold = 0.25
```

Trigger, anchor-time veto, support count and confirmation semantics are exact
as registered in the idea document.

## 6. Fixed Epoch Thinning

Use the authority's exact epoch ledger:

```text
60s absolute Unix epochs
complete 3,000-checkpoint single-segment eligibility
core = [15s,45s)
```

For every `(capture_id, epoch_id, direction, variant)`:

1. enumerate trigger candidates after anchor-time veto;
2. retain the earliest by `(candidate_ts_ns, candidate_event_seq)`;
3. suppress every later trigger in that key;
4. assign cluster `capture_id:epoch_id`;
5. evaluate only the retained trigger's confirmation.

Opposite directions share the same dependence cluster.

## 7. Formal Ledger

One trigger-ledger row per retained trigger:

```text
research_date
capture_id
variant
epoch_id
epoch_start_ns
core_open_ns
core_close_ns
segment_id
direction
candidate_id
candidate_ts_ns
candidate_event_seq
dependence_cluster_id
leader_channel
secondary_same_direction_count
secondary_opposite_count
leader_age_ms
secondary_age_json
additional_same_leader_update_count
opposite_update_count
confirmation_event_seq
confirmation_ts_ns
confirmation_status
cancel_reason
```

Allowed `cancel_reason`:

```text
none
insufficient_confirmation_history
segment_boundary
no_additional_same_leader_update
explicit_opposite_update
multiple
```

Anchor-time vetoed triggers and same-key suppressed triggers are represented
only in exact counters by date/variant/direction; they do not receive retained
trigger rows.

## 8. Slice/Reset Invariance

Use the baseline artificial-start schedule:

```text
nominal stride = 600s
guard = 122s
```

Every slice must rebuild features directly from sliced raw cache.

Comparable epochs must:

- be complete and eligible in full and sliced analyses;
- belong to the artificial-start segment;
- start no earlier than `ceil((actual_start + 122s)/60s)`.

Compare:

```text
confirmed candidate identity:
  (capture_id, variant, epoch_id, candidate_ts_ns,
   candidate_event_seq, direction, cluster_id)

support identity:
  (capture_id, epoch_id, checkpoint_ts_ns,
   channel_index, action_int, memory_int)
```

Required:

```text
identity mismatches = 0
support mismatches = 0
cross-segment compared checkpoints = 0
represented slice dates >= 4
distinct comparable epochs >= 30
```

## 9. Formal Builds

Execute exactly:

```text
Build A: canonical source caches
Build B: fresh canonical source caches
Build P: poisoned unconsumed fields with canonical consumed authority
```

Each build runs the complete 29-cache pipeline once.

After all three preseal outputs exist:

1. compare exact output path sets and bytes;
2. verify poison attestation;
3. write identical final determinism/outcome evidence to all roots;
4. rebuild manifests;
5. require final A/B/P byte equality.

No formal build may be repaired or replaced. Execution failure ends the task
as `阻塞` or execution-integrity failure.

## 10. Required Outputs

Exact 14 non-cache outputs:

```text
classification.json
contracts/authority_binding.json
contracts/detector_contract.json
contracts/fixed_epoch_contract.json
contracts/outcome_access_ledger.json
reports/A_minus1_summary.json
run_manifest.json
support/channel_action_by_date.csv
support/epoch_support.csv
support/slice_invariance.csv
support/source_cache_inventory.csv
support/support_by_date.csv
support/trigger_ledger.csv
support/variant_summary.csv
```

`run_manifest.json` excludes itself and lists exactly 13 entries.

## 11. Gates

Sequential gates:

### A-1-0 Authority and Source

Required:

- baseline verifier passes;
- idea and plan SHA256 match the frozen task;
- authority file/blob/AST binding passes;
- exact 29-cache source closure passes;
- Build A/B/P final difference count is zero.

### A-1-1 Outcome Boundary

Required:

- forbidden field access count is zero;
- poisoned unconsumed fields change no output;
- consumed field mismatch count is zero.

### A-1-2 Detector Integrity

Required:

- action partition violations = 0;
- source invalid contribution count = 0;
- TTL unauthorized refresh = 0;
- cross-segment memory carry = 0;
- candidate identity violations = 0;
- fixed epoch contract violations = 0;
- slice candidate mismatches = 0;
- slice support mismatches = 0;
- cross-segment compared checkpoints = 0;
- represented slice dates >= 4;
- distinct comparable epochs >= 30.

### A-1-3 Primary Structural Support

For `TRADE_LED` confirmed clusters:

```text
distinct cluster count >= 30
represented date count >= 4
maximum single-date cluster share <= 0.50
```

Sensitivity results cannot rescue this gate.

## 12. Classification

First failed gate determines the only classification:

```text
A-1-0:
  Aminus1_authority_or_source_failed

A-1-1:
  Aminus1_outcome_boundary_violated

A-1-2:
  Aminus1_detector_integrity_failed

A-1-3:
  Aminus1_trade_led_structural_support_not_estimable

all pass:
  Aminus1_trade_led_recurrent_structural_candidate
```

Passing all gates authorizes only drafting a separately reviewed structural
null audit. It does not authorize A0 or future outcomes.

## 13. Pre-Execution Review Lock

Before any 29-cache execution:

```text
independent plan review P0/P1/P2/P3 = 0/0/0/0
idea SHA256 frozen in task
plan SHA256 frozen in task
implementation commit frozen
focused hostile tests pass
baseline/current/predecessor regressions pass
```

Until then:

```text
data execution = locked
future outcomes = locked
A0 = locked
live/private/order = locked
```

## 14. Post-Execution Immutability

The first formal Build A command activates the post-execution lock.

After that command:

- no idea or plan edits;
- no detector or test edits;
- no threshold, gate or output-schema edits;
- no repair command;
- no diagnostic rerun;
- no alternative parameter execution;
- no additional research question;
- no result-driven code correction.

Only these actions remain permitted:

1. complete the already registered Build B/P/finalize sequence if Build A
   completed successfully;
2. write the exact observed result and execution report;
3. run read-only independent QA.

If the observed result differs from the implicit prediction, record the
difference and stop.
