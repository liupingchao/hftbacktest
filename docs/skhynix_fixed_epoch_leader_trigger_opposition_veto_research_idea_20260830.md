# SKHYNIX Fixed Epoch Leader Trigger With Opposition Veto Research Idea

Date: 2026-08-30

Hypothesis ID:
`FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1`

Revision: 4, pre-execution

## 1. Starting Evidence

The accepted fixed-epoch baseline produced `2,219` common candidates. Its
selected `F010` retained only `2` clusters, while the least strict `F000`
retained `34`.

The dominant cancellation reasons were:

```text
F010:
  persistence_abstain = 1,581
  consensus_lost      =   614
  opposite_consensus  =    22

F000:
  persistence_abstain = 1,040
  consensus_lost      = 1,135
  opposite_consensus  =     9
```

The previous detector therefore rejected candidates mainly because all three
channels did not remain simultaneously fresh and directional, not because
the market produced explicit opposite evidence.

## 2. Research Idea

Replace:

```text
trade_d AND depletion_d AND ofi_d
```

with:

```text
one registered leader onset_d
-> fixed-epoch/core admission
-> confirmation-edge admission
-> explicit-opposition veto
-> fixed-epoch thinning
-> explicit-evidence confirmation
```

Secondary channels may support or explicitly oppose the leader. Neutral,
stale or absent secondary evidence does not automatically delete an event.

This is not a `2-of-3` vote. Depletion and OFI are both depth-derived and are
not independent votes.

## 3. Registered Variants

Unique primary:

```text
TRADE_LED
```

Non-rescue sensitivities:

```text
DEPLETION_LED
OFI_LED
```

Each variant is executed and reported. Only `TRADE_LED` enters the support
gate. A sensitivity cannot rescue the primary.

## 4. Frozen Thresholds

All variants use:

```text
fast 100ms ratio:
  positive >= +0.50
  negative <= -0.50

medium 500ms ratio:
  positive >= +0.25
  negative <= -0.25

margin = 0.00
TTL = 100ms, fresh iff age <= 100ms
prestate = 120ms = six prior 20ms checkpoints
confirmation window = 200ms = ten later checkpoints
```

This version does not lower directional amplitude. It isolates conjunction
and persistence semantics.

## 5. Authority Primitives

The successor must directly call the accepted authority for:

```text
source_preflight
base_eligibility
channel_actions
channel_memories
epoch_support_ledger
materialize_poisoned_cache_set
verify_poison_attestation
```

Full-cache and sliced-cache features must also directly call the single bound
`build_features` authority described in the execution plan. The same callable
must build A, B, P and every artificial slice.

New-evidence masks are therefore exactly:

```text
trade:     trade_total > 0
depletion: bid_depletion + ask_depletion > 0
ofi:       ofi_abs > 0
```

No successor reimplementation may supply source, action, TTL memory or epoch
semantics.

## 6. Checkpoint-Exact Causal Order

For the complete cache, raw source preflight runs before any action, memory,
onset or trigger construction. Any source defect stops at A-1-0.

At checkpoint `t`, the order is:

1. authority base eligibility and channel actions process raw evidence at
   `t`;
2. authority channel memories process segment clear, global/new invalid,
   neutral overwrite, directional update and expiry;
3. memory age exactly `100ms` remains fresh;
4. leader prestate reads only `t-120ms ... t-20ms`, never `t`;
5. raw leader onset is evaluated from the leader's `NEW_d` action at `t`;
6. fixed-epoch eligibility and core membership are evaluated;
7. confirmation-edge admission requires `t+200ms <= core_close_ns`;
8. anchor-time support and opposition read post-action memory at `t`;
9. veto-admitted triggers enter fixed-epoch thinning;
10. only the retained trigger enters confirmation.

Consequences:

- `NEW_NEUTRAL` at `t` clears an older opposite memory before veto;
- `NEW_-d` at `t` is visible to veto;
- segment reset at `t` clears memory and makes the complete epoch
  structurally ineligible;
- the six-point prestate excludes all same-checkpoint updates.

## 7. State Vocabulary

The mutually auditable layers are:

```text
raw leader onset
-> epoch/core omitted OR confirmation-edge omitted
   OR anchor-time vetoed OR veto-admitted trigger
-> retained trigger OR same-key suppressed trigger
-> confirmed retained trigger OR cancelled retained trigger
```

Per `(capture, epoch, variant, direction)`:

```text
raw_onset
  = epoch_core_omitted + confirmation_edge_omitted
  + anchor_vetoed + veto_admitted

veto_admitted
  = retained + same_key_suppressed

retained
  = confirmed + cancelled
```

No object at a later layer may exist without its preceding layer.

## 8. Raw Leader Onset

For variant leader channel `c` and direction `d`, checkpoint `t` is a raw
leader onset iff:

1. leader action at `t` is `NEW_POS` for `d=+1` or `NEW_NEG` for `d=-1`;
2. leader memory at each of the six prior checkpoints is exactly
   `BACKGROUND=0`;
3. those six checkpoints and `t` belong to one segment.

Unknown, stale, directional or unavailable prestate does not qualify.

## 9. Epoch/Core and Anchor-Time Veto

The authority fixed-epoch contract remains:

```text
checkpoint = 20ms
epoch origin = Unix epoch 0
epoch width = 60s
complete single-segment grid = 3,000 checkpoints
eligible core = [epoch_start + 15s, epoch_start + 45s)
```

A raw onset outside a structurally eligible core is `epoch_core_omitted`.

An onset inside the core but with:

```text
t + 200ms > core_close_ns
```

is `confirmation_edge_omitted`. It is removed before anchor veto and thinning,
does not occupy a thinning key, and never receives a retained-trigger row.
Equality is admitted: `t+200ms == core_close_ns` proceeds to veto.

For an onset inside the eligible core, support/veto reads post-action
memories at `t`:

```text
secondary_same_direction_count =
  number of non-leader memories equal to d

secondary_opposite_count =
  number of non-leader memories equal to -d
```

If `secondary_opposite_count > 0`, the onset is `anchor_vetoed`. Otherwise it
is a `veto_admitted trigger`.

Same-direction support count `0,1,2` is descriptive, not an admission gate.

## 10. Fixed-Epoch Thinning

For every:

```text
(capture_id, epoch_id, variant, direction)
```

retain the earliest veto-admitted trigger by:

```text
(candidate_ts_ns, candidate_event_seq)
```

All later veto-admitted triggers in the same key are
`same_key_suppressed`.

Cluster identity remains:

```text
capture_id : epoch_id
```

Opposite directions and all variants in the same epoch share the same
dependence cluster.

Thinning precedes confirmation. A later trigger cannot replace an earlier
retained trigger that later fails confirmation.

## 11. Explicit-Evidence Confirmation

For retained trigger time `t`, confirmation checkpoints are exactly:

```text
t+20ms, t+40ms, ..., t+200ms
```

Confirmation-edge admission already guarantees the registered close lies in
the core. The retained window must still exist and remain in the trigger
segment. Missing rows or a segment change are cancellation conditions, not a
new admission decision.

The decision is completed only at `t+200ms`, never at the first additional
update.

Record separately:

```text
first_additional_same_update_ts_ns/event_seq
confirmation_window_close_ts_ns/event_seq
```

Confirmation requires:

1. at least one leader `NEW_d` action in the ten checkpoints;
2. zero `NEW_-d` actions from any channel in the ten checkpoints.

`NO_UPDATE`, `NEW_NEUTRAL` and stale secondary memory neither confirm nor
veto.

Cancellation booleans are independent:

```text
insufficient_confirmation_history
confirmation_segment_boundary
explicit_opposite_update
no_additional_same_leader_update
```

Canonical primary cancel reason uses the first true atom in that exact order.
If no atom is true, status is `CONFIRMED` and reason is `none`.

## 12. Implicit Prediction

The idea predicts that `TRADE_LED` confirmed structure satisfies:

```text
distinct epoch clusters >= 30
represented research dates >= 4
maximum single-date cluster share <= 0.50
```

The prediction is stronger than merely increasing the two selected F010
clusters.

If the primary fails, the result contradicts this registered support
prediction. Sensitivities cannot rescue it.

## 13. Claim Boundary

The strongest possible claim is:

```text
historical outcome-blind recurrent structural candidate
```

No future price, target, fill, fee, PnL, economic precision or trading claim
is authorized.

## 14. Post-Execution Immutability

The formal one-shot attempt consumes a precommitted tracked `armed` claim and
creates an audited claim-consumption commit/tag before reading any cache.
Completion creates a separate terminal-receipt commit/tag binding the final
tree and result hashes. From claim-consumption commit creation:

- this idea is immutable;
- the execution plan is immutable;
- successor runner and tests are immutable;
- no repair, diagnosis, replacement attempt, alternative parameter run or
  result-driven plan change is allowed.

Contradictory results must be recorded and left unchanged.
