# SKHYNIX Fixed Epoch Leader Trigger With Opposition Veto Research Idea

Date: 2026-08-30

Hypothesis ID:
`FIXED_EPOCH_LEADER_TRIGGER_OPPOSITION_VETO_MSTATE_V1`

## 1. Starting Evidence

The accepted fixed-epoch baseline established:

- local onsets are not scarce: `2,219` common candidates;
- fixed epoch thinning is deterministic and slice/reset invariant;
- the selected `F010` retained only `2` clusters;
- the least strict `F000` retained `34` clusters.

The dominant cancellation reasons were not explicit opposite direction:

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

The evidence therefore suggests that the previous detector mostly rejected
candidates because all three channels did not remain simultaneously fresh
and directional. It did not mostly reject candidates because the market
produced explicit opposite evidence.

## 2. Research Idea

Replace the symmetric three-channel hard intersection:

```text
trade_d AND depletion_d AND ofi_d
```

with an asymmetric and interpretable state:

```text
one registered leader trigger_d
AND no explicit opposite evidence
WITH optional same-direction secondary support
```

The leader channel defines why the event exists. Secondary channels may:

- support the event;
- explicitly veto it if they point in the opposite direction;
- remain neutral, stale or absent without automatically deleting it.

This is not a `2-of-3` vote. Depletion and OFI are both depth-derived and
must not be treated as independent votes.

## 3. Registered Variants

Unique primary:

```text
TRADE_LED
```

The aggressive-trade channel is the leader. Depletion and OFI are optional
support or explicit opposition veto.

Non-rescue sensitivities:

```text
DEPLETION_LED
OFI_LED
```

Each sensitivity uses exactly the same state rules with a different leader.
Neither sensitivity may rescue a failed `TRADE_LED` primary.

## 4. Fixed Directional Thresholds

The previous directional amplitude thresholds remain unchanged:

```text
fast 100ms ratio:
  positive >= +0.50
  negative <= -0.50

medium 500ms ratio:
  positive >= +0.25
  negative <= -0.25

margin = 0.00
```

This version does not test whether weaker directional moves are useful. It
tests only whether the three-channel conjunction and all-checkpoint
persistence were unnecessarily destructive.

## 5. Leader Trigger

At checkpoint `t`, leader channel `c` produces trigger direction `d` iff:

1. the leader receives a new explicit evidence update at `t`;
2. the leader action is `NEW_POS` for `d=+1` or `NEW_NEG` for `d=-1`;
3. the preceding six 20ms checkpoints are all observable leader
   `BACKGROUND=0` in the same segment;
4. no secondary channel has fresh memory equal to `-d` at `t`;
5. `t` lies in a structurally eligible fixed-epoch core.

Fresh same-direction secondary memories are recorded as support count
`0, 1, 2`; they are not an admission requirement.

## 6. Explicit-Evidence Persistence

The retained leader trigger is confirmed over the following `200ms`:

```text
confirmation window = (t, t + 200ms]
```

Confirmation requires:

1. at least one additional explicit `NEW_d` update from the leader;
2. zero explicit `NEW_-d` updates from any of the three channels;
3. complete same-segment checkpoint support through the window.

`NO_UPDATE`, `NEW_NEUTRAL` and stale secondary memory do not automatically
veto the candidate. They also do not count as confirmation.

This differs from the predecessor requirement that every checkpoint remain
in complete three-channel `SIGNAL_d`.

## 7. Fixed Epoch Infrastructure

The accepted baseline remains unchanged:

```text
checkpoint = 20ms
epoch origin = Unix epoch 0
epoch width = 60s
eligible core = [15s,45s)
thinning key = (capture_id, epoch_id, direction)
retain earliest eligible trigger for each key
cluster key = (capture_id, epoch_id)
```

Thinning occurs after anchor-time opposition veto and before persistence
confirmation. A later same-epoch trigger cannot replace an earlier retained
trigger that later fails confirmation.

## 8. Implicit Prediction

The idea predicts that `TRADE_LED` will produce non-vacuous recurrent
confirmed structure:

```text
confirmed distinct epoch clusters >= 30
represented research dates >= 4
maximum single-date cluster share <= 0.50
```

The prediction is intentionally stronger than "more than two events."

If the primary does not satisfy these conditions, the result contradicts
the idea's support prediction. The task must record that result and stop.

## 9. Claim Boundary

This idea can establish only:

```text
historical outcome-blind recurrent structural support
```

It cannot establish:

- future-price direction;
- economic precision;
- maker or taker profitability;
- fill probability;
- live-trading readiness.

No future price, target, fill, fee or PnL field may be accessed.

## 10. No Post-Result Repair

After formal execution begins:

- this idea document is immutable;
- the execution plan is immutable;
- detector code and tests are immutable;
- thresholds and gates are immutable;
- no failed result may trigger repair, additional diagnosis, alternative
  parameter execution or plan revision inside this task.

Unexpected or contradictory results must be recorded as observed.
