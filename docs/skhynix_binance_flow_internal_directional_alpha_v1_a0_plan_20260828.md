# SKHYNIX Binance FLOW_INTERNAL_DIRECTIONAL_ALPHA_V1 A0 Plan - 2026-08-28

Date: 2026-08-28

Status:

```text
frozen zero-target A0 design contract
execution not authorized by this document
```

Hypothesis identifier:

```text
FLOW_INTERNAL_DIRECTIONAL_ALPHA_V1
```

Research family:

```text
continuous active flow
  -> causal directional dominance transition
  -> direction-adjusted competing risks
  -> incremental alpha beyond current book state
```

Methodology lineage:

```text
docs/conditional_risk_research_methodology_kernel_v1.md
current workspace SHA256:
dd1adee720f613a51393ff97ae3fd026a79d9b553eaa984e629ab5ddbcab7505
```

This lineage document is informative, not normative. It is currently outside
the tracked predecessor evidence set. The present plan is self-contained:
implementation authority comes only from the formulas, transitions, gates and
hash bindings frozen below. A later edit or absence of the lineage document
cannot change this V1 contract.

## 1. Decision

Register a new hypothesis.

The predecessor:

```text
SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1
```

successfully compressed a near-continuous pressure process, but its canonical
A0 result was:

```text
490,307 raw micro crossings
253 candidates
42 confirmed excursions
1 safe-reentry anchor
episode duration p50 172,060ms
episode duration p90 962,582ms
direction switches p50 410.5
direction switches p90 3,170.6
34/42 recovered_without_wide_spread
```

The state machine therefore found long-lived active-flow regimes rather than
short excursions whose complete termination still left a maker spread
opportunity.

The new route does not wait for flow completion.

It asks:

> While active flow is still present, does a causally confirmed transition
> from mixed flow to directional dominance, or from one persistent dominance
> direction to the opposite direction, change the future continuation versus
> reversal law beyond current OBI, spread, depth, recent return, volatility and
> activity?

Primary alignment:

```text
directional_dominance_confirmed_at
```

It is not:

```text
the first raw pressure crossing
the maximum future imbalance
the maximum future return
the end of a complete flow regime
the first future price barrier
an offline trend center
a future-confirmed backdated onset
```

## 2. Version Boundary

`FLOW_INTERNAL_DIRECTIONAL_ALPHA_V1` is not:

- a robustness run of either predecessor;
- a lower novelty threshold for safe reentry;
- a favorable subset of the 42 previous excursions;
- a rule that uses future return to label historical dominance;
- current OBI renamed as alpha;
- a maker fill model;
- a trading strategy.

Load-bearing changes are:

1. the primary object is a directional state transition inside active flow;
2. the anchor no longer requires global quiet or flow termination;
3. pressure features are bounded directional ratios, not quiet-regime
   z-scores;
4. mixed-to-dominant transitions and persistent directional flips are both
   explicit;
5. repeated same-direction pressure remains inside one directional state;
6. local refractory limits repeated anchors without requiring global quiet;
7. future direction is tested only in later competing-risk stages;
8. quote-side recovery is an execution interaction, not the primary anchor.

Any later change to these semantics creates another hypothesis version.

## 3. Layered Research Question

The research chain has three layers.

### Layer 1: Directional State

```text
Should current flow be interpreted as:
  mixed
  upward dominant
  downward dominant
  releasing
  persistently flipping
```

This is the A0 object.

### Layer 2: Directional Outcome

```text
After confirmation in direction d:
  continuation barrier first
  reversal barrier first
  timeout
```

This begins only after A0 passes.

### Layer 3: Execution

```text
Given directional alpha:
  aggressive entry
  passive entry after local quote-side recovery
  reduced size
  no trade
```

Quote-side recovery belongs here. It cannot define or rescue the primary V1
directional anchor.

## 4. Primary Conditional Hypothesis

Let:

```text
C_t = current observable static and recent-price context
F_t = causally observed directional flow path
R_t = local quote-side recovery state
```

The primary nested comparison is:

```text
H0:
  P(T,J | C_t)

H1:
  P(T,J | C_t,F_t)
```

A later execution interaction may compare:

```text
H2:
  P(T,J | C_t,F_t,R_t,F_t x R_t)
```

The directional hypothesis is supported only if `H1` adds out-of-sample
information over `H0`.

`H2` cannot rescue a failed `H1`.

## 5. A0 Purpose And Authority

A0 is a zero-target state-support stage.

A0 may:

- replay admitted Binance public messages;
- reconstruct causal L1-L5 book state;
- aggregate event contributions into 20ms causal bins;
- calculate trailing bounded directional ratios;
- calibrate activity support using the frozen calibration role;
- materialize active, mixed, candidate, dominant, release, flip and local
  refractory states;
- emit causal directional-dominance anchors;
- construct outcome-blind controls;
- audit cadence, duration, compression, direction balance, date support and
  dependence clusters;
- inspect current and trailing-only spread, OBI, depth, return, volatility and
  activity;
- select follow-up horizon using boundary geometry only;
- freeze downstream directional target definitions without materializing
  them.

A0 may not:

- read future midpoint or BBO changes after an anchor;
- determine whether continuation or reversal later occurred;
- choose barriers or horizons using directional returns;
- calculate future markout;
- calculate fill, fees, slippage or PnL;
- fit H0, H1 or H2;
- select thresholds using model loss;
- drop dates, directions or flip states based on future outcomes;
- access private APIs or orders;
- collect new data.

Passing A0 means only:

```text
directional state transitions are causal,
not near-continuous,
historically supported,
and sufficiently distributed for later testing
```

It does not mean:

```text
direction predicts price
gross alpha covers costs
the signal is executable
maker or taker PnL is positive
```

## 6. Historical Evidence Boundary

Existing admitted source:

```text
29 captures
9 research dates
35.9172008142 hours
2026-07-29 through 2026-08-27
```

All dates precede 2026-08-28 and have already been inspected.

Frozen tracked source authority:

```text
predecessor execution commit:
91cc0770c4de3c41c6a27c1980c23414b1f21dbd

predecessor QA record commit:
f7d10df93688c9ce1a6b66c0b751f806494e955d

source manifest:
local_live_analysis/skhynix_safe_reentry_after_flow_excursion_a0_0828T011/
  contracts/source_manifest.json
SHA256:
63cc8eb6cbe4db17b61ab0ce782104d8a6984471180edc5add7b91e812e5f15f

capture inventory:
local_live_analysis/skhynix_safe_reentry_after_flow_excursion_a0_0828T011/
  support/source_inventory.csv
SHA256:
62baf3b498ecafcda901cbea1f61f52489cc01e8c3da95427fdf8294ec27d235

session roles:
local_live_analysis/skhynix_safe_reentry_after_flow_excursion_a0_0828T011/
  contracts/session_role_ledger.csv
SHA256:
5910288825e89dd37c1db3ef98c8025ede0d593cf404c7b8038c45e851c4da3e

predecessor run manifest:
local_live_analysis/skhynix_safe_reentry_after_flow_excursion_a0_0828T011/
  run_manifest.json
SHA256:
c1564f84e793253a39c5d6cccdab2559638bd485b3a9ecf02e9d0d3eb692423b
```

The implementation must load these four blobs from the exact predecessor
commit and verify their hashes before discovering raw files. The admitted
source set is the exact 29-row inventory keyed by:

```text
capture_id
research_date
role
start_utc
end_utc
duration_seconds
raw_size_bytes
raw_sha256
depth_gap_count
```

`raw_path` may be relocated, but every other identity field must match. No
additional capture, replacement hash or role reassignment is allowed under
V1.

Session roles remain:

| Dates | Role |
| --- | --- |
| 2026-07-29 | normalization/activity calibration |
| 2026-07-30, 2026-08-03, 2026-08-04 | historical method development |
| 2026-08-07, 2026-08-24, 2026-08-25 | blocked historical validation |
| 2026-08-26, 2026-08-27 | historical no-refit replay |

No true prospective claim is available.

No date may be reclassified after A0 begins.

## 7. Causal Event Ordering

Every raw row receives:

```text
event_key = (local_receive_ts_ns,event_seq_in_file)
```

Every 20ms checkpoint receives:

```text
checkpoint_event_key =
  (checkpoint_local_receive_ts_ns,last_causally_visible_event_seq)
```

Rules:

1. Preserve file order for equal receive timestamps.
2. Emit a checkpoint before applying a same-timestamp later message.
3. Never use exchange timestamp to move information earlier.
4. Fail closed on snapshot, sequence, capture or quality boundaries.
5. Do not carry bins, windows, candidates or states across a reset.
6. Bind every transition and anchor to exact checkpoint event key.

Checkpoint grid:

```text
segment first checkpoint =
  smallest Unix-epoch multiple of 20ms strictly greater than snapshot receive ts

later checkpoints =
  first checkpoint + n*20ms
```

No checkpoint is emitted before a valid snapshot initializes the segment.

## 8. Causal 20ms Flow Bins

The primary measurement unit is a non-overlapping, right-open 20ms bin ending
at checkpoint `c`:

```text
B_c = [c-20ms,c)
```

An event whose local receive time equals `c` is not visible at checkpoint `c`
and belongs to the next bin. Equal-timestamp rows retain file order. Bins never
cross snapshot, reconnect, sequence-gap, capture or quality-reset boundaries.

### 8.1 Admitted Message Universe

Only:

```text
valid sequence-admitted depthUpdate
valid public trade
```

contribute to flow bins.

Initial or reconnect snapshots establish book state and reset all windows, but
contribute zero flow and zero activity. `bookTicker`, wrappers, subscription
acks and unknown messages contribute zero and cannot modify the reconstructed
L1-L5 state used by this detector.

Activity is message count, not atomic-level count:

```text
C_e = 1 for each admitted depthUpdate message
C_e = 1 for each admitted trade message
C_e = 0 otherwise
```

L1-L5 weights remain:

```text
w_0..w_4 = [1,1/2,1/3,1/4,1/5]
```

### 8.2 Atomic Depth Contribution

Apply each level row inside a `depthUpdate` sequentially in received array
order.

For an updated side `s`, price `p`, old quantity `q_pre` and new quantity
`q_post`:

```text
delta_q = q_post - q_pre
```

Rank is frozen as:

```text
r(p) =
  pre-update top-five rank, if p is in the pre-update top five;
  otherwise post-update top-five rank, if p enters the post-update top five;
  otherwise unavailable.
```

An unavailable rank contributes zero. Rank is zero-based.

For every ranked atomic level update `j`:

```text
ask_depletion_j =
  w_r * max(-delta_q,0) if side=ask else 0

bid_depletion_j =
  w_r * max(-delta_q,0) if side=bid else 0

ofi_j =
  +w_r * delta_q if side=bid
  -w_r * delta_q if side=ask

ofi_abs_j = abs(ofi_j)
```

Thus bid addition and ask removal are upward-positive; bid removal and ask
addition are downward-negative.

Price-level deletion, insertion and top-five migration use the same atomic
rule. There is no separate synthetic contribution for a best-price move. A
snapshot replacement starts a new segment and contributes nothing.

### 8.3 Atomic Trade Contribution

For an admitted Binance public `trade` with quantity `q` and
`buyer_is_maker=m`:

```text
trade_signed_e = -q if m=true
trade_signed_e = +q if m=false
trade_total_e  = q
```

Seller-aggressive flow is negative; buyer-aggressive flow is positive.

### 8.4 Bin Aggregation

For every bin `B_c`, sum atomic contributions:

```text
T_c = sum(trade_signed_e)
V_c = sum(trade_total_e)
A_c = sum(ask_depletion_j)
B_c_dep = sum(bid_depletion_j)
O_c = sum(ofi_j)
U_c = sum(ofi_abs_j)
C_c = sum(C_e)
```

`U_c` is the sum of absolute atomic level contributions, not the absolute
value of net bin OFI.

The bins are event-additive. The implementation must not sum overlapping
upstream 50ms windows as if they were independent observations.

Required identity tests:

```text
V_c >= abs(T_c)
A_c >= 0
B_c_dep >= 0
U_c >= abs(O_c)
sum over adjacent bins == sum over their atomic events
mirrored bid/ask and buy/sell input negates T and O and swaps A/B_dep
```

## 9. Bounded Directional Ratios

Frozen path windows:

```text
fast:    100ms
medium:  500ms
slow:   2000ms
```

For each trailing window `W`, define:

```text
D_trade(W) =
  sum(T_c) / sum(V_c)

D_dep(W) =
  (sum(A_c)-sum(B_c_dep))
  /
  (sum(A_c)+sum(B_c_dep))

D_ofi(W) =
  sum(O_c) / sum(U_c)
```

Each valid component lies in:

```text
[-1,1]
```

Denominator rules:

```text
denominator > 0:
  ratio is valid

denominator == 0:
  ratio is unavailable
```

Forbidden:

```text
replace zero denominator with epsilon
apply a global scale floor
carry the previous ratio through inactivity
interpret unavailable as zero
```

At least two of the three ratios must be available for any state decision.

Composite dominance:

```text
D(W) = median of available directional ratios
```

Agreement count for direction `d in {-1,+1}`:

```text
A_d(W,theta) =
  number of valid components satisfying d * D_component(W) >= theta
```

The component ratios and agreement count remain separately available. The
median cannot hide component disagreement.

## 10. Activity Support

Directional ratios during almost empty bins are not admitted.

Define trailing medium activity:

```text
activity_500ms =
  sum(C_c for atomic events in [checkpoint-500ms,checkpoint))
```

The calibration checkpoint universe contains every 20ms checkpoint on the
frozen calibration role that:

- is at least `2000ms` after the current segment start;
- has a complete contiguous 2000ms bin history;
- has a valid uncrossed reconstructed L1-L5 book;
- is not at or across a reset/quality boundary.

Zero-activity checkpoints remain in the universe.

Using only this frozen universe, calculate:

```text
Q_activity_60 =
  60th percentile of valid activity_500ms
```

Quantile rule:

```text
sort n values ascending
h = (n-1)*0.60
linearly interpolate between floor(h) and ceil(h)
active tie rule: activity_500ms >= Q_activity_60
```

Active flow requires:

```text
activity_500ms >= Q_activity_60
at least two directional component denominators are positive
valid uncrossed L1-L5 book
complete contiguous 2000ms history inside the current segment
```

The quantile level `60%` is frozen. Its numeric value is an A0 calibration
output.

No per-date or per-role refit is allowed.

All primary anchors therefore have a complete 2000ms causal slow window.
For any component/window whose denominator is zero, the ratio remains
unavailable. Model encoding is frozen later in Section 21; missing values are
never silently converted into observed neutral flow.

Current spread, OBI, bilateral depth, trailing return and trailing volatility
do not participate in active-flow, candidate, persistence, confirmation,
flip, release or refractory decisions. They are recorded causally at each
checkpoint only for A0 support description, control matching and the later
`H0` information set.

## 11. Frozen Constants

Primary tuple:

| Item | Value |
| --- | ---: |
| Causal bin/checkpoint | `20ms` |
| Fast path window | `100ms` |
| Medium path window | `500ms` |
| Slow diagnostic window | `2000ms` |
| Activity calibration quantile | `0.60` |
| Mixed-state lookback | `500ms` |
| Fast component dominance | `0.50` |
| Medium composite dominance | `0.25` |
| Required component agreement | `2 of 3` |
| Candidate qualification window | `300ms` |
| Required qualifying exposure | `120ms` |
| Release composite magnitude | `<0.10` |
| Release dwell | `200ms` |
| Local anchor refractory | `300ms` |
| Control stride | `250ms` |
| Recent dominance exclusion for controls | `1000ms` |
| Dependence cluster | `30s` |

Interpretation:

```text
fast component threshold 0.50
  -> at least 75/25 directional split in that component

medium composite threshold 0.25
  -> path-level direction must agree beyond a small instantaneous burst
```

Frozen diagnostics:

```text
fast window 200ms
medium window 1000ms
required exposure 80ms
required exposure 200ms
local refractory 150ms
local refractory 600ms
```

Diagnostics cannot replace or rescue the primary after outcomes are visible.

Frozen checkpoint predicates:

```text
Q_d:
  A_d(100ms,0.50) >= 2
  and d * D(500ms) >= 0.25

Q_both:
  Q_+1 and Q_-1

Q_release:
  abs(D(500ms)) < 0.10
  and A_+1(100ms,0.50) < 2
  and A_-1(100ms,0.50) < 2

Q_mixed:
  abs(D(500ms)) < 0.25
  and A_+1(100ms,0.50) < 2
  and A_-1(100ms,0.50) < 2
```

Whenever `Q_both` is true, the checkpoint is direction-ambiguous:

- it contributes no directional persistence;
- it cannot confirm, renew, reject or flip a direction;
- it may not emit an anchor.

## 12. Active Mixed State

Canonical background is not quiet.

It is:

```text
MIXED_ACTIVE_FLOW
```

`MIXED_ACTIVE_FLOW` requires active-flow support and:

```text
abs(D(500ms)) < 0.25
no direction has 2-of-3 fast components >=0.50
no active dominance candidate
no active confirmed dominance state
```

For a mixed-to-dominance candidate, this state must hold for the complete
preceding `500ms`.

The market may remain busy throughout the mixed period.

This is the central difference from global quiet novelty.

## 13. Mixed-To-Dominance Candidate

After a valid 500ms mixed-active history, direction `d` becomes a candidate
when the current checkpoint satisfies:

```text
Q_d is true
Q_both is false
active-flow support is valid
```

If `Q_both` is true:

```text
candidate_status = direction_ambiguous
candidate rejected
```

Candidate type:

```text
mixed_onset
```

The candidate checkpoint is descriptive. It is not the alignment anchor.

## 14. Persistence Confirmation

Open a causal `300ms` qualification window after candidate creation.

For candidate direction `d`, a complete 20ms interval contributes only when
its ending checkpoint satisfies:

```text
Q_d is true
Q_both is false
active-flow support remains valid
```

The candidate checkpoint itself contributes zero elapsed exposure.
Qualifying exposure is cumulative complete 20ms exposure inside the frozen
300ms window; a nonqualifying checkpoint contributes zero but does not erase
earlier qualifying exposure unless one of the explicit rejection transitions
below fires.

Confirmation requires:

```text
qualifying_exposure_ms >= 120ms
within 300ms after candidate_at
```

The alignment anchor is:

```text
directional_dominance_confirmed_at =
  first checkpoint where 120ms elapsed exposure is causally known
```

It is never backdated.

If support is insufficient:

```text
candidate_status = transient_rejected
at candidate_at + 300ms
reset candidate exposure to zero
enter MIXED_ACTIVE_BUILDING with mixed_elapsed_ms=0
```

If active-flow support disappears before confirmation:

```text
candidate_status = flow_support_lost
candidate rejected
enter INACTIVE_FLOW
```

If opposite direction becomes dominant before confirmation:

```text
candidate_status = pre_confirmation_direction_switch
Q_-d is true
Q_both is false
reset candidate exposure to zero
```

The detector then returns to active mixed-state building.

## 15. Confirmed Directional State

At confirmation, enter:

```text
DOMINANT_ACTIVE_d
```

Anchor identity:

```text
anchor_id =
  SHA256(canonical JSON bytes of:
    {
      "anchor_type": string,
      "capture_id": string,
      "confirmation_event_seq": integer,
      "confirmation_ts_ns": integer,
      "direction": integer,
      "hypothesis_id": string,
      "segment_id": integer
    }
  )
```

Record:

- anchor type;
- direction;
- all component ratios at 100ms, 500ms and 2000ms;
- agreement counts;
- activity;
- dominance acceleration `D(100ms)-D(500ms)`;
- current spread, OBI and bilateral depth;
- trailing-only returns and realized volatility;
- cumulative dominance exposure;
- same-direction renewal count;
- attempted opposite-flip count;
- local recovery state by quote side;
- exact event key.

While active:

- repeated same-direction qualification updates the same state;
- repeated same-direction pressure does not emit another anchor;
- raw sign flips do not immediately change direction;
- a direction change requires a separately persistent flip;
- no future price participates in state maintenance.

`same_direction_renewal_count` increments on a rising edge:

```text
Q_d is true and Q_both is false
after at least one preceding active checkpoint where Q_d was false
```

The confirmation checkpoint is not a renewal. Renewals during local
refractory are counted but never emit anchors.

## 16. Local Refractory

After every confirmed directional anchor:

```text
no new anchor may be emitted for 300ms
```

During this period:

- same-direction evidence updates the active state;
- opposite evidence may be recorded as a flip precursor;
- no opposite anchor is emitted;
- no checkpoint is deleted from the path.

The refractory period limits duplicate anchors. It does not require flow to
become quiet.

## 17. Persistent Directional Flip

After local refractory expires, an active state in direction `d` opens an
opposite flip candidate when:

```text
Q_-d is true
Q_both is false
active-flow support remains valid
```

Candidate type:

```text
persistent_flip
```

The same `300ms / 120ms` persistence contract applies.

Upon confirmation:

```text
close DOMINANT_ACTIVE_d with exit_reason=persistent_flip
emit one directional_dominance_confirmed_at anchor for -d
enter DOMINANT_ACTIVE_-d
```

The flip confirmation is not backdated to its first opposite checkpoint.

If the original direction `d` reasserts before flip confirmation:

```text
flip_status = flip_rejected_original_reasserted
Q_d is true
Q_both is false
return to DOMINANT_ACTIVE_d
reset flip exposure to zero
emit no anchor
```

If the 300ms qualification window expires with less than 120ms opposite
exposure:

```text
flip_status = transient_flip_rejected
return to DOMINANT_ACTIVE_d
reset flip exposure to zero
emit no anchor
```

If active-flow support disappears before flip confirmation:

```text
flip_status = flow_support_lost
close DOMINANT_ACTIVE_d with exit_reason=flow_became_inactive
enter INACTIVE_FLOW
```

## 18. Release To Mixed Flow

An active directional state begins release when:

```text
Q_release is true
active-flow support remains valid
```

Require continuous:

```text
200ms release dwell
```

At causal completion:

```text
close directional state with exit_reason=release_to_mixed
enter MIXED_ACTIVE_FLOW building
```

No anchor is emitted at release.

If direction `d` reasserts before the release dwell completes:

```text
release_status = release_rejected_same_direction_reasserted
Q_d is true
Q_both is false
return to DOMINANT_ACTIVE_d
reset release exposure
```

If the release predicate breaks without either direction qualifying:

```text
release_status = release_interrupted
return to DOMINANT_ACTIVE_d
reset release exposure
```

If `Q_-d` becomes true and `Q_both` is false, release does not emit an anchor.
After local refractory has expired, the detector closes the release candidate
with `release_status=superseded_by_flip` and opens the separately defined
opposite flip candidate at the current checkpoint.

If active-flow support disappears:

```text
close with exit_reason=flow_became_inactive
enter INACTIVE_FLOW
```

Reset or quality failure censors the state.

## 19. Canonical State Machine

States:

```text
INACTIVE_FLOW
MIXED_ACTIVE_BUILDING
MIXED_ACTIVE_READY
DOMINANCE_CANDIDATE_d
DOMINANT_ACTIVE_d
FLIP_CANDIDATE_-d
RELEASE_CANDIDATE
```

### 19.1 Global Transition Precedence

At each checkpoint, apply this precedence:

1. **Reset/quality boundary.** Censor every open candidate/state, clear all
   elapsed counters and enter `INACTIVE_FLOW`.
2. **Active support false.** Close any open candidate with
   `flow_support_lost`; close any confirmed dominant state with
   `flow_became_inactive`; clear all counters and enter `INACTIVE_FLOW`.
3. **Direction ambiguity normalization.** If `Q_both`, record ambiguity and
   treat both directional predicates as false for this checkpoint.
   Qualification exposure does not increase, but candidate age still advances
   and a qualification timeout may still fire.
4. **State-specific rule.** Apply exactly the first matching row in the table
   below.
5. **No predicate matched.** Remain in the current state and apply the row's
   explicit counter behavior.

Reset and active-support rules therefore dominate every state-specific
predicate. One checkpoint can produce at most one transition and at most one
anchor.

### 19.2 Total State Transition Table

| Current state | First matching predicate after global rules | Action | Next state |
| --- | --- | --- | --- |
| `INACTIVE_FLOW` | active support true | set `mixed_elapsed=0`; the opening checkpoint contributes no prior active exposure | `MIXED_ACTIVE_BUILDING` |
| `MIXED_ACTIVE_BUILDING` | `Q_mixed` and cumulative contiguous mixed exposure reaches `500ms` | freeze complete mixed history | `MIXED_ACTIVE_READY` |
| `MIXED_ACTIVE_BUILDING` | not `Q_mixed` | reset `mixed_elapsed=0` | same |
| `MIXED_ACTIVE_BUILDING` | otherwise | add one contiguous 20ms mixed interval | same |
| `MIXED_ACTIVE_READY` | `Q_d` for exactly one direction | open `mixed_onset`; candidate checkpoint exposure `0` | `DOMINANCE_CANDIDATE_d` |
| `MIXED_ACTIVE_READY` | `Q_mixed` | retain ready history | same |
| `MIXED_ACTIVE_READY` | otherwise | reset mixed history | `MIXED_ACTIVE_BUILDING` |
| `DOMINANCE_CANDIDATE_d` | `Q_-d` | reject `pre_confirmation_direction_switch`, clear exposure and mixed history | `MIXED_ACTIVE_BUILDING` |
| `DOMINANCE_CANDIDATE_d` | `Q_d` and exposure reaches `120ms` | confirm at current checkpoint, emit one anchor, start refractory | `DOMINANT_ACTIVE_d` |
| `DOMINANCE_CANDIDATE_d` | candidate age reaches `300ms` before confirmation | reject `transient_rejected`, clear exposure and mixed history | `MIXED_ACTIVE_BUILDING` |
| `DOMINANCE_CANDIDATE_d` | `Q_d` | add 20ms qualifying exposure | same |
| `DOMINANCE_CANDIDATE_d` | otherwise | exposure unchanged | same |
| `DOMINANT_ACTIVE_d` | local refractory expired and `Q_-d` | open opposite candidate with exposure `0` | `FLIP_CANDIDATE_-d` |
| `DOMINANT_ACTIVE_d` | `Q_release` | open release candidate with exposure `0` | `RELEASE_CANDIDATE` |
| `DOMINANT_ACTIVE_d` | rising edge of `Q_d` | increment renewal count, emit no anchor | same |
| `DOMINANT_ACTIVE_d` | otherwise | maintain state | same |
| `FLIP_CANDIDATE_-d` | `Q_d` | reject `flip_rejected_original_reasserted`, clear flip exposure | `DOMINANT_ACTIVE_d` |
| `FLIP_CANDIDATE_-d` | `Q_release` | reject `flip_released_before_confirmation`, start release exposure `0` | `RELEASE_CANDIDATE` |
| `FLIP_CANDIDATE_-d` | `Q_-d` and exposure reaches `120ms` | close old state, confirm flip at current checkpoint, emit one anchor, restart refractory | `DOMINANT_ACTIVE_-d` |
| `FLIP_CANDIDATE_-d` | candidate age reaches `300ms` before confirmation | reject `transient_flip_rejected`, clear flip exposure | `DOMINANT_ACTIVE_d` |
| `FLIP_CANDIDATE_-d` | `Q_-d` | add 20ms qualifying exposure | same |
| `FLIP_CANDIDATE_-d` | otherwise | exposure unchanged | same |
| `RELEASE_CANDIDATE` | local refractory expired and `Q_-d` | close release as `superseded_by_flip`, open flip exposure `0` | `FLIP_CANDIDATE_-d` |
| `RELEASE_CANDIDATE` | `Q_d` | reject `release_rejected_same_direction_reasserted`, clear release exposure | `DOMINANT_ACTIVE_d` |
| `RELEASE_CANDIDATE` | `Q_release` and continuous exposure reaches `200ms` | close dominant state `release_to_mixed`; set mixed exposure `0` | `MIXED_ACTIVE_BUILDING` |
| `RELEASE_CANDIDATE` | `Q_release` | add 20ms continuous release exposure | same |
| `RELEASE_CANDIDATE` | otherwise | close `release_interrupted`, clear release exposure | `DOMINANT_ACTIVE_d` |

For `RELEASE_CANDIDATE`, `d` always refers to the still-open parent dominant
direction. For all candidate ages and refractory clocks, elapsed time is
checkpoint time minus the causal opening/confirmation checkpoint; no opening
checkpoint contributes elapsed exposure.

Canonical mixed-onset path:

```text
INACTIVE_FLOW
  -> MIXED_ACTIVE_BUILDING
  -> MIXED_ACTIVE_READY
  -> DOMINANCE_CANDIDATE_d
  -> DOMINANT_ACTIVE_d
```

Canonical flip path:

```text
DOMINANT_ACTIVE_d
  -> FLIP_CANDIDATE_-d
  -> DOMINANT_ACTIVE_-d
```

Rejected flip:

```text
DOMINANT_ACTIVE_d
  -> FLIP_CANDIDATE_-d
  -> DOMINANT_ACTIVE_d
```

Canonical release:

```text
DOMINANT_ACTIVE_d
  -> RELEASE_CANDIDATE
  -> MIXED_ACTIVE_BUILDING
```

Rejected release:

```text
DOMINANT_ACTIVE_d
  -> RELEASE_CANDIDATE
  -> DOMINANT_ACTIVE_d
```

Forbidden:

```text
raw sign flip -> immediate new anchor
same-direction renewal -> new anchor
future return -> confirm current direction
future barrier -> move anchor earlier
reset boundary -> continued state
```

## 20. Current-State Controls

Controls represent active-flow checkpoints with comparable current state but
without a recently confirmed directional path.

Generate candidates on a:

```text
250ms calendar stride
```

Control calendar grid is Unix-epoch aligned at multiples of `250ms`. For each
grid time, choose the first valid 20ms checkpoint at or after that grid time
inside the same segment. If two grid times map to the same checkpoint, keep
only the earlier grid identity.

A control candidate must:

- have active-flow support;
- have valid current L1-L5 state;
- have no active candidate;
- have no confirmed directional state;
- have no directional anchor in the preceding `1000ms`;
- have no future outcome exclusion.

Each eligible checkpoint is expanded into two deterministic pseudo-candidates:

```text
(checkpoint_event_key,control_direction=+1)
(checkpoint_event_key,control_direction=-1)
```

`control_direction` is only a matching label. It does not classify the
checkpoint as directionally dominant, does not inspect future price, and does
not enter anchor generation. Once either pseudo-candidate is matched, both
copies for that checkpoint are removed so the underlying control checkpoint
cannot be reused.

Match anchors without reuse on:

1. same research date;
2. same direction;
3. same spread-tick value, adjacent value only as final relaxation;
4. same direction-adjusted OBI bin `d*OBI` of width `0.10`;
5. same bid-depth quintile;
6. same ask-depth quintile;
7. same activity quintile;
8. same trailing 500ms return bin;
9. same trailing 2s volatility quintile;
10. same 30-minute time block;
11. nearest timestamp.

All matching strata are frozen from the Section 10 calibration checkpoint
universe:

```text
OBI:
  fixed direction-adjusted bins [-1.0,-0.9),...,[0.9,1.0]

bid depth, ask depth, activity, trailing 500ms return,
trailing 2s volatility:
  calibration empirical quintile edges using the same linear interpolation

edge tie:
  searchsorted(edges,value,side="right"), clipped to bins 0..4
```

Matching relaxation is deterministic:

1. exact frozen strata;
2. only if no exact candidate exists, allow spread difference of one tick;
3. select minimum absolute time distance;
4. break ties by:

```text
(
  control_capture_id,
  control_segment_id,
  control_checkpoint_ts_ns,
  control_checkpoint_event_seq,
  control_direction
)
```

No other covariate, date-specific edge or future-price criterion may relax the
match.

Global no-reuse assignment is a deterministic chronological greedy pass:

```text
anchor processing key =
  (
    research_date,
    anchor_ts_ns,
    anchor_event_seq,
    anchor_id
  )
```

Sort anchors ascending by this key. For each anchor, apply the frozen exact
then one-tick-relaxed candidate ordering above against controls not previously
used. Assign the first eligible control. If no control remains, mark the
anchor unmatched. Once a checkpoint is assigned, remove both direction
pseudo-labels before processing the next anchor.

No parallel matching, randomized ordering, maximum-cardinality rematching or
post-hoc swap is permitted under V1.

Control meaning:

```text
similar current observable context
without a recently confirmed directional dominance path
```

For every matched pair:

```text
anchor entry:
  entry_at = directional_dominance_confirmed_at
  d = confirmed direction
  Z_transition = 1

control entry:
  entry_at = control checkpoint
  d = control_direction
  Z_transition = 0

for both:
  m0 = midpoint causally visible at entry_at
  target clock origin = entry_at
  target orientation = d

pair_id =
  SHA256(canonical JSON bytes of:
    {
      "anchor_id": lowercase hex string,
      "control_capture_id": string,
      "control_checkpoint_event_seq": integer,
      "control_checkpoint_ts_ns": integer,
      "control_direction": integer,
      "control_segment_id": integer,
      "hypothesis_id": string
    }
  )
```

Canonical identity serialization for every V1 SHA:

```text
UTF-8 encoded JSON
object keys sorted lexicographically
separators exactly "," and ":"
ensure_ascii=true
integers encoded in base-10 with no quotes
strings encoded as JSON strings
no whitespace or trailing newline
SHA256 output as lowercase hexadecimal
```

The primary later population is the union of the two entries from every
no-reuse matched pair. Unmatched anchors and unmatched controls are support
diagnostics only and do not enter the primary H0/H1 estimand.

Pair weighting is frozen:

```text
each pair has total weight 1
anchor row weight 0.5
control row weight 0.5
dates receive equal total score weight
```

## 21. Frozen H0, H1 And H2 Information Sets

The primary scientific question is whether the causally confirmed transition
state matters beyond a comparable snapshot:

```text
H0:
  current spread ticks
  direction-adjusted weighted L1-L5 OBI
  log supporting-side weighted depth
  log vulnerable-side weighted depth
  log activity_500ms
  direction-adjusted trailing return 100ms
  direction-adjusted trailing return 500ms
  direction-adjusted trailing return 2000ms
  trailing realized volatility 2000ms
  direction sign main effect
  30-minute time block
```

Frozen causal definitions:

```text
weighted_bid_depth = sum_r w_r*bid_qty_r
weighted_ask_depth = sum_r w_r*ask_qty_r

OBI =
  (weighted_bid_depth-weighted_ask_depth)
  /
  (weighted_bid_depth+weighted_ask_depth)

mid_t = (best_bid_t+best_ask_t)/2

trailing_return_W =
  (mid_t-mid_(t-W))/tick_size

trailing_realized_volatility_2000ms =
  sqrt(sum of squared 20ms log-midpoint changes over [t-2000ms,t))
```

Frozen orientation map:

```text
oriented_OBI = d*OBI
oriented_return_W = d*trailing_return_W

if d=+1:
  supporting_depth = weighted_bid_depth
  vulnerable_depth = weighted_ask_depth

if d=-1:
  supporting_depth = weighted_ask_depth
  vulnerable_depth = weighted_bid_depth

H0 depth fields = log1p(supporting_depth), log1p(vulnerable_depth)
H0 activity field = log1p(activity_500ms)
H0 volatility = unchanged
H0 spread = unchanged
H0 direction main effect = d
H0 time block =
  floor((UTC nanoseconds since midnight at entry_at) / 30 minutes)
  with fixed integer levels 0..47
```

Raw bid/ask depth, raw un-oriented OBI and raw un-oriented returns do not enter
the primary H0 in addition to these oriented fields.

All lag endpoints must exist inside the same valid segment. H0 fields are
measured at `entry_at` before any same-timestamp later message.

```text
H1 primary adds:
  Z_transition
```

`H1` therefore adds one coefficient per competing cause. It does not add a
large flexible path vector that could rediscover the detector in sample.

A secondary descriptor model `H1b` may add:

```text
  mixed_onset versus persistent_flip indicator
  D_trade at 100/500/2000ms
  D_dep at 100/500/2000ms
  D_ofi at 100/500/2000ms
  composite D at 100/500/2000ms
  component agreement
  dominance acceleration
  persistence exposure
  prior dominant-state duration
  same-direction renewal count
  flip precursor count
```

`H1b` uses confirmed anchor entries only. It is a separate within-anchor
strength estimand, not part of the primary matched anchor-versus-control
comparison.

For each unavailable component ratio in `H1b`:

```text
model value = 0
availability indicator = 0
```

For an available ratio:

```text
model value = observed bounded ratio
availability indicator = 1
```

Availability indicators are frozen before target access. `H1b` is secondary
and cannot rescue a failed primary `H1`.

```text
H2 adds:
  intended quote side
  attacked-side replenishment
  local depth recovery
  local adverse-pressure release
  H1 x local-recovery interactions
```

`H2` is fitted only on entries where the corresponding execution-side local
state is causally observable at or after entry under a separately frozen
landmark. It cannot change the primary entry time.

A0 fits none of these models.

## 22. Frozen Downstream Directional Targets

At later stages, for either anchor or matched control entry let:

```text
m0 = current midpoint at entry_at
d  = confirmed direction or frozen control_direction
k  = barrier in ticks
```

Direction-adjusted competing risks:

```text
n_continuation:
  midpoint first reaches m0 + d*k*tick

n_reversal:
  midpoint first reaches m0 - d*k*tick

n_timeout:
  neither barrier is reached at or before entry_at + tau

n_ambiguous:
  both barriers occur at indistinguishable event order
```

Frozen barrier candidates:

```text
primary: 1 tick
diagnostic only: 2 ticks
diagnostic only: 3 ticks
```

The one-tick symmetric first-passage target is the only primary V1 target.
The two- and three-tick targets cannot select, replace or rescue the primary.

Frozen horizon candidates:

```text
100ms
250ms
500ms
1000ms
2000ms
5000ms
```

A0 may inspect only boundary/quality coverage for these horizons.

It may not inspect barrier outcomes.

Primary target event ordering scans causally reconstructed event-level BBO
states after `entry_at`, preserving `(local_receive_ts_ns,event_seq_in_file)`.
Checkpoint coarsening is not used for barrier order.

The primary observation window is:

```text
event_key > entry_event_key
and local_receive_ts_ns <= entry_ts_ns + tau_ns
```

A barrier first reached exactly at `entry_at + tau` is an observed cause, not
a timeout. Timeout is assigned only after the complete right-closed endpoint
has been observed with neither barrier hit.

`n_ambiguous` means the event order cannot be distinguished because both
barrier labels would be assigned to the same event key or the first observable
post-entry state appears beyond a reset/quality discontinuity.

Frozen primary disposition:

```text
n_ambiguous:
  administratively censor immediately before the ambiguous event key
  do not assign continuation or reversal
  retain the entry in denominator and ambiguity diagnostics
```

Discrete risk-row disposition for any administrative censor inside elapsed
bin `j`:

```text
retain all fully completed prior bins
omit the partially observed current bin j
omit all later bins
do not add a no-event row for the partial bin
```

The same partial-bin rule applies to reset, quality and capture-end censoring.
An observed continuation or reversal inside bin `j` retains bin `j` with the
cause label.

A1 target support requires:

```text
overall ambiguous share <= 0.01
per-date ambiguous share <= 0.05
```

If either ambiguity gate fails, V1 stops at A1. No tie-breaking by future
return, later quote or favorable direction is allowed.

## 23. Incremental Directional Test

The primary later claim is not:

```text
continuation probability > 0.5
```

It is:

```text
H1 improves direction-adjusted competing-risk prediction over H0
out of sample and across dates
```

### 23.1 Primary Estimator

Use one discrete-time multinomial competing-risk hazard on the matched-pair
population.

Frozen elapsed bins:

```text
event_key > entry_event_key and 0ms <= elapsed <= 100ms
100ms < elapsed <= 250ms
250ms < elapsed <= 500ms
500ms < elapsed <= 1000ms
1000ms < elapsed <= 2000ms
2000ms < elapsed <= 5000ms
```

Truncate this list mechanically at the `primary_tau` frozen by A0-7. A1, A2
and A3 read that value and may not reject, extend, shorten or reselect it.

For cause `k in {continuation,reversal}`:

```text
eta_H0,k(i,u) = alpha_k(u) + gamma_k' C_i
eta_H1,k(i,u) = alpha_k(u) + gamma_k' C_i + beta_k*Z_transition_i
```

The no-event state is the multinomial reference. H0 and H1 use identical
entries, risk rows, targets, censoring, weights, preprocessing and elapsed
bins.

Preprocessing:

- continuous H0 fields: median imputation and standardization fitted on
  development dates only;
- categorical fields: frozen levels plus explicit unknown level;
- no feature selection after target access;
- direction orientation uses the exact Section 21 map before fitting.

Preprocessing statistics use each unique development entry once, without
hazard-row expansion and without pair/date weights:

```text
impute continuous field with unweighted development-entry median
standardize with unweighted development-entry mean and population std
if population std=0, standardized value is fixed to 0 and field is retained
categorical reference = lexicographically first frozen level
```

LODO preprocessing is fold-local:

```text
for held-out development date v:
  fold training dates = other two development dates
  fit continuous medians, means and population std only on fold training entries
  transform fold training and held-out entries with those statistics
  fit H0 on fold training dates
  score H0 on held-out date

after lambda selection:
  refit preprocessing on all three development dates
  refit final H0 and H1 on all three development dates
  freeze preprocessing and coefficients
  apply unchanged to blocked validation and historical no-refit replay
```

Primary categorical handling is not data-discovered:

```text
time block levels: fixed integers 0..47, reference 0
direction sign: numeric value -1 or +1, not one-hot encoded
unknown time block: invalid entry, not an unknown category
```

No held-out development entry may influence its fold's imputation,
standardization, feature validity or fitted coefficients.

Use ridge penalties:

```text
lambda grid = [0.01,0.1,1,10,100]
```

Select one lambda using H0-only, leave-one-development-date-out,
date-equal entry negative log loss. Freeze the same lambda for H0 and H1.

Lambda selection tie rule:

```text
compute OOF NLL in float64
minimum = smallest finite OOF NLL
tied = lambdas with abs(OOF_NLL-minimum) <= 1e-12
select max(tied)
```

Primary optimizer contract:

```text
algorithm: deterministic float64 L-BFGS-B
initial coefficients: all zeros
maximum iterations: 2000
gradient tolerance: 1e-8
function tolerance: 1e-12
parameter bounds: none
warm start across lambdas or H0/H1: forbidden
```

Convergence requires:

```text
optimizer success=true
all coefficients finite
objective finite
maximum absolute analytic gradient <= 1e-6
```

Any failed fold, lambda fit, final H0 or final H1 fit fails A3. No solver,
initialization or tolerance switch is allowed as rescue.

Exact weighted fitting objective:

```text
objective =
  mean over development dates(
    mean over matched pairs on date(
      0.5*entry_NLL(anchor)
      +
      0.5*entry_NLL(control)
    )
  )
  +
  lambda/2 * sum(theta_j^2 for j in penalized coefficients)
```

`entry_NLL` is the sum of multinomial hazard-row negative log likelihood over
that entry's at-risk elapsed bins.

Penalty mask:

```text
unpenalized:
  all cause-specific elapsed-bin baselines alpha_k(u)

penalized with the same selected lambda:
  all continuous H0 coefficients
  all non-reference categorical dummy coefficients
  direction main-effect coefficient
  all explicit unknown-level coefficients
  beta_continuation and beta_reversal in H1
```

There is no additional global intercept outside `alpha_k(u)`. H0 and H1 use
the identical preprocessing statistics and penalty mask; only the two
`Z_transition` cause coefficients are added in H1.

Frozen role chain:

```text
calibration:
  2026-07-29

model development and ridge selection:
  2026-07-30, 2026-08-03, 2026-08-04

blocked validation:
  2026-08-07, 2026-08-24, 2026-08-25

historical no-refit replay:
  2026-08-26, 2026-08-27
```

No blocked-validation or no-refit-replay target may influence preprocessing,
lambda, coefficients, thresholds or model structure.

### 23.2 Primary Evaluation And Materiality

Primary score:

```text
Delta_NLL =
  date-equal entry NLL_H0
  -
  date-equal entry NLL_H1
```

Passing directional increment requires all:

```text
Delta_NLL >= 0.002 nats per entry
95% dependence-preserving bootstrap lower bound > 0
beta_continuation > 0
beta_reversal < 0
at least 4 of 5 blocked/replay dates have Delta_NLL > 0
```

Bootstrap unit is the frozen pair-dependence connected component defined in
Section 24. Within each date:

1. sample the date's pair components with replacement, using the original
   number of components;
2. include every matched pair and both entries from each sampled component;
3. recompute pair-weighted, date-equal `Delta_NLL`;
4. aggregate dates with equal weight.

This preserves matched pairs, shared 30s clusters and overlapping follow-up
intervals simultaneously.

Frozen bootstrap details:

```text
replicates: 2000
seed: 20260828
model handling: fixed-model score bootstrap
refit inside replicate: false
lower bound: one-sided 5th percentile of Delta_NLL replicates
```

H0/H1 preprocessing, coefficients and entry predictions are fitted once under
the frozen development procedure. Bootstrap replicates resample only the
blocked/replay evaluation pair components and recompute the fixed-prediction
score difference.

Percentile rule:

```text
sort B=2000 finite replicate values ascending
h = (B-1)*0.05
linearly interpolate between floor(h) and ceil(h)
ties remain repeated observations
non-finite replicate values cause A3 failure
```

Exact random draw contract:

```text
PRNG: NumPy Generator(PCG64(20260828))
date order: ascending ISO research_date
component order within date:
  ascending pair_dependence_component_id lowercase hex
replicate order: r=0..1999
loop nesting:
  for replicate r
    for date in ascending order
draw for a date with n components:
  rng.integers(0,n,size=n,endpoint=false,dtype=int64)
sampling probability: equal 1/n with replacement
duplicate draws: retain full multiplicity
reseed between dates or replicates: forbidden
```

The generator stream is created once before replicate zero. Component members
are expanded in ascending `pair_id` order before fixed-prediction scoring.

Frozen evaluation families:

- blocked-date out-of-sample log score;
- cause-specific calibration;
- continuation-minus-reversal probability separation;
- block-bootstrap confidence intervals;
- date-level effect consistency;
- H1 coefficient/path stability;
- placebo direction and time-shift nulls.

An apparent signal that disappears after current OBI, recent return or
volatility enters H0 is not incremental flow alpha.

The secondary `H1b` descriptor model, alternative barrier, alternative
horizon, stronger H0 and H2 execution interaction cannot rescue a failed
primary H1.

## 24. Dependence And Effective Sample Size

Directional anchors may cluster inside long active-flow periods.

Every anchor and underlying control checkpoint receives:

```text
dependence_cluster_id =
  (capture_id,floor(entry_at/30s))

pair_cluster_id =
  (
    anchor_dependence_cluster_id,
    control_dependence_cluster_id
  )
```

A0 reports:

- anchor count;
- unique anchor, control and pair clusters;
- anchors, controls and matched pairs per cluster;
- maximum anchor, control and pair cluster share;
- maximum date share;
- mixed-onset versus flip share;
- direction balance;
- same-state duration;
- inter-anchor distribution.

For each predeclared follow-up horizon `tau`, create a geometry-only overlap
graph:

```text
node = one matched anchor or control entry
edge = same capture and [entry_at,entry_at+tau] intervals overlap
```

The graph uses timestamps and quality boundaries only. It does not inspect
midpoint or barrier outcomes.

A0 reports:

- unique overlap components;
- maximum component share;
- p50/p90/p99 component size;
- fraction of entries in components larger than 10;
- the same metrics by date.

For the mechanically selected primary horizon, create the unique inference
graph used by the bootstrap:

```text
node = one complete matched pair

edge between pair a and pair b if either:
  any entry from a and any entry from b share the same 30s
  dependence_cluster_id;

  or any same-capture primary follow-up intervals from a and b overlap.

pair_dependence_component_id =
  lexicographically smallest pair_id in the connected component
```

Because matching is same-date, every component is date-contained. The graph
is deterministic, undirected and uses timestamp geometry only.

A0/A1 report:

- unique pair-dependence components overall and by date;
- pair and entry counts per component;
- maximum component share overall and by date;
- component p50/p90/p99 sizes;
- exact component membership SHA.

Later inference clusters at least by capture and 30s block. Row-level iid
standard errors are forbidden.

## 25. Trading-Cost Boundary

Directional predictability is not tradability.

Later cost analysis must keep separate:

```text
gross direction-adjusted midpoint move
taker entry hurdle
passive-entry queue bound
fees
slippage
latency buffer
inventory/exit cost
```

For an aggressive entry, a later conservative hurdle is:

```text
expected signed midpoint move
  >
entry half-spread
+ taker fee
+ slippage
+ latency buffer
+ exit assumption
```

For passive entry, public contact is not a real fill.

No cost or fill field is materialized in A0.

## 26. Required A0 Outputs

Formal execution must publish:

```text
contracts/
  source_manifest.json
  event_ordering_contract.json
  flow_bin_contract.json
  directional_ratio_contract.json
  activity_support_contract.json
  dominance_state_machine.json
  persistence_contract.json
  local_refractory_contract.json
  control_support_contract.json
  downstream_target_stub.json
  H0_H1_H2_contract.json
  dependence_contract.json
  transition_precedence_contract.json
  gate_contract.json
  session_role_ledger.csv
  outcome_access_ledger.json

support/
  feature_availability_by_date.csv
  activity_support_by_date.csv
  mixed_state_support_by_date.csv
  candidate_support_by_date.csv
  candidate_rejection_composition.csv
  directional_anchor_ledger.csv
  directional_anchor_support_by_date.csv
  anchor_type_composition.csv
  direction_balance_by_date.csv
  dominant_state_ledger.csv
  dominant_state_duration_distribution.csv
  flip_transition_composition.csv
  release_transition_composition.csv
  inter_anchor_distribution.csv
  dependence_cluster_support.csv
  control_dependence_support.csv
  pair_dependence_support.csv
  followup_overlap_components.csv
  pair_dependence_components.csv
  active_flow_burst_density.csv
  crossing_to_anchor_compression.csv
  current_spread_distribution.csv
  control_candidates.csv
  matched_control_pairs.csv
  control_overlap_by_date.csv
  followup_geometry.csv

reports/
  A0_summary.json

classification.json
run_manifest.json
```

Large ledgers and caches remain outside Git. Git tracks compact contracts,
summaries, support tables and exact hashes.

## 27. A0 Gates

### 27.1 Frozen Gate Statistic Formulas

Detector-ready interval:

```text
one complete 20ms interval
whose ending checkpoint:
  is inside one capture and one segment;
  has valid uncrossed L1-L5 state;
  has complete contiguous 2000ms history;
  is not at a reset or quality boundary.
```

Durations:

```text
detector_ready_hours =
  20ms * count(detector-ready intervals) / 3.6e6ms

active_flow_hours =
  20ms * count(detector-ready intervals whose ending checkpoint
               has active-flow support=true) / 3.6e6ms
```

Zero denominator:

```text
if detector_ready_hours=0 or active_flow_hours=0:
  corresponding rate is unavailable
  corresponding rate gate fails
```

Rates:

```text
anchor_rate_per_hour =
  total confirmed anchors / detector_ready_hours

active_flow_anchor_rate =
  total confirmed anchors / active_flow_hours
```

Raw qualifying checkpoints:

```text
one checkpoint-direction pair (checkpoint_event_key,d) where:
  detector-ready=true
  active-flow support=true
  Q_d=true
  Q_both=false
```

Each pair is counted once regardless of current state. If anchor count is
zero, `raw_qualifying_checkpoints/anchors` is unavailable and its gate fails.

Inter-anchor intervals:

```text
sort anchors by event key within each (capture_id,segment_id)
take differences only between consecutive anchors in that same group
pool those positive differences across groups
```

Median rule:

```text
sort n pooled values
h=(n-1)*0.50
linearly interpolate floor(h),ceil(h)
no pooled gap values -> gate fails
```

Same-capture 5s burst:

```text
for every anchor a:
  count anchors in the same (capture_id,segment_id)
  with anchor_ts in [a.anchor_ts,a.anchor_ts+5s)

maximum 5s burst = maximum of those counts
```

The right endpoint is excluded. Equal-timestamp anchors are ordered by event
sequence but counted in the same window.

Direction and type shares:

```text
minority_direction_share =
  min(count(d=+1),count(d=-1)) / total anchors

mixed_onset_share =
  count(anchor_type=mixed_onset) / total anchors

single_date_anchor_share =
  max_date(anchor count on date) / total anchors
```

Zero total anchors makes all shares unavailable and their gates fail.

Component-pair family assignment at each anchor uses direction-adjusted
100ms component ratios at threshold `0.50`:

```text
qualifying component set =
  {trade,dep,ofi where d*D_component(100ms) >=0.50}

if exactly two qualify:
  contribute their one unordered pair family

if all three qualify:
  contribute all three unordered pair families:
    dep_trade
    dep_ofi
    trade_ofi
```

The gate's family count is the cardinality of the union across anchors.
Unavailable components do not qualify. Pair names and ordering are exactly:

```text
dep_trade
dep_ofi
trade_ofi
```

### Gate A0-0: Source Closure

Require:

- predecessor execution commit exactly `91cc0770`;
- all four frozen authority blob SHA256 values match Section 6;
- exact 29-row admitted inventory identity;
- exact size and SHA closure;
- zero unhandled depth gaps;
- deterministic replay;
- all reset and quality boundaries represented.

### Gate A0-1: Zero-Outcome Boundary

Require:

```text
future midpoint fields read: []
future BBO fields read: []
continuation/reversal targets materialized: false
future markout fields read: []
cost/fill/PnL fields read: []
H0/H1/H2 fitted: false
new collection: false
private/order access: false
```

### Gate A0-2: Feature Availability

Require:

```text
valid bounded ratios at every anchor: at least 2 of 3
all valid ratios inside [-1,1]: true
zero denominator represented as unavailable: true
epsilon/floor denominator substitutions: 0
all anchors have complete contiguous 2000ms segment history: true
20ms right-open bin boundary violations: 0
non-admitted message contributions: 0
U below abs(O) violations: 0
calibration role only for Q_activity_60: true
calibration quantile interpolation and tie rule exact: true
per-date refits: 0
overall active-checkpoint two-component availability >= 0.90
minimum per-date availability >= 0.80
```

### Gate A0-3: Directional Anchor Support

Require:

```text
minimum anchors:                         500
minimum represented research dates:     8
minimum anchors per represented date:   30
anchor rate envelope:                    5 to 150 per hour
maximum single-date anchor share:        0.35
minority direction share:                0.25
minimum mixed-onset anchor share:        0.20
```

Persistent flips are reported but are not required to reach a minimum share.

### Gate A0-4: State Semantics And Compression

Require:

```text
mixed-onset anchors violating 500ms mixed history: 0
anchors with <120ms elapsed persistence: 0
backdated confirmations: 0
same-direction renewal anchors: 0
anchors inside 300ms local refractory: 0
raw sign flip directly creating anchor: 0
overlapping dominant-state IDs: 0
checkpoints with multiple transitions: 0
checkpoints with multiple anchors: 0
candidate/flip/release counters surviving reset or support loss: 0
raw qualifying checkpoints / anchors >= 3
median inter-anchor interval >= 1000ms
maximum anchors in any same-capture 5s window <= 6
active-flow anchor rate <= 300 per active-flow hour
```

Both directions and at least two component-pair families must appear.

### Gate A0-5: Dependence Support

Require:

```text
minimum unique 30s dependence clusters: 100
minimum unique control 30s clusters:     100
minimum unique pair clusters:            100
minimum represented dates:               8
maximum single-date cluster share:        0.35
maximum single-cluster anchor share:      0.05
maximum single-cluster control share:     0.05
maximum single-pair-cluster share:        0.05
median anchors per cluster <=             5
```

### Gate A0-6: Control Common Support

Require:

```text
minimum unique matched pairs:             500
overall anchor-to-control support:         0.90
minimum per-date common support:           0.75
maximum single-date matched-pair share:    0.35
control reuse:                             0
underlying checkpoint reuse through opposite pseudo-label: 0
matched entries with missing target-origin midpoint: 0
```

### Gate A0-7: Follow-Up Geometry

Require at least one horizon with:

```text
overall complete boundary/quality coverage >= 0.95
minimum per-date complete coverage >= 0.80
minimum overlap components:                100
maximum overlap-component entry share:     0.05
maximum per-date overlap-component share:  0.10
minimum pair-dependence components:         100
maximum pair-dependence pair share:         0.05
maximum per-date pair-dependence share:     0.10
```

Select the largest predeclared horizon satisfying all eight geometry conditions.
This selection reads timestamps and quality boundaries only.

A0-7 writes:

```text
primary_tau
primary_tau_geometry_metrics
eligible_horizons_in_ascending_order
```

to `downstream_target_stub.json` and `classification.json`. If no horizon
passes, `primary_tau=null`, A0 fails, and A1 is unauthorized.

Once A0 passes:

```text
A1/A2/A3 primary_tau authority = read-only A0-7 output
horizon reselection in A1/A2/A3 = forbidden
```

## 28. A0 Classifications

Passing:

```text
A0_directional_state_contract_supported
```

Failures:

```text
A0_source_not_admissible
A0_zero_outcome_boundary_violated
A0_directional_feature_support_failed
A0_directional_anchor_support_insufficient
A0_directional_anchor_near_continuous
A0_directional_state_semantics_failed
A0_directional_anchor_date_concentrated
A0_dependence_support_insufficient
A0_control_common_support_insufficient
A0_followup_geometry_insufficient
```

Canonical multi-failure rule:

```text
gate evaluation order:
  A0-0, A0-1, A0-2, A0-3, A0-4, A0-5, A0-6, A0-7

failed_gates:
  every failed gate ID in that order

failed_conditions:
  every failed atomic condition in gate order and in the written order
  inside each gate
```

Primary `classification` is unique:

```text
no failed gates:
  A0_directional_state_contract_supported

first failed gate A0-0:
  A0_source_not_admissible

first failed gate A0-1:
  A0_zero_outcome_boundary_violated

first failed gate A0-2:
  A0_directional_feature_support_failed

first failed gate A0-3:
  if anchor rate >150/hour:
    A0_directional_anchor_near_continuous
  else if maximum single-date anchor share >0.35:
    A0_directional_anchor_date_concentrated
  else:
    A0_directional_anchor_support_insufficient

first failed gate A0-4:
  if any zero-violation invariant fails, either direction is absent,
  or fewer than two component-pair families appear:
    A0_directional_state_semantics_failed
  else:
    A0_directional_anchor_near_continuous

first failed gate A0-5:
  A0_dependence_support_insufficient

first failed gate A0-6:
  A0_control_common_support_insufficient

first failed gate A0-7:
  A0_followup_geometry_insufficient
```

`classification.json` must contain:

```text
classification
gate_results in canonical gate order
failed_gates
failed_conditions
A1_authorized
```

`A1_authorized=true` only when `failed_gates=[]`.

Only:

```text
A0_directional_state_contract_supported
```

may authorize A1 target-support work.

## 29. Verification Requirements

Formal implementation must include focused tests for:

- exact event ordering and checkpoint event key;
- exact `[c-20ms,c)` bin membership, including same-timestamp events;
- admitted message-type activity count;
- pre-rank/post-rank fallback for atomic depth updates;
- no synthetic snapshot or best-price-move contribution;
- non-overlapping 20ms bin additivity;
- mirrored trade, depletion and OFI signs;
- atomic `U=sum(abs(ofi_j))` identity;
- bounded-ratio identities;
- zero-denominator unavailable semantics;
- complete 2000ms segment warm-up;
- calibration-only activity threshold;
- exact quantile interpolation and active tie rule;
- no per-date refit;
- complete 500ms mixed-active history;
- candidate checkpoint contributes zero persistence exposure;
- causal 120ms persistence confirmation;
- no confirmation backdating;
- transient rejection;
- candidate loss of active-flow support;
- pre-confirmation direction switch rejection;
- same-direction renewal under one state;
- no anchor during 300ms local refractory;
- persistent flip confirmation;
- rejected flip returning to the original dominant state;
- release dwell;
- rejected release returning to the original dominant state;
- interrupted release and release-to-flip precedence;
- exhaustive state/predicate transition precedence;
- at most one transition and one anchor per checkpoint;
- reset censoring;
- no raw sign-flip anchor;
- static book and recent-price fields excluded from anchor decisions;
- deterministic dual-direction control labels;
- chronological global no-reuse matching assignment;
- no future-anchor control exclusion;
- no control reuse;
- matched-pair risk-origin and weight identity;
- anchor/control/pair dependence cluster construction;
- geometry-only overlap-component construction;
- pair-dependence component construction preserving complete pairs;
- unique primary one-tick target;
- right-closed `tau` endpoint barrier handling;
- ambiguous-target administrative censoring;
- partial-bin censor omission and observed-cause bin retention;
- exact H0 direction-orientation map;
- exact primary penalty mask and weighted objective;
- canonical anchor/control/pair SHA serialization;
- fold-local LODO preprocessing and final all-development refit;
- deterministic lambda tie rule and optimizer convergence;
- fixed-model component bootstrap and one-sided percentile interpolation;
- exact PCG64 component sampling stream;
- canonical multi-gate classification and ordered failure ledger;
- zero-target outcome ledger;
- deterministic double-build identity.

Required static checks:

```text
focused pytest
Python compile
ruff
Markdown fence parity
git diff --check
artifact size/SHA closure
```

## 30. Explicitly Forbidden Rescue

After A0 begins, do not:

- lower dominance ratios because anchors are sparse;
- shorten mixed history;
- shorten 120ms persistence;
- shorten local refractory;
- raise the anchor-rate ceiling after observing density;
- drop persistent flips;
- drop dates with weak direction balance;
- select component families using future return;
- add recent return to the anchor after seeing outcomes;
- select a favorable barrier or horizon using the same dates;
- wait for future price confirmation and backdate the anchor;
- call gross midpoint predictability executable alpha;
- call public contact a real fill;
- fit H0/H1/H2 before A0 passes;
- describe a changed tuple as V1.

Any such change creates:

```text
FLOW_INTERNAL_DIRECTIONAL_ALPHA_V2
or another new hypothesis identifier
```

## 31. Ordered Stage Chain

```text
A0 directional state and support
  -> A1 barrier observability and competing-risk support
  -> A2 direction-adjusted target materialization
  -> A3 blocked-date H0 versus H1 incremental test
  -> A4 dependence nulls and historical transport
  -> A5 H2 local-recovery interaction and execution-cost analysis
  -> prospective confirmation on future dates
```

Later stages cannot rescue a failed A0 or H1.

## 32. Frozen Claim Boundary

This document asserts only:

```text
FLOW_INTERNAL_DIRECTIONAL_ALPHA_V1 is a new,
causal and outcome-blind hypothesis for directional
state transitions inside continuous active flow.
```

It does not assert:

```text
the detector has been implemented
directional anchors have historical support
flow direction predicts price
H1 adds information over H0
gross alpha covers spread or fees
local recovery improves execution
the strategy is actionable or profitable
```

Those claims remain locked behind the ordered gates.
