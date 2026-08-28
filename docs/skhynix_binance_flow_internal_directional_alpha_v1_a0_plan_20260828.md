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

Methodology dependency:

```text
docs/conditional_risk_research_methodology_kernel_v1.md
```

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

## 8. Causal 20ms Flow Bins

The primary measurement unit is a non-overlapping 20ms bin.

For every bin `b`, record:

```text
T_b:
  signed aggressive trade quantity
  buyer-initiated positive
  seller-initiated negative

V_b:
  total aggressive trade quantity

A_b:
  positive weighted ask-side net depletion

B_b:
  positive weighted bid-side net depletion

O_b:
  signed weighted L1-L5 order-flow imbalance contribution
  upward positive
  downward negative

U_b:
  sum of absolute weighted order-flow contributions

C_b:
  admitted public-message activity count
```

L1-L5 weights remain:

```text
[1,1/2,1/3,1/4,1/5]
```

The bins are event-additive. The implementation must not sum overlapping 50ms
windows as if they were independent observations.

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
  sum(T_b) / sum(V_b)

D_dep(W) =
  (sum(A_b)-sum(B_b))
  /
  (sum(A_b)+sum(B_b))

D_ofi(W) =
  sum(O_b) / sum(U_b)
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
  sum(C_b over previous 500ms)
```

Using only the frozen calibration role, calculate:

```text
Q_activity_60 =
  60th percentile of valid activity_500ms
```

Active flow requires:

```text
activity_500ms >= Q_activity_60
at least two directional component denominators are positive
valid uncrossed L1-L5 book
```

The quantile level `60%` is frozen. Its numeric value is an A0 calibration
output.

No per-date or per-role refit is allowed.

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
A_d(100ms,0.50) >= 2
d * D(500ms) >= 0.25
active-flow support is valid
```

If both directions satisfy the fast rule:

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
A_d(100ms,0.50) >= 2
d * D(500ms) >= 0.25
active-flow support remains valid
```

The candidate checkpoint itself contributes zero elapsed exposure.

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
```

The detector then returns to active mixed-state building.

## 15. Confirmed Directional State

At confirmation, enter:

```text
DOMINANT_ACTIVE_d
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
A_-d(100ms,0.50) >= 2
(-d) * D(500ms) >= 0.25
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
return to DOMINANT_ACTIVE_d
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
abs(D(500ms)) < 0.10
both directions have agreement count <2 at fast threshold
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
return to DOMINANT_ACTIVE_d
reset release exposure
```

If the opposite direction qualifies, release does not emit an anchor. After
local refractory has expired, the detector opens the separately defined
opposite flip candidate.

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
4. same absolute OBI bin of width `0.10`;
5. same bid-depth quintile;
6. same ask-depth quintile;
7. same activity quintile;
8. same trailing 500ms return bin;
9. same trailing 2s volatility quintile;
10. same 30-minute time block;
11. nearest timestamp.

Control meaning:

```text
similar current observable context
without a recently confirmed directional dominance path
```

## 21. Frozen H0, H1 And H2 Information Sets

```text
H0:
  current spread ticks
  current equal-weight L1-L5 OBI
  current bid and ask weighted depth
  activity_500ms
  trailing return 100ms
  trailing return 500ms
  trailing return 2000ms
  trailing realized volatility 2000ms
  30-minute time block
  source-quality state
```

```text
H1 adds:
  anchor type
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

```text
H2 adds:
  intended quote side
  attacked-side replenishment
  local depth recovery
  local adverse-pressure release
  H1 x local-recovery interactions
```

A0 fits none of these models.

## 22. Frozen Downstream Directional Targets

At later stages, let:

```text
m0 = current midpoint at anchor
d  = confirmed direction
k  = barrier in ticks
```

Direction-adjusted competing risks:

```text
n_continuation:
  midpoint first reaches m0 + d*k*tick

n_reversal:
  midpoint first reaches m0 - d*k*tick

n_timeout:
  neither barrier is reached before tau

n_ambiguous:
  both barriers occur at indistinguishable event order
```

Frozen barrier candidates:

```text
1 tick
2 ticks
3 ticks
```

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

## 24. Dependence And Effective Sample Size

Directional anchors may cluster inside long active-flow periods.

Every anchor receives:

```text
dependence_cluster_id =
  (capture_id,floor(anchor_ts/30s))
```

A0 reports:

- anchor count;
- unique 30s clusters;
- anchors per cluster;
- maximum cluster share;
- maximum date share;
- mixed-onset versus flip share;
- direction balance;
- same-state duration;
- inter-anchor distribution.

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

### Gate A0-0: Source Closure

Require:

- 29 admitted captures;
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
calibration role only for Q_activity_60: true
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
raw qualifying checkpoints / anchors >= 3
median inter-anchor interval >= 300ms
```

Both directions and at least two component-pair families must appear.

### Gate A0-5: Dependence Support

Require:

```text
minimum unique 30s dependence clusters: 100
minimum represented dates:               8
maximum single-date cluster share:        0.35
maximum single-cluster anchor share:      0.05
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
```

### Gate A0-7: Follow-Up Geometry

Require at least one horizon with:

```text
overall complete boundary/quality coverage >= 0.95
minimum per-date complete coverage >= 0.80
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

Only:

```text
A0_directional_state_contract_supported
```

may authorize A1 target-support work.

## 29. Verification Requirements

Formal implementation must include focused tests for:

- exact event ordering and checkpoint event key;
- non-overlapping 20ms bin additivity;
- mirrored trade, depletion and OFI signs;
- bounded-ratio identities;
- zero-denominator unavailable semantics;
- calibration-only activity threshold;
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
- reset censoring;
- no raw sign-flip anchor;
- static book and recent-price fields excluded from anchor decisions;
- deterministic dual-direction control labels;
- no future-anchor control exclusion;
- no control reuse;
- dependence cluster construction;
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
