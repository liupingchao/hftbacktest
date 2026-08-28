# SKHYNIX Binance LIQUIDITY_BREAK_ONSET_V1 A0 Causal-Anchor Contract - 2026-08-28

Date: 2026-08-28

Revision: 1

Status: frozen A0 design contract; execution not authorized by this document

Hypothesis identifier:

```text
LIQUIDITY_BREAK_ONSET_V1
```

Research family:

```text
continuous_background_interpretable_transition_alignment
```

Methodology dependency:

```text
docs/conditional_risk_research_methodology_kernel_v1.md
```

Predecessor result:

```text
OBI_REVERSAL_V1
  -> A3_no_increment_over_H0
```

## 1. Decision

Retain the alignment idea, but move the alignment point from a completed or
confirmed state transition to the first causally observable structural break.

The new route is:

```text
continuous market background
  -> earliest observable coherent liquidity pressure
  -> start_anchor at the first causal detection message
  -> distinguish adverse continuation from liquidity recovery
  -> test incremental transition information against current static state
```

The primary alignment is:

```text
liquidity_break_onset_detected_at
```

It is not:

```text
completed OBI reversal
confirmed future dwell
retrospectively estimated change point
first future price move
maximum-pressure timestamp
episode center
```

## 2. Version Boundary

`LIQUIDITY_BREAK_ONSET_V1` is a new hypothesis version.

It must not be described as:

- a faster OBI reversal;
- an alternative confirmation dwell for `OBI_REVERSAL_V1`;
- a barrier or horizon robustness check;
- an A4 rescue of the failed reversal result;
- a rediscovered `N -> S -> P -> R -> N` phase cycle.

The following predecessor facts remain binding:

- queue-shock candidates can form a near-continuous process;
- discrete HSMM phases were inferior to continuous autoregressive baselines;
- completed OBI reversal history did not improve the frozen H0 out of sample;
- all existing SKHYNIX dates are historically consumed.

## 3. Primary Hypothesis

Let `d` be the pressure direction:

```text
d = +1:
  upward pressure
  ask side is vulnerable
  buyer-initiated trades are same-direction flow

d = -1:
  downward pressure
  bid side is vulnerable
  seller-initiated trades are same-direction flow
```

The hypothesis is:

> When vulnerable-side net depletion, aggressive trade pressure and
> whole-book flow pressure first become coherently abnormal, that causal
> onset changes the subsequent competing-risk transition law beyond the
> information in the current static book state.

The claim is conditional:

```text
current market state C_t
transition-pressure state M_t
  ->
P(T, J | C_t, M_t)
```

The pattern is not a rigid path. It is an interpretable conditional
state-transition law.

## 4. A0 Purpose And Authority

A0 is a zero-target causal-anchor stage.

A0 may:

- reconstruct the Binance public event stream;
- compute causal order-book and trade-flow components;
- calibrate covariate-only robust scales;
- materialize onset and release anchors;
- construct outcome-blind control support;
- measure anchor cadence, overlap, duration and coverage;
- freeze the downstream target and H0/H1 contracts without materializing
  their outcomes.

A0 may not:

- read future midpoint or best-quote transitions;
- calculate adverse or recovery event rates;
- plot future prices aligned to anchors;
- select thresholds using future returns, first passage, markout or PnL;
- fit H0 or H1;
- select a target barrier or horizon from predictive results;
- access private APIs, orders, fills or execution data;
- collect new data.

Passing A0 means only:

```text
the causal anchor is well-defined and has sufficient historical support
```

It does not mean:

```text
the anchor predicts price
the anchor is actionable
the anchor supports a maker strategy
```

## 5. Source And Information Filtration

Primary source:

```text
existing admitted Binance public SKHYNIXUSDT captures
```

Allowed public channels:

```text
snapshot
depthUpdate
trade
bookTicker for quality and current-book consistency only
```

The primary detector may consume:

- local receive timestamp;
- deterministic file sequence;
- snapshot and depth-update sequence identifiers;
- L1-L5 prices and quantities;
- depth quantity increases and decreases;
- public trade quantity and aggressor side;
- current source age and reconstruction quality.

The detector must not consume:

- future messages;
- exchange timestamps that arrive after the current local decision point;
- future-confirmed dwell;
- midpoint delta;
- future best-price movement;
- microprice movement;
- post-onset return;
- any target label.

Current book prices may be used only to maintain ranked L1-L5 levels, infer
tick size, orient the vulnerable side and enforce crossed-book quality rules.
Price movement itself is not an onset component.

## 6. Deterministic Event Ordering

Every raw row receives:

```text
event_key = (local_receive_ts_ns, event_seq_in_file)
```

Ordering rules:

1. Sort by `local_receive_ts_ns`.
2. Preserve file order for equal receive timestamps.
3. Apply a message to the causal book before evaluating the detector at that
   message.
4. Bind the decision point to the exact `event_key`, not timestamp alone.
5. Never reorder equal-timestamp depth and trade messages by event type.

Depth continuity remains:

```text
snapshot bridge
  -> U/u/pu continuity
  -> fail closed on a sequence gap
```

Reset boundaries are:

- capture start;
- snapshot rebootstrap;
- sequence gap;
- crossed or incomplete top-five book;
- source-quality failure;
- capture end.

An onset cannot span a reset boundary.

## 7. Event-Time Windows

The detector is event driven. It does not wait for a 100ms observation grid.

Frozen primary pressure window:

```text
W = 50ms
```

Frozen diagnostic-only windows:

```text
20ms
100ms
```

The diagnostic windows may report support and cadence, but they cannot replace
the 50ms primary after downstream outcomes are visible.

At event `e_t`, every rolling window is left-open and right-closed:

```text
(t - W, t]
```

Only events whose `event_key` is no later than the current decision event may
enter the window.

## 8. Level Weights And Quantity Semantics

Use L1-L5 with frozen distance-rank weights:

```text
w_l = 1 / l

w = [1.0, 0.5, 0.3333333333, 0.25, 0.2]
```

Missing levels are unavailable, not zero.

An event is detector-eligible only when all five levels on both sides are
present and finite.

For each level:

```text
add_qty:
  max(new_qty - old_qty, 0)

remove_qty:
  max(old_qty - new_qty, 0)
```

`remove_qty` is an observed displayed-depth reduction. It must not be called
an exact cancellation because public depth data cannot reliably separate
cancellation, trade consumption and level migration in every case.

Frozen denominator floors are fitted on the historical normalization role
only:

```text
depth_scale_floor =
  1st percentile of positive weighted L1-L5 side depth

trade_scale_floor =
  10th percentile of positive trailing median 50ms total-trade quantity
```

Both floors are reused unchanged on every later date. Nonfinite or nonpositive
floors fail closed.

## 9. Three Interpretable Pressure Components

All components are computed for both `d=+1` and `d=-1`.

### 9.1 Vulnerable-Side Net Depletion

Let `v(d)` be ask for `d=+1` and bid for `d=-1`.

Let:

```text
Q_v,start =
  sum_l w_l * Q_v,l at the last admissible event no later than t-W
```

Then:

```text
X_dep,d(t) =
  sum_l w_l * (remove_v,l - add_v,l) over (t-W, t]
  ---------------------------------------------------
  max(Q_v,start, depth_scale_floor)
```

Positive values mean that the side vulnerable to direction `d` lost more
displayed depth than it replenished.

### 9.2 Aggressive Trade Pressure

Let:

```text
signed_trade_qty =
  buyer_initiated_qty - seller_initiated_qty

total_trade_qty =
  buyer_initiated_qty + seller_initiated_qty

trade_scale(t) =
  median 50ms total_trade_qty at 20ms calendar checkpoints
  over the causal normalization history [t-60s, t-500ms)
```

Then:

```text
X_trade,d(t) =
  d * signed_trade_qty over (t-W, t]
  ---------------------------------
  max(trade_scale, trade_scale_floor)
```

The aggressor-side convention follows the admitted Binance public trade
contract and must be unit-tested with mirrored fixtures.

### 9.3 Whole-Book Flow Pressure

For each level:

```text
bid_net_l = bid_add_l - bid_remove_l
ask_net_l = ask_add_l - ask_remove_l

total_depth_start =
  sum_l w_l * (Q_bid,l + Q_ask,l)
  at the last admissible event no later than t-W
```

Define:

```text
X_ofi,d(t) =
  d * sum_l w_l * (bid_net_l - ask_net_l)
  ----------------------------------------
  max(total_depth_start, depth_scale_floor)
```

Positive `X_ofi,d` means whole-book displayed flow is oriented in direction
`d`.

### 9.4 Deliberate Exclusions

The primary onset predicate does not include:

- current OBI level;
- OBI reversal history;
- midpoint delta;
- microprice delta;
- spread widening;
- realized volatility;
- future queue refill;
- a learned latent state.

These fields may later enter H0 context or diagnostics, but cannot determine
the A0 primary anchor.

## 10. Causal Robust Normalization

Each component uses a trailing calendar-time robust baseline.

Frozen baseline window:

```text
60 seconds
```

Frozen guard interval:

```text
500ms immediately before the current event
```

For component `k` and direction `d`:

```text
history(t) = [t-60s, t-500ms)

center_k,d(t) = median of 20ms calendar checkpoints in history(t)
scale_k,d(t)  = 1.4826 * MAD of the same checkpoints

Z_k,d(t) =
  (X_k,d(t) - center_k,d(t))
  --------------------------------------
  max(scale_k,d(t), global_scale_floor_k)
```

The 20ms checkpoints are a normalization sampling device, not detector
decision times.

`global_scale_floor_k` is fitted only on the frozen historical normalization
role and then reused unchanged for all later dates. It is the 10th percentile
of positive rolling MAD values for that component.

Requirements:

- at least 30 seconds of valid baseline exposure after every reset;
- no current-window value in its own baseline;
- no future or same-event value in center or scale;
- no per-date refit on blocked validation or replay roles;
- zero or nonfinite scale fails closed.

## 11. Primary Onset Predicate

Frozen component threshold:

```text
z_star = 3.0
```

For each direction:

```text
I_dep,d   = 1[Z_dep,d   >= 3.0]
I_trade,d = 1[Z_trade,d >= 3.0]
I_ofi,d   = 1[Z_ofi,d   >= 3.0]

coherence_count_d =
  I_dep,d + I_trade,d + I_ofi,d

pressure_score_d =
  max(Z_dep,d, 0)
  + max(Z_trade,d, 0)
  + max(Z_ofi,d, 0)
```

The structural onset predicate is:

```text
coherence_count_d >= 2
and pressure_score_d >= 6.0
and X_dep,d > 0
and detector_quality_valid
```

No dwell or future confirmation is required.

Anchor-emission eligibility is separate from the structural predicate:

```text
detector_state == NEUTRAL:
  either direction may emit a start anchor

detector_state == ACTIVE(d):
  direction d cannot emit another anchor
  direction -d may emit an opposite-onset anchor
```

### 11.1 Simultaneous Direction Conflict

If both directions satisfy the predicate at the same event:

```text
score_gap =
  abs(pressure_score_+ - pressure_score_-)
```

Rules:

- if `score_gap >= 0.5`, select the direction with the larger score;
- otherwise classify the event as `direction_ambiguous`;
- an ambiguous event creates no primary anchor.

The `0.5` gap is frozen before target access.

## 12. start_anchor

The primary `start_anchor` is:

```text
start_anchor =
  the first event_key at which the structural onset predicate becomes true
  while the detector is NEUTRAL
```

The stored decision fields are:

```text
onset_detected_local_ts_ns
onset_detected_event_seq
direction_d
Z_dep
Z_trade
Z_ofi
coherence_count
pressure_score
current L1-L5 book state
quality and reset epoch
```

The anchor is not backdated.

In particular:

- do not backdate to the first component threshold crossing;
- do not backdate to the beginning of the 50ms window;
- do not move the anchor to the largest later pressure score;
- do not move the anchor to a later confirmation dwell;
- do not use an offline change-point estimate.

For diagnostics only, record:

```text
precursor_at =
  the earliest causal event in the current active candidate sequence
  at which any one component first crossed 3.0

precursor_to_detection_ms =
  start_anchor - precursor_at
```

`precursor_at` is not an alignment point and cannot replace `start_anchor`.

## 13. end_anchor And Duplicate Suppression

The detector state is:

```text
NEUTRAL
ACTIVE(+1)
ACTIVE(-1)
```

After a primary start anchor in direction `d`, the detector enters
`ACTIVE(d)`.

While active:

- no additional same-direction start anchor is emitted;
- pressure components continue to be computed causally;
- active duration is descriptive support, not a target.

Frozen release condition:

```text
coherence_count_d == 0
and pressure_score_d < 1.5
continuously for 100ms of valid event-time exposure
```

The `end_anchor` is the event at which the 100ms release condition is first
causally known to be complete.

It is not backdated to the beginning of the release interval.

If a valid opposite-direction onset appears while `ACTIVE(d)`:

1. terminate the old active state at the current event;
2. mark old exit reason `opposite_onset`;
3. emit a new start anchor for `-d` at the same current event;
4. preserve one total order through `event_seq`.

Capture end, reset or quality failure closes the active interval as censored.
There is no forced maximum active duration.

`end_anchor` is used only for:

- duplicate suppression;
- active-time occupancy;
- recurrence geometry;
- support diagnostics.

It is not the downstream `n_recovery` outcome.

## 14. Outcome-Blind Control Support

Controls are required to establish common support for a later nested H0/H1
test.

Control candidate times are generated on a causal:

```text
250ms calendar stride
```

A control candidate must:

- be outside an active interval at its decision time;
- fail the structural onset predicate in both directions;
- have complete baseline warmup and valid L1-L5 book state;
- use only information available by its decision time;
- not be excluded because of a future anchor.

For every onset anchor, match an outcome-blind control without reuse using:

1. same research date;
2. same pressure direction;
3. same absolute current OBI bin of width `0.10`;
4. same spread-tick value when support permits, otherwise adjacent value;
5. same total-depth quintile;
6. same trailing activity quintile;
7. same 30-minute time-of-day block;
8. nearest timestamp as the final tie-break.

Matching may relax only in this frozen order:

```text
time-of-day block
activity quintile
depth quintile
adjacent spread value
```

It may never relax:

```text
research date
direction
current OBI bin
quality eligibility
```

No future target, future price or future anchor may participate in matching.

## 15. Frozen Downstream Target Stub

A0 does not materialize these targets. It freezes their meaning so later
stages cannot select favorable outcomes.

At decision time `t0`, record:

```text
m0:
  current midpoint

tick:
  frozen tick size

Q_v,baseline:
  median weighted vulnerable-side depth over [t0-500ms, t0-100ms]
```

The later competing risks are:

### 15.1 n_adverse_continuation

```text
d = +1:
  midpoint first reaches m0 + 1 tick

d = -1:
  midpoint first reaches m0 - 1 tick
```

### 15.2 n_liquidity_recovery

Recovery is first causally complete when all conditions hold:

```text
weighted vulnerable-side depth >= 0.8 * Q_v,baseline
same-direction pressure_score < 1.5
same-direction coherence_count == 0
conditions persist for 100ms
no prior n_adverse_continuation
```

Recovery detection is not backdated.

### 15.3 Censoring And Simultaneity

Right-censor on:

- frozen `tau_max`;
- capture end;
- sequence gap;
- snapshot reset;
- source-quality failure.

If adverse continuation and recovery become identifiable at the same ordered
event, classify the row as interval-ambiguous and do not pointify it.

Candidate `tau_max` values are:

```text
500ms
1000ms
2000ms
5000ms
10000ms
```

A0 selects the largest value with:

```text
overall complete future-coverage geometry >= 0.95
minimum per-date complete coverage >= 0.80
```

This selection uses capture boundaries and quality geometry only. It does not
read which target occurs.

## 16. Frozen H0/H1 Information Contract

The downstream model family remains a low-parameter discrete-time
multinomial competing-risk hazard.

H0 contains current static state:

- current equal-weight L1-L5 OBI;
- current per-level imbalance summary;
- spread;
- total bid/ask depth and concentration;
- current source age and quality;
- trailing unsigned activity;
- trailing volatility ending at decision time;
- frozen time-of-day context.

H0 must not contain a deterministic reconstruction of the onset predicate.

H1 adds the candidate transition-pressure family:

```text
onset_indicator_R
Z_dep,d
Z_trade,d
Z_ofi,d
coherence_count_d
```

The same fields are present for onset and matched-control entries.

H1 therefore tests:

```text
current static state
  versus
current static state + causal transition pressure
```

The primary result cannot be reduced to an unconditional anchor event rate.

A stronger-H0 diagnostic may later add generic signed 50ms book and trade
changes. It cannot rescue a failed primary H1.

## 17. Session Roles

Frozen historical roles:

| Dates | Role | A0 use |
| --- | --- | --- |
| 2026-07-29 | normalization calibration | global scale floors and reconstruction support |
| 2026-07-30, 2026-08-03, 2026-08-04 | historical development | anchor support and control design |
| 2026-08-07, 2026-08-24, 2026-08-25 | blocked historical validation | no threshold or normalization refit |
| 2026-08-26, 2026-08-27 | historical no-refit replay | support replay only |

All dates are historical and previously inspected.

A0 must regenerate the exact capture inventory, sizes, hashes, durations,
roles, reset counts and sequence-gap counts.

No true prospective claim is available.

## 18. Required A0 Outputs

The formal A0 execution must publish:

```text
contracts/
  source_manifest.json
  event_ordering_contract.json
  pressure_component_contract.json
  normalization_contract.json
  onset_state_machine.json
  control_support_contract.json
  downstream_target_stub.json
  H0_H1_contract.json
  gate_contract.json
  session_role_ledger.csv
  outcome_access_ledger.json

support/
  component_support_by_date.csv
  normalization_scale_support.csv
  onset_anchor_ledger.csv
  active_interval_ledger.csv
  anchor_support_by_date.csv
  inter_anchor_distribution.csv
  component_pair_composition.csv
  precursor_detection_delay.csv
  control_candidates.csv
  matched_control_pairs.csv
  control_overlap_by_date.csv
  followup_geometry.csv

reports/
  A0_summary.json

classification.json
run_manifest.json
```

Large raw data and reconstructed caches remain outside Git. Git tracks only
compact contracts, summaries, ledgers when size permits, and exact hashes.

## 19. A0 Gates

### Gate A0-0: Source Closure

Require:

- all admitted raw paths present;
- exact size and SHA closure;
- snapshot/depth continuity;
- zero unhandled sequence gaps;
- deterministic replay;
- all reset boundaries represented.

### Gate A0-1: Zero-Outcome Boundary

Require:

- no future midpoint or best-price field read;
- no target label materialized;
- no future-aligned price plot;
- no H0/H1 fit;
- no new data collection;
- no private API or order activity.

### Gate A0-2: Causal Anchor Support

Require:

```text
minimum total primary anchors:          800
minimum represented research dates:    8
minimum anchors on each admitted date:  25
maximum single-date anchor share:       0.35
anchor rate envelope:                   5 to 300 per hour
minority direction share:               at least 0.20
```

### Gate A0-3: Earliest-Detection Geometry

Require:

```text
no future confirmation used:                   true
p90 precursor_to_detection_ms <=              100ms
median same-direction inter-anchor interval >= 250ms
active-time occupancy <=                       0.60
direction-ambiguous fraction <=                0.10
```

Failing this gate means the detector is still too late, too dense or too
ambiguous to serve as an alignment landmark.

### Gate A0-4: Component Diversity

Require:

- every primary component appears in at least 10% of anchors;
- no single component pair accounts for more than 85% of anchors;
- both directions contain all three possible component pairs;
- global scale floors are finite and positive;
- at least 95% of anchors use non-floor local MAD on at least two components.

This gate prevents a nominal three-channel detector from collapsing into one
unacknowledged trigger.

### Gate A0-5: Control Common Support

Require:

```text
minimum unique matched pairs:                 500
overall onset-to-control common support:      0.90
minimum per-date common support:              0.75
maximum single-date matched-pair share:       0.35
```

### Gate A0-6: Follow-Up Geometry

Require that at least one candidate `tau_max` satisfies:

```text
overall complete coverage >= 0.95
minimum per-date coverage >= 0.80
```

This gate inspects geometry only, not outcomes.

## 20. Frozen A1 Timeliness Stop Gate

A0 cannot evaluate price timeliness because future price access is forbidden.
It freezes the later A1 stop rule:

```text
At precursor_at:
  m_precursor = causal midpoint at precursor_at

pre_detection_adverse =
  d = +1 and midpoint reaches m_precursor + 1 tick before start_anchor
  or
  d = -1 and midpoint reaches m_precursor - 1 tick before start_anchor

pre_detection_adverse_fraction =
  fraction of entries with pre_detection_adverse
```

Require in A1:

```text
pre_detection_adverse_fraction <= 0.30
```

Also require:

```text
median post-anchor identified transition time
  >= 3 * median precursor_to_detection_ms
```

If either condition fails:

```text
classification:
  causal_anchor_too_late_for_target

action:
  stop LIQUIDITY_BREAK_ONSET_V1 before H0/H1 fitting
```

This threshold cannot be changed after A1 outcomes are visible.

## 21. A0 Classifications

Passing classification:

```text
A0_causal_anchor_contract_supported
```

Failure classifications:

```text
A0_source_not_admissible
A0_zero_outcome_boundary_violated
A0_anchor_support_insufficient
A0_anchor_near_continuous
A0_anchor_detection_too_delayed
A0_anchor_direction_ambiguous
A0_component_family_collapsed
A0_control_common_support_insufficient
A0_followup_geometry_insufficient
```

Only the passing classification may authorize A1 target materialization.

## 22. Explicitly Forbidden Rescue Operations

After A0 execution begins, do not:

- change the 50ms primary window;
- change `z_star=3.0`;
- change the two-of-three coherence rule;
- add OBI, midpoint or spread movement to the onset predicate;
- backdate the start anchor;
- add a future confirmation dwell;
- split anchors by favorable component pair;
- tune release rules using target outcomes;
- select controls using future anchors or prices;
- replace event-time detection with offline change-point segmentation;
- use 20ms or 100ms diagnostics to rescue a failed 50ms primary;
- lower support gates because anchors are sparse;
- increase thresholds because anchors are dense;
- describe a later variant as robustness of this version.

Any such change creates:

```text
LIQUIDITY_BREAK_ONSET_V2 or another new hypothesis identifier
```

## 23. Verification Requirements

Formal A0 implementation must include focused tests for:

- equal-timestamp event ordering;
- snapshot bridge and `U/u/pu` continuity;
- mirrored `d=+1` and `d=-1` component signs;
- top-five level rank changes;
- add/remove quantity semantics;
- left-open/right-closed rolling windows;
- baseline guard interval;
- no current value in its own robust normalization;
- onset at the second coherent channel without dwell;
- no anchor backdating;
- simultaneous-direction ambiguity;
- active lock and causal release;
- opposite-onset state transition;
- reset censoring;
- no future-anchor control exclusion;
- no control reuse;
- target-stub non-materialization;
- outcome-access fail-closed behavior;
- deterministic double-build artifact identity.

Required static checks:

```text
focused pytest
Python compile
git diff --check
Markdown fence parity
artifact size/SHA closure
```

## 24. Stage Boundary

The stage chain is:

```text
A0 causal-anchor support and contract execution
  -> A1 target materialization and timeliness stop gate
  -> A2 target variation and common-risk-set closure
  -> A3 nested H0/H1 competing-risk increment test
  -> A4 dependence nulls and historical transport
  -> prospective confirmation under a new collection task
```

A0 passing does not automatically authorize A2 or A3.

The next formal task after this document is accepted may implement and execute
only A0. It may not read downstream outcomes.

## 25. Frozen Constants Summary

| Item | Frozen value |
| --- | --- |
| Hypothesis ID | `LIQUIDITY_BREAK_ONSET_V1` |
| Primary detection mode | raw-message event driven |
| Primary pressure window | `50ms` |
| Diagnostic windows | `20ms`, `100ms` |
| Levels | L1-L5 |
| Level weights | `1/l` |
| Robust baseline | trailing `60s` |
| Baseline guard | `500ms` |
| Normalization checkpoints | `20ms` calendar time |
| Component threshold | `z_star=3.0` |
| Coherence | at least 2 of 3 components |
| Aggregate pressure threshold | `6.0` |
| Direction conflict gap | `0.5` |
| Release score | `<1.5` |
| Release dwell | `100ms` causal |
| Control stride | `250ms` |
| OBI match bin | `0.10` |
| Candidate tau_max | `0.5/1/2/5/10s` |
| A1 pre-detection maximum | `0.30` |
| Replication unit | research date or session |

## 26. Frozen Claim Boundary

This contract asserts only:

```text
LIQUIDITY_BREAK_ONSET_V1 is a new, explicit and causally evaluable
alignment hypothesis.
```

It does not assert:

```text
the detector has passed support gates
the market reaction is predictable
the effect transports across dates
the signal survives execution latency
the signal has maker economic value
```

Those claims remain locked behind the ordered research gates.
