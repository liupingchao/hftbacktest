# SKHYNIX Binance SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1 A0 Plan - 2026-08-28

Date: 2026-08-28

Status:

```text
frozen zero-target A0 design contract
execution not authorized by this document
```

Hypothesis identifier:

```text
SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1
```

Research family:

```text
continuous_flow_excursion
  -> causal recovery
  -> maker_safe_reentry_alignment
```

Methodology dependency:

```text
docs/conditional_risk_research_methodology_kernel_v1.md
```

Predecessor:

```text
LIQUIDITY_BREAK_ONSET_V1
  -> A0_anchor_near_continuous
  -> A1_blocked_by_A0
```

## 1. Decision

Register a new hypothesis.

The failed predecessor treated a two-of-three 50ms pressure crossing as a new
alignment event. Historical execution produced:

```text
340,068 anchors
9,468.1098 anchors/hour
60.26% opposite-onset exits
active-duration p50 176.3133ms
```

This means the crossing detector described a near-continuous order-flow
process with rapid sign changes.

The new route is:

```text
continuous background
  -> novel pressure candidate after a genuine quiet state
  -> causally persistent flow excursion
  -> direction changes remain inside one episode
  -> bilateral recovery
  -> refractory quiet completion
  -> current wide-spread opportunity still exists
  -> safe_reentry_anchor
```

Primary alignment:

```text
safe_reentry_after_flow_excursion
```

It is not:

```text
the first pressure crossing
the maximum-pressure timestamp
the first opposite-direction crossing
an offline episode center
a future-confirmed backdated onset
the first future fill
the first favorable markout
```

## 2. Version Boundary

`SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1` is not:

- a higher threshold for `LIQUIDITY_BREAK_ONSET_V1`;
- a longer release dwell for the old detector;
- a favorable subset of the old 340,068 anchors;
- an A1 rescue using future outcomes;
- proof that spread capture exists;
- a maker strategy.

Load-bearing changes are:

1. pressure crossing becomes a micro-observation, not an event;
2. novelty requires a prior bilateral quiet state;
3. persistence requires cumulative causal exposure;
4. direction flips remain inside one excursion;
5. refractory completion defines episode separation;
6. alignment moves from shock onset to safe-reentry availability;
7. current spread is part of the exposure definition.

Any later change to these semantics creates another hypothesis version.

## 3. Primary Hypothesis

Let `C_t` denote current static market state and `E_t` the causally completed
excursion history.

The hypothesis is:

> After a genuinely novel and persistent order-flow excursion has ended,
> bilateral liquidity has recovered, the market has remained quiet through a
> refractory period, and a spread opportunity is still currently visible,
> the recent excursion history may change the competing-risk law of passive
> quote contact, adverse move-through and spread collapse beyond the
> information in the current static book state.

Conditional form:

```text
H0:
  P(T,J | current safe-looking state C_t)

H1:
  P(T,J | C_t, completed excursion history E_t)
```

The alignment claim is therefore:

```text
same current spread/depth/OBI context
  +
different recent excursion/recovery path
  ->
possibly different maker transition risk
```

## 4. A0 Purpose And Authority

A0 is a zero-target support stage.

A0 may:

- reconstruct admitted Binance public data;
- reuse the predecessor pressure-component projection as micro-observations;
- materialize causal background, candidate, excursion, recovery and
  refractory states;
- materialize safe-reentry anchors;
- construct outcome-blind controls;
- measure episode compression, cadence, duration and cross-date support;
- inspect current spread, depth, OBI, activity and source quality;
- select follow-up horizon using boundary geometry only;
- freeze downstream public-contact, queue-bound and adverse targets without
  materializing them.

A0 may not:

- read future midpoint or best-price transitions;
- inspect future trade contact at a hypothetical quote;
- simulate queue fill;
- calculate future spread collapse;
- inspect markout, PnL, fees or inventory outcome;
- fit H0 or H1;
- choose constants using future event rates or model loss;
- access private APIs, orders or fills;
- collect new data.

Passing A0 means only:

```text
the excursion and safe-reentry exposure are causal,
semantically distinct from continuous background,
and historically supported
```

It does not mean:

```text
safe reentry predicts fill
safe reentry avoids adverse selection
spread can be captured
the strategy is actionable or profitable
```

## 5. Historical Evidence Boundary

Existing admitted data:

```text
29 captures
9 research dates
35.9172008142 hours
2026-07-29 through 2026-08-27
```

All dates precede 2026-08-28 and have already been inspected in predecessor
research.

Session roles remain:

| Dates | Role |
| --- | --- |
| 2026-07-29 | normalization and state calibration |
| 2026-07-30, 2026-08-03, 2026-08-04 | historical method development |
| 2026-08-07, 2026-08-24, 2026-08-25 | blocked historical validation |
| 2026-08-26, 2026-08-27 | historical no-refit replay |

No true prospective claim is available.

The current spread support fact from predecessor A0 is:

```text
onset-anchor spread p50:                 1 tick
outcome-blind control spread p50:       1 tick
onset anchors with spread > 1.5 ticks:  20.81%
controls with spread > 1.5 ticks:       21.53%
```

Therefore the old detector did not identify spread widening. The new exposure
must explicitly require a currently visible spread opportunity.

## 6. Source And Event Ordering

Primary source:

```text
existing admitted Binance public SKHYNIXUSDT captures
```

Allowed channels:

```text
snapshot
depthUpdate
trade
bookTicker for current quality consistency only
```

Every raw row receives:

```text
event_key = (local_receive_ts_ns,event_seq_in_file)
```

Rules:

1. Preserve file order for equal receive timestamps.
2. Apply the current message before updating causal state.
3. Do not use a later exchange timestamp as earlier local information.
4. Fail closed on snapshot/depth continuity gaps.
5. Do not carry state across capture, snapshot, sequence-gap or quality
   boundaries.
6. Bind every state transition and anchor to exact `event_key`.

## 7. Micro-Pressure Projection

The predecessor projection is reused only as a micro-observation layer.

Primary window:

```text
50ms
```

Levels and weights:

```text
L1-L5
w_l = 1/l
```

Directional components:

```text
vulnerable-side net depletion
aggressive trade pressure
whole-book flow pressure
```

Robust normalization:

```text
20ms completed checkpoints
60s trailing history
500ms guard
rolling median
IQR / 1.349
calibration-only global scale floors
```

Micro structural predicate:

```text
component threshold: z >= 3.0
coherence: at least 2 of 3
aggregate positive score >= 6.0
vulnerable-side net depletion > 0
```

Important:

```text
micro predicate true
  !=
new excursion
```

It is only one observation consumed by the excursion state machine.

## 8. Frozen Time Constants

Primary A0 constants:

| Item | Value |
| --- | ---: |
| Causal checkpoint | `20ms` |
| Micro-pressure window | `50ms` |
| Novelty pre-quiet | `1000ms` |
| Persistence qualification window | `250ms` |
| Required qualifying exposure | `100ms` |
| Candidate timeout | `250ms` |
| Bilateral recovery fraction | `0.80` |
| Recovery pressure score | `<1.5` both directions |
| Refractory quiet duration | `1000ms` |
| Primary spread opportunity | `>=2 ticks` |
| Maximum absolute current OBI | `0.50` |
| Control stride | `250ms` |
| Control recent-excursion exclusion | `5000ms` |

Rationale uses only predecessor support geometry:

```text
active-duration p50:            176ms
active-duration p90:            406ms
same-direction inter-anchor p50: 475ms
same-direction inter-anchor p90: 1180ms
```

`1000ms` lies above the predecessor active-duration p90 and near the
inter-anchor p90. It is intended to require genuine separation rather than
reduce event count by raising the pressure threshold.

Frozen diagnostics:

```text
refractory 500ms
refractory 2000ms
persistence exposure 50ms
persistence exposure 200ms
```

Diagnostics cannot replace or rescue the primary after outcomes are visible.

## 9. Novelty Semantics

Novelty means the excursion candidate begins only from a bilateral
`BACKGROUND_READY` state.

`BACKGROUND_READY` requires for the full preceding `1000ms` of valid
checkpoint exposure:

```text
micro structural predicate false in both directions
pressure score < 1.5 in both directions
no active candidate
no active excursion
no recovery or refractory state
valid uncrossed L1-L5 book
normalization available
```

The first later checkpoint satisfying a directional micro predicate becomes:

```text
candidate_at
```

Properties:

- a repeated crossing inside an existing process is not novel;
- an opposite crossing during an episode is not novel;
- state reset or missing exposure invalidates the 1000ms quiet history;
- novelty does not inspect future persistence;
- `candidate_at` is descriptive and is not the downstream decision anchor.

Candidate direction:

```text
d0 = direction with larger pressure score
```

If the score gap is below the inherited conflict gap:

```text
candidate_status = direction_ambiguous
candidate rejected
```

## 10. Persistence Semantics

Persistence removes single-message and short-lived threshold noise.

After `candidate_at`, open a causal `250ms` qualification window.

For the initial candidate direction `d0`, accumulate:

```text
qualifying_exposure_ms
```

Only valid 20ms checkpoints satisfying the directional micro structural
predicate contribute.

Confirmation requires:

```text
qualifying_exposure_ms >= 100ms
within 250ms of candidate_at
```

The excursion starts at:

```text
excursion_confirmed_at =
  the checkpoint where 100ms qualifying exposure first becomes known
```

It is not backdated to `candidate_at`.

If confirmation does not occur within 250ms:

```text
candidate_status = transient_rejected
```

If an opposite-direction structural crossing becomes dominant before
confirmation:

```text
candidate_status = pre_confirmation_direction_switch
candidate rejected
```

This does not open an opposite candidate. The detector must rebuild the full
`1000ms` bilateral novelty history before another candidate can begin.

The detector may return to `BACKGROUND_READY` only if the complete novelty
quiet condition is again satisfied.

No future midpoint, spread transition or trade contact participates in
confirmation.

## 11. Excursion State

At `excursion_confirmed_at`, enter:

```text
EXCURSION_ACTIVE
```

Record:

- initial direction;
- current dominant direction;
- peak directional pressure scores;
- integrated qualifying exposure by direction;
- micro-crossing count;
- direction-switch count;
- maximum depth deficit by side;
- minimum and maximum current spread;
- current OBI path summaries;
- duration;
- reset/censor reason.

While active:

- no new excursion may begin;
- same-direction crossings update the same episode;
- opposite-direction crossings update direction-switch count;
- an opposite crossing does not close the episode;
- an opposite crossing does not emit another anchor;
- all state remains within the same reset/quality segment.

This is the primary refractory strengthening over the predecessor.

## 12. Pre-Excursion Baseline

At `candidate_at`, freeze a causal background baseline from:

```text
[candidate_at-1000ms,candidate_at-100ms)
```

Record:

```text
Q_bid,baseline
Q_ask,baseline
spread_baseline
OBI_baseline
activity_baseline
```

Depth uses weighted L1-L5 quantities.

The final 100ms is excluded so the developing candidate cannot contaminate
its own recovery reference.

If the baseline is incomplete or crosses a reset:

```text
candidate rejected
```

## 13. Recovery Candidate

An active excursion enters `RECOVERY_CANDIDATE` when current directional
pressure has become bilaterally quiet:

```text
micro structural predicate false in both directions
pressure score < 1.5 in both directions
valid uncrossed L1-L5 book
```

While in `RECOVERY_CANDIDATE`, depth recovery is evaluated causally. Entry to
`REFRACTORY` requires:

```text
weighted bid depth >= 0.8 * Q_bid,baseline
weighted ask depth >= 0.8 * Q_ask,baseline
```

Recovery is bilateral. Restoring only the previously vulnerable side is
insufficient, and no refractory exposure accumulates before both sides meet
the recovery condition.

If either direction becomes structurally active again:

```text
return to EXCURSION_ACTIVE
retain the same episode identity
increment recovery_reset_count
```

Recovery does not create the safe-reentry anchor.

## 14. Refractory Semantics

When bilateral depth recovery first becomes known while pressure remains
quiet, enter:

```text
REFRACTORY
```

Require `1000ms` of continuous valid checkpoint exposure during which:

```text
micro predicate remains false in both directions
pressure score remains <1.5 in both directions
both side depths remain >=0.8 baseline
book remains valid and uncrossed
```

Any renewed pressure:

```text
returns to EXCURSION_ACTIVE
does not create a new episode
resets the refractory clock
```

The refractory period is therefore an episode-merging rule, not a sample
thinning rule.

At the checkpoint where the full 1000ms condition first becomes causally
known, define:

```text
refractory_completed_at
```

It is not backdated.

## 15. Safe-Reentry Anchor

At `refractory_completed_at`, evaluate only current observable state.

Primary opportunity conditions:

```text
current spread >= 2 ticks
abs(current equal-weight L1-L5 OBI) <= 0.50
weighted bid depth >= 0.8 * Q_bid,baseline
weighted ask depth >= 0.8 * Q_ask,baseline
source quality valid
```

If all hold:

```text
safe_reentry_anchor = refractory_completed_at
episode_status = safe_reentry_available
```

If recovery is complete but spread is below 2 ticks:

```text
episode_status = recovered_without_wide_spread
no primary anchor
```

If OBI remains extreme:

```text
episode_status = recovered_but_imbalanced
no primary anchor
```

The detector does not wait for a later favorable spread. Waiting would create
a different exposure and could mix a new market process into the old episode.

After any terminal episode status:

```text
return to BACKGROUND_BUILDING
```

A new excursion requires a new full 1000ms novelty pre-quiet history.

## 16. State Machine

Canonical states:

```text
BACKGROUND_BUILDING
BACKGROUND_READY
EXCURSION_CANDIDATE
EXCURSION_ACTIVE
RECOVERY_CANDIDATE
REFRACTORY
TERMINAL
```

Canonical path:

```text
BACKGROUND_BUILDING
  -> BACKGROUND_READY
  -> EXCURSION_CANDIDATE
  -> EXCURSION_ACTIVE
  -> RECOVERY_CANDIDATE
  -> REFRACTORY
  -> TERMINAL
  -> BACKGROUND_BUILDING
```

Allowed loops:

```text
EXCURSION_ACTIVE
  -> EXCURSION_ACTIVE through direction switches

RECOVERY_CANDIDATE or REFRACTORY
  -> EXCURSION_ACTIVE through renewed pressure
```

Forbidden:

```text
opposite crossing -> new episode
future dwell -> backdated candidate
future spread -> relocated safe-reentry anchor
reset boundary -> continued episode
```

## 17. Episode Identity And Compression

Every candidate receives a deterministic candidate ID.

Every confirmed excursion receives:

```text
episode_id =
  SHA256(
    hypothesis_id,
    capture_id,
    segment_id,
    candidate_event_key,
    confirmation_event_key
  )
```

All micro crossings between confirmation and terminal state map to the same
episode ID.

Required compression metrics:

```text
raw micro-crossing count
confirmed excursion count
micro-crossings per excursion
direction switches per excursion
transient rejection count
refractory reset count
safe-reentry anchor count
```

The purpose is to demonstrate that a near-continuous crossing process has
been converted into separated, interpretable excursions.

## 18. Outcome-Blind Controls

Controls are quiet, currently safe-looking wide-spread states without a
recent completed excursion.

Generate candidates on a:

```text
250ms calendar stride
```

A control candidate must:

- have valid current L1-L5 state;
- have complete current normalization;
- satisfy current spread `>=2 ticks`;
- satisfy `abs(OBI)<=0.50`;
- satisfy bilateral depth eligibility;
- have no active candidate/excursion/recovery/refractory;
- have no confirmed excursion or safe-reentry anchor in the preceding
  `5000ms`;
- use no future anchor or outcome exclusion.

Match every safe-reentry anchor without reuse on:

1. same research date;
2. same current spread-tick value, adjacent value only as final relaxation;
3. same absolute OBI bin of width `0.10`;
4. same bid-depth quintile;
5. same ask-depth quintile;
6. same activity quintile;
7. same 30-minute time block;
8. nearest timestamp.

The control meaning is:

```text
same current safe-looking context
without the recent completed excursion path
```

No future contact, adverse move, spread collapse or fill may participate.

## 19. Current-State H0 And Excursion-History H1

Frozen downstream information families:

```text
H0:
  current spread ticks
  current equal-weight L1-L5 OBI
  current bid and ask weighted depth
  current activity
  current time block
  current source-quality state

H1 adds:
  recent-excursion indicator
  excursion duration
  initial direction
  peak pressure by direction
  integrated pressure by direction
  direction-switch count
  recovery-reset count
  maximum bid and ask depth deficits
  recovery duration
  refractory reset history
```

A0 does not fit either model.

This nested comparison is essential. A later positive result must show that
the excursion/recovery path adds information beyond the current wide-spread
book state.

## 20. Frozen Downstream Maker Target Stub

Binance public data cannot identify a real own-order fill.

Later stages must keep separate:

```text
public quote contact
public price move-through
conservative queue-depletion fill bound
real own-order fill
realized maker PnL
```

Candidate competing risks after safe-reentry:

```text
n_contact:
  public aggressive trade first contacts the hypothetical passive quote

n_adverse:
  public midpoint or best quote first moves through the quote

n_spread_collapse:
  spread first falls below 2 ticks before contact

n_timeout:
  none is identified before tau_max
```

A separate conservative queue layer may later define:

```text
n_queue_fill_bound
```

using frozen ahead quantity, cancels, trades and no optimistic hidden-liquidity
assumption.

Public contact or a queue bound must never be called a real fill.

## 21. Follow-Up Geometry

Candidate horizons:

```text
500ms
1000ms
2000ms
5000ms
10000ms
```

A0 may select the largest horizon satisfying:

```text
overall complete boundary/quality coverage >= 0.95
minimum per-date coverage >= 0.80
```

This selection uses only capture, segment and source-quality geometry.

It may not inspect contact, adverse movement, spread collapse or queue fill.

## 22. Required A0 Outputs

Formal execution must publish:

```text
contracts/
  source_manifest.json
  event_ordering_contract.json
  micro_pressure_contract.json
  normalization_contract.json
  novelty_contract.json
  persistence_contract.json
  excursion_state_machine.json
  recovery_refractory_contract.json
  safe_reentry_contract.json
  control_support_contract.json
  downstream_target_stub.json
  H0_H1_contract.json
  gate_contract.json
  session_role_ledger.csv
  outcome_access_ledger.json

support/
  micro_crossing_support_by_date.csv
  candidate_support_by_date.csv
  transient_rejection_support.csv
  excursion_ledger.csv
  excursion_support_by_date.csv
  excursion_duration_distribution.csv
  direction_switch_composition.csv
  recovery_reset_composition.csv
  refractory_completion_support.csv
  terminal_episode_composition.csv
  safe_reentry_anchor_ledger.csv
  safe_reentry_support_by_date.csv
  safe_reentry_spread_distribution.csv
  crossing_to_episode_compression.csv
  control_candidates.csv
  matched_control_pairs.csv
  control_overlap_by_date.csv
  followup_geometry.csv

reports/
  A0_summary.json

classification.json
run_manifest.json
```

Large event ledgers and caches remain outside Git. Git tracks compact
contracts, summaries, support tables and exact hashes.

## 23. A0 Gates

### Gate A0-0: Source Closure

Require:

- 29 admitted captures present;
- exact size and SHA closure;
- zero unhandled depth sequence gaps;
- deterministic replay;
- all reset and quality boundaries represented.

### Gate A0-1: Zero-Outcome Boundary

Require:

```text
future midpoint fields read: []
future best-price fields read: []
future contact fields read: []
queue-fill targets materialized: false
markout/PnL fields read: []
H0/H1 fitted: false
new collection: false
private/order access: false
```

### Gate A0-2: Normalization Support

Require:

- calibration-only denominator and global scale floors;
- finite positive scales;
- no per-date refit after calibration;
- at least 95% of confirmed excursions and safe-reentry anchors have
  non-floor local scale on at least two directional components;
- zero current checkpoint self-inclusion;
- zero equal-timestamp future-message inclusion.

### Gate A0-3: Excursion Support

Require:

```text
minimum confirmed excursions:          300
minimum represented research dates:    8
minimum excursions per represented date: 15
excursion rate envelope:                2 to 100 per hour
maximum single-date excursion share:    0.35
minority initial-direction share:       0.20
```

### Gate A0-4: Novelty, Persistence And Compression

Require:

```text
confirmed excursions violating 1000ms pre-quiet: 0
confirmed excursions with <100ms qualifying exposure: 0
backdated confirmations: 0
overlapping episode IDs: 0
new episodes emitted during active/refractory: 0
raw crossing / confirmed excursion ratio >= 5
median inter-excursion interval >= 2000ms
```

Also require both directions and all three component pairs to appear across
confirmed excursions.

### Gate A0-5: Recovery And Safe-Reentry Support

Require:

```text
minimum primary safe-reentry anchors:       200
minimum represented research dates:         8
minimum safe-reentry anchors per date:      10
safe-reentry rate envelope:                 1 to 50 per hour
maximum single-date anchor share:           0.35
spread >=2 ticks at every primary anchor:   true
abs(OBI)<=0.50 at every primary anchor:     true
bilateral depth recovery at every anchor:   true
```

Report but do not target-select:

```text
recovered_without_wide_spread
recovered_but_imbalanced
reset_or_quality_censored
never_recovered
```

### Gate A0-6: Control Common Support

Require:

```text
minimum unique matched pairs:                 200
overall safe-reentry-to-control support:      0.90
minimum per-date common support:              0.75
maximum single-date matched-pair share:       0.35
control reuse:                                0
```

### Gate A0-7: Follow-Up Geometry

Require at least one horizon with:

```text
overall complete coverage >= 0.95
minimum per-date complete coverage >= 0.80
```

## 24. A0 Classifications

Passing:

```text
A0_safe_reentry_contract_supported
```

Failures:

```text
A0_source_not_admissible
A0_zero_outcome_boundary_violated
A0_normalization_support_failed
A0_excursion_support_insufficient
A0_excursion_still_near_continuous
A0_novelty_contract_failed
A0_persistence_contract_failed
A0_refractory_contract_failed
A0_safe_reentry_support_insufficient
A0_safe_reentry_current_state_invalid
A0_control_common_support_insufficient
A0_followup_geometry_insufficient
```

Only:

```text
A0_safe_reentry_contract_supported
```

may authorize A1 target-support work.

## 25. Verification Requirements

Formal implementation must include focused tests for:

- equal-timestamp event ordering;
- snapshot bridge and U/u/pu continuity;
- mirrored pressure components;
- no current value in normalization;
- no future equal-timestamp checkpoint inclusion;
- full 1000ms novelty pre-quiet;
- reset invalidation of novelty history;
- transient candidate rejection;
- pre-confirmation direction-switch rejection;
- causal 100ms persistence confirmation;
- no confirmation backdating;
- opposite-direction crossing retained in one episode;
- multiple direction switches under one episode ID;
- bilateral depth recovery;
- refractory reset on renewed pressure;
- causal 1000ms refractory completion;
- recovered-without-wide-spread terminal state;
- recovered-but-imbalanced terminal state;
- exact safe-reentry anchor;
- no future-anchor control exclusion;
- no control reuse;
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

## 26. Explicitly Forbidden Rescue

After A0 execution begins, do not:

- raise the micro z threshold to lower the episode rate;
- shorten novelty pre-quiet because support is sparse;
- reduce persistence exposure because candidates are rejected;
- shorten refractory because safe-reentry anchors are sparse;
- wait for a future wider spread after refractory completion;
- change the OBI threshold using future outcomes;
- drop episodes with many direction switches;
- select favorable dates, directions or component pairs;
- use public contact as a real fill;
- inspect future return, fill, markout or PnL;
- fit H0/H1;
- describe a changed tuple as the same hypothesis version.

Any such change creates:

```text
SAFE_REENTRY_AFTER_FLOW_EXCURSION_V2
or another new hypothesis identifier
```

## 27. Stage Boundary

Ordered chain:

```text
A0 excursion and safe-reentry support
  -> A1 target observability and public-contact support
  -> A2 conservative queue-bound and competing-risk materialization
  -> A3 nested H0/H1 incremental-risk test
  -> A4 dependence nulls and historical transport
  -> A5 maker economics under explicit queue/fee/inventory assumptions
  -> prospective confirmation on future dates
```

Later stages cannot rescue a failed A0.

## 28. Frozen Claim Boundary

This document asserts only:

```text
SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1 is a new,
causal and outcome-blind research hypothesis
with explicit novelty, persistence and refractory semantics.
```

It does not assert:

```text
the A0 detector has been implemented
the state machine has historical support
safe-reentry anchors are frequent enough
recent excursion history adds information over current state
public contact implies fill
maker spread can be captured
the strategy is actionable or profitable
```

Those claims remain locked behind the ordered gates.
