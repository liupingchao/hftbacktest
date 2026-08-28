# SKHYNIX Binance Track A: Continuous Background, OBI Reversal Alignment And Competing-Risk H0/H1 Test - 2026-08-28

Date: 2026-08-28

Revision: 2

Status: review draft; design-only research contract; not execution authority

Primary hypothesis identifier:

```text
OBI_REVERSAL_V1
```

Research family:

```text
continuous_background_interpretable_m_state_competing_risk_transition
```

Methodology dependency:

```text
docs/conditional_risk_research_methodology_kernel_v1.md
```

Revision history:

```text
revision 1:
  generic withdrawal + flow persistence + replenishment deficit M-state

revision 2:
  OBI reversal becomes the first and only primary hypothesis
  generic composite M-state becomes a separately versioned successor
```

Authority boundary:

- This document designs one Binance public-data structural transition study.
- It grants no new collection, private endpoint, order, cancel, strategy,
  deployment, live-capital, maker economics, or Track B authority.
- Every execution stage requires a formal workflow task and frozen inputs.

## 1. Decision

Use the prior empirical experience:

> After a confirmed order-book-imbalance reversal, short-term price direction
> appears to follow a stable conditional probability distribution.

as the source of one interpretable and falsifiable hypothesis.

The primary route is:

```text
continuous market background
  -> OBI enters one directional state
  -> OBI reverses and confirms the opposite state
  -> align at the causal confirmation timestamp
  -> model first price-direction transition
  -> compare H0 snapshot state with H1 snapshot + reversal path
```

The study does not begin by segmenting arbitrary time-series fragments,
running DTW, or clustering high-dimensional paths.

## 2. Why OBI Reversal Is A Suitable Primary M-State

OBI reversal has:

- a direct order-book interpretation;
- a deterministic causal state machine;
- an observable alignment point;
- a natural mirrored side orientation;
- a specific future transition target;
- a low-parameter H0/H1 comparison;
- an online filter implementation if the hypothesis survives.

The important claim is not:

```text
current OBI predicts price
```

It is:

```text
conditional on the same current OBI and market context,
arriving there through an OBI reversal changes the future
price-direction transition law
```

This is a path-dependence hypothesis.

## 3. Prior Experience Contract

The user's experience may determine:

- the choice of `OBI_REVERSAL_V1`;
- the expected effect direction;
- the state-transition interpretation;
- the initial low-capacity model family.

It is not counted as statistical evidence.

If that experience was formed from any date in the existing SKHYNIX dataset:

- that date remains historically consumed;
- it may be used for discovery or calibration only;
- it cannot become a fresh validation or prospective holdout.

No remembered threshold, horizon or favorable event rate is imported unless
it is written into the A0 contract before target access.

## 4. Primary Research Question

Track A asks:

> Among comparable directional OBI entries with the same current book state,
> does a causally confirmed OBI reversal change the probability and timing of
> the next pressure-oriented price move beyond a context-only snapshot model?

Let:

```text
R_i = 1  when landmark i is a confirmed OBI reversal entry
R_i = 0  when landmark i is a matched non-reversal OBI entry

T_i = elapsed time to first admitted directional price transition
J_i = follow-new-OBI direction or fail/opposite direction
```

The primary claim concerns:

```text
P(T_i, J_i | current state, R_i)
```

## 5. H0/H1 Contract

The frozen nested models are:

```text
H0:
  P(T, J | current OBI snapshot, ordinary observable context)

H1:
  P(T, J | current OBI snapshot, ordinary context,
           OBI reversal path indicator)
```

Primary hypothesis:

> Adding the frozen reversal-path indicator produces a material,
> uncertainty-bounded and cross-session-stable improvement in held-out
> competing-risk prediction after current OBI is controlled.

This distinguishes:

```text
state effect:
  price responds to current OBI level

path effect:
  price responds differently because OBI arrived at that level
  through a reversal
```

H1 fails when it does not materially improve H0, even if reversal events show
an attractive unconditional win rate.

## 6. Meaning Of Pattern

The pattern is:

```text
interpretable conditional path-dependent transition law
```

not:

```text
rigid geometric template
```

An accepted result may have variable paths and durations. The stable object
is the cause-specific cumulative incidence:

```text
F_follow(u | R, C)
F_fail(u   | R, C)
```

where `C` contains the current OBI snapshot and frozen H0 context.

## 7. Inherited Evidence

The accepted earlier classification remains:

```text
continuous_state_no_discrete_phase_support
```

It rejects a closed `N -> S -> P -> R -> N` phase-cycle interpretation. It
does not reject a specific OBI path-dependence hypothesis.

The continuous background remains the baseline environment. Old HSMM states,
phase labels, medoids, transition matrices and duration prototypes cannot
initialize or tune `OBI_REVERSAL_V1`.

## 8. Research Tuple

Before target access, freeze:

```text
source:
  admitted Binance public depth and trade data

instrument:
  SKHYNIX research symbol under the existing source contract

decision timestamp:
  reversal_detected_at or matched entry_detected_at

information filtration:
  public observations available by decision timestamp

primary M-state:
  confirmed OBI reversal path indicator

control:
  current-OBI-matched non-reversal directional entry

target:
  first pressure-oriented price move after decision time

target causes:
  n_follow, n_fail

follow-up:
  support-selected elapsed-time range

censoring:
  horizon, capture, gap, reconnect, reset, quality and ambiguity

model comparison:
  H0 snapshot versus H0 plus reversal path H1

replication unit:
  session or research date

claim boundary:
  short-horizon structural price-direction transition only
```

Changing any load-bearing tuple member creates a new hypothesis version.

## 9. Information And Outcome Boundary

### 9.1 Allowed decision-time information

At decision time `t`, H0 and H1 may use:

- current and trailing BBO;
- current and trailing L1-L5 depth;
- public depth updates and trades;
- causal OBI history ending by `t`;
- spread, midpoint and microprice ending by `t`;
- causal add, cancel, depletion and replenishment;
- signed OFI and public-trade flow ending by `t`;
- trailing movement, activity and volatility ending by `t`;
- source age, sequence continuity, reconnect epoch and quality masks;
- frozen calendar context.

All preprocessing is fitted on calibration or training-role data only.

### 9.2 Permitted future target

After A0 freezes the tuple, the target stage may inspect future public quotes
only to determine:

```text
first price-transition type J
transition time T
censoring status
```

Future target values never enter the feature vector.

### 9.3 Forbidden outcomes

Track A may not use:

```text
post-transition return magnitude
future markout beyond the first-passage target
quote contact
own-order fill
spread capture
fees or rebates
inventory
PnL
optimal quote distance or size
maker action labels
```

Track A tests structural directional transition, not economic value.

## 10. Data Roles

Recommended immutable historical roles:

| Dates | Role | Permitted use |
|---|---|---|
| 2026-07-29 | reconstruction/normalization calibration | support only |
| 2026-07-30, 2026-08-03, 2026-08-04 | historical development | state and estimator development |
| 2026-08-07, 2026-08-24, 2026-08-25 | blocked historical validation | frozen model comparison |
| 2026-08-26, 2026-08-27 | historical no-refit replay | final historical replay |
| newly collected sessions | prospective validation/final holdout | required for final confirmation |

All existing dates have been inspected by prior research and are not fresh
prospective evidence.

The exact inventory and hashes must be regenerated in A0.

## 11. Stage Chain

```text
A0 support and OBI_REVERSAL_V1 tuple freeze
  -> A1 OBI reconstruction and causal reversal state machine
  -> A2 common risk set and first-passage target materialization
  -> A3 competing-risk H0/H1 path-dependence test
  -> A4 dependence nulls, transport and online actionability
  -> A5 prospective confirmation and primary classification
```

No later stage may redefine OBI, reversal, controls, target barriers or
follow-up after favorable outcomes become visible.

## 12. A0: Zero-Target Support Stage

A0 has zero target access and fits no transition model.

It must publish:

- exact source inventory and SHA identities;
- timestamp and deterministic reconstruction contract;
- accepted causal grid;
- L1-L5 depth availability and freshness;
- gap, reconnect, reset and censoring geometry;
- standard OBI formula and level weights;
- OBI numerical support and missing-level rules;
- reversal state-machine thresholds, dwell and hysteresis;
- `cross_at` and `detected_at` timestamp semantics;
- non-reversal control-entry rules;
- H0 context family;
- first-passage quote convention and barrier;
- maximum follow-up selection rule;
- hazard elapsed-time basis and parameter budget;
- proper-score and materiality gates;
- dependence-preserving null family;
- session-role and outcome-access ledgers.

A0 may inspect:

- cadence and freshness;
- depth support;
- OBI marginal distribution;
- complete future-coverage geometry;
- calendar blocks and censoring geometry;
- numerical state-machine behavior without target joins.

A0 may not inspect:

- follow/fail event rates;
- reversal-conditioned price direction;
- favorable horizons;
- H0/H1 loss;
- model coefficients;
- price-transition plots aligned to reversal.

## 13. Causal Reconstruction And Continuous Background

Primary reconstruction:

```text
100ms causal grid
```

Coarser views may be used only under A0-frozen causal aggregation:

```text
200ms
500ms
1s
2s
5s
```

These are feature scales, not target horizons.

The continuous background model describes ordinary evolution:

```text
B_t = E[Z_t | Z_(<=t-1), quality_(<=t)]
```

Primary background:

```text
low-parameter diagonal robust AR or grouped AR
```

It supplies:

- ordinary persistence control;
- causal scale normalization;
- current context for H0;
- OOD and quality diagnostics.

It does not create event boundaries or unsupervised motifs.

## 14. Standard OBI Definition

For accepted levels `l = 1..L`, with primary `L=5`:

```text
OBI_t =
  sum_l w_l * (Q_bid,t,l - Q_ask,t,l)
  /
  sum_l w_l * (Q_bid,t,l + Q_ask,t,l)
```

where:

- `Q_bid,t,l` and `Q_ask,t,l` are causally reconstructed displayed quantities;
- `w_l` is frozen before target access;
- the denominator must exceed the frozen minimum support;
- missing required levels produce unavailable/OOD, not silent zero fill.

Primary weights:

```text
equal level weights
```

Permitted A0-frozen robustness:

- L1-only OBI;
- L1-L3 aggregate;
- distance-decayed L1-L5;
- per-level log-depth difference vector already present in the repository.

Robustness variants cannot rescue a failed primary equal-weight L1-L5 result.

## 15. OBI Reversal State Machine

The reversal is a causal transition between interpretable OBI bands.

For an old direction `s_old in {-1,+1}`:

### 15.1 Pre-state

Require:

```text
s_old * OBI_t >= h_pre
```

for at least frozen dwell `d_pre`.

This establishes that the book occupied a meaningful old directional state,
not merely crossed zero because of noise.

### 15.2 Crossing

`reversal_cross_at` is the first timestamp after the accepted pre-state where:

```text
s_old * OBI_t <= h_neutral_exit
```

and the path proceeds toward the opposite band.

Crossing is a retrospective structural anchor, not the online decision time.

### 15.3 New-state confirmation

Let:

```text
s_new = -s_old
```

Require:

```text
s_new * OBI_t >= h_post
```

for frozen confirmation dwell `d_confirm`.

The timestamp at which confirmation becomes causally available is:

```text
reversal_detected_at
```

### 15.4 Quality and ambiguity

Reject or censor reversal candidates with:

- gap or reconnect inside the pre-state/cross/confirmation path;
- stale or missing required levels;
- ambiguous side orientation;
- denominator below OBI support;
- repeated neutral chatter exceeding the frozen state-machine rule;
- target transition already making the causal interpretation unsupported.

Thresholds and dwell values are frozen without price-target access.

## 16. Alignment Contract

Primary alignment:

```text
t0 = reversal_detected_at
```

Diagnostic timestamp:

```text
reversal_cross_at
```

The system may report:

```text
detection_delay =
  reversal_detected_at - reversal_cross_at
```

It may not backdate the online decision to `reversal_cross_at`.

Price movement between `cross_at` and `detected_at` is:

- recorded as `pre_detection_transition`;
- included in structural/actionability diagnostics;
- excluded from claims that the signal was causally capturable before that
  movement.

The primary competing-risk clock starts at `detected_at`.

## 17. Common Risk Set And Controls

To distinguish reversal history from current OBI level, construct a common
directional-entry risk set.

Every admitted landmark enters a frozen new-direction OBI band and has:

```text
entry_detected_at
new direction s_new
current OBI
spread/depth/activity context
```

Classify the history:

```text
R_i = 1:
  the entry was preceded by an accepted opposite OBI pre-state
  and completed the frozen reversal path

R_i = 0:
  the entry reached the same current OBI band without an accepted
  opposite pre-state in the frozen lookback
```

Controls must share:

- the same primary OBI formula;
- the same new-direction band;
- the same confirmation semantics;
- the same quality and censoring rules;
- comparable current OBI support.

H0 controls remaining snapshot differences statistically. Optional matching
is diagnostic and must use only decision-time context.

This comparison asks:

```text
same current OBI region
different arrival path
```

## 18. Primary M-State

The primary interpretable M-state is:

```text
M_i = OBI reversal history indicator R_i
```

No high-dimensional embedding is required.

Primary H1 adds only:

```text
R_i
```

Secondary path descriptors may be reported after the primary model is frozen:

- old-state OBI strength;
- old-state dwell;
- reversal amplitude;
- crossing speed;
- L1-L5 sign coherence;
- confirmation strength;
- confirmation dwell;
- pre-detection price transition.

These descriptors cannot rescue a failed primary reversal-indicator test.

## 19. H0 Snapshot Model

H0 must make current-state OBI prediction difficult to confuse with reversal
path dependence.

The frozen low-parameter H0 may include:

- current aggregate OBI;
- current per-level imbalance summary;
- spread;
- total bid/ask depth and concentration;
- contemporaneous OFI/book-flow pressure;
- contemporaneous signed trade flow;
- trailing price movement ending by decision time;
- trailing activity and volatility;
- time-of-day basis;
- source age and quality state.

H0 must include current OBI.

H0 must not include a deterministic reconstruction of the accepted reversal
state machine. Otherwise it would absorb the exact H1 candidate.

Secondary robustness may add generic OBI slope or lag summaries to test
whether the path effect reduces to simple momentum. That stronger H0 cannot
create a positive result if the primary H1 fails.

## 20. First-Passage Competing-Risk Target

Orient every landmark so:

```text
new OBI direction = +1
old OBI direction = -1
```

Freeze reference quote `P_i,0` at decision time using one quote convention.

### 20.1 n_follow

The first admitted reference-price barrier is reached in the new OBI
direction:

```text
s_new * (P_t - P_i,0) >= k_ticks
```

### 20.2 n_fail

The first admitted barrier is reached in the old OBI direction:

```text
s_new * (P_t - P_i,0) <= -k_ticks
```

### 20.3 No transition

If neither barrier is reached before maximum follow-up:

```text
right_censored_at_tau_max
```

Capture end, gap, reconnect, reset or quality failure also censors the
interval.

### 20.4 Quote convention and barrier

A0 freezes:

- midpoint, touch or another explicit public quote convention;
- tick rounding;
- barrier `k_ticks`;
- simultaneous-hit and interval ambiguity rules.

One-tick first passage is the recommended primary target when supported by the
quote/tick contract. Other barriers are predeclared robustness only.

The model does not use post-barrier return magnitude or markout.

## 21. Risk-Set Construction

For directional entry `i`:

```text
t_i   = entry/reversal detected time
R_i   = reversal history indicator
C_i   = frozen H0 snapshot
T_i   = time to first follow/fail barrier or censor
J_i   = n_follow or n_fail when observed
```

At elapsed time `u`, the row remains at risk only when:

```text
T_i >= u
```

Dependence controls:

- one active interval per directional OBI state;
- no duplicate entry while that interval remains active;
- frozen refractory and hysteresis rules;
- no interval crosses capture, reconnect or quality boundaries;
- same-session rows remain in the same split;
- uncertainty is aggregated by session/date, not hazard row.

## 22. Data-Determined Time Scale

No single favorable horizon is chosen after outcomes are visible.

A0 examines complete follow-up geometry only on a log-spaced support grid:

```text
100ms
200ms
500ms
1s
2s
5s
10s
30s
60s
120s
300s
```

The primary `tau_max` is selected mechanically from:

- complete future coverage;
- independent-date support;
- censoring geometry;
- parameter budget.

It is selected before follow/fail event rates are inspected.

The elapsed-time baseline uses:

- 4-6 piecewise log-time bins; or
- a 3-5 degree-of-freedom restricted spline.

Primary scoring integrates over the complete frozen follow-up range. A
favorable individual horizon is diagnostic only.

## 23. Primary Competing-Risk Model

Use a discrete-time multinomial hazard.

For cause `k in {follow, fail}` at elapsed time `u`:

```text
eta_H0,k(i,u)
  = alpha_k(u)
  + gamma_k' C_i

eta_H1,k(i,u)
  = alpha_k(u)
  + gamma_k' C_i
  + beta_k * R_i
```

Conditional event probability:

```text
P(J_i = k at u | T_i >= u)
  = exp(eta_k(i,u))
    /
    (1 + sum_j exp(eta_j(i,u)))
```

The denominator's `1` is no price transition in the current elapsed-time bin.

H1 adds one degree of freedom per cause. The added information is exactly the
reversal path indicator.

## 24. Directional Expectations

The frozen experience-derived expectation is:

```text
beta_follow > 0
beta_fail   < 0
```

Equivalent cumulative-incidence expectation:

```text
F_follow(u | R=1, C)
  >
F_follow(u | R=0, C)

F_fail(u | R=1, C)
  <
F_fail(u | R=0, C)
```

over the predeclared early elapsed-time region or integrated primary range.

If H1 improves proper scores but coefficient directions contradict the frozen
expectation, the original interpretation fails and requires a new version.

## 25. Model Fitting

Requirements:

- preprocessing fitted on calibration/training roles only;
- train periods precede validation/replay where practical;
- purge and embargo around capture/session boundaries;
- H0 and H1 use identical risk rows, targets, censoring and elapsed-time basis;
- H1 differs from H0 only through `R_i`;
- regularization selected within development data;
- no-refit replay freezes OBI, reversal, controls, coefficients and thresholds;
- session/date is the replication and uncertainty unit.

Secondary models:

- H1 plus frozen reversal-strength descriptors;
- stronger H0 plus generic OBI slope/lag summaries;
- cause-specific logistic or Cox hazards;
- Aalen additive hazard;
- L1-only and L1-L3 OBI robustness.

Secondary models cannot rescue a failed primary H0/H1 result.

## 26. Primary Metrics

### 26.1 Proper scores

- held-out competing-risk negative log loss;
- integrated Brier score;
- cause-specific Brier scores;
- cumulative-incidence calibration;
- calibration slope/intercept;
- OOD and unsupported-entry rates.

Primary increments:

```text
Delta_log_loss = loss_H0 - loss_H1
Delta_IBS      = IBS_H0  - IBS_H1
```

Positive values favor H1.

### 26.2 Effect metrics

- `beta_follow` and `beta_fail`;
- reversal versus control cumulative incidence;
- probability of follow before fail within `tau_max`;
- restricted mean time to follow/fail;
- reversal/control entry rate per eligible hour;
- event and censor counts by date;
- maximum single-date contribution.

### 26.3 Alignment and online metrics

- pre-state dwell;
- cross-to-detection delay;
- fraction with pre-detection price transition;
- residual time from detection to first barrier;
- OBI reversal chatter/revision rate;
- reversal detector opportunity rate;
- fraction censored before useful elapsed time.

## 27. Statistical Uncertainty

Row-IID uncertainty is forbidden.

Use:

- date/session-block bootstrap;
- dependence-preserving calendar blocks;
- leave-one-date-out influence analysis;
- session-aggregated proper-score increments;
- event-count and effective-sample diagnostics.

The materiality threshold and confidence rule are frozen before target access.

P-values alone do not establish the pattern.

## 28. Dependence-Preserving Nulls

Required nulls:

```text
history_block_shift:
  shift reversal-history labels within session while preserving
  current OBI entry times and target paths

current_OBI_stratified_permutation:
  permute R only within frozen current-OBI/context strata

prehistory_time_reversal:
  reverse the pre-entry OBI path while preserving current OBI

side_orientation_disruption:
  preserve event rates and current state, break correct new-direction mapping

pseudo_reversal_controls:
  create matched entries with similar current OBI and activity
  but no accepted opposite pre-state

cross_detection_delay_null:
  test whether apparent effect occurred before causal confirmation
```

Each null:

- preserves session and quality boundaries;
- uses the same risk set and censoring;
- refits H0/H1 under the same procedure;
- is evaluated with the same primary score;
- participates in frozen multiplicity control.

## 29. Gate Chain

Exact numeric thresholds are frozen in A0 before target access.

### Gate A0: support and tuple admissibility

- source and reconstruction identities close;
- OBI has adequate L1-L5 support;
- reversal and control definitions are immutable;
- complete follow-up supports `tau_max`;
- session roles and access ledger close;
- target surfaces remain inaccessible.

### Gate A1: OBI state-machine validity

- reversal detection is causal and deterministic;
- current-OBI control support overlaps reversal support;
- reversal/control entries are not primarily session or quality fingerprints;
- opportunity and chatter rates are finite;
- cross-to-detection delay is auditable.

A1 inspects no future price direction.

### Gate A2: target variation and identification

- both follow and fail causes have adequate independent-date support;
- censoring and simultaneous-hit ambiguity remain below frozen caps;
- no single date supplies the target variation;
- pre-detection transitions are separately identified.

Insufficient support yields an inconclusive classification. It does not
authorize a different barrier or horizon under the same version.

### Gate A3: H0/H1 path-dependence increment

- H1 improves H0 on the frozen primary proper score;
- blocked confidence lower bound exceeds materiality;
- integrated Brier/calibration do not contradict the result;
- `beta_follow` and `beta_fail` follow frozen directions;
- no single date or elapsed-time bin carries the claim.

### Gate A4: null separation and historical transport

- observed increment beats all required dependence nulls;
- frozen model transports to historical no-refit replay;
- stronger-H0 diagnostics do not reduce the effect to simple OBI slope;
- session identity and quality artifacts do not explain the result.

### Gate A5: causal actionability and prospective confirmation

- a material fraction of transitions remains after `reversal_detected_at`;
- protocol-frozen prospective sessions reproduce the increment;
- calibration, opportunity rate and direction remain within tolerance;
- final holdout does not revise the same hypothesis version.

Only Gate A5 can produce the strongest positive classification.

## 30. Pass Semantics

An accepted result means:

```text
after controlling current OBI and ordinary context,
the path of arriving through a confirmed OBI reversal
changes the future first-passage price-direction distribution
on independent sessions
```

It does not mean:

- every reversal is followed by price in the new direction;
- OBI reversal produces deterministic alpha;
- post-transition return magnitude is positive;
- the signal survives execution latency;
- maker intervention is profitable;
- one threshold works on another venue or instrument.

## 31. Failure Classifications

Track A ends with exactly one primary classification:

```text
insufficient_OBI_support

OBI_reversal_state_machine_unstable

current_OBI_control_overlap_insufficient

price_transition_target_not_identified

target_variation_insufficient

OBI_reversal_no_increment_over_current_OBI_H0

increment_explained_by_generic_OBI_slope

increment_occurs_before_causal_detection

increment_driven_by_single_session_or_horizon

increment_not_dependence_null_distinct

historical_OBI_reversal_increment_not_transportable

historical_support_only_pending_prospective

stable_OBI_reversal_path_dependence_confirmed
```

The final classification requires prospective confirmation.

## 32. Generic Composite M-State Status

The revision-1 candidate:

```text
pressure-side withdrawal
  + aggressive-flow persistence
  + replenishment deficit
```

is not part of the `OBI_REVERSAL_V1` primary family.

It may become a separately versioned successor only after:

- OBI_REVERSAL_V1 closes; or
- an independent controller explicitly authorizes parallel multiplicity.

It cannot:

- rescue a failed OBI reversal result;
- alter OBI thresholds or targets;
- enter H0/H1 feature selection;
- be combined post hoc with favorable OBI cases.

## 33. Secondary Motif Diagnostics

Only after Gate A3 passes may the study inspect:

- aligned OBI paths;
- reversal-strength subgroups;
- bounded soft-DTW;
- k-medoids;
- interpretable shapelets;
- follow/fail path heterogeneity.

These diagnostics cannot modify OBI, reversal/control membership, targets or
the primary classification.

## 34. Track B Unlock Contract

Track B remains locked unless Track A reaches at least:

```text
historical_support_only_pending_prospective
```

and a separate review authorizes outcome expansion.

The Track A handoff contains:

- OBI formula and weights;
- reversal/control state machine;
- decision and cross timestamps;
- H0/H1 models;
- causal entry predictions;
- follow/fail/censoring ledger;
- calibration and OOD state;
- source, config, model and code hashes.

Track B may separately define markout, maker risk, fill or economics. Those
outcomes cannot retroactively tune `OBI_REVERSAL_V1`.

## 35. Required Artifacts

```text
artifacts/skhynix_obi_reversal_track_a/
  contracts/
    source_manifest.json
    session_role_ledger.csv
    research_tuple.json
    OBI_contract.json
    reversal_state_machine.json
    control_entry_contract.json
    target_contract.json
    censoring_contract.json
    H0_H1_contract.json
    gate_contract.json
    outcome_access_ledger.json
  support/
    admitted_intervals.csv
    OBI_support.csv
    followup_support.csv
    effective_sample_support.csv
  states/
    directional_entry_ledger.parquet
    reversal_ledger.parquet
    control_overlap.csv
    detection_delay.csv
  targets/
    first_passage_ledger.parquet
    censoring_summary.csv
    target_variation.csv
    predetection_transition.csv
  models/
    H0_spec.json
    H1_spec.json
    coefficients.parquet
    validation_predictions.parquet
  metrics/
    proper_scores.csv
    cumulative_incidence.csv
    calibration.csv
    session_influence.csv
    online_actionability.csv
  nulls/
    null_manifest.json
    null_scores.csv
    multiplicity_results.csv
  replay/
    no_refit_predictions.parquet
    transport_metrics.csv
  reports/
    track_a_report.md
    failure_analysis.md
  classification.json
  run_manifest.json
```

Every artifact carries source hashes, code commit, config hash, model hash,
timestamp semantics and session role.

## 36. Parameter And Sample Discipline

Primary model capacity is intentionally small:

- one aggregate OBI;
- one reversal indicator;
- one low-parameter H0 context;
- two competing causes;
- one H1 coefficient per cause;
- one frozen elapsed-time basis;
- date-blocked validation;
- explicit censoring.

Effective support is measured by independent entries, transitions and
sessions, not 100ms hazard rows.

When support is insufficient, the valid result is inconclusive. Model
capacity, target horizon or barrier cannot be expanded to manufacture support.

## 37. Design Risks

### Current OBI masquerades as reversal effect

Require current OBI in H0 and current-OBI overlap diagnostics.

### Price moves before confirmation

Record `pre_detection_transition`; start the primary clock at
`reversal_detected_at`.

### Reversal threshold is outcome-tuned

Freeze OBI bands, dwell and hysteresis under zero-target A0.

### OBI slope explains everything

Run a stronger-H0 robustness with generic slope/lag features. Classify
`increment_explained_by_generic_OBI_slope` when appropriate.

### Threshold entry creates dense dependent samples

Use one active interval, hysteresis, refractory rules and session-block
uncertainty.

### One date supplies the result

Use date influence caps, blocked intervals and no-refit replay.

### Experience is mistaken for evidence

Treat experience only as hypothesis provenance and expected direction.

### Favorable horizon selection

Use support-selected `tau_max` and integrated proper scoring.

### Historical replay is called prospective

Require new protocol-frozen sessions for final confirmation.

## 38. First Execution Task

The next formal task should be:

```text
SKHYNIX-BINANCE-OBI-REVERSAL-TRACK-A0-SUPPORT-AND-TUPLE-FREEZE
```

It must have zero future price-target access.

It should:

1. close source and reconstructed-cache identities;
2. freeze data roles and admitted intervals;
3. freeze equal-weight L1-L5 OBI and missing-level semantics;
4. freeze old/new bands, neutral band, pre-dwell, confirmation dwell and
   hysteresis;
5. freeze `cross_at`, `detected_at` and pre-detection handling;
6. freeze reversal and same-current-OBI non-reversal entry rules;
7. freeze current-state H0 context;
8. freeze first-passage quote convention, tick barrier and censoring;
9. select `tau_max` from complete support without event-rate access;
10. freeze proper-score, uncertainty, null and materiality gates;
11. verify zero access to follow/fail outcomes and aligned price plots.

Only after A0 QA acceptance may the first-passage targets be materialized.

## 39. Final Nonclaims

This plan does not claim:

- that OBI reversal occurs frequently enough;
- that current-OBI-matched controls have adequate support;
- that reversal history improves H0;
- that the expected direction is correct;
- that the effect survives causal confirmation delay;
- that historical data provide prospective confirmation;
- that first-passage direction implies positive markout;
- that the signal is actionable for a maker;
- that any quoting strategy is profitable;
- that the result transfers to another venue or symbol.

Its purpose is to turn one interpretable empirical experience into a sharply
defined path-dependence hypothesis that can fail.

## 40. A0 Execution Result - 2026-08-28

Formal task `0828T004` executed the zero-target
`OBI_REVERSAL_V1` support and tuple-freeze stage against the closed historical
dataset.

The result is:

```text
status: passed
classification: A0_support_and_tuple_frozen
future_price_target_access: false
next_stage_authorized:
  A1 OBI state-machine materialization
  A2 first-passage target materialization under a new formal task
```

This pass means that the data, support and frozen contracts are sufficient to
run the next historical hypothesis test. It does not mean that OBI reversal
has predictive value.

### 40.1 Source And Outcome-Access Closure

A0 verified:

- 29 public Binance captures across 9 research dates;
- 35.9172 hours of reconstructed 100ms top-5 book state;
- raw size and SHA closure, including a fresh raw-hash verification;
- reconstructed-cache size and SHA closure;
- zero declared depth gaps;
- no private API access, orders or new collection.

The A0 loader read only:

- `bid_qty_log_l1` through `bid_qty_log_l5`;
- `ask_qty_log_l1` through `ask_qty_log_l5`;
- `ts_ns`, `valid`, capture identity, date and role.

It did not read future price fields, materialize follow/fail labels or create
reversal-aligned future-price plots.

### 40.2 OBI Support

Equal-weight L1-L5 standard OBI was reconstructed as:

```text
OBI =
  sum_l(Q_bid_l - Q_ask_l)
  /
  sum_l(Q_bid_l + Q_ask_l)
```

Twenty-three captures were informative. Six captures had constant or
near-constant reconstructed OBI and were excluded from state/control support
while remaining in the source-provenance ledger:

```text
2026-07-29_c092d6b4402c
2026-07-30_1b18a29f3daf
2026-08-03_098332acc16a
2026-08-03_b7b76aed76b8
2026-08-03_d8e3322d6f2c
2026-08-26_a055aa6c7a87
```

All 9 research dates retained at least one informative capture.

### 40.3 State-Machine Freeze

The support-only threshold trace used 1 second old-state dwell and 1 second
opposite-state confirmation:

| Absolute OBI threshold | Reversals | Rate/hour | Maximum date share | Qualified | Selected |
| ---: | ---: | ---: | ---: | :---: | :---: |
| 0.40 | 5,830 | 162.318 | 0.2506 | no | no |
| 0.50 | 4,021 | 111.952 | 0.2502 | yes | yes |
| 0.60 | 2,537 | 70.635 | 0.2495 | yes | no |

The primary absolute OBI threshold is frozen at `0.50`. It is the smallest
candidate passing the predeclared support-only event-rate, date-count and
date-concentration gates. No future outcome was used for selection.

The frozen primary alignment remains `reversal_detected_at`, after the full
confirmation dwell. `reversal_cross_at` remains diagnostic only.

### 40.4 Common Risk-Set Support

A0 found:

```text
reversal entries:                  4,021
non-reversal control candidates:   3,162
reversals inside common support:   4,005
overall common-support coverage:   0.9960
minimum per-date coverage:         0.8764
```

The primary H0/H1 comparison uses the full reusable common risk set defined by
date, direction and current-OBI bin. A single eligible non-reversal state may
therefore support multiple risk-set comparisons in the hazard model.

A no-reuse one-to-one matching diagnostic produced 2,801 pairs and 0.6966
matched coverage. That lower number is a matching-allocation constraint, not
a lack of covariate overlap, and is not the primary support gate.

The first implementation incorrectly used no-reuse matched coverage as the
control-overlap gate. This was corrected before any future-price target access:

- primary support gate: reusable common risk-set coverage;
- secondary diagnostic: unique no-reuse matched pairs;
- H0/H1 estimand: full risk-set path-dependence increment.

This distinction is now protected by a focused regression test.

### 40.5 Follow-Up Geometry And Frozen Horizon

Follow-up support was selected only from capture boundaries and completeness,
without event direction or price outcomes.

The largest qualifying horizon is:

```text
tau_max:                         120 seconds
complete reversals:             3,910
complete diagnostic pairs:      2,680
overall reversal completeness:  0.9724
minimum per-date completeness:  0.8276
dates with complete pairs:      9
```

The next candidate, 300 seconds, failed support with 0.9363 overall and 0.6897
minimum per-date completeness. The primary competing-risk analysis therefore
freezes `tau_max = 120 seconds`; shorter elapsed-time structure remains inside
the integrated hazard model rather than becoming separately selected
horizons.

### 40.6 Gate Result And Next Boundary

All A0 gates passed:

- source closure;
- cache closure;
- OBI support;
- state-machine support;
- common risk-set overlap;
- follow-up support;
- zero future-target access.

The next stage may materialize the frozen reversal/control ledgers and
symmetric one-tick first-passage outcomes under a new formal task. It must not
change the OBI formula, threshold, dwell, alignment time, H0/H1 increment,
barrier, censoring or `tau_max` after viewing the targets.

The strongest current claim is:

```text
A0 support and tuple freeze completed.
OBI reversal predictive value remains untested.
Historical results cannot be called prospective.
```

## 41. A1/A2 Execution Result - 2026-08-28

Formal task `0828T005` materialized the frozen causal state ledger and
first-passage target without fitting H0 or H1.

The result is:

```text
status: passed
classification: A1_A2_state_and_targets_materialized
H0_H1_increment_tested: false
predictive_value_claim_allowed: false
next_stage: A3 H0/H1 competing-risk test
```

This pass means that both target causes vary across dates and entry types with
enough support to fit the frozen model comparison. It is not evidence that the
reversal-history indicator improves H0.

### 41.1 A1 State Ledger

A1 reproduced the frozen A0 ledgers exactly:

```text
all reversal entries:       4,021
all control candidates:     3,162
all entries:                7,183
```

The primary reusable common risk set contains:

```text
primary reversal entries:   4,005
primary control entries:    3,159
primary entries:            7,164
research dates:             9
```

Each entry carries the frozen decision identity and only information available
by `decision_timestamp`:

- current equal-weight L1-L5 OBI;
- L1 and L1-L3 imbalance summaries;
- per-level imbalance dispersion;
- total depth and depth concentration;
- spread;
- contemporaneous signed book flow and public trade flow;
- trailing 1-second and 5-second midpoint movement;
- trailing 5-second absolute movement;
- source age, activity and no-new-information state;
- frozen UTC time-of-day basis.

Generic OBI slope and lag features remain outside primary H0 and are reserved
for the stronger-H0 diagnostic.

### 41.2 A2 Target Materialization

The target uses the frozen convention:

```text
quote:                 midpoint
grid:                  100ms grid close
clock start:           decision timestamp
first future input:    next grid after decision
orientation:           side * future midpoint displacement
follow barrier:        +1 tick
fail barrier:          -1 tick
tau_max:               120 seconds
```

The output retains only:

- first transition type;
- cause code;
- first transition time;
- censoring and observed interval-ambiguity status.

It does not retain post-hit displacement, return, markout or any path after the
first hit.

### 41.3 Target Variation

Across 7,164 primary entries:

| Cause | Count | Fraction | Date count | Maximum date share | Time p50 | Time p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Follow | 4,247 | 0.5928 | 9 | 0.3421 | 300ms | 1,500ms |
| Fail | 2,917 | 0.4072 | 9 | 0.2756 | 400ms | 1,600ms |

Both causes occur in both entry types:

| Entry type | Follow | Fail |
| --- | ---: | ---: |
| Reversal | 2,345 | 1,660 |
| Control | 1,902 | 1,257 |

There were no `tau_max`, capture-end or quality censored entries in the
primary ledger. No observed grid-close interval was classified as an
ambiguous dual hit.

The raw marginal follow fractions are approximately:

```text
reversal: 0.5855
control:  0.6021
```

These marginals are not the H0/H1 result. They do not control current state,
date, activity, elapsed time or the reusable risk-set structure, and they
cannot be used to accept or reject `OBI_REVERSAL_V1`.

### 41.4 Temporal Resolution Warning

The one-tick target is short relative to the 100ms observation grid:

| Elapsed time | Fraction with a first hit |
| ---: | ---: |
| 100ms | 0.2446 |
| 200ms | 0.4008 |
| 500ms | 0.6602 |
| 1s | 0.8332 |
| 2s | 0.9344 |
| 5s | 0.9876 |
| 10s | 0.9992 |
| 30s | 1.0000 |

This is recorded as:

```text
one_tick_barrier_near_100ms_grid_resolution
```

The A0 barrier cannot be changed after viewing these outcomes. A3 must use the
frozen one-tick target and report elapsed-time-bin influence. A different
barrier or event-time quote reconstruction would be a separately versioned
hypothesis, not a robustness result that can rescue this version.

The ambiguity rate of zero means that no dual hit is observable at grid-close
sampling. It is not proof that both barriers were never touched inside a
100ms interval.

### 41.5 Confirmation-Delay Warning

Among the 4,005 primary reversal entries, the cross-to-detection interval
contains:

```text
pre-detection follow hit:  2,986
pre-detection fail hit:      716
no pre-detection hit:        303
any pre-detection hit:    0.9243
```

This is recorded as:

```text
high_pre_detection_transition_fraction
```

The result does not leak into the post-decision target. It is a separate
actionability diagnostic using information already known at
`reversal_detected_at`.

It changes the interpretation of a future positive A3 result. Even if
reversal history improves post-detection first-passage prediction, the signal
would often be detected after an earlier one-tick transition had already
occurred. A3 must therefore distinguish:

- incremental continuation risk after confirmation;
- price movement that occurred before causal confirmation;
- statistical path dependence from maker actionability.

### 41.6 Gate And Access Result

All frozen A1 and A2 identification gates passed:

- upstream A0 contracts and artifacts closed;
- state features were finite and deterministic;
- both entry types had common support on all 9 dates;
- both target causes exceeded 200 events and covered all 9 dates;
- maximum per-date cause shares remained below 0.35;
- censoring and observed ambiguity remained below their caps;
- both reversal and control entries contained both causes.

The access ledger confirms:

```text
A1 future fields read:                         []
A2 future field read:                          midpoint_delta_ticks
future values retained:                        J, T, censoring
post-first-passage return/markout materialized: false
H0/H1 fitted:                                  false
private access/orders/new collection:          false
```

The strongest current claim is:

```text
A1 state ledger and A2 target materialization completed.
The target is identifiable but temporally short.
Confirmation delay is a major actionability risk.
OBI reversal path-dependence remains untested until A3.
```

## 42. A3 Execution Result - 2026-08-28

Formal task `0828T006` fitted the frozen H0/H1 discrete-time multinomial
competing-risk models.

The primary result is:

```text
status: failed
classification: A3_no_increment_over_H0
selected ridge: 0.001
```

This is a scientific primary-hypothesis failure, not an execution failure. The
models converged, the H1 design differed from H0 only by `R`, the validation
split remained untouched, and the result reproduced deterministically.

### 42.1 Frozen Model

The role split was:

| Role | Dates | Entries |
| --- | --- | ---: |
| Train/preprocess | Jul29, Jul30, Aug03, Aug04 | 1,296 |
| Blocked validation | Aug07, Aug24, Aug25 | 3,187 |
| No-refit replay | Aug26, Aug27 | 2,681 |

H0 used:

- six frozen elapsed-time baseline bins;
- nineteen side-oriented and train-standardized decision-time features;
- 50 parameters across the two causes.

H1 used the identical risk rows, target, scaler, elapsed-time basis and ridge,
and added only:

```text
beta_follow * R
beta_fail   * R
```

H1 therefore had 52 parameters. Both models converged with maximum absolute
gradients below `1e-7`.

The ridge grid was:

```text
0.0001, 0.001, 0.01, 0.1
```

It was selected using H0-only leave-one-development-date-out entry NLL. The
blocked validation and replay outcomes did not participate in selection.

### 42.2 Primary Proper Scores

Positive deltas favor H1:

```text
Delta_NLL = NLL_H0 - NLL_H1
Delta_IBS = IBS_H0 - IBS_H1
```

Results:

| Dataset | Delta NLL | Delta IBS |
| --- | ---: | ---: |
| Historical train | +0.003015 | +0.002690 |
| Blocked validation | -0.001603 | -0.000675 |
| No-refit replay | -0.008676 | -0.001749 |

H1 learned a small in-sample increment, but the increment reversed sign on
every untouched historical stage. On blocked validation:

```text
H0 date-equal entry NLL: 3.078672
H1 date-equal entry NLL: 3.080276
```

The frozen materiality threshold was `+0.002 nats/entry`. H1 did not merely
miss materiality; it made the primary proper score worse.

Both cause-specific integrated Brier deltas were also negative:

```text
follow Brier delta: -0.000453
fail Brier delta:   -0.000221
```

Thus the NLL result is not contradicted by a favorable Brier result.

### 42.3 Coefficient Direction Is Not Enough

The fitted H1 coefficients followed the experience-derived direction:

```text
beta_follow = +0.164963
beta_fail   = -0.158056
```

This means that inside the training fit, `R=1` shifted hazard toward follow
and away from fail after ordinary context was controlled.

However, correct coefficient signs do not establish a stable conditional
pattern. The out-of-sample proper scores show that applying those shifts to
new dates worsened prediction.

The result therefore distinguishes:

```text
training-period directional association: present
cross-date incremental predictive law:   not supported
```

### 42.4 Date-Block Evidence

Every blocked-validation date had negative NLL increment:

| Date | Entries | Delta NLL |
| --- | ---: | ---: |
| 2026-08-07 | 948 | -0.000245 |
| 2026-08-24 | 1,062 | -0.003007 |
| 2026-08-25 | 1,177 | -0.001558 |

The 5,000-replicate date-block bootstrap was:

```text
lower 95%: -0.003007
median:    -0.001603
upper 95%: -0.000245
```

Even the upper bound remained negative. The result is not carried by one
adverse validation date.

The no-refit replay dates were also negative:

| Date | Entries | Delta NLL |
| --- | ---: | ---: |
| 2026-08-26 | 2,257 | -0.016701 |
| 2026-08-27 | 424 | -0.000651 |

Replay cannot determine the A3 classification and is not needed to establish
the failure, but it supplies additional non-rescue evidence.

### 42.5 Elapsed-Time Influence

Blocked-validation increments by the event-time bin were:

| Event-time bin | Entries | Delta NLL |
| --- | ---: | ---: |
| 100ms | 844 | +0.001295 |
| 200-500ms | 1,475 | +0.000858 |
| 600-1,000ms | 570 | -0.007745 |
| 1,100-2,000ms | 255 | -0.011495 |
| 2,100-5,000ms | 43 | -0.006310 |

The only positive increments were in the first two, sub-500ms bins, and both
were below materiality.

Removing first-100ms entries produced:

```text
Delta_NLL = -0.002711
```

Removing all entries completed by 500ms produced:

```text
Delta_NLL = -0.008775
```

Thus the weak favorable component is concentrated exactly where the frozen
one-tick target is closest to the 100ms grid resolution. It does not persist
into the slower residual transition process.

### 42.6 Ridge Sensitivity

Ridge sensitivity was run after the primary fit and was diagnostic only:

| Ridge | Delta NLL | Delta IBS | beta follow | beta fail |
| ---: | ---: | ---: | ---: | ---: |
| 0.0001 | -0.001747 | -0.000759 | +0.1692 | -0.1639 |
| 0.001 primary | -0.001603 | -0.000675 | +0.1650 | -0.1581 |
| 0.01 | -0.000725 | -0.000163 | +0.1322 | -0.1158 |
| 0.1 | +0.000118 | +0.000195 | +0.0472 | -0.0267 |

Heavy shrinkage reduces H1 toward H0 and produces a negligible positive
increment at ridge `0.1`. It remains far below the frozen `0.002`
materiality threshold and cannot rescue the primary model.

### 42.7 Interpretation

The current evidence supports:

```text
current OBI and ordinary context contain short-horizon price information
```

but does not support:

```text
after current OBI and ordinary context are controlled,
binary accepted-opposite-state ancestry R
adds a stable cross-date competing-risk increment
```

Likely contributors, which are interpretations rather than separately tested
causes, include:

- the binary indicator compresses heterogeneous reversal paths into one bit;
- H0 already absorbs current OBI, flow and trailing price movement;
- 92.43% of reversals had a one-tick transition before causal confirmation;
- the frozen one-tick target is concentrated near the 100ms grid;
- the training relationship changes across market dates.

These observations do not authorize post-hoc path descriptors, interactions,
different barriers, faster alignment or stronger/weaker ridge to rescue
`OBI_REVERSAL_V1`.

### 42.8 Gate Result And Research Boundary

Passed:

- H0 and H1 convergence;
- H1 equals H0 plus `R` only;
- `beta_follow > 0`;
- `beta_fail < 0`;
- cause-specific Brier contradiction tolerances.

Failed:

- validation NLL improvement and materiality;
- date-block confidence lower bound;
- integrated Brier improvement;
- positive-date support;
- elapsed-bin breadth;
- positive increment after excluding first 100ms.

The canonical classification is:

```text
A3_no_increment_over_H0
```

Therefore:

- A4 nulls and stronger-H0 diagnostics are not required to reject the primary;
- no historical path-dependence candidate is declared;
- no predictive-value, maker-actionability or prospective claim is allowed;
- `OBI_REVERSAL_V1` must stop as the primary version.

A future study may register a new interpretable hypothesis, but it must be a
new version with new provenance and frozen contracts. It cannot be described
as a robustness rescue of this result.
