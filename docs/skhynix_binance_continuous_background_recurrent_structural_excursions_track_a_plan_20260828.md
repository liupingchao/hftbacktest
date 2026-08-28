# SKHYNIX Binance Phase Alignment Track A: Continuous Background With Recurrent Structural Excursions - 2026-08-28

Date: 2026-08-28

Status: review draft; design-only research contract; not execution authority

Research identifier:

```text
continuous_background_with_recurrent_structural_excursions
```

Predecessor:

```text
docs/skhynix_binance_phase_alignment_track_a_outcome_blind_motif_discovery_plan_20260827.md
```

Authority boundary:

- This document defines Track A structural research only.
- It grants no new collection, private endpoint, order, cancel, strategy,
  deployment, live-capital, Track B outcome access, or Track C economics
  authority.
- Each execution stage requires a formal workflow task and frozen inputs.

## 1. Decision

Continue researching the alignment idea, but reject the assumption that the
market must repeatedly complete:

```text
N -> S -> P -> R -> N
```

The accepted historical result from `0827T004` is:

```text
continuous_state_no_discrete_phase_support
```

The data supported a continuous process more strongly than a closed,
finite-state phase cycle. Extending duration support from seconds to minutes
did not reverse that result. A small diagonal AR(1) continuous baseline still
outperformed the minute-scale Student-t HSMM, and complete long-duration
episodes were too sparse to identify a common closed cycle.

The new hypothesis is therefore:

```text
continuous background
  -> recurrent local structural excursion
  -> background
     or another excursion
     or persistent displacement
     or out-of-distribution state
     or censored observation
```

Track A will align variable-length structural excursions, not fixed future
windows and not a predefined state cycle.

## 2. Falsifiable Hypothesis

Let `X_t` be the causally reconstructed public Binance order-book and
trade-flow tensor. Represent it as:

```text
X_t = B_t + R_t
```

where:

- `B_t` is a continuous, slowly changing conditional background;
- `R_t` is the innovation or structural residual not explained by the
  background model using information available by `t`.

An excursion candidate `E_i` is a variable-duration residual segment:

```text
E_i = {R_t : tau_i_start <= t <= tau_i_end}
```

The research hypothesis is:

> After accounting for a causal continuous background, some residual
> excursions recur across independent sessions with stable multilevel
> order-book geometry, temporal shape, and branch behavior, and at least some
> can be recognized online before they terminate.

A useful descriptive model is:

```text
R_t ~= A_i * M_k((t - tau_i_start) / d_i) + epsilon_t
```

where:

- `M_k` is a neutral structural motif;
- `A_i` contains signed or side-symmetric amplitude information;
- `d_i` is variable duration;
- `epsilon_t` is unexplained residual variation.

This equation is not a requirement that every excursion be a warped copy of
one template. It defines the strongest form of recurrence that Track A may
test and reject.

## 3. What Alignment Means In This Plan

The alignment object is no longer:

```text
one queue-depletion crossing
  -> fixed next 500ms
```

Track A permits three structural anchors:

```text
1. onset alignment:
   causal change-point or innovation threshold crossing

2. internal alignment:
   outcome-blind local shapelet or medoid landmark

3. termination alignment:
   return to background, transition to another excursion,
   persistent displacement, OOD, or censoring
```

Duration is estimated from the data. It is not assumed to be 500ms, 12.8s,
51.2s, or one minute. Candidate durations may span milliseconds to minutes,
subject to actual support and censoring.

Retrospective warping may be used only to measure structural similarity after
segments have been extracted. It may not supply future information to the
online recognizer.

## 4. Explicit Rejection Of The Old Positive Seed

The following objects from the predecessor route are negative evidence only:

```text
N / S / P / R semantic state names
fixed N -> S -> P -> R -> N grammar
old HSMM state assignments
old phase medoids or prototypes
old duration assignments
old transition matrix
old retrospective boundaries
```

They may be used to document why the route changed. They may not:

- initialize new clusters;
- seed shapelets;
- define excursion boundaries;
- choose `K`;
- choose a duration range;
- set distance weights;
- define success thresholds.

Existing reconstructed causal feature caches may be reused only after source,
schema, timestamp, grid, and hash closure. Reuse of the observations does not
authorize reuse of the failed labels.

## 5. Track A Question And Claim Limit

Track A asks:

> Does the public Binance SKHYNIX order book contain recurrent, transportable,
> outcome-blind structural excursions around a continuous background, and can
> those excursions be detected causally?

Track A does not ask whether a discovered motif predicts:

```text
future return
future volatility
future midpoint or BBO movement
future adverse selection
future quote contact
future fill
future spread capture
future PnL
maker profitability
```

The strongest conclusion available from already inspected historical data is:

```text
historical_support_only_pending_prospective
```

A final positive structural claim requires new, protocol-frozen prospective
Binance public-data sessions.

## 6. Outcome-Blind Information Boundary

At time `t`, Track A may use only information observable by `t`:

- current and trailing BBO;
- current and trailing L1-L5 depth;
- public depth updates and trades;
- causal add, cancel, depletion and replenishment estimates;
- current and trailing spread, midpoint and microprice;
- current and trailing book slope, concentration and imbalance;
- current and trailing signed trade and order-flow variables;
- current and trailing realized movement ending no later than `t`;
- source age, sequence continuity, reconnect epoch and quality masks;
- session identity for splitting and diagnostics only.

No `t+h` response join is permitted. Future observations may be used only to
label the retrospective end of an already extracted segment for offline
structural evaluation. They may not enter causal onset, prefix recognition,
feature normalization, hyperparameter selection, or model ranking.

## 7. Data Roles

The existing inventory contains 29 captures over 9 dates, approximately
35.917 admitted hours, and 1,292,945 valid 100ms reconstructed rows. This is
adequate for historical method development and structural replay, but no
existing date is a fresh prospective final holdout.

Freeze roles before fitting:

| Dates | Role | Permitted use |
|---|---|---|
| 2026-07-29 | normalization and reconstruction calibration | data support only |
| 2026-07-30, 2026-08-03, 2026-08-04 | historical method development | estimator and support development |
| 2026-08-07, 2026-08-24, 2026-08-25 | blocked historical structural validation | model selection under frozen blocks |
| 2026-08-26, 2026-08-27 | historical no-refit replay | final historical replay only |
| newly collected dates | prospective validation and final holdout | required for final positive claim |

The exact ledger must be regenerated from admitted files and hashes in A0. If
the inventory differs, execution stops and publishes the discrepancy instead
of silently changing roles.

## 8. Stage Chain

```text
A0 support and surface freeze
  -> A1 continuous background and innovation residual
  -> A2 causal change-points and variable-length excursions
  -> A3 outcome-blind motif dictionary discovery
  -> A4 cross-session transport and structural nulls
  -> A5 causal online recognition and final classification
```

Each stage is separately reviewable. A later stage may not repair an earlier
failure by redefining its inputs.

## 9. A0: Support And Surface Freeze

A0 fits no motif and reads no future response.

It must publish:

- exact source inventory and SHA identities;
- timestamp semantics and deterministic reconstruction contract;
- sequence gaps, reconnects, resets and unavailable intervals;
- accepted depth, initially L1-L5;
- eligible calendar exposure by date and capture;
- complete dependence blocks at each candidate scale;
- session-role ledger;
- feature and normalization contract;
- background estimator candidates;
- change-point candidate family;
- segment termination and censoring rules;
- distance family and weight-freezing method;
- null family;
- model-complexity budgets;
- quantitative gates;
- outcome-blind access ledger.

Primary reconstruction remains:

```text
100ms causal grid
```

Derived structural views may include:

```text
200ms
500ms
1s
2s
5s
```

These are representation scales, not fixed excursion horizons. Multiscale
features must be computed causally and must not duplicate future samples
through centered filters.

A0 also publishes an effective-sample-size table. Parameter budgets must be
defined against independent calendar blocks, not raw grid rows.

## 10. Causal Structural Representation

### 10.1 Spatial order-book channels

For levels `L1-L5` on both sides:

- displayed quantity;
- distance from same-side best and midpoint;
- level occupancy;
- add intensity;
- cancel intensity;
- depletion intensity;
- replenishment intensity;
- persistence and age where reconstructable;
- migration of depth toward or away from the touch.

### 10.2 Flow and price-state channels

- signed public-trade quantity and count;
- aggressive-flow persistence;
- spread and spread change;
- midpoint and microprice change ending by `t`;
- top-level and multilevel imbalance;
- depth concentration and slope;
- two-sided contraction or expansion;
- causal realized movement over trailing windows.

### 10.3 Quality channels

- source age;
- sequence continuity;
- no-new-information flag;
- reconnect epoch;
- missing-level mask;
- stale-state mask;
- segment censoring mask.

Quality variables control admission and OOD classification. They may not be
allowed to become an accidental session fingerprint.

### 10.4 Normalization

Normalization is:

- causal or calibration-only;
- robust to heavy tails;
- session-transportable;
- frozen before replay;
- independent of future outcomes.

Primary normalization uses calibration-set median and robust scale by channel.
A causal exponentially weighted scale is a robustness variant only if its
half-life is frozen in A0.

Both signed orientation and side-symmetric geometry are useful. The primary
representation must freeze one of:

```text
absolute bid/ask orientation
aggressor-relative mirrored orientation
side-unoriented symmetric representation
```

Orientation may not be selected using future market response.

## 11. A1: Continuous Background

The background is not a single quiet-state centroid. It is the expected
short-horizon evolution conditional on observable history:

```text
B_t = E[X_t | X_(<=t-1), quality_(<=t)]
R_t = X_t - B_t
```

### 11.1 Primary estimator

Use a low-parameter diagonal robust AR(1) or channel-group AR model:

```text
X_t,j = alpha_j + phi_j * X_(t-1,j) + beta_j' * quality_t + eta_t,j
```

Reasons:

- it already provided a strong historical continuous baseline;
- it is causal and auditable;
- its parameter count remains controlled at minute scales;
- residuals have a direct interpretation.

### 11.2 Secondary estimators

Use only as frozen robustness comparisons:

- ridge VAR with low-rank or grouped cross-channel structure;
- robust linear state-space model;
- trailing local-level model;
- seasonal intraday background only if support exists across dates.

Deep sequence models are excluded from the primary Track A study. They add
capacity before recurrence and sample support have been established.

### 11.3 Background acceptance

A1 reports by date and block:

- held-out predictive log density or robust loss;
- innovation autocorrelation;
- cross-channel residual dependence;
- tail rate;
- stability of fitted coefficients;
- effective parameter count;
- comparison with persistence and unconditional robust baselines.

Track A does not require perfectly white residuals. It requires a frozen
background that removes ordinary local persistence without absorbing the
candidate structural excursions.

The background family and regularization may be selected only on development
and blocked validation roles. Historical replay dates are no-refit.

## 12. A2: Change-Points And Variable-Length Excursions

### 12.1 Onset candidates

Use an ensemble of outcome-blind causal detectors:

- robust multivariate CUSUM on innovation energy;
- Bayesian online change-point detection with a frozen hazard family;
- covariance or subspace change detector;
- channel-group threshold crossings for sparse localized excursions.

No single queue depletion is an excursion by definition. It is one possible
component of an innovation vector.

### 12.2 Start rule

An onset requires:

- detector evidence above its A0-frozen threshold;
- minimum affected-channel breadth or a predeclared sparse exception;
- quality admissibility;
- minimum separation from a prior onset unless declared a branch;
- support in more than one representation scale for the primary detector.

### 12.3 End rule

An excursion ends at the earliest causally observable condition:

```text
return_to_background:
  residual energy and detector evidence remain below exit thresholds
  for a frozen dwell period

next_excursion:
  a new incompatible change-point begins before return

persistent_displacement:
  residual structure remains but exceeds the supported duration cap

OOD:
  feature or quality state leaves calibrated support

censored:
  capture end, gap, reconnect, reset, or missing required observations
```

### 12.4 Segment policy

- Preserve variable duration.
- Do not force every start to have a return.
- Preserve aborted, branched and persistent excursions.
- Use maximal non-overlapping primary segments.
- Store overlapping detector proposals in an audit table.
- Never concatenate across reconnects or unsupported gaps.
- Record left and right censoring explicitly.

The duration cap is a support and computation boundary, not an assertion that
the excursion ends at that time. Candidates may span milliseconds to minutes.

### 12.5 A2 outputs

- change-point table;
- detector agreement table;
- excursion registry;
- branch and termination labels;
- censoring table;
- duration-support survival table;
- per-date and per-session onset rates;
- admitted residual tensor slices;
- rejected-candidate reason table.

## 13. A3: Outcome-Blind Motif Discovery

Track A seeks a small neutral dictionary:

```text
M0, M1, ..., M(K-1), OOD
```

`K` is selected using structural compression, stability and null separation,
not future response.

### 13.1 Primary: variable-length k-medoids

Use actual excursion segments as medoids. The primary distance is:

```text
D(E_i, E_j)
  = w_spatial  * d_spatial(E_i, E_j)
  + w_shape    * d_soft_dtw(E_i, E_j)
  + w_duration * abs(log(d_i / d_j))
```

where:

- `d_spatial` measures L1-L5 channel geometry and propagation;
- `d_soft_dtw` permits bounded local temporal deformation;
- duration remains a separate signal and cannot be erased by warping.

Distance components are whitened on development support. Weights are frozen
mechanically from scale normalization or an A0-declared equal-weight rule.
They may not be tuned for attractive clusters.

Warping constraints must prevent:

- reversing temporal order;
- matching onset to termination;
- collapsing minute excursions into millisecond bursts;
- ignoring level identity;
- aligning through censored regions.

### 13.2 Complementary: unsupervised shapelet dictionary

Shapelets represent recurring local primitives inside excursions. Candidate
examples may later be interpreted as:

- one-sided withdrawal propagation;
- same-price absorption and refill;
- depth migration away from the touch;
- two-sided liquidity contraction;
- persistent signed-flow corridor;
- aborted impulse or reversal.

These descriptions are not labels, seeds, or required discoveries.

Shapelets must be selected without future response labels. Selection uses:

- repeated occurrence count;
- support across independent dates;
- reduction in structural reconstruction error;
- non-redundancy;
- separation from matched nulls;
- causal prefix detectability.

Matrix-profile or nearest-neighbor subsequence search may generate candidates.
The final dictionary must satisfy a frozen parameter and multiplicity budget.

### 13.3 Complementary: change-point segment clustering

Cluster fixed-statistic summaries of variable segments:

- onset jump vector;
- peak residual geometry;
- cumulative channel displacement;
- propagation order across L1-L5;
- signed-flow persistence;
- termination and branch type;
- duration and path length.

This estimator checks whether recurrence exists without relying on temporal
warping.

### 13.4 Conditional HSMM

HSMM is not the primary discovery model.

It may be fitted only after A3 has independently established stable discrete
motifs and only to test whether within-excursion substructure benefits from
explicit duration modeling. It must:

- condition on extracted excursion intervals;
- use neutral states;
- obey a small frozen parameter budget;
- include censoring;
- beat the accepted continuous background plus motif baseline on held-out
  structural likelihood or compression.

Failure of conditional HSMM does not invalidate recurrent excursions. Success
does not revive the old `N -> S -> P -> R -> N` grammar.

## 14. Structural Model Selection

Candidate dictionaries are compared using:

- held-out medoid or dictionary reconstruction loss;
- description length or compression gain over the continuous background;
- cluster stability under block bootstrap;
- medoid and shapelet reproducibility;
- cross-date membership entropy;
- maximum single-date support share;
- OOD rate;
- sensitivity to the allowed duration range;
- structural-null separation;
- parameter count and effective sample support.

Compactness on the training set alone is never sufficient.

The accepted dictionary is the smallest one that improves frozen held-out
structural criteria and passes recurrence and transport gates.

## 15. A4: Cross-Session Transport

Freeze the dictionary on development plus blocked validation data, then apply
it without refitting to historical replay dates.

Report:

- motif count and exposure-normalized rate by date;
- nearest-medoid distance by date;
- OOD fraction by date;
- motif profile drift;
- duration and branch distributions;
- prefix-recognition coverage;
- maximum support contribution from one date;
- classifier accuracy for predicting session identity from motif features.

A motif is not transportable if it primarily identifies capture date,
connection epoch, source cadence, or a one-off quality artifact.

Historical replay is an anti-overfitting check, not prospective confirmation.

## 16. Structural Null Suite

Each accepted motif must outperform matched nulls that preserve simpler
properties while destroying the proposed structure.

Required nulls:

```text
temporal_block_permutation:
  preserve marginal channels and local blocks, destroy excursion order

channel_block_shift:
  preserve each channel path, destroy cross-channel synchronization

level_identity_permutation:
  preserve values, destroy L1-L5 spatial propagation

side_orientation_disruption:
  preserve activity, destroy coherent bid/ask geometry

time_reversal:
  test whether onset-to-termination ordering matters

change_point_rate_matched_surrogate:
  preserve onset rate and duration support, destroy repeated shapes

session_identity_null:
  test whether motifs are session fingerprints
```

Null generation must preserve quality masks, admitted exposure, censoring and
the relevant marginal distribution. Easy white-noise nulls are insufficient.

Multiplicity control applies across:

- candidate `K`;
- detector families;
- representation scales;
- distance variants;
- shapelet lengths;
- null families.

The correction method and hypothesis family are frozen in A0.

## 17. A5: Causal Online Recognition

Retrospective motif membership is not enough. For each frozen motif, construct
an online recognizer using only the observed prefix:

```text
P(M_k | X_(<=t), onset_detected_by_t)
```

The recognizer may use:

- distance from the observed prefix to medoid prefixes;
- detected shapelet primitives;
- residual energy path;
- elapsed duration;
- causal branch evidence;
- quality and OOD masks.

It may not use:

- retrospective end time;
- total duration;
- future warping path;
- future peak;
- future return-to-background label.

Report:

- onset delay;
- motif decision delay;
- fraction of motif duration elapsed at first decision;
- prefix precision and coverage;
- false-entry rate during background;
- label revision rate;
- remaining-duration calibration;
- early OOD rate;
- performance by date and duration bucket.

The primary online result must distinguish:

```text
early_recognizable
late_only_recognizable
retrospective_only
not_recognizable
```

## 18. Quantitative Gates

Exact numeric thresholds must be frozen in the A0 support supplement before
motif fitting. They may use support, effective sample size, null simulation
and model complexity, but never Track B outcomes.

### Gate A0: admissibility

- source and reconstruction identities close;
- session roles are immutable;
- complete blocks support the declared parameter budget;
- gaps and censoring are auditable;
- forbidden outcome surfaces are absent.

### Gate A1: background adequacy

- the selected background beats persistence or unconditional baselines on
  blocked held-out data;
- coefficients and residual scales remain stable enough for replay;
- residual tails are not explained primarily by quality failures;
- no-refit replay remains within calibrated OOD limits.

### Gate A2: excursion support

- onset and termination rules produce enough uncensored excursions across
  independent dates;
- recurrence is not created by one date or one connection epoch;
- duration support covers the model budget;
- detector agreement exceeds matched chance levels;
- candidate rate remains operationally finite.

### Gate A3: recurrent motif evidence

- the accepted dictionary improves held-out compression or reconstruction
  over the continuous-background-only baseline;
- medoids or shapelets are stable under blocked resampling;
- motifs beat all required matched structural nulls after multiplicity
  control;
- no accepted motif is dominated by a single date;
- OOD is explicitly retained rather than forced into clusters.

### Gate A4: transport

- frozen motifs recur on historical no-refit replay;
- support, distance and OOD remain within A0-frozen tolerances;
- motif features do not primarily identify session identity;
- spatial and temporal structure both contribute beyond their nulls.

### Gate A5: online recognizability

- at least one transported motif is recognized from a causal prefix before
  most of its duration has elapsed;
- background false-entry and label-revision rates remain within frozen caps;
- recognition survives date-blocked replay;
- a final positive claim remains capped pending prospective sessions.

Failure at a gate stops later Track A stages unless a new versioned protocol
is approved. The failed run remains the primary result for that protocol.

## 19. Failure Classifications

Track A must end with exactly one primary classification:

```text
insufficient_structural_support

continuous_background_no_recurrent_excursions

recurrent_excursions_not_null_distinct

session_specific_excursions_only

recurrent_excursions_not_transportable

retrospective_excursions_not_online_recognizable

historical_support_only_pending_prospective

stable_recurrent_structural_excursions_online_recognizable
```

The last classification requires prospective validation. Existing historical
data alone cannot produce it.

## 20. Track B Unlock Contract

Track B remains locked unless Track A produces at least:

```text
historical_support_only_pending_prospective
```

and a separate review decides whether prospective confirmation is sufficient
to inspect outcomes.

The Track A handoff contains only:

- frozen motif dictionary;
- causal recognizer;
- onset and prefix timestamps;
- neutral motif IDs;
- confidence and OOD flags;
- duration and branch information observable by each timestamp;
- complete provenance and model hashes.

Future price, markout, fill and PnL fields are joined only in a separately
authorized Track B task after Track A artifacts are frozen.

## 21. Required Artifact Layout

```text
artifacts/skhynix_continuous_excursion_track_a/
  contracts/
    source_manifest.json
    session_role_ledger.csv
    feature_contract.json
    outcome_blind_access_ledger.json
    gate_contract.json
  support/
    admitted_intervals.csv
    quality_summary.csv
    effective_sample_support.csv
  background/
    model_spec.json
    coefficients.parquet
    validation_metrics.csv
    residual_diagnostics.csv
  changepoints/
    detector_spec.json
    candidate_onsets.parquet
    detector_agreement.csv
  excursions/
    excursion_registry.parquet
    termination_summary.csv
    censoring_summary.csv
    duration_support.csv
  motifs/
    dictionary_spec.json
    medoids/
    shapelets/
    assignments.parquet
    stability_metrics.csv
  nulls/
    null_manifest.json
    null_metrics.csv
    multiplicity_results.csv
  transport/
    replay_metrics.csv
    session_identity_diagnostics.csv
  online/
    recognizer_spec.json
    prefix_decisions.parquet
    online_metrics.csv
  reports/
    track_a_report.md
    failure_analysis.md
  classification.json
  run_manifest.json
```

Every artifact must carry source hashes, code commit, config hash, model hash,
timestamp semantics and role identity.

## 22. Parameter And Sample Control

Minute-scale support is limited even when grid-row count is large. The plan
therefore uses:

- a low-parameter continuous background as primary;
- small neutral dictionaries;
- medoids rather than unrestricted generative templates;
- bounded shapelet count and lengths;
- grouped or diagonal covariance before full covariance;
- date-blocked validation;
- effective samples based on calendar blocks and excursions;
- explicit censoring;
- minimum occurrences and minimum independent-date support per motif.

Parameter count must be reported for every estimator. A model is ineligible
when its effective independent support does not exceed the A0-frozen ratio,
even if numerical fitting succeeds.

## 23. Design Risks And Diagnostics

### Background absorbs the excursion

If the background adapts too quickly, genuine excursions disappear. Compare
frozen adaptation speeds and report residual-energy attenuation around
change-points.

### Background creates the excursion

If the background is misspecified, ordinary autocorrelation appears as
recurrent residual structure. Require held-out residual diagnostics and
background-family robustness.

### Warping manufactures similarity

Soft-DTW can over-align unrelated paths. Use bounded warping, retain duration
penalties and require segment-summary clustering agreement.

### Thresholds manufacture segment boundaries

Report sensitivity around frozen thresholds and compare detector families.
Do not optimize thresholds for cluster compactness.

### One session supplies a motif

Cap maximum date contribution and require independent-date recurrence.

### Quality artifacts masquerade as structure

Audit reconnect, source age, no-new-information and sequence state by motif.

### Retrospective shape is not online state

Keep retrospective and prefix metrics separate. A late-only recognizer cannot
support maker risk control at onset.

### Too many candidate families consume the sample

Freeze a small primary path and treat alternatives as robustness checks:

```text
primary:
  diagonal robust AR background
  multiscale causal CUSUM ensemble
  variable-length k-medoids
  bounded tensor + soft-DTW distance
  frozen structural null suite

secondary:
  low-rank ridge VAR
  Bayesian online change-point detector
  unsupervised shapelet dictionary
  segment-summary clustering

conditional only:
  within-excursion HSMM
```

## 24. First Execution Task

The next task should be:

```text
SKHYNIX-BINANCE-CONTINUOUS-EXCURSION-TRACK-A0-SUPPORT-AND-SURFACE-FREEZE
```

It must not fit motifs.

It should:

1. close source and reconstructed-cache hashes;
2. freeze session roles and admitted intervals;
3. freeze the 13-dimensional L1-L5 primary projection and richer robustness
   tensor;
4. measure multiscale complete-block and duration support;
5. freeze background candidates and parameter budgets;
6. freeze change-point detectors and segment termination rules;
7. freeze distance, warping, shapelet and null contracts;
8. publish numeric gates based only on support and matched null simulation;
9. verify that no outcome columns or joins are reachable.

Only after A0 QA acceptance may A1 background fitting begin.

## 25. Final Nonclaims

This plan does not claim:

- that recurrent excursions exist;
- that they form a closed cycle;
- that every excursion returns to background;
- that a motif predicts price direction or volatility;
- that a motif is profitable to trade;
- that historical replay is prospective evidence;
- that an offline motif can be recognized early enough for maker control;
- that Binance structural findings transfer to Hyperliquid or any other venue.

The purpose of Track A is to make those distinctions observable and
falsifiable while preserving the original alignment idea in a form consistent
with the historical evidence.
