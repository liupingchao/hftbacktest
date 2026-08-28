# SKHYNIX Binance Track A: Continuous Background, Interpretable M-State Filter, Competing-Risk Transition Hazard And H0/H1 Incremental Test - 2026-08-28

Date: 2026-08-28

Status: review draft; design-only research contract; not execution authority

Research identifier:

```text
continuous_background_interpretable_m_state_competing_risk_transition
```

Methodology dependency:

```text
docs/conditional_risk_research_methodology_kernel_v1.md
```

Supersedes:

```text
docs/skhynix_binance_continuous_background_recurrent_structural_excursions_track_a_plan_20260828.md
```

Authority boundary:

- This document designs one Binance public-data structural transition study.
- It grants no new collection, private endpoint, order, cancel, strategy,
  deployment, live-capital, maker economics, or Track B authority.
- Each execution stage requires a formal workflow task and frozen inputs.

## 1. Decision

Retain the broad alignment idea and the accepted continuous-background result,
but change the primary research route from unsupervised motif discovery to a
hypothesis-driven conditional transition test.

The superseded route was:

```text
continuous background
  -> generic segmentation
  -> DTW / k-medoids / shapelet discovery
  -> statistical filtering of discovered motifs
```

That route contains excessive researcher degrees of freedom:

```text
segment boundaries
feature projection
distance definition
warping path
cluster count
assignment radius
motif interpretation
```

In high-dimensional dependent data, this pipeline can almost always produce
retrospectively compact clusters. Cluster existence is therefore not accepted
as primary evidence.

The new primary route is:

```text
continuous background
  -> predeclared interpretable M-state
  -> predeclared mutually exclusive N transitions
  -> competing-risk transition hazard
  -> H0 versus H1 out-of-sample incremental test
```

DTW, k-medoids and shapelets are demoted to secondary diagnostics that may run
only after the primary conditional-transition gate passes.

## 2. Research Question

Track A asks:

> Conditional on the ordinary continuous market background and unavoidable
> context, does an interpretable state composed of pressure-side depth
> withdrawal, persistent aggressive flow and replenishment deficit change the
> probability and timing of subsequent recovery, propagation or reversal
> transitions across independent sessions?

This is a probabilistic state-transition hypothesis:

```text
{m1, m2, m3}
  ->
{n_recovery, n_propagation, n_reversal}
```

It is not a rigid waveform, fixed phase cycle, fixed 500ms response window, or
claim that every occurrence follows the same path.

## 3. Primary Falsifiable Hypothesis

Let:

```text
C_t = unavoidable observable background context at landmark time t
M_t = interpretable candidate state at t
T   = elapsed time from t to the first admitted N transition
J   = type of the first admitted N transition
```

The nested models are:

```text
H0:
  P(T, J | C_t, additive main effects of m1_t, m2_t, m3_t)

H1:
  P(T, J | C_t, additive main effects, joint M-state q_M(t))
```

The primary hypothesis is:

> Adding the frozen joint M-state to an H0 that already controls for context
> and the three component main effects produces a material,
> uncertainty-bounded and cross-session-stable improvement in held-out
> competing-risk prediction.

The study is rejected when H1 does not improve H0 by the frozen effect
threshold, even if individual coefficients, horizons or sessions appear
favorable.

## 4. Meaning Of Pattern

In this contract, a pattern means:

```text
an interpretable conditional transition law
```

not:

```text
a repeated geometric template
```

An accepted pattern may therefore have heterogeneous paths and durations. Its
stable object is:

```text
P(J = k, T <= u | C_t, M_t)
```

or the cause-specific hazard:

```text
lambda_k(u | C_t, M_t)
```

The pattern exists only if this conditional law transports out of sample and
beats the context-only law.

## 5. Inherited Evidence

The following accepted historical result remains in force:

```text
continuous_state_no_discrete_phase_support
```

It means:

- a closed `N -> S -> P -> R -> N` finite-state grammar was not supported;
- seconds-to-minutes HSMM duration extensions did not reverse the result;
- a low-parameter continuous AR baseline remained stronger;
- historical replay was not prospective evidence.

This negative result supports retaining a continuous background. It does not
prove that the M-to-N conditional transition proposed here exists.

Old N/S/P/R labels, HSMM assignments, medoids, transition matrices, durations
and prototypes may not initialize or tune this study.

## 6. Research Tuple

The execution supplement must freeze one tuple before target access:

```text
source:
  admitted Binance public depth and trade data

instrument:
  SKHYNIX research symbol under the existing source contract

decision time:
  causal reconstructed grid timestamp

information filtration:
  public observations available by decision time

landmark:
  broad side-oriented pressure activation

candidate M-state:
  withdrawal + flow persistence + replenishment deficit

target:
  first recovery, propagation or reversal transition

follow-up:
  support-selected elapsed-time range

censoring:
  horizon, capture, gap, reconnect, reset, quality and ambiguity

model comparison:
  additive-component H0 versus H0 plus frozen joint M-state H1

replication unit:
  session or research date

claim boundary:
  structural transition predictability only
```

Changing a load-bearing tuple member creates a new hypothesis version.

## 7. Information And Target Boundary

### 7.1 Information available at landmark time

H0 and H1 inputs may use only values available by `t`:

- current and trailing BBO;
- current and trailing L1-L5 depth;
- public depth updates and trades;
- causal add, cancel, depletion and replenishment estimates;
- spread, midpoint and microprice ending no later than `t`;
- signed trade and book-flow variables ending no later than `t`;
- trailing activity and realized movement ending no later than `t`;
- source age, sequence continuity, reconnect epoch and quality masks;
- frozen calendar and session context.

All preprocessing must be fitted on prior or training-role data.

### 7.2 Future structural targets permitted

After the support and hypothesis tuple are frozen, the target-construction
stage may inspect future public structural states solely to determine:

```text
transition type J
transition time T
censoring status
```

The future target is used for model fitting and scoring, never as a feature.

### 7.3 Forbidden outcomes

Track A may not use:

```text
post-transition return magnitude
future markout
future adverse selection beyond the frozen structural target
quote contact
own-order fill
spread capture
fees or rebates
inventory
PnL
optimal quote distance or size
maker action labels
```

Track A is not outcome-blind in the literal sense because it has a future
structural target. It is price-economics-blind beyond the exact frozen N-state
contract.

## 8. Data Roles

Before target access, every session receives an immutable role.

Recommended historical treatment:

| Dates | Role | Permitted use |
|---|---|---|
| 2026-07-29 | reconstruction and normalization calibration | support only |
| 2026-07-30, 2026-08-03, 2026-08-04 | historical development | state and estimator development |
| 2026-08-07, 2026-08-24, 2026-08-25 | blocked historical validation | frozen model comparison |
| 2026-08-26, 2026-08-27 | historical no-refit replay | final historical replay |
| newly collected sessions | prospective validation/final holdout | required for final positive claim |

All existing dates have already been inspected by earlier research. None may
be relabeled as a fresh prospective final holdout.

The exact inventory, roles and hashes must be regenerated in Stage A0.

## 9. Stage Chain

```text
A0 support and hypothesis-tuple freeze
  -> A1 continuous background and interpretable state construction
  -> A2 landmark risk set and structural target materialization
  -> A3 competing-risk H0/H1 incremental test
  -> A4 dependence nulls, cross-session transport and robustness
  -> A5 prospective confirmation and primary classification
```

No later stage may redefine an earlier surface after favorable targets become
visible.

## 10. A0: Support And Hypothesis-Tuple Freeze

A0 has zero target access and fits no transition model.

It must publish:

- exact source inventory and SHA identities;
- deterministic reconstruction and timestamp contract;
- accepted causal grid;
- L1-L5 availability and freshness;
- gap, reconnect, reset and censoring geometry;
- eligible calendar exposure;
- complete follow-up support at a predeclared log-spaced duration grid;
- session-role ledger;
- continuous-background candidates and parameter budgets;
- H0 context plus additive M-component family;
- M-state formulas and orientation;
- broad landmark and hysteresis rules;
- N-state formulas, precedence and ambiguity rules;
- maximum follow-up selection rule;
- hazard time basis and parameter budget;
- proper scoring metrics;
- effect-size and uncertainty gates;
- dependence-preserving null family;
- outcome-access ledger.

A0 may inspect:

- cadence;
- complete future coverage geometry;
- quality and censoring support;
- feature marginal support;
- independent calendar-block counts.

A0 may not inspect:

- N-state event rates;
- M-conditioned transition rates;
- favorable elapsed-time regions;
- H0 or H1 loss;
- model coefficients;
- effect sizes.

## 11. Causal Reconstruction And Continuous Background

Primary reconstruction remains:

```text
100ms causal grid
```

Coarser causal views may be used only under A0-frozen aggregation:

```text
200ms
500ms
1s
2s
5s
```

These are feature scales, not target horizons.

Let `Z_t` be the robustly normalized observable feature vector. Define:

```text
B_t = E[Z_t | Z_(<=t-1), quality_(<=t)]
R_t = Z_t - B_t
```

Primary background:

```text
low-parameter diagonal robust AR or grouped AR
```

Secondary robustness:

- low-rank ridge VAR;
- robust local-level state-space model;
- slower causal scale variant.

The background is accepted only if it beats persistence/unconditional
baselines on blocked held-out data and remains stable on no-refit replay.

The purpose of `R_t` is to express candidate M variables relative to ordinary
continuous market evolution. It is not used to generate unsupervised
segments.

## 12. Pressure-Side Orientation

Each landmark receives a pressure side:

```text
s_t in {bid_pressure, ask_pressure}
```

All candidate variables are transformed into pressure-relative coordinates so
that positive values have the same interpretation:

```text
positive withdrawal:
  liquidity removed from the pressure-facing side

positive flow persistence:
  aggressive flow continues toward that side

positive refill deficit:
  depletion exceeds replenishment on that side
```

The side-orientation rule must be frozen before N targets are visible.

When side orientation is ambiguous:

- the landmark is excluded under a frozen reason; or
- both orientations are retained only under a predeclared dependence rule.

The primary route may not choose the orientation that later produces the more
favorable transition.

## 13. Broad Landmark

The broad landmark defines the risk-set entry opportunity. It is intentionally
less specific than the candidate M-state.

Recommended landmark:

```text
causal activation of directional public flow or book pressure
```

The landmark:

- establishes side orientation;
- requires quality admissibility;
- uses a low threshold calibrated without N targets;
- uses entry/exit hysteresis;
- applies a frozen minimum separation or refractory rule;
- does not require all three M components.

This produces both:

```text
M-positive pressure landmarks
M-negative or weak-M pressure landmarks
```

H0 and H1 are therefore compared on a common pressure-activation risk set,
not on hand-matched episodes selected after outcomes are known.

## 14. Interpretable M-State

At each broad landmark `t_i`, define three pressure-oriented candidate
components.

### 14.1 m1: pressure-side depth withdrawal

`m1` measures whether displayed L1-L5 liquidity on the pressure-facing side
has fallen below its causal background while opposite-side and total-depth
context are controlled.

It must preserve:

- level identity;
- depth concentration;
- migration toward or away from the touch;
- causal trailing reference.

Positive `m1` means greater pressure-side withdrawal.

### 14.2 m2: aggressive-flow persistence

`m2` measures sustained pressure-oriented public trade and book-flow activity,
not a single trade or depletion crossing.

It may use a small A0-frozen bank of causal exponential summaries. The bank is
selected from cadence and support, not from transition outcomes.

Positive `m2` means stronger and more persistent directional pressure.

### 14.3 m3: replenishment deficit

`m3` compares pressure-side replenishment with depletion and cancellation over
the same causal information set.

Positive `m3` means:

```text
depletion and cancellation
  >
new displayed replenishment
```

### 14.4 M-state score

Robust standardized components are oriented so that larger values all mean
stronger candidate mechanism.

Primary continuous score:

```text
q_M(t) = min(
  clip(m1_t),
  clip(m2_t),
  clip(m3_t)
)
```

The minimum implements an interpretable soft conjunction: the joint state is
only as strong as its weakest required component.

Primary filter:

```text
M_positive(t) =
  q_M(t) >= h_M_entry
  under the frozen dwell and hysteresis rule
```

`q_M` is the primary model input. `M_positive` is used for reporting,
opportunity counts and online filter diagnostics.

No M threshold may be selected using N outcomes.

## 15. H0 Context

H0 represents context and component-wise persistence that any honest joint
pattern must beat.

The frozen low-parameter context includes the additive main effects:

```text
m1
m2
m3
```

and may include:

- spread;
- total and opposite-side depth;
- unsigned public activity;
- generic directional pressure used to define the landmark;
- causal trailing volatility/activity;
- time-of-day basis;
- source age and quality state;
- prior accepted landmark/transition history.

H0 may not include interactions or deterministic transforms that reconstruct
`q_M`.

H0 must be strong enough to prevent the candidate from receiving credit for
ordinary high-activity or wide-spread regimes, or for ordinary autocorrelation
in any individual M component.

## 16. Mutually Exclusive N Transitions

After a landmark, follow the oriented public state until the first admitted
transition.

### 16.1 n_recovery

Recovery occurs when:

- pressure-side depth and replenishment return to the frozen background band;
- the M-state exits through hysteresis;
- no pressure-direction propagation transition has occurred;
- no opposite-side reversal transition has occurred.

This includes absorption or normalization without pressure-direction price
relocation.

### 16.2 n_propagation

Propagation occurs when the pressure moves through the local structure before
recovery:

- pressure-side touch is exhausted or relocated;
- the public reference price moves in the pressure direction under the frozen
  quote convention;
- and the frozen spatial confirmation rule is satisfied, such as continued
  deeper-level depletion or migration.

The target is the first structural transition. Post-transition return
magnitude and markout are forbidden.

### 16.3 n_reversal

Reversal occurs when:

- the original directional pressure loses dominance;
- an opposite-side pressure state satisfying the frozen minimum conditions
  begins;
- recovery or propagation has not already occurred.

### 16.4 Persistence and no event

Persistence is not forced into a fourth event type.

If none of the three transitions occurs before maximum supported follow-up:

```text
right_censored_at_tau_max
```

Capture end, gap, reconnect, reset or quality failure also right-censors the
risk interval.

### 16.5 Precedence and ties

N-state definitions must be mutually exclusive.

When two candidates occur inside one grid interval:

- use exchange-event ordering only if the source contract identifies it;
- otherwise classify the interval as ambiguous and censor it;
- never apply a favorable semantic precedence after outcome inspection.

The exact precedence and ambiguity rules are frozen in A0.

## 17. Start And End Anchors

This route does not search for segment boundaries that maximize shape
similarity.

```text
start_anchor:
  first causal broad-landmark entry timestamp

M_entry_at:
  equal to start_anchor when M_positive is already true at the landmark;
  otherwise absent for that risk interval

end_anchor:
  first admitted N-transition timestamp

end_type:
  recovery, propagation or reversal
```

If no N transition occurs:

```text
end_anchor:
  absent

status:
  right-censored
```

Anchors are outputs of predeclared state predicates. DTW, clustering and
future economic outcomes cannot move them.

## 18. Risk-Set Construction

For landmark `i`:

```text
t_i     = landmark time
C_i     = frozen H0 context at t_i
q_M,i   = frozen M-state score at t_i
T_i     = time to first N event or censor
J_i     = event type, if observed
```

At elapsed-time bin `u`, a row remains at risk only when:

```text
T_i >= u
```

Dependence controls:

- one active primary risk interval per pressure side;
- no duplicate landmark while the same interval remains active;
- frozen refractory rule after terminal transition;
- opposite-side entry treated according to the reversal contract;
- no interval crosses capture, reconnect or quality boundaries;
- calendar/session blocks remain the uncertainty unit.

Raw hazard rows are not treated as IID observations.

## 19. Data-Determined Time Scale

The study does not choose one favorable fixed horizon.

A0 inspects only complete follow-up geometry on a log-spaced support grid, for
example:

```text
1s
2s
5s
10s
30s
60s
120s
300s
600s
900s
```

The primary `tau_max` is the largest predeclared support point satisfying
frozen complete-block and independent-session requirements before N-state
event rates are inspected.

The hazard baseline uses a small elapsed-time basis:

```text
alpha_k(u)
```

Recommended choices:

- 4-6 piecewise log-time bins; or
- a 3-5 degree-of-freedom monotone/restricted spline.

Parameter count is controlled against independent landmark and session
support. A visually favorable horizon cannot rescue a failed integrated
primary score.

## 20. Primary Competing-Risk Model

Use a discrete-time multinomial hazard because it:

- handles multiple first-event types;
- supports interval and right censoring;
- permits a flexible but low-parameter elapsed-time baseline;
- produces calibrated cumulative incidence;
- remains auditable and causally deployable.

For event `k` at elapsed time `u`:

```text
eta_H0,k(i,u)
  = alpha_k(u)
  + gamma_k' C_i
  + rho_k1*m1_i
  + rho_k2*m2_i
  + rho_k3*m3_i

eta_H1,k(i,u)
  = alpha_k(u)
  + gamma_k' C_i
  + rho_k1*m1_i
  + rho_k2*m2_i
  + rho_k3*m3_i
  + beta_k * q_M,i
```

Conditional event probability:

```text
P(J_i = k at u | T_i >= u)
  = exp(eta_k(i,u))
    /
    (1 + sum_j exp(eta_j(i,u)))
```

The denominator's `1` is no transition in the current elapsed-time bin.

H1 adds one primary degree of freedom per event cause beyond the additive
component model. This keeps the candidate test focused on whether the
predeclared conjunction matters beyond ordinary persistence of `m1`, `m2`
and `m3`.

## 21. Attribution And Robustness Models

For diagnostic decomposition, the study may also report:

```text
H_base:
  context only

H0:
  H_base + additive m1 + m2 + m3

H1:
  H0 + joint q_M
```

`H_base` shows whether the component family contains information at all.
Only `H0` versus `H1` tests the claimed interpretable joint pattern. A
favorable `H_base` versus `H0` result may not rescue a failed primary
`q_M` increment.

Additional robustness estimators:

- cause-specific logistic hazard;
- cause-specific regularized Cox model;
- Aalen additive hazard;
- grouped background variant;
- alternative A0-frozen M threshold;
- L1-L3 versus L1-L5 spatial projection.

No deep sequence model, unconstrained neural hazard or broad feature search is
part of the primary study.

## 22. Directional Expectations

The primary mechanism predicts:

```text
larger q_M
  -> higher propagation cumulative incidence
  -> lower or delayed recovery cumulative incidence
```

The reversal direction is exploratory unless a separate directional
hypothesis is frozen before target access.

Passing proper-score improvement with coefficients opposite to the frozen
mechanism does not authorize the original interpretation. It requires a new
versioned hypothesis or a negative/misspecified classification.

## 23. H0/H1 Model Fitting

Requirements:

- train periods strictly precede validation/replay periods where practical;
- preprocessing and normalization fit on training data only;
- purge and embargo around capture/session boundaries;
- regularization selected within development roles;
- H0 and H1 share the same risk rows, censoring and elapsed-time basis;
- H1 may differ from H0 only through the frozen M-state term;
- no-refit replay uses frozen coefficients and thresholds;
- session/date remains the replication and uncertainty unit.

Primary scoring uses all admitted elapsed-time bins up to `tau_max`, not the
best post-hoc horizon.

## 24. Primary Metrics

### 24.1 Proper scoring

- held-out competing-risk negative log loss;
- integrated Brier score;
- cause-specific Brier score;
- cumulative-incidence calibration;
- calibration slope/intercept;
- OOD and unsupported-row rate.

Primary increment:

```text
Delta_log_loss = loss_H0 - loss_H1

Delta_IBS = IBS_H0 - IBS_H1
```

Positive values favor H1.

### 24.2 Effect and transition metrics

- cause-specific `beta_k`;
- cause-specific cumulative incidence by frozen M-score groups;
- restricted mean transition time;
- recovery/propagation/reversal probability within `tau_max`;
- M-positive opportunity rate per eligible hour;
- event and censor counts by date;
- maximum single-date contribution.

### 24.3 Online filter metrics

- M-entry detection delay;
- M-state dwell before N transition;
- residual dwell after a declared online decision budget;
- false M-entry rate on low-pressure exposure;
- M-state revision/chatter rate;
- fraction censored before any actionable dwell.

Track A reports structural actionability only. It does not select a maker
action.

## 25. Uncertainty

Row-IID standard errors are forbidden.

Use:

- date/session-block bootstrap;
- dependence-preserving calendar blocks;
- leave-one-date-out influence analysis;
- confidence intervals over session-aggregated proper-score increments;
- event-count and effective-sample diagnostics.

The primary confidence interval and effect threshold are frozen before target
access.

A single favorable day or one large event cannot carry the claim.

## 26. Dependence-Preserving Nulls

Required nulls preserve simpler market structure while breaking the candidate
M-to-N relationship:

```text
within_session_block_shift:
  shift q_M relative to future N transitions by admissible calendar blocks

pressure_side_disruption:
  preserve activity and marginals, break correct directional orientation

M_component_desynchronization:
  preserve each m component, break their contemporaneous conjunction

matched_context_permutation:
  permute q_M only within frozen H0 context strata

time_reversal_diagnostic:
  test whether the proposed transition direction is asymmetric
```

For every null:

- reconstruct the same risk set and censoring;
- refit both H0 and H1 under the same procedure;
- evaluate the same best frozen primary metric;
- preserve session and quality boundaries.

Multiplicity covers the complete predeclared hypothesis family. Nulls may not
be selected after seeing which one is easiest to beat.

## 27. Gate Chain

Numeric thresholds must be frozen in the A0 execution supplement before
N-target access.

### Gate A0: data and support admissibility

- source and reconstruction identity close;
- session roles are immutable;
- complete follow-up geometry supports `tau_max`;
- parameter budget is supported by independent sessions and landmarks;
- gaps, ambiguity and censoring are auditable;
- target surfaces remain inaccessible.

### Gate A1: state-construction validity

- continuous background beats simple baselines;
- M components are causal and numerically stable;
- M score is not primarily a quality/session fingerprint;
- broad landmark and M filter have finite, auditable opportunity rates;
- orientation and hysteresis are deterministic.

This gate inspects M marginals, not M-conditioned N outcomes.

### Gate A2: target variation and identification

- each primary N state has adequate identified event support;
- censoring does not dominate the primary follow-up range;
- ties and ambiguous events remain below the frozen cap;
- no single date supplies the effective target variation.

If any primary cause fails the frozen support requirement, the three-cause
primary family is ineligible and yields an inconclusive classification.
Reduced-cause models are diagnostic only and require a new version before
they can become primary.

### Gate A3: H0/H1 incremental predictability

- H1 improves H0 on the frozen primary proper score;
- the blocked confidence lower bound exceeds the frozen materiality threshold;
- integrated Brier/calibration do not contradict the primary result;
- propagation/recovery directions are consistent with the frozen mechanism;
- no single date or elapsed-time bin carries the result.

### Gate A4: null separation and historical transport

- observed H1 increment beats all required dependence-preserving nulls after
  multiplicity control;
- frozen coefficients and state definitions transport to no-refit replay;
- session identity and quality artifacts do not explain the increment;
- sensitivity models do not reveal a load-bearing arbitrary threshold.

### Gate A5: prospective confirmation

- protocol-frozen prospective sessions reproduce the proper-score increment;
- effect direction, calibration and opportunity rate remain within tolerance;
- the final holdout is not used to revise the same hypothesis version.

Only Gate A5 can produce the strongest positive classification.

## 28. Pass Semantics

An accepted conditional transition pattern means:

```text
the frozen joint M-state changes the probability/timing distribution
of the frozen N transitions beyond context and additive component effects
on independent sessions
```

It does not mean:

- all M occurrences reach the same N state;
- the path between M and N is geometrically similar;
- transition probability is deterministic;
- the pattern predicts post-transition return magnitude;
- the pattern is economically tradable;
- maker intervention has positive value.

## 29. Failure Classifications

Track A ends with exactly one primary classification:

```text
insufficient_transition_support

continuous_background_m_state_not_stable

target_transition_not_identified

target_variation_insufficient

H1_no_increment_over_H0

increment_driven_by_single_session_or_horizon

increment_not_dependence_null_distinct

historical_transition_increment_not_transportable

structural_increment_not_online_actionable

historical_support_only_pending_prospective

stable_interpretable_conditional_transition_confirmed
```

The last classification requires prospective confirmation.

## 30. Secondary Pattern Diagnostics

Only after Gate A3 passes may the study examine residual heterogeneity within
the accepted M-to-N family.

Permitted secondary diagnostics:

- aligned residual plots by N event;
- variable-length medoids;
- bounded soft-DTW;
- interpretable shapelets;
- conditional subgroups;
- anchor-perturbation robustness.

These diagnostics:

- cannot change M or N definitions;
- cannot rescue a failed H0/H1 test;
- cannot create a stronger primary classification;
- may generate a separately versioned successor hypothesis.

## 31. Track B Unlock Contract

Track B remains locked unless Track A reaches at least:

```text
historical_support_only_pending_prospective
```

and the controller separately decides whether prospective evidence is
sufficient.

The frozen handoff contains:

- background model;
- broad landmark and orientation rule;
- M component formulas and score;
- N state definitions;
- hazard model and coefficients;
- causal decision timestamps;
- transition/censoring ledger;
- calibration and OOD state;
- source, config, model and code hashes.

Track B may then define a new tuple for markout, contact, fill, maker risk or
economics. Those outcomes do not retroactively tune Track A.

## 32. Required Artifact Layout

```text
artifacts/skhynix_interpretable_transition_track_a/
  contracts/
    source_manifest.json
    session_role_ledger.csv
    research_tuple.json
    information_contract.json
    M_state_contract.json
    N_state_contract.json
    censoring_contract.json
    gate_contract.json
    outcome_access_ledger.json
  support/
    admitted_intervals.csv
    quality_summary.csv
    followup_support.csv
    effective_sample_support.csv
  background/
    model_spec.json
    coefficients.parquet
    validation_metrics.csv
    residual_diagnostics.csv
  states/
    landmark_ledger.parquet
    M_state_ledger.parquet
    M_marginal_summary.csv
    orientation_diagnostics.csv
  targets/
    transition_ledger.parquet
    censoring_summary.csv
    target_variation.csv
    ambiguity_summary.csv
  models/
    H0_spec.json
    H1_spec.json
    H0_coefficients.parquet
    H1_coefficients.parquet
    validation_predictions.parquet
  metrics/
    proper_scores.csv
    cumulative_incidence.csv
    calibration.csv
    session_influence.csv
    online_filter_metrics.csv
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

Every artifact must carry source hashes, code commit, config hash, model hash,
timestamp semantics and session role.

## 33. Parameter And Sample Discipline

The effective sample is measured by independent landmarks, events and
sessions, not 100ms hazard rows.

Required controls:

- one primary M score;
- one primary H0 context plus additive-component family;
- one primary elapsed-time basis;
- three mutually exclusive causes;
- one added H1 coefficient per cause;
- grouped or diagonal background before high-dimensional alternatives;
- date-blocked validation;
- explicit censoring;
- minimum event-per-parameter and session-support ratios;
- parameter count reported for every model.

When support is insufficient, the valid result is
`insufficient_transition_support`, not a more flexible model.

## 34. Design Risks

### M-state chosen after seeing N

Prevent with support-before-target access, immutable formulas and access
ledger.

### H0 intentionally weak

Prevent with a predeclared context family containing generic activity,
spread, depth and trailing movement.

### Favorable horizon selection

Use support-selected `tau_max`, integrated proper scores and a frozen
elapsed-time basis.

### Dense overlapping landmarks

Use one active interval per side, refractory rules and session-block
uncertainty.

### Transition labels overlap

Freeze mutually exclusive predicates, precedence and ambiguity censoring.

### State threshold creates the result

Use continuous `q_M` as the primary model input; threshold is a filter and
diagnostic. Run only A0-frozen threshold sensitivity.

### Price relocation leaks economics

Use relocation only as the exact `n_propagation` structural endpoint. Forbid
post-transition magnitude, markout and strategy labels.

### Rare events create large coefficients

Require event support, regularization, blocked confidence intervals and
proper-score improvement.

### Historical data appear prospective

Cap all existing dates at historical replay and require newly collected
protocol-frozen sessions for final confirmation.

## 35. First Execution Task

The next formal task should be:

```text
SKHYNIX-BINANCE-INTERPRETABLE-TRANSITION-TRACK-A0-SUPPORT-AND-TUPLE-FREEZE
```

It must have zero N-target access.

It should:

1. close source and reconstructed-cache identities;
2. freeze session roles and admitted intervals;
3. freeze causal grid, background and normalization;
4. freeze broad landmark and pressure-side orientation;
5. freeze exact `m1`, `m2`, `m3`, `q_M` and hysteresis formulas;
6. freeze mutually exclusive N predicates, tie and censoring rules without
   materializing their event rates;
7. measure complete follow-up geometry and select `tau_max` mechanically;
8. freeze H0 context plus additive M-component main effects, H1 joint `q_M`
   increment and elapsed-time basis;
9. freeze proper-score, uncertainty, null and materiality gates;
10. verify zero access to target rates, transition-conditioned plots and
    economics outcomes.

Only after A0 QA acceptance may target materialization begin.

## 36. Final Nonclaims

This plan does not claim:

- that the proposed M-state is stable;
- that M changes any N transition;
- that the joint effect survives H0 context and component main effects;
- that the transition time scale is already known;
- that historical data provide prospective confirmation;
- that a structural transition predicts post-transition return;
- that a maker can act before the state exits;
- that any quoting response is profitable;
- that the result transfers to another venue or instrument.

Its purpose is to test one interpretable conditional transition family with
fewer researcher degrees of freedom and a clear nested baseline.
