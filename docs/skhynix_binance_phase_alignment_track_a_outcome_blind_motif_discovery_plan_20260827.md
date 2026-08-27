# SKHYNIX Binance Phase Alignment Track A: Outcome-Blind Motif Discovery Plan - 2026-08-27

Date: 2026-08-27

Status: review draft; design-only research contract; not execution authority

Related documents:

```text
docs/skhynix_trigger_aligned_episode_research_implementation_plan.md
docs/skhynix_continuous_hazard_maker_research_framework_v2.md
docs/skhynix_post_h0b_future_research_path_20260825.md
docs/conditional_risk_research_methodology_kernel_v1.md
docs/binance_single_venue_maker_reusable_assets_and_framework_20260825.md
```

Authority boundary:

- This document designs Track A only.
- It grants no data collection, private endpoint, order, cancel, strategy,
  deployment, live-capital, Track B outcome access, or Track C economics
  authority.
- Existing SKHYNIX sessions may be used only under explicitly assigned
  historical-development roles.
- A new formal workflow task, independently reviewed execution contract, and
  separately authorized data access are required before implementation.

## 1. Route Reset

The earlier event-alignment route used a single queue-depletion trigger as an
event-time origin:

```text
one queue-depletion trigger
  -> align the next fixed response window
```

Accepted density evidence showed that this construction is unsuitable for
SKHYNIX. Trigger candidates are too dense, response windows overlap, and
multiple candidate crossings frequently belong to one continuous order-flow
run.

The successor research object is a phase trajectory:

```text
normal liquidity
  -> stress onset
  -> peak pressure / price discovery
  -> liquidity recovery
  -> normalized liquidity
```

For compact notation, this candidate ontology is written:

```text
N -> S -> P -> R -> N
```

This sequence is a hypothesis, not a required output. Track A must be able to
reject it.

## 2. Track A Purpose

Track A asks one structural question:

> Using only Binance public order-book and trade-flow information observable
> by time `t`, does SKHYNIX exhibit recurrent, cross-session-stable,
> finite-duration market phases and transition grammars that can be recognized
> causally before the phase has ended?

Track A does not ask whether a phase predicts returns, markout, fills, maker
profitability, or an optimal quoting posture. Those questions belong to later
tracks.

The strongest Track A result is:

```text
stable_cross_session_phase_grammar_online_recognizable
```

This result only authorizes a separately reviewed Track B response study.

## 3. Why Track A Is A Standalone Go/No-Go Study

Track A is not a preprocessing step whose output is assumed to exist. It must
be allowed to stop the program.

Failure modes that must remain valid conclusions include:

```text
continuous_state_no_discrete_phase_support
stable_states_but_no_recurrent_transition_grammar
session_specific_motifs_only
retrospective_motifs_not_online_recognizable
insufficient_structural_support
```

Track B and Track C must remain locked for all of these classifications.

## 4. Outcome-Blind Boundary

### 4.1 Allowed information

At grid time `t`, Track A may consume only values observable by `t`:

- current and historical Binance BBO;
- current and historical Binance top-N depth;
- current and historical depth updates;
- current and historical public trades;
- current and trailing spread, midpoint, microprice and book-shape state;
- current and trailing signed flow, add, cancel, depletion and replenishment;
- current and trailing realized movement, calculated strictly from timestamps
  no later than `t`;
- source age, sequence state, connection epoch and quality masks;
- session and segment identity used for splitting, blocking and quality
  control.

### 4.2 Forbidden information

Track A may not construct, read, rank, visualize, or select models using:

```text
future return
future midpoint or BBO movement
future realized volatility
future adverse move-through
future markout
future quote contact
future fill or order lifecycle
future spread capture
future PnL
Track B response labels
Track C economics labels
```

No `t+h` join is permitted in Track A.

### 4.3 Naming discipline

Learned states are initially named:

```text
Q0, Q1, ..., Q(K-1)
```

They may not be named `normal`, `stress`, `peak`, `recovery`, `safe`,
`toxic`, `profitable`, or similar terms until the neutral state model and
state-profile tables are frozen.

Semantic names must be assigned from contemporaneously observable state
profiles only. Future response behavior may not influence state naming.

## 5. Data Roles And Consumption Lifecycle

Before any Track A model fitting, every session must receive one immutable
role:

```text
historical_method_development
historical_structural_replay
prospective_structural_validation
prospective_structural_final_holdout
```

Recommended initial role treatment:

- Jul30 and Aug03: historical method development;
- Aug04 and Aug07: historical structural replay only;
- newly collected Binance-only sessions: prospective structural validation
  and final holdout.

The historical sessions have already been inspected in earlier research. They
cannot be promoted to a fresh Track A final holdout.

Track A outcome-blind status does not make an old session prospectively fresh.

## 6. Track A Stage Chain

Track A is divided into six separately reviewable stages:

```text
A0 support and data admission
  -> A1 causal state representation
  -> A2 neutral phase segmentation
  -> A3 maximal-run and motif discovery
  -> A4 causal online recognition
  -> A5 prospective structural validation and classification
```

No later stage may silently repair or redefine an earlier frozen surface.

## 7. Stage A0: Support And Data Admission

A0 reads no future response and fits no phase model.

It must publish:

- exact source inventory and SHA identities;
- symbol, market type, tick size and quantity semantics;
- snapshot and incremental-depth sequence continuity;
- accepted book depth, initially top 5 unless support justifies another fixed
  depth;
- source cadence and source-age distributions;
- gap, reconnect, reset and unavailable-state intervals;
- eligible calendar exposure;
- complete calendar and dependence blocks;
- candidate grid support;
- session-role ledger;
- outcome-blind access ledger.

Candidate calendar grids may include:

```text
10ms
20ms
50ms
```

The primary grid must be selected mechanically from cadence, freshness,
no-new-information rate, complete-block support and deterministic replay. It
may not be selected from phase compactness, future response, model loss, or
economic convenience.

Event-arrival sampling may be published only as a robustness view. It may not
replace equal calendar-time exposure.

### 7.1 Observation resolution is not phase timescale

The selected calendar grid defines only the minimum observation resolution at
which the public market state can be reconstructed consistently. It does not
define:

```text
phase duration
cycle duration
episode window
maximum response horizon
required number of grid steps per phase
```

Track A must not pre-register a fixed window such as `500ms`, `2s` or `10s`
inside which N/S/P/R is expected to complete.

The research hypothesis is instead:

> If recurrent phase structure exists, the market data will reveal its state
> dwell, transition and complete-cycle timescale distributions.

The allowed relationship is:

```text
data-supported observation grid
  -> variable-length state estimation
  -> variable-length maximal runs
  -> data-discovered phase and cycle timescales
```

The forbidden relationship is:

```text
researcher-selected episode window
  -> force every candidate path into that window
```

Track A must allow:

- no identifiable phase timescale;
- one dominant timescale;
- state-specific timescales;
- the same transition grammar at multiple timescales;
- session-specific timescale drift;
- a continuous process with no stable phase boundary.

## 8. Stage A1: Causal State Representation

### 8.1 State tensor

For every eligible grid endpoint `t`, construct:

\[
Z_t = f(\mathcal F_t^{Binance})
\]

where every input belongs to the Binance public filtration observed by `t`.

The primary representation preserves both time and book-level space:

\[
Z_t(\tau, l, c)
\]

with:

```text
tau = trailing time coordinate
l   = relative book level, initially L1-L5
c   = observable state channel
```

### 8.2 Primary channel families

The initial allowlist is:

```text
book geometry
  bid/ask relative price by level
  bid/ask normalized quantity by level
  depth concentration and slope
  spread and level spacing

book dynamics
  add quantity by side and level
  cancel quantity by side and level
  depletion velocity by side and level
  replenishment velocity by side and level

trade flow
  aggressive buy/sell quantity
  signed trade-flow imbalance
  trade count and intensity
  touch and strict-through public quantity

price-relative state
  midpoint change through t
  microprice displacement through t
  trailing realized movement through t

quality state
  source age
  no-new-information mask
  sequence continuity
  connection/segment epoch
  market-view quality mask
```

The tensor must retain per-level values. Top-level imbalance or one scalar OFI
may be included as summaries, but they may not replace the spatial surface.

### 8.3 Causal normalization

Normalization must be frozen before state discovery:

- prices are represented relative to the current midpoint and in instrument
  ticks;
- quantities use a prior-only rolling robust scale derived from accepted book
  depth;
- no global full-session mean or variance may be used;
- bullish and bearish orientations use a frozen side-mirroring rule;
- no value carries across segment, sequence, reconnect, source-gap or quality
  boundaries;
- every derived value carries `observed_at`, source identity, calculation
  version, source age and availability status;
- missing values and no-new-information intervals remain explicit.

### 8.4 Direction orientation

Direction may not be selected from future price movement.

The primary side orientation must use a frozen observable pressure rule, such
as a combination of signed trade flow and same-side book withdrawal, evaluated
only through `t`.

Rows without a sufficiently identified orientation remain neutral or OOD.
They may not be assigned the favorable side after observing the completed run.

### 8.5 Multi-resolution causal feature bank

An unknown phase timescale cannot be discovered from a representation that
contains only one researcher-selected trailing window.

The primary state representation must therefore include:

```text
instantaneous/base-grid state
+ a bounded multi-resolution bank of trailing causal summaries
```

The candidate lookbacks or exponential half-lives must be generated
mechanically from:

- the accepted observation grid;
- measured channel cadence and freshness;
- uninterrupted complete-block support;
- a bounded approximately logarithmic spacing rule;
- compute and memory feasibility.

They may not be selected from motif compactness, cycle completion rate, future
response or economic performance.

Every trailing feature remains an observation operator, not a proposed phase
duration. The HSMM may use evidence from several resolutions simultaneously,
and Track A must report whether inferred state boundaries and duration modes
survive removal of the shortest and longest feature scales.

If the inferred grammar appears only under one narrow lookback choice and does
not transport across adjacent accepted scales, it is classified as
scale-specific exploratory structure rather than stable phase support.

## 9. Stage A2: Neutral Phase Segmentation

### 9.1 Primary model family

The primary estimator is an interpretable multivariate Student-t emission
sticky hidden semi-Markov model:

```text
causal interpretable observation vector
  -> multivariate Student-t emission sticky HSMM
  + explicit state duration
  + neutral state identifiers
  + blocked held-out structural scoring
```

The sticky transition prior limits state flicker. Explicit duration prevents
one continuous run from becoming hundreds of event samples.

The HSMM must not hard-code:

```text
N -> S -> P -> R -> N
```

It first estimates neutral states `Q0, ..., Q(K-1)` and their unconstrained
outcome-blind transition graph. The N/S/P/R ontology is tested only after the
neutral model, duration model and state profiles are frozen.

Neural encoders, autoencoders and unconstrained deep sequence models are not
primary Track A estimators.

### 9.2 Emission and duration contract

The canonical spatial tensor remains the source representation, but the HSMM
emission layer consumes a reviewed compact observation vector derived from it.
The projection must:

- remain causal and outcome-blind;
- preserve separate L1-L5 depth, add, withdrawal, depletion and replenishment
  surfaces rather than replacing them with one aggregate imbalance;
- preserve signed trade-flow, spread, microprice displacement and quality
  state;
- use prior-only robust scaling;
- be fixed before state-count or emission-family comparison;
- publish an exact field-to-tensor provenance map.

For state `k`, the primary emission family is:

\[
X_t \mid Q_t=k \sim
t_{\nu_k}(\mu_k,\Sigma_k)
\]

where:

- `X_t` is the frozen interpretable observation vector;
- `mu_k` is the state profile;
- `Sigma_k` is diagonal or reviewed block-diagonal covariance;
- `nu_k` is a bounded or shared degrees-of-freedom parameter;
- no unrestricted full covariance is allowed unless support and numerical
  stability are demonstrated out of sample.

Student-t emissions are primary because order-book and trade-flow features are
heavy-tailed and contain genuine bursts that must not automatically create a
new state merely because a Gaussian emission treats them as extreme
outliers.

The primary parametric duration family is a shifted negative-binomial
distribution in calendar-grid steps:

\[
D_k \sim 1 + \operatorname{NegBin}(r_k,p_k)
\]

Its scale parameters are estimated from eligible market data rather than fixed
from a desired cycle horizon. A geometric duration is retained only as the
memoryless baseline.

Implementation may require finite computational support. Any truncation must:

- be derived only from uninterrupted block support, memory limits and
  deterministic replay feasibility;
- be materially wider than the dwell region observed only in
  `historical_method_development` fits before prospective roles are opened;
- treat boundary hits as censored or overflow observations;
- publish sensitivity to wider support;
- never be interpreted as the market's maximum phase duration.

Segment, sequence, reconnect, quality and collection-end boundaries create
explicit right censoring. They do not imply that the latent phase naturally
ended at the boundary.

A reviewed flexible discrete-hazard or duration-histogram estimator must be
included as a robustness model. If it reveals stable multimodal dwell
distributions that the negative-binomial family cannot represent, the
parametric HSMM must not collapse those modes into one artificial timescale.

The execution supplement must freeze:

```text
emission field list
emission covariance blocks
degrees-of-freedom treatment
duration family and support
computational duration support and overflow treatment
initial-state treatment
transition-prior strength
numerical convergence criteria
```

### 9.3 Mandatory model baselines

The Student-t sticky HSMM is accepted only if it improves structural evidence
over all frozen simpler alternatives:

```text
Gaussian sticky HSMM
Student-t memoryless HMM
continuous autoregressive or state-space model
single-state heavy-tailed null
```

The comparison must use blocked held-out structural likelihood, predictive
one-step density, duration calibration, state-profile stability,
transition-graph stability, session concentration and OOD rate.

If the continuous autoregressive/state-space baseline explains held-out
structure as well as or better than the discrete models, Track A must classify:

```text
continuous_state_no_discrete_phase_support
```

It may not retain the HSMM merely because discrete phase plots are easier to
interpret.

### 9.4 Candidate state counts

A reviewed execution supplement must freeze a bounded candidate set, for
example:

```text
K in {3, 4, 5, 6, 7, 8}
```

State count may be selected only using outcome-blind structural criteria:

- blocked held-out structural likelihood;
- minimum-description-length or another frozen complexity penalty;
- state-profile stability;
- duration identifiability;
- transition-graph stability;
- cross-session occupancy and concentration;
- blocked-bootstrap label stability;
- prospective structural replay performance.

No future-response metric may break a tie.

### 9.5 Offline and online estimands

Track A must distinguish:

```text
offline smoother:
  P(Q_t | Z_1, ..., Z_T)

online filter:
  P(Q_t | Z_1, ..., Z_t)
```

The offline smoother tests whether a stable structural segmentation exists.
It is not a deployable recognizer.

Only the online filter may be handed to Track B as decision-time phase state.

## 10. Stage A3: Maximal Runs And Motif Discovery

### 10.1 Maximal-run rule

A motif sample is a maximal connected phase run, not a grid row and not every
threshold crossing.

Each run must have:

```text
run_id
session_id
segment/epoch identity
run_start
neutral state sequence
state entry and exit times
state duration vector
orientation
quality and censoring state
reset reason
```

Adjacent runs may be merged only under a frozen hysteresis and reset contract.

Overlapping sliding windows must not be counted as independent motifs.

### 10.2 Transition grammar

The neutral state sequence is mined for recurrent transition paths:

```text
Q0 -> Q2 -> Q5 -> Q3 -> Q0
Q0 -> Q2 -> Q0
Q1 -> Q4 -> Q3
```

The transition graph and motif family are learned without future-response
labels.

Cycle discovery therefore has two distinct estimators:

```text
sticky HSMM
  -> estimates neutral state and duration sequence

maximal-run grammar miner
  -> estimates recurrent state paths and their cross-session support
```

The grammar miner operates on maximal runs, not grid rows. It must publish path
support by session, duration vectors, censoring, maximum session share,
transition-block null frequency and cross-channel-shift null frequency.

The same neutral transition path may occur with different dwell vectors and
complete-cycle durations. Track A must first report the joint distribution:

\[
\left(D_{Q_0},D_{Q_2},D_{Q_5},D_{Q_3},D_{\text{cycle}}\right)
\]

and then test whether it supports:

```text
one recurrent timescale family
multiple recurrent timescale families
continuous duration variation
session-specific duration only
no stable duration support
```

Any fast/medium/slow labels are post-estimation reporting labels defined from
frozen duration-distribution features. They may not be preselected windows
used to create the runs.

The framework must permit:

- aborted stress paths;
- direct normalization;
- repeated stress/peak loops;
- recovery failure;
- OOD transitions;
- no identifiable cycle.

### 10.3 Within-grammar prototypes

Runs with the same or compatible transition grammar may be grouped into a
small number of interpretable prototypes using:

```text
multichannel k-medoids
or
reviewed soft-DTW distance over the structural tensor
```

Examples of possible prototypes, not guaranteed outputs:

- L1 depletion followed by L2-L5 withdrawal;
- L1 depletion absorbed by deeper same-side liquidity;
- depletion followed by same-price replenishment;
- depletion followed by price-level migration;
- two-sided liquidity withdrawal followed by symmetric recovery.

Prototype selection uses recurrence, compactness, stability and cross-session
support only.

Soft-DTW or full-run distance may be used for retrospective structural
discovery. Track B recognition must use the frozen causal online filter or a
separately frozen prefix-only matcher.

## 11. Stage A4: Causal Online Recognition

### 11.1 Online phase filter

For each offline structural transition, A4 measures:

```text
t_offline_entry
t_online_detect
detection_delay
false_entry_count
state_flicker_count
remaining_structural_dwell
online posterior at detection
online OOD status
```

The online recognizer must use only observations received by
`t_online_detect`.

The primary recognizability question is:

> Does the online filter identify the phase while a material portion of the
> structural run remains, rather than only after offline hindsight has revealed
> the completed path?

Recognition thresholds, debounce, hysteresis and posterior persistence must be
fit on past structural blocks only.

The deployable state estimate is the HSMM forward filter:

\[
\pi_t(k,d)=P(Q_t=k,\text{current dwell}=d\mid X_1,\ldots,X_t)
\]

It must expose both state posterior and duration/hazard information. The
offline smoother or completed run may provide evaluation labels, but it may
not provide an online detector input.

### 11.2 Causal transition-prefix detector

The cycle detector is not a second unconstrained clustering model. It is a
frozen finite-state prefix matcher over an accepted neutral transition
grammar, driven by the online HSMM posterior.

Before semantic mapping, it operates on neutral paths such as:

```text
Q0
Q0 -> Q2
Q0 -> Q2 -> Q5
Q0 -> Q2 -> Q5 -> Q3
Q0 -> Q2 -> Q5 -> Q3 -> Q0
```

After the neutral state-profile mapping is frozen, the same prefixes may be
rendered as:

```text
N
N -> S
N -> S -> P
N -> S -> P -> R
N -> S -> P -> R -> N
```

A prefix may advance only when all frozen conditions pass:

```text
minimum state posterior
minimum posterior persistence
allowed neutral-state transition
duration support or transition hazard
debounce and hysteresis
accepted source quality and freshness
not OOD
no reset boundary
```

No absolute cycle-completion timeout may be selected from intuition or economic
convenience. If a detector timeout is required operationally, it must be
derived from past-only fitted duration tails, remain separate from the
structural definition of a completed cycle, and publish timeout sensitivity.
Crossing that timeout produces a censored or unresolved path, not evidence
that the phase or cycle ended.

The detector must emit decision-time events rather than waiting for a complete
cycle:

```text
phase_entry_candidate
phase_entry_confirmed
transition_prefix_advanced
transition_prefix_aborted
recovery_failed
cycle_completed
online_ood
detector_reset
```

It must preserve incomplete and contradictory paths, including:

```text
N -> S -> N
N -> S -> P -> N
N -> S -> P -> S
N -> S -> P -> R -> P
N -> OOD
```

The detector may not reinterpret a skipped or failed phase as a complete cycle.
Thresholds and state-machine rules must be frozen using past structural blocks
and replayed without refitting on prospective sessions.

### 11.3 Online cycle metrics

In addition to state-level recognition metrics, A4 must publish:

- prefix precision and false-entry rate by session;
- time from offline phase entry to online prefix advancement;
- remaining structural dwell at each prefix milestone;
- complete-cycle precision and recall against frozen offline structural runs;
- aborted-path and recovery-failure rates;
- reset, OOD and quality-gated detector counts;
- the fraction of detections that occur only after the relevant phase has
  effectively ended;
- prospective no-refit performance by session.

## 12. Stage A5: Semantic Mapping To N/S/P/R

Only after the neutral model is frozen may states be mapped to the candidate
phase ontology.

The mapping uses contemporaneous state profiles:

| Candidate phase | Outcome-blind observable profile |
| --- | --- |
| `N` normal liquidity | low directional pressure, stable two-sided depth, ordinary spread and low transition velocity |
| `S` stress onset | increasing signed flow or cancel pressure, impacted-side depth decline and rising directional asymmetry |
| `P` peak pressure / price discovery | elevated pressure, low impacted-side depth, wide spread or strong microprice displacement |
| `R` liquidity recovery | pressure decay, positive replenishment, improving two-sided depth, but state not yet normalized |

The mapping must be deterministic from frozen state-profile fields.

After mapping, Track A tests whether the observed transition graph actually
supports:

```text
N -> S -> P -> R -> N
```

The following are valid contradictory findings:

```text
N -> S -> N dominates
S and P are not structurally separable
P transitions directly to N
R cannot be distinguished from ordinary N
one session contains most complete cycles
the market is better represented by a continuous factor than discrete phases
```

Track A must publish these findings rather than force a four-state story.

## 13. Structural Null Models

Track A must compare observed recurrence against dependence-preserving nulls.

### Null A: Cross-channel block shift

Shift trade-flow channels relative to book channels by one frozen no-wrap
block mapping within session and quality boundaries.

Purpose:

- preserve each channel's local autocorrelation;
- break the joint trade/book evolution.

### Null B: Transition block permutation

Permute complete state runs or time blocks within session.

Purpose:

- preserve state occupancy and duration distribution;
- break the observed transition grammar.

### Null C: Book-level identity permutation

Permute or reverse L1-L5 identities under a frozen valid transformation.

Purpose:

- test whether recurrence depends on spatial propagation across book levels;
- reject motifs driven only by aggregate depth.

### Null D: Side-orientation disruption

Disrupt the causal side-orientation assignment without reading future price.

Purpose:

- test whether direction-normalized prototypes rely on real contemporaneous
  pressure rather than arbitrary mirroring.

### Null E: Session-identity diagnostic

Fit a diagnostic classifier from state or motif identity to session identity.

Purpose:

- identify states that mostly encode collection day, volatility level or
  feed regime rather than reusable market structure.

Session classification is a diagnostic, not a favorable objective. Excessive
session separability blocks cross-session claims.

## 14. Primary Structural Metrics

Track A must publish at least:

### State-level metrics

- state occupancy by session and side;
- state duration distribution;
- transition matrix by session;
- blocked held-out structural loss;
- bootstrap state-profile stability;
- label-matched variation of information or equivalent partition stability;
- session concentration;
- OOD rate.

### Run-level metrics

- maximal-run count;
- complete and censored run count;
- run duration;
- state-dwell vector and complete-cycle duration;
- within-grammar duration modes and multimodality diagnostics;
- duration-distribution transport across sessions;
- grid-resolution sensitivity of phase boundaries and duration estimates;
- reset and merge counts;
- transition-path frequency;
- longest continuous run;
- overlap and dependence-block counts.

### Prototype-level metrics

- within-prototype distance;
- between-prototype distance;
- medoid stability;
- cross-session support;
- maximum session share;
- null-relative compactness;
- prospective assignment and OOD rate.

### Online-recognition metrics

- offline-to-online detection delay;
- false entry rate;
- state flicker rate;
- online posterior calibration against the frozen offline structural labels;
- remaining structural dwell at first detection;
- session-level recognizability.

These metrics describe structural recurrence only. They may not be renamed
alpha, edge, toxicity, safety or maker value.

## 15. Dependence And Replication

The independent replication unit is the session or collection day.

Within-session uncertainty must use:

- complete maximal runs;
- calendar-time blocks;
- flow or state-transition blocks;
- block bootstrap or another reviewed dependence-preserving estimator.

Grid rows are not independent observations.

Thousands of state rows or overlapping subsequences cannot substitute for
multiple independent sessions.

Every cross-session claim must report:

```text
session count
session role
per-session result
maximum session contribution
cross-session aggregation rule
claim-strength cap
```

## 16. Pre-Registered Track A Gates

Numeric thresholds are not finalized by this design draft. A support-only
execution supplement must freeze them before model fitting.

### Gate A0: Data admissibility

Require:

- deterministic Binance top-N reconstruction;
- accepted sequence continuity and source freshness;
- sufficient eligible calendar exposure;
- sufficient complete structural blocks;
- explicit separation of observation grid from inferred phase timescale;
- no future joins;
- explicit quality and reset boundaries.

Failure classification:

```text
insufficient_structural_support
```

### Gate A1: State stability

Require:

- stable state profiles under blocked refits;
- non-degenerate duration distributions;
- acceptable OOD and missing-state rates;
- no single session carrying the state definition;
- structural held-out improvement over the frozen Gaussian HSMM, memoryless
  HMM, continuous autoregressive/state-space and single-state heavy-tailed
  baselines.

Failure classification:

```text
continuous_state_no_discrete_phase_support
or
session_specific_motifs_only
```

### Gate A2: Transition grammar recurrence

Require:

- recurrent transition paths in multiple sessions;
- observed grammar stronger than transition-block and cross-channel nulls;
- stable maximal-run and reset semantics;
- data-supported dwell and cycle-duration distributions;
- grammar and duration conclusions robust to accepted observation grids;
- no interpretation based on overlapping window counts.

Failure classification:

```text
stable_states_but_no_recurrent_transition_grammar
```

### Gate A3: Prototype transport

Require:

- frozen prototypes assign prospectively without refitting;
- acceptable OOD rate;
- within-prototype compactness exceeds null expectations;
- no prototype is dominated by one session or one short calendar interval.

Failure classification:

```text
session_specific_motifs_only
```

### Gate A4: Online recognizability

Require:

- causal online phase detection;
- frozen neutral-grammar and transition-prefix detector identities;
- bounded false entry, false prefix advancement, abort and flicker rates;
- detection before the structural phase is effectively over;
- explicit recovery-failure, OOD and reset handling;
- consistency across prospective sessions without refitting.

Failure classification:

```text
retrospective_motifs_not_online_recognizable
```

### Gate A5: Final Track A classification

Only all-gate success yields:

```text
stable_cross_session_phase_grammar_online_recognizable
```

This classification authorizes a Track B design review only.

## 17. Track A Output Package

The minimum Track A package should contain:

```text
contracts/
  task.md
  execution_plan.md
  feature_schema.json
  source_and_quality_contract.json
  session_roles.json
  null_contract.json
  gate_contract.json

support/
  source_inventory.csv
  cadence_and_grid_support.csv
  observation_resolution_contract.json
  quality_intervals.csv
  dependence_support.csv

state/
  normalization_contract.json
  emission_and_duration_contract.json
  state_model_manifest.json
  model_baseline_comparison.csv
  neutral_state_profiles.csv
  state_duration_by_session.csv
  transition_matrix_by_session.csv
  structural_holdout_scores.csv

motifs/
  maximal_run_ledger.csv.gz
  transition_grammar.csv
  phase_and_cycle_duration_by_session.csv
  duration_mode_stability.csv
  grid_timescale_robustness.csv
  motif_prototypes.json
  prototype_assignment_by_session.csv
  prototype_stability.csv

online/
  online_decoder_manifest.json
  transition_prefix_detector_manifest.json
  transition_prefix_events.csv.gz
  offline_online_transition_comparison.csv
  online_recognition_by_session.csv
  online_cycle_recognition_by_session.csv
  online_ood_summary.csv

nulls/
  null_results.csv
  null_replicate_manifest.json

reports/
  phase_alignment_track_a_report.md

primary_classification.json
track_a_manifest.json
```

The package must preserve code, configuration, input identity, deterministic
seed conventions, output inventory and zero-outcome-access evidence.

## 18. Track B Handoff Contract

Track B may begin only after Track A is independently accepted as:

```text
stable_cross_session_phase_grammar_online_recognizable
```

The Track A handoff must freeze:

- feature schema;
- causal normalization;
- neutral state definitions;
- N/S/P/R mapping, if supported;
- transition and duration model;
- maximal-run and reset rules;
- motif prototypes;
- online posterior and detection rules;
- OOD policy;
- state and motif model identities;
- prospective assignment ledger;
- outcome-blind access proof.

Track B may then ask whether the frozen online phase state predicts future
response. It may not recluster states, rename phases, change run boundaries, or
select a new motif after reading future outcomes.

## 19. Explicit Non-Claims

Even a positive Track A result does not prove:

- future price direction is predictable;
- a phase has positive or negative expected return;
- a phase is safe or toxic for a maker;
- recovery creates a profitable quoting window;
- public contact equals a fill;
- queue position or hidden liquidity is known;
- Binance maker PnL is positive;
- any strategy should be implemented or deployed.

Track A proves only recurrent, transportable and causally recognizable market
structure.

## 20. Design Risks

The execution review must explicitly attack:

1. forcing four states because the desired story has four names;
2. using full-run hindsight to claim online recognizability;
3. selecting direction from future return;
4. fitting normalization on the full session;
5. allowing one continuous run to create many overlapping samples;
6. selecting state count from future-response performance;
7. mistaking session identity for a market phase;
8. collapsing the spatial book into one imbalance scalar;
9. allowing message-rate sampling to replace calendar exposure;
10. treating attractive plots as evidence of recurrence;
11. using retrospective soft-DTW assignment in Track B;
12. promoting historical sessions to prospective holdout status;
13. assigning `recovery` because future volatility later declined;
14. reopening Track A after Track B outcomes become visible.

## 21. Recommended First Formal Task

The next task should remain design/support-only:

```text
SKHYNIX-BINANCE-PHASE-ALIGNMENT-TRACK-A0-SUPPORT-AND-SURFACE-FREEZE
```

It should:

- inventory available Binance raw/top5 data;
- freeze session roles;
- measure cadence and candidate calendar grids;
- freeze the separation between observation resolution and inferred market
  timescale;
- freeze the state-tensor schema and causal normalization candidates;
- freeze the Student-t emission, covariance-block and duration candidates;
- freeze Gaussian, memoryless and continuous-state model baselines;
- freeze neutral-grammar and causal prefix-detector surfaces;
- freeze allowed and forbidden data surfaces;
- define the prospective collection requirement;
- define the Track A model-selection and null surfaces;
- read no future response;
- fit no phase model;
- grant no private, order, strategy or live authority.

Only after A0 is reviewed and accepted should an A1-A5 execution plan be
drafted.
