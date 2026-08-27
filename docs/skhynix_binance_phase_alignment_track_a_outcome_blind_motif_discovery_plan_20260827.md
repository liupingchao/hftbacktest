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

## 9. Stage A2: Neutral Phase Segmentation

### 9.1 Primary model family

The recommended primary estimator is an interpretable sticky hidden
semi-Markov model:

```text
sticky HSMM
  + explicit state duration
  + neutral state identifiers
  + blocked held-out structural scoring
```

The sticky transition prior limits state flicker. Explicit duration prevents
one continuous run from becoming hundreds of event samples.

Neural encoders, autoencoders and unconstrained deep sequence models are not
primary Track A estimators.

### 9.2 Candidate state counts

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

### 9.3 Offline and online estimands

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
- structural held-out improvement over a simpler continuous or memoryless
  baseline.

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
- bounded false entry and flicker;
- detection before the structural phase is effectively over;
- consistency across prospective sessions.

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
  quality_intervals.csv
  dependence_support.csv

state/
  normalization_contract.json
  state_model_manifest.json
  neutral_state_profiles.csv
  state_duration_by_session.csv
  transition_matrix_by_session.csv
  structural_holdout_scores.csv

motifs/
  maximal_run_ledger.csv.gz
  transition_grammar.csv
  motif_prototypes.json
  prototype_assignment_by_session.csv
  prototype_stability.csv

online/
  online_decoder_manifest.json
  offline_online_transition_comparison.csv
  online_recognition_by_session.csv
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
- freeze the state-tensor schema and causal normalization candidates;
- freeze allowed and forbidden data surfaces;
- define the prospective collection requirement;
- define the Track A model-selection and null surfaces;
- read no future response;
- fit no phase model;
- grant no private, order, strategy or live authority.

Only after A0 is reviewed and accepted should an A1-A5 execution plan be
drafted.
