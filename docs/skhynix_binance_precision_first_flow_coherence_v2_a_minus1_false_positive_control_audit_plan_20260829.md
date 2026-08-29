# SKHYNIX Binance Precision-First Flow Coherence V2
# A-1 Structural False-Positive Control Audit Plan

Date: 2026-08-29

Task: `0829T001`

Hypothesis ID: `PRECISION_FIRST_FLOW_COHERENCE_V2`

Audit ID: `PRECISION_FIRST_FLOW_COHERENCE_V2_A_MINUS1`

Status: frozen draft pending independent plan review

Revision: 6

Review history:

```text
Round 1:
  P0/P1/P2/P3 = 0/5/2/0
  recommendation = FAIL
  data execution lock = retained

Round 2:
  P0/P1/P2/P3 = 0/1/3/1
  recommendation = FAIL
  data execution lock = retained

Round 3:
  P0/P1/P2/P3 = 0/1/1/1
  recommendation = FAIL
  data execution lock = retained

Round 4:
  P0/P1/P2/P3 = 0/0/1/0
  recommendation = FAIL
  data execution lock = retained

Round 5:
  P0/P1/P2/P3 = 0/0/1/0
  recommendation = FAIL
  data execution lock = retained
```

## 1. Decision Context

The accepted `0828T014` result established that the first
`FLOW_COHERENCE_TRANSITION_V1` contract failed. Its first failed gate was
trade-plus-depth availability, but later gates also showed sparse support,
zero stable variants and no separation from the registered structural null.

The new task does not overwrite or rescue that result. It registers a
different detector objective:

```text
cost(false positive) >> cost(false negative)
```

For an entry detector:

```text
false positive
  -> detector fires, downstream research/trading acts on background noise

false negative
  -> a real opportunity is missed, but no trade loss is created
```

This asymmetry is valid only for an entry detector. It must not be reused for
a protective risk-off detector, where missed danger can create direct loss.

## 2. Research Question

The primary question is:

```text
Can a causal, interpretable and deliberately sparse flow-coherence detector
keep its structural-null false-cluster rate below a frozen ceiling and still
show historical count separation after accounting for filter selection?
```

This is not a recall study. The detector may abstain almost everywhere.

The task does not ask:

- how many opportunities were missed;
- whether market-time coverage exceeds a fixed percentage;
- whether firing rate exceeds a minimum;
- whether the signal predicts future price;
- whether a strategy is profitable.

## 3. Claim Boundary

This is a zero-outcome A-1 structural audit.

Forbidden inputs:

- future midpoint or BBO;
- future return or markout;
- barrier outcomes;
- fill, queue, fee, rebate or PnL;
- model loss measured on a future target;
- private account or order data.

Permitted inputs:

- current and trailing-only 20ms trade, depletion and OFI features;
- current book validity and segment identity;
- current activity support;
- trailing-only detector state;
- outcome-blind structural-null assignments.

Passing this audit can produce only:

```text
historical_structural_false_fire_control_candidate
```

It cannot produce:

```text
prospective support
economic precision
positive expectancy
maker viability
live readiness
```

## 4. Historical Reuse And Validation Status

All 29 caches and all nine research dates were consumed by `0828T014`.
Therefore:

```text
every date role = historically_reused_post_selection
```

Leave-one-date-out cross-fitting prevents direct same-fold fitting and
evaluation, but it does not restore a genuinely unseen holdout.

Any passing result requires a later prospective task using newly collected
dates before a confirmatory economic claim.

## 5. Source Authority

Repository baseline:

```text
accepted predecessor commit = 45544ecc
cache-generating source commit =
  5603a670e617636b9994d605faef833164d3add4
```

Inputs:

```text
29 accepted deterministic replay caches
9 research dates
20ms causal checkpoints
cache schema v4
```

The task must verify:

- predecessor commit ancestry;
- source inventory and session-role files against their frozen Git blobs;
- cache authority SHA;
- all 29 cache size/SHA/schema rows;
- exact consumed-field whitelist;
- no unexpected cache fields.

No raw reparse is required if cache and source authority close exactly.

### 5.1 Exact Cache Schema

Allowed cache fields are exactly:

```text
activity
ask_depletion
ask_depth
bid_depletion
bid_depth
bin_boundary_violations
cache_schema_version
event_seq
initial_bridge_failure_count
midpoint
non_admitted_message_contributions
obi
ofi
ofi_abs
quality_boundary_count
ready
reset_count
segment_end_ids
segment_end_ts
segment_id
sequence_gap_count
spread_ticks
tick_size
trade_signed
trade_total
ts_ns
valid_book
```

The detector may load values only from:

```text
activity
ask_depletion
bid_depletion
event_seq
ofi
ofi_abs
ready
segment_id
trade_signed
trade_total
ts_ns
valid_book
```

Field-name inspection for schema admission is allowed. Values from midpoint,
OBI, spread, depth snapshots or any field outside the consumed whitelist may
not be loaded.

### 5.2 Inherited Code Authority

Accepted predecessor file:

```text
path =
  examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py

commit = 45544ecc
Git blob OID = 494c203e7195f292e057f7708c99f52096259a02
blob SHA256 =
  f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c
```

The new runner must import the predecessor module from the exact path above
only after verifying the complete blob SHA256. It must call, not reimplement,
the following inherited symbols:

```text
symbol                            normalized AST SHA256
conflict_primitives               760bbedc04aac79841ac1ce82a89ece851eea6529ba1d02ee756abcd7f05b128
coherence_predicates              9f0564156193492f5aafc4c4b06d79c1f38c7606947582ed38a554a29a0157ab
fixed_opposite_orientation_pairs  04cef064fdaf5cba94421d6d3250760ccf623b2d0531a883154d2a3bdb4b293d
null_layout                       2def320606fa9caf45e5878845e91026b1284b57b1dd3ff8265038de92c8dcf5
permute_trade_direction_paths     b870945f3a079f34337912776001c8bbe8af41c2644e2c0b4acf76277e7637ce
```

Normalization is:

```text
ast.dump(function_node, annotate_fields=True, include_attributes=False)
SHA256 over ASCII bytes
```

The runner must fail if the module blob, symbol AST, callable module/name or
callable code object differs. Independent validation must compare
checkpoint-level conflict/coherence arrays, null layout masks and paired
assignments against direct calls to the bound predecessor module.

Any semantic change requires a new contract revision.

## 6. Three-State Detector Contract

Every checkpoint-direction pair has exactly one state:

```text
SIGNAL
BACKGROUND
ABSTAIN
```

### 6.1 ABSTAIN

`ABSTAIN` is mandatory when any required condition is unavailable:

- invalid or not-ready book;
- startup/reconnect cooldown;
- missing required trade/depth denominator;
- cross-segment or non-contiguous feature history;
- insufficient causal history for a registered filter.

An abstained checkpoint:

- cannot open or confirm a candidate;
- cannot contribute persistence;
- cannot enter a decision-supported-time denominator;
- is reported as coverage loss only;
- is never labelled noise or false negative.

### 6.2 BACKGROUND

`BACKGROUND` requires complete feature support but failure of one or more
registered signal predicates.

### 6.3 SIGNAL

`SIGNAL` requires complete support and successful completion of the exact
causal state machine in Sections 7-9.

Partition invariants:

```text
SIGNAL & BACKGROUND = false
SIGNAL & ABSTAIN = false
BACKGROUND & ABSTAIN = false
SIGNAL | BACKGROUND | ABSTAIN = true
```

## 7. Base Interpretable Transition

The base transition remains:

```text
direct contiguous trade-depth conflict
  -> first coherent rising edge in direction d
  -> persistent coherent state
```

This retains the alignment point at the first causally visible change.

Base windows:

```text
fast = 100ms
medium = 500ms
prestate lookback = 500ms
minimum active checkpoints = 15 of prior 25
minimum contiguous conflict = 120ms
maximum candidate confirmation window = 1000ms
```

Base conflict and coherence formulas are inherited byte-for-byte from the
accepted `0828T014` V0 implementation. Their semantics are not retuned.

The new version adds only registered precision filters after a valid base
candidate is formed.

For V2, the maximum precision-confirmation deadline is separately frozen as:

```text
candidate_ts + 1000ms
```

This permits the registered 800ms persistence filter. Failure, opposition,
conflict, abstention or segment loss cancels the candidate before that
deadline.

## 8. Precision Filter Family

The family is the Cartesian product of three monotonic dimensions:

```text
coherence persistence:
  200ms, 400ms, 800ms

minimum normalized coherence margin:
  0.00, 0.10, 0.20

same-direction novelty lookback:
  500ms, 1000ms, 2000ms
```

Total:

```text
3 x 3 x 3 = 27 filters
```

Filter ID ordering:

```text
F{persistence_index}{margin_index}{novelty_index}
```

Indices use ascending list order.

### 8.1 Coherence Margin

V2 decision support requires both the fast and medium ratio vectors to have
finite trade, depletion and OFI components. If either depth component is
missing, the checkpoint is `ABSTAIN`.

At each supported checkpoint in direction `d`:

```text
trade_margin =
  d * fast_trade_ratio - 0.50

depth_margin =
  min(d * fast_depletion_ratio, d * fast_ofi_ratio) - 0.50

medium_margin =
  d * medium_composite_ratio - 0.25

normalized_coherence_margin =
  min(
    trade_margin,
    depth_margin,
    medium_margin
)
```

The medium composite is the median of all three finite medium components.
Missing values are never substituted or excluded from an extremum.

The candidate must maintain:

```text
normalized_coherence_margin >= registered threshold
```

for every supported checkpoint in its persistence interval.

### 8.2 Novelty

At the candidate rising edge in direction `d`, the exact same base coherence
predicate for direction `d` must be false at every supported checkpoint in
the registered trailing novelty interval.

The interval is closed on the left and open on the candidate:

```text
[candidate_ts - novelty_ms, candidate_ts)
```

Any `ABSTAIN` checkpoint inside the novelty interval makes the candidate
`ABSTAIN`; it does not count as non-coherence.

### 8.3 Persistence

Candidate exposure starts after the rising-edge checkpoint. Confirmation
occurs only after the registered number of subsequent supported checkpoints
all satisfy base coherence and the margin threshold.

The candidate checkpoint contributes zero persistence exposure.

Opposite coherence, conflict, `ABSTAIN` or segment change cancels the
candidate.

### 8.4 Refractory And Dependence

Fixed values:

```text
refractory = 30s per capture and direction
dependence cluster = 30s per capture across directions
```

The refractory is applied once to the common chronological base-candidate
rising-edge ledger before any precision filter is evaluated. A suppressed
base candidate cannot reappear under a stricter filter. No filter-specific
second refractory is allowed.

This deliberately allows a base candidate that later fails a strict filter to
suppress a later candidate. The recall loss is accepted and preserves exact
monotonic nesting across the 27-filter family.

After common refractory, fixed dependence-cluster IDs are assigned once,
before filtering, using base-candidate rising-edge timestamps:

```text
sort key =
  capture_id, candidate_ts_ns, candidate_event_seq, direction

same cluster when =
  same capture_id
  and same segment_id
  and current candidate_ts - previous candidate_ts <= 30s
```

Every filter may retain or delete fixed cluster IDs but may never recompute
them from filter-specific confirmation timestamps. Candidate-set and
unique-cluster-count monotonicity must both hold.

Every segment/quality/reset boundary forces a new cluster, regardless of
timestamp gap.

## 9. Candidate Ledger And Exact Family Evaluation

The implementation may build one lax candidate ledger for efficiency, but
each filter result must be exactly equivalent to running its full state
machine independently.

For every candidate record:

- capture/date/segment/direction;
- candidate and confirmation timestamp/event sequence;
- conflict duration and family;
- novelty support;
- minimum persistence-interval coherence margin;
- persistence exposure;
- cancel reason;
- every filter admission bit.

Hostile tests must compare ledger filtering with independent state-machine
runs for all 27 filters.

## 10. Outcome-Blind Structural Null

The accepted `0828T014` paired opposite-orientation path-swap null is reused:

```text
five-minute parents
activity/intensity-matched opposite-orientation microblock pairs
independent Bernoulli(0.5) label swap per fixed pair
trade magnitude, zero mask, missingness and denominator invariant
depth bundle and activity unchanged
```

Durations:

```text
10s sensitivity
30s primary
60s sensitivity
```

Null banks:

```text
selection bank:
  duration = 30s primary only
  replicates = 199
  bank_code = 1

evaluation bank:
  durations = 10s, 30s, 60s
  replicates = 199 per duration
  bank_code = 2

SeedSequence root:
  [20260829,bank_code,microblock_ms,replicate_id,capture_ordinal]

PRNG = numpy PCG64
```

Selection-bank and evaluation-bank RNG stream identities must be disjoint.
No generator state or stream key may be reused across banks. Independently
generated banks may coincidentally produce the same aggregate swap vector;
such collisions are reported but are not bank overlap.

Pairing, identifier and fingerprint rules are inherited from the accepted
Revision 6 contract, including `capture_ordinal`.

Any conservation, caliper, balance, fingerprint-diversity or boundary-censor
failure makes the null inadmissible.

### 10.1 External Audit Censor

The null comparison mask is an external audit censor, not detector input.

The causal detector runs without knowing pairability or microblock boundaries.
It assigns tri-state outputs, candidates, refractory and fixed cluster IDs
first.

Freeze the maximum trailing influence required to compute one V2 state:

```text
W_state = 2000ms
```

For duration `h`, filter `f` and possible candidate rising-edge checkpoint
`t`, define:

```text
causal_path_supported_f(t) =
  every checkpoint in
  [t - filter_novelty_ms, t + filter_persistence_ms]
  has complete V2 causal decision support

comparison_path_supported_{h,f}(t) =
  duration-h external comparison mask is true at every checkpoint in
  [
    t - filter_novelty_ms - W_state,
    t + filter_persistence_ms
  ]

E_{h,f}(t) =
  causal_path_supported_f(t)
  and comparison_path_supported_{h,f}(t)
```

`E_{h,f}(t)=false` produces `audit_censored`, not `ABSTAIN`, and does not alter
detector state or later candidates.

The left extension is additive. It is not
`max(filter_novelty_ms,W_state)`: the earliest novelty checkpoint has its own
two-second trailing feature influence.

A fixed cluster is included when at least one retained signal in that cluster
has `E_{h,f}(candidate_ts)=true`. It is counted once regardless of direction
or the number of included confirmations.

The filter-duration exposure is frozen in three explicit units:

```text
H_checkpoint_count_{h,f,D} =
  count of unique capture-time checkpoints t on date set D where
  E_{h,f}(t)=true

H_seconds_{h,f,D} =
  0.020 * H_checkpoint_count_{h,f,D}

H_hours_{h,f,D} =
  H_seconds_{h,f,D} / 3600
```

Exposure is counted once per capture-time checkpoint, never once per
direction. If directional support ever differs, `E_{h,f}(t)` is true only when
both directions are supported.

Observed and null use exactly the same `E_{h,f}`,
`H_checkpoint_count_{h,f,D}`, `H_seconds_{h,f,D}` and `H_hours_{h,f,D}`.
Because the null preserves support and external masks, any exposure mismatch
is a hard failure. Every rate stated as `per hour` uses `H_hours`, never a raw
checkpoint count.

## 11. Null-Only Filter Selection

Filter selection may use:

- training-date decision-supported hours;
- training-date null anchor/cluster counts;
- filter strictness ordering.

It may not use:

- training or held-out observed anchor counts;
- observed date distribution;
- future outcomes;
- any economic metric.

Filter selection uses only the 30s primary selection bank.

For each outer held-out date:

1. Exclude the held-out date.
2. For every filter, calculate the selection-bank null false-cluster rate for
   each of 199 replicates on the remaining dates.
3. Compute the Type-7 p95 false-cluster rate using the unique capture-time
   comparison-supported denominator frozen in Section 10.1.
   If the training-date `H_hours_{30s,f,D_train}` is zero, non-finite or
   otherwise not estimable, the filter is not admitted. A zero null count
   never converts zero exposure into a zero rate.
4. Admit filters whose p95 null false-cluster rate is at most:

```text
0.10 per comparison-supported capture hour
```

5. Select the first admitted filter in this frozen lexicographic order:

```text
lowest persistence
then lowest margin
then shortest novelty
then filter_id
```

If no filter qualifies, the complete held-out date is `ABSTAIN`.

This rule deliberately does not reward observed firing count.
It makes no claim that the lexicographic order is a total ordering of every
mixed-dimension notion of strictness.

The selected filter is then frozen for that fold and used unchanged for all
10s, 30s and 60s evaluation-bank audits.

Because the selection bank is independent of the evaluation bank and
selection reads no observed count, evaluation-bank randomization p-values are
interpreted conditional on the frozen selection-bank realization. The 27-way
filter search does not reuse evaluation draws.

## 12. Leave-One-Date-Out Cross-Fitting

The nine outer folds are identified by held-out research date.

For each fold:

- the filter is selected from the other eight dates using Section 11 only;
- the selected filter is applied once to the held-out observed date;
- the same selected filter is applied to every independent held-out
  evaluation-bank null replicate at all three durations;
- no held-out row enters selection;
- no filter may change after held-out evaluation.

The cross-fitted observed ledger is the concatenation of the nine held-out
observed ledgers.

For evaluation-bank null replicate `r` and duration `h`, the cross-fitted null
ledger concatenates held-out replicate `(h,r)` from all nine folds using each
fold's already selected 30s filter.

This produces a null conditional on an independently generated null-only
selection bank. It neither selects on observed anchors nor evaluates on the
Monte Carlo draws used for selection.

## 13. Structural False-Fire Estimators

Primary units are 30-second dependence clusters, not raw confirmations.

For each duration `h in {10s,30s,60s}`, let fold `j` select filter `f_j`.
Define:

```text
O_h =
  observed cross-fitted unique fixed-cluster count after duration-h censor

N_{h,r} =
  evaluation-bank cross-fitted unique fixed-cluster count for duration h,
  replicate r

H_checkpoint_count_h =
  sum over held-out folds j of
  H_checkpoint_count_{h,f_j,{held-out date j}}

H_seconds_h =
  0.020 * H_checkpoint_count_h

H_hours_h =
  H_seconds_h / 3600
```

Estimators:

```text
null_count_{h,p95} = Type-7 p95 of N_{h,r}

null_false_cluster_rate_{h,p95} =
  null_count_{h,p95} / H_hours_h

structural_null_burden_ratio_{h,p95} =
  null_count_{h,p95} / max(O_h,1)

count_tail_p_h =
  (1 + count(N_{h,r} >= O_h)) / 200
```

Fail-closed denominator rules:

```text
H_hours_h <= 0 or non-finite ->
  null_false_cluster_rate_{h,p95} = NOT_ESTIMABLE
  Gate A-1-5 fails
  Gate A-1-6 cannot pass

O_h = 0 ->
  date-share and burden-ratio metrics = NOT_ESTIMABLE
  Gate A-1-5 fails before those metrics are interpreted

any NaN or infinity in count, exposure or rate fields ->
  hard failure
```

The suffix `p95` means empirical Type-7 95th percentile, not a confidence
limit.

`structural_null_burden_ratio_{h,p95}` is a descriptive null-burden ratio, not
a false-discovery confidence bound. It must never be renamed or interpreted
as:

```text
FDP upper confidence bound
precision lower confidence bound
economic false-positive probability
```

Date support:

```text
represented observed dates
maximum single-date cluster share
dates with observed count above date-specific null p90
```

The identifiable claims are limited to null false-fire rate and historical
randomization separation.

## 14. Primary Gates

### Gate A-1-0: Authority

Require:

- accepted predecessor/source ancestry;
- source Git blob closure;
- 29-cache authority closure;
- exact 27-field schema and 12-field consumed whitelist closure;
- Build A/B use distinct resolved output roots;
- same-root finalization is rejected;
- preseal, pending and final full non-cache SHA difference count = 0;
- deterministic Build A/B becomes true only after final comparison;
- manifest closure.

### Gate A-1-1: Zero Outcome

Require:

- exact allowed cache schema;
- exact consumed-field whitelist;
- hostile unexpected future/outcome field causes hard failure;
- no future/outcome fields consumed;
- no target/model/economic artifacts.

### Gate A-1-2: Tri-State And Abstention

Require:

- partition violations = 0;
- `ABSTAIN -> SIGNAL` violations = 0;
- cross-segment/quality feature windows = 0;
- candidate persistence through abstention = 0;
- external comparison mask access by detector = 0;
- decision-supported denominator direction double-count = 0;
- no availability or recall lower bound.

Availability and abstention shares are diagnostics only.

### Gate A-1-3: Null Admissibility

For the 30s selection bank and 10s/30s/60s evaluation banks require:

- 199 replicates;
- at least 190 distinct fingerprints;
- cross-bank RNG stream-identity overlap = 0;
- pair-label, magnitude, zero-mask, missingness and denominator mismatch = 0;
- caliper and boundary-censor violations = 0;
- every date has at least three matched pairs;
- p95 joint distance <= 0.60.

Any duration may invalidate and none may rescue another.

### Gate A-1-4: Selection Integrity

Require:

- exactly nine outer folds;
- held-out row leakage = 0;
- observed-count access during selection = 0;
- selection/evaluation null bank overlap = 0;
- only the 30s selection bank chooses filters;
- selected filter identity is unchanged across all evaluation durations;
- selected filter equals independent null-only recomputation;
- candidate-ledger and independent detector results exact for all filters;
- candidate-set and fixed-cluster-count monotonicity violations = 0;
- fixed cluster ID recomputation after filtering = 0;
- cross-segment fixed-cluster merge = 0;
- observed/null `E_{h,f}`, checkpoint-count, seconds or hours exposure
  mismatch = 0;
- slice/reset mismatches = 0.

Slice/reset audit covers all 27 filters, tri-state counts, base-candidate
identity, common refractory admission, fixed cluster IDs, filter bits,
fold-selected signals and all three duration-specific external censor bits.

Every segment at least 10 minutes long receives artificial starts at 10-minute
spacing. Comparison starts after:

```text
30s cooldown + 2s maximum causal history + 30s common refractory
```

All post-guard identities and metrics must be exact.

### Gate A-1-5: Structural Support Estimability

Require on the 30s primary evaluation bank:

```text
H_checkpoint_count_30s > 0
H_seconds_30s > 0 and finite
H_hours_30s > 0 and finite
O_30s >= 30 independent clusters
represented dates >= 4
maximum single-date cluster share <= 0.50
```

These are not recall targets and do not estimate economic precision. They are
the minimum support needed to interpret null burden and cross-date count
separation.

Failure classification:

```text
Aminus1_structural_support_not_estimable
```

### Gate A-1-6: Structural False-Fire Control

Require on the 30s primary:

```text
null_false_cluster_rate_{30s,p95} <= 0.10 per hour
structural_null_burden_ratio_{30s,p95} <= 0.10
count_tail_p_30s <= 0.01
dates above date-null p90 >= 4
```

Sensitivity requirements:

```text
null_false_cluster_rate_{10s,p95} <= 0.20 per hour
null_false_cluster_rate_{60s,p95} <= 0.20 per hour
structural_null_burden_ratio_{10s,p95} <= 0.20
structural_null_burden_ratio_{60s,p95} <= 0.20
count_tail_p_10s <= 0.05
count_tail_p_60s <= 0.05
```

Failure classification:

```text
Aminus1_structural_false_fire_control_failed
```

### Gate A-1-7: Sparse Firing Guard

This gate measures actual selected-filter detector behavior before external
audit censoring.

Define:

```text
O_raw =
  cross-fitted selected-filter observed unique fixed-cluster count
  before external censor

raw_supported_checkpoint_count =
  sum over held-out folds j of unique capture-time checkpoints t where
  causal_path_supported_{f_j}(t)=true

H_raw_seconds =
  0.020 * raw_supported_checkpoint_count

H_raw_hours =
  H_raw_seconds / 3600

raw_cluster_rate_per_hour =
  O_raw / H_raw_hours, only when H_raw_hours > 0 and finite

raw_5s_burst =
  maximum selected-filter confirmation count in any same-capture 5s window
  before external censor
```

Require:

```text
H_raw_hours > 0 and finite
raw_cluster_rate_per_hour is finite
raw_cluster_rate_per_hour <= 5
raw_5s_burst <= 2
```

If `H_raw_hours <= 0` or is non-finite, the raw rate is
`NOT_ESTIMABLE` and Gate A-1-7 cannot pass. Zero raw clusters do not convert
zero raw exposure into a zero rate.

The external comparison mask, `O_h` and `H_hours_h` do not enter this gate.
There is no minimum firing-rate gate.

## 15. Classification Precedence

Exactly one classification is written:

```text
Aminus1_source_not_admissible
Aminus1_zero_outcome_boundary_violated
Aminus1_abstention_contract_violated
Aminus1_structural_null_not_admissible
Aminus1_selection_integrity_failed
Aminus1_structural_support_not_estimable
Aminus1_structural_false_fire_control_failed
Aminus1_signal_not_sparse
Aminus1_historical_structural_false_fire_control_candidate
```

Precedence follows gate order A-1-0 through A-1-7.

## 16. A0 Authority

If and only if every gate passes:

```text
draft_exploratory_a0_contract = true
exploratory_a0_execution_authorized = false
confirmatory_a0_authorized = false
future_target_access_authorized = false
prospective_precision_validation_required = true
```

Passing A-1 authorizes drafting only. A separate reviewed task is required
before reading any future target.

If any gate fails, all A0 flags remain false.

## 17. Required Outputs

```text
contracts/
  source_cache_contract.json
  tri_state_detector_contract.json
  precision_filter_family_contract.json
  structural_null_contract.json
  cross_fit_selection_contract.json
  gate_contract.json
  outcome_access_ledger.json
  execution_evidence_contract.json

support/
  source_cache_inventory.csv
  tri_state_support_by_date.csv
  candidate_ledger.csv
  filter_support_by_date.csv
  fold_selection_ledger.csv
  cross_fitted_signal_ledger.csv
  cross_fitted_null_summary.csv
  structural_false_fire_summary.csv
  parameter_monotonicity.csv
  slice_invariance.csv

reports/
  A_minus1_summary.json

classification.json
run_manifest.json
```

Large candidate/null ledgers may remain ignored if compact summaries contain
exact hashes and complete gate evidence.

Required negative tests:

```text
unexpected future_return_500ms cache field -> hard failure
same resolved Build A/B root -> hard failure
selection code reads observed count -> hard failure
selection/evaluation RNG stream-identity overlap -> hard failure
comparison mask changes detector state -> hard failure
filter-specific cluster recomputation -> hard failure
cross-segment cluster merge -> hard failure
direction-time denominator double count -> hard failure
novelty or persistence support-gap checkpoint enters H -> hard failure
comparison path omits earliest-state W_state influence -> hard failure
180000 supported checkpoints and 5 raw clusters ->
  H_raw_hours=1 and raw_cluster_rate_per_hour=5
zero selection exposure and zero null clusters -> filter not admitted
zero evaluation exposure -> Aminus1_structural_support_not_estimable
zero raw exposure and zero raw clusters -> sparse gate does not pass
NaN or infinity in exposure/rate -> hard failure
poisoned allowed-but-unconsumed midpoint/OBI/spread -> no output change
callable replacement or inherited symbol byte change -> hard failure
```

## 18. Stop Rules

Do not rescue a failed result by:

- rewarding a filter for more observed anchors;
- lowering persistence/margin/novelty after seeing results;
- changing the null false-fire threshold;
- replacing clusters with raw confirmations;
- dropping difficult dates;
- treating `ABSTAIN` as background;
- adding a minimum firing-rate target;
- reading future price.

The next action after this draft is independent plan review. Data execution
remains locked until that review reports no P0-P2 defects.
