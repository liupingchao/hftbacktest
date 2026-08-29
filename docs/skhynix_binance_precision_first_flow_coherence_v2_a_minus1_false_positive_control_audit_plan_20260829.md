# SKHYNIX Binance Precision-First Flow Coherence V2
# A-1 Structural False-Positive Control Audit Plan

Date: 2026-08-29

Task: `0829T001`

Hypothesis ID: `PRECISION_FIRST_FLOW_COHERENCE_V2`

Audit ID: `PRECISION_FIRST_FLOW_COHERENCE_V2_A_MINUS1`

Status: frozen draft pending independent plan review

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
produce historical SIGNAL anchors whose structural false-discovery upper
bound is at most 10%, after accounting for filter selection?
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
historical_precision_support_candidate
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
- structural-null comparison mask false;
- insufficient causal history for a registered filter.

An abstained checkpoint:

- cannot open or confirm a candidate;
- cannot contribute persistence;
- cannot enter a denominator for precision claims;
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

At each supported checkpoint in direction `d`:

```text
trade_margin =
  d * fast_trade_ratio - 0.50

depth_margin =
  max(d * fast_depletion_ratio, d * fast_ofi_ratio) - 0.50

medium_margin =
  d * medium_composite_ratio - 0.25

opposition_margin =
  min(
    d * fast_depletion_ratio + 0.50,
    d * fast_ofi_ratio + 0.50
  )

normalized_coherence_margin =
  min(
    trade_margin,
    depth_margin,
    medium_margin,
    opposition_margin
)
```

For the two fast depth components:

```text
available_depth_components =
  finite values among depletion and OFI

depth_margin =
  max margin over available_depth_components

opposition_margin =
  min opposition margin over available_depth_components
```

At least one depth component must be finite. A missing component is excluded
from both extrema and is never substituted with zero. Missing trade or medium
composite makes the checkpoint `ABSTAIN`.

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

Opposite coherence, conflict, `ABSTAIN`, segment change or comparison-mask
loss cancels the candidate.

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
same observed/null boundary censor
trade magnitude, zero mask, missingness and denominator invariant
depth bundle and activity unchanged
```

Durations:

```text
10s sensitivity
30s primary
60s sensitivity
```

Replicates:

```text
199 per duration
root seed = 20260829
PRNG = numpy PCG64
```

Pairing, identifier and fingerprint rules are inherited from the accepted
Revision 6 contract, including `capture_ordinal`.

Any conservation, caliper, balance, fingerprint-diversity or boundary-censor
failure makes the null inadmissible.

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

For each outer held-out date and duration:

1. Exclude the held-out date.
2. For every filter, calculate the null false-cluster rate for each of 199
   replicates on the remaining dates.
3. Compute the Type-7 p95 false-cluster rate per decision-supported hour,
   where decision-supported checkpoints are exactly `SIGNAL | BACKGROUND`
   and exclude `ABSTAIN`.
4. Admit filters whose p95 null false-cluster rate is at most:

```text
0.10 per decision-supported hour
```

5. Select the least strict admitted filter using:

```text
lowest persistence
then lowest margin
then shortest novelty
then filter_id
```

If no filter qualifies, the complete held-out date is `ABSTAIN`.

This rule deliberately does not reward observed firing count.

## 12. Leave-One-Date-Out Cross-Fitting

The nine outer folds are identified by held-out research date.

For each fold:

- the filter is selected from the other eight dates using Section 11 only;
- the selected filter is applied once to the held-out observed date;
- the same selected filter is applied to every held-out null replicate;
- no held-out row enters selection;
- no filter may change after held-out evaluation.

The cross-fitted observed ledger is the concatenation of the nine held-out
observed ledgers.

For null replicate `r`, the cross-fitted null ledger concatenates held-out
replicate `r` from all nine folds using each fold's already selected filter.

This produces a selection-aware null without selecting on observed anchors.

## 13. Structural False-Positive Estimators

Primary units are 30-second dependence clusters, not raw confirmations.

Let:

```text
O = observed cross-fitted cluster count
N_r = cross-fitted null cluster count in replicate r
```

Estimators:

```text
null_count_p95 = Type-7 p95 of N_r

structural_FDP_U95 =
  min(1, null_count_p95 / max(O,1))

structural_precision_L95 =
  1 - structural_FDP_U95

count_tail_p =
  (1 + count(N_r >= O)) / 200
```

Date support:

```text
represented observed dates
maximum single-date cluster share
dates with observed count above date-specific null p90
```

These quantities describe structural false-positive control only.

## 14. Primary Gates

### Gate A-1-0: Authority

Require:

- accepted predecessor/source ancestry;
- source Git blob closure;
- 29-cache authority closure;
- deterministic Build A/B;
- manifest closure.

### Gate A-1-1: Zero Outcome

Require:

- exact allowed cache schema;
- exact consumed-field whitelist;
- no future/outcome fields consumed;
- no target/model/economic artifacts.

### Gate A-1-2: Tri-State And Abstention

Require:

- partition violations = 0;
- `ABSTAIN -> SIGNAL` violations = 0;
- cross-segment/quality feature windows = 0;
- candidate persistence through abstention = 0;
- no availability or recall lower bound.

Availability and abstention shares are diagnostics only.

### Gate A-1-3: Null Admissibility

For 10s, 30s and 60s require:

- 199 replicates;
- at least 190 distinct fingerprints;
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
- selected filter equals independent null-only recomputation;
- candidate-ledger and independent detector results exact for all filters;
- monotonic-family violations = 0;
- slice/reset mismatches = 0.

### Gate A-1-5: Precision Estimability

Require on the 30s primary:

```text
O >= 30 independent clusters
represented dates >= 4
maximum single-date cluster share <= 0.50
```

These are not recall targets. They are the minimum evidence required to bound
false-positive risk.

Failure classification:

```text
Aminus1_precision_not_estimable
```

### Gate A-1-6: Structural False-Positive Control

Require on the 30s primary:

```text
structural_FDP_U95 <= 0.10
structural_precision_L95 >= 0.90
count_tail_p <= 0.01
dates above date-null p90 >= 4
```

Sensitivity requirements:

```text
10s and 60s structural_FDP_U95 <= 0.20
10s and 60s count_tail_p <= 0.05
```

Failure classification:

```text
Aminus1_structural_false_positive_control_failed
```

### Gate A-1-7: Sparse Firing Guard

Require:

```text
observed cluster rate <= 5 per decision-supported hour
maximum same-capture 5s burst <= 2
```

There is no minimum firing-rate gate.

## 15. Classification Precedence

Exactly one classification is written:

```text
Aminus1_source_not_admissible
Aminus1_zero_outcome_boundary_violated
Aminus1_abstention_contract_violated
Aminus1_structural_null_not_admissible
Aminus1_selection_integrity_failed
Aminus1_precision_not_estimable
Aminus1_structural_false_positive_control_failed
Aminus1_signal_not_sparse
Aminus1_historical_precision_support_candidate
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
  structural_false_positive_summary.csv
  parameter_monotonicity.csv
  slice_invariance.csv

reports/
  A_minus1_summary.json

classification.json
run_manifest.json
```

Large candidate/null ledgers may remain ignored if compact summaries contain
exact hashes and complete gate evidence.

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
