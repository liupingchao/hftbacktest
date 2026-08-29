# SKHYNIX Binance Precision-First Fresh-Channel Consensus M-State V1
# A-1 Structural False-Positive Control Audit Plan

Date: 2026-08-29

Task: `0829T002`

Hypothesis ID: `FRESH_CHANNEL_CONSENSUS_MSTATE_V1`

Audit ID: `FRESH_CHANNEL_CONSENSUS_MSTATE_V1_A_MINUS1`

Status: candidate contract Revision 2; data execution locked

Revision: 2

Review history:

```text
Round 1:
  reviewed commit = 2dc0ada6
  P0/P1/P2/P3 = 0/4/2/0
  recommendation = FAIL
  data execution lock = retained
```

## 1. Decision Context

The accepted `0829T001` result is:

```text
Aminus1_structural_support_not_estimable
```

Its implementation and evidence passed QA, but the registered detector
failed scientifically. Even the least restrictive `F000` admitted zero of
350 common candidates:

```text
ABSTAIN              279
not_novel             34
conflict              22
coherence_lost         8
opposite_coherence     5
margin                 2
```

The dominant failure was not the amplitude threshold. V2 required trade,
depletion and OFI to be simultaneously finite at every 20ms checkpoint
through the complete novelty and persistence path.

This successor does not:

- lower the directional thresholds;
- accept two of three channels;
- reinterpret missing data as background;
- tune a filter using observed candidate count;
- inspect future price or economic outcomes.

It registers a new M-state representation:

```text
three independently observed channel states
  -> bounded fresh-memory projection
  -> first fresh three-channel consensus onset
  -> persistent fresh consensus
```

The scientific change is asynchronous observability, not weaker directional
evidence.

## 2. Methodology Reference

This plan follows the local reusable reference:

```text
path =
  /Users/liu/Documents/
  hftbacktest-0814t001-skhynix-episode-research/
  docs/conditional_risk_research_methodology_kernel_v1.md

content SHA256 =
  dd1adee720f613a51393ff97ae3fd026a79d9b553eaa984e629ab5ddbcab7505
```

The kernel is a reusable design reference, not execution authority. Its path
is recorded for provenance; execution does not depend on reading it at
runtime.

The applicable order remains:

```text
support-only structural audit
  -> separately reviewed future-outcome task
  -> latency and economics only after predictive support
```

## 3. Research Question

Primary question:

```text
Can an interpretable causal M-state built from fresh asynchronous trade,
depletion and OFI channel memories produce recurrent, sparse consensus-onset
landmarks whose conditional trade-orientation-null false-cluster burden is
below the frozen precision-first ceiling?
```

The task is not a recall study.

Allowed:

- high abstention;
- missed opportunities;
- a no-filter fold sentinel;
- a negative or not-estimable result.

Forbidden interpretations:

- zero firing means high precision;
- structural-null separation means economic alpha;
- historical cross-fitting creates a prospective holdout;
- passing A-1 authorizes A0 execution or live trading.

## 4. Claim And Outcome Boundary

This is an outcome-blind A-1 task.

Forbidden values:

- midpoint, BBO or spread values;
- future returns or markouts;
- barrier outcomes;
- fill, queue, fee, rebate or PnL;
- model loss measured against a future target;
- private account or order data.

Permitted causal values:

- current and trailing trade, depletion and OFI ratios;
- current readiness, activity and segment identity;
- trailing channel-memory state;
- outcome-blind null assignments and external comparison masks.

The strongest possible result is:

```text
historical_structural_false_fire_control_candidate
```

It is not:

```text
economic precision
predictive alpha
maker viability
prospective support
live readiness
```

## 5. Historical Reuse

The same 29 caches and nine dates were used by predecessor studies.

Every date role is:

```text
historically_reused_post_selection
```

Leave-one-date-out cross-fitting prevents direct same-fold selection leakage,
but it does not create new holdout evidence.

No result from this task may be called prospective.

## 6. Source And Code Authority

Inputs:

```text
29 accepted deterministic replay caches
9 research dates
20ms causal checkpoints
cache schema v4
```

Frozen source cache root:

```text
/Users/liu/Documents/
hftbacktest-0829t001-precision-first-flow-coherence-audit/
local_live_analysis/
skhynix_precision_first_flow_coherence_a_minus1_0829T001/cache
```

The execution runner copies/verifies those 29 cache files into its own output
root. It may not discover or substitute additional caches.

Accepted predecessor/controller authority:

```text
accepted controller commit =
  af0d38f432a343ec3fb5d89c7567f93745565f91

accepted evidence parent =
  f1ad26a6e891f5580e4d335a93c21eac1746f739
```

Frozen cache/source authority artifacts at that commit:

```text
source cache inventory path =
  local_live_analysis/
  skhynix_precision_first_flow_coherence_a_minus1_0829T001/
  support/source_cache_inventory.csv

Git blob OID =
  c1c877b65a25e8976f972883b311c5a086ed4536

file SHA256 =
  e6f8f3fedb76eeed6d99cb8cb5306732af54f20bcb0882b56983dc61273a39e1

inventory payload SHA256 =
  e554793e98d9a000b1b8c0049897ed42a14167d07f16b1602acfcbb2f048b12c

source cache contract path =
  local_live_analysis/
  skhynix_precision_first_flow_coherence_a_minus1_0829T001/
  contracts/source_cache_contract.json

Git blob OID =
  e72e34b2f9fe78eb4fed6515f7dcac8e14fc2c59

file SHA256 =
  4b6b2c570093725a2f83a356d453b0c022bb3c47c23e8daf441c51ded55de494
```

The runner must verify controller ancestry, both Git blobs, both file hashes,
the internal inventory payload hash and all 29 cache size/SHA/schema rows.

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

Values may be loaded only from:

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

Values from midpoint, OBI, spread, depth snapshots or any future/economic
field may not be loaded.

Accepted causal/null primitive authority:

```text
path =
  examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py

accepted commit = 45544ecc
Git blob OID = 494c203e7195f292e057f7708c99f52096259a02
blob SHA256 =
  f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c
```

The runner must verify and directly call the accepted implementations with
these normalized AST SHA256 values:

```text
symbol                            normalized AST SHA256
base_masks                        bc2155a38bd1707fcdb77bdebea611da3889934a47d415bdfd0d5a95842d7114
build_features                    e5cca6c2b7627ef8e3719e4fdecb5a540a42a2028e2fb141ea9fff4f7c246933
fixed_opposite_orientation_pairs  04cef064fdaf5cba94421d6d3250760ccf623b2d0531a883154d2a3bdb4b293d
null_layout                       2def320606fa9caf45e5878845e91026b1284b57b1dd3ff8265038de92c8dcf5
permute_trade_direction_paths     b870945f3a079f34337912776001c8bbe8af41c2644e2c0b4acf76277e7637ce
prior_count                       abbd39099a876c70713b05871152fdfa56d1ee8d142611debe849f1afcc018a8
run_length                        074d37d93e2b66de5d94c9112494fd3f615a5f24aa3ce77a2f5da64a5b37e37d
```

Normalization is:

```text
ast.dump(function_node, annotate_fields=True, include_attributes=False)
SHA256 over ASCII bytes
```

Reimplementation or callable rebinding is a hard authority failure.

## 7. Primitive Channel Observations

Channels:

```text
C = {trade, depletion, ofi}
```

For channel `c`, use the already accepted causal ratio features:

```text
r100_c(t)
r500_c(t)
```

No cross-channel imputation is allowed.

The rolling ratios are feature values, not channel-arrival timestamps.
Therefore a finite ratio alone may not refresh channel memory.

Freeze channel-specific new-evidence indicators from the underlying 20ms
causal contribution bins:

```text
new_trade_evidence(t) =
  trade_total(t) > 0

new_depletion_evidence(t) =
  bid_depletion(t) + ask_depletion(t) > 0

new_ofi_evidence(t) =
  ofi_abs(t) > 0
```

All four contribution magnitudes must be finite and non-negative. A negative
or non-finite value is a source/numeric integrity failure, not `NO_UPDATE`.

The indicators are invariant under the registered null because trade
magnitude, depth magnitude, OFI magnitude, zero masks and missingness are
preserved exactly.

Define:

```text
base_eligible(t) =
  detector_ready(t)
  and activity_supported(t)
  and valid_book(t)
```

When `base_eligible(t)=false`:

- all three current observations are `UNKNOWN`;
- all three stored memories are cleared immediately;
- the aggregate M-state is `ABSTAIN`;
- no anchor or persistence checkpoint may cross that checkpoint.

This prevents bounded memory from carrying a favorable sign through startup,
reconnect, invalid-book or inactive-flow intervals.

For filter margin `m` and direction `d in {-1,+1}`, define a current strong
observation only at a checkpoint with new evidence for that channel:

```text
strong_{c,d,m}(t) =
  new_channel_evidence_c(t)
  and finite(r100_c(t))
  and finite(r500_c(t))
  and d * r100_c(t) >= 0.50 + m
  and d * r500_c(t) >= 0.25 + m
```

The channel input action is:

```text
A_{c,m}(t) =
  GLOBAL_INVALID
    if base_eligible(t) = false

  NEW_INVALID
    if new_channel_evidence_c(t)
    and either required ratio is non-finite

  NEW_POS
    if strong_{c,+1,m}(t)

  NEW_NEG
    if strong_{c,-1,m}(t)

  NEW_NEUTRAL
    if new_channel_evidence_c(t)
    and both ratios are finite
    and neither direction is strong

  NO_UPDATE
    if base_eligible(t)
    and not new_channel_evidence_c(t)
```

The positive and negative conditions must be mutually exclusive. Every
channel/checkpoint must have exactly one action. Any overlap or gap is an
integrity failure.

`NO_UPDATE` never changes `last_observed_at_c`. Repeated finite rolling ratios
without a new underlying contribution cannot refresh TTL.

The amplitude thresholds are not lower than V2:

```text
fast threshold   = 0.50 + margin
medium threshold = 0.25 + margin
```

## 8. Fresh Channel Memory

For TTL `tau`, each channel has a deterministic memory:

```text
K_{c,tau,m}(t) in {-1,0,+1,UNKNOWN}
```

Update rules, in chronological order:

1. Every segment/reset boundary or `base_eligible=false` checkpoint clears
   all channel memories.
2. `NEW_POS`, `NEW_NEG` or `NEW_NEUTRAL` overwrites that channel memory and
   stores the current checkpoint timestamp and event sequence.
3. `NEW_INVALID` clears that channel memory immediately.
4. `NO_UPDATE` does not overwrite the memory or refresh its timestamp.
5. A stored memory is usable only while:

```text
0 <= t - last_observed_at_c <= tau
```

6. Once its age exceeds `tau`, the channel state is `UNKNOWN`.
7. A neutral `0` observation invalidates an older directional state
   immediately. It may not be skipped in favor of an older sign.
8. An opposite observation invalidates the previous direction immediately.
9. No memory may cross capture, segment, reconnect or quality boundaries.

This is bounded event-evidence memory, not rolling-feature timestamp carry and
not unrestricted forward filling. The TTL is measured from the checkpoint of
the most recent new valid underlying channel contribution.

## 9. Aggregate M-State

For each filter `(tau,m)`, aggregate the three fresh channel memories:

```text
M_{tau,m}(t) =
  ABSTAIN
    if any channel memory is UNKNOWN

  SIGNAL_POS
    if all three channel memories equal +1

  SIGNAL_NEG
    if all three channel memories equal -1

  BACKGROUND
    otherwise
```

Therefore:

- all three channels remain required;
- asynchronous observations are allowed only within `tau`;
- disagreement and neutral evidence are background;
- stale or unavailable evidence is abstention;
- missingness never votes for a direction.

Every checkpoint belongs to exactly one of:

```text
SIGNAL_POS
SIGNAL_NEG
BACKGROUND
ABSTAIN
```

`SIGNAL_POS` and `SIGNAL_NEG` are mutually exclusive.

## 10. Alignment Anchor

The common anchor ledger uses the most permissive registered instantaneous
state:

```text
tau = 100ms
margin = 0.00
```

For direction `d`, a candidate anchor occurs at checkpoint `t` only when:

```text
M_{100ms,0}(t) = SIGNAL_d
```

and the immediately preceding six checkpoints, covering 120ms, are all:

```text
M_{100ms,0} = BACKGROUND
```

The six checkpoints must:

- be in the same segment as `t`;
- contain no `ABSTAIN`;
- contain no `SIGNAL_POS` or `SIGNAL_NEG`.

Thus an anchor cannot be created by:

- data returning after a stale interval;
- startup/reconnect recovery;
- an `ABSTAIN -> SIGNAL` transition;
- repeated checkpoints inside one existing consensus state.

The anchor is the first causally visible:

```text
observable background -> fresh three-channel consensus
```

No filter may create a new later anchor that is absent from this common
ledger. Stricter filters may only delete common anchors. This deliberately
accepts false negatives and preserves exact candidate identity nesting.

## 11. Precision Filter Family

The family is:

```text
freshness TTL:
  100ms, 60ms, 40ms

coherence persistence:
  200ms, 400ms, 800ms

minimum channel margin:
  0.00, 0.10, 0.20
```

Total:

```text
3 x 3 x 3 = 27 filters
```

Filter IDs:

```text
F{ttl_index}{persistence_index}{margin_index}
```

Index order is the list order above, from least restrictive to most
restrictive.

For a common anchor at `t` and direction `d`, filter `f=(tau,p,m)` admits it
only when:

1. The preceding 120ms are all `BACKGROUND` under `M_{tau,m}`.
2. The anchor checkpoint is `SIGNAL_d` under `M_{tau,m}`.
3. Every subsequent checkpoint in:

```text
(t, t + p]
```

is `SIGNAL_d` under `M_{tau,m}`.
4. No segment boundary occurs before confirmation.

Any opposite, neutral, expiry or missing channel state cancels the filter.

Cancellation reasons are mutually exclusive and evaluated in this order:

```text
insufficient_history
segment_boundary
prestate_abstain
prestate_not_background
anchor_abstain
anchor_not_consensus
persistence_abstain
opposite_consensus
consensus_lost
```

The family partial order is:

```text
smaller TTL is stricter
longer persistence is stricter
larger margin is stricter
```

Every stricter admitted candidate set and fixed cluster set must be a subset
of every directly comparable looser set.

This is a:

```text
common-anchor-conditioned delete-only confirmation family
```

It is not a claim that the natural onset timestamps of 27 independently run
M-state detectors are nested.

For diagnosis only, independently compute each `(tau,m)` natural onset ledger
using its own 120ms background prestate before common-anchor restriction.
Report:

```text
orphan_strict_onset =
  natural onset under a stricter (tau,m)
  that has no exact common-anchor identity
```

Orphan onsets:

- are false negatives accepted by the registered common-anchor family;
- may not enter filter selection;
- may not enter observed or null estimators;
- may not rescue any gate;
- must be reported by date and filter.

## 12. Refractory And Dependence

Apply once to the common chronological anchor ledger:

```text
refractory = 30s per capture and direction
dependence cluster = 30s per capture across directions
```

The refractory and cluster IDs are assigned before filter evaluation.

Every segment/reset/quality boundary forces a cluster break.

No filter-specific refractory or cluster recomputation is allowed.

## 13. Candidate And State Evidence

Every common candidate row must include:

- capture/date/segment/direction;
- candidate timestamp and event sequence;
- channel last-observation timestamps and ages at the anchor;
- common 120ms prestate counts;
- fixed refractory and dependence-cluster identity;
- all 27 admission bits;
- filter-specific confirmation identity or cancellation reason.

State evidence by date and filter must include:

- `SIGNAL_POS` checkpoint count;
- `SIGNAL_NEG` checkpoint count;
- `BACKGROUND` checkpoint count;
- `ABSTAIN` checkpoint count;
- exact partition check;
- per-channel expiry count;
- per-channel neutral-overwrite count;
- maximum observed memory age.

## 14. Outcome-Blind Structural Null

Reuse the accepted paired opposite-orientation path-swap null:

```text
five-minute parents
activity/intensity-matched opposite-orientation microblock pairs
independent Bernoulli(0.5) swap per fixed pair
trade magnitude, zero mask, missingness and denominators invariant
depth bundle and activity unchanged
```

The exact conditional null hypothesis is:

```text
H0_conditional:
  conditional on the observed depth bundle, OFI/depletion path,
  activity/intensity path, missingness, denominators, segment boundaries
  and fixed matched-pair structure,
  trade orientation labels are exchangeable within each registered
  opposite-orientation pair.
```

The null asks whether trade orientation aligns with the fixed depletion/OFI
path more often than this conditional exchangeability law permits.

It does not estimate:

- an unconditional market-background false-positive rate;
- economic false positives;
- the chance that a live trade loses money;
- a null that randomizes depth or OFI structure.

Every null replicate must start from randomized causal features and rerun the
complete pipeline in this order:

```text
channel new-evidence actions
-> filter-specific channel memories
-> M-states
-> common anchors
-> common refractory
-> replicate-specific fixed dependence clusters
-> 27 delete-only confirmations
-> external duration censor
-> cluster counts
```

Observed channel memory, anchors, refractory decisions, cluster IDs,
admission bits or counts may not be copied into a null replicate.

Null banks:

```text
selection:
  duration = 30s
  replicates = 199
  bank_code = 3

evaluation:
  durations = 10s, 30s, 60s
  replicates = 199 per duration
  bank_code = 4

SeedSequence root:
  [20260829,bank_code,duration_ms,replicate_id,capture_ordinal]

PRNG = numpy PCG64
```

Selection and evaluation streams must be disjoint.

Required null invariants:

- exact magnitude preservation;
- exact zero-mask preservation;
- exact missingness preservation;
- exact denominator preservation;
- fixed pair identity;
- comparison-mask identity;
- at least 190 distinct fingerprints among 199 replicates;
- minimum three valid date-pairs per duration;
- maximum date p95 joint distance at most 0.60.

Any invariant failure makes the null inadmissible.

## 15. External Comparison Censor And Exposure

The comparison mask is never detector input.

The detector first produces channel memories, M-states, common anchors,
refractory and fixed cluster IDs without knowing null pairability.

For filter `f=(tau,p,m)`, duration `h` and possible anchor checkpoint `t`,
define:

```text
causal_mstate_supported_f(t) =
  every checkpoint in [t-120ms, t+p]
  has non-ABSTAIN M_{tau,m}

comparison_supported_{h,f}(t) =
  external comparison mask is true throughout
  [t-120ms-500ms-tau, t+p]

E_{h,f}(t) =
  causal_mstate_supported_f(t)
  and comparison_supported_{h,f}(t)
```

The left influence includes:

```text
120ms prestate
500ms ratio history
tau memory freshness
```

It is additive and must not be replaced by a maximum.

`E=false` is `audit_censored`; it does not alter causal detector state.

Exposure units:

```text
H_checkpoint_count = count of unique supported capture-time checkpoints
H_seconds = 0.020 * H_checkpoint_count
H_hours = H_seconds / 3600
```

Exposure is counted once per capture-time checkpoint, not per direction.

Observed and null use exactly the same filter-duration exposure mask and
units. Negative, non-finite or inconsistent units are A-1 integrity failures.

## 16. Null-Only Filter Selection

Use leave-one-date-out outer folds.

For each held-out date:

1. Use the other eight dates only.
2. Use only the independent 30s selection null bank.
3. For each filter, calculate the 199 training-date null false-cluster rates.
4. If training exposure is exactly zero, the filter is ordinarily not
   admitted in that fold.
5. Negative, non-finite or unit-inconsistent exposure is an A-1-4 integrity
   failure, not an ordinary filter rejection.
6. Admit filters whose Type-7 p95 null false-cluster rate is at most:

```text
0.10 per comparison-supported capture hour
```

7. Select the first admitted filter in frozen least-restrictive-first order:

```text
largest TTL
then shortest persistence
then lowest margin
then filter_id
```

Selection may not inspect observed candidate/cluster count.

If no filter qualifies:

```text
fold_state = META_ABSTAIN
selected_filter = NONE
observed/null/exposure contribution = 0
```

The selected 30s filter is reused unchanged for 10s and 60s evaluation.

## 17. Estimators

Primary unit:

```text
fixed 30s dependence cluster
```

For duration `h`:

```text
O_h =
  observed cross-fitted unique fixed clusters after duration-h censor

N_{h,r} =
  evaluation-bank cross-fitted unique fixed clusters for replicate r

null_count_p95_h =
  Type-7 p95 of N_{h,r}

null_false_cluster_rate_p95_h =
  null_count_p95_h / H_hours_h

structural_null_burden_ratio_p95_h =
  null_count_p95_h / O_h

count_tail_p_h =
  (1 + count_r[N_{h,r} >= O_h]) / 200
```

Optional-value semantics:

```text
H_hours_h = 0
  -> null_false_cluster_rate_p95_h = null
  -> estimator is not estimable

O_h = 0
  -> structural_null_burden_ratio_p95_h = null
  -> maximum_single_date_share_h = null
  -> estimator is not estimable

count_tail_p_h
  -> remains finite in [0,1], including when O_h = 0
```

These exact null values are legitimate support semantics, not numeric
corruption.

Negative, NaN, infinite, type-invalid or unit-inconsistent count/exposure/
rate values are integrity failures.

Primary:

- exact zero exposure or fewer than 30 observed clusters fails A-1-5;
- `0 observed / 0 null` is not estimable and is never high precision.

Sensitivities:

- zero exposure or zero observed clusters at 10s or 60s makes that
  sensitivity non-estimable and fails A-1-6.

### 17.1 Raw Sparsity Estimator

For held-out fold `j`, use its already selected filter `f_j` on the observed
held-out date without any external comparison mask.

Define:

```text
H_raw_checkpoint_count =
  sum across non-NONE folds of unique checkpoints t where
  every M-state checkpoint in [t-120ms,t+p_{f_j}] is non-ABSTAIN

H_raw_seconds =
  0.020 * H_raw_checkpoint_count

H_raw_hours =
  H_raw_seconds / 3600

O_raw =
  observed unique fixed clusters retained by the selected filters
  before external comparison censor

raw_cluster_rate =
  O_raw / H_raw_hours when H_raw_hours > 0
  else null
```

`NONE` folds contribute exactly zero raw clusters and zero raw exposure.

The maximum 5s burst is calculated from selected-filter confirmation
timestamps before external comparison censor, separately within captures;
windows may not cross segment boundaries.

Raw exact-zero exposure is legitimate but cannot pass A-1-7. Negative,
non-finite or unit-inconsistent raw values fail A-1-4.

## 18. Primary Gates

Gate A-1-0, Authority:

- plan SHA and accepted ancestry;
- source/cache/schema closure;
- exact callable binding;
- Build A/B distinct roots;
- preseal, pending and final difference counts all zero;
- exact Required Outputs and manifest closure.

Gate A-1-1, Zero Outcome:

- consumed-field whitelist exact;
- future/economic fields not loaded;
- poison of unconsumed values changes no output;
- no target/economic artifacts.

Gate A-1-2, M-State Integrity:

- exact four-state partition;
- exact six-action channel-input partition;
- zero sign overlap;
- zero TTL refresh without new channel evidence;
- zero unknown-to-background conversion;
- zero neutral-skip violations;
- zero expiry violations;
- zero cross-segment memory carry;
- zero `ABSTAIN -> SIGNAL` anchors;
- zero non-background-prestate anchors;
- zero filter monotonicity violations;
- zero slice/reset invariance mismatches.

Gate A-1-3, Structural Null Admissibility:

- 199 replicates in every bank;
- at least 190 distinct fingerprints;
- zero stream overlap;
- zero conservation/invariant mismatches;
- minimum date-pair count at least 3;
- maximum date p95 joint distance at most 0.60.

Gate A-1-4, Selection And Numeric Integrity:

- nine folds;
- observed selection access zero;
- held-out isolation exact;
- null-bank overlap zero;
- checkpoint/seconds/hours exact;
- all required counts and exposures non-negative, finite and correctly typed;
- rate is null if and only if its exposure is exactly zero;
- burden/share is null if and only if observed count is exactly zero;
- legitimate zero support is not a numeric violation;
- negative, NaN, inf and unit mismatch are numeric violations;
- filter-duration numerator/denominator identity exact.

Gate A-1-5, Structural Support Estimability:

```text
primary H_hours > 0
primary observed clusters >= 30
represented observed dates >= 4
maximum single-date share <= 0.50
```

These are estimability requirements, not recall targets.

Gate A-1-6, Structural False-Fire Control:

```text
30s null false-cluster rate p95 <= 0.10/hour
30s structural null burden ratio p95 <= 0.10
30s count-tail p <= 0.01
dates above date-specific null p90 >= 4

10s and 60s estimable
10s and 60s rate p95 <= 0.20/hour
10s and 60s burden p95 <= 0.20
10s and 60s count-tail p <= 0.05
```

Gate A-1-7, Raw Sparsity:

```text
raw supported exposure > 0
raw selected-filter cluster rate <= 5/hour
maximum 5s burst <= 2
```

Coverage and minimum firing rate are diagnostics only.

## 19. Classification

First failed gate determines the unique classification:

```text
A-1-0 -> Aminus1_source_not_admissible
A-1-1 -> Aminus1_zero_outcome_boundary_violated
A-1-2 -> Aminus1_mstate_integrity_failed
A-1-3 -> Aminus1_structural_null_not_admissible
A-1-4 -> Aminus1_selection_integrity_failed
A-1-5 -> Aminus1_structural_support_not_estimable
A-1-6 -> Aminus1_structural_false_fire_control_failed
A-1-7 -> Aminus1_signal_not_sparse
all pass -> Aminus1_historical_structural_false_fire_control_candidate
```

No later gate may rescue an earlier failure.

## 20. Required Outputs

```text
contracts/
  source_cache_contract.json
  mstate_detector_contract.json
  precision_filter_family_contract.json
  structural_null_contract.json
  cross_fit_selection_contract.json
  gate_contract.json
  outcome_access_ledger.json
  execution_evidence_contract.json

support/
  source_cache_inventory.csv
  channel_state_support_by_date.csv
  mstate_support_by_date.csv
  candidate_ledger.csv
  filter_support_by_date.csv
  fold_selection_ledger.csv
  cross_fitted_signal_ledger.csv
  cross_fitted_null_summary.csv
  structural_false_fire_summary.csv
  parameter_monotonicity.csv
  orphan_strict_onset_by_date.csv
  slice_invariance.csv

reports/
  A_minus1_summary.json

classification.json
run_manifest.json
```

The non-cache artifact set must equal this list exactly.

## 21. Hostile Tests

At minimum:

- current strong observation threshold boundaries;
- mutual exclusion of positive/negative observations;
- no new underlying event means `NO_UPDATE`;
- repeated finite rolling ratios without new evidence cannot refresh TTL;
- channel-specific event masks cannot refresh another channel;
- new evidence with non-finite required ratio clears that channel;
- neutral immediately overwrites sign;
- unknown retains sign only within TTL;
- expiry at `tau + one checkpoint`;
- reset/segment clears memory;
- no cross-capture carry;
- `ABSTAIN -> SIGNAL` cannot anchor;
- six-checkpoint background prestate is exact;
- stricter filters cannot create common anchors;
- independently detected strict natural onsets missing from the common ledger
  are counted as orphan diagnostics and never enter estimators;
- all comparable filter candidate/cluster subsets are monotonic;
- the conditional H0 wording and fixed conditioning variables are exact;
- every null replicate recomputes actions, memories, M-state, anchors,
  refractory, clusters, filters and censor counts;
- null cannot copy any observed memory/candidate/cluster/admission identity;
- missingness and denominator invariants;
- selection cannot read observed counts;
- held-out date cannot enter selection;
- `NONE` folds contribute zero observed/null/raw counts and exposure;
- primary exact zero routes to A-1-5;
- sensitivity exact zero routes to A-1-6;
- raw exact zero fails A-1-7 without becoming integrity corruption;
- negative/NaN/inf count, exposure and rate route to A-1-4;
- exposure unit conversion;
- comparison censor cannot alter causal candidates;
- artificial slice/reset invariance;
- Build A/B preseal, pending and final equality;
- exact Required Outputs and manifest closure;
- source/controller/cache inventory blob mutation fails authority;
- every inherited callable AST/callable binding mutation fails authority;
- poison future/unconsumed fields changes no output.

## 22. Execution Lock And Next Authority

Revision 2 is not execution authority.

Before any 29-cache run:

```text
independent plan review must close at P0/P1/P2/P3 = 0/0/0/0
plan SHA256 must be frozen
formal task must bind that SHA
```

Until then:

```text
data execution = locked
future outcome access = false
A0 drafting = false
A0 execution = false
live trading = false
```

If A-1 passes, it authorizes only drafting a separately reviewed exploratory
A0 contract. It does not authorize reading future outcomes.
