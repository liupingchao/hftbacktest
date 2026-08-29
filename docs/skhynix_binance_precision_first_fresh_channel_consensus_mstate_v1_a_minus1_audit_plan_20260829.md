# SKHYNIX Binance Precision-First Fresh-Channel Consensus M-State V1
# A-1 Structural False-Positive Control Audit Plan

Date: 2026-08-29

Task: `0829T002`

Hypothesis ID: `FRESH_CHANNEL_CONSENSUS_MSTATE_V1`

Audit ID: `FRESH_CHANNEL_CONSENSUS_MSTATE_V1_A_MINUS1`

Status: candidate contract; data execution locked

Revision: 1

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
landmarks whose structural-null false-cluster burden is below the frozen
precision-first ceiling?
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

The exact cache authority, source inventory, session-role ledger, schema and
consumed-field whitelist are inherited from accepted task `0829T001`.

Allowed cache fields remain exactly those frozen by `0829T001`.

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

The runner must verify and directly call the accepted implementations for:

```text
base_masks
build_features
fixed_opposite_orientation_pairs
null_layout
permute_trade_direction_paths
prior_count
run_length
```

Normalized AST SHA256 values must be frozen before execution. Reimplementation
or callable rebinding is a hard authority failure.

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
observation:

```text
strong_{c,d,m}(t) =
  finite(r100_c(t))
  and finite(r500_c(t))
  and d * r100_c(t) >= 0.50 + m
  and d * r500_c(t) >= 0.25 + m
```

The channel observation is:

```text
O_{c,m}(t) =
  +1       if strong_{c,+1,m}(t)
  -1       if strong_{c,-1,m}(t)
   0       if both ratios are finite but neither direction is strong
  UNKNOWN  otherwise
```

The positive and negative conditions must be mutually exclusive. Any overlap
is an integrity failure.

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
2. A finite `+1`, `-1` or `0` observation overwrites that channel memory and
   stores its observation timestamp and event sequence.
3. `UNKNOWN` does not overwrite the memory.
4. A stored memory is usable only while:

```text
0 <= t - last_observed_at_c <= tau
```

5. Once its age exceeds `tau`, the channel state is `UNKNOWN`.
6. A neutral `0` observation invalidates an older directional state
   immediately. It may not be skipped in favor of an older sign.
7. An opposite observation invalidates the previous direction immediately.
8. No memory may cross capture, segment, reconnect or quality boundaries.

This is bounded causal carry, not forward filling. The TTL is part of the
registered state model and is never inferred from later data.

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

The M-state must be recomputed from the randomized causal features. Channel
memory may not be copied from the observed run.

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
4. Reject filters with zero, negative, non-finite or inconsistent exposure.
5. Admit filters whose Type-7 p95 null false-cluster rate is at most:

```text
0.10 per comparison-supported capture hour
```

6. Select the first admitted filter in frozen least-restrictive-first order:

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

`0 observed / 0 null` is not estimable and may not be called high precision.

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
- zero sign overlap;
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
- all counts, rates and exposures non-negative and finite;
- zero/non-finite semantics fail closed;
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
- neutral immediately overwrites sign;
- unknown retains sign only within TTL;
- expiry at `tau + one checkpoint`;
- reset/segment clears memory;
- no cross-capture carry;
- `ABSTAIN -> SIGNAL` cannot anchor;
- six-checkpoint background prestate is exact;
- stricter filters cannot create common anchors;
- all comparable filter candidate/cluster subsets are monotonic;
- null recomputes M-state rather than copying observed memory;
- missingness and denominator invariants;
- selection cannot read observed counts;
- held-out date cannot enter selection;
- zero/negative/NaN/inf exposure and rate routing;
- exposure unit conversion;
- comparison censor cannot alter causal candidates;
- artificial slice/reset invariance;
- Build A/B preseal, pending and final equality;
- exact Required Outputs and manifest closure;
- poison future/unconsumed fields changes no output.

## 22. Execution Lock And Next Authority

Revision 1 is not execution authority.

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
