# SKHYNIX Binance FLOW_COHERENCE_TRANSITION_V1 A-1 Primitive Support Audit - 2026-08-28

Date: 2026-08-28

Status:

```text
revision 6 independently accepted
execution requires task 0828T014
```

Hypothesis identifier:

```text
FLOW_COHERENCE_TRANSITION_V1
```

Audit identifier:

```text
FLOW_COHERENCE_TRANSITION_V1_A_MINUS1
```

Predecessor:

```text
FLOW_INTERNAL_DIRECTIONAL_ALPHA_V1
accepted execution status
rejected_at_A0 scientific status
```

## 1. Decision

Do not draft another complete A0 state machine yet.

Run one outcome-blind hypothesis-admission audit asking:

> Does active SKHYNIX order flow repeatedly move from a locally conflicting
> trade-depth configuration into persistent directional
> coherence, outside startup and reconnect recovery, with enough cross-date
> and dependence support to justify a separately reviewed A0 contract?

The proposed structural transition is:

```text
DISCORDANT_ACTIVE
  -> COHERENCE_CANDIDATE_d
  -> COHERENT_ACTIVE_d
```

The alignment point is the first causal checkpoint where persistence is known.
It is not backdated to the first raw crossing.

## 2. Why This Is A New Hypothesis

The failed predecessor required:

```text
500ms uninterrupted MIXED_ACTIVE_FLOW
  -> directional candidate
  -> 120ms persistence
```

Its accepted A0 evidence contained:

```text
33.897367 detector-ready hours
14.287006 active-flow hours
904,401 raw directional qualifying checkpoint-direction pairs
3 confirmed anchors
all 3 anchors in one 2026-08-24 reconnect-adjacent episode
```

The present audit does not lower the predecessor's mixed threshold. It changes
the primary object from:

```text
neutral or mixed flow -> directional flow
```

to:

```text
locally conflicting trade-depth evidence
  -> synchronized directional propagation
```

Changing this object requires a new hypothesis identifier.

## 3. Authority And Claim Boundary

This audit may decide only whether a complete A0 contract is worth drafting.

It may not claim:

- future price direction;
- continuation or reversal predictability;
- incremental H1 information;
- spread capture;
- maker or taker actionability;
- fill probability;
- fee-adjusted or realized PnL.

A positive audit authorizes only:

```text
draft_a0_contract = true
a0_execution_authorized = false
future_target_access_authorized = false
```

## 4. Source Authority

Use the accepted `0828T013` source and replay layer at commit:

```text
5603a670e617636b9994d605faef833164d3add4
```

Required inherited source:

```text
29 admitted Binance captures
9 research dates
2026-07-29 through 2026-08-27
20ms causal atomic replay cache
```

Frozen cache authority:

```text
docs/skhynix_flow_internal_directional_alpha_a0_cache_authority_20260828.csv
SHA256:
49bd38bb974e6bbda28db84e5116b850c1d75e234408d41fb429e08cc3b2300f
```

The audit must:

1. verify the tracked 29-row source inventory;
2. verify every raw file size and SHA256;
3. verify each primary and deterministic replay cache against the frozen
   cache-authority row;
4. bind every consumed cache size, SHA256, row count and schema;
5. copy or hard-link admitted cache bytes into the task-owned output root;
6. never consume a cache whose paired deterministic identity differs.

No new collection is allowed.

## 5. Historical Consumption Roles

Every existing date is historically consumed by hypothesis development or
predecessor review. None is an independent validation or prospective holdout.

The operational roles are:

```text
2026-07-29:
  historical_normalization_calibration

all other existing dates:
  historically_consumed_reused_structural_support
```

No existing date may be represented as independent validation. Even a passing
audit has the claim limit:

```text
historical_support_only_pending_prospective
```

## 6. Zero-Outcome Boundary

Permitted inputs end at each checkpoint:

- atomic public trade flow;
- atomic bid/ask depletion;
- atomic OFI and absolute OFI;
- current L1-L5 depth and OBI;
- current spread and midpoint for source validity diagnostics only;
- trailing activity;
- segment, reset, reconnect and quality metadata.

Forbidden reads and computations:

```text
future midpoint fields: []
future BBO fields: []
future return or markout fields: []
continuation/reversal targets materialized: false
barrier or horizon outcome scans: false
fill, fee, slippage or PnL fields: []
H0/H1/H2 fitted: false
model loss inspected: false
new collection: false
private or order access: false
```

Trailing midpoint returns from the predecessor are not used to define,
filter, rank or validate provisional coherence anchors.

## 7. Causal Grid And Features

Use the inherited 20ms right-open checkpoint grid.

For each window:

```text
W in {100ms, 200ms, 500ms, 1000ms, 2000ms}
```

calculate:

```text
r_trade(W) =
  signed aggressive trade quantity / total aggressive trade quantity

r_dep(W) =
  (ask depletion - bid depletion)
  / (ask depletion + bid depletion)

r_ofi(W) =
  signed OFI / absolute OFI
```

Zero denominator remains unavailable. No epsilon or floor is permitted.

For at least two available components:

```text
D(W) = median(r_trade(W), r_dep(W), r_ofi(W))
```

For at least one available depth component:

```text
D_depth(W) = median(r_dep(W), r_ofi(W))
```

Activity remains:

```text
activity_500ms =
  atomic admitted messages in [t-500ms,t)
```

The primary active threshold reuses the accepted calibration-only rule:

```text
Jul29 calibration Q60
expected accepted value = 44 messages/500ms
active tie rule = >=
```

## 8. Nuisance Exclusion

Every eligible audit checkpoint must satisfy:

```text
valid uncrossed L1-L5 book
complete contiguous 2000ms history
activity_500ms >= Q60
trade available plus at least one depth component at 100ms and 500ms
time from capture or segment start >= 30s
time from reconnect or reset >= 30s
no sequence or quality boundary in the trailing feature window
```

The 30-second cooldown is a transport-nuisance exclusion. It is not selected
from future price behavior.

Provisional candidates may be reported by distance to the next boundary for
geometry diagnostics, but future boundary distance cannot enter the causal
detector.

## 9. Reference Trade-Depth Coherence Predicate

For direction:

```text
d in {-1,+1}
```

Reference directional coherence:

```text
Q_d(t):
  d * r_trade(100ms) >= 0.50
  and at least one of:
    d * r_dep(100ms) >= 0.50
    d * r_ofi(100ms) >= 0.50
  and neither available depth component <= -0.50 in direction d
  and d * D(500ms) >= 0.25
```

Depth-split ambiguity:

```text
Q_ambiguous_d(t):
  d * r_trade(100ms) >= 0.50
  and one available depth component >= +0.50 in direction d
  and another available depth component <= -0.50 in direction d
```

`Q_ambiguous_d` makes `Q_d=false`. Because the trade component has one sign,
`Q_-1` and `Q_+1` are mutually exclusive by construction.

This prevents the mechanically related depletion/OFI pair from establishing
coherence without aggressive-trade participation.

## 10. Trade-Depth Conflict Pre-State

At an active checkpoint, define two non-exclusive primitive labels.

Component conflict:

```text
abs(r_trade(100ms)) >= 0.25
and at least one available depth component points opposite trade by >= 0.25
```

Horizon conflict is diagnostic only:

```text
abs(r_trade(100ms)) >= 0.25
abs(D_depth(500ms)) >= 0.10
sign(r_trade(100ms)) != sign(D_depth(500ms))
```

The primary conflict predicate is:

```text
Q_conflict = component_conflict
```

Generic `not Q_d`, neutral flow and weak unresolved flow are not conflict.
Absorption is not claimed by this audit because no replenishment or
flow-versus-price-response predicate is frozen.

For each primary conflict checkpoint, assign one descriptive subtype:

```text
depletion_only
ofi_only
depletion_and_ofi
```

`horizon_conflict` and the subtype composition are reported diagnostics. They
cannot admit an anchor or rescue failed primary support.

## 11. Provisional Candidate And Confirmation

Maintain at most one global candidate per capture. Open it on a rising edge of
`Q_d` only when the immediately preceding checkpoint is `Q_conflict=true`,
the contiguous conflict run ending at that preceding checkpoint is at least
120ms, and the previous 500ms causal window contains:

```text
active exposure >= 300ms
same-direction Q_d exposure <= 80ms
```

Scattered conflict exposure earlier in the 500ms window is insufficient. A
single 20ms neutral gap between conflict and the coherence rising edge makes
the checkpoint ineligible. This freezes a directly adjacent
`conflict -> coherence` transition rather than a generic recent-history
association.

The candidate window is 300ms.

The candidate checkpoint contributes zero persistence exposure.

After candidate creation, apply this precedence at every later checkpoint:

```text
1. quality or active support lost:
   cancel as support_lost

2. Q_-d true:
   cancel as opposite_coherence
   no opposite candidate may open at the same checkpoint

3. elapsed time >= 300ms before confirmation:
   cancel as timeout

4. Q_d true:
   add one complete 20ms exposure

5. otherwise:
   add zero exposure and keep candidate open
```

Confirmation requires:

```text
qualifying exposure >= 120ms within 300ms
```

The provisional alignment anchor is:

```text
coherence_confirmed_at =
  first checkpoint where 120ms qualifying exposure is causally known
```

It is never backdated.

While a candidate is open:

- same-direction rising edges are counted but do not open another candidate;
- an opposite candidate cannot coexist;
- confirmation closes the candidate.

Apply a direction-agnostic 1000ms refractory after confirmation. Any candidate
opening or confirmation inside refractory is suppressed and counted.
Refractory never extends unless a new admitted anchor is emitted.

## 12. Diagnostic Mechanism Families

The only primary family is:

```text
P1: discordant_active -> coherent_active_d
```

Two diagnostic labels may describe P1 anchors:

```text
D1 fast_to_medium:
  100ms coherence appears before 500ms coherence

D2 trade_depletion_to_ofi:
  trade and depletion agree before OFI joins direction d
```

D1 or D2 cannot replace or rescue failed P1 support. A diagnostic family that
appears stronger requires a separately registered hypothesis version.

## 13. Frozen One-Factor Parameter Envelope

Do not execute a Cartesian search.

Execute exactly these nine variants:

| ID | Fast | Medium | Component | Medium D | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| V0 | 100ms | 500ms | 0.50 | 0.25 | 120ms |
| V1 | 200ms | 500ms | 0.50 | 0.25 | 120ms |
| V2 | 100ms | 1000ms | 0.50 | 0.25 | 120ms |
| V3 | 100ms | 500ms | 0.40 | 0.25 | 120ms |
| V4 | 100ms | 500ms | 0.60 | 0.25 | 120ms |
| V5 | 100ms | 500ms | 0.50 | 0.15 | 120ms |
| V6 | 100ms | 500ms | 0.50 | 0.35 | 120ms |
| V7 | 100ms | 500ms | 0.50 | 0.25 | 80ms |
| V8 | 100ms | 500ms | 0.50 | 0.25 | 200ms |

V0 is the only admission-primary variant. Diagnostics cannot replace it.

Variant substitution is exact:

- `fast` replaces 100ms only inside `Q_d`, ambiguity and same-direction
  pre-exposure;
- `medium` replaces 500ms only inside the `Q_d` median condition;
- `component` replaces every `0.50` in `Q_d` and ambiguity;
- `medium D` replaces only the directional median threshold in `Q_d`;
- `persistence` replaces only confirmation exposure;
- conflict primitives, pre-state exposure, cooldown, candidate window and
  refractory remain at the V0 definitions for every variant.

The only adjacent one-factor comparisons are:

```text
(V0,V1), (V0,V2), ... , (V0,V8)
```

## 14. Structural Null

The null asks whether persistent trade-depth directional coherence exceeds
what would be observed if aggressive-trade direction were unrelated to the
depth-response direction while regime timing and depth mechanics were held
fixed.

Use fixed five-minute parent blocks and direction microblocks inside each
segment. Partial parent or microblocks are not null-comparable. Blocks begin
at segment start and are not wall-clock rounded.

For every microblock define one path-level orientation from atomic aggressive
trade flow:

```text
net(block) =
  float64 accumulator initialized to +0.0;
  add each finite float64 trade_signed_20ms value once in ascending
  checkpoint timestamp order, rounding to float64 after every addition

if any value is non-finite:
  fail closed

g(block) = sign(net(block)) using exact comparison with 0.0

if g(block) = 0:
  the microblock has no directional-information label
  and is not null-comparable

U(W,t) = g(block_of_t) * r_trade(W,t)
```

For a labeled block `g` is exactly `-1` or `+1`. `U(W,t)` retains the complete
target trade path relative to its own block orientation, including all
within-block sign changes, absolute magnitudes, zeros, missingness and
cross-horizon relative structure.

For each five-minute parent and chosen microblock duration:

1. summarize every microblock by median `log1p(activity_500ms)`,
   active-checkpoint fraction, median `log1p(trade_qty_100ms)`, median
   `log1p(depletion_denominator_100ms)`, median `log1p(ofi_abs_100ms)`,
   nonzero-fast-trade fraction and complete trade-bundle availability
   fraction;
2. convert the four continuous intensity summaries to within-parent
   fractional ranks in `[0,1]` using average tie ranks divided by `n-1`; a
   constant column receives rank `0.5`;
3. define joint matching distance as Euclidean distance over the four ranks
   and the three untransformed fractions;
4. create edges only between one `g=+1` and one `g=-1` microblock, and admit
   an edge only if every intensity-rank difference is
   `<=0.40`, active/nonzero fraction differences are each `<=0.15`, and
   availability-fraction difference is `<=0.10`;
5. compute the fixed bipartite matching with this exact bitmask dynamic
   program:
   - use the smaller orientation side as the bitmask side; ties put `g=-1`
     on the bitmask side;
   - visit the other side in ascending microblock start timestamp;
   - each state may leave that block unmatched or match one unused admitted
     opposite block;
   - quantize each joint distance as
     `round_half_even(distance * 1e12)` to an integer;
   - compare complete solutions by, in order:
     maximum pair count, minimum summed quantized distance, then
     lexicographically smallest sorted tuple of
     `(plus_start_ts_ns,minus_start_ts_ns)`;
   - the selected complete solution is the unique fixed pairing;
6. unmatched, unlabeled or inadmissible microblocks are not null-comparable;
7. for each replicate and matched pair, independently draw
   `swap ~ Bernoulli(0.5)`;
8. if `swap=false`, retain both labels; if `swap=true`, exchange the pair's
   `+1/-1` labels;
9. reconstruct
   `r_trade_null(W,t)=g_assigned(block_of_t)*U(W,t)` and rerun exact V0.

This is an exact paired randomization. Conditional on the frozen pair
construction, its law is uniform over the `2^K` swap vectors for `K` matched
pairs. It preserves every target trade path's magnitude, zero mask,
missingness, denominator and within-block relative sign structure; preserves
the complete depth bundle; and conserves one positive and one negative label
inside every pair.

The registered H0 exchangeability assumption is:

```text
conditional on five-minute parent identity and the frozen pair-matching
variables, the positive/negative path-level orientation labels inside one
matched pair are exchangeable with respect to contemporaneous depth-response
direction.
```

For observed and randomized runs, comparison support is false within 2000ms
of every microblock boundary. The same mask applies to opening, confirmation,
persistence and dwell. Candidate or dwell terminates when support becomes
false.

For every date and replicate report:

```text
matched opposite-orientation pair count
median and p95 joint matching distance
p95 difference for each intensity rank and each fraction
pair label-count difference
target magnitude mismatches
target zero-mask mismatches
target missingness mismatches
target denominator mismatches
```

The null fails closed if a pair does not retain one positive and one negative
label; any target invariant changes; any caliper is exceeded; or p95 joint
distance exceeds `0.60`.

Run the complete algorithm for:

```text
10s root: SeedSequence([20260828,10000])
30s root: SeedSequence([20260828,30000])  # primary
60s root: SeedSequence([20260828,60000])
PRNG: numpy PCG64
replicates per duration: 199
per-capture stream:
  SeedSequence([20260828,microblock_ms,replicate_id,capture_ordinal])
within-stream draw order:
  segment_id, parent_id, matched_pair_id
```

Identifiers and ordering:

```text
capture_ordinal =
  zero-based row number after bytewise-ASCII sorting the 29 cache_name values

parent_id =
  (segment_id, zero-based five-minute parent number within segment)

matched_pair_id =
  zero-based row number after sorting selected pairs by
  (plus_start_ts_ns,minus_start_ts_ns)

fingerprint row order =
  capture_ordinal, segment_id, parent_number, matched_pair_id
```

At each duration require all count, dwell, date, balance and conservation
conditions in Gate A-1-5. The 10s and 60s paths may invalidate but never rescue
the 30s primary. Insufficient support at any duration is
`Aminus1_structural_null_not_admissible`; failed separation at any duration is
`Aminus1_structural_null_not_rejected`.

Assignment-diversity gate:

```text
distinct aggregate swap-vector fingerprints across 199 replicates >= 190
```

The fingerprint is SHA256 over ordered
`(capture_ordinal,segment_id,parent_id,pair_id,swap)` rows. Ties fail.

The structural null reads no future price.

Independent plan review:

```text
round = 6
P0/P1/P2/P3 = 0/0/0/0
recommendation = PASS
data execution lock = released
```

## 15. Frozen Metric Estimators

Detector-ready interval:

```text
one 20ms interval ending at a checkpoint that passes book, history,
segment, quality and 30s cooldown requirements, before applying activity
```

Durations:

```text
detector_ready_hours =
  detector-ready intervals * 20ms / 3.6e6ms

active_flow_hours =
  detector-ready intervals also passing activity and
  trade-plus-depth availability
  * 20ms / 3.6e6ms
```

All anchor-rate gates use `detector_ready_hours`.

Availability estimator:

```text
denominator =
  detector-ready checkpoints passing activity before component availability

numerator =
  denominator checkpoints where trade and >=1 depth component
  are available at both 100ms and 500ms

availability = numerator / denominator
```

Overall and every per-date estimate use this exact formula. A zero denominator
is unavailable and fails the feature-support gate.

Coherence dwell for one confirmed anchor:

```text
continuous complete 20ms intervals beginning at confirmation for which:
  eligible active support is true
  Q_d is true
  null-comparison support is true when evaluating the structural null

stop before the first interval where either condition is false
```

The confirmation checkpoint contributes the first 20ms dwell interval.

Null quantiles use Type-7 linear interpolation:

```text
sort n values
h=(n-1)*q
linearly interpolate floor(h),ceil(h)
```

Observed-versus-null comparisons are strict `>`. Ties fail.

## 16. Slice And Reset Invariance

For every segment long enough for testing, create artificial starts every
10 minutes.

Rerun the detector on the suffix. Compare full-run and suffix-run anchors only
after:

```text
artificial_start + 32s
```

The 32-second guard is:

```text
30s cooldown + 2000ms maximum feature history
```

Required:

```text
anchor identity and timestamp equality after guard: exact
variant metrics after guard: exact
```

This prevents detector initialization from becoming the alignment mechanism.

## 17. Dependence And Compression

Assign:

```text
dependence_cluster_id =
  (capture_id, floor(anchor_ts / 30s))
```

Report:

- raw `Q_d` checkpoint-direction pairs;
- rising-edge candidates;
- confirmed provisional anchors;
- raw-to-anchor compression;
- inter-anchor distribution;
- pre-refractory confirmations and their inter-confirmation distribution;
- maximum same-capture 5s pre-refractory confirmation burst;
- anchors per 30s cluster;
- direction balance;
- date concentration;
- component-conflict subtype composition and diagnostic horizon conflict.

Row-IID effective sample size is forbidden.

Pre-refractory confirmations come from a parallel V0 shadow detector. It is
identical to admitted V0 in candidate concurrency, precedence, timeout,
persistence and confirmation semantics, except:

```text
refractory duration = 0ms
```

Inter-confirmation intervals are computed only inside the same capture and
segment. Shadow confirmations never enter the admitted anchor ledger.

## 18. Admission Gates

### Gate A-1-0: Source And Cache Closure

Require exact source inventory, raw size/SHA closure, paired cache identity,
schema closure and deterministic artifact build.

### Gate A-1-1: Zero-Outcome Boundary

Require every forbidden field and action in Section 6 to remain empty or false.

### Gate A-1-2: Nuisance And Feature Support

Require:

```text
anchors inside 30s startup/reconnect cooldown: 0
pre-exclusion anchors inside cooldown / all pre-exclusion anchors <= 0.10
cross-segment or cross-quality feature windows: 0
ratio bound violations: 0
denominator substitutions: 0
overall active trade-plus-depth availability >= 0.90
minimum per-date trade-plus-depth availability >= 0.80
```

The pre-exclusion detector differs only by replacing the 30s cooldown with the
minimum 2000ms feature-history requirement. If it has zero anchors, define the
cooldown-zone share as zero; the reference-support gate still evaluates the
post-exclusion anchor count normally.

### Gate A-1-3: Reference Structural Support

Require V0:

```text
confirmed provisional anchors >= 500
represented dates >= 7
minimum anchors per represented date >= 20
anchor rate between 5 and 150 per detector-ready hour
maximum single-date share <= 0.30
minority direction share >= 0.25
unique 30s anchor clusters >= 150
maximum single-cluster share <= 0.05
pre-refractory median inter-confirmation interval >= 2000ms
maximum same-capture 5s pre-refractory confirmation burst <= 4
admitted anchors / pre-refractory confirmations >= 0.70
```

### Gate A-1-4: Parameter Stability

Require:

```text
at least 6 of 9 variants satisfy:
  represented dates >= 7
  maximum single-date share <= 0.35
  minority direction share >= 0.20
  anchor rate between 2 and 300 per detector-ready hour

maximum adjacent one-factor anchor-count ratio <= 5
minimum adjacent one-factor anchor-count ratio >= 0.20
```

### Gate A-1-5: Structural Null Separation

Require:

```text
null-comparable observed V0 anchor count > null p95
null-comparable observed V0 median coherence dwell > null p95
dates with null-comparable observed count > date-specific null p90 >= 6
null activity matching diagnostics pass
observed and null boundary-censor violations = 0
```

Apply that complete set independently to 10s, 30s and 60s. Also require:

```text
comparable_active_coverage =
  eligible active intervals inside paired microblock interiors
  / all eligible active intervals before null pairing and boundary censor

30s overall/minimum per-date comparable active coverage >= 0.60/0.50
10s overall/minimum per-date comparable active coverage >= 0.35/0.25
60s overall/minimum per-date comparable active coverage >= 0.70/0.60
matched opposite-orientation pairs >= 50 overall at each duration
matched pairs >= 3 per active-support date at each duration
null-comparable observed anchors >= 200 at each duration
null-comparable observed represented dates >= 7 at each duration
distinct aggregate swap-vector fingerprints >= 190 at each duration
```

### Gate A-1-6: Slice And Dependence Support

Require:

```text
slice invariance mismatches: 0
unique artificial starts tested >= 10
unique 30s anchor clusters >= 150
maximum cluster share <= 0.05
```

## 19. Classifications

Passing:

```text
Aminus1_historical_flow_coherence_support_supported
```

Failures:

```text
Aminus1_source_not_admissible
Aminus1_zero_outcome_boundary_violated
Aminus1_nuisance_dominated
Aminus1_feature_support_failed
Aminus1_primitive_support_sparse
Aminus1_transition_near_continuous
Aminus1_transition_date_concentrated
Aminus1_structural_null_not_rejected
Aminus1_structural_null_not_admissible
Aminus1_parameter_unstable
Aminus1_detector_not_slice_invariant
```

Evaluate every gate and publish every failed atomic condition. Choose the
unique primary classification by the first failed gate in written order, with
the following semantic refinements:

```text
A-1-2 startup/reconnect contamination
  -> Aminus1_nuisance_dominated

A-1-3 too few anchors or dates
  -> Aminus1_primitive_support_sparse

A-1-3 excessive rate, pre-refractory burst or short pre-refractory gaps
  -> Aminus1_transition_near_continuous

A-1-3 date concentration
  -> Aminus1_transition_date_concentrated

A-1-4 failure
  -> Aminus1_parameter_unstable

A-1-5 support, conservation or balance failure
  -> Aminus1_structural_null_not_admissible

A-1-5 separation failure
  -> Aminus1_structural_null_not_rejected

A-1-6 slice mismatch
  -> Aminus1_detector_not_slice_invariant
```

Passing still writes:

```text
claim_limit = historical_support_only_pending_prospective
prospective_validation_required = true
```

## 20. Required Outputs

```text
contracts/
  source_cache_contract.json
  primitive_contract.json
  nuisance_contract.json
  variant_contract.json
  structural_null_contract.json
  slice_invariance_contract.json
  gate_contract.json
  outcome_access_ledger.json
  session_role_ledger.csv

support/
  source_cache_inventory.csv
  feature_support_by_date.csv
  primitive_exposure_by_date.csv
  provisional_anchor_ledger.csv
  provisional_anchor_support_by_date.csv
  primitive_family_composition.csv
  direction_balance_by_date.csv
  compression_by_date.csv
  inter_anchor_distribution.csv
  dependence_cluster_support.csv
  parameter_stability.csv
  structural_null_summary.csv
  structural_null_by_date.csv
  structural_null_balance_by_date_replicate.csv
  structural_null_support_by_duration_date.csv
  pre_refractory_confirmation_support.csv
  slice_invariance.csv

reports/
  A_minus1_summary.json

classification.json
run_manifest.json
```

Large cache and replicate ledgers remain local. Compact contracts, summaries,
support tables and exact hashes are tracked.

## 21. Stop Rule

If the audit fails:

```text
draft_a0_contract = false
a0_execution_authorized = false
future_target_access_authorized = false
```

Do not rescue the hypothesis by:

- reducing cooldown;
- selecting only 2026-08-24;
- choosing the most favorable variant;
- dropping structural-null replicates;
- replacing P1 with D1 or D2;
- inspecting future price response.

If the audit passes, the next task is only to draft and independently review a
complete A0 contract.
