# SKHYNIX Binance Precision-First Fixed Causal Epoch M-State V2
# A-1 Structural False-Positive Control Audit Plan

Date: 2026-08-29

Task: `0829T003`

Hypothesis ID: `FIXED_CAUSAL_EPOCH_MSTATE_V2`

Audit ID: `FIXED_CAUSAL_EPOCH_MSTATE_V2_A_MINUS1`

Status: candidate contract Revision 4; data execution locked

## 1. Decision Context

Accepted predecessor `0829T002 / FRESH_CHANNEL_CONSENSUS_MSTATE_V1`
finished with:

```text
classification = Aminus1_mstate_integrity_failed
common anchors = 3,783
true slice/reset identity mismatches = 8
30s cross-fitted observed clusters = 0
30s conditional-null p95 false-cluster rate = 1.1411327645/hour
```

The M-state representation reduced simultaneous complete-path support
dependence, but its accepted-anchor-driven 30s refractory was a renewal
process:

```text
accept anchor
-> suppress next 30s
-> accept first later onset
```

When replay starts from a different point, the first accepted anchor changes.
Dense natural onsets can preserve that phase difference indefinitely.

This successor changes only anchor thinning and dependence identity. It does
not:

- lower the `0.50 / 0.25` directional thresholds;
- change three-channel consensus to two-of-three;
- convert UNKNOWN to BACKGROUND;
- inspect future prices, outcomes, fills, fees or PnL;
- tune epoch geometry using observed candidate counts.

## 2. Research Question

```text
After replacing renewal refractory and renewal clustering with a fixed,
absolute-time causal epoch contract, does the same fresh-channel M-state
produce reset-invariant, sparse landmarks whose conditional structural-null
false-cluster burden satisfies the frozen precision-first gates?
```

This is still an outcome-blind A-1 audit. Passing authorizes only drafting a
separately reviewed A0 contract.

All nine research dates have already participated in repeated hypothesis
revision. They are historical development data, not a prospective holdout.
The strongest positive claim available here is:

```text
historical structural false-fire-control candidate
```

No pass may be described as prospective precision, economic value, live
capture probability or deployability.

## 3. Frozen Predecessor And Data Authority

Accepted predecessor:

```text
QA/controller commit =
  460f649063cf2e344f6db29f4855821c97e68ab9

accepted implementation commit =
  094ad7b55f7fa2f1cf4ba8d9fdcdfbdfb63911c1

runner path =
  examples/hyperliquid/
  skhynix_fresh_channel_consensus_mstate_a_minus1.py

runner Git blob OID =
  4e48d1262d408c6c98e4b58057ff574416ce9fe4

runner SHA256 =
  8a9ce6ed18c32f0027e700cfa795cebe6de9285643c80ecdd8d7c31dfaf9156a
```

The successor must verify ancestry, blob OID, whole-file SHA and directly bind
the accepted causal primitives. Normalized AST SHA256:

```text
source_preflight       9089e2f348965c6601d23315be625f333558b688dda4318b034fff38d80d57a6
base_eligibility       c0121463187a6678679059b6b8cf9d2948a525fb14c5df0bb25377bce7d7da6a
channel_actions        0dc86f04ff18fe490f6f7c2f6332acadc8d441c68deb798e4829bfae5d0277ab
channel_memory_family  c3ee8828cefbbb7ec5b4163fa456a08ad4af12a2f4f9a9d0712f7035920fb501
aggregate_mstate       0eb6a67e685c8f4a65d7f48316010f4b7448788857da29eb758eadacd525e9e0
build_mstate_family    17fce29166a82d70b5896929f62feab924a31a1fd4e2e9f0e0e4d2652ffdfde2
natural_onset_mask     13a369282bd36c5a13a8a183ea427d38a5b98640e8c5a768f16bafcdbb09af8d
evaluate_filter_family de30351c480f7dd72850f995d0d3dc8e09fe1f516f098e4b70bfa2a13addaf8f
interval_all_mask      c1be90232e704c7e86bb8fc354f621c1b78ddc14a511332b59d1f93b7d990868
stream_root            c39e6f07ba0ab630785b9737ca3ba051bf3fd9971c29f3bdbdd013df1fed121b
rng_for                8ce7fb9579622cc6762b216b2972eb562c5918776ef135bc623558bd3db504ce
select_filters         98ebcfbfed5bbd4f6aa37dfc9d112347ceef15990f47dcbc8a47184a0e6903ef
estimator              04e8bfcb0178a34ef926809eae9a242e39bdcb2089fc62954d9932c84dd883e7
pair_identities         57926d61e747c162ba68722ab621c31d5089abcba218ca4487a15745b7d4fcaf
pair_distances          afb245ece524c3a80090da9ec0cb164e3b88daa7a24e298b02713d3166e56ae1
```

The conditional-null implementation is also bound directly:

```text
null authority commit =
  45544ecc3901623ca7c2e34a059afca6c551d625

null authority path =
  examples/hyperliquid/skhynix_flow_coherence_a_minus1_audit.py

null authority blob OID =
  494c203e7195f292e057f7708c99f52096259a02

null authority SHA256 =
  f7dc1565bf0a45363dadf3204d827e0d13687f6cc3307c2e7c5e77aeb321400c
```

Normalized null callable AST SHA256:

```text
base_masks                        bc2155a38bd1707fcdb77bdebea611da3889934a47d415bdfd0d5a95842d7114
build_features                    e5cca6c2b7627ef8e3719e4fdecb5a540a42a2028e2fb141ea9fff4f7c246933
fixed_opposite_orientation_pairs  04cef064fdaf5cba94421d6d3250760ccf623b2d0531a883154d2a3bdb4b293d
null_layout                       2def320606fa9caf45e5878845e91026b1284b57b1dd3ff8265038de92c8dcf5
permute_trade_direction_paths     b870945f3a079f34337912776001c8bbe8af41c2644e2c0b4acf76277e7637ce
prior_count                       abbd39099a876c70713b05871152fdfa56d1ee8d142611debe849f1afcc018a8
run_length                        074d37d93e2b66de5d94c9112494fd3f615a5f24aa3ce77a2f5da64a5b37e37d
```

The successor must direct-call the bound M-state and null primitives.
Reimplementation, transitive rebinding or fallback import is an A-1-0
authority failure.

Source cache root:

```text
/Users/liu/Documents/
hftbacktest-0829t002-fresh-channel-consensus-mstate-a-minus1/
local_live_analysis/
skhynix_fresh_channel_consensus_mstate_a_minus1_0829T002/cache
```

Authority:

```text
29 caches / 9 dates / cache schema v4
inventory file SHA256 =
  e6f8f3fedb76eeed6d99cb8cb5306732af54f20bcb0882b56983dc61273a39e1
typed inventory payload SHA256 =
  e554793e98d9a000b1b8c0049897ed42a14167d07f16b1602acfcbb2f048b12c
predecessor classification SHA256 =
  d28ba2875ff382765c1ed2603b18a2bef440077677db07e4fe50ce68e3c9a2f0
predecessor summary SHA256 =
  9613375bb221085f10f8a34d3d13094d05ff4e9d1c8f13d3f64b8830aa289ddd
```

Allowed and consumed cache fields remain exactly those frozen in `0829T002`.
Midpoint, OBI, spread, depth snapshots and future/economic fields remain
unconsumed.

## 4. Inherited M-State

Channels:

```text
trade, depletion, ofi
```

New-evidence masks:

```text
trade:      trade_total > 0
depletion:  bid_depletion + ask_depletion > 0
ofi:        ofi_abs > 0
```

The six channel actions, TTL memory rules and four aggregate states are
unchanged:

```text
actions:
  GLOBAL_INVALID, NEW_INVALID, NEW_POS, NEW_NEG, NEW_NEUTRAL, NO_UPDATE

memory:
  -1, 0, +1, UNKNOWN

M-state:
  SIGNAL_NEG, BACKGROUND, SIGNAL_POS, ABSTAIN
```

All raw-source preflight, no-refresh, neutral overwrite, expiry, reset and
segment-boundary integrity rules from the accepted predecessor remain exact.

Filter family remains:

```text
TTL:          100ms, 60ms, 40ms
persistence:  200ms, 400ms, 800ms
margin:       0.00, 0.10, 0.20
total:        27 filters
```

Natural onset remains:

```text
M_{tau,m}(t) = SIGNAL_d
and immediately preceding 120ms under M_{tau,m} are all BACKGROUND
```

The common natural-onset ledger uses `(TTL=100ms, margin=0.00)`.

## 5. Fixed Causal Epoch Contract

Absolute epoch origin:

```text
origin_ns = 0  # Unix epoch
epoch_width = 60s
epoch_id(t) = floor(ts_ns(t) / 60_000_000_000)
epoch_start_ns(t) = epoch_id(t) * 60_000_000_000
```

Eligible central core:

```text
core_open_ns  = epoch_start_ns + 15s
core_close_ns = epoch_start_ns + 45s
eligible iff core_open_ns <= ts_ns < core_close_ns
```

The two 15s edge guards are intentional false-negative zones. They ensure
that eligible cores in adjacent epochs are separated by at least 30s.

### 5.1 Complete Single-Segment Epoch

Epoch universe for a capture is every integer epoch ID from:

```text
floor(capture_first_ts_ns / 60s)
through floor(capture_last_ts_ns / 60s), inclusive
```

An intermediate epoch with zero checkpoints still receives one evidence row.
For epoch `e`, define the ordered expected set:

```text
E_e = [epoch_start_ns + j * 20ms for j in 0..2999]
O_e = every observed timestamp satisfying
      epoch_start_ns <= ts_ns < epoch_end_ns
```

An epoch is structurally eligible only if its entire 60s checkpoint grid is
present in one capture and one segment:

```text
expected checkpoints = 60s / 20ms = 3,000
first timestamp       = epoch_start_ns
last timestamp        = epoch_end_ns - 20ms
all adjacent deltas   = 20ms
all segment_id values = one identical segment
```

Assign exactly one disposition using this ordered `if/elif` partition:

```text
partial_capture_start
  iff capture_first_ts_ns > epoch_start_ns

partial_capture_end
  iff not partial_capture_start
  and capture_last_ts_ns < epoch_end_ns - 20ms

missing_checkpoint
  iff neither partial predicate is true
  and O_e is a strictly increasing proper subsequence of E_e

irregular_checkpoint
  iff no earlier predicate is true
  and O_e is neither a strictly increasing proper subsequence of E_e
  nor exactly equal to the ordered E_e

segment_boundary
  iff no earlier predicate is true
  and O_e == E_e in raw row order
  and cardinality(unique segment_id in O_e) != 1

eligible
  iff no earlier predicate is true
  and O_e == E_e in raw row order
  and cardinality(unique segment_id in O_e) == 1
```

`O_e` is never sorted before these predicates. A permutation, duplicate,
non-increasing timestamp or wrong adjacent delta is therefore irregular.
The predicates are evaluated over raw timestamps before M-state analysis.
Duplicate and off-grid counts are retained even if an earlier disposition
wins. No anchor, exposure checkpoint, observed cluster or null cluster may
come from an ineligible epoch. A reset inside the core therefore removes the
epoch; it never creates a second per-segment opportunity.

Common thinning:

```text
for each structurally eligible capture and epoch_id, and each direction:
  retain the earliest common natural onset in the eligible core
  suppress every later onset in the same key
```

Tie-break:

```text
(candidate_ts_ns, candidate_event_seq)
```

At most two anchors may exist per epoch: one per direction.

Dependence cluster:

```text
cluster_id = capture_id : epoch_id
```

The retained candidate still records the unique epoch segment for audit, but
segment is not a cluster discriminator. Both directions in the same epoch
share one cluster. No accepted-anchor timestamp participates in suppression
or cluster identity.

Consequences:

- slicing before an epoch cannot change decisions in later comparable
  complete epochs;
- process restart cannot permanently shift thinning phase;
- adjacent counted clusters have at least 30s temporal separation;
- an event near an epoch edge is deliberately omitted rather than assigned
  to a potentially under-clustered boundary pair.

The epoch origin, width, guards and earliest-onset rule are frozen. No
alternative offset, width, core or tie-break may be selected from observed
counts.

## 6. Filter Confirmation

The 27 filters remain a common-anchor-conditioned delete-only family.

For retained common anchor `(t,d)`, filter `(tau,p,m)` admits only if:

1. Its own previous 120ms M-state is all BACKGROUND.
2. Its M-state at `t` is `SIGNAL_d`.
3. Every checkpoint in `(t,t+p]` is `SIGNAL_d`.
4. The complete interval stays in the same segment.

Cancellation order is unchanged:

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

Strict natural onsets absent from the common ledger remain orphan diagnostics
and cannot enter selection or estimators.

## 7. Slice And Reset Invariance

Artificial start schedule is deterministic within each original segment:

```text
stride = 600s
K_segment = {
  k in positive integers:
  first_segment_ts_ns + k * 600s <= last_segment_ts_ns
}
nominal start(k) = first_segment_ts_ns + k * 600s
actual start index = searchsorted(ts_ns, nominal_start, side="left")
```

`K_segment` is computed once from the full preflighted arrays; iteration ends
after its largest member. There is no open-ended loop and no early stop after
a skipped start.

Skip a nominal start when the actual index is absent or belongs to another
segment.

The slice is rebuilt from raw cache authority, never from full-run derived
features or M-states:

1. Source-preflight the full raw cache.
2. Slice `[actual_start_index:n)` for exactly these row-aligned fields:

```text
activity, ask_depletion, ask_depth, bid_depletion, bid_depth,
event_seq, midpoint, obi, ofi, ofi_abs, ready, segment_id,
spread_ticks, trade_signed, trade_total, ts_ns, valid_book
```

3. Copy exactly these non-row-aligned fields unchanged:

```text
bin_boundary_violations, cache_schema_version,
initial_bridge_failure_count, non_admitted_message_contributions,
quality_boundary_count, reset_count, segment_end_ids, segment_end_ts,
sequence_gap_count, tick_size
```

4. Preserve field names, dtypes and trailing shapes exactly.
5. Write the sliced raw cache in a temporary root excluded from outputs.
6. Direct-call the bound `build_features(sliced_cache_path)`.
7. Run source preflight and the full M-state/epoch pipeline again.

The slice-source identity is canonical JSON over sorted tuples:

```text
(field_name, dtype.str, shape_as_integer_list,
 SHA256(C-contiguous raw bytes))
```

using `sort_keys=True,separators=(",",":"),ensure_ascii=True`.
Reusing or slicing full-run rolling features is prohibited.
Full and sliced capture endpoints are always the corresponding preflighted
`ts_ns[0]` and `ts_ns[-1]`; copied metadata may not supply an endpoint.

Define:

```text
guard_ns = 122s
comparison_floor_ns = actual_start_ts_ns + guard_ns
first_comparable_epoch_id =
  ceil(comparison_floor_ns / 60s)
```

The nominal start is qualifying only when at least one epoch satisfies:

```text
epoch_id >= first_comparable_epoch_id
the full epoch is structurally eligible in both full and sliced analyses
the full epoch belongs to the artificial-start segment
```

Otherwise the nominal start is deterministically skipped and counted as
`no_comparable_epoch`. Partial final epochs and every later segment are
excluded symmetrically from both expected and actual sets. Candidate identity
is:

```text
(capture_id, epoch_id, candidate_ts_ns, candidate_event_seq,
 direction, filter_id, epoch_cluster_id)
```

M-state support identity includes all four states and is compared
checkpoint-by-checkpoint for every filter on the union of comparable epoch
cores:

```text
(capture_id, epoch_id, checkpoint_ts_ns, filter_index, mstate_int)
```

where `mstate_int` is exactly `-1,0,1,2`. There is one tuple per checkpoint
and filter; no deduplication is permitted. Typed sort order is:

```text
capture_id ASCII, epoch_id numeric, checkpoint_ts_ns numeric,
filter_index numeric, mstate_int numeric
```

Canonical JSON uses
`sort_keys=True,separators=(",",":"),ensure_ascii=True`.

One CSV row is emitted only per qualifying artificial start with exact
expected/actual candidate identity hashes, counts, support hashes/counts,
comparable epoch count and mismatch reason. Absent/wrong-segment and
no-comparable starts appear only in the exact summary counters, never as CSV
rows. Required:

```text
zero identity mismatch
zero M-state support identity mismatch
zero row with any compared checkpoint outside the artificial-start segment
at least one qualifying artificial start on at least four research dates
at least 30 distinct comparable epochs globally
positive compared support checkpoint count
```

Failure of the last three requirements is an A-1-2 integrity/support audit
failure, never a vacuous pass.

Distinct comparable epoch identity is exactly `(capture_id, epoch_id)`,
deduplicated across artificial-start rows.
`compared_support_checkpoint_count` counts unique
`(capture_id, epoch_id, checkpoint_ts_ns)` identities, also deduplicated
across rows. Per-row `expected_support_tuple_count` and
`actual_support_tuple_count` count the 27 filter-state tuples and therefore
equal `27 * comparable_core_checkpoint_count`.

## 8. Outcome-Blind Structural Null

Direct-call the bound paired opposite-orientation path-swap null:

```text
five-minute parents
activity/intensity-matched opposite-orientation microblock pairs
independent Bernoulli(0.5) swap per fixed pair
trade magnitude, zero mask, missingness and denominators invariant
depth bundle and activity unchanged
```

The exact conditional null is:

```text
H0_conditional:
  conditional on the observed depth bundle, OFI/depletion path,
  activity/intensity path, missingness, denominators, segment boundaries,
  complete-epoch eligibility and fixed matched-pair structure,
  trade orientation labels are exchangeable within each registered
  opposite-orientation pair.
```

It tests whether trade orientation aligns with the fixed depth/OFI path more
often than this conditional exchangeability law permits. It is not an
unconditional market false-positive rate, economic loss probability or live
fill claim.

Banks:

```text
selection: 30s, 199 replicates, bank_code=5
evaluation: 10s/30s/60s, 199 replicates each, bank_code=6
SeedSequence root:
  [20260829, bank_code, duration_ms, replicate_id, capture_ordinal]
```

Every replicate recomputes:

```text
channel actions
-> memories
-> M-states
-> natural onsets
-> fixed epoch eligibility/thinning
-> epoch cluster IDs
-> 27 confirmations
-> external censor
-> cluster counts
```

No observed memory, onset, retained anchor, epoch admission or count may be
copied into a null replicate.

Required replicate invariants:

- exact magnitude, zero-mask, missingness and denominator preservation;
- exact three channel-specific new-evidence masks;
- exact pair identity and canonical pair ordering;
- exact comparison mask;
- exact complete-epoch eligibility and disposition;
- both directions deduplicate to one fixed epoch cluster;
- observed and null use the identical filter-duration exposure masks;
- at least 190 distinct fingerprints among 199 replicates;
- zero selection/evaluation stream overlap;
- minimum three valid date-pairs per duration;
- maximum date p95 joint distance at most 0.60.

All magnitude, zero-mask, missingness, denominator, new-evidence-mask,
pair-identity and comparison-mask invariants remain exact.

## 9. Exposure And Selection

The detector runs before the comparison mask is applied.

For filter `f`, duration `h`, checkpoint `t`, exposure requires:

```text
the containing epoch is structurally eligible
t is inside that epoch's fixed eligible core
all M-states in [t-120ms, t+p] are non-ABSTAIN
comparison mask is true throughout [t-120ms-500ms-tau, t+p]
```

Exposure is counted once per unique capture-time checkpoint.
The same epoch-eligibility mask is used by observed, selection-null,
evaluation-null and raw exposure. For every filter-duration-date, the
numerator candidate domain must be a subset of the exact denominator
checkpoint domain; observed and every null replicate must report the same
denominator identity hash.

Leave-one-date-out selection remains null-only:

```text
training dates = other eight dates
selection bank = independent 30s bank
threshold = Type-7 p95 null false-cluster rate <= 0.10/hour
order = largest TTL, shortest persistence, lowest margin, filter_id
no qualifying filter -> META_ABSTAIN
```

Observed counts may not influence selection.

## 10. Estimators And Gates

Estimator unit is the fixed epoch cluster.

Primary and sensitivity formulas, optional-value semantics and numeric
integrity rules remain those of `0829T002`.

Raw sparsity adds a support-conditioned epoch occupancy:

```text
raw_supported_epoch_count =
  distinct structurally eligible (capture_id, epoch_id) across non-NONE folds
  with at least one raw exposure checkpoint under the held-out selected filter

occupied_epoch_count =
  distinct raw-supported epoch clusters admitted by those selected filters

occupied_supported_epoch_share =
  occupied_epoch_count / raw_supported_epoch_count
```

If `raw_supported_epoch_count=0`, the share is null. A-1-4 verifies that the
null is denominator-consistent; if and only if execution reaches A-1-7, zero
raw support fails A-1-7. If an earlier gate fails, A-1-7 is
`NOT_EVALUATED`. If support is positive and occupied count is zero, the share
is exactly `0.0`. Counts are deduplicated across directions. Structurally
eligible market-time occupancy is reported only as a diagnostic and cannot
satisfy sparsity gates.

Gates are sequential and fail closed:

```text
any A-1-0 failure:
  detector/null interpretation stops
  A-1-1 through A-1-7 = NOT_EVALUATED

A-1-1 failure:
  A-1-2 through A-1-7 = NOT_EVALUATED

any A-1-k failure for k in 2..6:
  every later gate = NOT_EVALUATED
```

Evidence needed to establish the failing gate may be produced, but no later
scientific gate is interpreted. First failed gate determines classification;
no later gate may rescue it.

Gate A-1-0, authority and determinism:

- reviewed plan SHA, accepted ancestry, runner blob and callable AST exact;
- direct null-authority binding exact;
- 29-cache size/SHA/schema and typed inventory closure exact;
- invalid raw source contribution count zero;
- Build A/B roots distinct;
- preseal, pending and final difference counts zero;
- exact 25-output set and manifest closure.

Gate A-1-1, zero outcome:

- consumed-field whitelist exact;
- future price/target and fill/fee/PnL never loaded;
- poison of unconsumed values changes no output;
- no target or economic artifact.

Gate A-1-2, M-state/epoch/reset integrity:

- four-state and six-action partitions exact;
- zero `NEW_INVALID`, sign overlap, unauthorized TTL refresh,
  UNKNOWN-to-BACKGROUND, neutral skip, expiry and cross-segment carry;
- zero invalid natural-anchor prestate;
- epoch arithmetic and six-way disposition partition exact;
- zero retained onset from an ineligible epoch or edge guard;
- at most one retained onset per capture/epoch/direction;
- every retained onset is exact earliest by frozen tie-break;
- at most two retained anchors and one cluster per capture/epoch;
- dual directions in one epoch share exact cluster identity;
- adjacent counted epoch clusters are separated by at least 30s;
- zero filter monotonicity violation;
- zero slice candidate/support identity mismatch;
- zero cross-segment slice comparison.
- qualifying artificial starts on at least four dates;
- at least 30 distinct comparable epochs and positive support comparison;
- maximum common retained-cluster 5s burst at most one.

Gate A-1-3, structural null admissibility:

- 199 replicates in selection and every evaluation bank;
- at least 190 distinct fingerprints in each bank/duration;
- zero stream overlap;
- zero channel mask, conservation, pair identity/order, comparison-mask,
  epoch-disposition, dual-direction-dedup or denominator-identity mismatch;
- minimum date-pair count at least 3;
- maximum date p95 joint distance at most 0.60.

Gate A-1-4, selection and numeric integrity:

- exactly nine folds and held-out isolation exact;
- observed selection access zero and bank overlap zero;
- filter-duration numerator/denominator identity exact;
- checkpoint/seconds/hours conversions exact;
- counts/exposures are integer, finite and non-negative;
- rates/burdens/shares are finite when defined;
- rate is null iff exposure is exactly zero;
- structural burden and maximum-single-date share are null iff observed
  cluster count is exactly zero;
- occupied-supported-epoch share is null iff raw-supported epoch count is
  exactly zero, and is `0.0` when its denominator is positive and occupied
  count is zero;
- legitimate zero support is not numeric corruption;
- NaN, infinity, negative values, invalid types or unit mismatch fail.

Gate order and classifications:

```text
A-1-0 authority/determinism
  -> Aminus1_source_not_admissible
A-1-1 zero outcome
  -> Aminus1_zero_outcome_boundary_violated
A-1-2 M-state/epoch/reset integrity
  -> Aminus1_mstate_integrity_failed
A-1-3 structural null admissibility
  -> Aminus1_structural_null_not_admissible
A-1-4 selection/numeric integrity
  -> Aminus1_selection_integrity_failed
A-1-5 support estimability
  -> Aminus1_structural_support_not_estimable
A-1-6 structural false-fire control
  -> Aminus1_structural_false_fire_control_failed
A-1-7 raw sparsity
  -> Aminus1_signal_not_sparse
all pass
  -> Aminus1_historical_structural_false_fire_control_candidate
```

Frozen thresholds:

```text
A-1-5:
  primary exposure > 0
  primary observed clusters >= 30
  represented dates >= 4
  maximum single-date share <= 0.50

A-1-6 primary:
  null p95 rate <= 0.10/hour
  null burden ratio <= 0.10
  count-tail p <= 0.01
  dates above date-null p90 >= 4

A-1-6 sensitivities:
  10s and 60s estimable
  rate <= 0.20/hour
  burden <= 0.20
  tail p <= 0.05

A-1-7:
  raw exposure > 0
  raw selected-filter cluster rate <= 5/hour
  occupied supported epoch share <= 0.10
```

Maximum 5s burst must be at most one and is an A-1-2 bookkeeping integrity
diagnostic, not empirical sparsity evidence. It uses every unique common
retained `dependence_cluster_id` before filter confirmation or selection,
one timestamp per cluster equal to
`epoch_start_ns`, separately per capture. It takes the maximum count in any
half-open `[s,s+5s)` window whose `s` is a cluster timestamp; windows never
cross captures and directions are deduplicated before counting. Coverage and
minimum firing rate remain diagnostics, not recall targets.

## 11. Required Outputs

Exactly 25 non-cache artifacts:

```text
contracts/
  source_cache_contract.json
  mstate_detector_contract.json
  fixed_epoch_thinning_contract.json
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
  epoch_support_by_date.csv
  candidate_ledger.csv
  filter_support_by_date.csv
  fold_selection_ledger.csv
  cross_fitted_signal_ledger.csv
  cross_fitted_null_summary.csv
  structural_false_fire_summary.csv
  parameter_monotonicity.csv
  orphan_strict_onset_by_date.csv
  slice_invariance.csv

reports/A_minus1_summary.json
classification.json
run_manifest.json
```

`run_manifest.json` lists the other 24 artifacts and excludes itself.
Missing or extra paths, duplicate paths, schema mismatch, manifest count
mismatch or SHA mismatch fails closed at A-1-0.

Frozen new/changed schemas:

`fixed_epoch_thinning_contract.json`:

```text
origin_ns, epoch_width_ns, checkpoint_ns, expected_checkpoint_count,
core_open_offset_ns, core_close_offset_ns, disposition_precedence,
thinning_key, tie_break, cluster_key, edge_omission_policy
```

`epoch_support_by_date.csv`, one row per
`research_date, capture_id, epoch_id`:

```text
research_date,capture_id,epoch_id,epoch_start_ns,epoch_end_ns,
core_open_ns,core_close_ns,segment_id,segment_count,segment_ids_json,
segment_set_sha256,disposition,observed_checkpoint_count,
unique_timestamp_count,duplicate_timestamp_count,off_grid_timestamp_count,
missing_expected_timestamp_count,grid_exact,
raw_natural_onset_neg_count,raw_natural_onset_pos_count,
edge_guard_omitted_neg_count,edge_guard_omitted_pos_count,
retained_neg_count,retained_pos_count,
same_key_suppressed_neg_count,same_key_suppressed_pos_count,
retained_neg_candidate_id,retained_pos_candidate_id,
dependence_cluster_id
```

`candidate_ledger.csv`, one row per retained common candidate:

```text
research_date,capture_id,epoch_id,epoch_start_ns,core_open_ns,core_close_ns,
segment_id,direction,candidate_id,candidate_ts_ns,candidate_event_seq,
dependence_cluster_id,channel_last_observation_ts_json,
channel_last_observation_age_ms_json,common_prestate_background_count,
common_prestate_abstain_count,common_prestate_signal_count,
admitted_filter_ids,confirmation_map_json,cancel_reason_map_json
```

`slice_invariance.csv`, one row per qualifying artificial start:

```text
research_date,capture_id,segment_id,nominal_start_ts_ns,
actual_start_ts_ns,comparison_floor_ns,first_comparable_epoch_id,
slice_source_sha256,
comparable_epoch_count,expected_identity_count,actual_identity_count,
expected_identity_sha256,actual_identity_sha256,identity_exact,
expected_support_tuple_count,actual_support_tuple_count,
expected_support_sha256,actual_support_sha256,support_identity_exact,
cross_segment_checkpoint_count,mismatch_reason
```

The summary additionally records:

```text
nominal_artificial_start_count
qualifying_artificial_start_count
skipped_absent_or_wrong_segment_count
skipped_no_comparable_epoch_count
represented_slice_date_count
distinct_comparable_epoch_count
compared_support_checkpoint_count
```

`reports/A_minus1_summary.json` freezes these changed objects:

```text
raw:
  observed_cluster_count: int
  exposure_checkpoint_count: int
  exposure_seconds: finite float
  exposure_hours: finite float
  cluster_rate_per_hour: finite float or null
  raw_supported_epoch_count: int
  occupied_epoch_count: int
  occupied_supported_epoch_share: finite float or null
  raw_supported_epoch_identity_sha256: ASCII SHA256
  occupied_epoch_identity_sha256: ASCII SHA256
  occupied_subset_violation_count: int
  structurally_eligible_epoch_count: int
  structurally_occupied_epoch_count: int
  structurally_occupied_epoch_share: finite float or null

integrity:
  common_cluster_maximum_5s_burst: int
  nominal_artificial_start_count: int
  qualifying_artificial_start_count: int
  skipped_absent_or_wrong_segment_count: int
  skipped_no_comparable_epoch_count: int
  represented_slice_date_count: int
  distinct_comparable_epoch_count: int
  compared_support_checkpoint_count: int
```

Epoch identity hashes use sorted `(capture_id, epoch_id)` tuples. Occupied
identities must be an exact subset of raw-supported identities. Structural
market-time occupancy uses all structurally eligible epochs as denominator
and selected occupied epochs as numerator; it is diagnostic only. The gate
contract repeats each gate condition with exact `actual`, `required`,
`passed` and `status` (`PASS`, `FAIL`, `NOT_EVALUATED`).

Candidate/support hashes use canonical JSON over lexicographically sorted
identity tuples. `mismatch_reason` is exactly one of:

```text
none
candidate_identity
support_identity
cross_segment
multiple
```

All integer fields are base-10 integers, booleans are `True/False`, absent
candidate and cluster IDs are empty strings. `segment_id` is populated only
for `eligible`, otherwise it is empty. `segment_ids_json` is the numeric
ascending unique segment list, including `[]` for an empty epoch.
`segment_set_sha256` is canonical JSON SHA256 of that list.

Typed row ordering is:

```text
research_date ASCII, capture_id ASCII, epoch_id numeric
```

Candidate rows add `direction numeric`, then `candidate_ts_ns numeric`,
then `candidate_event_seq numeric`. Canonical JSON everywhere uses
`sort_keys=True,separators=(",",":"),ensure_ascii=True`; NumPy scalars are
converted to Python `int`, `float`, `bool` or `str` before serialization.

## 12. Hostile Tests

At minimum:

- absolute epoch boundaries at exactly 0s, 15s, 45s and 60s;
- core is `[15s,45s)`, not closed on the right;
- partial, missing-grid, irregular-grid and segment-crossing epochs are
  ineligible everywhere;
- disposition overlap/precedence, empty intermediate epoch, duplicate and
  off-grid timestamp mutations;
- complete timestamp set with permuted/non-monotonic raw row order;
- reset inside core with same-direction onsets on both sides yields no anchor;
- reset inside core with opposite-direction onsets on both sides yields no
  cluster;
- earliest onset per capture/direction/eligible epoch wins;
- opposite directions share one epoch cluster but have independent thinning;
- a later onset in the same epoch cannot replace the first;
- cutting before an earlier same-epoch onset may change only that epoch;
- decisions in later complete epochs are identical after reset;
- slice is rebuilt from sliced raw cache; derived-feature reuse mutation
  fails;
- exact row-aligned field-set and stale full-capture endpoint mutations;
- multi-segment slice comparison excludes later segments;
- finite `K_segment`, last nominal boundary and no early-stop behavior;
- skipped starts never create CSV rows and exact summary counters mutate
  closed;
- zero qualifying starts, zero comparable epochs and zero support comparison
  cannot pass;
- exact support identity tuple, typed sorting and hash mutation;
- comparable epochs deduplicate across starts by `(capture_id,epoch_id)`;
- support checkpoint count cannot be substituted by 27x support tuple count;
- epoch cluster IDs never depend on accepted-anchor timestamps;
- adjacent eligible cores are separated by at least 30s;
- no edge-guard anchor enters candidates, estimators or exposure;
- boundary epoch disposition is identical in observed, selection-null,
  evaluation-null and raw paths;
- strict filters remain delete-only;
- null recomputes epoch thinning and cluster identity;
- observed/null dual directions deduplicate to one distinct cluster;
- every filter-duration numerator is a subset of its denominator identity;
- every null replicate shares the exact denominator hash;
- positive raw-supported epochs with zero occupied epochs produce `0.0`, not
  null;
- raw support/occupied hashes, subset relation and summary field mutation;
- selection/evaluation banks are disjoint;
- A-1-2/A-1-3/A-1-4 zero, nonfinite and `NOT_EVALUATED` precedence;
- non-source A-1-0 and A-1-1 failure force all later gates to
  `NOT_EVALUATED`;
- corrupt selection/null cannot create an earlier A-1-2 burst failure;
- A-1-5 or A-1-6 failure forces A-1-7 to `NOT_EVALUATED`;
- segment-boundary epoch schema and empty N/A sentinels round-trip;
- numeric epoch ordering cannot be replaced by ASCII ordering;
- exact new evidence schemas, row grains and field types;
- manifest self-exclusion and exact 25-path mutation;
- all predecessor source/action/memory/null hostile tests remain passing;
- Build A/B preseal, pending and final equality;
- exact 25-output and manifest closure.

## 13. Execution Lock

Before any 29-cache execution:

```text
independent plan review must close at P0/P1/P2/P3 = 0/0/0/0
reviewed plan SHA256 must be frozen in the formal task
```

Until then:

```text
data execution = locked
future outcome access = false
A0 execution = false
live/private/order execution = false
```

No failure may be remediated by changing threshold, epoch geometry, core
width, offset or filter order inside this task.
