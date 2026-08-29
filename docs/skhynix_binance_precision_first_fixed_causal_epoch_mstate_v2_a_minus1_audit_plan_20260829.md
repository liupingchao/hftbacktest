# SKHYNIX Binance Precision-First Fixed Causal Epoch M-State V2
# A-1 Structural False-Positive Control Audit Plan

Date: 2026-08-29

Task: `0829T003`

Hypothesis ID: `FIXED_CAUSAL_EPOCH_MSTATE_V2`

Audit ID: `FIXED_CAUSAL_EPOCH_MSTATE_V2_A_MINUS1`

Status: candidate contract Revision 1; data execution locked

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
```

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

Common thinning:

```text
for each capture, segment, direction and epoch_id:
  retain the earliest common natural onset in the eligible core
  suppress every later onset in the same key
```

Tie-break:

```text
(timestamp, event_seq, direction)
```

At most two anchors may exist per epoch: one per direction.

Dependence cluster:

```text
cluster_id = capture_id : segment_id : epoch_id
```

Both directions in the same epoch share one cluster. No accepted-anchor
timestamp participates in suppression or cluster identity.

Consequences:

- slicing before an epoch cannot change decisions in later complete epochs;
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

Artificial slice tests must compare only the same segment.

For each eligible artificial start:

- rebuild causal features and all M-states from the slice;
- rebuild natural onsets, fixed epoch thinning, epoch clusters and filters;
- compare after:

```text
guard = 122s
```

The guard exceeds:

```text
one full 60s epoch
+ one additional complete epoch/core opportunity
+ 500ms feature history
+ 100ms maximum TTL
+ 120ms prestate
+ 800ms maximum persistence
```

Expected and actual identities are:

```text
(candidate_ts_ns, direction, filter_id, epoch_cluster_id)
```

Required:

```text
zero identity mismatch
zero M-state support-count mismatch
zero cross-segment comparison
```

## 8. Outcome-Blind Structural Null

Reuse the accepted conditional opposite-orientation path-swap null.

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

All magnitude, zero-mask, missingness, denominator, new-evidence-mask,
pair-identity and comparison-mask invariants remain exact.

## 9. Exposure And Selection

The detector runs before the comparison mask is applied.

For filter `f`, duration `h`, checkpoint `t`, exposure requires:

```text
t is inside the fixed eligible core
all M-states in [t-120ms, t+p] are non-ABSTAIN
comparison mask is true throughout [t-120ms-500ms-tau, t+p]
```

Exposure is counted once per unique capture-time checkpoint.

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
  maximum 5s burst <= 2
```

Coverage and firing rate remain diagnostics, not recall targets.

## 11. Required Outputs

Exactly 24 non-cache artifacts:

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

## 12. Hostile Tests

At minimum:

- absolute epoch boundaries at exactly 0s, 15s, 45s and 60s;
- core is `[15s,45s)`, not closed on the right;
- earliest onset per capture/segment/direction/epoch wins;
- opposite directions share one epoch cluster but have independent thinning;
- a later onset in the same epoch cannot replace the first;
- cutting before an earlier same-epoch onset may change only that epoch;
- decisions in later complete epochs are identical after reset;
- multi-segment slice comparison excludes later segments;
- epoch cluster IDs never depend on accepted-anchor timestamps;
- adjacent eligible cores are separated by at least 30s;
- no edge-guard anchor enters candidates, estimators or exposure;
- strict filters remain delete-only;
- null recomputes epoch thinning and cluster identity;
- selection/evaluation banks are disjoint;
- all predecessor source/action/memory/null hostile tests remain passing;
- Build A/B preseal, pending and final equality;
- exact 24-output and manifest closure.

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
