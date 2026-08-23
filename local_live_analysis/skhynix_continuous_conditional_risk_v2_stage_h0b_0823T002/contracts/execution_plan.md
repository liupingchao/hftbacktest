# SKHYNIX Stage H0-B Conditional-Risk Audit Execution Plan V2

Date: 2026-08-23

Status: V2 candidate; it becomes execution authority only after independent
review and exact task pinning.

Candidate formal task ID after review: `0823T002`.

This document does not dispatch `0823T002` and does not authorize outcome
access by itself.

V2 preserves every primary-analysis contract from V1. It supersedes V1 only
for the post-primary-seal Stage 4 landmark crosscheck in Section 23. The
revision was required after a fail-closed execution exposed accepted Stage 2
boundary candidates whose floored landmark has no jointly identifiable H0-B
`50ms` endpoint. The repair is diagnostic-only and cannot change models,
thresholds, gates, primary hashes or primary classification.

The authority split is exact:

```text
primary_plan_path =
  docs/skhynix_stage_h0b_conditional_risk_audit_plan_20260823.md

primary_plan_sha256 =
  c1be0fdbd58f19c201c2faa7251621402486e6ebabf259af316b98bcf4c92b10

diagnostic_plan_path =
  docs/skhynix_stage_h0b_conditional_risk_audit_plan_v2_20260823.md

diagnostic_plan_sha256 =
  exact independently reviewed and dispatch-pinned V2 raw SHA256
```

V1 remains the sole authority for pre-outcome support, estimator, RQ1/RQ2/RQ3,
latency roles and primary classification. V2 is the sole authority for the
post-seal Stage 4 diagnostic, its permit/receipt, its expanded output schema
and the final package integration that includes those diagnostic bytes. Both
identities must be present in the task, matrix, permits, primary seal,
runtime-contract bridge and final evidence. Neither identity may substitute
for the other.

## 1. Authority And Position In The Chain

This plan defines the first outcome-bearing stage under the user-approved
continuous conditional-risk v2 framework.

The accepted execution chain is:

```text
accepted Research Package Trust Kernel v1
-> accepted Stage H0-A support-only package
-> accepted c6in execution-latency measurement
-> accepted H0-B superseding primary tuple
-> reviewed H0-B execution plan
-> formal H0-B task
-> pre-outcome contract and support replay
-> outcome-bearing Build A / Build B
-> independent QA
-> controller closure
```

Accepted authorities:

- `0820T001 / RESEARCH-PACKAGE-TRUST-KERNEL-LAYERED-IDENTITY-AND-STAGE4-PARITY`:
  `已通过`;
- `0821T001 / SKHYNIX-STAGE-H0A-SUPPORT-ONLY`:
  `已通过`;
- `0822T002 /
  SKHYNIX-C6IN-HYPERLIQUID-EXECUTION-LATENCY-MEASUREMENT-REVISION-2`:
  `已通过`;
- `0823T001 / SKHYNIX-H0B-PRIMARY-TUPLE-SUPERSESSION`:
  `已通过`;
- `h0b_tuple_authority=accepted_superseding_tuple`.

Master research authority:

```text
docs/skhynix_continuous_hazard_maker_research_framework_v2.md
```

Accepted H0-A execution authority:

```text
docs/skhynix_stage_h0a_support_only_execution_plan.md
```

Accepted superseding-tuple authority:

```text
docs/skhynix_h0b_primary_tuple_supersession_plan_20260823.md
.workflow/reports/0823T001-controller-closure.md
```

No chat statement, later diagnostic, H0-B result or implementation convenience
may override these accepted identities or semantics.

## 2. Purpose

Stage H0-B answers three bounded questions:

1. **RQ1, time variation:** does the frozen `50ms` public adverse-event risk
   vary materially over calendar time after preserving local dependence and
   conditioning on accepted segment/cadence structure?
2. **RQ2, coarse predictability:** does a frozen cross-spread feature set
   improve out-of-fold interval likelihood over a time/context baseline?
3. **RQ3, primary-latency actionability:** do out-of-fold high-risk regimes
   retain positive residual dwell after the accepted `6600ms` latency?

The stage publishes a small, auditable research package. It does not build a
new GB-scale row-level data plane.

## 3. Stage Decision Scope

H0-B is a screening audit before the main H0-H4 modeling stage.

It may:

- reject the reactive-risk route when RQ1 fails;
- determine whether the frozen coarse H1/H0 screen is promising enough for
  main modeling;
- determine whether the coarse signal is compatible with the accepted
  `6600ms` latency;
- identify a non-rescuing difference between the `850ms` diagnostic and the
  `6600ms` primary;
- declare that more sessions or better coverage are required.

It may not:

- issue the final framework classification
  `conditional_quote_risk_signal_supported`;
- claim formal H3-vs-H2 queue-shock increment;
- select among H2/H3/H4 estimators;
- claim maker profitability, quote posture, fill probability, execution
  quality or live action value;
- promote `850ms`, `100ms` or any other scenario over `6600ms`;
- use an H0-B result to change target, distance, horizon, side aggregation,
  latency or session role.

The strongest positive H0-B result is:

```text
h0b_main_modeling_candidate
```

That result permits controller review of a later main-modeling plan. It does
not itself establish a deployable or final risk signal.

## 4. Frozen Primary Tuple

The exact primary tuple is:

```text
stage_id = stage_h0b
feature_set_id = feature_set_h0
target = public_bbo_moves_through_quote
target_venue = hyperliquid
target_channel = bbo
distance_definition = target_visible_best_quote
delta_ticks = 0
horizon_ms = 50
gate_latency_ms = 6600
side_aggregation = equal_weight_bid_ask_session_scores
calendar_grid_ms = 10
primary_block_seconds = 60
```

The authoritative tuple file is:

```text
local_live_analysis/skhynix_h0b_primary_tuple_supersession_0823T001/
superseding_primary_tuple.json
```

Its SHA256 is:

```text
e3badf4c179a9e717ea49ff6617b637ec7e4c967c3dec365bebe78ef9399457c
```

Accepted tuple package identities:

```text
R = 08ada07165297f72dc05eec402bcfb70d748c6386986ec555b8c1b609e406079
C = a5f40d41226066291afcfc31d473cbadf8ed1edb322be6843b0f7aca45ea66b5
E = 32ed6e541183683e2279860d9deef30ab7b0d230acff3ef84dd8e8f865632dc6
composite =
    5ec515e00ab2765a281084a64fbe0e1962727059e42734be33b7362258a22f76
```

Latency roles are immutable:

| Latency | Role | Primary | Diagnostic | Can rescue primary |
| ---: | --- | --- | --- | --- |
| `25ms` | `legacy_sensitivity` | false | false | false |
| `50ms` | `legacy_sensitivity` | false | false | false |
| `100ms` | `historical_optimistic_sensitivity` | false | false | false |
| `250ms` | `legacy_sensitivity` | false | false | false |
| `500ms` | `legacy_sensitivity` | false | false | false |
| `850ms` | `terminal_observability_normal_path_diagnostic_only` | false | true | false |
| `6600ms` | `measurement_selected_primary` | true | false | false |

## 5. Session Roles

The accepted session roles are:

| Session | Evidence label | Formal eligible | H0-B role |
| --- | --- | --- | --- |
| Jul30 | `historical_discovery_and_internal_validation` | true | formal |
| Aug03 | `historical_transfer` | false | diagnostic only |
| Aug04 | `historical_consumed_validation` | true | formal |

Aug03 is diagnostic only because accepted H0-A records
`evidence_label=historical_transfer` and `formal_eligible=false`.

Aug03 may never:

- satisfy a two-session formal gate;
- rescue a Jul30/Aug04 disagreement;
- determine a threshold, bin edge, scaler or regularization setting used by
  a formal session;
- be pooled with formal sessions under a formal score.

The accepted tuple records:

```text
underlying_market_state = unknown_calendar_state
future_calendar_inference = false
```

H0-B must not infer KRX open, close, auction or holiday state from the date or
wall clock. All underlying-regime outputs therefore use:

```text
underlying_regime = unknown_calendar_state
```

## 6. Accepted Input Pins

### 6.1 Trust Kernel

```text
kernel_name = research_package_trust_kernel
kernel_version = v1
registry_revision = 1
registry_entry_sha256 =
    cae21d65bf447435bafc37508b8ca00643a0742b37e0f404148cab92818c90c9
kernel_source_tree_sha256 =
    cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203
kernel_api_contract_sha256 =
    2cd5a67ba15d67e59bcddcdbb21593696d3dc3dc27d81986c39ddf3e91e91f5f
kernel_negative_matrix_sha256 =
    f6247594b6f024945a52c0dccf421bac93538d9357f92ff6027d024199fc6b97
kernel_qa_report_sha256 =
    8fe01f85f8a68581b79ee410167769f2a105d9cc74ca6528af9496808a626be8
```

### 6.2 Accepted H0-A

```text
task_id = 0821T001
package =
  local_live_analysis/
  skhynix_continuous_conditional_risk_v2_stage_h0a_support_only
primary_tuple_sha256 =
  e5d1b132248ff1a6933678c32a54e6b4147c1c6f47dab25103011ecbd7a68eca
support_projection_identity =
  2bdc9b127e065f19d3166e3821df846383a2bc8eaa770f1cd6d267195f8ea7a4
support_projection_commitments_sha256 =
  a266184403a830fc422764e90c6dd48a5c1900af13333246ca7e220d664728df
R = 7176c78c2b6bfadf11432e9f5a1c1eaf9627c8028a4a22257d67d5a5404dc8fd
C = 4e8ccc7466f84d9eb2f71f681557424b59ca58249d67426325ecdbd661189636
E = 8745458fcf4e0ab31f8ad3b2bc2d1d93704f18b13a776c515f14195b51213969
composite =
  2682c32eefac427eed1899a3d492b3fc7545520f72021723e8fef7a0d4d8d9d0
qa_report_sha256 =
  337cb9990adc84376e2083fa4076ba40f9f56c8709d687fd1487502f6992dcac
controller_closure_sha256 =
  9cc1b1d24d29cd2b55a8c1774a9d9e6e59338242c95c3af861460a0b8b07aded
```

### 6.3 Accepted Execution-Latency Measurement

```text
task_id = 0822T002
accepted_at = 2026-08-23T15:02:27Z
source_commit =
  0c0c5b1c232fce18b3ea5e9efa53a78da3ee503f
package =
  local_live_analysis/
  skhynix_c6in_hyperliquid_execution_latency_0822T002
primary_interval =
  risk_decision_ready_to_authoritative_terminal_confirm
primary_quantile = nearest_rank_p95
p95_cancel_effective_latency_us = 6561052
recommended_gate_latency_ms = 6600
R = e8b118bfcf9cbad4c0d95d070084aa9a268f62c13140728c80e373966388eb55
C = 20a5837162d63763ee42e3fc8ed7bef824316e102eb9325a15f83fe901b37ea9
E = 103dbe0d2e02392b5e45d61bb106bbdf7d4235b982cea44f895c72aea98ff958
composite =
  7d851ab161ec02c621dffff63ef2f3e962a2aa0c7ed8b4382b285df9534bb0df
package_inventory_sha256 =
  1750474bdd04e1ff5b4beaddf1d93c3e79177060abd2cf7e3c6bacad0876af43
measurement_manifest_sha256 =
  8ac3b362e8d64cbd81232eaf7ed5856bada63ece20408e0d0b3fb5f84c562afd
qa_report_sha256 =
  f8f8f534013ebeb0fcb7d5b6c87efa6e655e23065d0ae516ae3399436471fe86
controller_closure_sha256 =
  96523141ad541f64ce952db84ac9f7ee82502e20fe13f83367bbb6cb9d114cf7
```

H0-B consumes this accepted measurement only through its immutable identity
and the accepted superseding tuple. It does not reopen private/order evidence
or recompute the latency bucket.

### 6.4 Accepted Superseding Tuple

```text
task_id = 0823T001
tuple_sha256 =
  e3badf4c179a9e717ea49ff6617b637ec7e4c967c3dec365bebe78ef9399457c
R = 08ada07165297f72dc05eec402bcfb70d748c6386986ec555b8c1b609e406079
C = a5f40d41226066291afcfc31d473cbadf8ed1edb322be6843b0f7aca45ea66b5
E = 32ed6e541183683e2279860d9deef30ab7b0d230acff3ef84dd8e8f865632dc6
composite =
  5ec515e00ab2765a281084a64fbe0e1962727059e42734be33b7362258a22f76
qa_report_sha256 =
  764ca3f3f7c7fe7e9a6884a25d9c9cbb1ebf31387892ca8c2c6fd954f0e0b9ae
controller_closure_sha256 =
  00fd9916e15d8e4079925e37990b7e30f9a51373a6f0829908eca3b99b49ac30
h0b_tuple_authority = accepted_superseding_tuple
```

### 6.5 Accepted Stage 1-4 Dependencies

H0-B inherits the exact Stage 1-4 identities embedded in accepted H0-A.
It must not independently substitute a newer or similarly named package.

The accepted Stage 2 and Stage 3 packages are used only for frozen
queue-shock dose and dependence identities. The accepted Stage 4 package is
diagnostic-only and is subject to the post-primary sealing rule in Section 23.

## 7. Non-Goals And Hard Boundary

H0-B does not:

- open Aug07 event rows;
- collect new data;
- use network access;
- read credentials or private/account state;
- call order or cancel endpoints;
- submit, cancel or simulate own orders;
- infer fills, fees, queue position, inventory or PnL;
- change accepted Stage 1-4, H0-A, latency or tuple artifacts;
- modify the Trust Kernel registry;
- fit H2, H3 or H4 formal models;
- select an estimator after comparing results;
- publish a row-level 10ms calendar grid;
- build a strategy, posture rule or GLFT feature integration;
- execute the staleness audit;
- use Stage 4 precomputed outcomes as the primary outcome oracle.

Forbidden field tokens include:

```text
own_order
own_fill
filled
fill_probability
execution_pnl
inventory
fee_adjusted
realized_pnl
```

## 8. Two-Envelope Execution Architecture

H0-B uses two separate processes and two separate authority envelopes.

### 8.1 H0B0: Pre-Outcome Contract Builder

H0B0 may read:

- accepted plans, tasks, QA and controller closures;
- accepted H0-A support-only package;
- accepted execution-latency package metadata and identities;
- accepted superseding tuple package;
- accepted Stage 1-3 contracts and metadata;
- raw/R0/R1 source rows only through the H0-A support projector's allowed
  support fields and same-row validity checks.

H0B0 must not:

- compare target BBO prices across time;
- evaluate the adverse-event predicate;
- compute adverse rates, losses, risk bins or dwell;
- open Stage 4 outcome/features/views;
- open Aug07 event rows.

H0B0 outputs and fsyncs:

```text
preoutcome_contract.json
support_replay_receipt.json
preoutcome_source_inventory.csv
outcome_access_permit.json
```

These are local names inside each isolated build root. The final package
preserves both independent evidence sets as exact `build_a`/`build_b` root
files listed in Section 25; no Build B evidence is overwritten by Build A.

Before task dispatch, the controller reconstructs the exact semantic source
inventory without evaluating any adverse-event predicate. The task pins:

```text
expected_semantic_source_inventory_sha256
source_inventory_contract_sha256
```

`preoutcome_source_inventory.csv` contains repository-relative semantic rows
only:

```text
session,segment_id,source_role,relative_path,bytes,sha256,header_sha256
```

Rows are ordered by the complete tuple above. Absolute roots, build labels,
inode values and filesystem timestamps are forbidden from the semantic
inventory. Therefore Build A and Build B must produce the same
`semantic_source_inventory_sha256`.

This semantic inventory covers H0-B primary sources only. The eight Stage 4
diagnostic files are not opened or hashed by H0B0; their accepted
path/bytes/SHA identities are pinned from the already accepted Stage 4
manifest and are reverified by the post-seal diagnostic opener.

### 8.2 H0B1: Outcome Runner

H0B1 is a fresh process. It may start only when:

```text
outcome_access_permit.status = admitted
outcome_access_permit.fsynced = true
support_replay_receipt.exact_commitment_match = true
accepted_input_identity_match = true
preoutcome_contract_sha256 = dispatch-pinned SHA256
semantic_source_inventory_sha256 =
  dispatch-pinned expected_semantic_source_inventory_sha256
source_inventory_contract_sha256 = dispatch-pinned SHA256
build_envelope_sha256 = hash of the current isolated build envelope
```

The permit binds:

- task and plan identity;
- Surface Matrix identity;
- runtime source tree identity;
- accepted H0-A, latency and superseding-tuple identities;
- the cross-build semantic source inventory;
- the current build-root-specific envelope;
- all likelihood branches and formulas;
- feature allowlist;
- OOF folds;
- bootstrap seeds;
- RQ1/RQ2/RQ3 gates;
- latency roles;
- output tree;
- forbidden-path policy.

Any missing or stale field returns:

```text
H0B_OUTCOME_PERMIT_MISMATCH
```

The permit contains both identities:

```text
semantic_source_inventory_sha256
build_envelope_sha256
```

The semantic hash must be byte-identical across Build A and Build B. The build
envelope hash is intentionally different and is computed over canonical JSON:

```text
{
  "build_label": "A" | "B",
  "resolved_build_root": absolute path,
  "runtime_pid": positive integer,
  "runtime_source_tree_sha256": lowercase SHA256,
  "semantic_source_inventory_sha256": lowercase SHA256,
  "preoutcome_contract_sha256": lowercase SHA256
}
```

The permit is single-build-root specific. It cannot be copied from Build A to
Build B. Rebinding changes only the build envelope and permit identities; it
must not change the semantic inventory.

## 9. Guarded Source Boundary

The authoritative source inventory is reconstructed from accepted H0-A
`input_bindings.csv` and accepted Stage 2/3 manifests.

Allowed source families:

1. Jul30/Aug03/Aug04 accepted raw manifests, timeline indexes and target BBO
   stores bound by H0-A;
2. H0-A-bound R0 Binance `bookTicker` event stores used only for strict-as-of
   public feature state after the outcome permit;
3. accepted R0 segment manifests and quality masks;
4. accepted R1 alignment manifests and quality rows;
5. accepted Stage 2 candidate membership and overlap-block identities;
6. accepted Stage 3 candidate audit projection and detector contract;
7. accepted H0-A support commitments;
8. accepted execution-latency package metadata and closure;
9. accepted H0-B superseding tuple;
10. accepted Trust Kernel v1.

R1 `decision_labels/*.csv.gz` contains future-horizon columns and is not
required for H0-B primary feature construction. It is forbidden throughout
this task. Current/past Binance and Hyperliquid quote state is rebuilt from
the accepted R0 public event stores instead.

Pre-outcome schema admission computes SHA256 over the exact UTF-8 header plus
one LF. The accepted current inventory has:

```text
Hyperliquid hot-event header:
  unique_header_count = 1 across 19 files
  sha256 =
    5589834f56360a64bb46ac3be4046bd2167310068022cd8a2aa3ff6b5187481d

Binance hot-event header:
  unique_header_count = 1 across 19 files
  sha256 =
    6660a3ce0c75a3653b112b89a21029cf3eb2807d0c7d90bf4573704e1af01ba5
```

No per-session schema adapter is active. A header mismatch fails closed and
requires independent plan review before any adapter may be introduced.

Forbidden source families:

- Aug07 event rows;
- any unaccepted or similarly named replay package;
- R1 `decision_labels/*.csv.gz`;
- Stage 4 outcomes/features/views before primary H0-B results are sealed;
- network, private, order or cancel sources;
- GLFT runtime or live decision logs;
- staleness-audit outputs.

The guarded opener must reject a forbidden path before reading any bytes.

Absolute host roots are provenance only. Content identity uses the accepted
scope, session, segment, repository-relative path, bytes and SHA256 fields.
The H0-A absolute-path identity observation must not be repeated as a false
semantic mismatch.

## 10. Calendar Grid And At-Risk Surface

The grid is:

```text
origin_ns = 0
step_ns = 10_000_000
interval = half_open_segment_epoch
```

Every nominal grid start `t` is reconstructed exactly as H0-A did.

Strict-as-of visibility is:

```text
source_receive_ts_ns <= t
same segment
same connection epoch
```

No message arrival at `t' > t` may influence a feature at `t`.

Every accepted R0 event store is first validated in exact source order:

```text
ordering_key = (local_ts_ns, event_seq)
event_seq is strictly increasing within the source file
local_ts_ns is non-decreasing within the source file
```

The state visible at `t` is the last valid row by `ordering_key` among rows
with `local_ts_ns <= t`. All same-timestamp rows at `local_ts_ns=t` are state
rows, never future outcome rows. A source-order violation fails closed.

Each interval-likelihood-eligible grid start expands into exactly two side
rows:

```text
side = maker_ask_risk
direction_sign = +1

side = maker_bid_risk
direction_sign = -1
```

This side expansion does not double statistical independence. Row counts are
structural exposure counts and are never interpreted as universal `N_eff`.

## 11. Vulnerable Quote And Adverse Event

At grid time `t`, use the last valid Hyperliquid BBO visible strict-as-of `t`.

```text
maker_ask_risk vulnerable quote = hyperliquid_ask(t)
maker_bid_risk vulnerable quote = hyperliquid_bid(t)
```

The primary endpoint and the observation-bound search limit are different
objects:

```text
endpoint_ns = t + 50ms

binary_identification_supported search limit:
  endpoint_ns

interval_likelihood_only_supported search limit:
  first qualifying target-BBO receive strictly after endpoint_ns
```

A qualifying receive must have `receive_ts_ns>t`, remain in the same accepted
segment/epoch/source-support component, and contain a finite positive
non-crossed BBO. For interval-only rows, the post-endpoint receive must exist
before the first segment, epoch, core-quality, source-gap or source-end
boundary; otherwise the row contradicts accepted H0-A support and fails
closed.

Later rows are scanned by exact `(local_ts_ns, event_seq)` order. When multiple
rows share one receive timestamp, `event_seq` determines which row is first.
Observation bounds remain clock-time bounds: a non-adverse row at the same
`local_ts_ns` as the first adverse row does not advance `L`. `L` uses the last
qualifying non-adverse receive at a strictly smaller receive timestamp. This
keeps `L < U` for an observed interval while respecting the source order. A
zero-width event interval is a contract failure, not a point event.

The primary event is:

```text
maker_ask_risk:
  first later Hyperliquid bid >= vulnerable ask

maker_bid_risk:
  first later Hyperliquid ask <= vulnerable bid
```

This is exactly:

```text
public_bbo_moves_through_quote
```

It is public-market evidence. It is not a fill, own-order event, exchange
matching-engine timestamp or proof that displayed size was executable.

The primary distance is `delta_ticks=0`.

One-tick-away distance is a secondary diagnostic and cannot enter any H0-B
gate.

## 12. H0-A Support-Class Disposition

The exact mapping is:

| H0-A class | H0-B treatment |
| --- | --- |
| `binary_identification_supported` | include in primary interval likelihood; event uses interval-event likelihood, no event through `t+h` uses full-horizon right-censor likelihood; include in binary diagnostics |
| `interval_likelihood_only_supported` | include in primary interval likelihood under the frozen observation-bound algorithm; exclude from binary diagnostics |
| `right_censored_segment` | exclude from fixed-horizon primary with exact reason |
| `right_censored_source_end` | exclude from fixed-horizon primary with exact reason |
| `epoch_censored` | exclude from fixed-horizon primary with exact reason |
| `core_quality_censored` | exclude from fixed-horizon primary with exact reason |
| `source_gap_censored` | exclude from fixed-horizon primary with exact reason |
| `reference_quote_unavailable` | exclude with exact reason; never impute |
| `invalid_quote_state` | exclude with exact reason; never impute |

H0-B must reproduce the accepted per-segment/per-horizon support projection
commitments before outcome access.

The current accepted `50ms` projection contains zero
`interval_likelihood_only_supported` rows. That observed count does not remove
the branch from the contract. Fixtures and hostile tests must execute every
interval-only branch.

## 13. Observation Bounds

The accepted observation-bound contract is:

```text
observation_bound_contract_id =
  h0a_hyperliquid_bbo_receive_interval_v1
```

For an observed event:

```text
T in (L, U]

L = max(
      t,
      last qualifying non-adverse target-BBO receive time before U
    )

U = first qualifying adverse target-BBO receive time
```

For no observed event through the full horizon:

```text
T > t + h
```

The executable branches are:

```text
1. first adverse U <= endpoint_ns:
     observed interval event (L,U]

2. no adverse through endpoint_ns and:
     a qualifying non-adverse receive exists at or after endpoint_ns
     OR identification_class=binary_identification_supported:
     full-horizon right censor T > endpoint_ns

3. identification_class=interval_likelihood_only_supported and the first
   qualifying post-endpoint receive is adverse with L < endpoint_ns < U:
     horizon-straddling interval; binary endpoint not identified
```

The interval-only search stops after that first qualifying post-endpoint
receive. It cannot scan outcome-adaptively farther into the future. Truncating
all searches at `endpoint_ns`, or scanning beyond the first qualifying
post-endpoint receive, is a contract violation.

For an interval that straddles the horizon:

```text
L < t + h < U
binary endpoint is not identified
```

The lower bound is open and the upper bound is closed. Equality changes are
contract violations.

## 14. Primary Interval-Likelihood Formula

The `50ms` horizon is divided into five fixed event-time bins:

```text
(0ms, 10ms]
(10ms, 20ms]
(20ms, 30ms]
(30ms, 40ms]
(40ms, 50ms]
```

For feature vector `x`, the frozen coarse discrete-time model emits:

```text
q_k(x) = P(T in bin k | T survived through bin k-1, x)
S_0(x) = 1
S_k(x) = product_{j=1..k}(1 - q_j(x))
F_k(x) = 1 - S_k(x)
```

The five-bin probabilities are interpreted as piecewise-constant continuous
hazards inside each `10ms` bin. Let `d=10_000_000ns`, `u` be elapsed
nanoseconds after `t`, and:

```text
bin(u) = min(5, floor(u / d) + 1) for 0 <= u < 50ms
fraction(u) = (u - (bin(u)-1)*d) / d

S_exact(0) = 1
S_exact(50ms) = S_5

S_exact(u) =
  S_{bin(u)-1} *
  (1 - q_{bin(u)}) ** fraction(u)
  for 0 < u < 50ms
```

This is equivalent to a constant continuous hazard
`lambda_k=-log(1-q_k)/d` inside bin `k`. Calculations use float64 and the
Section 19 clipped `q_k`. Exact bin boundaries use the already-defined
`S_k`; no floating boundary search is permitted.

Observed event likelihood:

```text
P(L < T <= U | x) =
  S_exact(L - t | x) - S_exact(U - t | x)
```

Full-horizon right-censor likelihood:

```text
P(T > t + 50ms | x) =
  S_5(x)
```

Horizon-straddling likelihood:

```text
P(T > L | x) =
  S_exact(L - t | x)
```

The primary row loss is:

```text
-log(max(likelihood, 1e-12))
```

The only scalar `50ms` risk score used for binary reliability, realized-rate
deciles, coarse empirical cells and RQ3 is:

```text
risk_score_50ms(x) = F_5(x) = 1 - S_5(x)
```

Invalid bound order, `L >= U`, zero/negative likelihood before flooring, a
bound outside the allowed observation geometry or a branch/class mismatch
fails closed.

No midpoint, lower-bound point, upper-bound point or first-message point
coercion is allowed.

## 15. Binary Diagnostics

Binary Brier score, binary log loss and reliability tables use only:

```text
identification_class = binary_identification_supported
```

Binary target:

```text
y = 1 when an adverse event is observed with U <= t + 50ms
y = 0 when no adverse event is observed through t + 50ms
```

`interval_likelihood_only_supported` rows never enter binary diagnostics.

Every binary table publishes:

- eligible count;
- identified fraction relative to the primary interval-likelihood surface;
- event count;
- non-event count;
- Brier score;
- binary log loss;
- ten equal-count training-defined reliability bins.

Binary diagnostics may confirm direction. They may not replace interval log
loss as the primary score.

## 16. Feature Contract

H0-B fits only the frozen coarse H0 and H1 estimators.

### 16.1 H0: Time/Context Baseline

H0 contains:

```text
side indicator
elapsed_session_fraction
elapsed_session_fraction_squared
elapsed_segment_fraction
target_bbo_update_count_1s
target_bbo_no_new_information_fraction_1s
```

All values are strict-as-of `t`.

The elapsed fractions are exact:

```text
formal_session_start_ns =
  minimum accepted segment_start_ts_ns in that formal session

formal_session_end_ns =
  maximum accepted segment_end_ts_ns in that formal session

elapsed_session_fraction =
  (t - formal_session_start_ns) /
  (formal_session_end_ns - formal_session_start_ns)

elapsed_session_fraction_squared =
  elapsed_session_fraction ** 2

elapsed_segment_fraction =
  (t - segment_start_ts_ns) /
  (segment_end_ts_ns - segment_start_ts_ns)
```

The session denominator spans absolute accepted calendar time, including
accepted gaps between segments; gap rows themselves remain excluded. Segment
and session denominators must be strictly positive. Fractions are computed
before any scaling and are not clipped.

The cadence features use the trailing half-open/closed window `(t-1s, t]`
inside the same segment and epoch:

```text
target_bbo_update_count_1s =
  qualifying Hyperliquid BBO receive count

target_bbo_no_new_information_fraction_1s =
  count of 10ms subintervals with zero qualifying receive / 100
```

A full trailing `1s` window is required; otherwise both cadence features are
missing. No receive-age timeout is introduced.

No calendar/underlying-state inference is added.

### 16.2 H1: Cross-Spread Screen

H1 contains H0 plus:

```text
risk_gap_bps
risk_gap_change_50ms_bps
binance_bbo_age_ms
hyperliquid_bbo_age_ms
trailing_basis_residual
```

Direction-normalized `risk_gap_bps` is:

```text
maker_ask_risk:
  10000 * (binance_bid - hyperliquid_ask) / reference_mid

maker_bid_risk:
  10000 * (hyperliquid_bid - binance_ask) / reference_mid
```

The denominator is exact:

```text
binance_mid = 0.5 * (binance_bid + binance_ask)
hyperliquid_mid = 0.5 * (hyperliquid_bid + hyperliquid_ask)
reference_mid = 0.5 * (binance_mid + hyperliquid_mid)
```

`risk_gap_change_50ms_bps` uses strict-as-of state at `t` and `t-50ms` in the
same segment/epoch. A missing historical endpoint remains explicitly missing.

Feature-state sources are exact:

```text
Binance BBO:
  last valid event_type=bookTicker row with local_ts_ns <= t

Hyperliquid BBO:
  last valid event_type=bbo row with local_ts_ns <= t

binance_bbo_age_ms =
  (t - Binance BBO local_ts_ns) / 1e6

hyperliquid_bbo_age_ms =
  (t - Hyperliquid BBO local_ts_ns) / 1e6
```

Rows must be finite, positive and non-crossed. State never carries across a
segment, epoch, core-quality or source-gap boundary.

The basis feature is frozen as:

```text
basis_level_bps(t) =
  10000 * (binance_mid(t) - hyperliquid_mid(t)) / reference_mid(t)

ewma_half_life = 60s
ewma_step = 10ms
ewma_alpha = 1 - exp(-ln(2) * ewma_step / ewma_half_life)

trailing_basis_residual(t) =
  basis_level_bps(t) - prior_only_ewma_basis_level_bps(t)
```

The EWMA at `t` uses only valid grid states strictly before `t`, initializes
from the first prior valid state and resets at every segment, epoch,
core-quality or source-gap boundary. The first valid state after a reset is
missing; no future or full-session demeaning is allowed.

### 16.3 Missing Values

Rows remain on the accepted primary support surface.

The raw feature order is:

```text
H0 numeric raw order:
  elapsed_session_fraction
  elapsed_session_fraction_squared
  elapsed_segment_fraction
  target_bbo_update_count_1s
  target_bbo_no_new_information_fraction_1s

H1 added numeric raw order:
  risk_gap_bps
  risk_gap_change_50ms_bps
  binance_bbo_age_ms
  hyperliquid_bbo_age_ms
  trailing_basis_residual
```

The exact transform order within each model and training fold is:

1. create `is_missing_<feature>` from the raw value;
2. compute the training-fold median from finite non-missing training values;
3. fail the fold if no finite training value exists for a feature;
4. fill missing train and test values with that training median;
5. compute training-fold `q25` and `q75` using one-based nearest rank;
6. set `scale=max(q75-q25, 1)` when the IQR is finite, otherwise fail;
7. emit `z_<feature>=(filled_value-training_median)/scale`;
8. append the unscaled `0/1` missing indicators.

No test or future value influences imputation or scaling. H0 columns embedded
in H1 use the same train-row universe and therefore the exact same fold
medians, IQRs and transformed bytes as standalone H0.

The exact design-matrix columns are:

```text
H0:
  01 side_maker_ask
  02 z_elapsed_session_fraction
  03 z_elapsed_session_fraction_squared
  04 z_elapsed_segment_fraction
  05 z_target_bbo_update_count_1s
  06 z_target_bbo_no_new_information_fraction_1s
  07 is_missing_elapsed_session_fraction
  08 is_missing_elapsed_session_fraction_squared
  09 is_missing_elapsed_segment_fraction
  10 is_missing_target_bbo_update_count_1s
  11 is_missing_target_bbo_no_new_information_fraction_1s

H1:
  H0 columns 01..11
  12 z_risk_gap_bps
  13 z_risk_gap_change_50ms_bps
  14 z_binance_bbo_age_ms
  15 z_hyperliquid_bbo_age_ms
  16 z_trailing_basis_residual
  17 is_missing_risk_gap_bps
  18 is_missing_risk_gap_change_50ms_bps
  19 is_missing_binance_bbo_age_ms
  20 is_missing_hyperliquid_bbo_age_ms
  21 is_missing_trailing_basis_residual
```

Side coding is exact:

```text
maker_ask_risk -> side_maker_ask=1.0
maker_bid_risk -> side_maker_ask=0.0
```

There is no ordinary intercept column because `alpha_1..alpha_5` are the five
unpenalized bin intercepts. Every listed beta column, including side and all
missing indicators, has ridge penalty weight `1.0`. No column is dropped for
zero variance or all-zero values.

The package reports missing fractions by feature/session/fold.

If any H1 primary feature has more than `5%` missing rows in either formal
session, the H1 screen is:

```text
inconclusive_feature_availability
```

It cannot be treated as a failed or passed predictability gate.

`inconclusive_feature_availability` is a stable gate reason only. Under the
Section 24 precedence it maps mechanically to the sole final classification:

```text
inconclusive_data_quality_or_coverage
```

## 17. Frozen Queue-Shock Dose Diagnostic

Queue-shock dose is diagnostic in H0-B. It is not an H3-vs-H2 test.

For side sign `s`, define:

```text
trailing_queue_shock_dose_500ms(t, s) =
  sum(queue_drop_ratio_i)
```

over accepted Stage 3 rows satisfying:

```text
primary_episode = true
direction_sign = s
decision_ts_ns > t - 500ms
decision_ts_ns <= t
queue_drop_ratio is finite and in [0, 1]
```

Only `decision_ts_ns <= t` rows are observable. Candidate/shock rows that are
not yet confirmed at `t` do not contribute.

The coarse empirical table uses:

```text
cross-spread bins:
  five equal-count bins from past training blocks

dose bins:
  exact zero
  three equal-count bins of positive training dose
```

Edges are fit independently inside each formal session/fold using past
training blocks only. Test values are clipped only to the outer open-ended
bins, not winsorized.

The table is descriptive and out-of-fold. It cannot claim formal dose
increment or rescue H1/H0.

## 18. Walk-Forward Contract

Evaluation is within-session and strictly forward.

For each formal session:

1. order accepted complete `60s` calendar blocks by absolute block ID;
2. use the first `60` complete blocks as the initial training window;
3. use consecutive `20`-block test windows;
4. use a final remainder as a test fold only when it contains at least `10`
   complete blocks;
5. expand the training window after every test fold;
6. never use future blocks for fitting, scaling, binning or thresholding.

Accepted `50ms` preflight counts imply:

| Session | Complete 60s blocks | Initial train | Valid test folds |
| --- | ---: | ---: | ---: |
| Jul30 | `232` | `60` | `8 x 20 blocks + final 12 = 9` |
| Aug04 | `119` | `60` | `2 x 20 blocks + final 19 = 3` |

Any mismatch from these accepted complete-block counts before outcome access
fails closed.

At every train/test boundary:

```text
purge = 500ms removed from the end of training
embargo = 500ms removed from the start of testing
```

The primary `50ms` label interval is therefore contained inside the frozen
purge/embargo budget.

No random row split, shuffled K-fold, leave-row-out or pooled-session split is
allowed.

Formal evaluation requires at least three valid OOF test folds per formal
session. Otherwise the result is:

```text
inconclusive_data_quality_or_coverage
```

## 19. Frozen Coarse Estimator

H0 and H1 use the same estimator family:

```text
five-bin discrete-time logistic hazard
logit(q_k(x)) = alpha_k + beta' x
```

Frozen fitting details:

```text
optimizer = L-BFGS
numeric_dtype = float64
ridge_lambda = 1.0
penalized_parameters = beta only
unpenalized_parameters = alpha_1..alpha_5
beta_penalty_weight = 1.0 for every design-matrix column
maximum_iterations = 500
gradient_tolerance = 1e-8
parameter_tolerance = 1e-10
initial_beta = 0
initial_alpha_k = -4.0
probability_clip = [1e-9, 1 - 1e-9]
```

The optimized objective is the exact sum of Section 14 interval,
full-horizon-right-censor and horizon-straddle losses plus the frozen ridge
penalty. No event bound is point-coerced to construct the objective or its
initial state.

There is:

- no lambda search;
- no estimator comparison;
- no outcome-driven feature removal;
- no post-hoc calibration;
- no session pooling.

Optimizer failure, non-finite objective, non-finite parameter or a failure to
meet the frozen convergence criterion makes the fold invalid. Invalid folds
are not silently dropped.

All empirical quantiles use one-based nearest rank:

```text
rank(p, n) = ceil(p * n)
```

Threshold equality is frozen:

```text
entry when risk_score_50ms >= entry_threshold
exit when risk_score_50ms <= exit_threshold
```

Training-defined equal-count bin edges use nearest-rank values. Duplicate
edges are retained, test assignment uses right-closed intervals and empty
tie-induced bins are reported rather than jittered. All seeded resampling
uses NumPy `Generator(PCG64)` with the frozen task seeds. A two-sided
percentile `90%` interval uses the nearest-rank `5%` and `95%` endpoints.

## 20. RQ1: Block Variation And Stationary Null

For each formal session, side and complete absolute `60s` block:

```text
block_rate =
  adverse binary endpoint count /
  binary-identified side-row count
```

The package also reports the interval-likelihood event incidence, but the
RQ1 dispersion statistic uses the identified binary block rate because the
accepted current `50ms` support has effectively complete binary
identification and the block-rate estimand is directly auditable.

The side statistic is:

```text
D_side = sample variance of complete 60s block_rate
```

The session statistic is:

```text
D_session = 0.5 * (D_maker_ask_risk + D_maker_bid_risk)
```

### 20.1 Dependence-Preserving Null

The primary null is a cadence-conditioned stationary bootstrap with mean run
length `5s = 500` grid rows. It operates on the paired-side binary outcome
path, never on independent side rows.

Before outcome access, each eligible `10ms` anchor receives:

```text
session
segment
absolute_60s_block_id
target_bbo_update_count_1s
cadence_stratum
```

Within each session/segment, quartile edges are the one-based nearest-rank
`q25/q50/q75` of finite non-missing `target_bbo_update_count_1s`. Duplicate
edges are retained. Assignment is right-closed:

```text
QM: cadence feature is missing
Q1: value <= q25
Q2: q25 < value <= q50
Q3: q50 < value <= q75
Q4: value > q75
```

Empty tie-induced strata remain explicit. Every observed target stratum must
have at least one binary-identified source anchor. The target support class,
binary-identification flag, segment, block membership and denominator are
fixed and are never resampled.

For each formal session and each of `2000` null replicates:

1. visit all target rows belonging to complete `60s` blocks in increasing
   `(segment,local_ts_ns)` order;
2. start or restart by sampling uniformly from binary-identified source
   anchors in the same session/segment/cadence stratum as the current target
   row;
3. after each emitted row, decide whether the next target row restarts with
   probability `1/500`;
4. otherwise advance to the next exact `10ms` binary-identified source anchor
   only when its cadence stratum equals the next target row's cadence stratum;
5. force a restart when either target or source timestamps are not exactly
   `+10ms`, either path crosses a real segment/epoch/quality/source-gap
   boundary, the current source stratum differs from the current target
   stratum, or the source row is not binary identified;
6. keep the target row's support/identification flags and copy only the
   complete paired-side binary outcomes from the selected source anchor;
7. allow a sampled run to continue across adjacent complete absolute `60s`
   target blocks when both target and source paths remain exact `+10ms` and
   compatible; a gap or omitted/incomplete target block forces restart;
8. rebuild every target block rate and compute `D_session`.

This is a geometric stationary bootstrap conditioned on the accepted cadence
path. It preserves exact local paired-side dependence inside sampled runs,
keeps target support, denominator and cadence composition fixed row by row,
and removes observed minute-scale outcome ordering. A disjoint
fixed-microblock permutation, resampled identification flag or source stratum
that differs from the current target stratum is forbidden.

Frozen seeds:

```text
primary_null_seed = 8232001
time_bootstrap_seed = 8232002
flow_bootstrap_seed = 8232003
rq3_bootstrap_seed = 8232004
```

Every concrete generator seed is derived as:

```text
derived_seed(base_seed, namespace) =
  unsigned big-endian integer represented by the first 16 bytes of
  SHA256("0823T002|" + decimal(base_seed) + "|" + namespace)
```

The integer initializes NumPy `Generator(PCG64)`. Primary null namespaces are
`rq1|<session>|mean_rows=500`. Robustness nulls use the same algorithm with
restart probabilities `1/250` and `1/1000` and namespaces
`rq1|<session>|mean_rows=250` and `rq1|<session>|mean_rows=1000`. They are
secondary and cannot replace the `5s` primary.

### 20.2 Gate H-A Screen

A formal session passes the H0-B RQ1 screen when:

```text
observed D_session >
  primary 5s stationary-null 95th percentile
```

The cross-session RQ1 screen passes only when both formal sessions pass.

Complete-block construction already excludes clipped segment, epoch, quality
and source-gap blocks. The cadence-stratified null prevents a simple cadence
composition difference from satisfying the gate.

## 21. RQ2: Out-Of-Fold Predictability

For each formal session:

```text
side_loss(model, side) =
  mean primary interval log loss across valid OOF rows for that side

session_loss(model) =
  0.5 * (
    side_loss(model, maker_ask_risk) +
    side_loss(model, maker_bid_risk)
  )
```

The primary normalized score is:

```text
normalized_interval_log_loss_H1_H0 =
  session_loss(H1) / session_loss(H0)
```

H1 passes the session screen only when:

```text
normalized_interval_log_loss_H1_H0 <= 0.99
```

and all of the following hold:

- binary Brier score ratio `H1/H0 <= 1.00`;
- binary log-loss ratio `H1/H0 <= 1.00`;
- top-decile minus bottom-decile OOF realized rate is positive;
- no single frozen cross-spread x dose cell contributes more than `50%` of
  the total positive H1-vs-H0 interval-loss improvement;
- time-block bootstrap `90%` CI upper bound is `<1.00`;
- flow-aware bootstrap `90%` CI upper bound is `<1.00`.

Both formal sessions must pass for the cross-session H1/H0 screen to pass.

Binary metrics and realized-rate spreads are computed per side and then
equal-weighted into the session score. Reliability and realized-rate
diagnostics use only binary-identified rows. Their fold-specific bins are
defined from past training `risk_score_50ms` values.

For the cross-spread x dose concentration check, row improvement is:

```text
improvement_i = loss_H0_i - loss_H1_i
positive_cell_contribution_side_c =
  max(0, sum_{i in side,c}(improvement_i))

side_positive_total =
  sum_c(positive_cell_contribution_side_c)

session_cell_share_c =
  0.5 * (
    positive_cell_contribution_ask_c / ask_positive_total
  ) +
  0.5 * (
    positive_cell_contribution_bid_c / bid_positive_total
  )
```

Both side denominators must be positive. Corresponding
`cross_spread_bin,dose_bin` identities are combined across sides by the exact
formula above. Cells and their edges are frozen from past training blocks, and
the formal session check is `max_c(session_cell_share_c) <= 0.50`. Net,
absolute, pooled-row or one-side-only denominators are forbidden.

### 21.1 Time-Block Bootstrap

The time bootstrap uses an exponential cluster-multiplier bootstrap over
complete `60s` OOF test blocks within session. For replicate `b`, draw one
independent `Exp(1)` multiplier for each block and apply that multiplier to
every paired-side row in the block. Predictions and fold assignments remain
fixed; models are not refit.

For each model and side, recompute the weighted mean interval loss. Recompute
the equal-weight session loss from the two weighted side means, then recompute
the H1/H0 ratio. A side with zero total multiplier weight or a non-positive H0
loss invalidates the replicate.

Use `2000` replicates and
`derived_seed(time_bootstrap_seed,"rq2_time|<session>")`. At least `20`
non-empty complete OOF blocks and at least `1900/2000` finite replicates are
required per formal session.

### 21.2 Flow-Aware Bootstrap

Accepted Stage 2 Family A `overlap_block_id` defines event-overlap components.
For each exact `(session,segment_id,connection_epoch_id,overlap_block_id)`,
the component interval is rebuilt from accepted
`candidate_episode_membership.csv.gz` as:

```text
component_start_ns = min(shock_ts_ns)
component_end_ns = max(window_end_ts_ns)
component_interval = [component_start_ns, component_end_ns]
```

The accepted merging contract guarantees that distinct component intervals
inside one segment/epoch do not overlap. Any overlap or membership/count drift
fails closed.

Each OOF grid row is assigned to:

- the accepted Family A component whose closed interval contains `t`; or
- a `2s` absolute-time background block when no accepted component
  contains `t`.

Background block identity is:

```text
background_block_id =
  floor(t / 2_000_000_000)
```

It is additionally keyed by session/segment/epoch. Component membership takes
precedence at both closed endpoints. Background units contain only rows not
assigned to a component. Assignments must be exhaustive and mutually
exclusive; ambiguity, duplication or row loss fails closed.

The flow-aware bootstrap uses one independent `Exp(1)` cluster multiplier per
complete component/background unit. The multiplier is applied to every row
and both sides in that unit, so the estimand remains the original
calendar-grid-weighted side loss rather than an equal-unit estimand. No PPS
draw or fixed row count is used.

Within every replicate:

1. compute weighted H0 and H1 mean interval loss separately for each side;
2. equal-weight the two side means into each session model loss;
3. compute the session H1/H0 ratio;
4. reject a replicate with zero side weight or non-positive H0 loss.

Use `2000` replicates and
`derived_seed(flow_bootstrap_seed,"rq2_flow|<session>")`.

Flow-bootstrap validity requires:

```text
distinct complete units >= 6
non-empty units per side >= 6
largest unit row share per side <= 0.50
finite replicate count >= 1900 of 2000
```

Failure of any rule is `inconclusive_data_quality_or_coverage`; a nominal
point loss ratio cannot bypass it. The package must separately report the
accepted Family A component counts (`Jul30=9`, `Aug04=6`) and the observed
background-unit counts before any outcome values.

## 22. RQ3: Regime Dwell And Latency

RQ3 uses only H1 OOF predictions from valid test folds.

At each fold, thresholds are fit from the past training predictions:

```text
entry_threshold = training risk_score_50ms q90
exit_threshold = training risk_score_50ms q70
entry_debounce = 3 consecutive 10ms endpoints
exit_debounce = 5 consecutive 10ms endpoints
```

For each fold, the threshold source is exact:

1. fit the joint-side H1 model on that fold's post-purge expanding training
   rows;
2. apply the same fitted preprocessing and H1 parameters back to those exact
   training rows;
3. compute `risk_score_50ms=F_5` for every interval-likelihood-eligible
   training row;
4. compute q90 and q70 separately for each side using only that side's
   training predictions and one-based nearest rank.

OOF test predictions, H0 predictions, binary-only subsets, another fold's
predictions and pooled-side predictions cannot define RQ3 thresholds.

For each side:

```text
t_detect =
  third endpoint of the first qualifying entry run

t_exit =
  fifth endpoint of the first later qualifying exit run
```

After entry, the side remains in-regime until exit or censoring; nested
re-entry is forbidden.

Regime intervals cannot cross:

- test-fold boundary;
- segment boundary;
- epoch boundary;
- core-quality/source-gap boundary.

If no valid exit is observed before the earliest boundary, the regime is
right-censored at that boundary.

For latency `L`:

```text
total_dwell =
  t_exit - t_detect

residual_dwell_L =
  total_dwell - L
```

Observed exits produce exact signed residual dwell. Censored exits produce a
right-censored lower bound.

Kaplan-Meier is fit to non-negative `total_dwell`, never directly to signed
residual values. For each latency, estimated dwell quantiles and their
confidence limits are shifted by `-L`. Report:

- regime count;
- observed-exit count;
- right-censored count;
- identified fraction =
  `observed_exit_count / admitted_regime_count`;
- dwell p10/p50/p90 where identified;
- Kaplan-Meier total-dwell p50 and shifted residual-dwell p50;
- one-sided `95%` per-side lower confidence bound for residual-dwell p50 using
  the dependency-aware cluster-multiplier bootstrap below;
- switching rate per minute.

The Kaplan-Meier ordering and inversion are exact. For each distinct observed
duration `u`, let `n_u` be the weighted risk set immediately before `u`,
`d_u` the weighted observed-exit mass at `u`, and `c_u` the weighted censor
mass at `u`. Update:

```text
S(u) = S(u-) * (1 - d_u / n_u)
next risk set removes d_u and c_u after the survival update
```

Thus observed exits are processed before censors at an equal duration. The KM
median is:

```text
median = inf{u >= 0 : S(u) <= 0.5}
```

If survival never reaches `0.5`, the median is not identified. Linear
interpolation, midpoint interpolation and treating the largest censor as an
event are forbidden.

### 22.1 Dependency-Aware RQ3 Uncertainty

Each admitted regime is assigned to the complete absolute `60s` OOF block
containing `t_detect`. Its complete duration/censor record remains attached to
that block even if the regime later crosses another absolute minute inside
the same test fold.

For each formal session and side:

1. draw one independent `Exp(1)` multiplier per distinct detection block;
2. apply the block weight to all regimes assigned to that block;
3. compute the weighted KM curve using the exact tie order above;
4. invert the weighted KM median;
5. repeat `2000` times using
   `derived_seed(rq3_bootstrap_seed,"rq3|<session>|<side>")`;
6. take the one-based nearest-rank `5%` endpoint of identifiable replicate
   medians as the one-sided `95%` lower confidence bound for total dwell;
7. subtract latency `L` from the point median and lower bound.

RQ3 is inconclusive for a formal session when either side has:

```text
distinct non-empty detection blocks < 20
point KM median not identified
identifiable bootstrap medians < 1900 of 2000
non-finite weighted KM state
```

This bootstrap is the formal uncertainty oracle. Greenwood values may be
published only as explicitly labeled descriptive diagnostics and cannot enter
a gate.

Every quantity is first computed per side. Session-level identified fraction
is the equal-weight mean of the two side fractions. The session residual-p50
lower bound is the equal-weight mean of the two one-sided `95%` side lower
bounds. By the two-side Bonferroni construction, this is the frozen one-sided
`90%` lower bound for the equal-weight mean of side-specific residual medians.
Pooled regime rows may not replace these session scores. A missing or
non-identifiable side value makes the formal session inconclusive.

### 22.2 Gate H-C Screen

A formal session passes the primary RQ3 screen only when:

```text
latency = 6600ms
equal_weight_side_identified_fraction >= 0.90
bonferroni_90_equal_weight_residual_p50_lower_bound_ms > 0
```

Both formal sessions must pass.

The `850ms` result is always:

```text
terminal_observability_normal_path_diagnostic_only
```

If `850ms` passes and `6600ms` fails:

- the primary RQ3 screen fails;
- `850ms` cannot rescue it;
- the report may set
  `execution_observability_gap_candidate=true`;
- only a later separately reviewed execution-observability plan may study
  that gap.

## 23. Stage 4 Diagnostic Crosscheck

Stage 4 Episode v3 is not the H0-B outcome oracle.

The exact pre-diagnostic primary research allowlist is:

```text
censoring_disposition.csv
exclusion_counts.csv
primary_classification.json
rq1_block_rates.csv
rq1_dispersion_tests.csv
rq2_coarse_conditional_risk.csv
rq2_feature_availability.csv
rq2_oof_fold_scores.csv
rq2_reliability.csv
rq2_risk_deciles.csv
rq2_session_scores.csv
rq3_latency_actionability.csv
rq3_regime_summary.csv
support_outcome_projection_commitments.csv
diagnostics/regime_intervals.csv.gz
diagnostics/latency_scenario_roles.csv
```

For every path, construct the canonical inventory row
`{"path":relative_path,"bytes":integer,"sha256":lowercase_hex}` and sort by
UTF-8 path bytes. Define:

```text
primary_results_sha256 =
  SHA256(canonical_json_bytes(complete sorted allowlist inventory))

primary_classification_sha256 =
  SHA256(raw canonical bytes of primary_classification.json)
```

`primary_result_seal.json` is canonical JSON with exactly these top-level keys:

```text
schema_version
task_id
primary_plan_sha256
diagnostic_plan_sha256
diagnostic_review_sha256
surface_matrix_sha256
semantic_source_inventory_sha256
build_a_primary_results_sha256
build_b_primary_results_sha256
primary_results_sha256
primary_classification_sha256
stage4_crosscheck_opened
sealed_fsynced
```

Build A and Build B primary hashes and classification hashes must match before
the seal is written. The final two booleans must be exactly `false` and `true`
respectively. The seal is fsynced before any Stage 4 opener process starts.

After the common seal is fsynced, each build must create and fsync its own
`stage4_diagnostic_permit.json` in a fresh diagnostic process. The permit has
exactly:

```text
schema_version
task_id
build_label
status
fsynced
primary_plan_sha256
diagnostic_plan_sha256
diagnostic_review_sha256
surface_matrix_sha256
runtime_source_tree_sha256
primary_seal_sha256
primary_results_sha256
primary_classification_sha256
stage4_projection_contract_sha256
```

The permit is valid only when both plan identities and the independent V2
review match dispatch, the current runtime source tree matches, and the
referenced seal/classification/results are byte-identical to the sealed
primary build. A copied, stale or pre-seal diagnostic permit fails closed
before any Stage 4 byte is opened.

Only then may a fresh diagnostic process open the following exact accepted
Jul30 Stage 4 paths:

| Path | Accepted raw SHA256 |
| --- | --- |
| `outcomes/segment_0001.csv.gz` | `669817ba04cdde44d087607d28218aaeb7bf05d4faee74c7809ef612afbb0eee` |
| `outcomes/segment_0002.csv.gz` | `4c88223823fc3af73c036bc96494c674cfd2d4368ca3393b0ae133273aee6e6d` |
| `outcomes/segment_0003.csv.gz` | `ff699a28b055028e04a2aff861446ff19685ca36684d7913b1678cd8a0fd6547` |
| `outcomes/segment_0004.csv.gz` | `88b024fb615b86e7de467911c1ad37ee91197791e0d0c276420fbd88ec0e82f2` |
| `outcomes/segment_0005.csv.gz` | `d348bc1d20477efa12da34d7bc957935a312e3a8217367fb9a30b6370ab6b3aa` |
| `outcomes/segment_0006.csv.gz` | `c6a255900e6e76dcad950bb7e43e6d1bf5f723beb34c9c494999f3b4082138ab` |
| `outcomes/segment_0007.csv.gz` | `b419533bb1a06f7cb7edaa131e66f83e4f95a73112802f3f256a8756402e73ae` |
| `outcomes/segment_0008.csv.gz` | `4c6e89b26eafb3999d70f4f10c7501302f5bc2296e198ab615b4580be941777d` |

The exact full UTF-8 header plus LF has `62` fields and SHA256:

```text
e7af9e9e84973ed074a09957435f1a10e146bb03ffa0e3365d238f8d79d01239
```

The guarded reader validates the full header but projects only:

```text
candidate_id
episode_id
t_candidate_ns
outcome_horizon_status
time_to_first_adverse_target_bbo_event_status
time_to_first_adverse_target_bbo_event_interval_lower_ns
time_to_first_adverse_target_bbo_event_interval_upper_ns
time_to_first_adverse_target_bbo_event_censor_time_ns
time_to_first_adverse_target_bbo_event_censor_reason
public_bbo_moves_through_quote
public_quote_risk_availability
```

All other Stage 4 columns are inaccessible to the diagnostic process. Stage 4
`anchors/`, `features/`, `views/` and `paths/` remain forbidden.

The aggregate crosscheck algorithm is exact:

1. join projected rows by `candidate_id` to accepted Jul30 Stage 2 Family A
   membership; require `t_candidate_ns=shock_ts_ns`, exact segment membership
   and one-to-one conservation;
2. map `direction_sign=+1` to `maker_ask_risk` and `-1` to
   `maker_bid_risk`;
3. map the landmark to
   `g=floor(t_candidate_ns/10_000_000)*10_000_000`; retain the exact accepted
   Stage 2 segment/epoch assignment and classify the H0-B landmark as:

   ```text
   identified_event
   identified_no_event
   diagnostic_censored_grid_boundary
   diagnostic_censored_interval_likelihood_only
   diagnostic_censored_right_censored_segment
   diagnostic_censored_right_censored_source_end
   diagnostic_censored_epoch_censored
   diagnostic_censored_core_quality_censored
   diagnostic_censored_source_gap_censored
   diagnostic_censored_reference_quote_unavailable
   diagnostic_censored_invalid_quote_state
   ```

   `diagnostic_censored_grid_boundary` applies when `g` is before the first
   or after the last nominal H0-B grid start for the accepted segment. The
   exact mutually exclusive precedence is:

   ```text
   1. grid_boundary
   2. accepted H0-A/H0-B support class at g
   3. identified event/no-event reconstruction
   ```

   Therefore a row that is both outside the nominal grid and beyond a segment
   endpoint is `diagnostic_censored_grid_boundary`, never segment-boundary.
   When `g` is nominal, map every legal non-binary H0-A support class to the
   same-named diagnostic-censored state above. Only
   `binary_identification_supported` proceeds to Step 4.
   `interval_likelihood_only_supported` is diagnostic-censored because this
   crosscheck compares binary `50ms` endpoints, even though the row remains
   valid for the primary interval likelihood. All nine support classes must
   be recognized and tested. These diagnostic-censored states are not
   projection mismatches and never enter an event-rate or agreement
   denominator. A missing/duplicate join, `t_candidate_ns` drift, Stage 4
   path-to-segment mismatch, accepted epoch drift, invalid direction, schema
   drift or unknown support class still fails closed with
   `H0B_STAGE4_PROJECTION_MISMATCH`;
4. only for an identified H0-B landmark, reconstruct the H0-B `50ms`
   diagnostic endpoint at `g` using the sealed event contract, without
   reading or changing model outputs;
5. define the Stage 4 `50ms` endpoint as event only when status is
   `interval_censored` and interval upper is `<=t_candidate_ns+50ms`; define
   no-event only when the first-event row is `right_censored` through at least
   that endpoint; all other rows are diagnostic-censored;
6. never use an interval midpoint, lower bound or Stage 4 Boolean as a
   substitute for Step 5;
7. define a row as eligible only when both the H0-B and Stage 4 `50ms`
   endpoints are identified. Count the union of H0-B-censored and
   Stage-4-censored rows once in `censored_count`. Aggregate by
   `segment_id,side` into joined count, H0-B identified/censored counts, every
   exact H0-B censor-reason count, Stage 4 identified/censored counts,
   H0B-only-censored, Stage4-only-censored, both-censored, eligible/censored
   counts, H0-B event rate, Stage 4 event rate, both-event, H0B-only,
   Stage4-only, neither and exact agreement fraction;
8. set `quote_risk_naming_match=true` only when every
   `public_quote_risk_availability=available` row satisfies
   `public_bbo_moves_through_quote ==
   (first-event status is interval_censored)`;
9. set `direction_mapping_match=true` only when every joined direction uses
   the exact Step 2 maker-side predicate;
10. publish a session/side summary as the deterministic sum of segment rows.

Every segment and session summary must satisfy all exact conservation rules:

```text
joined_count =
  h0b_identified_count + h0b_censored_count

h0b_censored_count =
  h0b_grid_boundary_censored_count
  + h0b_interval_likelihood_only_censored_count
  + h0b_right_censored_segment_count
  + h0b_right_censored_source_end_count
  + h0b_epoch_censored_count
  + h0b_core_quality_censored_count
  + h0b_source_gap_censored_count
  + h0b_reference_quote_unavailable_count
  + h0b_invalid_quote_state_count

joined_count =
  stage4_identified_count + stage4_censored_count

h0b_censored_count =
  h0b_only_censored_count + both_censored_count

stage4_censored_count =
  stage4_only_censored_count + both_censored_count

censored_count =
  h0b_only_censored_count
  + stage4_only_censored_count
  + both_censored_count

joined_count = eligible_count + censored_count

eligible_count =
  both_event_count + h0b_only_count
  + stage4_only_count + neither_count

h0b_event_count =
  both_event_count + h0b_only_count

stage4_event_count =
  both_event_count + stage4_only_count
```

When `eligible_count>0`, the exact finite ratios are:

```text
h0b_event_rate =
  h0b_event_count / eligible_count

stage4_event_rate =
  stage4_event_count / eligible_count

agreement_fraction =
  (both_event_count + neither_count) / eligible_count
```

When `eligible_count=0`, all three ratio cells are empty. Zero, NaN, infinity
or any other sentinel is forbidden. Any negative count, count conservation
failure, ratio mismatch or zero-denominator serialization violation is
`H0B_STAGE4_PROJECTION_MISMATCH`.

The crosscheck records whether direction mapping, endpoint sign and quote-risk
naming agree. It is not expected to produce identical row labels because the
two contracts use different vulnerable-quote landmarks.

The crosscheck:

- is Jul30-only;
- is diagnostic;
- uses only the jointly identified endpoint subset for rates and agreement;
- preserves accepted boundary candidates through explicit diagnostic
  censoring rather than silently dropping them;
- cannot modify models, thresholds, gates or primary classification;
- cannot repair a failed Build A/Build B primary comparison.

After the diagnostic file is complete, final research identity `R` is computed
from the Section 27 research allowlist, which equals the pre-diagnostic
allowlist plus exactly:

```text
diagnostics/stage4_landmark_crosscheck.csv
```

The primary seal remains byte-for-byte unchanged. The final manifest must
prove that `primary_results_sha256` still rebuilds from the pre-diagnostic
subset.

## 24. Decision Precedence

The exact precedence is:

1. data/support/feature/fold failure;
2. RQ1 cross-session variation;
3. RQ2 H1/H0 cross-session screen;
4. RQ3 `6600ms` cross-session actionability;
5. positive modeling-candidate result.

Allowed H0-B classifications:

### 24.1 `inconclusive_data_quality_or_coverage`

Use when:

- support replay does not match accepted H0-A;
- a formal session loses the accepted support gate;
- H1 primary feature missingness exceeds `5%`;
- fewer than three valid OOF folds remain;
- required binary/interval/regime denominators are zero;
- optimizer or bootstrap validity fails.

### 24.2 `quote_risk_flat_reactive_signal_not_indicated`

Use only when both formal sessions fail the RQ1 variation screen.

This does not mean maker is good, bad or unnecessary. It means the current
reactive-risk path is not indicated by RQ1.

### 24.3 `cross_session_unstable_needs_more_sessions`

Use when exactly one formal session passes any current required
cross-session screen and the other is valid but does not pass.

Aug03 cannot resolve the disagreement.

### 24.4 `h0b_coarse_cross_spread_predictability_not_indicated`

Use when both formal sessions pass RQ1 but both fail the frozen H1/H0 screen.

This rejects the current coarse cross-spread screen. It does not claim that
every possible H2/H3/H4 state vector is unpredictable.

### 24.5 `predictable_but_not_latency_actionable`

Use when both formal sessions pass RQ1 and H1/H0, but both fail the `6600ms`
RQ3 screen.

The `850ms` diagnostic is recorded but cannot change this classification.

### 24.6 `h0b_main_modeling_candidate`

Use only when both formal sessions pass:

```text
RQ1 variation
H1/H0 predictability
RQ3 at 6600ms
```

This permits a later main H0-H4 modeling plan. It does not equal
`conditional_quote_risk_signal_supported`.

## 25. Required Research Outputs

Candidate formal package root:

```text
local_live_analysis/
skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002
```

Required directories:

```text
contracts
diagnostics
reports
runtime_source
runtime_tests
```

Required root files:

```text
accepted_input_bindings.json
censoring_disposition.csv
exclusion_counts.csv
h0b_manifest.json
outcome_access_ledger_build_a.json
outcome_access_ledger_build_b.json
outcome_access_permit_build_a.json
outcome_access_permit_build_b.json
preoutcome_contract.json
preoutcome_source_inventory.csv
primary_classification.json
primary_result_seal.json
rq1_block_rates.csv
rq1_dispersion_tests.csv
rq2_coarse_conditional_risk.csv
rq2_feature_availability.csv
rq2_oof_fold_scores.csv
rq2_reliability.csv
rq2_risk_deciles.csv
rq2_session_scores.csv
rq3_latency_actionability.csv
rq3_regime_summary.csv
support_outcome_projection_commitments.csv
support_replay_receipt_build_a.json
support_replay_receipt_build_b.json
stage4_diagnostic_permit_build_a.json
stage4_diagnostic_permit_build_b.json
stage4_diagnostic_receipt_build_a.json
stage4_diagnostic_receipt_build_b.json
```

Required contract files:

```text
contracts/accepted_kernel_pin.json
contracts/execution_plan.md
contracts/surface_matrix.json
contracts/task.md
contracts/v2_framework.md
```

Required diagnostics:

```text
diagnostics/regime_intervals.csv.gz
diagnostics/stage4_landmark_crosscheck.csv
diagnostics/latency_scenario_roles.csv
```

Required report:

```text
reports/h0b_conditional_risk_audit.md
```

Required runtime:

```text
runtime_source/skhynix_stage_h0b.py
runtime_source/skhynix_stage_h0b_contracts.py
runtime_tests/test_skhynix_stage_h0b.py
runtime_tests/test_skhynix_stage_h0b_package.py
```

No row-level 10ms grid or row-level OOF prediction file is published.
Canonical per-segment/per-side streaming commitments provide replay identity.

Maximum formal package size:

```text
134217728 bytes
```

## 26. Output Semantics

### 26.1 `support_outcome_projection_commitments.csv`

One row per:

```text
session
segment
side
```

Fields include:

```text
support_row_count
interval_likelihood_row_count
binary_row_count
event_observed_count
full_horizon_right_censor_count
horizon_straddle_count
geometric_exclusion_count
canonical_projection_sha256
first_grid_ts_ns
last_grid_ts_ns
```

The canonical projection hashes the complete ordered derived row tuple without
publishing it.

### 26.2 `rq1_block_rates.csv`

One row per session/side/complete block. It includes exact denominators and
does not include a pooled formal rate.

### 26.3 `rq2_oof_fold_scores.csv`

One row per session/fold/model. It includes train/test block ranges, purge,
embargo, convergence, denominators and losses.

### 26.4 `rq3_regime_summary.csv`

One row per session/side/latency role. `6600ms` is the only primary row.

### 26.5 `primary_classification.json`

It contains exactly one allowed classification, exact gate facts and the
precedence path. It must not contain strategy recommendations.

### 26.6 Canonical Serialization

All CSV files use UTF-8, LF, comma delimiter, one exact header, RFC 4180
quoting only when required and no blank lines. Integers are base-10 without
leading zeros; booleans are `true`/`false`; missing cells are empty; finite
float64 values use `format(value,".17g")`; NaN and infinity are forbidden.
CSV rows use the key order stated below.

All JSON uses accepted Trust Kernel
`canonical_pretty_json_bytes(indent=2,sort_keys=true,trailing_newline=true)`.
Unknown keys, bool-as-integer, NaN and infinity fail closed.

Deterministic gzip uses:

```text
filename = empty
mtime = 0
compresslevel = 1
```

### 26.7 Exact CSV Headers

The exact ordered headers are:

```text
censoring_disposition.csv
session,segment_id,side,identification_class,disposition,row_count,reason_code

exclusion_counts.csv
session,segment_id,side,stage,reason_code,row_count

preoutcome_source_inventory.csv
session,segment_id,source_role,relative_path,bytes,sha256,header_sha256

support_outcome_projection_commitments.csv
session,segment_id,side,support_row_count,interval_likelihood_row_count,binary_row_count,event_observed_count,full_horizon_right_censor_count,horizon_straddle_count,geometric_exclusion_count,canonical_projection_sha256,first_grid_ts_ns,last_grid_ts_ns

rq1_block_rates.csv
session,side,absolute_block_id,block_start_ns,block_end_ns,binary_identified_count,event_count,block_rate

rq1_dispersion_tests.csv
session,ask_variance,bid_variance,observed_d_session,primary_mean_run_rows,null_replicates,null_p95,primary_pass,robustness_250_p95,robustness_1000_p95

rq2_coarse_conditional_risk.csv
session,fold_id,side,cross_spread_bin,dose_bin,row_count,binary_identified_count,event_count,realized_rate,mean_loss_h0,mean_loss_h1,positive_improvement,cell_share

rq2_feature_availability.csv
session,fold_id,model,feature,row_count,missing_count,missing_fraction,training_median,training_q25,training_q75,scale,availability_gate_pass

rq2_oof_fold_scores.csv
session,fold_id,model,train_first_block,train_last_block,test_first_block,test_last_block,purge_ns,embargo_ns,train_row_count,test_row_count,ask_test_rows,bid_test_rows,converged,iterations,objective,ask_interval_log_loss,bid_interval_log_loss,session_interval_log_loss,ask_brier,bid_brier,session_brier,ask_binary_log_loss,bid_binary_log_loss,session_binary_log_loss

rq2_reliability.csv
session,fold_id,model,side,reliability_bin,training_lower_edge,training_upper_edge,test_count,event_count,mean_predicted_risk,realized_rate

rq2_risk_deciles.csv
session,fold_id,model,side,decile,training_lower_edge,training_upper_edge,test_count,event_count,mean_predicted_risk,realized_rate

rq2_session_scores.csv
session,h0_interval_log_loss,h1_interval_log_loss,normalized_interval_log_loss_h1_h0,h0_brier,h1_brier,brier_ratio_h1_h0,h0_binary_log_loss,h1_binary_log_loss,binary_log_loss_ratio_h1_h0,top_bottom_realized_rate_spread,max_positive_cell_share,time_ci_lower,time_ci_upper,flow_ci_lower,flow_ci_upper,rq2_pass,gate_reason

rq3_latency_actionability.csv
session,latency_ms,latency_role,primary,ask_identified_fraction,bid_identified_fraction,equal_weight_identified_fraction,ask_residual_p50_ms,bid_residual_p50_ms,equal_weight_residual_p50_ms,ask_lower95_ms,bid_lower95_ms,bonferroni90_equal_weight_lower_ms,session_pass,can_rescue_primary

rq3_regime_summary.csv
session,side,latency_ms,latency_role,regime_count,observed_exit_count,right_censored_count,identified_fraction,dwell_p10_ms,dwell_p50_ms,dwell_p90_ms,km_total_dwell_p50_ms,residual_dwell_p50_ms,residual_dwell_lower95_ms,switching_rate_per_minute,distinct_detection_blocks,identifiable_bootstrap_replicates

diagnostics/regime_intervals.csv.gz
session,fold_id,side,regime_id,t_detect_ns,t_exit_ns,censor_time_ns,censored,total_dwell_ns,detection_block_id,entry_threshold,exit_threshold

diagnostics/stage4_landmark_crosscheck.csv
scope,session,segment_id,side,joined_count,h0b_identified_count,h0b_censored_count,h0b_grid_boundary_censored_count,h0b_interval_likelihood_only_censored_count,h0b_right_censored_segment_count,h0b_right_censored_source_end_count,h0b_epoch_censored_count,h0b_core_quality_censored_count,h0b_source_gap_censored_count,h0b_reference_quote_unavailable_count,h0b_invalid_quote_state_count,stage4_identified_count,stage4_censored_count,h0b_only_censored_count,stage4_only_censored_count,both_censored_count,eligible_count,censored_count,h0b_event_count,stage4_event_count,both_event_count,h0b_only_count,stage4_only_count,neither_count,h0b_event_rate,stage4_event_rate,agreement_fraction,direction_mapping_match,quote_risk_naming_match,primary_seal_unchanged

diagnostics/latency_scenario_roles.csv
latency_ms,latency_role,primary,diagnostic,can_rescue_primary,source_authority
```

Row ordering is the lexical/numeric tuple implied by the header's leading key
columns: session order `jul30,aug03,aug04`; side order
`maker_ask_risk,maker_bid_risk`; model order `H0,H1`; latency numeric
ascending; segment/fold/block/bin/regime numeric ascending. Summary rows use
`scope=session` and `segment_id=ALL` after segment rows.

### 26.8 Exact JSON Key Universes

The root JSON objects and exact top-level keys are:

```text
accepted_input_bindings.json:
  schema_version
  task_id
  primary_plan_sha256
  diagnostic_plan_sha256
  diagnostic_review_sha256
  surface_matrix_sha256
  expected_semantic_source_inventory_sha256
  bindings

h0b_manifest.json:
  schema_version
  task_id
  status
  primary_plan_sha256
  diagnostic_plan_sha256
  diagnostic_review_sha256
  primary_results_sha256
  primary_classification_sha256
  stage4_crosscheck_sha256
  research_data_identity
  runtime_contract_identity
  publication_envelope_identity
  composite_package_identity
  package_file_count
  package_total_bytes

outcome_access_ledger_build_a.json and
outcome_access_ledger_build_b.json:
  schema_version
  task_id
  build_label
  events

outcome_access_permit_build_a.json and
outcome_access_permit_build_b.json:
  schema_version
  task_id
  build_label
  status
  fsynced
  primary_plan_sha256
  diagnostic_plan_sha256
  diagnostic_review_sha256
  surface_matrix_sha256
  runtime_source_tree_sha256
  preoutcome_contract_sha256
  source_inventory_contract_sha256
  semantic_source_inventory_sha256
  build_envelope
  build_envelope_sha256
  support_replay_receipt_sha256
  accepted_input_bindings_sha256

preoutcome_contract.json:
  schema_version
  task_id
  primary_plan_sha256
  diagnostic_plan_sha256
  diagnostic_review_sha256
  surface_matrix_sha256
  source_inventory_contract_sha256
  likelihood_contract_sha256
  design_matrix_contract_sha256
  walk_forward_contract_sha256
  resampling_contract_sha256
  classification_contract_sha256
  output_contract_sha256
  runtime_source_tree_sha256

primary_classification.json:
  schema_version
  task_id
  classification
  precedence_path
  gate_reasons
  formal_session_facts
  latency_roles
  claim_limit

primary_result_seal.json:
  schema_version
  task_id
  primary_plan_sha256
  diagnostic_plan_sha256
  diagnostic_review_sha256
  surface_matrix_sha256
  semantic_source_inventory_sha256
  build_a_primary_results_sha256
  build_b_primary_results_sha256
  primary_results_sha256
  primary_classification_sha256
  stage4_crosscheck_opened
  sealed_fsynced

stage4_diagnostic_permit_build_a.json and
stage4_diagnostic_permit_build_b.json:
  schema_version
  task_id
  build_label
  status
  fsynced
  primary_plan_sha256
  diagnostic_plan_sha256
  diagnostic_review_sha256
  surface_matrix_sha256
  runtime_source_tree_sha256
  primary_seal_sha256
  primary_results_sha256
  primary_classification_sha256
  stage4_projection_contract_sha256

stage4_diagnostic_receipt_build_a.json and
stage4_diagnostic_receipt_build_b.json:
  schema_version
  task_id
  build_label
  primary_plan_sha256
  diagnostic_plan_sha256
  diagnostic_review_sha256
  diagnostic_permit_sha256
  stage4_crosscheck_sha256
  primary_results_sha256
  primary_classification_sha256
  primary_seal_sha256
  stage4_path_count
  stage4_projected_field_count
  joined_count
  eligible_count
  censored_count
  primary_seal_unchanged

support_replay_receipt_build_a.json and
support_replay_receipt_build_b.json:
  schema_version
  task_id
  build_label
  accepted_h0a_commitments_sha256
  observed_h0a_commitments_sha256
  exact_commitment_match
  forbidden_outcome_access_count
  replay_row_count
```

Each `bindings[]` object has exactly:

```text
binding_id,authority_path,bytes,sha256,status
```

Each `events[]` ledger object has exactly:

```text
sequence,process_role,phase,relative_path,access_kind,bytes_read,permit_sha256,admitted
```

Each `build_envelope` object has exactly the six keys frozen in Section 8.2.
`formal_session_facts` has exact keys `jul30,aug04`; each value has exact keys
`rq1,rq2,rq3,data_quality`. `latency_roles` has exact keys `850,6600`.
`precedence_path`, `gate_reasons` and `bindings` are ordered arrays;
all other unknown nested keys fail closed. The canonical Surface Matrix
contains recursive JSON Schema objects for these nested values and their
exact scalar types.

### 26.9 Exact Package Report Template

`reports/h0b_conditional_risk_audit.md` is a package-internal evidence file,
not the workflow business report. It uses UTF-8, LF, no trailing spaces,
exactly one terminal LF and this exact line template:

```text
# Stage H0-B Conditional-Risk Audit

- task: `0823T002`
- status: `待验收`
- classification: `<classification>`
- primary_plan_sha256: `c1be0fdbd58f19c201c2faa7251621402486e6ebabf259af316b98bcf4c92b10`
- diagnostic_plan_sha256: `<diagnostic_plan_sha256>`
- diagnostic_review_sha256: `<diagnostic_review_sha256>`
- formal_sessions: `jul30,aug04`
- primary_tuple: `public_bbo_moves_through_quote/delta=0/horizon=50ms/latency=6600ms/equal_weight_bid_ask_session_scores`
- rq1_jul30_pass: `<true_false_or_NA>`
- rq1_aug04_pass: `<true_false_or_NA>`
- rq2_jul30_ratio_time_upper_flow_upper_pass: `<ratio>/<time_upper>/<flow_upper>/<true_false_or_NA>`
- rq2_aug04_ratio_time_upper_flow_upper_pass: `<ratio>/<time_upper>/<flow_upper>/<true_false_or_NA>`
- rq3_6600_jul30_lower_ms_pass: `<lower_ms>/<true_false_or_NA>`
- rq3_6600_aug04_lower_ms_pass: `<lower_ms>/<true_false_or_NA>`
- rq3_850_role: `terminal_observability_normal_path_diagnostic_only/non_rescue`
- primary_results_sha256: `<primary_results_sha256>`
- primary_classification_sha256: `<primary_classification_sha256>`
- stage4_crosscheck_sha256: `<stage4_crosscheck_sha256>`
- research_data_identity: `<R>`
- code_contract_identity: `<C>`
- evidence_identity/composite_identity: bound by `h0b_manifest.json` to avoid report self-reference
- outcome_access: `public_only_after_build_specific_admitted_permits`
- stage4_access: `post_primary_seal_build_specific_diagnostic_permits_exact_projection_only`
- aug07_access: `false`
- network/private/order/cancel/live_access: `false`
- claim_limit: `screening_audit_not_final_signal_or_strategy`
```

Placeholder rendering is exact:

- classification is one Section 24 enum;
- evaluated gate states are lower-case `true`/`false`;
- an inconclusive or precedence-not-evaluated gate state is literal `NA`;
- finite evaluated numeric values use Section 26.6 `.17g`;
- an unavailable, inconclusive or not-evaluated numeric is literal `NA`;
- SHA and identity placeholders are lower-case 64-hex.

The source binding is exact:

```text
rq1_<session>_pass =
  rq1_dispersion_tests.primary_pass for that session

rq2 ratio =
  rq2_session_scores.normalized_interval_log_loss_h1_h0

rq2 time_upper =
  rq2_session_scores.time_ci_upper

rq2 flow_upper =
  rq2_session_scores.flow_ci_upper

rq2 pass =
  rq2_session_scores.rq2_pass

rq3 lower_ms =
  rq3_latency_actionability.bonferroni90_equal_weight_lower_ms
  where latency_ms=6600 for that session

rq3 pass =
  rq3_latency_actionability.session_pass
  where latency_ms=6600 for that session
```

In the canonical CSVs, gate-state cells are `true`/`false` when evaluated and
empty when inconclusive or not evaluated. The package report maps an empty
gate-state or numeric cell to `NA`; it never maps inconclusive to `false`.

No package report line may contain E, composite, business commit, QA status or
controller acceptance because those values do not exist before E is computed.

The external workflow business report is:

```text
.workflow/reports/0823T002-business.md
```

It is outside the package tree and outside R/C/E. Gate 7 refers to this
external report. It may record final E/composite, the business commit and
`待验收` handoff after the package manifest has been sealed.

## 27. Layered Identity

The package uses accepted Trust Kernel v1.

Research-data identity `R` includes:

```text
censoring_disposition.csv
exclusion_counts.csv
primary_classification.json
rq1_block_rates.csv
rq1_dispersion_tests.csv
rq2_coarse_conditional_risk.csv
rq2_feature_availability.csv
rq2_oof_fold_scores.csv
rq2_reliability.csv
rq2_risk_deciles.csv
rq2_session_scores.csv
rq3_latency_actionability.csv
rq3_regime_summary.csv
support_outcome_projection_commitments.csv
diagnostics/regime_intervals.csv.gz
diagnostics/stage4_landmark_crosscheck.csv
diagnostics/latency_scenario_roles.csv
```

Runtime-contract identity `C` includes:

```text
preoutcome_contract.json
contracts/accepted_kernel_pin.json
contracts/execution_plan.md
contracts/surface_matrix.json
contracts/task.md
contracts/v2_framework.md
runtime_source/skhynix_stage_h0b.py
runtime_source/skhynix_stage_h0b_contracts.py
runtime_tests/test_skhynix_stage_h0b.py
runtime_tests/test_skhynix_stage_h0b_package.py
```

Evidence identity `E` includes:

```text
accepted_input_bindings.json
outcome_access_ledger_build_a.json
outcome_access_ledger_build_b.json
outcome_access_permit_build_a.json
outcome_access_permit_build_b.json
preoutcome_source_inventory.csv
primary_result_seal.json
support_replay_receipt_build_a.json
support_replay_receipt_build_b.json
stage4_diagnostic_permit_build_a.json
stage4_diagnostic_permit_build_b.json
stage4_diagnostic_receipt_build_a.json
stage4_diagnostic_receipt_build_b.json
reports/h0b_conditional_risk_audit.md
```

Every R/C/E inventory row has exactly:

```text
path,bytes,sha256
```

The exact inventory cardinalities are:

```text
R files = 17
C files = 10
E files = 14
manifest files excluded from E = 1
exact regular files in tree = 42
h0b_manifest.package_file_count = 41
```

Paths are canonical POSIX package-relative strings, rows are sorted by UTF-8
path bytes, and bytes/SHA are raw-file values. Absolute root, host, build
label, mtime, ctime and inode never enter an inventory row.

The exact `runtime_contract_bridge` object passed to
`compute_runtime_contract_identity` has these top-level keys:

```text
schema_version
task_id
kernel_source_tree_sha256
primary_plan_sha256
diagnostic_plan_sha256
diagnostic_review_sha256
surface_matrix_sha256
files
research_surface_assignments_sha256
output_schema_contract_sha256
contract_versions
hard_boundary
```

Their exact scalar values/rules are:

```text
schema_version = skhynix_stage_h0b_runtime_contract_bridge_v2
task_id = 0823T002
kernel_source_tree_sha256 =
  cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203
primary_plan_sha256 =
  c1be0fdbd58f19c201c2faa7251621402486e6ebabf259af316b98bcf4c92b10
diagnostic_plan_sha256 =
  dispatch-pinned independently reviewed V2 SHA256
diagnostic_review_sha256 =
  dispatch-pinned independent V2 review SHA256
surface_matrix_sha256 = dispatch-pinned canonical matrix SHA256
files = complete sorted C inventory above
research_surface_assignments_sha256 =
  SHA256(canonical JSON of the exact projection below)
output_schema_contract_sha256 =
  preoutcome_contract.output_contract_sha256
```

The assignment projection is built only from fields admitted by
`research-package-surface-matrix-v1`. For every
`surface in matrix.surfaces` and every `artifact in surface.artifacts`, emit:

```text
{
  "surface_id": surface.surface_id,
  "path": artifact.path,
  "entry_type": artifact.entry_type,
  "required": artifact.required,
  "identity_layer": surface.identity_layer,
  "exact_contract_sha256":
    TrustKernel.canonical_json_sha256(surface.exact_contract)
}
```

Include every surface/artifact row, including directory artifacts and all
identity layers. Sort rows by the exact tuple
`(path,surface_id,entry_type)`, then hash the canonical JSON array. Missing
artifact fields, a changed `exact_contract`, hashing the unsorted projection,
an extra projection key or reuse of the old C identity must fail the
`layered_identity` negative fixture.

Before hashing, package-owned assignments must pass an exact set oracle.
Define:

```text
package_prefix =
  local_live_analysis/
  skhynix_continuous_conditional_risk_v2_stage_h0b_0823T002/
```

An artifact is package-owned only when its repository-relative `path` starts
with the exact prefix. Normalize it by removing that prefix; any empty,
absolute, `..`, alternate-root or similarly named prefix fails closed.

For package-owned regular files require:

```text
entry_type = regular_file
required = true
exactly one matrix surface owns the path
surface.identity_layer matches the set below
```

The exact normalized set equalities are:

```text
normalized package-owned layer R regular files
  == exact Section 27 R file list

normalized package-owned layer C regular files
  == exact Section 27 C file list

normalized package-owned layer E regular files
  == exact Section 27 E file list
     union {"h0b_manifest.json"}

union of normalized package-owned R/C/E regular files
  == exact Section 25 required file list
```

For package-owned directories require `entry_type=directory`,
`required=true`, layer `E`, and exact normalized set:

```text
{"contracts","diagnostics","reports","runtime_source","runtime_tests"}
```

No package-owned path may appear twice, be omitted, use the wrong layer or
normalize from the wrong prefix. External workflow evidence artifacts remain
in the complete projection hash but are excluded from the package-owned set
equalities. Hostile fixtures must cover omitted assignment, wrong layer,
wrong entry type, `required=false` and wrong package prefix.

`contract_versions` has exactly:

```text
observation_bounds = h0a_hyperliquid_bbo_receive_interval_v1
interval_likelihood = h0b_piecewise_constant_hazard_v1
design_matrix = h0b_h0_h1_design_matrix_v1
walk_forward = h0b_expanding_60_20_v1
resampling = h0b_dependency_resampling_v1
classification = h0b_screening_classification_v1
output_schema = h0b_output_schema_v1
identity_bridge = h0b_layered_identity_bridge_v1
```

`hard_boundary` has exactly these keys, all `false`:

```text
aug07_event_rows_read
stage4_before_primary_seal
r1_decision_labels_read
network_accessed
private_endpoint_accessed
order_or_cancel_accessed
live_action_executed
```

The exact `publication_envelope_bridge` object passed to
`compute_publication_envelope_identity` has these top-level keys:

```text
schema_version
task_id
files
expected_files
expected_directories
manifest_excluded_path
manifest_self_binding_normalization
package_file_count_rule
package_total_bytes_rule
atomic_publication
verify_only_zero_write
archive_scope
kernel_package_admission_portable
full_source_semantic_replay_portable
outcome_values_present
```

Its exact values/rules are:

```text
schema_version = skhynix_stage_h0b_publication_envelope_bridge_v1
task_id = 0823T002
files = complete sorted E inventory above
expected_files =
  UTF-8 sorted exact union of every Section 25 required file,
  including h0b_manifest.json
expected_directories =
  ["contracts","diagnostics","reports","runtime_source","runtime_tests"]
manifest_excluded_path = h0b_manifest.json
manifest_self_binding_normalization =
  manifest_excluded_from_E_inventory_and_package_count_bytes
package_file_count_rule =
  exact_regular_file_count_excluding_h0b_manifest
package_total_bytes_rule =
  exact_regular_file_bytes_excluding_h0b_manifest
verify_only_zero_write = true
archive_scope = package_only
kernel_package_admission_portable = true
full_source_semantic_replay_portable = false
outcome_values_present = true
```

`atomic_publication` has exactly:

```text
no_overwrite = true
fsync_tree_before_rename = true
atomic_rename = true
```

Accordingly, `h0b_manifest.json.package_file_count` and
`package_total_bytes` exclude only `h0b_manifest.json` itself. Admission also
checks that the actual exact tree has one additional regular file, the
manifest, and no additional directory, symlink or special entry. Mutating any
bridge key, scalar, inventory row/order, count rule, portability flag or
publication boolean changes C or E and must fail old-identity admission.

`h0b_manifest.json` is the package's unique identity seal. Following accepted
Trust Kernel v1 self-reference exclusion, it is present in the exact tree but
excluded from the publication-envelope inventory used to compute `E`. The
manifest is written only after R, C, E and composite are known and contains
their exact values. Admission validates its canonical raw bytes against those
recomputed values.

There is no other excluded file. In particular, `primary_result_seal.json`
participates in `E`, and `diagnostics/stage4_landmark_crosscheck.csv`
participates in final `R`.

The final identity relation is:

```text
R = TrustKernel.compute_research_data_identity(
      complete Section 27 R inventory, including Stage 4 diagnostic
    )

C = TrustKernel.compute_runtime_contract_identity(
      R,
      exact runtime_contract_bridge above
    )

E = TrustKernel.compute_publication_envelope_identity(
      R,
      C,
      exact publication_envelope_bridge above
    )

composite = TrustKernel.compute_composite_package_identity(R,C,E)
```

Mutating the post-seal Stage 4 diagnostic must change R and therefore require
new C, E, composite and manifest bytes, while leaving
`primary_result_seal.json` unchanged. Reusing old C/E after any R mutation is
rejected before a trusted composite is returned.

Build-root absolute paths are evidence fields but are excluded from semantic
research identity through the accepted Trust Kernel normalization contract.

## 28. Load-Bearing Surface Matrix

The formal Surface Matrix contains exactly the following `61` load-bearing
surfaces. Every row has one unique stable failure code and one independently
executed negative mutation:

| Surface | Exact contract | Required negative mutation | Stable failure code |
| --- | --- | --- | --- |
| `kernel_pin` | accepted Trust Kernel v1 exact | alter registry/kernel pin | `H0B_KERNEL_PIN_MISMATCH` |
| `master_framework_pin` | exact accepted v2 framework | alter framework SHA | `H0B_MASTER_FRAMEWORK_MISMATCH` |
| `accepted_h0a_binding` | exact H0-A tuple/package/QA/closure | alter one identity | `H0B_H0A_IDENTITY_MISMATCH` |
| `accepted_latency_binding` | exact 0822T002 package/QA/closure | alter one identity | `H0B_LATENCY_IDENTITY_MISMATCH` |
| `accepted_tuple_binding` | exact 0823T001 tuple/package/QA/closure | alter one identity | `H0B_TUPLE_IDENTITY_MISMATCH` |
| `accepted_stage1_4_binding` | exact inherited dependency set | substitute package | `H0B_DEPENDENCY_IDENTITY_MISMATCH` |
| `session_roles` | Jul30/Aug04 formal; Aug03 diagnostic | promote Aug03 | `H0B_SESSION_ROLE_MISMATCH` |
| `underlying_state_boundary` | unknown; no calendar inference | inject KRX state | `H0B_UNDERLYING_STATE_INFERENCE_FORBIDDEN` |
| `semantic_source_inventory` | dispatch-pinned cross-build semantic inventory | add/remove source | `H0B_SEMANTIC_INVENTORY_MISMATCH` |
| `build_envelope` | exact root/process/runtime envelope | copy A envelope to B | `H0B_BUILD_ENVELOPE_MISMATCH` |
| `source_schema` | exact R0 and Stage 4 header identities | alter one header | `H0B_SOURCE_SCHEMA_MISMATCH` |
| `source_ordering` | `(local_ts_ns,event_seq)` exact | swap same-ts sequence | `H0B_SOURCE_ORDERING_MISMATCH` |
| `guarded_opener` | reject before forbidden read | open forbidden path | `H0B_FORBIDDEN_PATH_ACCESS` |
| `feature_source_boundary` | R0 BBO/bookTicker only | open R1 future labels | `H0B_FEATURE_SOURCE_BOUNDARY_MISMATCH` |
| `two_envelope_boundary` | H0B0 then fresh H0B1 | reuse process | `H0B_OUTCOME_ACCESS_BEFORE_PERMIT` |
| `outcome_access_permit` | exact fsynced semantic+build binding | stale/copy permit | `H0B_OUTCOME_PERMIT_MISMATCH` |
| `support_replay` | exact H0-A commitments | alter one commitment | `H0B_SUPPORT_COMMITMENT_MISMATCH` |
| `calendar_grid` | exact 10ms absolute grid | drift origin by 1ns | `H0B_CALENDAR_GRID_MISMATCH` |
| `side_expansion` | exactly paired bid/ask rows | drop one side | `H0B_SIDE_PAIR_MISMATCH` |
| `event_definition` | opposing BBO crosses vulnerable quote | use midpoint/retreat | `H0B_EVENT_DEFINITION_MISMATCH` |
| `support_class_mapping` | exact nine-class disposition | coerce interval-only | `H0B_SUPPORT_CLASS_DISPOSITION_MISMATCH` |
| `observation_bounds` | endpoint/search-limit branches and source-order-aware `(L,U]` exact | truncate interval-only search at horizon | `H0B_OBSERVATION_BOUND_MISMATCH` |
| `interval_likelihood` | exact piecewise-constant `S(L)-S(U)` | round bounds to bins | `H0B_INTERVAL_LIKELIHOOD_MISMATCH` |
| `right_censor_likelihood` | exact `S_5` | encode no-event as zero-time | `H0B_RIGHT_CENSOR_MISMATCH` |
| `horizon_straddle` | exact `S_exact(L)` | drop straddle row | `H0B_HORIZON_STRADDLE_MISMATCH` |
| `risk_score` | exact `F_5=1-S_5` | use one-bin hazard | `H0B_RISK_SCORE_MISMATCH` |
| `binary_subset` | binary class only | include interval-only | `H0B_BINARY_SUBSET_MISMATCH` |
| `h0_features` | exact context allowlist | add future/calendar feature | `H0B_H0_FEATURE_ALLOWLIST_MISMATCH` |
| `h1_features` | exact cross-spread allowlist | add outcome/post-t feature | `H0B_H1_FEATURE_ALLOWLIST_MISMATCH` |
| `design_matrix` | exact columns/order/coding/penalty mask | reorder/drop indicator | `H0B_DESIGN_MATRIX_MISMATCH` |
| `basis_residual` | prior-only 60s EWMA | full-session demean | `H0B_BASIS_RESIDUAL_MISMATCH` |
| `missing_value_policy` | train-only median/IQR + indicator | use test median | `H0B_MISSING_VALUE_POLICY_MISMATCH` |
| `dose_definition` | exact accepted Stage 3 trailing dose | use shock before confirm | `H0B_DOSE_RECONSTRUCTION_MISMATCH` |
| `walk_forward` | exact 60/20 blocks, purge/embargo | random split | `H0B_WALK_FORWARD_MISMATCH` |
| `estimator` | exact five-bin ridge logistic | tune lambda after outcome | `H0B_ESTIMATOR_CONTRACT_MISMATCH` |
| `numeric_seed_conventions` | float64/nearest-rank/PCG64/derived seeds | change quantile/RNG | `H0B_NUMERIC_CONVENTION_MISMATCH` |
| `rq1_statistic` | exact side/session variance | pool side rows | `H0B_RQ1_STATISTIC_MISMATCH` |
| `rq1_stationary_null` | row-matched cadence strata, fixed support and geometric runs | resample support/use source-only stratum | `H0B_RQ1_NULL_MISMATCH` |
| `rq2_score` | exact equal-side H1/H0 interval loss | choose favorable metric | `H0B_RQ2_SCORE_MISMATCH` |
| `rq2_concentration` | exact positive-cell contribution and 50% cap | use net/absolute cell sum | `H0B_RQ2_CONCENTRATION_MISMATCH` |
| `time_bootstrap` | 60s Exp(1) cluster multipliers | row bootstrap | `H0B_TIME_BOOTSTRAP_MISMATCH` |
| `flow_component_assignment` | exact Family A closed components/background | double-assign endpoint | `H0B_FLOW_COMPONENT_MISMATCH` |
| `flow_bootstrap` | exact unit multipliers/validity rules | equal-unit estimand | `H0B_FLOW_BOOTSTRAP_MISMATCH` |
| `rq3_threshold_source` | same-fold H1 training predictions per side | use OOF/pooled threshold | `H0B_RQ3_THRESHOLD_MISMATCH` |
| `rq3_regime` | q90/q70 and 3/5 debounce | post-hoc threshold/debounce | `H0B_RQ3_REGIME_MISMATCH` |
| `rq3_km_ties` | events-before-censors, exact inversion | censor first/interpolate | `H0B_RQ3_KM_MISMATCH` |
| `rq3_cluster_bootstrap` | detection-block Exp(1) KM bootstrap | Greenwood gate CI | `H0B_RQ3_BOOTSTRAP_MISMATCH` |
| `rq3_side_aggregation` | side p50/LB equal weight + Bonferroni | pooled regimes | `H0B_RQ3_SIDE_AGGREGATION_MISMATCH` |
| `latency_roles` | 6600 primary, 850 diagnostic | promote/rescue with 850 | `H0B_PRIMARY_LATENCY_MISMATCH` |
| `classification_precedence` | exact allowed exits and gate-reason mapping | issue final signal claim | `H0B_CLASSIFICATION_MISMATCH` |
| `primary_result_seal` | exact pre-diagnostic allowlist/hash/schema plus distinct V1 primary and V2 diagnostic identities | omit/mutate sealed path or collapse plan identities | `H0B_PRIMARY_SEAL_MISMATCH` |
| `stage4_projection` | eight exact paths/header/projected fields, full support-state mapping and censor precedence | read extra Stage 4 field or alter boundary disposition | `H0B_STAGE4_PROJECTION_MISMATCH` |
| `stage4_crosscheck` | post-seal build-specific diagnostic permits, aggregate conservation and unchanged primary | open before permit/seal, copy permit or change primary | `H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL` |
| `aug07_nonaccess` | zero event-row access | open one Aug07 row | `H0B_AUG07_ACCESS_FORBIDDEN` |
| `deterministic_build` | Build A/B primary and final research bytes exact | mutate Build B | `H0B_BUILD_MISMATCH` |
| `output_schema` | exact Section 26 CSV/JSON/report byte contracts | add/reorder field or report line | `H0B_OUTPUT_SCHEMA_MISMATCH` |
| `package_tree` | exact path/type universe | add extra/symlink | `H0B_PACKAGE_TREE_MISMATCH` |
| `layered_identity` | exact R/C/E bridge payloads and reverse binding | mutate bridge/reuse old C/E | `H0B_IDENTITY_BINDING_MISMATCH` |
| `manifest_self_exclusion` | only manifest excluded from E inventory/count/bytes | include/exclude another file | `H0B_MANIFEST_SELF_REFERENCE_MISMATCH` |
| `atomic_publication` | no overwrite, fsync then rename | precreate final | `PUBLICATION_FINAL_EXISTS` |
| `zero_external_action` | no network/private/order/cancel/live | attempt endpoint | `H0B_EXTERNAL_ACTION_FORBIDDEN` |

Every surface requires:

- authoritative source;
- decision/as-of time;
- exact field universe;
- rebuild oracle;
- stable failure code;
- current and frozen negative execution;
- durable evidence;
- identity-layer assignment.

## 29. Stable Failure Codes

The implementation must use exactly the following `61` surface codes:

```text
H0B_KERNEL_PIN_MISMATCH
H0B_MASTER_FRAMEWORK_MISMATCH
H0B_H0A_IDENTITY_MISMATCH
H0B_LATENCY_IDENTITY_MISMATCH
H0B_TUPLE_IDENTITY_MISMATCH
H0B_DEPENDENCY_IDENTITY_MISMATCH
H0B_SESSION_ROLE_MISMATCH
H0B_UNDERLYING_STATE_INFERENCE_FORBIDDEN
H0B_SEMANTIC_INVENTORY_MISMATCH
H0B_BUILD_ENVELOPE_MISMATCH
H0B_SOURCE_SCHEMA_MISMATCH
H0B_SOURCE_ORDERING_MISMATCH
H0B_FORBIDDEN_PATH_ACCESS
H0B_FEATURE_SOURCE_BOUNDARY_MISMATCH
H0B_OUTCOME_ACCESS_BEFORE_PERMIT
H0B_OUTCOME_PERMIT_MISMATCH
H0B_SUPPORT_COMMITMENT_MISMATCH
H0B_CALENDAR_GRID_MISMATCH
H0B_SIDE_PAIR_MISMATCH
H0B_EVENT_DEFINITION_MISMATCH
H0B_SUPPORT_CLASS_DISPOSITION_MISMATCH
H0B_OBSERVATION_BOUND_MISMATCH
H0B_INTERVAL_LIKELIHOOD_MISMATCH
H0B_RIGHT_CENSOR_MISMATCH
H0B_HORIZON_STRADDLE_MISMATCH
H0B_RISK_SCORE_MISMATCH
H0B_BINARY_SUBSET_MISMATCH
H0B_H0_FEATURE_ALLOWLIST_MISMATCH
H0B_H1_FEATURE_ALLOWLIST_MISMATCH
H0B_DESIGN_MATRIX_MISMATCH
H0B_BASIS_RESIDUAL_MISMATCH
H0B_MISSING_VALUE_POLICY_MISMATCH
H0B_DOSE_RECONSTRUCTION_MISMATCH
H0B_WALK_FORWARD_MISMATCH
H0B_ESTIMATOR_CONTRACT_MISMATCH
H0B_NUMERIC_CONVENTION_MISMATCH
H0B_RQ1_STATISTIC_MISMATCH
H0B_RQ1_NULL_MISMATCH
H0B_RQ2_SCORE_MISMATCH
H0B_RQ2_CONCENTRATION_MISMATCH
H0B_TIME_BOOTSTRAP_MISMATCH
H0B_FLOW_COMPONENT_MISMATCH
H0B_FLOW_BOOTSTRAP_MISMATCH
H0B_RQ3_THRESHOLD_MISMATCH
H0B_RQ3_REGIME_MISMATCH
H0B_RQ3_KM_MISMATCH
H0B_RQ3_BOOTSTRAP_MISMATCH
H0B_RQ3_SIDE_AGGREGATION_MISMATCH
H0B_PRIMARY_LATENCY_MISMATCH
H0B_CLASSIFICATION_MISMATCH
H0B_PRIMARY_SEAL_MISMATCH
H0B_STAGE4_PROJECTION_MISMATCH
H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL
H0B_AUG07_ACCESS_FORBIDDEN
H0B_BUILD_MISMATCH
H0B_OUTPUT_SCHEMA_MISMATCH
H0B_PACKAGE_TREE_MISMATCH
H0B_IDENTITY_BINDING_MISMATCH
H0B_MANIFEST_SELF_REFERENCE_MISMATCH
PUBLICATION_FINAL_EXISTS
H0B_EXTERNAL_ACTION_FORBIDDEN
```

Optimizer convergence failures are domain gate reasons serialized in output;
they do not replace any surface code. Unknown failures do not become accepted
generic errors.

## 30. Hostile Preflight

Before any formal outcome access, hostile preflight must execute every
Surface Matrix mutation on:

```text
current runtime
frozen runtime copy
```

Required targeted cases include:

1. one accepted H0-A identity mutation;
2. one accepted execution-latency identity mutation;
3. one accepted tuple identity mutation;
4. Aug03 formal promotion;
5. calendar-state inference injection;
6. one source-inventory addition and deletion;
7. one R0 header mutation;
8. one R1 decision-label open;
9. one forbidden Aug07 open;
10. one Stage 4 early open;
11. stale permit copied from another build root;
12. 1ns calendar-grid drift;
13. same-timestamp future-event inclusion;
14. ask/bid event predicate swap;
15. interval-only-to-binary coercion;
16. `(L,U]` to `[L,U]` mutation;
17. event midpoint point coercion;
18. full-horizon no-event encoded as binary-only;
19. horizon-straddle row deletion;
20. `F_5` risk score replaced by one-bin hazard;
21. full-session basis demeaning;
22. test-fold scaler leakage;
23. random row split;
24. post-hoc lambda change;
25. quantile method or PRNG substitution;
26. shock-time dose before confirmation;
27. row-level IID bootstrap;
28. signed residual passed directly to Kaplan-Meier;
29. `850ms` primary promotion;
30. `850ms` rescue of failed `6600ms`;
31. positive final-signal claim from H0-B;
32. Build B output mutation;
33. extra package path, symlink and special entry;
34. publication overwrite;
35. semantic inventory polluted with absolute build root;
36. Build A permit/envelope copied into Build B;
37. same-timestamp `event_seq` swap;
38. tied non-adverse row incorrectly advances `L` to `U`;
39. `(9ms,11ms]` rounded to whole `0-20ms` bins;
40. H0/H1 design column reorder or missing-indicator drop;
41. side/missing-indicator ridge penalty mask change;
42. RQ1 fixed disjoint microblock shuffle;
43. RQ2 cell contribution changed from positive-only to net/absolute;
44. Family A component endpoint assigned to two units;
45. flow bootstrap changed to equal-unit rather than row-weighted estimand;
46. RQ3 threshold derived from OOF or pooled-side predictions;
47. KM tie changed to censor-before-event or interpolated median;
48. Greenwood CI substituted for cluster-multiplier gate CI;
49. pooled regimes substituted for equal-side Bonferroni aggregation;
50. one primary allowlist path omitted from the seal;
51. one unprojected Stage 4 field read;
52. grid/segment overlap changed from grid-first, one legal support class
    rejected, one censor-source count omitted, or one conservation equation
    bypassed;
53. old C/E reused after Stage 4 diagnostic mutates R;
54. manifest included in its own E inventory or another file excluded;
55. one C/E bridge key, inventory ordering, count rule or portability value
    mutated while old identities are retained;
56. package report injected with E/composite or external business-report
    fields before E exists;
57. one package-owned matrix artifact omitted, assigned to the wrong layer,
    marked optional or moved under a similarly named package prefix.
58. V1 primary and V2 diagnostic identities collapsed into one plan pin;
59. Build A diagnostic permit copied into Build B or used before fsync;
60. one event total or rate changed while preserving the four-cell counts, or
    a zero-denominator ratio serialized as zero/NaN/infinity;
61. one output header/key reordered or extended.

Fail-open count must equal zero before formal Build A begins.

## 31. Execution Gates

### Gate 0: Plan, Task, Matrix And Kernel

- V1 primary plan SHA and V2 diagnostic plan SHA match their distinct task and
  matrix pins;
- independent V2 review SHA matches dispatch and final severity is
  `P0/P1/P2=0/0/0`;
- formal task and canonical matrix validate;
- accepted Kernel/H0-A/latency/tuple identities match;
- every surface has one distinct executed negative mutation;
- all six H0-B classification exits are defined;
- all seven canonical Stage exit criteria `EC1..EC7` are defined and validate.

### Gate 1: Focused Contract Tests

Fixtures must cover:

- every support class;
- every legal Stage 4 landmark support-class disposition;
- outside-grid plus Stage4-available;
- simultaneous grid-boundary and segment-boundary with grid-boundary
  precedence;
- all nine H0-B/Stage4 identified-event, identified-no-event and censored
  state combinations;
- union-censor de-duplication and every Stage 4 crosscheck conservation rule;
- event-total/four-cell/rate identities and the exact empty ratio cells when
  `eligible_count=0`;
- every event/no-event/straddle branch;
- ask and bid event definitions;
- strict-as-of equality;
- same-timestamp `event_seq` ordering and non-zero interval geometry;
- exact within-bin `S_exact(L)-S_exact(U)` likelihood;
- train/test leakage;
- exact design-matrix columns, transform order and penalty mask;
- optimizer determinism and failure;
- RQ1 null determinism;
- time/flow component assignment and multiplier estimands;
- regime entry/exit/censoring;
- KM tie order, median inversion and dependency-aware bootstrap;
- equal-side Bonferroni aggregation;
- latency role non-rescue;
- classification precedence.

### Gate 2: Pre-Outcome Replay And Permit

- rebuild H0-A support projection;
- match every accepted commitment;
- match the dispatch-pinned semantic source inventory;
- bind a distinct current-build envelope;
- freeze all contracts;
- write and fsync the outcome permit;
- prove zero outcome predicate evaluation before permit.

### Gate 3: Outcome Build A And Build B

- run in isolated roots;
- each root creates its own admitted permit;
- open only allowed public outcome sources;
- produce identical pre-diagnostic primary outputs and commitments;
- prove semantic source inventory identical and build envelopes distinct;
- prove zero external action.

### Gate 4: Primary Seal

- bind all RQ1/RQ2/RQ3 outputs;
- compute exact primary classification;
- bind the V1 primary-plan and V2 diagnostic-plan identities separately;
- fsync `primary_result_seal.json`;
- verify `stage4_crosscheck_opened=false`.

### Gate 5: Diagnostic Crosscheck

- create and fsync distinct Build A/B diagnostic permits bound to the common
  primary seal, V2 plan/review, current runtime and exact Stage 4 projection;
- in two fresh diagnostic roots, open only the eight accepted Jul30 Stage 4
  outcome paths and eleven projected fields after Gate 4;
- produce byte-identical aggregate diagnostics;
- verify every censor-source count and all conservation equations;
- publish the exact diagnostic from diagnostic Build A;
- prove primary bytes and classification unchanged.

### Gate 6: Package Admission And Archive

- exact tree/type/field universe;
- R/C/E/composite reverse binding;
- zero-write admission;
- atomic no-overwrite publication;
- amdserver durable archive with package-only admission;
- local/archive exact-tree parity.

Full source-semantic replay remains Mac-local unless a separately accepted
portable source bundle exists.

### Gate 7: Business Handoff

- external `.workflow/reports/0823T002-business.md` status is `待验收`;
- commit ID and message are exact;
- tuple and upstream identities are recorded;
- R/C/E/composite and package tree are recorded;
- the external report is absent from the package R/C/E inventories;
- no controller acceptance or final signal claim is made.

## 32. Independent Review Gate

Before dispatch, an independent reviewer must verify:

1. every accepted identity and source role;
2. the two-envelope outcome barrier;
3. event direction for both maker sides;
4. every support-class and likelihood branch;
5. bound inclusivity and five-bin likelihood math;
6. H0/H1 feature availability at decision time;
7. dose use only after accepted confirmation time;
8. walk-forward, purge, embargo and minimum folds;
9. estimator determinism and absence of model selection;
10. RQ1 stationary-null validity;
11. time and flow-aware bootstrap units;
12. RQ3 threshold, debounce, censoring and latency non-rescue;
13. H0-B screening classifications versus final framework claims;
14. exact output tree, identity layers and archive scope;
15. Surface Matrix completeness and stable-code uniqueness.

Any P0, P1 or unresolved P2 finding blocks formal dispatch.

## 33. Independent QA Gate

QA must use a fresh work root and:

1. verify candidate lineage and exact business commit;
2. verify reviewed plan/task/matrix/kernel pins;
3. independently rebuild H0-A support commitments before outcomes;
4. independently inspect the fsynced outcome permit;
5. execute all current/frozen hostile mutations;
6. run focused tests, Ruff, compileall and diff checks;
7. independently rebuild outcome Build A/B from accepted sources;
8. compare complete research output bytes and streaming commitments;
9. recompute event intervals and interval losses on independent fixtures and
   sampled full-source partitions;
10. recompute RQ1 null quantiles and bootstrap CIs from frozen seeds;
11. recompute RQ3 regimes and every latency role;
12. verify the primary classification precedence;
13. verify Stage 4 was opened only after the immutable primary seal;
14. verify zero Aug07/network/private/order/cancel/live access;
15. verify package admission, archive parity and atomic no-overwrite behavior.

QA may end only at:

```text
已通过
未通过
阻塞
```

QA acceptance does not authorize a strategy or main modeling stage. A later
controller closure must accept the exact package and decide the next task.

## 34. Controller Closure

After QA passes, controller closure must record:

- task status `已通过`;
- exact business and QA commits;
- reviewed plan and matrix identities;
- accepted upstream H0-A, latency and superseding-tuple identities;
- H0-B package R/C/E/composite;
- primary classification;
- RQ1 formal-session facts;
- H1/H0 formal-session normalized losses and CIs;
- `6600ms` RQ3 facts;
- `850ms` diagnostic facts and non-rescue status;
- Stage 4 diagnostic-open chronology;
- zero Aug07/external-action facts;
- whether main H0-H4 modeling, more sessions, execution-observability study
  or no further current-route work is eligible.

Controller closure does not modify H0-B package bytes.

## 35. Completion Definition

H0-B is complete only when:

- this plan passes independent review;
- a formal task and canonical Surface Matrix are dispatched;
- pre-outcome support replay and permit pass;
- hostile preflight has fail-open count zero;
- Build A and Build B are deterministic;
- the primary result is sealed before Stage 4 crosscheck;
- the exact small-output package passes Trust Kernel admission;
- independent QA returns `已通过`;
- controller closure accepts the exact package and next-state decision.

Until formal dispatch, no H0-B outcome access is authorized.

Until controller closure, no H0-B classification is accepted.
