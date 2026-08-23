# SKHYNIX Stage H0-B Conditional-Risk Audit Execution Plan

Date: 2026-08-23

Status: controller draft; independent review is required before dispatch.

Candidate formal task ID after review: `0823T002`.

This document does not dispatch `0823T002` and does not authorize outcome
access by itself.

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

### 8.2 H0B1: Outcome Runner

H0B1 is a fresh process. It may start only when:

```text
outcome_access_permit.status = admitted
outcome_access_permit.fsynced = true
support_replay_receipt.exact_commitment_match = true
accepted_input_identity_match = true
preoutcome_contract_sha256 = dispatch-pinned SHA256
source_inventory_sha256 = dispatch-pinned SHA256
```

The permit binds:

- task and plan identity;
- Surface Matrix identity;
- runtime source tree identity;
- accepted H0-A, latency and superseding-tuple identities;
- complete source inventory;
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

The permit is single-build-root specific. It cannot be copied from Build A to
Build B without rebinding the isolated build root and source inventory.

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

Only later target BBO receives with:

```text
receive_ts_ns > t
receive_ts_ns <= t + 50ms
same segment
same epoch
valid finite positive non-crossed BBO
```

may define an event.

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

For exact nanosecond bounds, define:

```text
left_bin =
  floor(max(0, L - t) / 10ms)

right_bin =
  min(5, ceil(max(0, U - t) / 10ms))
```

`left_bin` is in `0..4`; `right_bin` is in `1..5`.

Observed event likelihood:

```text
P(L < T <= U | x) =
  S_left_bin(x) - S_right_bin(x)
```

Full-horizon right-censor likelihood:

```text
P(T > t + 50ms | x) =
  S_5(x)
```

Horizon-straddling likelihood:

```text
P(T > L | x) =
  S_left_bin(x)
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

Invalid bin order, zero/negative likelihood before flooring, a bound outside
the allowed observation geometry or a branch/class mismatch fails closed.

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

For model fitting:

- each numeric feature gets an explicit missing indicator;
- the numeric value is filled with the training-fold median only;
- scaling uses the training-fold median and IQR only;
- zero or unavailable IQR is replaced by `1`;
- no test or future value influences imputation or scaling.

The package reports missing fractions by feature/session/fold.

If any H1 primary feature has more than `5%` missing rows in either formal
session, the H1 screen is:

```text
inconclusive_feature_availability
```

It cannot be treated as a failed or passed predictability gate.

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

The primary null uses `5s` moving blocks.

Each source `5s` microblock is labeled before outcome access by:

```text
session
segment
target-BBO update-count quartile
```

The update-count quartile edges are computed from support/cadence metadata
before outcome access.

For each of `2000` null replicates:

1. preserve every original microblock position and its segment/cadence
   stratum;
2. sample with replacement one source microblock from the same
   session/segment/cadence stratum;
3. copy the complete paired-side outcome path for that microblock;
4. reconstruct the absolute `60s` block rates;
5. compute `D_session`.

This preserves local overlap/dependence and accepted segment/cadence
structure while removing the observed minute-scale ordering.

Frozen seeds:

```text
primary_null_seed = 8232001
time_bootstrap_seed = 8232002
flow_bootstrap_seed = 8232003
```

Robustness nulls use `2.5s` and `10s` moving blocks with the same strata and
seed derivation. They are secondary and cannot replace the `5s` primary.

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
positive_cell_contribution_c =
  max(0, sum_{i in c}(improvement_i))
cell_share_c =
  positive_cell_contribution_c /
  sum_c(positive_cell_contribution_c)
```

The denominator must be positive. Cells and their edges are frozen from past
training blocks, and the maximum cell share is computed per formal session
with bid/ask cell contributions equal-weighted before the `50%` check.

### 21.1 Time-Block Bootstrap

The time bootstrap resamples complete `60s` OOF test blocks within session.
Predictions and fold assignments remain fixed; models are not refit.

Use `2000` replicates and `time_bootstrap_seed`.

### 21.2 Flow-Aware Bootstrap

Accepted Stage 2 `overlap_block_id` defines event-overlap components.

Each OOF grid row is assigned to:

- the accepted overlap block whose frozen outcome window contains `t`; or
- a `2s` absolute-time background block when no accepted overlap block
  contains `t`.

Assignments must be mutually exclusive. Ambiguity fails closed.

The flow-aware bootstrap resamples these complete units within session,
preserving paired sides and all rows in a unit. Use `2000` replicates and
`flow_bootstrap_seed`.

## 22. RQ3: Regime Dwell And Latency

RQ3 uses only H1 OOF predictions from valid test folds.

At each fold, thresholds are fit from the past training predictions:

```text
entry_threshold = training risk_score_50ms q90
exit_threshold = training risk_score_50ms q70
entry_debounce = 3 consecutive 10ms endpoints
exit_debounce = 5 consecutive 10ms endpoints
```

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
  the
  log-log Greenwood interval;
- switching rate per minute.

Every quantity is first computed per side. Session-level identified fraction
is the equal-weight mean of the two side fractions. The session residual-p50
lower bound is the equal-weight mean of the two one-sided `95%` side lower
bounds. By the two-side Bonferroni construction, this is the frozen one-sided
`90%` lower bound for the equal-weight mean of side-specific residual medians.
Pooled regime rows may not replace these session scores. A missing or
non-identifiable side value makes the formal session inconclusive.

### 22.1 Gate H-C Screen

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

The H0-B primary package must first seal:

```text
primary_results_sha256
primary_classification_sha256
stage4_crosscheck_opened = false
```

Only then may a fresh diagnostic process open the accepted Jul30 Stage 4:

```text
outcomes/
```

for an aggregate landmark crosscheck of:

- event-direction parity near accepted trigger candidates;
- broad adverse-rate direction;
- no point-coercion disagreement;
- no quote-risk naming disagreement.

Stage 4 `features/` and `views/` remain forbidden unless independent review
adds exact diagnostic paths before dispatch.

The crosscheck:

- is Jul30-only;
- is diagnostic;
- cannot modify models, thresholds, gates or primary classification;
- cannot repair a failed Build A/Build B primary comparison.

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
outcome_access_ledger.json
outcome_access_permit.json
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
support_replay_receipt.json
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
contracts/*
runtime_source/*
runtime_tests/*
```

Evidence identity `E` includes:

```text
accepted_input_bindings.json
h0b_manifest.json
outcome_access_ledger.json
outcome_access_permit.json
preoutcome_source_inventory.csv
primary_result_seal.json
support_replay_receipt.json
reports/h0b_conditional_risk_audit.md
```

Build-root absolute paths are evidence fields but are excluded from semantic
research identity through the accepted Trust Kernel normalization contract.

## 28. Load-Bearing Surface Matrix

The formal Surface Matrix must contain at least these surfaces:

| Surface | Exact contract | Required negative mutation |
| --- | --- | --- |
| `kernel_pin` | accepted Trust Kernel v1 exact | alter registry/kernel pin |
| `master_framework_pin` | exact accepted v2 framework | alter framework SHA |
| `accepted_h0a_binding` | exact H0-A tuple/package/QA/closure | alter one identity |
| `accepted_latency_binding` | exact 0822T002 package/QA/closure | alter one identity |
| `accepted_tuple_binding` | exact 0823T001 tuple/package/QA/closure | alter one identity |
| `accepted_stage1_4_binding` | exact inherited dependency set | substitute package |
| `session_roles` | Jul30/Aug04 formal; Aug03 diagnostic | promote Aug03 |
| `underlying_state_boundary` | unknown; no calendar inference | inject KRX state |
| `source_inventory` | exact accepted source universe | add/remove source |
| `source_schema` | exact two R0 header identities | alter one header |
| `guarded_opener` | reject before forbidden read | open Aug07/Stage4 early |
| `feature_source_boundary` | R0 BBO/bookTicker only | open R1 future labels |
| `two_envelope_boundary` | H0B0 then fresh H0B1 | reuse process |
| `outcome_access_permit` | exact fsynced build-root binding | stale/copy permit |
| `support_replay` | exact H0-A commitments | alter one commitment |
| `calendar_grid` | exact 10ms absolute grid | drift origin by 1ns |
| `side_expansion` | exactly paired bid/ask rows | drop one side |
| `event_definition` | opposing BBO crosses vulnerable quote | use midpoint/retreat |
| `support_class_mapping` | exact nine-class disposition | coerce interval-only |
| `observation_bounds` | `(L,U]` exact | change inclusivity |
| `interval_likelihood` | exact five-bin formulas | point-coerce event |
| `right_censor_likelihood` | exact `S_5` | encode no-event as zero-time |
| `horizon_straddle` | exact `S_left` | drop straddle row |
| `risk_score` | exact `F_5=1-S_5` | use one-bin hazard/other score |
| `binary_subset` | binary class only | include interval-only |
| `h0_features` | exact context allowlist | add future/calendar feature |
| `h1_features` | exact cross-spread allowlist | add outcome/post-t feature |
| `basis_residual` | prior-only 60s EWMA | full-session demean |
| `missing_value_policy` | train-only median/IQR + indicator | use test median |
| `dose_definition` | exact accepted Stage 3 trailing dose | use shock time before confirm |
| `walk_forward` | exact 60/20 blocks, purge/embargo | random split |
| `estimator` | exact five-bin ridge logistic | tune lambda after outcome |
| `numeric_conventions` | nearest-rank/PCG64/tie rules exact | change quantile/RNG |
| `rq1_statistic` | exact side/session variance | pool side rows |
| `rq1_null` | exact stratified 5s null | unstratified IID shuffle |
| `rq2_score` | exact H1/H0 interval loss | choose favorable metric |
| `time_bootstrap` | exact 60s OOF block units | row bootstrap |
| `flow_bootstrap` | overlap/background units exact | duplicate/drop row |
| `rq3_regime` | q90/q70 and 3/5 debounce | post-hoc threshold |
| `rq3_survival` | KM on total dwell then latency shift | KM signed residual |
| `latency_roles` | 6600 primary, 850 diagnostic | promote/rescue with 850 |
| `classification_precedence` | exact allowed exits | issue final signal claim |
| `stage4_crosscheck` | after primary seal, diagnostic only | open before seal |
| `aug07_nonaccess` | zero event-row access | open one Aug07 row |
| `deterministic_build` | Build A/B research bytes exact | mutate Build B |
| `package_tree` | exact path/type universe | add extra/symlink |
| `layered_identity` | exact R/C/E reverse binding | stale manifest binding |
| `atomic_publication` | no overwrite, fsync then rename | precreate final |
| `zero_external_action` | no network/private/order/cancel/live | attempt endpoint |

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

The implementation must use exact stable codes including:

```text
H0B_KERNEL_PIN_MISMATCH
H0B_MASTER_FRAMEWORK_MISMATCH
H0B_H0A_IDENTITY_MISMATCH
H0B_LATENCY_IDENTITY_MISMATCH
H0B_TUPLE_IDENTITY_MISMATCH
H0B_DEPENDENCY_IDENTITY_MISMATCH
H0B_SESSION_ROLE_MISMATCH
H0B_UNDERLYING_STATE_INFERENCE_FORBIDDEN
H0B_SOURCE_INVENTORY_MISMATCH
H0B_SOURCE_SCHEMA_MISMATCH
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
H0B_FEATURE_ALLOWLIST_MISMATCH
H0B_FUTURE_FEATURE_ACCESS
H0B_BASIS_RESIDUAL_MISMATCH
H0B_MISSING_VALUE_POLICY_MISMATCH
H0B_DOSE_RECONSTRUCTION_MISMATCH
H0B_WALK_FORWARD_MISMATCH
H0B_ESTIMATOR_CONTRACT_MISMATCH
H0B_OPTIMIZER_FAILURE
H0B_NUMERIC_CONVENTION_MISMATCH
H0B_RQ1_NULL_MISMATCH
H0B_BOOTSTRAP_UNIT_MISMATCH
H0B_RQ3_REGIME_MISMATCH
H0B_RQ3_SURVIVAL_MISMATCH
H0B_PRIMARY_LATENCY_MISMATCH
H0B_DIAGNOSTIC_RESCUE_FORBIDDEN
H0B_CLASSIFICATION_MISMATCH
H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL
H0B_AUG07_ACCESS_FORBIDDEN
H0B_BUILD_MISMATCH
H0B_PACKAGE_TREE_MISMATCH
H0B_IDENTITY_BINDING_MISMATCH
H0B_EXTERNAL_ACTION_FORBIDDEN
PUBLICATION_FINAL_EXISTS
```

Unknown failures do not become accepted generic errors.

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
34. publication overwrite.

Fail-open count must equal zero before formal Build A begins.

## 31. Execution Gates

### Gate 0: Plan, Task, Matrix And Kernel

- reviewed plan SHA matches dispatch;
- independent review final severity is `P0/P1/P2=0/0/0`;
- formal task and canonical matrix validate;
- accepted Kernel/H0-A/latency/tuple identities match;
- every surface has one distinct executed negative mutation;
- all seven exit criteria are defined.

### Gate 1: Focused Contract Tests

Fixtures must cover:

- every support class;
- every event/no-event/straddle branch;
- ask and bid event definitions;
- strict-as-of equality;
- train/test leakage;
- optimizer determinism and failure;
- RQ1 null determinism;
- bootstrap unit assignment;
- regime entry/exit/censoring;
- latency role non-rescue;
- classification precedence.

### Gate 2: Pre-Outcome Replay And Permit

- rebuild H0-A support projection;
- match every accepted commitment;
- freeze source inventory and all contracts;
- write and fsync the outcome permit;
- prove zero outcome predicate evaluation before permit.

### Gate 3: Outcome Build A And Build B

- run in isolated roots;
- each root creates its own admitted permit;
- open only allowed public outcome sources;
- produce identical research outputs and commitments;
- prove source inventory unchanged;
- prove zero external action.

### Gate 4: Primary Seal

- bind all RQ1/RQ2/RQ3 outputs;
- compute exact primary classification;
- fsync `primary_result_seal.json`;
- verify `stage4_crosscheck_opened=false`.

### Gate 5: Diagnostic Crosscheck

- open accepted Jul30 Stage 4 outcome paths only after Gate 4;
- publish aggregate diagnostic;
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

- business report status is `待验收`;
- commit ID and message are exact;
- tuple and upstream identities are recorded;
- R/C/E/composite and package tree are recorded;
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
