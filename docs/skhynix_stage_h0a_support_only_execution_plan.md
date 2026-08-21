# SKHYNIX Stage H0-A Support-Only Execution Plan

Date: 2026-08-21

Revision: review draft 1

Status: controller-authored review draft. This document expands Stage H0-A
from `docs/skhynix_continuous_hazard_maker_research_framework_v2.md` into an
independent execution contract. It does not dispatch a task, authorize a data
scan, create a Surface Matrix, build a package, open an outcome aggregate, or
unlock Stage H0-B.

## 0. Review And Authority

The active authority chain is:

```text
user-approved v2 master framework
-> this H0-A execution-plan review
-> controller remediation of review findings
-> separate formal task ID and canonical Surface Matrix
-> Stage H0-A business execution
-> independent QA
-> controller decision on Stage H0-B
```

The formal H0-A task ID is intentionally unassigned during plan review.
Creating `.workflow/tasks/<TASK_ID>.md` or
`.workflow/contracts/<TASK_ID>-surface-matrix.json` before this review closes
would prematurely convert design choices into execution authority.

If this plan conflicts with the v2 master framework before review closure, the
v2 master framework controls and the conflict must be resolved explicitly.
After user approval, the controller must synchronize any accepted
clarifications back into the v2 master framework before dispatch.

## 1. Purpose

Stage H0-A answers one support question:

```text
Can the accepted Jul30/Aug03/Aug04 public data support a fixed 10ms
calendar-time conditional-risk study at one mechanically selected horizon
from 50/100/250/500ms, without using adverse-event outcomes to choose it?
```

H0-A has three deliverables:

1. Rebuild calendar-grid coverage, source cadence, censoring, endpoint
   identification and complete dependence-block support.
2. Select the first eligible primary horizon using only the frozen support
   rules.
3. Publish an immutable primary-tuple freeze that H0-B can consume only after
   independent QA.

H0-A is an engineering and data-support audit. It does not answer whether
adverse risk is high, time-varying, predictable, actionable or profitable.

## 2. Frozen Non-Goals

H0-A must not:

- publish adverse-event rates, conditioned rates, markouts, loss, effect
  sizes, model scores, calibration or risk regimes;
- compare a quote at `t` with a quote after `t` to derive an adverse label;
- fit, select or tune an estimator, model, threshold, bin edge or policy;
- run H0-B, RQ1, RQ2, RQ3, H0-H4 feature ablations or actionability work;
- open any Aug07 event row;
- read Stage 4 `outcomes/`, `features/` or `views/`;
- reinterpret public-market contact as a real fill;
- access a private endpoint, own order, cancel, inventory, fee or PnL field;
- collect new data, use network access or authorize live behavior;
- mutate accepted Stage 1-4 packages or their external source data;
- change the frozen queue-shock detector;
- select a horizon from 1000ms or 2000ms;
- use Aug03 to rescue a failed two-formal-session gate;
- publish a row-level 10ms research data plane.

## 3. Preconditions And Accepted Pins

### 3.1 Trust Kernel v1

The formal task and canonical Surface Matrix must use `mode=accepted` with
this exact schema-valid pin:

```text
kernel_name = research_package_trust_kernel
kernel_version = v1
registry_path = baselines/research_package_trust_kernel/accepted_versions.json
registry_entry_sha256 = cae21d65bf447435bafc37508b8ca00643a0742b37e0f404148cab92818c90c9
kernel_source_tree_sha256 = cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203
kernel_api_contract_sha256 = 2cd5a67ba15d67e59bcddcdbb21593696d3dc3dc27d81986c39ddf3e91e91f5f
kernel_negative_matrix_sha256 = f6247594b6f024945a52c0dccf421bac93538d9357f92ff6027d024199fc6b97
kernel_qa_report_sha256 = 8fe01f85f8a68581b79ee410167769f2a105d9cc74ca6528af9496808a626be8
kernel_acceptance_task_id = 0820T001
```

The registry itself has the separate exact precondition
`registry_revision=1`. `registry_revision` is not inserted into the
Surface Matrix `kernel_pin` object because the frozen schema does not permit
that extra field.

Before the first H0-A research-source row is read, preflight must validate:

1. the registry schema and exact revision;
2. the unique accepted v1 entry and its raw entry identity;
3. the exact acceptance package;
4. the accepted kernel source tree;
5. the task/Markdown/Surface Matrix pin parity;
6. all seven Surface Matrix exit criteria.

Any mismatch stops before research input access.

### 3.2 Accepted Package Dependencies

| Dependency | Role | Core identity | Full identity | Contract / manifest identity |
| --- | --- | --- | --- | --- |
| Stage 1, `0814T001` | authoritative source admission, session topology, source inventory and cadence prior | `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96` | `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590` | contract `8657c6a81cb541c8df1c7696f86b3b4cbfb0c2e01d0bb99a98bbecd158041a6e`; manifest raw SHA `9d8bf64c1a95ea378e88fe8d23243ce2896d727661988402c4e5dfd8600d55c2` |
| Stage 2, `0815T001` | evidence labels, formal eligibility and dependence prior | `7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8` | `bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833` | contract `0cd8b44ce2e39aebe8e8039f1dab0cf4d74dd555a59378e0a5acd33ca3b83815`; manifest raw SHA `ac5f4e50cfb653dffa6e6d6fb84ab451da70da415fd3d2785fff52330b9cd43b` |
| Stage 3, `0815T002` | transitive detector/anchor authority bound by Stage 4; no direct H0-A feature generation | `4939d1c1addce493edb2f368297d56b37edd0b123de01497dcdee6e77637eb9b` | `ff8e3434672226371051151cea838503877dca79ac7860cf179256362d75e404` | contract `a894079e405073400d86f8471fd20e2a3a7116d3b804952bab8db56587f93b3d`; manifest raw SHA `a73d4a15a6f58bd5c533944131a9cadae4e74d1c6f0c23a65d75c35973282fa1` |
| Stage 4, `0815T003` | Jul30 landmark, source-catalog and censoring cross-check only | `78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157` | `669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433` | contract `b80b9bae2d6cf18cb6e7be4f133467138527fec20eef7054ab38b7f8c70cefde`; manifest raw SHA `2c802336c1446eaf50eaf5ef546b115046abe3a1b69fdd1fcaea31d22a8e64a6` |

Stage 3 is a transitive accepted dependency. H0-A does not rebuild trigger
candidates or use detector outputs to select the primary horizon.

### 3.3 Authoritative Historical Sessions

The task must bind exact files through Stage 1 `input_inventory.csv` and
`data_admission/input_manifest_bindings.csv`. The current authoritative
session roots are:

| Session | Evidence role | Formal gate eligible | Raw campaign manifest SHA | Accepted R0 manifest SHA | Accepted R1 alignment manifest SHA |
| --- | --- | --- | --- | --- | --- |
| Jul30 | `historical_discovery` plus internal-validation segments | yes | `fc4d962bd96792b87671a35974601827864267f7a3d3181ff8237b050b497b12` | `c46c735d7933587af6eece4a9dd1bce241b3c093866efce976b1c3f952e72ce0` | `6123b408cbdfdc95758c8d27e6e0664959caaebeb0de843a5978cfee6c6c8ffd` |
| Aug03 | `historical_transfer` diagnostic | no | `c9d7f65ed85157838f55a41d56745e76e8cee845fcd20d066cd0f81ff61f5fe9` | `4ca8b2e0be7f989eabf11294d9a60da491c0f3fecd8124e9bf097bf1beff5e7a` | `1c4119daa4aa1f2575e4184e11058c81b626a91c3e1f1497f6424cf00fe752a3` |
| Aug04 | `historical_consumed_validation` | yes | `1853924b91ad9751239387c303a18c7906510c63d6c7449b28c59668d377885b` | `ed47cd97b55f5305b0527ac93e6b65d2344ef453b18e7f510f7c7c4e88bcd0cd` | `9d14558be47731fa274b2ede853a0f36ac27d3a0640da8ef9de56c136fcca221` |

The accepted Aug03 R1 authority is the `0804T001_old5h_replay` alignment
package. The failing or superseded alignment under the original Aug03
research root is not authoritative. Selecting it must fail with
`H0A_NONAUTHORITATIVE_AUG03_R1`.

All three sessions may appear in support diagnostics. Only Jul30 and Aug04
count toward the requirement that a primary horizon pass in at least two
formal sessions. Aug03 cannot rescue either formal session.

### 3.4 Authoritative Source Plane

The accepted source plane has distinct roles:

| Source | H0-A role |
| --- | --- |
| per-segment R0 `hyperliquid_hot_events.csv.gz` | authoritative target BBO event sequence, receive timestamps and quote-state validity |
| per-segment R0 `binance_hot_events.csv.gz` | cadence diagnostic only |
| raw/common timeline manifests | exact segment bounds, source inventory and schema authority |
| `common_l2_timeline.csv.gz` | strict-as-of/cadence cross-check for Binance and Hyperliquid fast/standard L2; not the target BBO event oracle |
| accepted R1 alignment package | accepted segment quality, reconciliation and censoring metadata |

The target `public_bbo_moves_through_quote` support surface must use
`event_type=bbo` rows from accepted R0 `hyperliquid_hot_events.csv.gz`.
`common_l2_timeline.csv.gz` does not contain the standalone Hyperliquid BBO
event sequence and cannot replace that oracle.

The exact per-segment paths, bytes and SHA256 values are not rediscovered by
glob. They are selected from the accepted Stage 1 inventory and then checked
against the live canonical files before and after execution.

## 4. Execution Architecture

H0-A is split into two ordered processes with different read authority.

```text
H0-A0 support projector
  accepted public source rows + accepted metadata
  -> support-only projection, aggregates and commitments

H0-A1 selector/freezer
  sealed support-only projection only
  -> mechanical horizon selection + immutable primary tuple
```

### 4.1 H0-A0 Support Projector

H0-A0 may:

- scan Jul30/Aug03/Aug04 accepted R0/R1 public source rows;
- decode timestamps, event/channel identity, segment/epoch identity, quality
  state, source timestamps and quote-state validity;
- decode individual public BBO price fields only to validate that each quote
  is finite, positive and has `bid <= ask`;
- derive availability, source age, no-new-information, censoring and support
  classifications;
- aggregate those support classifications and emit canonical commitments.

H0-A0 must not:

- pair price values across time to compute direction, movement or markout;
- compare reference and future quote prices;
- construct `public_bbo_moves_through_quote` truth values;
- count adverse events or emit any price value;
- read Stage 4 outcome-bearing surfaces;
- expose raw source rows to H0-A1.

The projector output contains no bid, ask, mid, trade price, return, markout,
adverse flag or label.

### 4.2 H0-A1 Selector And Freezer

H0-A1 runs in a fresh process whose allowed input root contains only the
support projection and frozen selection contract. It receives no raw source,
Stage 4 or accepted package path argument.

Its path guard must reject reads outside that sealed root, except for:

- the accepted Trust Kernel v1 package and registry;
- its own frozen runtime source;
- the task and canonical Surface Matrix needed for admission.

H0-A1:

1. validates the support projection identity and exact schema;
2. computes each session/horizon gate from frozen fields;
3. evaluates horizons in exact order `50, 100, 250, 500`;
4. selects the first horizon passing both Jul30 and Aug04;
5. publishes the full primary tuple or an explicit inconclusive result.

H0-A1 cannot access a value from which adverse direction, rate, effect or loss
could be reconstructed.

### 4.3 Process Boundary Evidence

The final package must record:

```text
projector_raw_public_rows_opened = true
projector_price_values_emitted = false
projector_cross_time_price_comparison_count = 0
projector_adverse_label_count = 0
selector_raw_public_rows_opened = false
selector_stage4_outcome_paths_opened = false
selector_forbidden_field_access_count = 0
aug07_event_rows_opened = false
aug07_event_row_read_count = 0
```

These fields are evidence claims and require runtime enforcement plus hostile
tests. They are not accepted as self-attestation alone.

## 5. Read Surface Contract

### 5.1 Stage 1

Allowed:

- `research_manifest.json`;
- `frozen_research_contract.json`;
- `input_inventory.csv`;
- `data_admission/input_manifest_bindings.csv`;
- `data_admission/session_topology.csv`;
- `data_admission/channel_inventory.csv`;
- `data_admission/hyperliquid_feed_cadence.csv`;
- `data_admission/unavailable_fields.csv`;
- `data_admission/underlying_regime_coverage.csv`;
- `consumption_ledgers/aug07_access_ledger.json`.

The Aug07 ledger may be read as boundary evidence. No path referenced by an
Aug07 event-row binding may be opened.

Stage 1 cadence is a prior and cross-check. H0-A must recompute the exact
cadence surface used by its support decision from the frozen Jul30/Aug03/
Aug04 input inventory.

### 5.2 Stage 2

Allowed:

- `density_manifest.json`;
- `frozen_density_contract.json`;
- `input_bindings.csv`;
- `trigger_density_by_session.csv`;
- `inter_trigger_distribution.csv`;
- `episode_merging_summary.csv`;
- `trigger_density_sensitivity_summary.csv`;
- `effective_sample_size.csv`;
- `reports/trigger_density_admission.md`.

The two large membership files are not needed for primary horizon selection.
If implementation review later proves one is necessary for a named
cross-check, it must be added to the task and Surface Matrix before execution.

Stage 2 values are dependence priors and trigger-dose cross-checks. They do
not replace H0-A calendar-grid support counts and do not establish adverse
risk.

### 5.3 Stage 4

Allowed:

- `episode_v3_manifest.json`;
- `frozen_episode_v3_contract.json`;
- `input_bindings.csv`;
- `source_event_store_catalog.csv`;
- `quality_intervals.csv`;
- `segment_summary.csv`;
- `anchors/segment_0001.csv.gz` through
  `anchors/segment_0008.csv.gz`.

Forbidden:

- every path under `outcomes/`;
- every path under `features/`;
- every path under `views/`;
- any future-derived value copied from those surfaces;
- the Stage 4 report if implementation would parse outcome summaries from it.

Within the allowed files, Stage 4 access is field-projected:

- `frozen_episode_v3_contract.json` is verified by raw SHA256 and is not
  semantically loaded into H0-A;
- `episode_v3_manifest.json` permits only package/contract/input identities,
  artifact records for allowed paths, boundary flags, `source_catalog_rows`,
  `quality_interval_rows` and `anchor_rows`;
- `segment_summary.csv` permits only `segment_id`, `evidence_label`,
  `candidate_rows`, `anchor_rows`, `auxiliary_degraded_grid_rows` and
  `core_degraded_grid_rows`;
- anchor files permit only `session_id`, `segment_id`, `candidate_id`,
  `shock_cluster_id`, `flow_cluster_id`, `overlap_block_id`, `candidate_seq`,
  `t_candidate_ns`, `t_confirm_ns`, `confirmed`, `classification`,
  `connection_epoch_id`, `segment_start_ts_ns`, `segment_end_ts_ns`,
  `evidence_label` and `source_manifest_sha256`.

All Stage 4 outcome statuses, outcome counts, event-time intervals, markouts,
public-risk flags and price values are opaque skipped fields.

The Stage 4 cross-check is limited to:

1. Jul30 source path/size/SHA agreement;
2. segment and connection-epoch boundaries;
3. quality/censoring interval agreement;
4. exact anchor count, candidate identity and ordering;
5. deterministic mapping of each anchor timestamp to its containing 10ms
   calendar cell;
6. confirmation that landmark mapping does not enter horizon selection.

### 5.4 Underlying Market State

Accepted inputs do not contain an authoritative KRX calendar-state feed.
H0-A must preserve:

```text
underlying_market_state = unknown_calendar_state
future_calendar_inference = false
```

Nominal exchange clock labels cannot make a grid row eligible or ineligible
and cannot select the horizon.

### 5.5 Source Row Field Access

For target support, the allowed R0 Hyperliquid hot-event fields are:

```text
segment_id
event_seq
source_raw_seq
source_item_index
local_ts_ns
exchange_ts_ns
event_type
coin
bid_px
ask_px
```

`bid_px` and `ask_px` are parsed only on `event_type=bbo` rows for individual
quote validity. Quantity/count fields are not required to identify the
default target and remain unopened values.

For cadence-only channels, H0-A may read only event identity, channel/type and
timestamp fields. Trade price, trade quantity, user, hash, side and all
depth-price/quantity values are forbidden.

For `common_l2_timeline.csv.gz`, H0-A may read:

```text
campaign_id
segment_id
profile_id
common_seq
common_ts_ns
trigger_track
trigger_kind
trigger_raw_seq
binance_local_ts_ns
binance_exchange_ts_ns
binance_age_ms
hyperliquid_fast_local_ts_ns
hyperliquid_fast_exchange_ts_ns
hyperliquid_fast_age_ms
hyperliquid_standard_local_ts_ns
hyperliquid_standard_exchange_ts_ns
hyperliquid_standard_age_ms
```

All common-timeline price, quantity and order-count cells are opaque skipped
tokens in H0-A. Unknown, duplicate, missing or reordered columns fail against
the frozen full-header schema even when the values are not selected.

## 6. Calendar Grid Contract

### 6.1 Clock And Alignment

The primary grid uses integer nanoseconds on the local collector receive-time
clock:

```text
grid_step_ns = 10_000_000
grid_origin_ns = 0
grid_ts_ns % grid_step_ns = 0
```

For each accepted structural segment and connection epoch, the nominal grid
covers the complete accepted half-open segment/epoch interval, including
opening or trailing periods where source state may be unavailable:

```text
first_grid_ts_ns = ceil(segment_epoch_start_ns / grid_step_ns)
                   * grid_step_ns
last_grid_ts_ns  = floor((segment_epoch_end_ns - 1) / grid_step_ns)
                   * grid_step_ns
```

Grid generation is per segment and epoch. No grid interval or endpoint may
cross a segment or connection-epoch boundary. The first supported quote does
not define the grid start; otherwise missing opening coverage would disappear
from the denominator.

The grid is calendar-time weighted:

- each eligible start contributes exactly 10ms exposure;
- multiple messages inside one cell do not increase exposure;
- a cell with no new message remains a calendar cell if strict-as-of state is
  valid and no explicit source/connection/quality gap has begun;
- row count is never interpreted as independent sample size.

### 6.2 Strict-As-Of State

At grid time `t`, an observation is visible only when:

```text
source_receive_ts_ns <= t
same structural segment
same connection epoch
no crossed core-quality or unavailable interval
source record passes exact schema and quote-validity checks
```

Every forward-carried state must retain:

```text
source_receive_ts_ns
source_event_ts_ns when available
source_age_ns
connection_epoch_id
quality_mask
no_new_information
```

No state may be carried across a segment, epoch, core-quality gap or source
unavailable interval.

For the target BBO stream, silence inside a connected, quality-eligible epoch
is `no_new_information`, not automatic staleness. H0-A applies no arbitrary
message-age timeout to the target quote. Only an explicit source gap,
connection boundary, quality interval, missing initial quote or invalid quote
state removes support.

### 6.3 Horizon Set

Primary candidates:

```text
50ms
100ms
250ms
500ms
```

Descriptive only:

```text
1000ms
2000ms
```

All six horizons use the same grid and support classifier. The descriptive
horizons must be emitted with `primary_selection_eligible=false`.

## 7. Support And Censoring Semantics

### 7.0 Primary Support Channel

The primary support gate is defined on the target-venue Hyperliquid public
BBO stream:

```text
target_venue = hyperliquid
target_channel = bbo
reference_state = strict_asof_target_bbo_at_grid_t
at_risk_support = target_bbo_observation_support_over_(t,t+h]
```

Binance BBO/trade/depth cadence, Hyperliquid trade cadence, fast/standard L2
cadence and Stage 2 trigger structure are diagnostics for later feature and
dependence work. They are published but cannot make a primary horizon pass or
fail. H0-B must preserve the H0-A horizon if a later feature is unavailable.

### 7.1 Per-Grid Support Facts

For every conceptual `(session, segment, epoch, grid_ts, horizon)` row, H0-A0
derives only these support facts:

```text
grid_inside_segment
reference_quote_available
reference_quote_valid
target_ts_inside_segment
target_ts_same_epoch
core_quality_eligible
target_feed_observation_supported
source_gap_intersects_at_risk_interval
interval_bounds_supported
binary_endpoint_identification_supported
interval_likelihood_eligible
identification_class
complete_60s_block_id
```

The row is not published. Its canonical support tuple contributes to a
per-segment/per-horizon streaming commitment.

### 7.2 Identification-Support Classes

The exact mutually exclusive `identification_class` enum is:

```text
binary_identification_supported
interval_likelihood_only_supported
right_censored_segment
right_censored_source_end
epoch_censored
core_quality_censored
source_gap_censored
reference_quote_unavailable
invalid_quote_state
```

`binary_identification_supported` means:

- a valid strict-as-of target BBO exists at `t`;
- the complete at-risk interval `(t, t+h]` remains in one segment and epoch;
- no core-quality or source-unavailable gap intersects the interval;
- the target BBO feed supplies enough ordered observations to decide the
  event later;
- no adverse-event truth value has been computed.

`interval_likelihood_only_supported` means the observation geometry cannot
guarantee exact binary identification at the horizon boundary, but valid
lower/upper observation bounds exist for later interval likelihood. It does
not assert that an adverse event occurred or that an actual outcome was
ambiguous. It is not coerced to binary support.

The remaining classes are not interval-likelihood eligible.

### 7.3 Fractions And Denominators

The count states are frozen as:

```text
nominal_calendar_grid_count =
    every absolute 10ms start in the accepted segment/epoch intervals

quality_eligible_grid_count =
    nominal starts with no core-quality, explicit connection or source-gap
    exclusion active at t

fully_identified_binary_count =
    rows classified binary_identification_supported

interval_likelihood_eligible_count =
    rows classified binary_identification_supported or
    interval_likelihood_only_supported
```

Quality eligibility does not require an initial quote or a supported future
endpoint. Those failures remain visible in the binary/interval fractions
instead of being removed from their denominators.

For each session and horizon:

```text
nominal_calendar_grid_count
quality_eligible_grid_count
fully_identified_binary_count
interval_likelihood_eligible_count
complete_60s_block_count
```

The frozen fractions are:

```text
quality_eligible_calendar_exposure_fraction =
    quality_eligible_grid_count / nominal_calendar_grid_count

fully_identified_binary_endpoint_fraction =
    fully_identified_binary_count / quality_eligible_grid_count

interval_likelihood_eligible_fraction =
    interval_likelihood_eligible_count / quality_eligible_grid_count
```

Zero denominators are explicit failures. Fractions are computed from integer
counts using exact decimal serialization; binary floating-point text is not
an oracle.

### 7.4 Complete 60-Second Blocks

Primary blocks use absolute receive-time anchoring:

```text
block_ns = 60_000_000_000
block_start_ns = floor(grid_ts_ns / block_ns) * block_ns
```

A session/horizon block is complete only when:

1. the full half-open calendar interval
   `[block_start_ns, block_start_ns + 60s)` lies inside one accepted segment
   and one connection epoch;
2. all 6000 expected 10ms grid starts exist;
3. every start is quality eligible;
4. every start plus the candidate horizon remains inside the same segment and
   epoch;
5. no core-quality or source gap invalidates the at-risk support.

Block completeness does not depend on trigger presence, adverse-event
presence or outcome class. Stage 2 occupied/overlap blocks are reported only
as diagnostics.

## 8. Mechanical Horizon Selection

### 8.1 Per-Session Gate

A primary candidate horizon passes a formal session only when all conditions
hold:

```text
quality_eligible_calendar_exposure_fraction >= 0.95
fully_identified_binary_endpoint_fraction >= 0.90
interval_likelihood_eligible_fraction >= 0.95
complete_60s_block_count >= 20
```

### 8.2 Cross-Session Gate

The formal session set is frozen to:

```text
jul30
aug04
```

Because there are exactly two formal-eligible historical sessions, both must
pass. Aug03 is always emitted as diagnostic and never increments the formal
pass count.

### 8.3 Selection Algorithm

```python
selected_horizon_ms = None

for horizon_ms in (50, 100, 250, 500):
    if passes("jul30", horizon_ms) and passes("aug04", horizon_ms):
        selected_horizon_ms = horizon_ms
        break
```

No ranking, tie-break, weighted score or "best coverage" fallback is allowed.
Once the first passing horizon is found, later horizons remain reported but
cannot replace it.

If none passes:

```text
selection_status = inconclusive_data_quality_or_coverage
selected_horizon_ms = null
```

This is a valid H0-A research result, not permission to relax a threshold.
H0-B remains locked.

## 9. Primary Tuple Freeze

Only `horizon_ms` is selected by H0-A. All other primary-tuple fields are
inherited constants:

```text
stage_id = stage_h0a
feature_set_id = feature_set_h0
target = public_bbo_moves_through_quote
distance_definition = target_visible_best_quote
delta_ticks = 0
horizon_ms = mechanically selected value or null
gate_latency_ms = 100
side_aggregation = equal_weight_bid_ask_session_scores
calendar_grid_ms = 10
primary_block_seconds = 60
formal_session_ids = [jul30, aug04]
diagnostic_session_ids = [aug03]
```

The `100ms` latency is frozen here because v2 primary multiplicity is
`target × delta × horizon × latency × side_aggregation`, and Gate H-C requires
the gate-relevant latency to be frozen before outcomes open.

Secondary values remain frozen as non-primary:

```text
descriptive_horizons_ms = [1000, 2000]
latency_sensitivity_ms = [25, 50, 250, 500]
distance_sensitivity = one_tick_secondary_only
single_side_results = secondary_only
```

`primary_tuple_freeze.json` must bind:

- the support projection identity;
- the exact selection trace;
- the accepted dependency identities;
- the accepted Trust Kernel pin;
- the exact input inventory identity;
- the precomputed H0-A code/contract identity C;
- `selection_status`;
- every inherited and selected tuple field;
- a statement that no H0-B outcome aggregate was opened.

It is written in the staging package, fsynced before publication, and becomes
immutable with the accepted package. H0-B cannot rewrite it.

The tuple does not contain final R, E or composite identities because it is
itself an R artifact. `h0a_manifest.json` later binds the tuple SHA256 and the
final R/C/E/composite identities without creating a self-reference.

## 10. Output Package

Proposed formal package root:

```text
local_live_analysis/
  skhynix_continuous_conditional_risk_v2_stage_h0a_support_only/
```

The exact artifact tree is frozen in the future task and Surface Matrix. The
reviewed minimum output set is:

```text
h0a_manifest.json
frozen_h0a_contract.json
input_bindings.csv
support_access_ledger.json
calendar_grid_support_by_segment.csv
calendar_grid_support_by_session.csv
source_cadence_by_session.csv
censoring_identification_by_horizon.csv
dependence_support_by_horizon.csv
support_projection_commitments.csv
horizon_selection_trace.csv
primary_tuple_freeze.json
landmark_crosscheck.csv
reports/h0a_support_only.md
contracts/task.md
contracts/surface_matrix.json
contracts/execution_plan.md
contracts/v2_framework.md
contracts/accepted_kernel_pin.json
runtime_source/
runtime_tests/
```

No row-level 10ms grid is published. The package proves the streamed grid
through exact counts, per-segment/per-horizon canonical commitments and
independent source replay.

Hard size contract:

```text
no published raw-event copy
no published row-level calendar grid
full package size <= 64 MiB
```

Exceeding the hard cap fails before publication with
`H0A_PACKAGE_SIZE_LIMIT_EXCEEDED`.

### 10.1 Canonical CSV Contract

All CSV files use UTF-8, LF endings, comma delimiters, one exact header row,
lowercase `true`/`false`, base-10 integers without separators and exact
decimal text derived from integer ratios. Unknown, missing, duplicate or
reordered columns fail.

| Artifact | Primary key |
| --- | --- |
| `input_bindings.csv` | `snapshot_phase, scope, role, session_id, segment_id, relative_path` |
| `calendar_grid_support_by_segment.csv` | `session_id, segment_id, connection_epoch_id, horizon_ms` |
| `calendar_grid_support_by_session.csv` | `session_id, horizon_ms` |
| `source_cadence_by_session.csv` | `aggregation_level, session_id, segment_id, venue, channel` |
| `censoring_identification_by_horizon.csv` | `session_id, horizon_ms, identification_class` |
| `dependence_support_by_horizon.csv` | `session_id, horizon_ms, block_seconds` |
| `support_projection_commitments.csv` | `session_id, segment_id, connection_epoch_id, horizon_ms` |
| `horizon_selection_trace.csv` | `evaluation_order, horizon_ms, session_id` |
| `landmark_crosscheck.csv` | `session_id, segment_id, check_id` |

Every CSV freezes exact ordered fields, exact row ordering, exact decimal
grammar and reject-on-unknown-field behavior.

The ordered fields are:

`input_bindings.csv`

```text
snapshot_phase
scope
role
session_id
segment_id
root
path
relative_path
bytes
sha256
authoritative_manifest_sha256
```

`calendar_grid_support_by_segment.csv`

```text
schema_version
session_id
segment_id
connection_epoch_id
evidence_label
formal_eligible
horizon_ms
primary_selection_eligible
segment_epoch_start_ns
segment_epoch_end_ns
first_grid_ts_ns
last_grid_ts_ns
nominal_calendar_grid_count
quality_eligible_grid_count
quality_eligible_calendar_exposure_fraction
fully_identified_binary_count
fully_identified_binary_endpoint_fraction
interval_likelihood_eligible_count
interval_likelihood_eligible_fraction
complete_60s_block_count
support_projection_row_count
support_projection_sha256
```

`calendar_grid_support_by_session.csv`

```text
schema_version
session_id
evidence_label
formal_eligible
horizon_ms
primary_selection_eligible
segment_count
connection_epoch_count
nominal_calendar_grid_count
quality_eligible_grid_count
quality_eligible_calendar_exposure_fraction
fully_identified_binary_count
fully_identified_binary_endpoint_fraction
interval_likelihood_eligible_count
interval_likelihood_eligible_fraction
complete_60s_block_count
quality_exposure_gate_pass
binary_identification_gate_pass
interval_likelihood_gate_pass
complete_block_gate_pass
session_support_gate_pass
```

`source_cadence_by_session.csv`

```text
schema_version
aggregation_level
session_id
segment_id
venue
channel
message_count
inter_arrival_count
inter_arrival_p01_ns
inter_arrival_p10_ns
inter_arrival_p50_ns
inter_arrival_p90_ns
inter_arrival_p99_ns
inter_arrival_max_ns
source_age_count
source_age_p01_ns
source_age_p10_ns
source_age_p50_ns
source_age_p90_ns
source_age_p99_ns
source_age_max_ns
no_message_calendar_grid_count
availability
semantic_note
```

`censoring_identification_by_horizon.csv`

```text
schema_version
session_id
horizon_ms
identification_class
grid_count
fraction_of_nominal_calendar_grid
binary_endpoint_identification_supported
interval_likelihood_eligible
formal_eligible
primary_selection_eligible
```

`dependence_support_by_horizon.csv`

```text
schema_version
session_id
horizon_ms
block_seconds
absolute_block_anchor
nominal_block_count
complete_block_count
clipped_segment_block_count
epoch_censored_block_count
quality_censored_block_count
source_gap_censored_block_count
complete_block_gate_pass
stage2_overlap_block_count_2000ms
stage2_overlap_block_interpretation
formal_eligible
primary_selection_eligible
```

For horizons other than 2000ms,
`stage2_overlap_block_count_2000ms` is empty and its interpretation remains
`not_applicable_non_2000ms_row`.

`support_projection_commitments.csv`

```text
schema_version
session_id
segment_id
connection_epoch_id
horizon_ms
support_projection_row_count
support_projection_sha256
first_grid_ts_ns
last_grid_ts_ns
```

`horizon_selection_trace.csv`

```text
schema_version
evaluation_order
horizon_ms
primary_selection_eligible
session_id
evidence_label
formal_eligible
quality_eligible_calendar_exposure_fraction
fully_identified_binary_endpoint_fraction
interval_likelihood_eligible_fraction
complete_60s_block_count
quality_exposure_gate_pass
binary_identification_gate_pass
interval_likelihood_gate_pass
complete_block_gate_pass
session_support_gate_pass
formal_session_pass_count
selected_at_this_horizon
selection_status_after_horizon
```

`landmark_crosscheck.csv`

```text
schema_version
session_id
segment_id
check_id
authoritative_source
expected_value
observed_value
status
enters_horizon_selection
```

### 10.2 Canonical JSON Contract

`primary_tuple_freeze.json` has this exact top-level key order:

```text
schema_version
task_id
stage_id
feature_set_id
frozen_date
selection_status
target
target_venue
target_channel
distance_definition
delta_ticks
horizon_ms
gate_latency_ms
side_aggregation
calendar_grid_ms
primary_block_seconds
formal_session_ids
diagnostic_session_ids
descriptive_horizons_ms
latency_sensitivity_ms
distance_sensitivity
single_side_results
support_projection_identity
horizon_selection_trace_sha256
input_inventory_sha256
code_contract_identity
accepted_dependency_identities
kernel_pin
boundary
```

`support_access_ledger.json` has this exact top-level key order:

```text
schema_version
task_id
stage_id
projector
selector
stage4
aug07
network
private_or_order
accepted_dependency_writes
source_inventory_before
source_inventory_after
source_inventory_unchanged
```

`frozen_h0a_contract.json` has this exact top-level key order:

```text
schema_version
task_id
frozen_date
authority
kernel_pin
accepted_dependencies
session_evidence
source_plane
source_field_access
calendar_grid
strict_asof
support_classifier
censoring
dependence_blocks
horizon_selection
primary_tuple
output_contract
boundary
identity_layers
atomic_publication
archive_portability
```

`h0a_manifest.json` has this exact top-level key order:

```text
schema_version
task_id
stage_id
frozen_date
package_path
kernel_pin
dependency_identities
contract_sha256
input_inventory_sha256_before
input_inventory_sha256_after
input_inventory_unchanged
primary_tuple_sha256
research_data_identity
code_contract_identity
evidence_identity
composite_identity
artifact_directory_allowlist
artifact_path_allowlist
artifacts
exact_counts
boundary
```

Every nested JSON object also uses a frozen exact key/type/value universe in
the formal task. Canonical JSON is UTF-8, sorted only where the frozen
contract says a map is semantically unordered, uses two-space indentation and
ends with one LF. Runtime wall-clock timestamps are excluded from package
identity; observed operation times belong in external receipts.

### 10.3 Support Projection Commitment

For each conceptual support row, the projector hashes a canonical tuple:

```text
schema_version
session_id
segment_id
connection_epoch_id
grid_ts_ns
horizon_ms
quality_eligible
binary_endpoint_identification_supported
interval_likelihood_eligible
identification_class
complete_60s_block_id_or_empty
```

The package publishes only the streaming row count and SHA256 per
segment/epoch/horizon. QA must independently rebuild these commitments.

## 11. Identity Layering

### 11.1 Research Identity R

R contains only scientific support outputs:

- calendar-grid support aggregates;
- cadence aggregates;
- censoring/identification aggregates;
- dependence-block aggregates;
- projection commitments;
- horizon selection trace;
- immutable primary tuple;
- landmark cross-check results.

R contains no runtime code, report prose, receipt or outcome value.

### 11.2 Code/Contract Identity C

C contains:

- `frozen_h0a_contract.json`;
- archived H0-A runtime source;
- archived H0-A runtime tests;
- exact frozen copies of the task, Surface Matrix, this execution plan and the
  v2 framework;
- `contracts/accepted_kernel_pin.json` and its accepted API binding.

### 11.3 Evidence Identity E

E contains:

- `h0a_manifest.json`;
- `input_bindings.csv`;
- `support_access_ledger.json`;
- canonical report;
- exact artifact and tree closure.

Hostile, build, admission, publication and archive receipts live outside the
immutable package under `.workflow/reports/`. They bind the final package
identity and chronology but do not alter it. In particular, the archive
receipt is created after package publication and cannot be a package artifact.

The composite identity is trusted only after R, C and E pass reverse binding.
Changing R while retaining old C/E, changing C while retaining old R/E, or
changing E while retaining old R/C must fail.

## 12. Proposed Surface Matrix

The formal task must instantiate these surfaces in canonical JSON and the
Markdown task in the same order:

| Surface ID | Authority | Required negative mutation | Stable error code |
| --- | --- | --- | --- |
| `kernel_pin` | accepted registry v1 entry and acceptance package | alter accepted entry pin | `KERNEL_PIN_MISMATCH` |
| `accepted_dependency_identity` | accepted Stage 1-4 manifests/contracts | coherent dependency identity substitution | `H0A_DEPENDENCY_IDENTITY_MISMATCH` |
| `session_authority` | Stage 1 bindings plus Stage 2 evidence labels | replace Aug03 accepted R1 with non-authoritative alignment | `H0A_NONAUTHORITATIVE_AUG03_R1` |
| `input_inventory` | exact Stage 1 source inventory and live files | mutate source bytes after before-snapshot | `H0A_INPUT_INVENTORY_CHANGED` |
| `source_schema_access` | frozen per-source schema and allowed-field map | request forbidden field or unknown column | `H0A_FORBIDDEN_SOURCE_FIELD_ACCESS` |
| `aug07_boundary` | Stage 1 Aug07 access ledger | open any Aug07 event-row path | `H0A_AUG07_EVENT_ACCESS_FORBIDDEN` |
| `stage4_crosscheck_boundary` | exact Stage 4 allowlist | open `outcomes/`, `features/` or `views/` | `H0A_STAGE4_SURFACE_FORBIDDEN` |
| `calendar_grid` | frozen integer-nanosecond grid oracle | shift origin by 1ns or use arrival grid | `H0A_GRID_ALIGNMENT_MISMATCH` |
| `strict_asof_state` | source timestamps, segment and epoch contract | use a future or cross-epoch observation | `H0A_STRICT_ASOF_VIOLATION` |
| `quality_censoring` | accepted quality intervals and source gaps | delete or shorten a censoring interval | `H0A_CENSORING_PROJECTION_MISMATCH` |
| `endpoint_identification` | frozen support classifier | reclassify interval-only support as binary support | `H0A_IDENTIFICATION_CLASS_MISMATCH` |
| `cadence_projection` | source-replay cadence oracle | alter an inter-arrival/source-age aggregate | `H0A_CADENCE_PROJECTION_MISMATCH` |
| `dependence_blocks` | absolute 60s block oracle | count a clipped/cross-epoch block as complete | `H0A_BLOCK_COMPLETENESS_MISMATCH` |
| `formal_session_eligibility` | Stage 2 frozen evidence strength | mark Aug03 formal eligible | `H0A_FORMAL_SESSION_ELIGIBILITY_MISMATCH` |
| `horizon_selection` | exact ordered first-pass algorithm | promote later passing or descriptive horizon | `H0A_HORIZON_SELECTION_MISMATCH` |
| `primary_tuple_freeze` | v2 constants plus selection result | alter target/delta/latency/side aggregation | `H0A_PRIMARY_TUPLE_MISMATCH` |
| `outcome_noninterference` | support-only projection contract | valid price mutation changes support output | `H0A_OUTCOME_NONINTERFERENCE_VIOLATION` |
| `support_projection_identity` | source replay canonical commitments | alter commitment/count pair | `H0A_SUPPORT_PROJECTION_MISMATCH` |
| `package_tree` | exact artifact/type allowlist | add symlink, special entry or extra file | `TREE_ENTRY_TYPE_FORBIDDEN` |
| `layered_identity` | Trust Kernel R/C/E/composite oracle | retain stale reverse binding | `COMPOSITE_IDENTITY_BINDING_MISMATCH` |
| `atomic_publication` | staging/fsync/rename contract | precreate a partial final package | `PUBLICATION_FINAL_EXISTS` |
| `durable_archive` | exact foreground archive envelope | alter archive tree or chronology | `ARCHIVE_TREE_MISMATCH` |

The final Surface Matrix may split a reviewed surface into smaller entries,
but it may not merge away a mutation, stable code, identity layer or evidence
requirement.

## 13. Hostile And Metamorphic Preflight

Hostile preflight runs before the first production-size source replay. At
minimum it must execute:

1. every Surface Matrix negative mutation against current runtime;
2. every mutation against the frozen runtime snapshot;
3. current/frozen exact stable-code parity;
4. direct tree attacks at source, staging and final package boundaries;
5. coherent manifest rehash attacks;
6. stale R/C/E reverse-binding attacks;
7. selector attempts to read raw inputs and Stage 4 forbidden surfaces;
8. Aug07 event-row open attempts;
9. non-authoritative Aug03 R1 substitution;
10. 1ns grid-origin drift;
11. cross-segment and cross-epoch endpoint construction;
12. interval-ambiguous-to-binary coercion;
13. Aug03 formal-promotion attack;
14. 1000/2000ms primary-promotion attack;
15. later-horizon-over-first-pass promotion;
16. partial publication and extra-tree-entry attacks.

### 13.1 Outcome Non-Interference Tests

On isolated input copies with refreshed source inventory:

1. change valid public BBO numeric values while preserving timestamps,
   non-nullness, positivity and `bid <= ask`;
2. do not change segment, epoch, quality, cadence or availability;
3. rerun H0-A0;
4. require all support outputs and selection inputs to be byte-identical.

This proves that support selection is not a hidden price-movement
calculation.

A separate control mutation changes an allowed support field such as a source
timestamp, segment boundary or quality interval and must change the
projection or fail with the declared stable code.

## 14. Business Execution Gates

### Gate 0: Dispatch Contract

- reviewed plan accepted and synchronized with v2;
- unique formal task ID created;
- task classified as `research_package`;
- canonical Surface Matrix validates against the frozen schema;
- Markdown/JSON surface parity passes;
- accepted Trust Kernel pin passes;
- EC1-EC7 are complete;
- no `TBD`, empty authority or free-text unavailable reason.

### Gate 1: Read-Only Input Preflight

- accepted Stage 1-4 identities pass;
- exact Jul30/Aug03/Aug04 authority passes;
- Aug03 accepted R1 resolves to `0804T001_old5h_replay`;
- before-inventory is frozen;
- accepted packages and source paths are read-only for the task;
- Aug07 event rows remain unopened;
- Stage 4 forbidden paths remain unopened.

### Gate 2: Hostile-First

- all declared current/frozen mutations execute;
- observed stable codes equal declared codes;
- fail-open count is zero;
- hostile receipt is content-addressed and consumed by the full runner.

### Gate 3: Two Independent Support Projections

- Build A and Build B run in isolated staging roots;
- both independently scan the accepted source inventory;
- both emit the exact support-only tree;
- all research outputs and support commitments are byte-identical;
- input before/after inventories are exact;
- accepted dependencies remain zero-write.

### Gate 4: Selector And Freeze

- H0-A1 sees only the sealed support projection;
- selection trace evaluates `50 -> 100 -> 250 -> 500`;
- Jul30/Aug04 are the only formal sessions;
- Aug03 is diagnostic only;
- 1000/2000ms remain descriptive;
- inherited tuple fields are exact;
- freeze manifest is written and fsynced before outcome access can exist.

### Gate 5: Package Admission

- current and archived H0-A verifiers agree;
- exact artifact tree and entry types pass;
- R/C/E/composite identities and reverse bindings pass;
- report values are mechanically derived;
- verify-only is zero-write for bytes, paths, SHA, mtime and entry types;
- package size is within the hard cap.

### Gate 6: Durable Archive

The proposed archive root is:

```text
/home/molly/project/durable_archives/
  skhynix_continuous_conditional_risk_v2/
  stage_h0a_support_only/
  <composite_identity>/
```

Publication must:

- run in the foreground and be explicitly waited;
- use a temporary sibling plus exact-tree verification and atomic rename;
- record observed archive start and completion times;
- prove strict `package publication < archive start < archive completion`;
- preserve any prior generation rather than overwrite it;
- bind local and remote R/C/E/composite identities.

Mac performs full source-semantic admission because the accepted external
inputs are present there. amdserver performs kernel/package byte admission
unless the exact external input inventory has separately been synchronized
and admitted. The archive must state:

```text
kernel_package_admission_portable = true
full_source_semantic_replay_portable = false
```

No amdserver command may claim full source-semantic replay while that flag is
false.

### Gate 7: Business Handoff

- business report status is `待验收`;
- formal package and archive identities are recorded;
- no controller acceptance or H0-B unlock is claimed;
- the independent QA entrypoint is exact and host-correct;
- worktree status and commit are recorded.

## 15. Independent QA Gates

Independent QA must use a fresh work root and must not reuse business Build A
or Build B as its source-semantic oracle.

### QA Gate 0: Contract

- reread workflow, task, plan, v2 and canonical Surface Matrix;
- validate accepted kernel pin and all EC1-EC7;
- verify exact surface order and stable codes.

### QA Gate 1: Authority And Boundary

- independently recompute dependency and input identities;
- verify session roles and the Aug03 accepted R1;
- prove Aug07 event rows and Stage 4 forbidden surfaces were not opened;
- compare before/after input inventories.

### QA Gate 2: Negative Topology

- execute every unique mutation on current and frozen implementations;
- execute tree attacks on every load-bearing boundary;
- assert exact codes, not only nonzero exits;
- require fail-open count zero.

### QA Gate 3: Fresh Source-Semantic Replay

- build one fresh full H0-A package from accepted inputs;
- independently recompute grid counts, cadence, censor classes, fractions,
  block counts and commitments;
- compare fresh/formal research outputs byte-for-byte.

### QA Gate 4: Outcome-Blind Selection

- rerun H0-A1 in a sealed selector root;
- execute valid-price metamorphic mutations;
- prove selector has no raw-source or Stage 4 outcome path;
- independently reproduce the first-pass horizon decision and tuple freeze.

### QA Gate 5: Trust Kernel Admission

- current and archived verification both pass;
- verify-only is zero-write;
- R/C/E/composite and reverse binding pass;
- exact tree, canonical serialization and report bindings pass.

### QA Gate 6: Archive And Portability

- remote archive tree is byte-exact;
- chronology is strict and observed;
- Mac full admission and amdserver kernel-only admission match the frozen
  portability contract;
- no background process remains running.

### QA Gate 7: Research Conclusion Boundary

- report contains support conclusions only;
- no adverse rate, effect, model, actionability or maker recommendation;
- no Aug07 event-row consumption;
- no Stage 1-4 mutation;
- QA status is exactly `已通过`, `未通过` or `阻塞`.

## 16. Verification Scope

The future task must freeze exact commands. The minimum categories are:

```text
workflow/schema validation
focused H0-A unit tests
current/frozen hostile matrix
outcome non-interference metamorphic tests
isolated Build A / Build B
current/frozen verify-only
fresh independent QA build
Ruff
compileall with external pycache
git diff --check
Mac full source-semantic admission
amdserver kernel-only archive admission
```

Unit coverage must include:

- integer grid ceil/floor and exact 10ms origin;
- segment/epoch boundary behavior;
- strict-as-of equality policy;
- source-age and no-new-information propagation;
- quote-validity checks without cross-time price comparison;
- every censoring class;
- exact fraction denominators and zero denominators;
- complete absolute 60s blocks;
- formal versus diagnostic session roles;
- ordered first-pass horizon selection;
- descriptive-horizon exclusion;
- immutable tuple serialization;
- selector path isolation;
- canonical commitments and report derivation.

## 17. Failure And Repair Policy

The task fails closed before publication when:

- an accepted dependency or source identity changes;
- the input inventory changes during execution;
- a forbidden path or field is accessed;
- any Aug07 event row is opened;
- Stage 4 outcome/feature/view data is opened;
- current/frozen stable-code parity fails;
- Build A and Build B differ;
- the selector can access raw inputs;
- outcome non-interference fails;
- a report value is not mechanically derived;
- R/C/E reverse binding fails;
- the final tree contains an unsupported entry;
- archive chronology or identity is not exact.

Repair must remain bounded to the failed surface. A repair may not:

- change support thresholds;
- promote Aug03;
- change horizon order;
- promote a descriptive horizon;
- change target, distance, latency or side aggregation;
- open H0-B outcomes to diagnose H0-A;
- silently rebuild an accepted dependency.

## 18. H0-B Unlock Rule

H0-B is eligible for a separate plan/task only when all are true:

1. H0-A independent QA status is `已通过`;
2. the controller records H0-A as accepted;
3. the accepted H0-A package has exact R/C/E/composite identities;
4. `primary_tuple_freeze.json` has
   `selection_status=selected`;
5. `selected_horizon_ms` is one of `50, 100, 250, 500`;
6. the H0-B task pins the exact accepted H0-A package and tuple identities.

If H0-A QA passes but selection is
`inconclusive_data_quality_or_coverage`, H0-A is complete but H0-B remains
locked. The controller must choose between data repair, new collection under
a new contract, or a v2 plan revision. It must not lower the gate in place.

## 19. Completion Definition

Stage H0-A is complete in workflow terms only when:

- this plan review is closed;
- a formal task and canonical Surface Matrix were created;
- business execution produced the exact small support-only package;
- hostile-first, deterministic builds and Trust Kernel admission passed;
- durable archive and portability evidence passed;
- independent QA returned `已通过`;
- controller tracking records the accepted H0-A identity and whether H0-B is
  unlocked or remains locked for inconclusive support.

Until then, "H0-A unlocked" means only that its prerequisite Trust Kernel is
accepted. It does not mean H0-A has started.

## 20. Review Checklist

The reviewer should explicitly accept, reject or revise these load-bearing
choices:

1. H0-A is two-process: support projector followed by sealed selector.
2. Individual prices may be decoded only for quote validity; cross-time price
   comparison and price emission are forbidden.
3. Aug03 is diagnostic and cannot count toward the two-formal-session gate.
4. Both Jul30 and Aug04 must pass a candidate horizon.
5. Grid origin is absolute epoch-aligned receive time at 10ms.
6. Fraction denominators are exactly those in §7.3.
7. Complete 60s blocks are absolute-time anchored and require all 6000 grid
   starts plus horizon support.
8. Stage 4 is limited to the explicit metadata/anchor allowlist.
9. H0-A freezes the default 100ms gate latency with the rest of the primary
   tuple.
10. No row-level 10ms grid is published; commitments plus replay are the
    audit mechanism.
11. H0-B remains locked when no primary candidate horizon passes.
12. Mac full replay and amdserver kernel-only admission are separate,
    truthful portability claims.
