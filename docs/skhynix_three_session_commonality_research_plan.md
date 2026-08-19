# SKHYNIX Three-Session Commonality Research Plan

Date: 2026-08-04

Task: `0804T006`

Amendment: `0804T007` adds directional cross-venue BBO dislocation signals.

Status: plan only; not executed.

## 1. Objective

Use three separately collected SKHYNIX Binance/Hyperliquid public-data
campaigns to
identify market structures that recur across sessions and can be described in
a stable, auditable way.

The target is not merely:

```text
three datasets have similar averages
```

The target is:

```text
same observable mechanism
-> same observed receipt-time event ordering
-> comparable episode shape
-> directionally consistent Hyperliquid response
-> explicit session-level prevalence and counterexamples
```

The strongest conclusion available from this plan is:

```text
cross-session recurrent mechanism candidate
-> candidate for later maker-rule research
```

It is not a validated signal, executable arbitrage, exact fill model, maker
identity claim or PnL result.

## 2. Frozen Input Inventory

| Dataset | Common timeline, CST | Duration | Segments | R0 | Current R1 |
| --- | --- | ---: | ---: | --- | --- |
| Jul30 | 2026-07-30 07:54:49 to 11:54:56 | 4h | 8 | pass | historical v2 pass; 1s/2s accepted |
| Aug03 | 2026-08-03 07:33:11 to 12:33:18 | 5h | 10 | pass | original v2 fail; current v3 replay pass |
| Aug04 | 2026-08-04 08:58:42 to 10:58:42 | 2h | 1 continuous | pass | current v3 pass |

Canonical source packages:

```text
local_live_analysis/cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m/
local_live_analysis/skhynix_cross_exchange_research_0730T013/

local_live_analysis/cross_exchange_collection_campaign_0802T001_skhynix_5h_10x30m/
local_live_analysis/skhynix_cross_exchange_research_0803T001/
local_live_analysis/skhynix_cross_exchange_research_0804T001_old5h_replay/alignment/

local_live_analysis/cross_exchange_collection_campaign_0804T001_skhynix_2h_continuous/
local_live_analysis/skhynix_cross_exchange_research_0804T001/
```

For Aug03, the original R0/research package remains
`skhynix_cross_exchange_research_0803T001/`, while the canonical current R1
v3 alignment is the isolated
`skhynix_cross_exchange_research_0804T001_old5h_replay/alignment/` replay.
The inventory manifest must bind these two paths explicitly and preserve both
their hashes.

Current normalized event inventory:

| Dataset | Binance hot rows | Hyperliquid hot rows | Auxiliary rows | Timeline rows |
| --- | ---: | ---: | ---: | ---: |
| Jul30 | 10,744,733 | 541,122 | 49,169 | 556,861 |
| Aug03 | 4,309,840 | 279,578 | 50,520 | 638,026 |
| Aug04 | 2,286,523 | 136,237 | 21,837 | 269,875 |

Known cadence commonality is already visible:

- Hyperliquid fast-L2 p50 source age is approximately `269-278ms` across all
  three campaigns.
- Hyperliquid fast-L2 p99 is approximately `596-650ms`.
- Hyperliquid standard-L2 p50 is approximately `2.62-2.76s`.
- Hyperliquid standard-L2 p99 is approximately `5.34-5.37s`.
- Hyperliquid BBO p50 is approximately `56-76ms`.

These are collection/microstructure observations, not proof of lead-lag alpha.

## 3. Why Existing Prototype IDs Cannot Be Compared Directly

The existing hierarchy produced:

| Dataset | ShockAtoms | ContinuousFlowEpisodes | Prototype v2 count | Formal supported |
| --- | ---: | ---: | ---: | ---: |
| Jul30 | 141,768 | 12,677 | 8 | 0 |
| Aug03 | 82,533 | 24,040 | 19 | 0 |
| Aug04 | not yet built | not yet built | not yet built | not applicable |

Jul30 and Aug03 were clustered independently. Therefore:

- `M0001` in one dataset is not the same object as `M0001` in another;
- prototype count differences may come from session activity, segmentation,
  graph geometry or sample density;
- Jul30 atom intensity is about `9.85/s`, while Aug03 is about `4.59/s`;
- Jul30 averages about `11.18 atoms/episode`, while Aug03 averages about
  `3.43 atoms/episode`.

Independent label names must never be used as evidence of cross-session
identity. Cross-session identity requires frozen transfer or explicit medoid
matching.

## 4. Definition Of “The Same”

Commonality is reported in five tiers.

### Tier C0: Data And Clock Commonality

The same information channel and timestamp semantics are available:

- same Binance and Hyperliquid instruments;
- same-host local-receipt join clock;
- same direction and price normalization;
- same no-future and segment-boundary rules;
- comparable source-age and degraded-interval fields.

This tier establishes comparability only.

### Tier C1: Structurally Recurrent

An Atom or Episode family is structurally recurrent when:

- assignment uses only the frozen outcome-free `structural_family_v1`
  feature contract;
- it occurs in all three datasets;
- each dataset contributes at least `100` eligible episodes;
- no single dataset supplies more than `70%` of pooled members;
- each session's out-of-distribution rate is at most `30%`;
- median assignment cost is inside the Jul30-frozen acceptance envelope;
- no more than `10%` of assigned members overlap warmup/degraded masks, and
  excluding those masks changes prevalence by no more than `20%`.

The primary support thresholds above are frozen. The `50/200` minimum support,
`60%/80%` dominance and `20%/40%` OOD alternatives are diagnostics only.

### Tier C2: Response Consistent

A C1 family becomes response consistent when:

- the primary outcome is direction-normalized Hyperliquid BBO midpoint
  response at `1000ms` and `2000ms`;
- those outcomes pass the dedicated classification-time label contract in
  Section 6.1 for every session; existing bookTicker-anchored R1 acceptance is
  input evidence but is not reused as Episode-end outcome qualification;
- the family has the same pre-registered response sign in all three sessions;
- at least two sessions have a within-session `95%` stratified-block-bootstrap
  interval excluding zero, while the third has the same sign and is not
  significantly opposite;
- the absolute median response at a passing primary horizon is at least
  `0.25` times that session's median Hyperliquid spread in bps;
- at least `80%` of within-session stratified-block-bootstrap draws preserve the
  pre-registered sign;
- after source-age, spread, volatility and shock-intensity adjustment, the
  family coefficient keeps its sign and at least `50%` of its unadjusted
  magnitude;
- first-after-target coverage is at least `80%` in every session and its
  `95%` interval is not entirely opposite with practical magnitude at least
  `0.25` spread.

The per-session null is that the pre-registered direction-normalized median
response is less than or equal to zero. If a family is pre-registered as
reversion rather than continuation, its sign and null are reversed before any
Aug04 outcome is read.

Response-curve Spearman correlation is a diagnostic, not a classification
gate, because the historical Jul30 contract formally accepts only two
horizons. It is reported only when canonicalization yields at least four
common-qualified horizons; the primary diagnostic target is pairwise
correlation `>= 0.70`.

### Tier C3: Confirmed In The Three Observed Sessions

A C2 family becomes confirmed in the finite three-session sample only when:

- its method, family definitions, transform, distance envelope, outcomes and
  hypotheses were frozen before Aug04 outcome access;
- Aug04 passes the full-pipeline state-preserving lag-shift test at
  empirical `p <= 0.05`;
- Benjamini-Hochberg correction across frozen families and primary horizons
  gives `q <= 0.10`;
- the practical effect threshold from C2 is met on Aug04;
- Jul30 and Aug03 retain the same sign, support and structural mapping;
- counterexamples and failure regimes are explicitly published.

This tier means only “confirmed in Jul30, Aug03 and Aug04.” It does not imply
population-level replication across future dates. Three sessions are three
independent session observations; no row-level bootstrap can increase that
sample size. Population-level replication requires a separately
pre-registered additional-session study and user authorization before any new
collection.

### Tier C4: Maker-Relevant Candidate

A C3 family may become maker-relevant only when:

- the vulnerable passive side follows mechanically from shock direction;
- the response occurs after the decision timestamp and before the proposed
  maker action becomes stale;
- effect size remains after spread, fees and conservative latency assumptions;
- the conclusion is stated as quote-protection or quote-intent evidence, not
  exact fill or PnL evidence.

This plan does not execute C4.

### Threshold Provenance And Sensitivity

The C1-C3 thresholds are versioned research priors, not discovered market
constants. They must be frozen before Aug04 outcome-level access. Their
scientific purpose is to prevent post-hoc threshold relaxation; they are not
asserted as universal market constants.

Required diagnostic views:

| Primary threshold | Diagnostic alternatives |
| --- | --- |
| minimum 100 episodes/session | 50 / 200 |
| maximum 70% pooled membership from one session | 60% / 80% |
| 80% bootstrap sign stability | 70% / 90% |
| diagnostic response-curve correlation 0.70 | 0.50 / 0.80 |

Only the primary contract determines classification. Diagnostics report
sensitivity and may not be used to tune the primary contract after Aug04 is
opened.

## 5. Research Architecture

Use two complementary tracks.

### Track A: Confirmatory Frozen Transfer

Purpose: test whether a structure defined without Aug04 recurs in Aug04.

```text
Jul30 discovery-only outcome-free structural geometry
-> frozen transfer to Aug03
-> no-refit confirmation on Aug04
```

Rules:

- Existing Jul30 `motif_v2` prototypes are designated
  `legacy_response_archetype`. Their schema contains
  `response_observed__*` and `response_residual__*`; they are descriptive
  historical outputs and are not confirmatory assignment anchors.
- A new `structural_family_v1` contract is fitted only on Jul30 discovery
  segments `0001-0003`.
- Jul30 supplies outcome-free boundary parameters, robust transforms,
  structural medoids and assignment envelopes.
- Aug03 is a historical transfer audit; it has already been consumed by prior
  diagnostic research and is not fresh held-out evidence.
- Aug04 remains unopened by Atom/Episode/Prototype and directional-BBO signal
  code until a freeze manifest is written.
- Aug04 is the confirmation session for the frozen family definitions.
- No feature, transform, distance threshold, unmatched penalty, hypothesis or
  family interpretation may change after the first Aug04 Atom or outcome read.

Aug04 first-read ledger:

1. Publish and fsync `freeze_manifest.json`, including contract SHA, code SHA,
   input inventory SHA and freeze timestamp.
2. Acquire one exclusive consumption lock.
3. Create `aug04_consumption_manifest.json` with `O_CREAT|O_EXCL`, state
   `read_started`, and fsync it before opening any Aug04 Atom/Episode,
   directional-BBO signal or outcome input.
4. Record run ID, process ID, command, first-read UTC nanoseconds, freeze SHA,
   canonical Aug04 input SHA and guarded-reader source SHA.
5. Route every Aug04 read through the guarded reader. A missing/mismatched
   freeze or existing ledger for another run fails closed.
6. On success, atomically replace the ledger with state `consumed`, exact
   consumed path/SHA list and output manifest SHA. A crash leaves
   `read_started`, which is an auditable failure and cannot be silently retried.

The freeze and consumption ledgers are immutable task evidence. Re-running
Aug04 confirmation requires a new run ID and an explicitly linked attempt
ledger; it may not overwrite the first-read record.

Allowed assignment features:

- complete Binance-only Episode path: atom/cluster count, duration,
  inter-arrival shape, signed/absolute shock, queue removal, direction
  persistence/reversal and Binance price/depth path;
- Hyperliquid BBO/fast-L2 liquidity and source age available at the decision
  timestamp;
- strictly trailing basis level/innovation context computed without future
  samples.

Forbidden assignment features:

- every `response_observed__*` and `response_residual__*` field;
- Hyperliquid markout, first-response latency or any post-decision liquidity
  path;
- basis closure or any post-decision basis path;
- labels or statistics fitted using Aug03/Aug04 outcomes.

Response and basis targets are evaluated only after family assignment has been
sealed. Full Episode structure is an offline mechanism taxonomy. It is not an
online maker signal; later live use requires a separate decision-time prefix
distillation and replay task.

Time anchor:

- an Atom's `classification_available_ts` is its frozen Binance depth
  confirmation timestamp;
- an Episode's `classification_available_ts` is the first timestamp at which
  the frozen boundary logic can assert that the Episode has ended;
- every confirmatory feature is available at or before that timestamp;
- C2/C3 primary response horizons start from that timestamp;
- paths measured from Episode start through Episode completion are
  contemporaneous descriptive mechanism views only and cannot support an
  actionable or no-lookahead claim.

This track can support a clean statement such as:

```text
The Jul30-defined adverse-flow family was observed in Aug03 and reappeared
under the frozen definition in Aug04.
```

### Track B: Descriptive Cross-Session Consensus

Purpose: discover structures missed by Jul30 anchors.

```text
fixed common feature contract
-> independent within-session prototypes
-> explicit prototype matching
-> pooled consensus family
```

This track uses all three datasets and is exploratory. It cannot call Aug04
held out.

Prototype matching uses outcome-free structural fields only:

```text
0.60 * Binance flow and episode-shape distance
+ 0.20 * Binance pre-state liquidity distance
+ 0.10 * Hyperliquid decision-time liquidity distance
+ 0.10 * strictly trailing basis-context distance
```

Robust median/IQR scaling is fitted on Jul30 discovery segments only. Pairwise
matching uses Hungarian assignment with dummy unmatched nodes. The unmatched
penalty is frozen as the Jul30 discovery `p95` nearest non-own-medoid distance.
A match is ambiguous and rejected when its second-best cost is at most `1.10`
times its best cost.

Each component distance is the median absolute difference over that
component's robust-scaled fields. The composite is undefined when a required
component has no valid fields; such a pair must remain unmatched. Ties are
resolved by frozen medoid SHA ordering. Structural prototype count, clustering
algorithm, seed and stopping rule are selected on Jul30 discovery structure
only and frozen before Aug03/Aug04 inspection.

Post-decision response curves and basis paths are evaluation targets only and
are never part of matching. Track B uses all sessions and is exploratory; it
cannot reach C3 in the current study.

## 6. Goal 0: Canonical Contract Normalization

Before commonality research:

1. Rebuild or validate all three campaigns under the same current code.
2. Publish R1 v3 for Jul30 in an isolated output.
3. Use the accepted Aug03 v3 replay semantics rather than the historical
   failed v2 result.
4. Preserve the original historical manifests and explain every qualification
   change.
5. Freeze a three-dataset inventory manifest containing every source path,
   row count, SHA, segment boundary and degraded interval.

Required gates:

- all R0 packages pass;
- all canonical R1 packages pass;
- future joins, timestamp regressions and cross-segment labels are zero;
- no label or feature crosses a segment boundary;
- source and runtime implementation hashes are frozen;
- raw data is not rewritten.

Outputs:

```text
commonality/
  dataset_inventory.csv
  canonical_input_manifest.json
  qualification_reconciliation.md
```

### 6.1 Classification-Time Outcome Contract

Build a dedicated label package after structural family assignment. Existing
R1 labels are anchored on Binance bookTicker price changes, so their passing
status is input evidence only and cannot qualify Episode-end outcomes.

For every eligible Atom/Episode anchor:

1. Require the anchor to be at or after the segment common-ready timestamp.
2. Require `classification_available_ts + 2000ms` to remain in the same
   segment.
3. Read segment-local Hyperliquid BBO strictly as-of the anchor, `+1000ms` and
   `+2000ms`.
4. Require every selected source local-receipt timestamp to be at or before its
   target local-receipt timestamp; exchange/source-native timestamps remain
   diagnostics. Future joins, timestamp regressions and cross-segment labels
   must be zero.
5. Preserve BBO source age as a diagnostic and apply reconnect/degraded masks
   as hard exclusions. An unchanged older BBO remains a valid step-function
   state under the accepted R1 v3 semantics.
6. Build first-after-target diagnostics from the first same-segment BBO event
   after each target; never substitute it for the primary as-of state.

Units:

```text
response_bps
= 10000 * direction * (target_mid / anchor_mid - 1)

anchor_spread_bps
= 10000 * (anchor_ask - anchor_bid) / anchor_mid
```

The practical threshold is
`abs(median(response_bps)) >= 0.25 * median(anchor_spread_bps)`.

Per-session gates:

- eligible-anchor reconciliation is exact;
- valid primary label coverage is at least `95%` at both horizons;
- future joins, timestamp regressions and cross-segment labels are zero;
- degraded/reconnect overlap is zero for formal labels;
- first-after diagnostic coverage is at least `80%`;
- a first-after contradiction exists only when its `95%` interval is entirely
  opposite the as-of sign and its absolute median is at least `0.25` median
  anchor spread.

Required package:

```text
classification_outcomes/
  labels/<session>/<segment>.csv.gz
  classification_outcome_quality_by_session.csv
  classification_outcome_source_age.csv
  classification_outcome_reconciliation.csv
  classification_outcome_manifest.json
```

The manifest binds anchor-table SHA, assignment SHA, canonical input SHA,
builder source SHA, exact row counts, coverage, exclusions and every
reconciliation counter.

## 7. Goal 1: Common ShockAtom Contract

Build the same ShockAtom definition in every dataset:

```text
Binance aggressive-flow threshold crossing
-> Binance depth confirmation
-> decision-time Hyperliquid pre-state
```

Freeze thresholds from the accepted Jul30 method. Do not recalibrate per
session.

Atom feature groups:

- direction and shock magnitude;
- confirmed queue removal and depleted levels;
- Binance spread, depth, imbalance and microprice displacement;
- Hyperliquid BBO and fast-L2 pre-state;
- strictly trailing basis level, innovation and residual;
- source ages and update flags;
- response state at `10/25/50/100/250/500/1000/2000ms`, stored only as
  evaluation outcomes and excluded from family construction.

All price and response values are direction-normalized. Raw USD price level is
not a clustering feature.

Common Atom reports:

- per-session count and rate;
- shock-size quantiles;
- direction balance;
- queue-removal distribution;
- source-age-conditioned response curve;
- cross-session standardized effect and heterogeneity;
- negative controls using reversed direction and shifted timestamps.

The primary trailing basis residual uses a `15min` rolling median and MAD
computed separately within each segment. `5min` and `30min` windows are
diagnostics. Warmup rows without a full trailing window are ineligible. Full
session demeaning is forbidden because it reads future session values.

## 8. Goal 2: Common Episode And Flow Dynamics

Apply one frozen Episode v2 boundary contract to every dataset:

- same cluster gap;
- same bridge gap;
- same recovery checkpoint;
- same reversal phase logic;
- same long-flow threshold;
- segment boundaries always terminate an episode.

Episode representation:

- atom and cluster count;
- duration and inter-arrival quantiles;
- signed and absolute cumulative shock;
- direction persistence and reversal count;
- peak queue removal;
- Binance price path;
- decision-time Binance and Hyperliquid liquidity;
- strictly trailing basis context;
- Hyperliquid first-response latency, response area, maximum
  adverse/favorable excursion, recovery time, post-decision basis path and
  liquidity replenishment path as outcome-only fields.

Do not require episode distributions to be equal. The research asks which
episode families recur despite different activity levels.

Acceptance:

- every Atom maps to exactly one Episode;
- no cross-segment membership;
- boundary support is replayable;
- family prevalence is reported per hour and per eligible decision, not only
  raw count;
- long-flow cases remain separate and are not forced into short prototypes.

## 9. Goal 3: Frozen Outcome-Free Structural Transfer

Do not use the accepted Jul30 prototype-v2 medoids for confirmatory transfer:
their feature schema includes response observability and response residuals.
Retain them only in a `legacy_response_archetype` appendix.

Build `structural_family_v1` from the Jul30 discovery split under the allowed
feature contract in Track A. Freeze:

- field names, direction normalization and missing-value policy;
- Jul30 discovery median/IQR transform;
- dimensionality reduction, when used, fitted on Jul30 discovery only;
- real Jul30 structural medoids;
- the Jul30 assignment envelope and out-of-distribution rule;
- family names based only on structural features.

For each Aug03 and Aug04 Episode:

1. Construct the frozen outcome-free feature vector.
2. Apply the Jul30 discovery transforms.
3. Calculate distance to each frozen structural medoid.
4. Assign only when distance is inside the frozen Jul30 envelope.
5. Record and validate `classification_available_ts`.
6. Seal assignment and its SHA before joining any later response outcome.
7. Otherwise label `out_of_distribution`.

For every transferred family publish:

- source Jul30 prototype identity and medoid SHA;
- per-session eligible and assigned count;
- assignment-distance p50/p95/max;
- out-of-distribution rate;
- episode-shape summary;
- independently joined response curve and post-decision basis path;
- closest counterexamples;
- classification ceiling.

Existing Jul30 outcome-based prototypes are not statistically supported.
Structural transfer can establish recurrence but cannot upgrade response
status without the C2/C3 tests.

## 10. Goal 4: Consensus Prototype Matching

Independently build descriptive prototypes in each dataset using identical
features and algorithm settings.

Match prototypes across sessions with a composite distance:

```text
0.60 * Binance_flow_and_episode_shape_distance
+ 0.20 * Binance_pre_state_liquidity_distance
+ 0.10 * Hyperliquid_decision_time_liquidity_distance
+ 0.10 * trailing_basis_context_distance
```

The weights, Jul30 discovery median/IQR scaler, dummy unmatched penalty and
ambiguity rule are research priors frozen before Aug04 prototype inspection.
Response curves and post-decision basis paths are forbidden matching inputs.

A consensus family requires:

- one matched prototype from every dataset;
- stable matching under pairwise and leave-one-session-out runs;
- no alternative match within `10%` of the selected cost;
- per-session minimum support;
- a real medoid from each session, never a synthetic centroid only.

In this three-session study Track B cannot pass C3 because Aug04 participates
in discovery. Its ceiling is C2-descriptive. Response sign agreement is tested
only after structural matching and does not participate in family selection.

## 11. Goal 5: Basis And Lead-Lag Mechanism Commonality

Prototype recurrence alone does not establish Binance lead or basis alpha.

For every common Atom/Episode family, estimate:

- Binance shock to Hyperliquid BBO response latency;
- Binance shock to Hyperliquid fast-L2 response latency;
- direction-normalized Hyperliquid markout curve;
- basis innovation and basis closure curve;
- impacted-side liquidity withdrawal and replenishment;
- Hyperliquid trade response;
- reverse-direction control: Hyperliquid event predicting Binance;
- shifted-time and sign-randomized controls.

Basis views:

- midpoint basis;
- microprice basis;
- executable bid/ask dislocation;
- rolling residual after session-local level removal;
- basis innovation rather than persistent level;
- basis response conditioned on source age and liquidity.

### 11.1 Directional Cross-Venue BBO Dislocation Signals

Study the two directions as separate state variables:

```text
Binance_price_Q(t) = Binance_native_price(t) * Binance_quote_to_Q(t)
Hyperliquid_price_Q(t) = Hyperliquid_native_price(t) * Hyper_quote_to_Q(t)

d_bh_Q(t) = Binance_bid1_Q(t) - Hyperliquid_ask1_Q(t)
d_hb_Q(t) = Hyperliquid_bid1_Q(t) - Binance_ask1_Q(t)

reference_mid_Q(t) = (Binance_mid_Q(t) + Hyperliquid_mid_Q(t)) / 2
d_bh_bps(t) = 10000 * d_bh_Q(t) / reference_mid_Q(t)
d_hb_bps(t) = 10000 * d_hb_Q(t) / reference_mid_Q(t)
```

`Q` is one frozen common quote unit. Preserve native prices alongside
quote-normalized prices. Quote-currency conversion source, timestamp, as-of
rule and SHA must be frozen. If the study uses a `USDT=USDC=USD=1` assumption
instead of an observed conversion stream, label it as a scenario and prohibit
unqualified executable-edge conclusions.

Interpretation:

- `d_bh > 0` is a gross top-of-book dislocation for buying the Hyperliquid
  ask and selling the Binance bid. It identifies the Hyperliquid ask as the
  potentially vulnerable passive side.
- `d_hb > 0` is the reverse gross dislocation for buying the Binance ask and
  selling the Hyperliquid bid. It identifies the Hyperliquid bid as the
  potentially vulnerable passive side.
- A positive value is not an executable-arbitrage claim. It excludes queue
  position, action latency, hidden liquidity, rejection, partial fill,
  account-specific fees and hedge failure.

Book invariant:

```text
d_bh_Q(t) + d_hb_Q(t)
= -Binance_spread_Q(t) - Hyperliquid_spread_Q(t)
<= 0
```

The identity must hold within numeric tolerance whenever both books are valid
and non-crossed. Both directional signals cannot be positive at the same
timestamp. Violations fail the signal input package.

#### Signal Input Contract

At each decision timestamp:

- use the deterministic union of Binance and Hyperliquid BBO state-change
  events as the primary decision population;
- after applying every event at one local timestamp in frozen
  `(local_ts, track_priority, source_sequence)` order, emit at most one state
  row for that timestamp;
- use a fixed `100ms` state grid only as an update-rate-bias diagnostic;
- join both venue BBO states strictly as-of one common same-host local-receipt
  timestamp;
- require every selected `source_local_receipt_ts <= decision_ts`;
- retain exchange/source-native timestamps as diagnostics only and never
  compare them directly with the local decision clock;
- require both books ready, same-segment, non-crossed and positive-sized;
- exclude reconnect, degraded and segment-warmup intervals;
- retain Binance and Hyperliquid BBO source age separately;
- retain all four top quantities and define
  `top_executable_qty=min(sell_bid_base_equivalent,
  buy_ask_base_equivalent)` for each direction;
- validate instrument price scale, quantity scale, tick size and symbol mapping;
- validate quote-currency conversion and contract multiplier provenance;
- store native-price, common-quote `Q`, bps, tick-normalized and
  trailing-residual forms.

Do not publish one ambiguous generic tick value. For each direction:

```text
sell_leg_ticks(t) = d_direction_Q(t) / sell_venue_tick_Q(t)
buy_leg_ticks(t) = d_direction_Q(t) / buy_venue_tick_Q(t)
conservative_common_ticks(t)
= d_direction_Q(t) / max(sell_venue_tick_Q(t), buy_venue_tick_Q(t))
```

For `d_bh`, the sell venue is Binance and the buy venue is Hyperliquid. For
`d_hb`, the sell venue is Hyperliquid and the buy venue is Binance. Native tick
sizes are converted to `Q` strictly as-of the same decision timestamp.

Venue-native quantities must be converted through frozen contract multipliers
before any cross-venue `min` or notional calculation. Missing or unverified
quantity conversion makes capacity outputs ineligible but does not invalidate
price-only signal analysis.

The primary trailing residual uses the same segment-local `15min` median/MAD
contract as basis research. `5min/30min` windows are diagnostics. Full-session
normalization is forbidden.

```text
dislocation_z(t)
= (dislocation_bps(t) - trailing_median_bps(t))
  / (1.4826 * trailing_MAD_bps(t))
```

Rows with zero/non-finite trailing MAD are ineligible for standardized-event
analysis and remain available for raw-level diagnostics.

#### Level, Change And Excursion Views

Run two complementary analyses.

Continuous local-projection view:

```text
future_outcome_h
= beta_level_h * level_z(t)
+ beta_change_h * change_z_100ms(t)
+ pre_state_controls
+ source_age_controls
+ segment_fixed_effect
```

Primary `delta_dislocation` lookback is `100ms`; `10/25/50/250/500ms` are
diagnostics. Coefficients are fitted separately for `d_bh` and `d_hb`.

Confirmatory predictor scaling:

```text
level_z(t)
= (dislocation_bps(t) - trailing_15m_median(dislocation_bps))
  / (1.4826 * trailing_15m_MAD(dislocation_bps))

change_z_100ms(t)
= (delta_dislocation_bps_100ms(t)
   - trailing_15m_median(delta_dislocation_bps_100ms))
  / (1.4826 * trailing_15m_MAD(delta_dislocation_bps_100ms))
```

Fit one joint OLS local projection per direction/outcome/horizon with
`level_z` and `change_z_100ms` together. The primary statistics are
`beta_level_h` and `beta_change_h`, interpreted as outcome change per one
trailing robust-sigma predictor increase. Rows with zero/non-finite predictor
MAD are ineligible. Controls are frozen as both venue spreads, log
base-equivalent top quantity, trailing Binance volatility, trailing basis
residual and both BBO source ages, plus segment fixed effects.

Interpretable event view:

- zero crossing into or out of positive gross dislocation;
- `dislocation_z` crossing `+2.0`;
- large widening/narrowing innovation using Jul30-discovery frozen quantiles;
- persistence episode from threshold entry until threshold exit;
- near-crossing states below zero, retained as controls rather than discarded.

The primary `dislocation_z=2.0` threshold is frozen before Aug04 access.
`1.5/2.5` are sensitivity diagnostics only.

#### Future Impact And Leg Decomposition

Evaluate strict as-of outcomes at
`10/25/50/100/250/500/1000/2000ms`. Each horizon receives its own coverage,
no-future, same-segment, quality-mask and source-age gate. Formal three-session
classification continues to use common-qualified horizons; shorter horizons
may remain diagnostic.

Directional-BBO primary horizons are frozen as
`100/250/500/1000/2000ms`. `10/25/50ms` are diagnostic only. A primary
horizon is `common_qualified` only when every session has at least `95%` valid
labels and zero future/cross-segment/quality-mask violations. A failed primary
horizon retains its frozen test slot with `quality_fail=true` and `p=1`; it
cannot be replaced by a diagnostic horizon.

The primary standalone hypotheses are:

- higher/rising `d_bh` predicts positive total closure and positive
  Hyperliquid-ask closure contribution;
- higher/rising `d_hb` predicts positive total closure and positive
  Hyperliquid-bid closure contribution;
- higher current dislocation predicts greater positive-threshold survival;
  survival is reported separately from closure and is not interpreted as
  favorable economics by itself.

Apply BH across the frozen family:

```text
30 joint fit keys
= 2 directions
  x 5 frozen primary horizons
  x 3 primary outcomes (total closure/Hyperliquid leg/survival)

60 primary hypothesis keys
= 30 joint fit keys x 2 predictors (level/change)
```

Depth depletion, trade arrival and replenishment use a separately corrected
secondary family:

```text
30 secondary joint fit keys
= 2 directions
  x 5 frozen primary horizons
  x 3 secondary liquidity outcomes
    (depth depletion/trade arrival/replenishment failure)

60 secondary hypothesis keys
= 30 secondary joint fit keys x 2 predictors (level/change)
```

Canonical identifiers:

```text
fit_key
= hex(SHA256(encode(["bbo-fit-v1", direction, horizon_ms, outcome])))

hypothesis_key
= hex(SHA256(encode(["bbo-hypothesis-v1", fit_key, predictor])))
```

`encode` is the plan's frozen length-prefixed canonical field encoder.

Secondary-family significance is descriptive and cannot raise the C1-C3 tier.

Primary outcome units and tests:

```text
total_closure_bps(h)
= 10000 * closure_direction_Q(h) / reference_mid_Q(t)

hyperliquid_leg_bps(h)
= 10000 * closure_from_hyperliquid_Q(h) / reference_mid_Q(t)

survival(h) = 1[d_direction_Q(t+h) > 0]
```

For both `level_z` and `change_z_100ms`, use one-sided nulls:

```text
H0_closure: beta <= 0
H0_hyperliquid_leg: beta <= 0
H0_survival: beta <= 0
```

Practical-effect gates per session:

- total-closure beta is at least `0.25` times median combined Binance plus
  Hyperliquid spread in bps;
- Hyperliquid-leg beta is at least `0.25` times median Hyperliquid spread in
  bps;
- survival linear-probability beta is at least `0.05`, meaning five percentage
  points per one robust-sigma predictor increase.

Spread units are:

```text
combined_spread_bps(t)
= 10000 * (Binance_spread_Q(t) + Hyperliquid_spread_Q(t))
  / reference_mid_Q(t)

hyperliquid_spread_bps(t)
= 10000 * Hyperliquid_spread_Q(t) / reference_mid_Q(t)
```

The survival model is the same joint OLS design interpreted as a linear
probability model. `2.5/10` percentage points are sensitivity diagnostics.

Directional-BBO C1 has two separately reported branches:

- continuous branch: at least `10,000` eligible union-BBO rows per session,
  no session contributes more than `70%`, and Jul30-frozen predictor-envelope
  OOD rate is at most `30%` in every session;
- event branch: at least `100` independent entries per event class per session,
  no session contributes more than `70%`, and maximum/minimum event rate per
  `1,000` eligible decisions is at most `4.0`.

Jul30 predictor envelope:

- build separately for `d_bh` and `d_hb` on Jul30 discovery rows;
- sort finite `level_z` and `change_z_100ms` values independently;
- freeze one-based nearest-rank `0.5%` and `99.5%` bounds with no
  interpolation;
- a row is OOD when either predictor falls outside its direction's frozen
  bounds;
- publish all four bounds, discovery row count and source SHA.

Independent event entries:

- process each direction and event class separately in canonical decision order;
- state events count only false-to-true entries; large-innovation point events
  count their first qualifying row and are considered exited immediately after
  that anchor row;
- primary refractory/merge gap is `500ms`;
- after one anchor, all same-class qualifying entries before
  `anchor_ts + 500ms` merge into that event and do not increase support;
- a later entry is independent only after the prior state has exited and its
  decision timestamp is at least `500ms` after the previous anchor;
- persistence entry remains one event until threshold exit; cross-class overlap
  is allowed but must retain separate event IDs;
- `250/1000ms` refractory gaps are diagnostics only.

Failure of one branch does not invalidate the other, but conclusions must name
the passing branch. C2 requires the same beta sign in all three sessions, at
least two session bootstrap intervals excluding zero, the third not
significantly opposite, and the practical gate above. C3 additionally requires
the frozen Aug04 lag-shift empirical `p <= 0.05` and primary-family BH
`q <= 0.10`.

C2/C3 are determined only by the frozen continuous-model betas. Zero-crossing,
z-excursion and persistence-event summaries have a C1-descriptive ceiling in
this study and cannot substitute an event contrast for a failed beta.

The `1/2/5ms` latency scenarios below are observed-state survival simulations,
not separately resolved market-response horizons. Horizons materially below
the BBO/source-age cadence cannot prove that the exchange had not changed
internally.

For `d_bh`:

```text
delta_d_bh_Q(h)
= [Binance_bid1_Q(t+h) - Binance_bid1_Q(t)]
- [Hyperliquid_ask1_Q(t+h) - Hyperliquid_ask1_Q(t)]

closure_bh_Q(h) = d_bh_Q(t) - d_bh_Q(t+h)
closure_from_binance(h) = Binance_bid1_Q(t) - Binance_bid1_Q(t+h)
closure_from_hyperliquid(h)
= Hyperliquid_ask1_Q(t+h) - Hyperliquid_ask1_Q(t)
```

For `d_hb`:

```text
delta_d_hb_Q(h)
= [Hyperliquid_bid1_Q(t+h) - Hyperliquid_bid1_Q(t)]
- [Binance_ask1_Q(t+h) - Binance_ask1_Q(t)]

closure_hb_Q(h) = d_hb_Q(t) - d_hb_Q(t+h)
closure_from_hyperliquid(h)
= Hyperliquid_bid1_Q(t) - Hyperliquid_bid1_Q(t+h)
closure_from_binance(h) = Binance_ask1_Q(t+h) - Binance_ask1_Q(t)
```

The two closure contributions must sum to total closure within numeric
tolerance. Quote conversion at `t` and `t+h` is independently strict as-of its
own local target timestamp. Conversion source age and changes are retained so
native quote movement and common-quote conversion movement can be reported
separately.

Report:

- future directional dislocation level and change;
- probability the signal is still positive at each horizon;
- time to zero, time to half-closure and maximum widening before closure;
- which venue/side moved first after signal formation;
- Binance-leg and Hyperliquid-leg closure contribution;
- future midpoint and microprice movement on each venue;
- top quantity and gross notional available at the observed BBO;
- outcome conditioning by source age, spread, volatility, liquidity,
  Atom/Episode family and TemporaryRegime.

First-mover ordering uses local receipt time and frozen source sequence.
Exact timestamp ties are `simultaneous`; `+/-1ms` ordering is reported as a
separate path-uncertainty sensitivity.

Path-outcome censoring:

- time-to-zero, time-to-half-closure and maximum-widening paths are defined
  only for events with initial `d_Q(t) > 0`;
- primary scan limit is `2000ms`; `5000/10000ms` are diagnostic only when the
  dedicated horizon/quality contract passes;
- time-to-zero is the first later union-BBO state with `d_Q <= 0`;
- time-to-half-closure is the first later state with
  `d_Q <= 0.5 * d_Q(t)`;
- maximum widening is the maximum `d_Q(u)-d_Q(t)` before the earlier of
  time-to-zero or censor time;
- censor at the earliest of scan limit, segment end, reconnect/degraded start
  or missing required quote/conversion state;
- never carry a path across a segment or quality interruption;
- unclosed paths retain `event_observed=0` and exact censor time; do not impute
  a closure time;
- publish Kaplan-Meier survival and restricted mean time within the frozen
  `2000ms` window. Fixed-horizon closure outcomes remain primary.

#### Formation Driver And Binance-Lead Test

The primary formation window for every event type is the strict as-of interval
`[t-100ms, t]`, where `t` is zero-crossing time, z-crossing time,
large-innovation decision time or persistence-entry time. Both endpoints must
remain in the same segment, and the full interval must avoid
warmup/reconnect/degraded masks. Diagnostic lookbacks
`10/25/50/250/500ms` rebuild driver labels independently and cannot replace the
primary label.

For `d_bh`:

```text
binance_contribution
= Binance_bid1_Q(t) - Binance_bid1_Q(t-100ms)

hyperliquid_contribution
= -(Hyperliquid_ask1_Q(t) - Hyperliquid_ask1_Q(t-100ms))
```

For `d_hb`:

```text
binance_contribution
= -(Binance_ask1_Q(t) - Binance_ask1_Q(t-100ms))

hyperliquid_contribution
= Hyperliquid_bid1_Q(t) - Hyperliquid_bid1_Q(t-100ms)
```

Both start states and quote conversions are strict as-of `t-100ms`. The two
contributions must sum to `d_Q(t)-d_Q(t-100ms)` within tolerance.

For every widening event, let:

```text
positive_binance_widening = max(binance_contribution, 0)
positive_hyperliquid_widening = max(hyperliquid_contribution, 0)

driver_share = positive_binance_widening
               / (positive_binance_widening
                  + positive_hyperliquid_widening)
```

The denominator must be positive. A window whose net signal widens but has no
positive leg contribution is invalid by construction and fails reconciliation.

Classify:

- `binance_driven` when `driver_share >= 0.70`;
- `hyperliquid_driven` when `driver_share <= 0.30`;
- `mixed` otherwise.

For `d_bh`, widening contributions are Binance bid rising and Hyperliquid ask
falling. For `d_hb`, they are Hyperliquid bid rising and Binance ask falling.

Binance receipt-time precedence is supported only when:

- Binance-driven widening is followed by the directionally expected
  Hyperliquid repricing more strongly than Hyperliquid-driven or matched
  no-shock controls;
- the result is consistent in all three sessions;
- source-age matching and `+/-1/2/5/10ms` timestamp sensitivity do not remove
  the effect;
- reverse-venue and state-preserving lag-shift controls do not reproduce it.

The conclusion remains receipt-time precedence association, not causal venue
leadership.

#### Maker-Side Hypotheses

Pre-register two Hyperliquid protection hypotheses:

```text
H-ASK:
high/rising d_bh
-> Hyperliquid ask is vulnerable
-> primary maker outcome:
   Hyperliquid_ask1_Q(t+h) - Hyperliquid_ask1_Q(t) > 0

H-BID:
high/rising d_hb
-> Hyperliquid bid is vulnerable
-> primary maker outcome:
   Hyperliquid_bid1_Q(t) - Hyperliquid_bid1_Q(t+h) > 0
```

These primary maker outcomes are exactly the direction-specific Hyperliquid
closure legs and are already included in the frozen BH family. Ask/bid depth
depletion, trade arrival and replenishment are secondary diagnostics in a
separate BH family; they cannot promote a signal to C2/C3 when the primary
repricing outcome fails.

Secondary liquidity labels use Hyperliquid fast-L2 as the primary source;
standard-L2 is diagnostic only.

Anchor:

- H-ASK uses the native Hyperliquid ask1 price and base-equivalent quantity at
  `t`;
- H-BID uses the native Hyperliquid bid1 price and base-equivalent quantity at
  `t`;
- require positive anchor quantity and valid fast-L2 state/source age.

Anchor-price quantity at `t+h`:

- when the native anchor price is present in fast-L2, use its reconstructed
  quantity;
- for H-ASK, if current best ask is above the anchor price, anchor quantity is
  exactly zero; if best ask is at/below anchor but the anchor price lies beyond
  observed depth, the label is unobserved;
- for H-BID, if current best bid is below the anchor price, anchor quantity is
  exactly zero; if best bid is at/above anchor but the anchor price lies beyond
  observed depth, the label is unobserved.

Labels:

```text
depth_depletion_fraction(h)
= max(0, 1 - anchor_price_qty(t+h) / anchor_price_qty(t))

trade_arrival(h)
= 1[at least one direction-consistent Hyperliquid aggressor trade in (t,t+h]]

replenishment_failure(h)
= 1[anchor qty first falls to <=50% of initial qty
     and does not recover to >=80% by t+h]
```

For H-ASK, direction-consistent trades are buyer-aggressor trades at native
price `>= anchor_ask_price`. For H-BID, they are seller-aggressor trades at
native price `<= anchor_bid_price`. Aggressor side must be present in source
data; unknown side makes the trade label ineligible. Trade base-equivalent
volume is reported as a diagnostic, while the primary outcome is the binary
arrival indicator.

Replenishment risk begins only after the `50%` depletion trigger. Rows without
that trigger are `not_at_risk` and excluded from the replenishment-failure
test. The first recovery to `80%` ends the failure state. Price-depth
unobservability, segment end, reconnect/degraded interval or the `2000ms` scan
limit right-censors the path. Require at least `100` at-risk rows per session
for the replenishment outcome.

All three secondary outcomes use one-sided `beta <= 0` nulls and the separate
secondary BH family. Their p/q values and coverage are published even though
they cannot raise the commonality tier.

Test the primary outcomes as quote-protection/quote-intent hypotheses. A later
maker-rule task must separately determine decision-time thresholds, action
latency, cancel success, queue position and fill economics.

#### Fee, Latency And Capacity Views

Publish:

- gross dislocation in native price, common quote `Q`, bps, sell-leg ticks,
  buy-leg ticks and conservative common ticks;
- break-even total fee/slippage hurdle in bps;
- scenario net edge under frozen public fee assumptions, clearly labelled
  non-account-specific;
- signal survival after `1/2/5/10/25/50/100ms` latency assumptions;
- top-of-book quantity and gross notional cap.

These are sensitivity views only. They cannot establish executable capacity,
simultaneous hedge execution, exact fills, arbitrage or PnL.

#### Cross-Session Acceptance

A directional BBO signal reaches:

- C1 when its level/change event definition and prevalence recur in all three
  sessions under the frozen input contract;
- C2 when future closure, leg attribution and maker-side response have the
  pre-registered primary repricing sign in all three sessions and meet the
  directional-BBO practical-effect and stratified-bootstrap gates;
- C3 only through the frozen Aug04 confirmation, BH correction and full
  state-preserving lag-shift controls already defined by this plan.

Analyze standalone signal effects first. Atom/Episode/Prototype conditioning is
a second-stage heterogeneity analysis and may not be used to retroactively
select the standalone signal threshold.

Lead-lag commonality requires:

- the observed Binance-to-Hyperliquid receipt-time association exceeds the
  reverse-direction control in all three datasets;
- peak response occurs after the locally received Binance decision event;
- sign and horizon ordering are stable;
- the association is not explained by stale Hyperliquid state;
- time-shift placebo and matched no-shock controls fail to reproduce it.

Clock and causality boundary:

- both venues use same-host local receipt timestamps, but their network paths,
  framing and server-side publication delays differ;
- exchange timestamps are diagnostics, not proof of synchronized clocks;
- rerun the full analysis after shifting Binance relative to Hyperliquid by
  `-10/-5/-2/-1/+1/+2/+5/+10ms`;
- report the smallest shift that removes sign or significance;
- control for pre-existing Hyperliquid moves, matched no-shock windows and the
  reverse venue direction;
- without an independent common-market proxy, residual common-factor
  confounding remains.

The strongest permitted wording is “Binance receipt-time precedence
association.” The study cannot establish causal venue leadership from these
public feeds alone.

## 12. Goal 6: Session-Aware Statistical Model

Separate within-session uncertainty from across-session generalization.

Primary confirmation:

- freeze `structural_family_v1` before Aug04 outcome access;
- use Aug04 as the finite-sample confirmation session;
- compute empirical significance by rerunning the full pipeline under the
  state-preserving multi-track lag-shift surrogate below;
- use exactly `999` deterministic surrogate seeds;
- apply BH across frozen families and `1000/2000ms` primary horizons.

The `1000/2000ms` statement above applies only to
`structural_family_v1` Atom/Episode confirmation. Directional-BBO confirmation
uses its separately frozen `100/250/500/1000/2000ms` primary family. The two
families receive separate BH correction and are both enumerated in the freeze
manifest; neither may borrow a passing horizon or q-value from the other.

Primary state-preserving surrogate:

1. Reconstruct every Hyperliquid track on its original native timeline:
   fast-L2, standard-L2, BBO, trades, asset context, main allMids and target-dex
   allMids.
2. For each segment and surrogate seed, draw one signed lag from the fixed
   millisecond grid `[-300000,-60000] U [60000,300000]`.
3. Pair Binance reference time `t` with every Hyperliquid track at native time
   `t + lag`. Apply the same lag to all Hyperliquid tracks and quality masks.
4. Do not rewrite local/source timestamps and do not reconstruct a book from
   reordered messages. Query every state, trade and auxiliary stream in its
   original order at the mapped native time.
5. Drop non-overlap rather than wrapping. Pre-trim `300s` at both segment
   edges, then apply the normal `15min` trailing-basis warmup and `2000ms`
   response-tail exclusion.
6. Recompute source ages in the mapped Hyperliquid native-time frame, recompute
   cross-venue basis features on the shifted pairing and propagate shifted
   degraded/reconnect masks.
7. Write lag by segment, seed, retained interval, eligible counts and source
   hashes into the surrogate manifest.

Lag selection and bootstrap sampling use one cross-language hash sampler.

Canonical field encoding:

```text
payload = ASCII("HFTBT-COMMONALITY-HASH-V1\\0")
for field in fields:
    bytes = UTF8(field)
    payload += uint32_big_endian(len(bytes)) || bytes
```

All task fields are ASCII. Integer fields use unsigned base-10 text with no
leading zeros except `"0"`. `run_seed` is a lowercase 64-character hex string
stored in the freeze manifest.

Unbiased index selection:

```text
attempt = 0
repeat:
    digest = SHA256(encode(fields + [decimal(attempt)])).digest()
    u = uint64_big_endian(digest[0:8])  # first 8 raw digest bytes
    limit = 2^64 - (2^64 mod population_size)
    if u < limit:
        return u mod population_size
    attempt += 1
```

Lag fields are exactly:

```text
["lag-v1", run_seed, surrogate_id, segment_id]
```

`surrogate_id` is `0..998`. The sorted lag grid contains every integer
millisecond from `-300000` through `-60000`, followed by `60000` through
`300000`, with no duplicates. The run seed, grid CSV SHA and sampler contract
SHA are frozen before Aug04 access.

This null preserves each venue's stateful order-book trajectory and within-
venue serial dependence while breaking contemporaneous cross-venue pairing.
The empirical one-sided p-value is:

```text
(1 + count(surrogate_effect >= observed_effect)) / (1 + surrogate_count)
```

Use the absolute value for a pre-registered two-sided hypothesis. The
surrogate effect includes the same practical-effect statistic used for C2.

Within each session, report stratified fixed-time-block bootstrap uncertainty
and an adjusted response model:

```text
response
= frozen_family
+ pre_state_controls
+ source_age_controls
+ segment_fixed_effect
```

A pooled hierarchical/random-effect fit may be published only as exploratory
descriptive evidence. With `n=3` sessions, its between-session variance and
population effect are not confirmatory and must not be used to claim
generalization to unobserved dates.

Recommended response targets:

- direction-normalized midpoint response;
- adverse-side BBO movement;
- impacted-side liquidity change;
- basis closure;
- response latency.

Serial dependence:

- use exactly `2000` stratified block-bootstrap draws per session, with
  `draw_id=0..1999`;
- use primary block length `60000ms`, with `30000/120000ms` diagnostics;
- for each segment, let `[eligible_start, eligible_end)` be the formal
  classification-time interval after warmup and response-tail exclusions;
- set `block_count=floor((eligible_end-eligible_start)/block_length)` and use
  exactly that many non-overlapping half-open blocks; drop the final partial
  tail and report its duration and excluded anchors;
- retain empty blocks because the resampling unit is time exposure;
- assign each Atom/Episode to the unique block containing its
  `classification_available_ts`; an Episode is never split;
- fail the statistic when any required segment has fewer than two full blocks;
- for each draw and segment, sample exactly `block_count` block indices with
  replacement using the canonical hash sampler fields:
  `["bootstrap-v1", run_seed, session_id, segment_id, block_length_ms,
  draw_id, draw_position]`;
- concatenate sampled blocks in canonical `segment_id`, then `draw_position`
  order. Duplicate blocks duplicate all member episodes; empty blocks add no
  episodes. Sampling the original number of blocks separately per segment
  preserves each segment's duration weight;
- compute the point estimate on the same full-block population used by the
  primary `60000ms` bootstrap;
- sort the `2000` finite draw statistics ascending and use one-based
  nearest-rank positions `50` and `1950` as the `2.5%/97.5%` interval, with no
  interpolation; any non-finite draw fails the statistic;
- sign stability is the exact fraction of the `2000` draw statistics with the
  pre-registered sign;
- never shuffle individual high-frequency rows independently;
- never move an event across segment boundaries;
- use the exact `999` lag-shift surrogates for publication tests.

Directional-BBO bootstrap extension:

- define each segment's primary BBO resampling interval as
  `[BBO_common_ready + 15min, segment_end - 2000ms)`;
- partition it with the same frozen non-overlapping `60000ms` block contract;
- assign every union-BBO row to the unique block containing its `decision_ts`;
- causal trailing predictors, controls, source ages and strict as-of outcomes
  are computed once on the original chronological stream and are never
  recomputed across concatenated bootstrap blocks;
- for each draw, copy every union-BBO row from each selected block, retaining
  direction, original segment ID and a draw-position duplicate ID;
- for each of the `2 x 5 x 3 = 30` primary joint fit keys, reconstruct its
  eligible sample from the copied rows using that direction/outcome/horizon's
  frozen predictor, control, conversion, source-age and quality masks;
- refit the complete joint OLS design, including `level_z`,
  `change_z_100ms`, frozen controls and segment fixed effects, independently
  for every draw and fit key;
- map the two beta coefficients from each fit to its `level` and `change`
  hypothesis keys, yielding exactly `60` primary hypothesis slots;
- percentile intervals and sign stability use `2000` beta draws per hypothesis
  key;
- a quality-failed horizon is not bootstrapped and sets all corresponding
  primary and secondary direction/outcome/predictor hypothesis slots to `p=1`;
- a rank-deficient, empty or non-finite refit fails that fit key; it cannot
  be dropped or replaced by another draw;
- the 30 secondary fit keys follow the same per-draw refit and two-beta mapping
  into their separate 60-hypothesis family;
- `30000/120000ms` diagnostics repeat the same fit-key refit contract.

The state-preserving lag-shift test reruns BBO state construction, predictor
construction, horizon eligibility and the complete joint OLS fit for every
surrogate. It does not permute only a final beta table.

For every negative or surrogate dataset, rerun the complete relevant path:

```text
Atom construction
-> Episode construction
-> structural prototype construction/transfer
-> matching and family selection
-> outcome join
-> family/horizon tests
-> BH correction
```

Permuting only the final outcome table is insufficient because it ignores
selection uncertainty.

Track-specific rule:

- Track A keeps the pre-frozen Jul30 medoids by design, but each surrogate
  reruns normalized event construction, Atom/Episode generation, frozen
  assignment, classification-time outcome construction and tests from the
  shifted venue relationship.
- Track B reruns within-session prototype construction, cross-session
  matching, family selection, outcome evaluation and correction for every
  surrogate.

Report:

- pooled effect;
- per-session effect;
- between-session variance;
- family x session interaction;
- empirical p-value;
- BH q-value;
- sign-stability rate;
- practical effect size.

The first two items are descriptive with three sessions. The primary
confirmatory quantities are Aug04 empirical `p`, BH `q`, practical effect and
the Jul30/Aug03 same-sign transfer audit.

## 13. Goal 7: Regime And Counterexample Analysis

Commonality must include its failure conditions.

Stratify by:

- Binance volatility;
- both-venue spread;
- Binance and Hyperliquid depth;
- shock intensity;
- fast-L2 and BBO source age;
- basis residual magnitude;
- episode direction persistence;
- local time bucket.

For every claimed common family publish:

- where it occurs;
- where it disappears;
- where response sign reverses;
- where latency exceeds a maker-useful window;
- where assignment becomes out of distribution;
- whether one session dominates the effect.

The existing Jul30 and Aug03 regime-v2 packages found zero internal
data-driven boundaries. This is evidence against inventing many fine-grained
regimes from the current sample. Start with coarse, pre-registered context
buckets.

## 14. Clear Description Contract

Every final “same point” must use this template:

```text
Commonality ID:
Name:
Observable mechanism:
Observed receipt-time event order:
Decision-time-visible inputs:
Per-session prevalence:
Episode shape:
Hyperliquid response curve:
Basis/lead-lag behavior:
BBO dislocation direction and state:
Formation driver:
Future leg decomposition and survival:
Source-age and quality boundary:
Counterexamples:
Statistical tier: C0 / C1 / C2 / C3
Maker interpretation:
Claims explicitly not supported:
```

Example of an acceptable statement:

```text
After a direction-persistent Binance sell-flow episode with confirmed top-level
queue removal, Hyperliquid bid-side response is delayed but directionally
consistent in all three sessions. The family is structurally recurrent and
response consistent, but it is not confirmed in the three-session sample after
multiple-testing correction.
```

An unacceptable statement is:

```text
Prototype M0006 works in all datasets and is profitable.
```

## 15. Negative Controls

Required controls:

- reverse venue direction;
- random shock sign;
- shifted Binance timestamps;
- matched no-shock windows;
- stale-source-age-only subsets;
- segment-start warmup subset;
- equalized session sample size;
- raw basis level without residualization;
- schema sentinel proving every forbidden outcome field is absent from
  assignment and matching;
- synthetic labels with preserved event cadence;
- source-age-matched directional BBO events with the signal magnitude removed;
- one-leg-only timestamp shifts for Binance and Hyperliquid BBO;
- positive-dislocation events after equalizing top quantity, spread and
  volatility distributions;
- formation-driver label permutation within session/segment/time bucket.

Every control reruns the full construction, matching, selection and testing
pipeline, including BH correction. A commonality claim fails when:

- any matched control reaches the same tier with at least `80%` of the primary
  practical effect;
- the time-shift sensitivity shows that plausible `1-10ms` path asymmetry
  reverses the claimed ordering;
- family mapping is unstable under the frozen unmatched/ambiguity rules;
- the adjusted effect changes sign or falls below `50%` of the unadjusted
  magnitude;
- Aug04 has `p > 0.05`, BH `q > 0.10`, or misses the practical effect
  threshold.

## 16. Ordered Execution Plan

### R0: Canonicalize Inputs

- Build current-contract Jul30 R1.
- Bind accepted Aug03 v3 replay and Aug04 v3.
- Freeze all source/runtime hashes.

### R1: Freeze Research Contract

- Freeze Atom/Episode thresholds.
- Freeze the outcome-free `structural_family_v1` features, transforms,
  horizons, matching weights, unmatched penalty, ambiguity rule, bootstrap
  block lengths, lag-shift grid and statistical thresholds.
- Freeze `d_bh/d_hb` definitions, quantity multipliers, 100ms primary
  innovation lookback, robust predictor scaling, OLS estimator, C1 support,
  practical-effect/null thresholds, tick units, secondary labels, fee
  scenarios and BBO outcome horizons.
- Mark existing outcome-based prototype-v2 artifacts as descriptive legacy
  outputs.
- Write and fsync the freeze manifest before opening Aug04 outcome-level
  Episode or directional-BBO data.
- Create the atomic Aug04 first-read consumption ledger through the guarded
  reader.

### R2: Build Common Atoms And Episodes

- Build all three sessions independently under the same contract.
- Publish conservation, quality and rate-normalized summaries.

### R3: Frozen Jul30 Structural Transfer

- Build Jul30 discovery-only structural anchors and transfer them to Aug03.
- Without refitting, evaluate Aug04.
- Seal assignment before joining response outcomes.
- Build and qualify dedicated classification-time outcomes for all sessions.
- Publish assignment, out-of-distribution and leakage-audit evidence.

### R4: Descriptive Consensus

- Build session-specific prototypes.
- Match medoids and publish consensus families.
- Keep this track exploratory.

### R5: Basis/Lead-Lag Response

- Estimate response and basis paths for recurrent families.
- Build standalone `d_bh/d_hb` state, innovation and excursion datasets.
- Estimate future dislocation, opportunity survival, first mover, formation
  driver and Binance/Hyperliquid leg decomposition.
- Test H-ASK and H-BID maker-side protection hypotheses before prototype
  conditioning.
- Run reverse-direction and time-shift controls.

### R6: Session-Aware Statistics

- Run stratified fixed-time-block bootstrap and full-pipeline state-preserving
  lag-shift surrogates.
- Treat hierarchical effects as exploratory because there are only three
  independent sessions.
- Apply BH correction.

### R7: Common Mechanism Catalog

- Publish clear descriptions, counterexamples and classification tiers.
- No new data collection is required for R0-R7.

If the result is `needs_more_sessions`, any additional collection requires a
new task, explicit user authorization and a user-confirmed active trading
window.

## 17. Deliverables

```text
local_live_analysis/skhynix_three_session_commonality_<task_id>/
  canonical/
    dataset_inventory.csv
    canonical_input_manifest.json
    freeze_manifest.json
    aug04_consumption_manifest.json
  atom/
    atom_commonality_by_session.csv.gz
    atom_response_curves.csv.gz
  episode/
    episode_commonality_by_session.csv.gz
    episode_counterexamples.csv.gz
  prototype_transfer/
    structural_family_assignments.csv.gz
    transfer_quality_by_session.csv
    assignment_outcome_separation_audit.json
  classification_outcomes/
    classification_outcome_quality_by_session.csv
    classification_outcome_source_age.csv
    classification_outcome_reconciliation.csv
    classification_outcome_manifest.json
  consensus/
    prototype_matching.csv
    consensus_family_catalog.csv
  mechanism/
    basis_lead_lag_by_family.csv.gz
    bbo_dislocation_state.csv.gz
    bbo_dislocation_events.csv.gz
    bbo_dislocation_response_by_session.csv.gz
    bbo_dislocation_leg_decomposition.csv.gz
    bbo_dislocation_formation_driver.csv.gz
    bbo_dislocation_path_survival.csv.gz
    bbo_dislocation_maker_hypotheses.csv.gz
    bbo_dislocation_secondary_liquidity.csv.gz
    bbo_dislocation_fee_latency_capacity.csv.gz
    bbo_dislocation_quality.csv
    bbo_dislocation_manifest.json
    negative_controls.csv.gz
    hierarchical_effects.csv
    lag_grid.csv
    bootstrap_sampling_manifest.json
    surrogate_manifest.json
  report/
    common_mechanism_catalog.md
    research_limitations.md
  commonality_manifest.json
```

## 18. Acceptance Gates

The package passes only when:

- canonical R0/R1 inputs pass and remain hash-stable;
- Aug04 is not read before the fsynced freeze manifest and atomic first-read
  ledger;
- no future feature join or cross-segment object exists;
- all three sessions use exactly the same Atom/Episode feature contract;
- every common family has per-session support and counterexamples;
- confirmatory assignment contains no response, markout, response-latency,
  post-decision liquidity or post-decision basis field;
- confirmatory outcomes begin no earlier than `classification_available_ts`;
- dedicated classification-time labels pass `95%` coverage, reconciliation,
  no-future, same-segment and quality-mask gates;
- Episode-start contemporaneous paths are labelled descriptive-only;
- structural transfer uses frozen Jul30 discovery transforms and medoids;
- family assignment is hash-sealed before outcome joins;
- consensus matching is label-independent and deterministic;
- outcome-based legacy prototype artifacts are explicitly excluded from
  confirmatory evidence;
- negative controls rerun the complete selection and testing pipeline;
- stratified fixed-time-block statistics preserve segment boundaries and
  serial dependence at the frozen block scale;
- bootstrap and lag selections are reproducible from the canonical
  length-prefixed hash sampler and frozen seed;
- lag-shift surrogates move every Hyperliquid track and quality mask together
  without rewriting or reordering native state;
- `d_bh + d_hb = -Binance_spread - Hyperliquid_spread` holds within the frozen
  tolerance for every eligible BBO row;
- all BBO signal and outcome joins are strict as-of, no-future and
  same-segment;
- venue quantities are converted to frozen base-equivalent units before
  cross-venue capacity calculations;
- quote conversion at every decision/outcome/formation endpoint is strict
  as-of and all closure/driver identities are evaluated in common quote `Q`;
- Binance and Hyperliquid closure legs sum to total directional-dislocation
  closure within the frozen tolerance;
- path outcomes use the frozen 2s scan, right-censor and no-cross-segment
  contract;
- the only primary maker outcome is direction-specific Hyperliquid BBO
  repricing; liquidity depletion/trades/replenishment remain separately
  corrected secondary diagnostics;
- standalone BBO signal thresholds and outcomes are evaluated before any
  Atom/Episode/Prototype-conditioned selection;
- Directional-BBO confirmation uses only its frozen
  `100/250/500/1000/2000ms` family; failed quality slots remain `p=1` and are
  not replaced;
- continuous/event C1 support, predictor scaling, OLS beta, one-sided null and
  practical-effect gates match the frozen BBO contract;
- sell-leg, buy-leg and conservative common ticks are reported separately;
- fast-L2 depletion, direction-consistent trade arrival and replenishment
  failure labels obey their frozen observability/censoring rules;
- H-ASK/H-BID conclusions remain quote-protection hypotheses and do not claim
  exact fills or executable arbitrage;
- multiple testing is corrected;
- repeated builds are byte-deterministic;
- every final statement uses the clear description contract;
- C3 is described only as confirmation in the three observed sessions;
- lead-lag language is limited to receipt-time precedence association;
- no output claims exact fill, maker identity, arbitrage or PnL.

## 19. Recommended Interpretation

The primary result should be a small catalog of mechanisms, not a large list
of clusters.

The preferred result shape is:

```text
2-5 clearly described recurrent mechanisms
+ their prevalence and response curves in each session
+ the regimes where they fail
+ an honest C0-C3 evidence tier
```

If no family reaches C2 or C3, publish the negative conclusion:

```text
The current three sessions share collection cadence and broad market-process
structure, but no response family is stable enough to support a cross-session
maker signal claim.
```

That conclusion is more useful than forcing unstable prototypes into a trading
rule.
