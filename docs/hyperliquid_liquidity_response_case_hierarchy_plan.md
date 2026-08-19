# Hyperliquid Liquidity-Response Case Hierarchy Plan

## 1. Purpose

This plan extends the QA-accepted Hyperliquid liquidity-response M1 dataset
into a four-level temporal case hierarchy:

```text
ShockAtom
-> ShockCluster / ContinuousFlowEpisode
-> Residual Motif / Prototype
-> TemporaryRegime
```

The guiding principle is to begin with replayable, mechanism-anchored evidence
and only then introduce statistical structure:

```text
observable shock
-> auditable market process
-> repeated conditional response
-> temporary market context
```

The final boundary is research-structure validation. This plan does not identify
a specific maker, reconstruct exact queue ownership, estimate exact fills,
produce a trading signal, run a strategy backtest, or claim maker PnL.

## 2. Accepted Starting Point

The immutable input is the QA-accepted M1 v2 package:

```text
local_live_analysis/skhynix_liquidity_response_0730T017/
```

Its accepted facts are:

- task ID: `0801T001`;
- schema: `hyperliquid_liquidity_response_motif_v2`;
- threshold candidates: `268,522`;
- primary records: `141,768`;
- buy/sell records: `71,507 / 70,261`;
- primary response horizons: `1000ms / 2000ms`;
- diagnostic horizons: `100ms / 250ms / 500ms`;
- isolated records at `100/250/500/1000/2000ms`:
  `25,561 / 4,949 / 540 / 19 / 1`.

The current M1 field name `episode_id` is retained for compatibility, but each
M1 primary record is interpreted by this hierarchy as a `ShockAtom`: a minimal
confirmed shock, not a complete independent market process.

No downstream task may mutate, relabel in place, or silently regenerate the M1
package. Every derived object must reference the accepted M1 manifest and
source-row identity.

## 3. Goal 1: ShockAtom Contract

### 3.1 Definition

A `ShockAtom` is the smallest replayable cross-venue research case:

```text
Binance aggressive-trade threshold crossing
-> Binance depth confirmation
-> decision-time Hyperliquid pre-state
```

Its identity and event evidence are inherited directly from M1:

- `shock_ts`: the first trade that crosses the frozen impact threshold;
- `decision_ts`: the first confirming Binance depth state;
- aggressor direction;
- pre-shock best queue and price;
- shock and queue-drop intensity;
- trade attribution;
- decision-time Binance and Hyperliquid state;
- source segment, source row and source manifest SHA.

### 3.2 Visibility Semantics

- `visible_from = decision_ts`.
- A response field is readable only after its recorded source timestamp.
- `outcome_known_from` is the latest source timestamp required by the selected
  outcome view, normally the accepted `2000ms` response source.
- Missing or uncovered outcomes remain missing; they are never forward-filled
  into an apparently observed outcome.

### 3.3 Output

```text
atom/
  shock_atom_catalog.csv.gz
  shock_atom_manifest.json
```

The catalog is compact and references the accepted M1 row instead of copying
the full M1 feature surface.

### 3.4 Acceptance

- Exactly `141,768` atoms map one-to-one to M1 primary records.
- `atom_id` reuses the stable M1 `episode_id`.
- Direction, timestamps, segment identity and evidence fields reconcile
  exactly with M1.
- Every atom is traceable to a source file, row number and accepted SHA.
- No atom crosses a segment boundary or reads a future response.

## 4. Goal 2: ShockCluster And ContinuousFlowEpisode

### 4.1 Why A New Level Is Required

The observed inter-atom gap has a median near `75ms`. A gap-only diagnostic
shows:

| Gap | Initial clusters | Median atoms | P95 atoms | P95 duration |
|---:|---:|---:|---:|---:|
| 75ms | 71,290 | 1 | 5 | 202ms |
| 100ms | 48,777 | 2 | 9 | 435ms |
| 125ms | 33,924 | 3 | 13 | 776ms |
| 150ms | 24,699 | 3 | 19 | 1,233ms |

The primary `100ms` gap is therefore a conservative middle point: it groups
obviously dependent shocks without immediately collapsing the four-hour stream
into a small number of long regimes. The `75ms` and `125ms` versions are
diagnostic sensitivity views only.

### 4.2 ShockCluster

Within each segment:

1. Sort atoms by `shock_ts`.
2. Connect consecutive atoms when their gap is `<=100ms`.
3. A maximal connected run is one primary `ShockCluster`.
4. Segment boundaries always terminate a cluster.

The cluster records:

- first/last atom and duration;
- atom count and inter-arrival distribution;
- signed and absolute cumulative impact;
- dominant direction and direction persistence;
- maximum individual shock;
- cumulative removed queue;
- Binance price displacement and depleted levels.

### 4.3 ContinuousFlowEpisode

Adjacent clusters may be merged only when:

- the cluster-to-cluster gap is `<=250ms`; and
- no market-recovery checkpoint exists between them.

A recovery checkpoint requires at least two Binance timeline states spanning
`>=50ms`, with all of the following true:

- spread is no wider than the episode pre-spread plus one tick;
- combined bid/ask top-5 depth is at least `80%` of episode pre-depth;
- midpoint has not extended the episode-direction extreme for `>=50ms`.

If recovery evidence is missing or ambiguous, clusters are not merged. This is
the fail-closed choice.

### 4.4 Boundary Parameter Provenance And Freeze

The episode-boundary values are versioned research priors, not exchange
mechanism facts:

- `100ms` primary cluster gap;
- `250ms` cluster bridge gap;
- `50ms` recovery-state span;
- `+1 tick` spread allowance;
- `80%` combined top-5 depth recovery;
- `100ms` rolling phase window;
- two-atom reversal confirmation;
- `5s` long-flow threshold.

The `80%` value is seeded from the accepted M1 replenishment convention, but
its use as a Binance episode-recovery boundary is a new research assumption.
The other values are engineering priors motivated by the observed atom-gap
distribution and the accepted M1 response horizons. None may be presented as
a discovered market constant.

All primary values are calibrated only on discovery segments `0001-0003`
before motif features or outcome distributions are inspected. Calibration may
use only structural diagnostics:

- episode and cluster count;
- atom-count and duration distributions;
- singleton and long-flow fractions;
- boundary support and missing-evidence rates;
- membership stability across diagnostic parameter views.

Motif coherence, markout, prototype quality and held-out results must not be
used to choose episode-boundary parameters.

The primary parameter set is named `episode_boundary_v1` and is frozen in
`episode_manifest.json`. The manifest records every value, its provenance,
the discovery input SHA and the sensitivity output SHA. Any value drift fails
closed and requires a new boundary version; it may not silently overwrite
`episode_boundary_v1`.

The required diagnostic grid is:

| Parameter | Primary | Diagnostic |
|---|---:|---:|
| cluster gap | 100ms | 75ms / 125ms |
| bridge gap | 250ms | 200ms / 300ms |
| recovery-state span | 50ms | 40ms / 60ms |
| depth recovery | 80% | 64% / 96% |
| spread allowance | +1 tick | 0 / +2 ticks |
| long-flow threshold | 5s | 4s / 6s |

The sensitivity report includes episode counts, duration and atom-count
quantiles, long-flow rate and atom-membership Jaccard relative to the primary
version. Diagnostic versions never feed the formal baseline or motif pipeline.

### 4.5 Episode Phases

For each episode, calculate rolling `100ms` signed shock impact.

- The initial sign defines the onset phase.
- A new reversal phase begins when the rolling sign changes and the changed
  sign is supported by at least two consecutive atoms.
- A single small opposite atom remains a perturbation inside the current phase.
- The recovery checkpoint closes the final phase and the episode.

Episodes longer than `5s` are retained as `long_flow_case` objects but excluded
from motif discovery. They are evidence for persistent flow or regime analysis,
not forcibly split into artificial short cases.

### 4.6 Output

```text
episode/
  shock_atom_membership.csv.gz
  shock_cluster_catalog.csv.gz
  continuous_flow_episode_catalog.csv.gz
  flow_episode_phases.csv.gz
  episode_boundary_audit.csv.gz
  episode_manifest.json
```

### 4.7 Acceptance

- Every atom belongs to exactly one primary cluster and one primary episode.
- No atom is duplicated, omitted or assigned across segments.
- Membership and boundaries are deterministic under repeated builds.
- All recovery decisions expose their supporting timeline rows.
- The complete boundary parameter set, provenance and discovery calibration
  are recorded and rehashed before publication.
- The required diagnostic grid is reported, but only
  `episode_boundary_v1` is published as primary.
- Long-flow cases are reported separately and never silently discarded.

## 5. Goal 3: Conditional Baseline, Residual Motif And Prototype

### 5.1 Frozen Data Split

- Discovery: segments `0001-0003`.
- Held out: segments `0004-0008`, evaluated in chronological order.
- Parameters, feature transforms, distance thresholds and prototypes are
  frozen before reading held-out outcomes.

Baseline-family selection and all hyperparameter decisions are made inside the
discovery set using leave-one-segment-out blocked cross-validation:

```text
train 0002-0003 -> validate 0001
train 0001+0003 -> validate 0002
train 0001-0002 -> validate 0003
```

The held-out segments are a one-time final-test resource:

- no file from `0004-0008` may be opened by the discovery or freeze command;
- before held-out evaluation, the system writes a frozen research contract
  containing input, feature, baseline, PCA, graph, prototype and threshold
  hashes;
- the first held-out read atomically writes
  `heldout_consumption_manifest.json` with the frozen-contract SHA, input SHAs,
  run ID and first-read time;
- after consumption, changed parameters or artifact hashes cannot produce
  another result labelled `held_out`;
- a rerun after any adjustment is labelled `post_selection` or
  `in_sample_adjusted` and cannot satisfy formal held-out acceptance;
- `needs_more_samples` requires new segments or dates rather than tuning and
  rereading `0004-0008`.

### 5.2 Conditional Response Model

For episode `i`, horizon `h` and response channel `k`:

```text
R[i,h,k]
= f(S_pre, ShockPath, InterveningFlow[0,h], DataQuality)
+ residual[i,h,k]
```

Inputs are:

- pre-state: spread, depth, imbalance, volatility, basis and source age;
- shock path: atom count, duration, signed/absolute impact, direction
  persistence, reversal count, queue removal and price displacement;
- intervening flow: subsequent same/opposite shock count and mass, continued
  aggressive flow and Binance price path up to the target horizon;
- data quality: coverage, source age, degraded interval and no-new-information
  flags.

This is an explanatory baseline. Because it uses realized intervening flow, it
must not be reported as a decision-time prediction or tradable signal.

### 5.3 Response Targets

The baseline covers:

- Hyperliquid spread change;
- impacted and opposite queue ratios;
- fast-L2 top-5 depth and imbalance response;
- withdrawal, retreat, replenishment and follow latency;
- short-horizon directional midpoint response.

The accepted `1000/2000ms` adverse markout remains an evaluation outcome and
does not enter motif clustering.

### 5.4 Baseline Implementations

Two frozen baselines are compared:

1. Matched-neighbor median:
   - robust-standardized pre-state and shock features;
   - up to `200` eligible discovery episodes;
   - a minimum of `50` eligible neighbors is required.
2. Quantile gradient boosting:
   - scikit-learn `HistGradientBoostingRegressor`;
   - quantiles `0.25 / 0.50 / 0.75`;
   - `max_iter=200`;
   - `max_leaf_nodes=15`;
   - `learning_rate=0.05`;
   - `min_samples_leaf=100`;
   - `l2_regularization=1`;
   - `random_state=0`.

The matched-neighbor purge/embargo rule is interval based:

```text
query_evidence_label_interval
  = [episode_start, outcome_known_from]

neighbor_evidence_label_interval
  = [neighbor_start, neighbor_outcome_known_from]

expanded_query_interval
  = [episode_start - embargo,
     outcome_known_from + embargo]
```

A neighbor is eligible only when its complete evidence/label interval is
disjoint from `expanded_query_interval` and it belongs to the current training
fold.

The primary embargo has a `60s` floor. On discovery data, primary response
targets are aggregated into `1s` wall-clock bins and their autocorrelation is
measured. `estimated_decorrelation_lag` is the first lag at which every primary
target has `|ACF| < 0.1` for three consecutive lags:

```text
embargo = max(60s, estimated_decorrelation_lag)
```

The required embargo sensitivity views are `30s / 60s / 120s`. They report
eligible-neighbor counts and baseline error but do not change the frozen
primary embargo. If fewer than `50` eligible neighbors remain, the matched
baseline is unavailable for that query; forbidden neighbors are never added
to reach the target count.

`baseline_manifest.json` records the interval rule, label horizon, ACF method,
estimated lag, frozen embargo, sensitivity results, unavailable-query count
and actual neighbor-count distribution.

Gradient boosting becomes the official baseline only if it beats the matched
baseline on median absolute error and quantile loss in at least two of three
discovery cross-validation folds for the primary response targets. Otherwise
the matched baseline is frozen. Held-out performance is reported but never
used to select or replace the official baseline.

### 5.5 Standardized Residual

```text
residual = observed - predicted_q50

conditional_scale
  = max((predicted_q75 - predicted_q25) / 1.349,
        discovery_scale_floor)

standardized_residual = residual / conditional_scale
```

The scale floor is `10%` of the discovery target MAD. Missing observations
remain masked and are not mean-filled.

### 5.6 Motif Discovery

Clustering input contains:

- episode shock shape;
- standardized response residuals;
- response latency and phase structure;
- observation masks.

It excludes segment ID, absolute timestamp, `1000/2000ms` adverse markout and
all PnL-like outcomes.

The primary discovery pipeline is:

1. Robust scaling fitted on discovery segments.
2. PCA retaining at least `90%` variance, bounded to `8-15` dimensions.
3. Euclidean `k=10` nearest-neighbor search.
4. Keep only mutual-nearest-neighbor edges.
5. Enforce a maximum of ten similarity edges per episode.
6. Build communities with NetworkX Louvain,
   `resolution=1.0`, `seed=0`.

A discovery community must:

- contain at least `100` episodes;
- appear in all three discovery segments;
- receive no more than `50%` of members from one segment.

### 5.7 Prototype

Each motif stores:

- a real medoid episode as its prototype;
- p10/p25/p50/p75/p90 response curves;
- pre-state and shock distributions;
- nearest supporting cases;
- nearest counterexamples;
- same-pattern/different-outcome cases;
- discovery member-distance distribution.

Held-out assignment is allowed only when prototype distance is no greater than
the discovery member p95 threshold.

### 5.8 Statistical Acceptance

A motif may be classified as `motif_candidate_supported` only if:

- it reappears in at least four of five held-out segments;
- no held-out segment contributes more than `50%` of matches;
- held-out median prototype distance is no more than `1.25x` discovery;
- residual response direction is stable at two adjacent observed horizons;
- block-bootstrap confidence intervals support the response;
- surrogate significance survives Benjamini-Hochberg `q<=0.05`.

Use `199` full-pipeline surrogate runs for screening and `999` matched-response
permutations for final candidates. Other allowed results are
`response_structure_only`, `not_supported` and `needs_more_samples`.

### 5.9 Output

```text
baseline/
  baseline_manifest.json
  baseline_predictions.csv.gz
  episode_residual_features.csv.gz
  baseline_calibration.csv
  frozen_research_contract.json
  heldout_consumption_manifest.json

motif/
  motif_manifest.json
  motif_membership.csv.gz
  motif_prototypes.csv
  motif_response_curves.npz
  motif_counterexamples.csv.gz
  walk_forward_stability.csv
  surrogate_test_results.csv
```

## 6. Goal 4: TemporaryRegime

### 6.1 Context Windows

Construct non-overlapping `1min` windows independently inside each segment.
Regime formation must not use motif labels or future outcomes.

Context features are:

- Binance and Hyperliquid spread and depth;
- realized volatility;
- basis level and change;
- shock and episode arrival rate;
- signed flow and directionality;
- source age and degraded-state rates.

All transforms use discovery-segment median and IQR.

### 6.2 Temporary Boundaries

For each one-minute boundary:

1. Compute the robust median context of the preceding three windows.
2. Compute the robust median context of the following three windows.
3. Use their robust standardized Euclidean distance as the change score.
4. Create a boundary when the score exceeds the discovery p95 threshold and
   is at least `3min` from the previous boundary.
5. Segment boundaries remain hard boundaries.

Intervals shorter than `3min` are merged with the adjacent interval having the
smaller context distance.

Because these boundaries are discovered by scanning the context sequence,
the p95 threshold alone is not sufficient for publication. Every internal
boundary must pass an upstream surrogate calibration before it enters
`regime_intervals.csv`.

The primary regime surrogate contract is:

- `999` surrogate context sequences;
- preserve segment boundaries;
- permute contiguous `5min` context blocks within each segment;
- diagnostic block sizes `3min / 5min / 7min`;
- rerun the complete detector for every surrogate, including p95 threshold,
  minimum `3min` spacing and short-interval merging;
- record each surrogate's maximum accepted change score and final boundary
  count.

For real boundary `b`, the family-wise empirical p value is:

```text
p_max_score(b)
  = (1 + count(surrogate_max_score >= real_score(b)))
    / (1 + surrogate_count)
```

Only boundaries with `p_max_score <= 0.01` are published as data-driven
regime boundaries. Non-significant candidates remain in the boundary audit
and the surrounding period remains `context_only`. Segment boundaries are
mechanical boundaries and are recorded separately without a surrogate claim.

`regime_boundaries.csv` records:

- raw change score;
- boundary origin (`segment` or `data_driven`);
- surrogate method and block length;
- surrogate count and seed;
- empirical max-score p value;
- null boundary-count p value;
- significance and publication status.

### 6.3 Descriptive Labels

Discovery q33/q67 thresholds classify each interval's:

- liquidity;
- volatility;
- basis;
- shock intensity;
- directionality

as `low`, `normal` or `high`. Labels are descriptive and versioned, for
example:

```text
high_volatility__wide_spread__positive_basis__high_shock
```

They are not permanent market classes.

### 6.4 Linking Motifs To Regimes

- Assign each episode to the regime visible at its `decision_ts`.
- Motif labels are used only after regime boundaries are frozen.
- Report motif prevalence, prototype distance, outcomes and transitions by
  regime.
- A regime is informative only if it lasts at least `3min`, appears in at
  least two segments, and its motif distribution differs from the global
  distribution under one-minute block bootstrap.

Allowed conclusions are:

- `temporary_regime_candidate`;
- `context_only`;
- `not_supported`.

### 6.5 Output

```text
regime/
  regime_manifest.json
  one_minute_context.csv.gz
  regime_boundary_audit.csv
  regime_boundaries.csv
  regime_surrogate_summary.csv
  regime_intervals.csv
  episode_regime_membership.csv.gz
  motif_by_regime.csv
  regime_transition_summary.csv
```

## 7. Unified Storage And Interfaces

The derived package root is:

```text
local_live_analysis/skhynix_liquidity_response_case_hierarchy/
```

It contains:

```text
atom/
episode/
baseline/
motif/
regime/
case_hierarchy_manifest.json
```

Every manifest must record:

- source manifest paths and SHA-256;
- schema and algorithm version;
- discovery and held-out segments;
- parameters and build time;
- input rehash before publication;
- output row counts and SHA-256;
- as-of visibility semantics;
- atomic publication contract;
- local-only/no-network/no-AWS/no-SSH/no-new-collection boundaries.

Implementation uses existing NumPy, SciPy, scikit-learn, Polars and NetworkX.
It does not add HDBSCAN, UMAP, LightGBM, XGBoost or a graph database.

## 8. Verification And QA

### 8.1 Focused Tests

Synthetic tests must cover:

- a single atom;
- same-direction dense shocks;
- a small opposite perturbation;
- a persistent reversal;
- successful and failed recovery checkpoints;
- missing recovery evidence;
- segment truncation;
- a `>5s` long-flow case;
- missing response observations;
- baseline fallback;
- mutual-neighbor and prototype assignment;
- regime boundary and minimum-duration behavior.

### 8.2 Failure Injection

Every stage must fail closed on:

- missing, duplicate or cross-segment atom membership;
- source identity or SHA drift;
- future response access;
- changed discovery/held-out split;
- fitting transforms on held-out data;
- selecting a baseline from held-out performance;
- reading consumed held-out data after a parameter or artifact change;
- neighbor evidence/label overlap inside the frozen purge/embargo interval;
- episode-boundary parameter or provenance drift;
- outcome fields entering motif features;
- unbounded graph degree;
- publishing an uncalibrated data-driven regime boundary;
- non-atomic output replacement.

### 8.3 Full-Data Acceptance

Independent QA must reconcile:

- atom count and M1 identity;
- membership conservation;
- cluster and episode boundary decisions;
- response coverage and no-future joins;
- baseline held-out predictions and calibration;
- held-out first-read and frozen-contract identity;
- matched-neighbor purge/embargo and eligible-neighbor counts;
- PCA/model/prototype versioning;
- graph edge budget and community support;
- surrogate and bootstrap results;
- recovery-parameter sensitivity and manifest freeze;
- regime max-score surrogate calibration;
- regime time coverage and episode assignment.

Each goal is a separate formal workflow task. The default execution chain is:

```text
business thread
-> independent QA
-> controller unlocks the next goal
```

Failure at one goal leaves all previously accepted packages unchanged.

## 9. Final Research Boundary

The strongest result supported by this four-hour, single-session dataset is:

```text
auditable continuous-flow cases
+ conditional liquidity-response structures
+ held-out motif candidates
+ temporary context regimes
```

It does not support:

- cross-day stable market laws;
- a permanent market ontology;
- identification of a specific maker;
- exact fill or queue-position claims;
- maker PnL or production-signal readiness.

Additional dates and sessions are required before any motif can be promoted
from a local response-structure candidate to a stable reusable market pattern.
