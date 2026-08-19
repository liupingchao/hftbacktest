# Prototype-Conditioned Lead-Lag Maker Rule Extraction Plan

## 0. Plan Status And Boundary

Status on `2026-08-01`:

- plan only;
- not executed;
- no implementation task has been created by this document;
- no artifact has been generated;
- no historical replay, public shadow, parameter search, network collection,
  private endpoint, order endpoint, or live process is authorized.

The purpose of this plan is to convert the existing 14 offline
response-structure prototypes into causal, decision-time-visible maker rule
candidates that can be tested on future market sessions.

The maximum conclusion available from the current single campaign is:

```text
prototype-conditioned maker rule candidate
-> candidate_for_future_public_shadow
```

It is not:

```text
validated trading signal
-> executable maker strategy
-> maker PnL evidence
-> production or live authorization
```

## 1. Problem Statement

The current data package contains one four-hour SKHYNIX campaign divided into
eight chronological segments. It does not contain independent dates or market
sessions.

The practical path is therefore:

```text
market mechanism constraints
+ offline response prototypes
+ causal decision-time feature extraction
-> provisional maker rules
-> historical transfer audit
-> future-session public shadow
-> rule modification and iteration
```

The central technical problem is that the 14 prototypes are full-episode,
offline research objects. Their membership uses information that may become
known only after the maker decision. A prototype ID cannot be used directly as
an online signal.

This plan introduces an explicit distillation layer:

```text
full-information offline prototype
        used only as a research label
                    |
                    v
decision-time-visible episode prefix
        -> online rule family / confidence / reject
        -> maker quote intent
```

Future outcomes remain labels. They never become trigger inputs.

## 2. Accepted Starting Point

### 2.1 Immutable SKHYNIX Inputs

The primary source package is:

```text
local_live_analysis/skhynix_liquidity_response_case_hierarchy/
```

Accepted hierarchy facts:

- `141,768` ShockAtoms;
- `48,777` ShockClusters;
- `12,677` ContinuousFlowEpisodes;
- `256` long-flow cases;
- `12,215` episodes with an available conditional-baseline prediction;
- `14` response-structure prototypes;
- all 14 prototypes appear in discovery segments `0001-0003`;
- all 14 prototypes match all held-out segments `0004-0008`;
- all 14 remain classified as `response_structure_only`;
- no prototype has final permutation support;
- the regime layer found zero internal data-driven regime boundaries.

Relevant immutable artifacts:

```text
atom/shock_atom_catalog.csv.gz
atom/shock_atom_manifest.json
episode/continuous_flow_episode_catalog.csv.gz
episode/flow_episode_phases.csv.gz
episode/episode_manifest.json
baseline/episode_residual_features.csv.gz
baseline/baseline_predictions.csv.gz
baseline/frozen_research_contract.json
motif/motif_prototypes.csv
motif/motif_membership.csv.gz
motif/walk_forward_stability.csv
motif/surrogate_test_results.csv
motif/motif_manifest.json
regime/one_minute_context.csv.gz
regime/regime_manifest.json
case_hierarchy_manifest.json
```

Every later implementation must fail closed when source paths, row counts,
schema versions, or SHA-256 identities differ from the accepted package.

### 2.2 Reusable Method Contracts

The following existing work may be reused as methodology and interface design:

- `docs/skhynix_cross_exchange_research_plan.md`;
- `docs/binance_led_hyperliquid_maker_data_input_contract.md`;
- `docs/cross_exchange_maker_mvp_plan.md`;
- `examples/hyperliquid/cross_exchange_lead_lag_join.py`;
- `examples/hyperliquid/cross_exchange_lead_lag_analysis.py`;
- `examples/hyperliquid/cross_exchange_shared_signal_kernel.py`.

The accepted BTC signal coefficients, normalization statistics, thresholds,
basis coefficients, and quote parameters must not be reused for SKHYNIX.

The reusable parts are limited to:

- same-host local-receipt as-of joining;
- explicit source-age tracking;
- no-future checks;
- nominal and effective horizon recording;
- pure decision-kernel structure;
- deterministic quote-intent and block-reason output;
- public-shadow and replay audit patterns.

## 3. Evidence Interpretation

### 3.1 Response Sign

The existing episode data direction-normalizes Hyperliquid markout:

- positive response means movement in the Binance shock direction;
- that movement is adverse for a hypothetical Hyperliquid passive quote on the
  vulnerable side;
- negative response means absorption, reversal, or less adverse movement than
  the conditional baseline.

Maker-side mapping:

| Binance shock | Directional risk | Vulnerable maker side |
| --- | --- | --- |
| aggressive buy | Hyperliquid price follows upward | passive ask |
| aggressive sell | Hyperliquid price follows downward | passive bid |

This mapping defines quote protection semantics. It does not prove fills.

### 3.2 Prototype Review Priors

The following grouping is a pre-registered research prior for rule extraction.
It is not a signal acceptance result.

| Family | Prototypes | Initial interpretation | Planned maker use |
| --- | --- | --- | --- |
| strong adverse-flow protection | `M0006` | large positive residual at both primary horizons | highest-priority vulnerable-side protection |
| directional continuation protection | `M0002`, `M0011`, `M0012` | repeated positive adverse response | side-specific cancel, size suppression, or widening |
| delayed adverse protection | `M0004` | weaker short response and stronger delayed response | quote-age and delayed cancel guard |
| cautious absorption/re-entry | `M0001`, `M0007`, `M0008` | negative residual at both primary horizons | delayed, small re-entry after no-follow-through |
| neutral controls | `M0009`, `M0014` | near-zero residual response | baseline quote-intent controls |
| watch or reject initially | `M0003`, `M0005`, `M0010`, `M0013` | broad, long, noisy, or near-neutral structure | no action until future evidence |

Additional interpretation constraints:

- `M0006` is the strongest defensive prior, not an automatic strategy.
- `M0012` and `M0008` are compact episodes and are useful candidates for
  early-prefix recognition.
- `M0003` and `M0013` are long or complex enough that a decision-time prefix
  may be ambiguous.
- `M0010` has a broad prototype-distance envelope and should default to
  out-of-distribution or watch-only treatment.
- `M0009` and `M0014` are controls; they must not be converted into aggressive
  size-increase rules merely because their average residual is near zero.

### 3.3 Statistical Boundary

The existing prototype review has already consumed information from segments
`0004-0008`. Those segments cannot be described as a fresh held-out test for
the new maker rules.

The new split terminology must be:

- `rule_discovery`: segments `0001-0003`;
- `historical_transfer_audit`: segments `0004-0008`;
- `fresh_session_validation`: future, previously unseen sessions or dates.

Only `fresh_session_validation` may support a formal out-of-sample maker-rule
claim.

## 4. Causal Clock Contract

Every future implementation must publish the following timestamps for each
candidate decision:

```text
shock_ts
-> decision_ts
-> prefix_evaluation_ts
-> latest_binance_source_ts
-> latest_hyperliquid_bbo_source_ts
-> latest_hyperliquid_fast_l2_source_ts
-> hypothetical_action_ts
-> future_label_source_ts
```

Definitions:

- `shock_ts`: first Binance trade that crosses the frozen shock threshold.
- `decision_ts`: first Binance depth state that confirms the shock.
- `prefix_evaluation_ts`: time at which the online prefix rule is evaluated.
- `hypothetical_action_ts`: decision output time after configured processing
  delay; it is not an exchange acknowledgement time.
- `future_label_source_ts`: source timestamp of the later response label.

Required inequalities:

```text
latest_binance_source_ts <= prefix_evaluation_ts
latest_hyperliquid_bbo_source_ts <= prefix_evaluation_ts
latest_hyperliquid_fast_l2_source_ts <= prefix_evaluation_ts
decision_ts <= prefix_evaluation_ts <= hypothetical_action_ts
future_label_source_ts > hypothetical_action_ts
```

Any violation rejects the row.

Lead-lag latency must be decomposed rather than reported as one number:

```text
shock confirmation lag
+ prefix observation delay
+ Hyperliquid source age
+ local decision processing delay
+ hypothetical action delay
= action-path information age
```

The plan does not assume the full market reaction remains available after this
delay. Freshness and effective-horizon evidence must decide that later.

## 5. Decision-Time Feature Contract

### 5.1 Allowed Binance Prefix Features

Only information visible at or before `prefix_evaluation_ts` may be used:

- current prefix atom count;
- current prefix cluster count;
- prefix duration;
- shock direction;
- direction persistence observed so far;
- signed and absolute cumulative shock impact observed so far;
- maximum observed individual shock impact;
- cumulative confirmed queue removal observed so far;
- inter-atom arrival times;
- time since latest same-direction atom;
- time since latest opposite-direction atom;
- reversal count confirmed so far;
- Binance midpoint displacement from episode start;
- Binance spread and top-5 depth state;
- continued aggressive-flow intensity;
- shock-confirmation lag;
- whether a frozen recovery checkpoint has already become observable.

Prefix fields must be recomputed from raw decision-time events. They must not be
copied from the completed `continuous_flow_episode_catalog.csv.gz` row when the
completed value includes future atoms.

### 5.2 Allowed Hyperliquid State Features

- BBO midpoint and spread;
- best bid and ask quantity;
- impacted-side and opposite-side quantity;
- BBO update flag since `decision_ts`;
- fast top-5 depth, imbalance, and microprice when fresh enough;
- latest first-response state only when its source timestamp is already visible;
- basis level or residual as context only;
- source age for every consumed channel;
- degraded, missing, duplicate, and no-new-information flags.

Source-age tiers inherit the SKHYNIX research contract:

| Track | Primary | Watch | Reject for decision use |
| --- | --- | --- | --- |
| Hyperliquid BBO | `<=250ms` | `250-500ms` | `>500ms` |
| Hyperliquid fast L2 | `<=500ms` | `500-1000ms` | `>1000ms` |
| Hyperliquid standard L2 | `<=3000ms` | `3000-6000ms` | `>6000ms` |

Standard L2 may provide slow liquidity context. It cannot trigger a
millisecond maker action.

### 5.3 Later Execution-State Inputs

When a public-shadow quote-intent kernel is eventually implemented, it may also
consume:

- current inventory and hard position cap;
- current working-order side and age;
- pending submit, cancel, or replace state;
- last action time and cooldown;
- post-only validity state;
- configured maximum quote distance and size.

These fields belong to the maker risk layer. They are not present in the
current public response-prototype evidence and must not be imputed.

### 5.4 Forbidden Runtime Inputs

The following may be used only as offline labels or diagnostics:

- final `motif_id`;
- final prototype distance;
- completed episode duration;
- completed atom or cluster count;
- atoms that occur after `prefix_evaluation_ts`;
- future recovery checkpoints;
- final phase count or final dominant direction;
- `h1000` or `h2000` response and residual fields;
- future markout, future spread, future depth, or future price movement;
- baseline predictions requiring intervening future flow;
- completed regime labels or boundaries discovered after the action;
- fill, fee, rebate, inventory transition, or PnL fields;
- any after-the-fact manual classification.

The runtime contract must contain an explicit allowlist. Unknown fields fail
closed.

## 6. Online Prefix Construction

### 6.1 Evaluation Checkpoints

Each shock episode is evaluated at causal checkpoints:

```text
T0 = decision_ts
T1 = decision_ts + 50ms
T2 = decision_ts + 100ms
T3 = decision_ts + 250ms
T4 = decision_ts + 500ms
```

An event-driven evaluation also occurs when:

- a new same-direction atom is confirmed;
- an opposite-direction atom is confirmed;
- a recovery checkpoint becomes visible;
- Hyperliquid BBO changes;
- a freshness tier changes to reject.

The primary maker candidate should act at the earliest checkpoint with adequate
classification confidence and fresh venue state. Later checkpoints may refine,
cancel, or expire the quote intent.

No checkpoint after `500ms` is intended as the initial fast lead-lag entry.
The accepted `1000/2000ms` horizons are outcome windows and possible defensive
cooldown windows.

### 6.2 Prefix Snapshot Output

The future prefix builder should emit one row per
`flow_episode_id x prefix_evaluation_ts` with:

- causal-clock fields;
- exact input source-row references;
- visible prefix features;
- missing and freshness flags;
- full-information prototype label in a separately marked training-label
  column;
- future response labels in a separate label table;
- leakage-audit result.

Feature and label tables should be physically separate so the runtime kernel
cannot accidentally read future columns.

## 7. Prototype Distillation

### 7.1 Objective

The distillation model answers:

```text
Given only the visible episode prefix, which prototype-conditioned rule family
is plausible enough to use, and when should the system reject classification?
```

It does not attempt to reproduce the final prototype ID at all costs.

### 7.2 Two-Stage Matcher

Stage A: mechanism rule envelope.

- deterministic constraints on shock direction, prefix atom count, duration,
  persistence, impact, reversal, and venue freshness;
- separate envelopes for defensive, absorption, neutral, and rejected
  structures;
- rules are readable and auditable;
- ambiguous overlap returns `no_action`.

Stage B: calibrated prototype-proximity model.

- trained only on `rule_discovery`;
- predicts prototype or rule-family probability from allowed prefix fields;
- simple regularized models are preferred;
- calibration is blocked by segment;
- confidence thresholds include an abstain class;
- no outcome label is used to improve prototype classification.

The initial implementation should compare:

1. deterministic quantile envelopes;
2. regularized multinomial logistic regression;
3. shallow gradient boosting with strict depth and minimum-leaf limits.

Model selection is based on blocked prefix-classification quality, calibration,
stability, and reject behavior. It is not selected by maker markout or PnL.

### 7.3 Abstention And Out-Of-Distribution Rules

Return `no_action` when any condition holds:

- source freshness is rejected;
- required prefix features are missing;
- two high-priority rule families conflict;
- maximum class probability is below the frozen threshold;
- probability margin between the top two families is too small;
- feature distance exceeds the discovery envelope;
- direction mapping is ambiguous;
- a long-flow or complex-flow signature appears;
- a new same-direction atom invalidates an absorption candidate;
- a risk or inventory guard blocks the action.

Coverage is secondary to correctness. The system must not force every episode
into one of the 14 prototypes.

## 8. Initial Maker Rule Hypotheses

All rules below produce quote intent only. They do not submit orders.

### 8.1 Defensive Priority

Defensive rules override neutral and re-entry rules.

`R-D1 / M0006 severe adverse-flow guard`

- trigger: high-confidence `M0006`-family prefix;
- action: cancel or suppress the vulnerable side;
- optional action: widen both sides when direction confidence is insufficient
  but adverse-flow confidence is high;
- expiry: frozen cooldown candidate inside the `1000-2000ms` response window;
- invariant: never increase vulnerable-side size.

`R-D2 / M0012 compact continuation guard`

- trigger: compact, often single-atom adverse prefix with fresh BBO evidence;
- action: protect the vulnerable side immediately;
- invariant: do not wait for the completed episode before acting.

`R-D3 / M0002-M0011 directional continuation guard`

- trigger: direction-persistent flow with prototype-conditioned continuation
  confidence;
- buy shock: suppress or widen the ask, optionally retain a constrained bid;
- sell shock: suppress or widen the bid, optionally retain a constrained ask;
- invariant: side mapping must be exactly mirrored in unit tests.

`R-D4 / M0004 delayed-adverse quote-age guard`

- trigger: `M0004`-family prefix or a quote surviving after a weaker initial
  response;
- action: shorten quote TTL and prevent a stale vulnerable-side quote from
  remaining through the delayed adverse window;
- invariant: this rule cannot create a new quote by itself.

### 8.2 Absorption And Re-Entry

Absorption candidates never act at `T0` solely from the offline prototype
prior. They require observable no-follow-through evidence.

`R-A1 / M0008 compact absorption re-entry`

- candidate after a compact shock;
- wait for `100-250ms` of causal evidence;
- require no new same-direction ShockAtom;
- require fresh Hyperliquid BBO;
- require no extension of Binance price in the shock direction;
- require stable or recovering Hyperliquid impacted-side liquidity;
- action: permit a small vulnerable-side re-entry quote intent;
- invariant: any renewed same-direction flow cancels the re-entry.

`R-A2 / M0007-M0001 multi-atom absorption re-entry`

- require a visible recovery checkpoint or equivalent frozen prefix evidence;
- require direction persistence to stop increasing;
- require no adverse freshness or degraded flags;
- action: staged re-entry at reduced size;
- invariant: defensive rules and inventory guards override it.

Re-entry means restoring a passive quote after protection. It does not mean
crossing the spread or taking the opposite position with a market order.

### 8.3 Neutral Controls

`R-N1 / M0009-M0014 neutral control`

- preserve the baseline maker quote-intent policy;
- do not add directional skew solely from prototype classification;
- do not increase size above the frozen baseline;
- record the row as a control for later comparison.

### 8.4 Watch And Reject Family

`M0003`, `M0005`, `M0010`, and `M0013` initially map to:

```text
watch_only_or_no_action
```

They may be reconsidered only when fresh-session evidence shows:

- stable early-prefix recognition;
- stable direction and response;
- no dependence on one session;
- a maker-relevant improvement over the neutral control.

## 9. Quote-Intent Action Space

The future pure kernel may emit only the following research actions:

```text
protect_bid
protect_ask
widen_both
reduce_bid_size
reduce_ask_size
hold_neutral
reentry_bid
reentry_ask
cancel_reentry
no_action
```

Each output must include:

- `rule_id`;
- parent prototype family;
- shock direction;
- classification confidence;
- evaluation timestamp;
- action;
- action intensity level;
- expiry or cooldown;
- block reason;
- feature-contract SHA;
- rule-contract SHA;
- source artifact SHA.

Candidate action-intensity grids may include:

- size multipliers: `0`, `0.25`, `0.5`, `1.0`;
- widening: `0`, `1`, `2`, `4` ticks;
- re-entry delay: `100`, `250`, `500ms`;
- cooldown or TTL: `250`, `500`, `1000`, `2000ms`.

These are shadow-evaluation candidates, not accepted parameters. Parameters
must be frozen before fresh-session outcome evaluation.

## 10. Rule Priority And Conflict Resolution

The kernel must use a deterministic priority:

```text
data-quality reject
-> hard risk / inventory block
-> pending order-lifecycle block
-> severe defensive guard
-> directional defensive guard
-> delayed quote-age guard
-> cancel re-entry
-> absorption re-entry
-> neutral control
-> no_action
```

Required conflict behavior:

- defensive and re-entry signals on the same side resolve to defense;
- ambiguous direction resolves to `widen_both` only when severe adverse-flow
  confidence passes its own frozen threshold, otherwise `no_action`;
- an existing cancel or replace in flight prevents a duplicate action;
- inventory rules may reduce or veto an action but may not reverse the
  prototype-conditioned direction;
- rule output must be idempotent for identical input state.

## 11. Historical Extraction And Audit

### 11.1 Rule Discovery

Use only segments `0001-0003` to:

- construct prefix features;
- fit prototype-distillation models;
- select model family;
- set probability, margin, and out-of-distribution thresholds;
- define rule envelopes;
- freeze freshness, latency, conflict, and expiry contracts.

Use leave-one-segment-out blocked validation inside discovery.

Prototype labels may supervise distillation. Future markout, residuals, and
maker proxies may not select the classifier.

### 11.2 Historical Transfer Audit

Segments `0004-0008` may be used once after freeze to report:

- causal rule coverage;
- abstain and out-of-distribution rate;
- prototype-family confusion;
- rule conflicts;
- side symmetry;
- source-age distribution;
- action latency;
- future adverse movement by rule;
- neutral-control comparison;
- sensitivity to hypothetical quote delay.

This output must be labelled:

```text
historical_transfer_audit
```

It must not be labelled `held_out`, because these segments have already been
reviewed in the prototype study.

### 11.3 Historical Acceptance Classification

Each rule receives one result:

- `candidate_for_future_public_shadow`;
- `watch_only_needs_fresh_sessions`;
- `reject_prefix_not_identifiable`;
- `reject_no_maker_relevance`;
- `reject_data_quality_dependent`.

No current historical result can be `strategy_validated`.

## 12. Public Counterfactual Maker Evaluation

The first maker evaluation remains public-only and no-submit.

### 12.1 Defensive Rule Metrics

- vulnerable-side future adverse markout;
- maximum adverse excursion;
- frequency of touch, trade-through, or queue depletion proxies;
- avoided quote-exposure duration;
- quote-cancel or widening frequency;
- missed spread opportunity;
- action age at the first Hyperliquid response;
- effect versus neutral-control rows matched on spread, depth, and freshness.

The primary question is:

```text
Does the guard reduce adverse passive exposure without suppressing nearly all
quoting?
```

### 12.2 Re-Entry Rule Metrics

- re-entry candidate count;
- renewed same-direction shock rate after re-entry;
- post-re-entry adverse markout;
- quote-survival proxy;
- touch and trade-through proxy;
- gross spread-capture range;
- fee and adverse-selection buffer sensitivity;
- comparison with immediate re-entry, delayed re-entry, and no re-entry.

Current public data cannot establish exact fills. Report optimistic, base, and
conservative ranges separately.

### 12.3 Lead-Lag Diagnostics

For every rule family report:

- Binance shock-to-decision lag;
- decision-to-rule lag;
- rule-to-first-Hyperliquid-BBO-response lag;
- effective information age;
- response by BBO and fast-L2 freshness tier;
- action outcome at `100`, `250`, `500`, `1000`, and `2000ms`;
- reverse-direction diagnostic where feasible;
- duplicate/no-new-Hyperliquid-information rows separately.

The result should establish whether the maker rule acts before the relevant
Hyperliquid response, not merely whether it correlates with a later outcome.

## 13. Fresh-Session Validation Loop

Fresh data collection is a separate future task and requires explicit
authorization. This plan does not start it.

Minimum preferred campaign:

- at least three independent future sessions;
- sessions separated by date or meaningful market-state interval;
- the same Binance `SKHYNIXUSDT` and Hyperliquid `xyz:SKHX` identity;
- unchanged event, timestamp, and source-age contracts;
- raw public artifacts retained with checksums;
- public-only and no-submit for the first validation stage.

Frozen-rule evaluation order:

1. run the unchanged prefix builder;
2. run the unchanged rule classifier;
3. record quote intents and blocks;
4. wait until the session closes;
5. build future labels;
6. evaluate each session independently;
7. aggregate only after per-session results are published.

A rule may advance only when:

- no-future violation count is zero;
- classification and action code are unchanged from the frozen contract;
- the intended direction is consistent in at least two adjacent horizons;
- the effect has the same sign in at least two of three independent sessions;
- no single session contributes more than `50%` of aggregate effect;
- primary evidence does not depend on stale or degraded rows;
- defensive rules reduce adverse exposure relative to neutral controls;
- re-entry rules do not become systematically adverse after conservative cost
  and latency buffers;
- action coverage is non-zero but abstention remains available;
- side mapping remains symmetric after conditioning on market direction.

Failure leads to rule modification and a new version. Failed sessions remain
part of the evidence and may not be silently excluded.

## 14. Promotion Boundary

### Stage P0: Contract And Prefix Dataset

Outputs only causal clocks, feature allowlists, prefix rows, labels, and leakage
audits.

Maximum conclusion:

```text
prefix_contract_valid
```

### Stage P1: Prototype Distillation

Outputs deterministic rule envelopes, model comparison, calibration, and
abstention behavior.

Maximum conclusion:

```text
prototype_prefix_recognizable
```

### Stage P2: Historical Transfer Audit

Outputs provisional maker actions and public counterfactual metrics on the
existing campaign.

Maximum conclusion:

```text
candidate_for_future_public_shadow
```

### Stage P3: Multi-Session Public Shadow

Consumes new independent public sessions with frozen rules.

Maximum conclusion:

```text
candidate_for_replay_and_execution_evidence
```

### Stage P4: Replay And Execution Evidence

Requires a separate accepted audit/replay task and actual execution-lifecycle
evidence.

Maximum conclusion:

```text
candidate_for_controlled_tiny_live
```

### Stage P5: Controlled Tiny Live

Requires explicit live authorization, independent risk review, post-only
constraints, position caps, tracked cancel, and final open-orders proof.

Only this stage may begin to collect fill, fee, inventory, and realized-PnL
evidence. It is outside this plan's current authorization.

## 15. Planned Artifact Contract

A future implementation should publish under a task-specific root:

```text
local_live_analysis/
  skhynix_prototype_conditioned_lead_lag_rules_<TASK_ID>/
    contract/
      source_identity_manifest.json
      causal_clock_contract.json
      online_feature_contract.json
      forbidden_feature_contract.json
      prototype_rule_hypotheses.csv
      rule_priority_contract.json
    prefix/
      prefix_features.csv.gz
      prefix_training_labels.csv.gz
      future_outcome_labels.csv.gz
      prefix_leakage_audit.csv.gz
      prefix_manifest.json
    extraction/
      rule_candidate_catalog.csv
      classifier_calibration.csv
      classifier_confusion.csv
      out_of_distribution_audit.csv.gz
      rule_conflict_audit.csv.gz
      frozen_rule_contract.json
      extraction_manifest.json
    historical_audit/
      rule_decisions.csv.gz
      rule_horizon_metrics.csv
      neutral_control_comparison.csv
      source_age_stability.csv
      latency_stress.csv
      historical_transfer_manifest.json
    shadow/
      public_shadow_contract.json
      quote_intents.csv.gz
      counterfactual_metrics.csv
      session_summary.csv
      shadow_manifest.json
    boundary_manifest.json
```

The `shadow/` directory is produced only in a later separately authorized
task.

## 16. Verification And Failure Injection

Every implementation stage must test:

- a future atom injected into a prefix;
- final prototype ID supplied to the runtime kernel;
- a future markout column added to the feature table;
- source timestamp after evaluation timestamp;
- missing source timestamp;
- stale BBO and fast-L2 rows;
- changed source manifest SHA;
- duplicate episode or prefix identity;
- segment-boundary crossing;
- buy/sell side inversion;
- defense and re-entry conflict;
- ambiguous top-two class probabilities;
- feature vector outside discovery support;
- repeated identical kernel input;
- pending-cancel and pending-replace states;
- inventory cap breach;
- empty or zero-action output;
- no new Hyperliquid information inside a nominal horizon;
- historical segments incorrectly labelled as fresh held-out;
- attempted order/private endpoint use in public-shadow mode.

Required invariants:

- future-feature count is zero;
- cross-segment leakage count is zero;
- runtime feature allowlist violations are zero;
- identical fixtures produce identical outputs;
- public-shadow order endpoint call count is zero;
- every decision has an explicit action or block reason;
- every published count reconciles with its source manifest.

## 17. Formal Task Sequence

If this plan is approved for execution later, use one formal task at a time:

1. `P0` causal clock, prefix feature, and leakage contract.
2. `P1` prototype-prefix dataset and distillation.
3. `P2` frozen maker rule catalog and historical transfer audit.
4. `P3` pure public-shadow quote-intent kernel.
5. `P4` future multi-session public collection and shadow validation.
6. `P5` replay and execution-evidence design.
7. `P6` optional controlled tiny-live proposal after all earlier QA gates.

Each task must follow:

```text
业务线程 -> QA验收线程 -> 总控
```

No later task is automatically authorized by an earlier task's existence.
Fresh collection, private endpoints, live-submit behavior, or risk-envelope
changes require their own explicit scope and approval.

## 18. Final Decision Rule

The present 14 prototypes are useful for extracting maker hypotheses because
they organize repeated cross-venue response structures. They are not directly
tradable because their final identity is known only after observing the
episode.

The viable approach is therefore:

```text
prototype as offline teacher
-> causal prefix as online state
-> abstaining rule classifier
-> defensive maker protection first
-> delayed absorption re-entry second
-> neutral controls and explicit rejects
-> future-session public validation
-> versioned iteration
```

The most defensible first strategy family is quote protection around
`M0006`, `M0012`, `M0002`, and `M0011`. The absorption family
`M0001`, `M0007`, and `M0008` is a second-stage re-entry hypothesis and must
wait for observable no-follow-through evidence. Neutral and complex prototypes
remain controls or no-action cases.

Until new independent sessions exist, the output can be a coherent,
mechanism-constrained maker rule catalog, but not a validated maker strategy.
