# SKHYNIX Trigger-Aligned Episode Research Implementation Plan

Date: 2026-08-14

Status: active controller goal since 2026-08-15; staged execution only.
No collection, private endpoint, order, cancel, deployment, or live
authorization is granted by this document.

Revision: 2026-08-14 review remediation; adds Candidate-aligned Family A,
Confirmed-shock Family B, trigger-density admission, interval-censored
Hyperliquid timing, underlying-market regimes, historical feed limitations,
pre-registered scoring, and per-feature observation-time invariants

## Execution Goal and Stage Governance

This complete document is the controller-level research goal. Its Ordered
Research Queue is executed as a gated sequence, not as one unbounded
implementation task.

The mandatory chain for every stage is:

```text
one formal workflow task
  -> business execution reaches 待验收
  -> independent QA reaches 已通过 / 未通过 / 阻塞
  -> controller decides whether the next stage is unlocked
```

Governance invariants:

- only one stage is the current formal task unless the plan explicitly proves
  that later work is independent and cannot consume gated information;
- every stage receives a unique task ID and a task file under
  `.workflow/tasks/`;
- the business thread may implement and self-test but may not accept its own
  result;
- the independent QA thread may inspect and verify but may not repair business
  code or silently widen scope;
- only an `已通过` QA report unlocks the next Ordered Research Queue item;
- `未通过` returns the same stage for bounded repair and independent re-QA;
- `阻塞` preserves all downstream locks until the controller records a
  resolved prerequisite;
- every QA result is written under `.workflow/reports/` and copied to
  `docs/qa-acceptance-report.md`;
- no stage may open a later session, outcome, feature, model, or actionability
  surface before the preceding freeze/consumption gate permits it;
- the overall goal is complete only after the final research classification
  receives independent QA. Completion of one stage is not completion of this
  plan.

## 1. 研究目标

本方案使用现有 SKHYNIX Binance/Hyperliquid public-data datasets，建立一套
以 trigger 为统一 event-time 原点的 episode case library，并研究：

\[
P\left(
R_i,\ O_i^{market}
\mid
S_i^{pre},\ T_i
\right)
\]

其中：

- \(S_i^{pre}\)：trigger 前的双交易所市场状态；
- \(T_i\)：Binance source-side queue shock 的严重程度和具体结构；
- \(R_i\)：trigger 后 Binance、Hyperliquid 和跨所价差的响应路径；
- \(O_i^{market}\)：未来价格、危险侧 quote 状态和公开市场结果。

本方案的核心不是寻找一个固定结果，而是回答：

```text
在当前 Spre 和 T 下，
历史上最相似的 episodes 后续出现了哪些路径，
这些路径的概率、分位数、时间分布和尾部风险是什么？
```

第一阶段只研究 public-data 可支持的市场条件分布。因为所有现有数据都
没有本账户下单和成交，本方案明确不估计：

\[
P(O_i^{own}(a)\mid S_i^{pre},T_i)
\]

也不声称真实 fill probability、fee、inventory、PnL 或
KEEP/CANCEL_RISK_SIDE 的因果 EV。

## 2. Trigger 的研究角色

Trigger 首先是时间配准机制，其次才是预测特征。

设双交易所连续市场过程为：

\[
X(t)=\left(X_B(t),X_H(t)\right)
\]

第 \(i\) 个 trigger candidate 的时刻为 \(\tau_i^0\)，定义：

\[
u=t-\tau_i^0
\]

对齐后的 episode path 为：

\[
\widetilde X_i(u)=X(\tau_i^0+u),
\qquad u\in[-L,H]
\]

所有 episode 因此具有相同的相对时间语义：

```text
u < 0   trigger 前状态
u = 0   candidate threshold 首次达到
u > 0   trigger 演化、target response 和市场结果
```

本研究必须分别保存：

```text
t_burst_start
t_candidate
t_confirm
t_research_decision
t_first_target_response
t_first_adverse_target_event
```

v1 的主时间原点固定为：

```text
u = 0 := t_candidate := existing shock_ts_ns
```

v1 的 research decision timestamp 固定为：

```text
t_research_decision := t_confirm := existing decision_ts_ns
```

因此，在决策时可使用：

\[
\left(
S_i^{pre},
T_i^{0:d_i},
R_i^{0:d_i}
\right),
\quad
d_i=t_{confirm}-t_{candidate}
\]

不能把 confirmation 之后的路径用于模拟 candidate-time 决策。

## 3. 当前可用数据清单

### 3.1 主研究数据

| Session | Duration | Segments | Current episode evidence | Research role |
| --- | ---: | ---: | ---: | --- |
| Jul30 | 4h | 8 | 268,522 candidates / 141,768 primary | discovery and internal validation |
| Aug03 | 5h | 10 | 127,622 candidates / 82,533 primary | historical transfer and robustness |
| Aug04 | 2h | 1 continuous | 67,468 candidates / 43,253 primary | historical validation, already consumed |
| Aug07 | 4h | 1 continuous | episode v3 not built | frozen-method retrospective evaluation |

Canonical sources:

```text
Jul30 raw:
local_live_analysis/cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m/

Jul30 R0/R1:
local_live_analysis/skhynix_cross_exchange_research_0730T013/

Jul30 historical episodes:
local_live_analysis/skhynix_liquidity_response_0730T017/

Aug03 raw:
local_live_analysis/cross_exchange_collection_campaign_0802T001_skhynix_5h_10x30m/

Aug03 R0:
local_live_analysis/skhynix_cross_exchange_research_0803T001/

Aug03 current R1 replay:
local_live_analysis/skhynix_cross_exchange_research_0804T001_old5h_replay/alignment/

Aug03 historical episodes:
local_live_analysis/skhynix_liquidity_response_0803T002/

Aug04 raw:
local_live_analysis/cross_exchange_collection_campaign_0804T001_skhynix_2h_continuous/

Aug04 R0/R1:
local_live_analysis/skhynix_cross_exchange_research_0804T001/

Aug04 historical episodes:
local_live_analysis/skhynix_liquidity_response_0804T008/

Aug07 immutable raw:
local_live_analysis/0807T001_skhynix_4h_continuous/

Aug07 full accepted R0/R1/basis on amdserver:
/home/molly/project/hftbacktest/local_live_analysis/0807T002_skhynix_basis_postprocess/

Aug07 local compact evidence:
local_live_analysis/0807T002_skhynix_postprocess_compact/
```

Aug07 accepted facts include:

- common L2 timeline: `501,220` rows;
- Binance hot events: `3,998,895`;
- Hyperliquid hot events: `207,634`;
- Hyperliquid auxiliary events: `42,704`;
- R1 labels: `714,063`;
- point-in-time basis state rows: `2,751,082`;
- feature-eligible basis rows: `1,976,897`;
- source inventory SHA256:
  `a30654f32cea375c49e4b26e39ed3c0de436e6d1f08de959601db749f262a003`.

### 3.2 Required session metadata

Before any episode outcome or model gate is inspected, `data_admission.md`
must publish one row per session containing:

- collection host, region/availability-zone class where known, process/runtime
  identity, collector source SHA, and local receipt clock source;
- `collection_topology_fingerprint`, derived from acquisition-time facts
  rather than from the later storage or postprocess host;
- research execution host and artifact location as a separate field;
- Binance and Hyperliquid channel inventory and observed inter-event cadence;
- Hyperliquid BBO/trade/fast-L2 source-age and inter-arrival p01/p10/p50/p90/p99;
- collection start/end in UTC, Asia/Shanghai, and the versioned underlying
  market calendar timezone;
- SK Hynix underlying-market regime coverage and duration;
- trigger count, trigger intensity, inter-trigger distribution, flow-cluster
  count, and estimated effective sample size.

Moving files from the collection host to local storage or amdserver does not
change historical local-receipt timestamps. It must not be described as an
acquisition-topology change. Timing outcomes may be transferred directly
across sessions only when their acquisition topology fingerprints and feed
cadence baselines are comparable; otherwise results must be stratified or
carry an explicit topology-confounding caveat.

### 3.3 Conditional input

Aug06 is not admitted into the primary study at plan freeze time:

```text
AMD raw:
/home/molly/project/hftbacktest/local_live_analysis/0806T001_skhynix_2h_continuous/

inventory SHA256:
a85e314ce9e2cb2f3867289ba9108c19129402c8614c02f9e1be43fba695af52
```

Its collection and transfer evidence is complete, but the formal task remains
`待验收` and no accepted R0/R1/basis package is currently bound to this plan.
It may enter only after:

1. collection QA is accepted;
2. immutable raw audit passes;
3. the current versioned postprocess pipeline produces accepted R0/R1/basis;
4. the episode contract is already frozen;
5. its role is recorded as `late_admitted_retrospective_evaluation`.

### 3.4 Smoke-only inputs

Jul29 60s/30m/2h public samples and research-max probes may be used for CLI,
schema, failure-injection, and runtime smoke tests only. They must not
contribute to effect estimates or model selection.

## 4. Evidence Classification

These datasets already exist and several have been inspected by prior
research. The study must not relabel them as prospective held-out evidence.

Use only these labels:

```text
historical_discovery
historical_internal_validation
historical_transfer
historical_consumed_validation
retrospective_method_holdout
late_admitted_retrospective_evaluation
smoke_only
```

Proposed roles:

| Dataset | Label |
| --- | --- |
| Jul30 segments 0001-0004 | historical_discovery |
| Jul30 segments 0005-0008 | historical_internal_validation |
| Aug03 | historical_transfer |
| Aug04 | historical_consumed_validation |
| Aug07 | retrospective_method_holdout |
| Aug06, if later admitted | late_admitted_retrospective_evaluation |

Aug07 may be kept unopened by the new episode/model code until the v1 freeze
manifest is written. This improves method discipline, but it does not turn
Aug07 into a fresh population-level experiment because aggregate Aug07 basis
and maker results are already known.

## 5. Primary Trigger Family

The primary episode family remains the existing Binance aggressive-trade queue
shock:

```text
same-side consecutive Binance trades within 10ms of burst start
-> cumulative touch quantity reaches 30% of strict pre-shock best queue
-> Candidate at shock_ts
-> Binance depth confirms depletion or <=70% remaining within 100ms
-> Confirmed at decision_ts
```

Direction mapping:

| Source shock | Direction sign | Vulnerable SKHX maker side |
| --- | ---: | --- |
| Binance aggressive buy / ask depletion | +1 | maker ask |
| Binance aggressive sell / bid depletion | -1 | maker bid |

The existing trigger audit remains authoritative for:

- every candidate;
- rejection reason;
- attribution class;
- deduplication;
- confirmation reuse;
- missing pre-state;
- insufficient response room.

No threshold may change inside episode v3. Sensitivity runs use separately
named contracts and cannot replace the primary result.

### 5.1 Historical Binance feed reality

The existing SKHYNIX campaigns were collected from:

```text
wss://fstream.binance.com/ws

SKHYNIXUSDT@trade
SKHYNIXUSDT@depth@0ms
SKHYNIXUSDT@bookTicker
```

They were not collected from `aggTrade`, and the archived `trade` payloads
contain ordinary quantity `q` but no `nq` field or equivalent RPI
participation marker. Historical data therefore cannot separate normal
visible-book-related quantity from trades involving liquidity that is absent
from the public displayed queue.

For this study:

- the numerator must be named `observed_trade_qty`, not `nq` or
  `visible_consumed_qty`;
- the ratio must be described as
  `observed_trade_qty_to_visible_prequeue_ratio`;
- `rpi_adjustment_status=unavailable_historical_feed`;
- the existing `shock_impact_ratio` field remains a legacy compatibility
  alias and cannot be interpreted as exact visible-queue consumption;
- depth timing must be described from the actual `@depth@0ms` archive and its
  measured arrival cadence, not from an assumed `100ms` feed contract.

Depth confirmation remains valuable because it independently requires an
observed visible-book reduction. It does not remove the numerator/denominator
measurement mismatch, so every severity and dose-response result must retain
this limitation.

## 6. Cross-Exchange Spread Contract

Define:

\[
d_{bh}=bid_{Binance}-ask_{Hyperliquid}
\]

\[
d_{hb}=bid_{Hyperliquid}-ask_{Binance}
\]

For the vulnerable maker side, define a direction-normalized risk gap:

\[
g_i^{risk}=
\begin{cases}
d_{bh}, & \text{aggressive buy / maker ask risk}\\
d_{hb}, & \text{aggressive sell / maker bid risk}
\end{cases}
\]

The primary queue-shock slice does not add a cross-spread threshold as a
second trigger. Cross-spread enters:

- \(S^{pre}\) as pre-trigger level, residual, age, velocity, and regime;
- \(R\) as the event-time gap path and its Binance/Hyperliquid leg
  decomposition;
- \(O^{market}\) as survival, closure, overshoot, and target adverse movement.

The research must compare the pre-registered nested model families in
Section 12:

```text
Family A Candidate-time: A0 -> A1 -> A2 -> A3
Family B Confirmed-time: B0 -> B1 -> B2 -> B3 -> B4
```

This directly answers whether queue-shock structure adds information beyond
the already strong cross-exchange spread signal without mixing Candidate-time
and Confirmed-time feature availability.

A separate cross-spread-triggered episode family may be planned later. It must
have a different trigger ID, contract, sample inventory, and report; it may not
be pooled with queue-shock episodes under one event origin.

## 7. Episode v3 Data Model

Episode v3 is one append-only candidate record with two linked landmark views.
Family A and Family B must not be built as unrelated datasets.

The shared record is:

\[
E_i=
\left(
\mathcal T_i,
S_i^{pre},
T_i^{prefix},
R_i^{prefix},
R_i^{future},
O_i^{market},
C_i
\right)
\]

where \(C_i\) contains censoring, overlap, data quality, and confirmation
status. No action or own-order outcome is fabricated.

### 7.1 Family A: Candidate-aligned policy view

Family A contains every candidate in the frozen trigger audit:

- later confirmed candidates;
- no-depth-confirmation candidates;
- mixed/uncertain attribution candidates;
- confirmation-reuse and same-direction-dedup exclusions;
- candidates censored by segment end, reconnect, missing state, or insufficient
  outcome room.

Every Family A candidate must receive:

- the Candidate-aligned fixed-grid path where observable;
- an outcome row with event-time intervals, availability, and censoring;
- confirmation probability target and final confirmation/rejection reason;
- Candidate-time feature ledger containing only information observed by
  `t_candidate`.

The primary Candidate-time estimand is:

\[
P\left(
R^{future},O^{market},confirmed
\mid
candidate,S^{pre},T^{candidate},R^{candidate}
\right)
\]

It must not be estimated only on confirmed rows. Otherwise it would estimate:

\[
P(O\mid candidate,confirmed=1)
\]

instead of the live-observable:

\[
P(O\mid candidate)
\]

All Family A outcome summaries must be built for the complete candidate
population. To avoid event duplication, exact sparse paths may reference a
shared immutable event store through membership indexes, but storage pressure
may not justify dropping rejected candidates. Rendered casebook examples may
use a frozen deterministic hash sample; model and outcome tables may not.

### 7.2 Family B: Confirmed-shock research view

Family B is a filtered landmark view over the same candidate IDs:

```text
confirmed = true
quality and outcome eligibility recorded explicitly
t_research_decision = t_confirm
```

It is used for Confirmed-time dose-response, response-path, cross-spread
closure, and public actionability research:

\[
E_i^{B}
=
E_i^{A}
\mid
(confirmed_i=1)
\]

Family B may further report clean attribution or quality subsets, but the
complete confirmed population remains visible and subset reasons are fields,
not silent filters.

Candidate-time conclusions use Family A. Confirmed-time conclusions use
Family B. The study must publish both:

\[
P(O\mid candidate)
\]

and:

\[
P(O\mid candidate,confirmed=1)
\]

so confirmation selection can be measured rather than hidden.

### 7.3 Anchor table

One row per candidate:

```text
campaign_id
session_id
segment_id
episode_id
flow_cluster_id
candidate_seq
direction_sign
t_burst_start_ns
t_candidate_ns
t_confirm_ns
confirmation_lag_ns
family_a_available
family_b_available
confirmed
classification
rejection_reason
censor_time_ns
censor_reason
quality_flags
source_manifest_sha256
```

`t_confirm_ns` is nullable for rejected or censored candidates. No artificial
confirmation timestamp may be filled in.

### 7.4 Exact sparse path

Preserve every relevant source event with:

```text
episode_id
relative_time_ns
absolute_receive_ts_ns
venue
channel
event_type
connection_epoch
source_age_ns
direction-normalized price/depth fields
raw_event_identity
```

This is the information-complete event-time view. Repeated forward-filled
states are not new events.

### 7.5 Fixed event-time grid

Build a reporting/model grid at:

```text
-2000, -1000, -500, -250, -100, -50, -25, -10,
0, 10, 25, 50, 100, 250, 500, 1000, 2000 ms
```

Every grid value must include:

- value;
- `observed_at_ns`;
- source event ID and source book version;
- strict-as-of source timestamp;
- effective relative time;
- source age;
- no-new-information flag;
- connection epoch and degraded mask;
- whether the requested point remains inside the segment.

Do not count repeated grid states as independent observations.

### 7.6 Event-count view

For market-intensity robustness, also record the first:

```text
1, 2, 3, 5, 10, 20
```

Hyperliquid BBO events, trades, and fast-L2 events after Candidate and after
Confirmed. This separates physical-time latency from market-event sequencing.

### 7.7 Feature observation ledger

Every decision feature, including derived and bucketed fields, must have:

```text
episode_id
family_view
decision_landmark
feature_name
value
observed_at_ns
source_event_id
source_book_version
calculation_version
availability_reason
```

The builder must assert for every non-null decision feature:

\[
observed\_at(feature)\le t_{decision}
\]

For Family A, `t_decision=t_candidate`. For Family B,
`t_decision=t_confirm`. This is a complete row/feature invariant, not only a
hostile-test scenario. Any violation fails the entire candidate build before
publication.

## 8. Feature Contract

### 8.1 \(S^{pre}\): trigger 前状态

All fields must be available strictly before `t_candidate`.

Cross-venue:

- `d_bh_bps`, `d_hb_bps`, and direction-normalized `risk_gap_bps`;
- trailing basis residual and robust z-score;
- `10/25/50/100/250/500ms` trailing gap changes;
- Binance and Hyperliquid contribution to recent gap formation;
- both venues' BBO ages, connection epochs, and update flags.

Binance:

- best bid/ask, spread, midpoint, and tick-normalized spread;
- impacted and opposite best queue;
- top-5 impacted/opposite depth and imbalance;
- trailing signed trade flow and OFI;
- recent queue replenishment/depletion;
- recent volatility and update intensity;
- same/opposite shock counts in trailing windows.

Hyperliquid:

- BBO, spread, impacted/opposite best quantity;
- fast-L2 top-5 depth and imbalance;
- recent BBO/trade/fast-L2 event intensity;
- recent directional midpoint movement;
- source age and no-new-information state.

Context:

- session ID and time-of-session;
- versioned `underlying_market_state` for the SK Hynix KRX spot market;
- time since the last underlying-market state transition;
- collection topology fingerprint and Hyperliquid feed-cadence regime;
- trailing volatility/spread/liquidity buckets;
- reconnect/degraded masks;
- recent continuous-flow density.

`underlying_market_state` is a first-class conditioning field, not a
presentation-only label. The versioned calendar contract must support at
least:

```text
pre_open_or_auction
continuous_trading
closing_or_post_close
closed
holiday_or_special_session
unknown_calendar_state
```

The exact mapping and timezone are frozen before outcome access. If the
calendar is unavailable or ambiguous, the value is
`unknown_calendar_state`; it may not be inferred from future price behavior.
The data-admission report must publish duration, candidate count, confirmed
count, flow-block count, and outcome coverage by underlying state.

All primary results must be stratified by `underlying_market_state`. If all
datasets cover only one state, the report must say that underlying-regime
generalization was not tested. If sessions cover different states, the report
must say that session effects and underlying-state effects are partially
confounded unless both appear within enough sessions for a within-state
comparison.

Own inventory, own order, queue position, fee, and fill fields do not exist in
these datasets and must not be imputed.

### 8.2 \(T\): trigger severity and structure

At Confirmed-time research decision:

- aggressor direction;
- pre-queue quantity;
- cumulative touch quantity at Candidate and Confirmed;
- shock impact ratio;
- burst trade count, quantity, and duration;
- price levels touched or crossed;
- queue drop ratio and depleted flag;
- confirmed removed quantity;
- trade-explained ratio and attribution class;
- confirmation lag;
- post-Candidate trade continuation observed before Confirmed;
- local source-event count between Candidate and Confirmed.

Every quantity-based trigger feature must also carry:

```text
trade_quantity_source = binance_trade_q
rpi_adjustment_status = unavailable_historical_feed
depth_stream = depth@0ms
```

Candidate-time diagnostic models may use only fields available at Candidate
and must have a separate feature allowlist.

### 8.3 Early \(R\) prefix

For delayed-decision diagnostics, build explicit prefixes ending at:

```text
Confirmed
Candidate + 25ms
Candidate + 50ms
Candidate + 100ms
```

Only target events received by the prefix endpoint may enter. A prefix that
extends beyond Confirmed is not part of the main Confirmed-time model.

## 9. Response and Outcome Contract

### 9.1 \(R\): aligned market response

For Binance and Hyperliquid, record:

- bid, ask, midpoint, spread, and impacted-side depth path;
- first BBO response;
- target impacted-side withdrawal, retreat, replenishment, and follow;
- target trades at, through, or away from the vulnerable quote;
- direction-normalized midpoint and microprice path;
- cross-spread path \(g^{risk}(u)\);
- gap closure through Binance leg versus Hyperliquid leg;
- source ages and no-new-information intervals.

### 9.2 \(O^{market}\): public-data outcomes

Primary outcomes:

- time to first adverse target BBO event;
- time to first target trade at or through the vulnerable pre-trigger quote;
- time to first target impacted-side price retreat;
- gap survival at `100/250/500/1000/2000ms`;
- direction-normalized target midpoint markout at
  `250/500/1000/2000ms`;
- maximum adverse excursion over `0-2000ms`;
- maximum favorable excursion over `0-2000ms`;
- target-leg and source-leg contribution to gap closure;
- whether the adverse event occurs before Confirmed.

### 9.3 Hyperliquid timing resolution and interval censoring

Hyperliquid public BBO, trades, and fast-L2 are discrete published
observations. The true market transition can occur between two observed
messages. Therefore event-time outcomes are not exact point timestamps.

For an adverse state first observed at \(t_{obs}\), with the last qualifying
non-adverse observation at \(t_{prev}\), record:

\[
T_{adverse}\in(t_{prev},t_{obs}]
\]

If the event is not observed before the episode horizon, record right
censoring. If data quality fails before the horizon, record quality censoring
at the failure time rather than treating the event as absent.

Per session and channel, publish:

- inter-arrival p01/p10/p50/p90/p99 and maximum;
- source-age p01/p10/p50/p90/p99 and maximum;
- repeated/no-new-information duration;
- point-observed, interval-censored, right-censored, and quality-censored
  outcome counts;
- the fraction of latency scenarios whose before/after ordering is identified
  rather than ambiguous.

The `25ms` and `50ms` scenarios are retained as diagnostics, but they are
eligible for an identified timing claim only when the session's observed feed
resolution supports that distinction. A latency bucket smaller than the
effective observation interval must be labelled
`unresolved_below_feed_resolution`, not positive or negative actionability.

Timing results may be pooled across sessions only inside the same acquisition
topology/fingerprint stratum or after a pre-registered topology adjustment.
Research execution on local or amdserver is recorded separately and does not
repair acquisition-time topology differences.

### 9.4 Public quote-risk proxies

```text
public_trade_reaches_quote
public_bbo_moves_through_quote
public_quote_survives_horizon
public_adverse_exposure
```

These fields describe public-market evidence only. Their names must not
contain `fill`, `filled`, `execution_pnl`, or `own_order`.

## 10. Episode Dependence and Effective Sample Size

The existing datasets contain many overlapping shocks:

- Jul30 contains `268,522` candidates and `141,768` primary confirmed events
  in four hours, approximately `18.65` candidates/second and `9.85` primary
  events/second;
- Jul30: only `1` episode isolated at `2000ms`;
- Aug03: `191` isolated at `2000ms`;
- Aug04: `6` isolated at `2000ms`.

Therefore row count is not the statistical sample size.

Before any response, model, or gate result, `data_admission.md` must publish:

- candidates and confirmed events per second/minute/hour;
- inter-candidate and inter-confirmed time p01/p10/p25/p50/p75/p90/p99;
- same-side and opposite-side inter-trigger distributions;
- fraction of time covered by at least one open `2000ms` outcome window;
- candidates per ShockCluster and ContinuousFlowEpisode;
- cluster/episode duration and member-count quantiles;
- longest near-continuous trigger run;
- row count, cluster count, flow-block count, time-block count, and estimated
  effective sample size.

Readers must see this density report before any result based on the hundreds
of thousands of candidate rows.

Required dependence treatment:

- preserve existing same-direction `50ms` deduplication;
- assign every Family A candidate, including rejected candidates, to a
  versioned episode-merging cluster;
- episodes whose outcome windows overlap belong to the same statistical
  block;
- bootstrap and permutation operate on session/time/flow blocks, not rows;
- publish raw row count, flow-block count, time-block count, and estimated
  effective sample size separately;
- report isolated and contaminated distributions side by side;
- never make the isolated subset the primary result when support is too small.

The primary `episode_merging_v1` contract reuses the accepted structural
boundary semantics:

1. within a segment, consecutive candidate atoms with gap `<=100ms` belong to
   one ShockCluster;
2. adjacent clusters may bridge only when their gap is `<=250ms` and no
   recovery checkpoint exists;
3. a recovery checkpoint requires at least two Binance states spanning
   `>=50ms`, spread no wider than pre-state plus one tick, combined top-5 depth
   at least `80%` of pre-state, and no extension of the directional midpoint
   extreme for `>=50ms`;
4. missing or ambiguous recovery evidence means no bridge;
5. segment and connection-epoch boundaries terminate every cluster/episode.

The complete parameter set, provenance, and membership must be frozen and
audited. `flow_cluster_id` is not an informal grouping label; it is the output
of this episode-merging contract.

Named density sensitivity:

```text
trigger_density_sensitivity_v1
```

It must report, without replacing the primary trigger:

- impact-ratio thresholds `0.50` and `0.70`;
- same-side refractory periods `100/250/500ms`;
- first Candidate per primary flow episode;
- trigger intensity, cluster geometry, effective sample size, response
  distribution, and dose-response stability under each view.

Sensitivity selection may use only trigger/pre-state structure. Future
response or outcome quality may not choose a stricter threshold. When the
trigger remains near-continuous, Palm and dose-response conclusions must be
described as conditional views of a continuously active order-flow process,
not as isolated-event effects.

The primary populations are:

```text
Family A:
all frozen trigger-audit candidates under observed continuous-flow reality

Family B:
all confirmed candidates, with quality subsets reported explicitly
```

Isolation is a diagnostic conditioning variable, not a hidden eligibility
filter.

This dataset contains only a few independent dates. Even perfect within-date
block handling can support only `cross-few-session consistency`; it cannot
establish statistical replication across the population of future sessions.

## 11. Case Retrieval and Conditional Distribution

### 11.1 Primary estimator: auditable case retrieval

Family A and Family B require separate retrieval contracts.

For a Family A Candidate-time query:

\[
x_i^A=(S_i^{pre},T_i^{candidate},R_i^{candidate})
\]

For a Family B Confirmed-time query:

\[
x_i^B=(S_i^{pre},T_i^{0:d_i},R_i^{0:d_i})
\]

retrieve historical cases:

\[
\mathcal N_k(x_i)
=
\operatorname{argmin}_{j}
d(x_i,x_j)
\]

The retrieval contract must freeze:

- family view and decision landmark;
- feature allowlist;
- direction normalization;
- discovery-only robust median/IQR scaling;
- feature-group weights;
- missingness penalties;
- session exclusion and purge/embargo;
- candidate `k` values and selection rule;
- maximum accepted distance and out-of-distribution rule.

Every prediction row must publish the exact neighbor episode IDs, distances,
weights, source sessions, flow blocks, and outcome availability.

Family A neighbor pools must contain rejected and confirmed historical
candidates under the same frozen candidate definition. Family B may use only
confirmed historical cases. A Family A query may not use confirmation status,
confirmation lag, confirmed queue removal, or any feature observed after
Candidate.

### 11.2 Required comparison estimators

Compare the case estimator with:

1. direction/session empirical baseline;
2. cross-spread-only empirical baseline;
3. regularized interpretable model;
4. quantile nonlinear model using the same decision-time feature allowlist.

The case estimator is not automatically accepted because it matches the
conceptual framework. It must demonstrate calibration and transfer relative
to simpler baselines.

### 11.3 Distribution outputs

For continuous outcomes:

- conditional mean;
- p10, p25, p50, p75, p90;
- empirical CDF;
- prediction-interval coverage;
- tail mean below p10 for adverse outcomes.

For binary outcomes:

- calibrated probability;
- Brier score;
- reliability table;
- support and effective sample size.

For event-time outcomes:

- interval-censored survival curve;
- cumulative hazard;
- p10/p50/p90 event time;
- probability that the event occurs before each latency threshold.

### 11.4 Pre-registered scoring contract

The phrase `normalized score` means the following fixed losses:

| Outcome family | Primary loss |
| --- | --- |
| continuous predictive distribution | CRPS |
| binary outcome probability | Brier score |
| interval/right-censored event time | interval log loss |

For an interval-censored event \(T\in(L,U]\), interval log loss is:

\[
\mathcal L_{interval}
=
-\log\left(
\max(F(U)-F(L),\epsilon)
\right)
\]

For a right-censored event \(T>L\):

\[
\mathcal L_{right}
=
-\log\left(
\max(1-F(L),\epsilon)
\right)
\]

The numerical floor \(\epsilon\), horizon, outcome list, and any censoring
weights must be frozen before evaluation.

For family view \(f\), model \(M\), outcome family \(k\), and session \(s\),
define the frozen baseline:

\[
M_0(f)=
\begin{cases}
A0, & f=Family\ A\\
B0, & f=Family\ B
\end{cases}
\]

Then:

\[
normalized\_loss_{M,k,s,f}
=
\frac{
loss_{M,k,s,f}
}{
loss_{M_0(f),k,s,f}
}
\]

Interpretation:

```text
< 1.00   better than the frozen A0/B0 baseline for this family view
= 1.00   equal to that baseline
> 1.00   worse than that baseline
> 1.05   more than 5% worse than that baseline
```

The primary model-selection score is the equal-weight arithmetic mean of the
three pre-registered outcome-family normalized losses. Scores are first
computed inside each session and then reported across sessions; pooled rows
cannot replace session scores.

CRPS, Brier, and interval log loss have fixed outcome roles. The study may
report calibration plots, quantile coverage, pinball loss, integrated Brier
score, or log score as diagnostics, but it may not choose whichever metric
looks favorable after evaluation.

## 12. Incremental Information Tests

The central comparison is:

\[
P(Y\mid \text{cross-spread state})
\]

versus:

\[
P(Y\mid S^{pre},T)
\]

where \(Y=(R,O^{market})\).

Pre-register these nested feature sets:

```text
Family A Candidate-time:
A0 = direction + underlying state + topology/cadence context
A1 = A0 + cross-spread level/change/age
A2 = A1 + full pre-trigger dual-venue state
A3 = A2 + Candidate-observable queue-shock prefix

Family B Confirmed-time:
B0 = direction + underlying state + topology/cadence context
B1 = B0 + cross-spread level/change/age
B2 = B1 + full pre-trigger dual-venue state
B3 = B2 + Confirmed-time queue-shock T
B4 = B3 + allowed early target-response prefix
```

Questions:

1. Does A1/B1 beat A0/B0 across sessions?
2. Does A3 add Candidate-time information without confirmation selection?
3. Does B3 beat B1/B2 after dependence-aware evaluation?
4. How different are
   \(P(O\mid candidate)\) and \(P(O\mid candidate,confirmed=1)\)?
5. Is the improvement concentrated in one direction, one session, one
   source-age regime?
6. Does B4 improve calibration enough to justify a later decision timestamp?
7. Does confirmation arrive after the target has already become adverse?

If B3 does not improve on B1/B2, the correct conclusion is:

```text
cross_spread_state_supported_queue_shock_increment_not_supported
```

The trigger remains useful as an event-time anchor even if its severity fields
add little predictive information.

## 13. Validation Design

### 13.1 Development

Use only Jul30 segments `0001-0004` to fit:

- robust transforms;
- distance weights;
- case support rules;
- model hyperparameters;
- outcome binning;
- calibration method.

Use Jul30 segments `0005-0008` for purged internal validation.

### 13.2 Historical transfer

Apply the frozen Jul30 contract to Aug03 without fitting on Aug03 outcomes.
Aug03 may be used to reject the current representation or to motivate a future
version, but not to silently refit v1.

### 13.3 Historical consumed validation

Apply the unchanged contract to Aug04. Results remain historical consumed
validation because Aug04 has already been used by commonality research.

### 13.4 Retrospective method holdout

Before opening Aug07 event rows with the new episode/model implementation:

1. write and fsync the full contract and code hashes;
2. write an immutable Aug07 consumption ledger;
3. bind the accepted AMD R0/R1/basis inventory;
4. run one deterministic build;
5. retain failed attempts and never overwrite the first-read record.

No threshold, feature, distance, outcome, hypothesis, or interpretation may
change after the first Aug07 row is opened under this implementation.

### 13.5 Statistical unit

Primary uncertainty uses:

- session-level reporting;
- flow-block bootstrap inside session;
- time-block bootstrap as a robustness view;
- no row-level IID bootstrap;
- exact count of independent dates fixed at the number of sessions.

### 13.6 Underlying regime and topology transfer

Every development/transfer report must show the joint support matrix:

```text
session
x underlying_market_state
x collection_topology_fingerprint
x Hyperliquid feed-cadence regime
x direction
```

An empty joint cell is a missing generalization regime, not a zero effect.
Timing-model transfer is accepted only inside supported topology/cadence
cells. Price/distribution transfer across topology cells may be reported with
the topology field included, but cannot be described as pure market-regime
transfer.

### 13.7 Evidence-strength boundary

Jul30, Aug03, Aug04, and Aug07 are at most four independent session/date
observations, with Aug04 already consumed and Aug07 only a retrospective
method holdout. This study can establish:

```text
cross-few-session consistency in the observed regimes
```

It cannot establish:

```text
population replication
stable future-session generalization
```

No bootstrap, candidate count, or effective sample size inside a session can
increase the number of independent observed dates.

## 14. Evaluation Gates

### Gate A: Data and replay truth

- every input manifest and source hash passes;
- future joins: `0`;
- cross-epoch labels: `0`;
- source timestamp regressions: `0`;
- exact degraded masks preserved;
- session topology fingerprints and research hosts recorded separately;
- underlying-market calendar version and per-state coverage published;
- Hyperliquid channel cadence/resolution distributions published;
- Binance `@trade q` / no-`nq` measurement status bound to every input;
- raw input unchanged before/after;
- Aug07 full artifacts validated on amdserver;
- local compact metadata never substituted for omitted large event files.

### Gate B: Episode truth

- Candidate/Confirmed/Rejection projections match the accepted historical
  detector on Jul30/Aug03/Aug04;
- Candidate uses no later confirmation state;
- all anchors satisfy their causal ordering;
- every candidate appears exactly once in trigger audit;
- every trigger-audit candidate has exactly one Family A outcome/censor row;
- every confirmed candidate has exactly one linked Family B view;
- rejected/unconfirmed candidates are not absent from Candidate-time outcomes;
- Family A and Family B share candidate identity and immutable source lineage;
- every non-null decision feature passes
  `observed_at_ns <= decision_landmark_ns`;
- all event-time paths remain inside segment/epoch boundaries;
- two independent complete builds are byte-identical.

### Gate C: Distribution calibration

For pre-registered primary outcomes:

- CRPS is the only primary continuous-distribution loss;
- Brier score is the only primary binary-probability loss;
- interval log loss is the only primary event-time loss;
- empirical interval coverage is reported against nominal coverage;
- binary probability reliability is reported by session;
- no accepted result relies only on pooled rows;
- the selected estimator has primary mean normalized loss `<1.00` in every
  outcome-qualified evaluation session required by the frozen contract;
- no outcome-family/session normalized loss exceeds `1.05`;
- out-of-distribution and unavailable rows remain explicit.

### Gate D: Incremental trigger information

Queue-shock \(T\) is considered incrementally supported only when B3:

- improves the pre-registered distribution score over both B1 and B2;
- preserves improvement in every outcome-qualified evaluation session;
- has no materially opposite result in another session;
- retains improvement under flow-block bootstrap;
- does not obtain the result from one post-hoc state bucket.

If fewer than three outcome-qualified evaluation sessions remain after data,
underlying-regime, topology, cadence, and censoring gates, incremental trigger
information cannot receive a supported classification. The result is
`inconclusive_data_quality_or_coverage`.

### Gate E: Public actionability proxy

For latency scenarios:

```text
25, 50, 100, 250, 500 ms
```

Family B computes a Confirmed-time interval margin. If:

\[
T_{adverse}\in(L_i,U_i]
\]

then:

\[
margin_i(\ell)
\in
\left(
L_i-(t_{confirm}+\ell),
U_i-(t_{confirm}+\ell)
\right]
\]

Classify:

```text
identified_positive:
  lower margin bound > 0

identified_non_positive:
  upper margin bound <= 0

timing_ambiguous:
  interval crosses 0

right_or_quality_censored:
  outcome ordering not fully observed
```

Report:

- identified-positive, identified-non-positive, ambiguous, and censored rates;
- interval-aware median and p10 margin bounds;
- side/session/regime breakdown;
- fraction already adverse before Confirmed;
- Family A Candidate-time results over all candidates, including rejected;
- Family B Confirmed-time results;
- sensitivity to Candidate-time versus Confirmed-time observation;
- scenario identification rate relative to each session's feed cadence.

These are hypothetical public actionability scenarios. They must never be
labelled observed cancel effectiveness. `25ms`/`50ms` results that are below
feed resolution remain diagnostics and cannot support
`quote_protection_research_candidate`.

## 15. Decision Outcomes

The study must end with exactly one primary classification:

```text
conditional_market_distribution_supported
  Spre/T case retrieval transfers and is calibrated across sessions.

cross_spread_supported_trigger_increment_not_supported
  Cross-spread state predicts response, but queue-shock T adds no stable value.

quote_protection_research_candidate
  Distribution and public actionability gates support a later no-order shadow.

candidate_time_research_required
  Family A supports a Candidate-time path and Confirmed-time is systematically
  too late; this is not inferred from confirmed-only candidates.

representation_not_supported
  Case similarity does not transfer or calibrate.

inconclusive_data_quality_or_coverage
  Dataset admission, outcome coverage, OOD, or effective support is inadequate.
```

`quote_protection_research_candidate` does not authorize live orders. It may
only unlock a separate production-equivalent no-order shadow plan.

## 16. Implementation Modules

Preserve all existing v1/v2 artifacts. Implement episode v3 in separate files:

```text
examples/hyperliquid/cross_exchange_trigger_aligned_episode_contract.py
examples/hyperliquid/cross_exchange_trigger_aligned_episodes.py
examples/hyperliquid/cross_exchange_episode_case_distribution.py
examples/hyperliquid/cross_exchange_episode_actionability_shadow.py
examples/hyperliquid/cross_exchange_episode_research_report.py
```

Focused tests:

```text
examples/hyperliquid/test_cross_exchange_trigger_aligned_episode_contract.py
examples/hyperliquid/test_cross_exchange_trigger_aligned_episodes.py
examples/hyperliquid/test_cross_exchange_episode_case_distribution.py
examples/hyperliquid/test_cross_exchange_episode_actionability_shadow.py
```

Reuse structured readers, parsers, detector logic, provenance validators,
atomic publication, and canonical serialization from the current repository.
Do not reimplement R0/R1/basis joins with ad hoc CSV matching.

If sharing the detector requires extraction, first move it to a pure,
versioned module with parity tests. Historical builders must continue to
produce identical bytes.

## 17. Output Layout

Proposed output:

```text
local_live_analysis/skhynix_trigger_aligned_episode_research_v1/
  research_manifest.json
  frozen_research_contract.json
  input_inventory.csv
  consumption_ledgers/
  data_admission/
    session_topology.csv
    underlying_regime_coverage.csv
    hyperliquid_feed_cadence.csv
    trigger_density_by_session.csv
    inter_trigger_distribution.csv
    effective_sample_size.csv
  episodes/
    jul30/
    aug03/
    aug04/
    aug07/
    family_a_candidate_view.csv.gz
    family_b_confirmed_view.csv.gz
    episode_merging_membership.csv.gz
  paths/
    sparse_event_paths/
    fixed_event_time_paths/
    event_count_paths/
  features/
    family_a_decision_features.csv.gz
    family_b_decision_features.csv.gz
    feature_observation_ledger.csv.gz
    feature_contract.json
  outcomes/
    family_a_market_outcomes.csv.gz
    family_b_market_outcomes.csv.gz
    event_time_intervals.csv.gz
    outcome_contract.json
  cases/
    neighbor_assignments.csv.gz
    case_support_by_session.csv
    conditional_distributions.csv.gz
    calibration_by_session.csv
  incremental/
    nested_model_scores.csv
    scoring_contract.json
    confirmation_selection_comparison.csv
    trigger_increment_bootstrap.csv.gz
  actionability/
    public_latency_scenarios.csv.gz
    actionability_by_session.csv
  reports/
    data_admission.md
    event_time_response_atlas.md
    conditional_distribution_report.md
    cross_spread_increment_report.md
    public_actionability_report.md
    research_limitations.md
    final_decision.md
  runtime_source/
```

The complete package must be content-addressed and atomically published.

## 18. Required Hostile Tests

At minimum:

- future target row injected into an \(S^{pre}\) feature;
- confirmation event leaked into Candidate-time features;
- one rejected candidate deleted from Family A outcomes;
- Family A built from confirmed rows only;
- nullable rejected `t_confirm_ns` replaced with a synthetic timestamp;
- a decision feature with `observed_at_ns > decision_landmark_ns`;
- cross-segment and cross-epoch path access;
- source age or degraded flag deletion;
- an interval-censored Hyperliquid event coerced to one point timestamp;
- a `25ms` scenario classified despite an unresolved feed interval;
- timing rows pooled across incompatible topology fingerprints;
- underlying-market state omitted or inferred from a future outcome;
- historical `trade.q` mislabeled as `nq` or RPI-adjusted quantity;
- historical `depth@0ms` mislabeled as a `depth@100ms` contract;
- direction normalization reversed for one side;
- `d_bh` and `d_hb` mapping swapped;
- repeated forward-filled state counted as a new event;
- overlapping rows treated as IID bootstrap units;
- outcome field added to the retrieval feature allowlist;
- Aug07 read before freeze/ledger creation;
- neighbor drawn from the query's embargoed flow block;
- unavailable outcome silently converted to zero;
- public quote-touch proxy renamed or interpreted as fill;
- same-version contract or input drift;
- existing historical v2 artifact changed during v3 build;
- interrupted publication exposing a partial output.

Every hostile case must fail closed and preserve accepted source artifacts.

## 19. Ordered Research Queue

If later dispatched, execute strictly in this order:

1. Freeze input inventory, acquisition topology, underlying-market calendar,
   feed-cadence, measurement limitations, evidence labels, scores, hypotheses,
   and primary outcomes.
2. Publish trigger density, inter-trigger distribution, episode-merging
   sensitivity, and effective-sample-size admission before outcome modeling.
3. Implement the shared trigger contract and prove historical detector parity.
4. Build Family A and Family B episode v3 for Jul30 only.
5. Validate anchors, all-candidate outcome coverage, per-feature observation
   times, interval censoring, exact paths, dependence blocks, and deterministic
   rebuild.
6. Freeze Family A/Family B case distance, baselines, outcomes, censoring
   model, and model selection on Jul30.
7. Apply without refit to Aug03.
8. Apply unchanged to Aug04.
9. Freeze code and create the Aug07 first-read ledger.
10. Build and evaluate Aug07 on amdserver.
11. Run confirmation-selection and incremental cross-spread versus queue-shock
    tests.
12. Run interval-censored public latency/actionability scenarios.
13. Produce the final classification and independent QA.
14. Decide whether to stop, redesign Candidate-time research, or create a
    separate production-equivalent no-order shadow plan.

Each numbered item should become a separate formal workflow task or an
explicitly bounded substage only after the preceding QA gate passes. This
document itself dispatches none of them.

## 20. Data Measurement Limitations

### 20.1 Binance quantity and RPI

Historical SKHYNIX data uses `fstream @trade` and stores `q`. It does not store
`aggTrade.nq`, an RPI participation flag, or another field that separates
trades against liquidity absent from the visible public queue. This cannot be
reconstructed from the archive.

Consequences:

- trigger severity is an observed pressure proxy, not exact displayed-queue
  consumption;
- ratio calibration and dose-response may be biased upward when the numerator
  contains quantity unrelated to the visible denominator;
- the bias may vary by session and market state;
- depth confirmation reduces false interpretation but does not identify the
  missing quantity decomposition;
- no result may claim that a ratio of `1.0` means the displayed queue was
  exactly traded through.

### 20.2 Binance depth stream

Historical campaigns subscribed to `@depth@0ms`, not the idealized
`@depth@100ms` design described in the earlier slice document. All cadence,
burst, confirmation, and latency statements must use the observed archive
distribution and recorded stream identity.

### 20.3 Hyperliquid public observation

Hyperliquid BBO, trades, and fast-L2 are aggregate public feeds. They do not
reveal the exact exchange-internal transition time, queue ownership, hidden
liquidity, or this account's fill. Event times are interval/right censored at
the public-feed resolution.

### 20.4 Underlying market state

The datasets do not contain a complete KRX SK Hynix spot order book. The
underlying-market state is a versioned calendar/context label, not a direct
measurement of spot price discovery. Holidays, special sessions, or calendar
ambiguity must be explicit.

### 20.5 Timing topology

Same-host receipt order is a fact about the acquisition topology. Postprocess
location does not alter it, but different acquisition topology fingerprints,
network paths, clock behavior, or feed-cadence regimes can change timing
outcomes. Cross-topology timing transfer is not a pure market statement.

### 20.6 Replication

The available dates provide only cross-few-session evidence. Hundreds of
thousands of trigger rows and resampled blocks do not create additional
independent dates or population replication.

## 21. Non-Goals

This study does not:

- collect new data;
- access private/account/order/cancel endpoints;
- submit or cancel orders;
- infer exact queue position;
- claim a public trade would have filled this account;
- estimate actual maker fee, inventory, or PnL;
- estimate causal KEEP/CANCEL EV;
- optimize GLFT spread, skew, size, or inventory parameters;
- promote a live strategy;
- claim population-level replication from the existing dates;
- mix queue-shock and cross-spread trigger families under one episode origin.

## 22. Completion Definition

The research is complete only when the accepted package answers:

1. Can every trigger-audit candidate, including rejected and censored cases,
   be transformed into one Family A record with a correctly linked Family B
   view where confirmed?
2. Are trigger density, merging geometry, feed resolution, topology,
   underlying regime, and effective sample size visible before result gates?
3. What response and market-outcome distributions follow each comparable
   \(S^{pre},T\) neighborhood?
4. Does cross-spread state alone explain the result, or does queue-shock
   severity add stable information?
5. Are CRPS, Brier, interval log loss, probability, quantile, survival, and
   tail estimates calibrated across the few sessions rather than only in
   pooled rows?
6. Under interval-censored public latency scenarios, is Candidate-time or
   Confirmed-time quote protection sufficiently identified to justify a later
   no-order production shadow?

Until these six questions pass their gates, the project may report an
auditable episode case library and descriptive response atlas, but must not
claim a deployable maker action rule.
