# SKHYNIX c6in Hyperliquid Execution Latency Measurement Plan

Date: 2026-08-22

Revision: review draft 2

Status: controller review draft only. This document does not create a formal
task, authorize credentials or private endpoints, authorize an order or
cancel, or unlock H0-B.

## 0. Review And Authority

This plan defines the measurement required before the controller chooses one
of the H0-A frozen pre-H0-B latency decisions:

```text
retain_100ms_as_preregistered_scenario
revise_primary_tuple_before_outcomes
```

The authority chain is:

```text
accepted v2 master framework
-> accepted H0-A execution contract and candidate package
-> independent H0-A QA
-> this measurement-plan review
-> separately dispatched formal latency-measurement task
-> business execution on the pinned c6in host
-> independent QA over the redacted measurement package
-> controller latency decision
-> retain the accepted H0-A tuple or publish an accepted superseding tuple
-> separately reviewed H0-B plan and task
```

No step in this document may be interpreted as permission to run measurement
traffic during H0-A QA. H0-A remains support-only and forbids private,
account, order and cancel access.

If the v2 framework, H0-A contract, formal measurement task, machine contract
or this plan disagree, execution fails closed before credentials or a private
endpoint are accessed.

## 1. Purpose

The plan answers one infrastructure question:

```text
After a risk decision is locally ready on the retained c6in execution host,
how long does the production-equivalent Hyperliquid path take to make the
tracked resting order authoritatively terminal?
```

It must separately measure:

1. local decision-to-dispatch overhead;
2. the synchronous cancel-call response RTT;
3. the relative ordering and gap between cancel response and authoritative
   terminal confirmation;
4. total decision-ready-to-terminal-confirmation latency;
5. cancellation reliability, timeout and fill-race rates.

The output is a pre-outcome recommendation for Gate H-C. It is not a trading
strategy result and does not claim that the SKHYNIX risk signal exists.

## 2. Current Prior Evidence

H0-A froze:

```text
target = public_bbo_moves_through_quote
horizon_ms = 50
gate_latency_ms = 100
gate_latency_basis = preregistered_gate_hc_scenario_not_execution_measurement
execution_latency_identified = false
```

The H0-A public cadence review found formal-session Hyperliquid BBO
inter-arrival p50 values below 100ms and did not challenge observation
resolution. It explicitly did not measure private execution latency.

Existing historical awsserver artifacts contain 18 unique local cancel-call
response durations:

```text
min = 614ms
p50 = 700ms
p90 = 786ms
p95 = 803ms
max = 816ms
```

Those rows are context only because they:

- were not produced on c6in;
- do not identify the current c6in runtime or network path;
- record local cancel response completion, not exact exchange cancellation
  time;
- do not provide the complete decision-ready-to-terminal interval required by
  this plan.

Historical awsserver rows must never be pooled with c6in primary samples.

## 3. Frozen Non-Goals

This measurement must not:

- open H0-B outcome aggregates or any adverse-event label;
- run RQ1, RQ2 or RQ3;
- inspect Aug07 research event rows;
- change the H0-A selected horizon, target, distance or side aggregation;
- mutate the accepted H0-A package or its primary tuple;
- tune a risk threshold, hysteresis, debounce, feature or model;
- estimate fill probability, queue position, fee, rebate, inventory value,
  markout, PnL or maker viability;
- optimize host region, route, DNS, SDK, connection reuse or request batching
  after seeing latency results;
- use ICMP ping, TCP connect time or a one-way estimate as a replacement for
  measured order lifecycle intervals;
- add a guessed "chain confirmation" duration to an already measured
  end-to-end RTT;
- treat a cancel API response by itself as authoritative terminal proof;
- treat a cancel error saying an order was already canceled or filled as a
  successful identified cancellation;
- hide slow, failed, timed-out, filled or terminal-unresolved attempts;
- use a BTC control-market sample as direct SKHYNIX target-market evidence;
- authorize deployment, default-on behavior or a live strategy.

## 4. Preconditions

### 4.1 H0-A Workflow State

The formal measurement task may be dispatched only after:

1. H0-A independent QA is `已通过`;
2. the controller accepts the exact H0-A R/C/E/composite identity;
3. H0-B outcome access remains false;
4. the controller records that the latency decision is pending this
   measurement.

The plan may be reviewed while H0-A QA is running, but measurement execution
must not be folded into QA.

As of `2026-08-21T17:07:45Z`, these H0-A workflow prerequisites are satisfied:

- QA acceptance commit:
  `43b088c315d0da18411b3a316def3030169b5039`;
- accepted R:
  `7176c78c2b6bfadf11432e9f5a1c1eaf9627c8028a4a22257d67d5a5404dc8fd`;
- accepted C:
  `4e8ccc7466f84d9eb2f71f681557424b59ca58249d67426325ecdbd661189636`;
- accepted E:
  `8745458fcf4e0ab31f8ad3b2bc2d1d93704f18b13a776c515f14195b51213969`;
- accepted composite:
  `2682c32eefac427eed1899a3d492b3fc7545520f72021723e8fef7a0d4d8d9d0`;
- controller closure:
  `.workflow/reports/0821T001-controller-closure.md`.

This satisfies the H0-A authority boundary only. The measurement plan still
requires review, a separate formal task and any separately approved active
live authority before execution.

### 4.2 Accepted Trust Kernel

Because the measurement can change the primary tuple used by H0-B, its formal
package must use accepted Research Package Trust Kernel v1 in
`mode=accepted`. The formal task must pin the then-current accepted registry
entry and source identities exactly.

The task must classify:

- redacted lifecycle samples and derived summaries as `R`;
- measurement code, schemas, tests and frozen rules as `C`;
- host/runtime receipts, execution boundary, publication and QA receipts as
  `E`.

### 4.3 c6in Host Identity

The formal task must pin the exact retained execution host before any private
access:

```text
ssh_alias
cloud_instance_id
cloud_account_or_project_identity
region
availability_zone
hostname
machine_id
boot_id
primary_network_interface
public_egress_identity
kernel_version
cpu_architecture
clocksource
ntp_synchronization_status
```

The historical c6in winner identity in prior collection tasks is context, not
automatic authority. A recreated instance, changed region or changed egress
identity is a different measurement population and requires a new host pin.

### 4.4 Runtime Identity

Before collection, the task must freeze:

```text
repository_commit
working_tree_clean
python_executable
python_version
hyperliquid_sdk_version
measurement_entrypoint_sha256
terminal_classifier_source_sha256
runtime_dependency_inventory_sha256
api_base_hostname
proxy_mode
ip_family
http_connection_reuse_mode
request_timeout_seconds
terminal_query_policy
terminal_query_retry_interval_ms
terminal_query_timeout_ms
upstream_strategy_runtime_mode
action_transport_type
live_order_allowed
passive_route_availability
```

The production-equivalent client path must be used. A standalone `curl`,
synthetic HTTP endpoint or a different SDK may be used only in diagnostics
and cannot enter the primary latency population.

The runtime preflight must prove whether the upstream strategy can emit a real
cancel. `production_dry`, `DryActionTransport` or `live_order_allowed=false`
means the passive route is unavailable because no venue mutation occurs.

### 4.5 Market Identity

The primary measurement must use the same Hyperliquid DEX/product/asset path
intended for the SKHYNIX live system. The formal task must pin:

```text
dex
canonical_asset
sdk_asset_identifier
asset_metadata_identity
tick_size
lot_size
minimum_valid_order_size
minimum_valid_order_notional
quote_distance_ticks
quote_distance_price
reference_mid_price
quote_distance_one_way_bps
minimum_safe_quote_distance_bps
quote_distance_safety_status
```

The historical public profile name `xyz:SKHX` is not sufficient by itself.
The formal task must resolve and freeze the current canonical private-order
identifier before active execution.

For the frozen `quote_distance_ticks=10`, the c6in preflight must derive:

```text
quote_distance_price = 10 * tick_size
quote_distance_one_way_bps =
    quote_distance_price / reference_mid_price * 10000
```

The formal dispatch must freeze `minimum_safe_quote_distance_bps` and its
public-data authority before the first private call. If tick size, reference
price or the market-specific safety predicate cannot be established, or if
`quote_distance_one_way_bps < minimum_safe_quote_distance_bps`, active
measurement fails closed before the first submit. The runner may not increase
the quote distance after inspecting market or latency results; changing the
frozen 10-tick choice requires a reviewed revision.

A more liquid control asset such as BTC may be measured first to validate
instrumentation. It remains `transport_control_only` and cannot select the
SKHYNIX Gate H-C primary latency.

### 4.6 Explicit Live Authorization

No prior BTC or awsserver standing authorization carries into this task.

Before an active measurement, the formal dispatch must explicitly set:

```text
active_private_read_authorized
active_order_submit_authorized
active_cancel_authorized
reduce_only_flatten_authorized
target_account_identity_token
target_market_identity
per_order_notional_cap_usdc
aggregate_position_cap_usdc
max_loss_usdc
max_loss_basis
max_open_orders
max_attempts_per_batch
max_total_attempts
max_batch_duration_seconds
```

If any required authorization is absent or false, the task may perform only
offline instrumentation tests and passive artifact admission.

## 5. Measurement Semantics

### 5.1 Clock Domains

Every interval used in a latency statistic must use a single process-local
monotonic nanosecond clock.

Required clocks:

```text
duration_clock = time.monotonic_ns or equivalent
audit_clock = UTC wall-clock nanoseconds
```

Wall time is recorded for audit and ordering across files. It must not be
subtracted to produce latency. NTP adjustments, leap seconds and wall-clock
steps must not affect a duration.

Every sample binds:

```text
boot_id
process_id
process_start_monotonic_ns
sample_sequence
```

A process restart, boot change or monotonic regression starts a new stratum.

### 5.2 Required Event Timestamps

For every tracked order attempt, record:

```text
t_submit_call_start_mono_ns
t_submit_response_end_mono_ns
t_resting_confirm_mono_ns
t_risk_decision_ready_mono_ns
t_cancel_enqueue_mono_ns
t_cancel_call_start_mono_ns
t_cancel_response_end_mono_ns
t_terminal_observation_start_mono_ns
t_terminal_confirm_mono_ns
t_final_open_orders_confirm_mono_ns
```

The primary start is `t_risk_decision_ready_mono_ns`. It represents the point
at which a production-equivalent risk controller has completed its decision
and requests removal of the tracked order.

The active calibration runner may generate this marker mechanically after
resting confirmation. It must not use H0-B outcomes or claim to benchmark the
future H0-B estimator.

### 5.3 Derived Intervals

For each sample:

```text
submit_response_rtt_ms =
    t_submit_response_end - t_submit_call_start

resting_confirmation_lag_ms =
    t_resting_confirm - t_submit_response_end

decision_to_enqueue_ms =
    t_cancel_enqueue - t_risk_decision_ready

enqueue_to_call_ms =
    t_cancel_call_start - t_cancel_enqueue

cancel_response_rtt_ms =
    t_cancel_response_end - t_cancel_call_start

terminal_minus_cancel_response_ms =
    t_terminal_confirm - t_cancel_response_end

cancel_effective_latency_ms =
    t_terminal_confirm - t_risk_decision_ready

final_safety_confirmation_ms =
    t_final_open_orders_confirm - t_risk_decision_ready
```

All values are derived from integer nanoseconds and published as exact integer
microseconds plus decimal milliseconds. Float timestamps are forbidden.

`terminal_minus_cancel_response_ms` is signed. An authenticated terminal event
may arrive before the synchronous cancel call returns. A negative value is
valid evidence of that ordering and must not be clamped to zero. The primary
`cancel_effective_latency_ms` remains nonnegative and is unaffected by which
of those two observations arrives first.

### 5.4 Gate H-C Mapping

Gate H-C defines:

```text
residual_dwell =
    t_exit - (t_detect + frozen_latency)
```

This plan maps `frozen_latency` to the production-equivalent post-detection
execution interval:

```text
t_risk_decision_ready -> t_terminal_confirm
```

It intentionally does not use:

- public BBO inter-arrival as an execution measurement;
- cancel-call response alone;
- final archive or report completion;
- an ICMP or TCP network estimate;
- a separately guessed exchange-confirmation add-on.

`final_safety_confirmation_ms` is reported as a conservative operational
diagnostic. It does not replace `cancel_effective_latency_ms` unless the
terminal classifier cannot authoritatively identify the tracked order before
the final open-orders proof.

This task measures the execution path after a decision is ready. It does not
benchmark the future H0-B estimator. The H0-B plan must separately prove,
before outcome access, either that its live decision runtime is already inside
the frozen `t_detect` boundary or that an outcome-blind runtime allowance has
been added through the same pre-outcome tuple-revision process.

## 6. Terminal Confirmation Contract

### 6.1 Exact Tracked Reference

Every terminal claim must bind the same tracked order through an exact
redaction-safe reference token derived from:

```text
account_identity_token
dex
asset
oid
cloid
submit_attempt_id
```

Raw credentials, private keys, signatures, nonces and unredacted account
identifiers must never enter the package.

### 6.2 Authority Order

The first authoritative terminal observation may come from:

1. an authenticated exact-reference order update classified as canceled;
2. an exact `query_order_by_oid` or exact `query_order_by_cloid` result
   classified as canceled;
3. an exact-reference terminal-history result plus a consistent empty
   open-orders snapshot;
4. a final empty open-orders proof only when the accepted terminal contract
   explicitly permits absence proof for that exact reference.

The formal task must reuse or extract the accepted exact-reference terminal
classifier. It must not create a weaker keyword or substring classifier.

### 6.3 Non-Terminal Responses

These are not authoritative successful cancellations:

```text
HTTP or SDK call returned without a classified cancel status
status=ok with an exchange error payload
already canceled or filled
unknown order
timeout
transport exception
reference mismatch
open-orders read failure
terminal query contradiction
```

They remain in the attempt denominator and are assigned an exact failure or
censoring class.

## 7. Sampling Design

### 7.1 Passive-First Route

The preferred first source is a production-equivalent c6in process that
already generates real, authorized cancels for the target market.

For the current review revision, the existing GLFT route is
`production_dry`, uses exact `DryActionTransport` objects and records
`live_order_allowed=false`. It therefore emits no real venue cancel and cannot
produce a passive primary row. Active calibration under §7.2 is the only
currently available measurement route, and it remains unavailable until a
separate live authorization is approved.

The passive preference is retained for a future revision in which a
production-equivalent GLFT source is independently proven to emit authorized
real cancels. Dry-action timing must never be relabeled as passive exchange
latency. The current GLFT observation is planning evidence, not a future
runtime pin; Gate 2 must re-establish the exact clean runtime and transport
identity.

Passive collection may add timing instrumentation, but it must not:

- create additional orders;
- change quote placement, hold time or cancel policy;
- change client reuse, retry or timeout behavior;
- choose only favorable periods;
- discard an attempt because it was slow or failed.

Passive rows are primary eligible only when the complete required timestamp
and exact terminal contract are present.

### 7.2 Active Calibration Route

For the current `production_dry` GLFT state, active calibration is the only
available route and requires a separately authorized live envelope under §8.
In a future passive-enabled revision, active calibration may instead be used
only when the passive source cannot reach the frozen sample gate.

Each active attempt is:

```text
public metadata and BBO preflight
-> one minimum-size post-only quote away from touch
-> exact resting confirmation
-> fixed pre-cancel settle interval
-> risk-decision-ready marker
-> production-equivalent tracked cancel
-> exact terminal confirmation
-> final open-orders and position safety proof
```

No attempt may submit a replacement order until the prior order is
authoritatively terminal and safety reconciliation is complete.

### 7.3 Primary Population

A row is `primary_latency_eligible=true` only when:

- it ran on the exact pinned c6in host and boot stratum;
- it used the exact pinned runtime and target-market path;
- the order was accepted and exactly confirmed resting;
- the cancel began from the frozen decision-ready marker;
- all monotonic timestamps are present and ordered;
- the cancel reference is exact and redaction-safe;
- terminal cancellation is authoritatively identified;
- no fill occurred before terminal confirmation;
- no runtime, network-mode or task-contract drift occurred;
- the attempt is not a warm-up or instrumentation-control row.

Eligibility is derived mechanically. It is not editable in the report.

### 7.4 Failure And Censoring Classes

Every non-eligible real attempt must have exactly one primary class:

```text
submit_rejected
resting_not_confirmed
filled_before_cancel_dispatch
filled_during_cancel_race
cancel_response_error
cancel_response_timeout
cancel_transport_exception
terminal_confirmation_timeout
terminal_reference_mismatch
terminal_query_contradiction
final_open_orders_unavailable
clock_contract_invalid
runtime_identity_drift
host_identity_drift
safety_stop
```

No class is silently removed from reliability denominators.

The reliability denominators are exact:

```text
target_cancel_attempt_count =
    target-market attempts with exact resting confirmation and
    t_risk_decision_ready emitted

terminal_identified_count =
    target_cancel_attempt_count rows with an authoritative exact-reference
    terminal classification before the frozen terminal timeout

terminal_identified_fraction =
    terminal_identified_count / target_cancel_attempt_count

fill_during_cancel_race_count =
    target_cancel_attempt_count rows classified filled_during_cancel_race

fill_during_cancel_race_fraction =
    fill_during_cancel_race_count / target_cancel_attempt_count
```

A zero `target_cancel_attempt_count` makes both fractions unavailable and
fails the sample gate.

### 7.5 Strata

Report separately by:

```text
UTC collection window
target versus control market
buy versus sell
warm versus cold HTTP connection
normal versus retry path
process and boot identity
```

Primary selection uses only:

```text
target market
production-equivalent connection lifecycle
all naturally occurring normal/retry paths under the frozen client policy
exact pinned runtime
```

Naturally occurring retries remain in the pooled primary population; removing
them would understate live latency. Cold-start or reconnect rows also remain
primary when they arise under the ordinary production lifecycle. A separately
forced cold/retry canary is diagnostic-only. The runner must not force a
reconnect, timeout or retry merely to change the primary distribution.

### 7.6 Sample Gate

The frozen primary sample requirement is:

```text
target_primary_eligible_count >= 100
target_primary_eligible_count_goal_current_active_only = 100
target_total_attempt_count <= 120
distinct_utc_collection_windows >= 3
eligible_count_per_window >= 20
largest_window_fraction <= 0.50
terminal_identified_fraction >= 0.99
unresolved_exposure_count = 0
clock_contract_failure_count = 0
runtime_or_host_drift_count = 0
```

The prior draft's 200-row aspirational goal is retired for this active-only
revision because it is incompatible with the 120-attempt hard cap. A future
passive-enabled revision may separately propose an extended 200-row target;
it cannot increase the active attempt cap or alter this task after results are
visible.

Exact window formulas:

```text
eligible_count_per_window =
    count(primary_latency_eligible=true for that preselected window)

largest_window_fraction =
    max(eligible_count_per_window) / target_primary_eligible_count
```

The three windows must be selected before collection and must not be chosen
from observed latency. A missed window requires a controller-recorded
replacement before its latency values are inspected.

Failure to reach the gate produces:

```text
latency_measurement_inconclusive_h0b_locked
```

It must not lower the sample requirement or switch to awsserver/control-market
rows.

The `max_total_attempts=120` cap gives exactly 20 attempts of headroom above
the 100-row eligible floor:

```text
maximum_noneligible_headroom = 120 - 100 = 20
maximum_noneligible_fraction_compatible_with_floor = 20 / 120 = 1 / 6
```

If more than 20 attempts are non-eligible, reaching 100 eligible rows is
mathematically impossible. Collection must stop at 120 total attempts and emit
`latency_measurement_inconclusive_h0b_locked`; no top-up, cap extension or
replacement campaign is permitted within the same task revision.

## 8. Active Micro-Live Safety Envelope

This section defines maximum reviewable bounds. It is not live authorization.

### 8.1 Order Bounds

The formal task may authorize no more than:

```text
post_only_required = true
time_in_force = Alo or exact current post-only equivalent
max_open_orders = 1
max_attempts_per_batch = 10
max_total_attempts = 120
max_batch_duration_seconds = 900
max_total_batches = 12
fixed_pre_cancel_settle_ms = 250
minimum_inter_attempt_seconds = 20
per_order_notional_cap_usdc = 5
aggregate_position_cap_usdc = 10
max_loss_usdc = 1
max_loss_basis = realized_reduce_only_flatten_slippage
```

Order size must be the smallest valid size whose notional is at or below the
cap. If the market minimum exceeds the cap, active target-market measurement
is blocked.

`max_loss_usdc` is not a mark-to-market drawdown limit. It is the realized
adverse price slippage from the separately authorized reduce-only flatten
after a measurement fill:

```text
if original_fill_side = buy:
    realized_flatten_slippage_loss_usdc =
        max(0, (fill_vwap - flatten_vwap) * flattened_quantity)

if original_fill_side = sell:
    realized_flatten_slippage_loss_usdc =
        max(0, (flatten_vwap - fill_vwap) * flattened_quantity)
```

The formula uses matched filled/flattened quantity and excludes fees and
rebates, which are recorded separately and are not an economic-PnL claim.
For a partial flatten, the loss is unavailable and exposure remains
unresolved.

This loss is knowable only after the reduce-only flatten is authoritatively
complete. It is therefore a retrospective stop against any later batch, not
an ex-ante guarantee that realized loss cannot exceed 1 USDC. The temporary
fill-induced position is bounded by `aggregate_position_cap_usdc` at fill
notional; that cap does not bound subsequent mark-to-market loss. This plan
does not require L0 to implement a separate real-time mark-to-market loss
monitor solely for `max_loss_usdc`.

### 8.2 Quote Placement

The active task must freeze side schedule and quote distance before the first
private call:

```text
side_schedule = balanced deterministic schedule
quote_distance_ticks = 10
```

A quote is placed away from touch:

```text
buy_price <= current_best_bid - 10 ticks
sell_price >= current_best_ask + 10 ticks
```

The quote must remain valid, post-only and non-crossing immediately before
submit. The distance is a safety device, not a strategy parameter or a fill
study. Before the first active submit, Gate 2 must confirm the exact tick size,
10-tick price distance, one-way bps distance and frozen market-specific safety
predicate from §4.5. An unresolved or insufficient distance stops before
submit and therefore before any resting confirmation.

### 8.3 Batch Stop Conditions

Stop the batch immediately on:

- any fill or position delta;
- any unresolved order reference;
- final open orders not exactly empty;
- account or market identity mismatch;
- credential or redaction failure;
- after an authoritatively completed reduce-only flatten,
  `realized_flatten_slippage_loss_usdc >= max_loss_usdc`;
- flatten incomplete, partially matched or otherwise unreconciled;
- runtime source, dependency, host, route or clock drift;
- two consecutive cancel response timeouts;
- one terminal confirmation timeout;
- post-only invariant failure;
- any unclassified response;
- user or controller stop.

After a fill, only the separately authorized reduce-only flatten and final
reconciliation path may run. No replacement measurement order is allowed in
that batch. The loss-cap comparison occurs only after flatten completion. If
the realized flatten slippage is at or above 1 USDC, the entire task stops and
no later batch or top-up attempt is permitted.

### 8.4 Batch Review

Every batch is sealed and admitted before the next batch. The next batch may
start only when:

```text
final_open_orders = 0
position_within_frozen_baseline = true
realized_flatten_slippage_loss_usdc < max_loss_usdc
    or no_fill_occurred = true
all_attempts_classified = true
artifact_inventory_exact = true
```

Latency values from a completed batch may not be used to alter later request
timeouts, connection mode, quote distance or sample eligibility.

## 9. Instrumentation Architecture

### 9.1 L0 Lifecycle Instrumenter

L0 runs on the pinned c6in host and owns:

- monotonic and audit timestamps;
- exact attempt sequence;
- submit/resting/cancel/terminal lifecycle events;
- terminal classifier calls;
- safety reconciliation;
- redaction before package publication.

The raw private response may exist only in the task-local protected runtime
directory for the minimum time needed to classify and redact it. It is not
pulled back or committed.

### 9.2 L1 Sealed Summarizer

L1 runs in a fresh process and may read only:

- the redacted sealed lifecycle ledger;
- the frozen measurement contract;
- the frozen host/runtime receipts;
- the accepted Trust Kernel and exact task contract.

L1 has no credentials, private endpoint client or network authority.

It derives:

- eligibility and censor classes;
- exact intervals;
- reliability fractions;
- nearest-rank quantiles;
- latency bucket recommendation;
- controller decision recommendation.

L1 must not receive a command-line option that excludes rows by latency,
status, period or side.

### 9.3 Process Boundary

Before L1 starts:

1. L0 closes every file;
2. the redacted ledger is canonicalized and fsynced;
3. the exact file inventory and SHA256 identity are sealed;
4. credentials and raw private responses are absent from the L1 root;
5. L1 runs with network disabled or guarded fail-closed;
6. any attempt to open a path outside the sealed root fails.

## 10. Frozen Statistical Rules

### 10.1 Quantile Definition

For sorted eligible values `x[1..n]`, use nearest-rank:

```text
Q(p) = x[ceil(p * n)]
```

Publish:

```text
min
p25
p50
p75
p90
p95
p99
max
mean
median_absolute_deviation
```

The primary latency statistic is:

```text
p95(cancel_effective_latency_ms)
```

`cancel_response_rtt_ms` and `final_safety_confirmation_ms` are separate
diagnostics and cannot replace it because they happen at different lifecycle
boundaries.

### 10.2 Bucket Rule

The recommended primary Gate H-C latency is:

```text
recommended_gate_latency_ms =
    max(100, 50 * ceil(p95_cancel_effective_latency_ms / 50))
```

Examples:

```text
p95 = 83ms  -> 100ms
p95 = 101ms -> 150ms
p95 = 723ms -> 750ms
p95 = 803ms -> 850ms
```

The bucket rule is frozen before live collection. No result-dependent choice
among p50, p90, p95, p99 or max is permitted.

### 10.3 Reliability Rule

A finite latency recommendation requires:

```text
sample_gate_pass = true
terminal_identified_fraction >= 0.99
unresolved_exposure_count = 0
fill_during_cancel_race_fraction <= 0.01
```

If these fail, the output is not a larger latency number. It is:

```text
execution_path_reliability_not_established_h0b_locked
```

### 10.4 Controller Recommendation

L1 emits exactly one recommendation:

```text
retain_100ms_as_preregistered_scenario
revise_primary_tuple_before_outcomes
latency_measurement_inconclusive_h0b_locked
execution_path_reliability_not_established_h0b_locked
```

Rules:

```text
sample/reliability gates pass and recommended latency = 100
  -> retain_100ms_as_preregistered_scenario

sample/reliability gates pass and recommended latency > 100
  -> revise_primary_tuple_before_outcomes

sample gate fails
  -> latency_measurement_inconclusive_h0b_locked

reliability gate fails
  -> execution_path_reliability_not_established_h0b_locked
```

The recommendation does not itself mutate the H0-A tuple.

## 11. Output Package

The formal package root is task-specific and frozen by the future task.

Required artifacts:

```text
measurement_manifest.json
frozen_measurement_contract.json
host_identity.json
runtime_identity.json
market_identity.json
authorization_envelope.json
collection_window_schedule.csv
attempt_ledger.csv
lifecycle_events.csv
failure_and_censoring.csv
latency_by_attempt.csv
latency_summary.csv
reliability_summary.json
historical_context_awsserver.csv
controller_latency_recommendation.json
boundary_manifest.json
sha256_inventory.csv
reports/execution_latency_measurement.md
```

### 11.1 `attempt_ledger.csv`

Required ordered fields:

```text
schema_version
task_id
sample_sequence
collection_window_id
batch_id
attempt_id
host_identity_token
boot_id
process_identity_token
runtime_identity_sha256
market_role
dex
asset
side
order_reference_token
post_only
quote_distance_ticks
tick_size
quote_distance_price
quote_distance_one_way_bps
order_size
order_notional_usdc
submit_status
resting_status
cancel_response_class
terminal_class
fill_race_class
filled_quantity
fill_vwap
flatten_status
flattened_quantity
flatten_vwap
realized_flatten_slippage_loss_usdc
flatten_fee_usdc
final_open_orders_count
position_delta
safety_status
primary_latency_eligible
primary_exclusion_reason
```

### 11.2 `lifecycle_events.csv`

One row per event:

```text
schema_version
task_id
sample_sequence
event_sequence
event_type
monotonic_ns
audit_utc_ns
order_reference_token
source
classification
detail_code
```

`event_sequence` and `monotonic_ns` must be strictly increasing within a
sample. Unknown event types fail closed.

### 11.3 `latency_by_attempt.csv`

Required fields:

```text
schema_version
task_id
sample_sequence
collection_window_id
market_role
side
connection_mode
retry_path
submit_response_rtt_us
resting_confirmation_lag_us
decision_to_enqueue_us
enqueue_to_call_us
cancel_response_rtt_us
terminal_minus_cancel_response_us
cancel_effective_latency_us
final_safety_confirmation_us
primary_latency_eligible
failure_or_censor_class
```

Missing intervals use a frozen explicit null representation. Zero is a real
duration and cannot represent missing.

### 11.4 `latency_summary.csv`

Primary key:

```text
market_role, collection_window_id, side, connection_mode, retry_path, metric
```

Each row contains:

```text
attempt_count
eligible_count
identified_count
failure_count
min_us
p25_us
p50_us
p75_us
p90_us
p95_us
p99_us
max_us
mean_us
mad_us
```

Pooled primary rows are mechanically derived from the exact eligible target
population. Per-window rows remain visible and cannot be replaced by the
pooled result.

### 11.5 Recommendation Binding

`controller_latency_recommendation.json` must bind:

- accepted H0-A tuple identity;
- measurement R/C/E/composite identity;
- host/runtime/market identities;
- sample and reliability gate facts;
- exact primary population identity;
- exact p95 and bucket rule;
- recommended latency;
- exactly one recommendation enum;
- `h0b_outcome_accessed=false`;
- `h0a_tuple_mutated=false`.

## 12. Identity And Publication

### 12.1 Research Identity R

R contains:

- redacted attempt and lifecycle ledgers;
- derived latency rows and summaries;
- failure/reliability facts;
- frozen recommendation.

### 12.2 Code/Contract Identity C

C contains:

- instrumenter and sealed summarizer source;
- exact schemas;
- fake-clock and fake-client tests;
- task/plan/machine contract;
- terminal-classifier source pin;
- quantile and bucket rules.

### 12.3 Evidence Identity E

E contains:

- host/runtime/market receipts;
- authorization envelope;
- collection-window and batch receipts;
- boundary and safety evidence;
- archive/publication receipts;
- independent QA receipt.

The composite identity is emitted only after R/C/E reverse bindings pass.

### 12.4 Durable Archive

The final accepted package must be archived outside the c6in execution
worktree and pulled back to the controller/QA environment with exact
relative-path, size and SHA256 parity.

Archive completion occurs only after:

- every batch is closed;
- final open orders are zero;
- position reconciliation passes;
- L1 summarization is complete;
- the exact package tree is fsynced;
- remote and pulled-back inventories match.

## 13. Hostile And Metamorphic Preflight

The formal task must freeze these stable error-code families before
implementation:

| Surface | Stable error code |
| --- | --- |
| wall clock used for duration | `LATENCY_WALL_CLOCK_DURATION_FORBIDDEN` |
| monotonic timestamp missing, decreasing or reordered | `LATENCY_MONOTONIC_ORDER_INVALID` |
| decision-ready marker missing | `LATENCY_DECISION_READY_MISSING` |
| cancel response promoted to terminal proof | `LATENCY_CANCEL_RESPONSE_NOT_TERMINAL` |
| terminal status unknown or contradictory | `LATENCY_TERMINAL_STATUS_UNIDENTIFIED` |
| foreign or inconsistent order reference | `LATENCY_ORDER_REFERENCE_MISMATCH` |
| raw secret/account/order-reference leak | `LATENCY_SECRET_OR_REFERENCE_LEAK` |
| duplicate attempt/event identity | `LATENCY_SAMPLE_IDENTITY_DUPLICATE` |
| real attempt missing from sealed ledger | `LATENCY_REAL_ATTEMPT_OMITTED` |
| control/awsserver row promoted to target | `LATENCY_TARGET_POPULATION_CONTAMINATED` |
| host or boot identity drift | `LATENCY_HOST_IDENTITY_DRIFT` |
| runtime or dependency identity drift | `LATENCY_RUNTIME_IDENTITY_DRIFT` |
| connection/retry policy drift | `LATENCY_CLIENT_POLICY_DRIFT` |
| quantile method or percentile changed | `LATENCY_QUANTILE_CONTRACT_MISMATCH` |
| recommended bucket rounded down | `LATENCY_BUCKET_ROUND_DOWN` |
| collection-window support invalid | `LATENCY_WINDOW_SUPPORT_INVALID` |
| sample gate claimed with insufficient rows | `LATENCY_SAMPLE_GATE_NOT_MET` |
| reliability denominator or class mismatch | `LATENCY_RELIABILITY_DENOMINATOR_MISMATCH` |
| unresolved order or exposure | `LATENCY_UNRESOLVED_EXPOSURE` |
| dry transport treated as passive exchange evidence | `LATENCY_PASSIVE_SOURCE_NOT_LIVE` |
| total-attempt cap exceeded or topped up | `LATENCY_ATTEMPT_CAP_EXHAUSTED` |
| tick-size or quote-distance safety unresolved/insufficient | `LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED` |
| mark-to-market or another basis substituted for flatten slippage | `LATENCY_LOSS_CAP_CONTRACT_MISMATCH` |
| H0-B outcome path opened | `LATENCY_H0B_OUTCOME_ACCESS_FORBIDDEN` |
| accepted H0-A tuple changed in place | `LATENCY_H0A_TUPLE_MUTATION_FORBIDDEN` |
| report and machine recommendation differ | `LATENCY_RECOMMENDATION_DIVERGENCE` |

Before any active private call, offline fixtures must prove fail-closed
behavior for:

1. wall-clock subtraction used as latency;
2. swapped or decreasing monotonic timestamps;
3. missing decision-ready marker;
4. cancel response treated as terminal confirmation;
5. "already canceled or filled" treated as a successful cancel;
6. foreign oid/cloid terminal result;
7. raw order reference leakage;
8. duplicate attempt or event sequence;
9. dropped slow attempt;
10. dropped timeout or fill-race attempt;
11. primary eligibility edited by report text;
12. BTC control rows promoted to target rows;
13. awsserver rows pooled with c6in;
14. host/boot/runtime identity change inside a stratum;
15. HTTP connection mode change after collection starts;
16. nearest-rank quantile replaced by interpolation;
17. p50 or p90 substituted for frozen p95;
18. bucket rounded down;
19. one window contributing more than 50%;
20. insufficient sample count declared conclusive;
21. terminal-identified denominator altered;
22. unresolved exposure hidden after cleanup;
23. H0-B outcome path opened;
24. H0-A tuple changed in place;
25. report recommendation diverging from machine recommendation;
26. `production_dry` or `DryActionTransport` row admitted as passive exchange
    latency;
27. attempt 121 submitted or a top-up added after the 120-attempt cap;
28. active submit allowed with unresolved tick size or insufficient 10-tick
    bps safety;
29. mark-to-market loss substituted for realized reduce-only flatten
    slippage, or the loss cap evaluated before flatten completion.

Every case must assert an exact stable error code.

## 14. Business Execution Gates

### Gate 0: Dispatch Contract

- plan review closed;
- one formal task ID exists;
- canonical machine contract exists;
- no conflicting formal task is active;
- accepted Trust Kernel pin passes;
- H0-A QA/controller acceptance prerequisites pass;
- explicit live authorization fields are present for active mode.

### Gate 1: Offline Instrumentation

- fake-clock ordering and duration tests pass;
- fake-client submit/resting/cancel/terminal branches pass;
- redaction and exact-reference classifier pass;
- all hostile cases pass before credentials are accessed.

### Gate 2: c6in Identity And Safety Preflight

- exact host/runtime/market identity passes;
- transport preflight proves passive availability or records the current
  `production_dry` route as unavailable;
- tick size, 10-tick price distance, one-way bps distance and the frozen
  market-specific safety predicate pass before submit;
- account identity token matches;
- final open orders and position baseline are captured;
- no conflicting service owns the same account/market path;
- caps and stop conditions are installed;
- raw credential values are not printed or persisted.

### Gate 3: Control-Market Canary

- optional control-market instrumentation canary passes;
- it proves only clock, lifecycle and artifact mechanics;
- it cannot satisfy the target sample gate.

### Gate 4: Target-Market Batches

- batches run only under the frozen schedule and caps;
- every attempt is reconciled before the next;
- each batch seals exact artifacts before continuation;
- any stop condition halts further submits.

### Gate 5: Sealed Summarization

- L1 receives only the redacted sealed root;
- network/private access is absent;
- every real attempt has one classification;
- summary and recommendation rebuild deterministically.

### Gate 6: Package Admission

- exact tree and schemas pass;
- R/C/E/composite and reverse bindings pass;
- current and archived verifier parity passes;
- verify-only is zero-write;
- credentials and raw private payloads are absent.

### Gate 7: Business Handoff

- final open orders are zero;
- position and loss are reconciled;
- remote and pulled-back package inventories match;
- business report ends in `待验收`;
- H0-B remains locked.

## 15. Independent QA Gates

QA does not place or cancel an order. It independently validates the sealed
business evidence.

### QA Gate 0: Authority

- exact task, plan, machine contract and authorization envelope agree;
- H0-A QA/controller prerequisite is accepted;
- no H0-B outcome access occurred.

### QA Gate 1: Source And Boundary

- c6in host/runtime/market identities are exact;
- current `production_dry`/`DryActionTransport` evidence is rejected as
  passive exchange latency;
- tick size, 10-tick price distance, bps conversion and safety predicate
  rebuild exactly;
- no awsserver/control rows enter the target primary population;
- no credential or raw private payload leaks.

### QA Gate 2: Negative Matrix

- QA reruns hostile cases against current and frozen implementations;
- exact stable codes pass;
- fail-open count is zero.

### QA Gate 3: Attempt Reconstruction

- independently reconstruct every attempt from lifecycle events;
- recompute eligibility, failure and censor classes;
- require total attempts at or below 120 and prove no top-up campaign exists;
- reconstruct fill/flatten quantities and realized flatten slippage when a
  fill occurred;
- detect duplicate, omitted, reordered and foreign-reference rows.

### QA Gate 4: Latency Reconstruction

- recompute every interval from integer monotonic timestamps;
- recompute nearest-rank quantiles and window fractions;
- require exact match with business summaries.

### QA Gate 5: Recommendation

- independently apply sample, reliability, p95 and bucket rules;
- require the same single recommendation;
- verify H0-A tuple remains immutable.

### QA Gate 6: Safety And Archive

- final open-orders, position, flatten and stop evidence pass;
- loss uses only the frozen realized reduce-only flatten-slippage formula and
  is evaluated after authoritative flatten completion;
- remote/pulled-back/archive trees and identities match;
- current/frozen admission is zero-write.

### QA Gate 7: Conclusion Boundary

- no signal, actionability, fill, economics or maker claim is made;
- QA concludes only whether the latency measurement and recommendation are
  accepted.

## 16. Failure And Repair Policy

Any failure stops H0-B.

A bounded repair may fix:

- timestamp instrumentation;
- exact-reference terminal classification;
- redaction;
- deterministic summary reconstruction;
- artifact schema, identity or publication;
- host/runtime receipt completeness.

A repair may not:

- delete or recollect an unfavorable accepted batch;
- change sample eligibility;
- lower sample or reliability gates;
- change p95 to another percentile;
- round the bucket down;
- switch target market after seeing results;
- change network/runtime configuration and pool before/after rows;
- extend `max_total_attempts` or add a top-up after the 120-attempt cap;
- change the frozen 10-tick distance after market preflight;
- replace realized flatten slippage with mark-to-market or another loss basis;
- mutate H0-A in place;
- open H0-B outcomes.

A host, route, SDK, terminal policy or target-market change after collection
starts creates a new measurement revision. The old and new rows remain
separate.

## 17. Controller Decision And H0-B Handoff

After QA `已通过`, the controller records one of:

### Route A: Retain 100ms

Allowed only when:

```text
recommendation = retain_100ms_as_preregistered_scenario
recommended_gate_latency_ms = 100
```

The controller records the accepted measurement identity and retains the
accepted H0-A tuple unchanged.

### Route B: Superseding Tuple

Required when:

```text
recommendation = revise_primary_tuple_before_outcomes
recommended_gate_latency_ms > 100
```

The controller must:

1. review a v2/H0-A latency revision;
2. create a separate formal tuple-supersession task;
3. preserve the accepted H0-A package immutably;
4. publish a new tuple that changes only the reviewed latency surfaces and
   required sensitivity set;
5. independently QA and accept that tuple;
6. make H0-B pin the superseding tuple.

If the selected latency exceeds the existing `500ms` sensitivity maximum,
the superseding plan must explicitly add the selected bucket and retain
`100ms` only as an optimistic sensitivity scenario.

### Route C: Remain Locked

Either inconclusive or reliability-not-established recommendation keeps H0-B
locked. The controller may choose a new collection revision or stop the
direct-live-applicability claim. H0-B may not silently keep 100ms.

## 18. Completion Definition

This review draft is complete when:

- the measurement object and clock contract are explicit;
- cancel response and authoritative terminal confirmation are separated;
- passive and active routes are separated;
- active limits and stop conditions are reviewable;
- sample, reliability, quantile and bucket rules are frozen;
- exact artifacts, hostile tests, QA and H0-B handoff are specified;
- no formal task or live authority has been implied.

The future measurement task is complete only after:

- business execution and safety reconciliation finish;
- independent QA is `已通过`;
- the controller accepts the exact measurement identity;
- the controller records the pre-H0-B latency decision.

## 19. Review Choices

Review of this plan accepts or revises these load-bearing choices:

1. H0-A QA and latency measurement remain separate tasks.
2. No historical live authorization carries into this measurement.
3. c6in target-market evidence is required for direct SKHYNIX applicability.
4. BTC and awsserver evidence remain diagnostic only.
5. Durations use monotonic nanoseconds; wall time is audit-only.
6. Primary latency runs from risk-decision-ready to authoritative terminal
   confirmation.
7. Cancel response RTT is reported separately and cannot prove terminal
   cancellation.
8. A passive row is eligible only when its production-equivalent source emits
   a real venue cancel; dry-action timing is never exchange-latency evidence.
9. Active calibration, if separately authorized, uses one minimum-size
   post-only order at a time under the frozen caps.
10. The primary population requires at least 100 eligible target samples over
    at least three preselected UTC windows.
11. The primary statistic is nearest-rank p95.
12. The recommended bucket rounds p95 upward to the next 50ms, with a 100ms
    floor.
13. Reliability failure produces a lock, not a larger invented latency.
14. The measurement package uses accepted Trust Kernel v1 and independent QA.
15. Any latency above 100ms requires a separately reviewed and accepted
    superseding tuple before H0-B.
16. `max_loss_usdc=1` means realized adverse price slippage from the
    reduce-only flatten, calculated only after authoritative flatten
    completion. It is not a mark-to-market limit or an ex-ante loss guarantee.
17. `max_total_attempts=120` provides 20% headroom over the 100-row eligible
    floor. More than 20 non-eligible attempts makes the result inconclusive;
    no top-up beyond 120 is permitted. The exact compatible non-eligible
    fraction is at most `1/6`; a 17% rate at the hard cap fails.
18. The passive preference is retained only for a future live
    production-equivalent cancel source. Current GLFT `production_dry`
    evidence cannot enter the latency population.
19. `quote_distance_ticks=10` must be converted using the confirmed target
    tick size and pass the frozen market-specific bps safety predicate in §4.5
    before the first active submit. Failure stops before resting confirmation.
