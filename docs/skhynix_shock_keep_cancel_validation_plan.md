# SKHYNIX Shock-Triggered KEEP/CANCEL Validation Plan

Date: 2026-08-13

Status: plan only; not dispatched; no collection, private endpoint, order,
cancel, deployment, configuration change, or live authorization is granted by
this document

## 1. Objective

This plan validates one narrow production question:

```text
When a confirmed Binance SKHYNIXUSDT queue shock occurs,
does canceling only the exposed Hyperliquid xyz:SKHX maker quote
improve event-level maker value relative to keeping that quote?
```

The shortest accepted path is:

```text
immutable public events
-> deterministic shock Candidate / Confirmed / Rejected state machine
-> native monotonic decision path
-> frozen KEEP / CANCEL_RISK_SIDE policy
-> actionability evidence
-> exact own-order lifecycle evidence
-> randomized event-level EV comparison
```

This is not a general market-making optimization plan. It does not search for
a new symbol, rebuild the GLFT quote model, optimize inventory skew, add
multi-level quoting, or claim profitability from public midpoint prediction
alone.

The instrument contract is fixed:

- lead venue and symbol: Binance USD-M `SKHYNIXUSDT`;
- target and execution venue: Hyperliquid `xyz:SKHX`;
- target activity and liquidity are treated as a project premise, not as a
  milestone to re-evaluate against BTC.

## 2. Current Baseline

The plan must reuse rather than duplicate the following accepted or completed
work:

- immutable public collection, inventory verification, R0/R1 provenance,
  connection epochs, stale masks, point-in-time joins, and deterministic
  rebuilds in `hftbacktest`;
- the offline Binance aggressive-trade shock and Hyperliquid response episode
  contract in `docs/skhynix_liquidity_response_motif.md`;
- the SKHYNIX GLFT public shadow, quote model, risk controls, native
  hftbacktest/Nautilus matrices, and deterministic policy replay in `glft`;
- accepted read-only SKHX account subscriptions and reconciliation evidence;
- the selected official-SDK HTTP mutation baseline and dedicated critical
  cancel-lane architecture;
- the accepted native single-writer, typed event, deterministic Python/Rust
  replay, and controlled monotonic-latency evidence.

The current evidence does not yet establish:

- that the offline detector runs causally in the connected native path;
- that one exact KEEP/CANCEL policy is isolated from existing GLFT and guard
  behavior;
- that a cancel can become effective before the target adverse event;
- that active SKHX private updates provide a complete order lifecycle;
- that randomized live cancellation improves maker EV.

Existing negative theoretical markout, cross-session arrival shift,
250ms-latency stress failure, and unidentified queue/partial-fill sensitivity
remain active warnings. No stage may reinterpret them as passed evidence.

## 3. Frozen Experimental Boundary

### 3.1 Shock direction and risk side

Direction mapping is fixed:

| Binance confirmed shock | Expected SKHX direction | Exposed maker quote |
| --- | --- | --- |
| aggressive buy / ask depletion | upward | maker ask |
| aggressive sell / bid depletion | downward | maker bid |

The mapping must be encoded once in the versioned contract. Runtime code,
replay, reports, and experiment analysis must consume the same field rather
than reconstructing it independently.

### 3.2 Allowed actions

Only two experiment actions exist:

```text
KEEP
  Leave the already-resting exposed-side order unchanged.

CANCEL_RISK_SIDE
  Request cancellation of only the already-resting exposed-side order.
```

The experiment action may not:

- change order size;
- widen, improve, or otherwise reprice either quote;
- cancel the opposite-side quote;
- submit a replacement inside the event outcome horizon;
- change GLFT fair value, spread, skew, inventory, or risk parameters;
- use taker, crossing, market, modify, batch, or multi-level behavior.

Independent safety controls remain authoritative. A safety cancellation is not
an experiment treatment and must be recorded as an override.

### 3.3 Eligibility before assignment

An event is experiment-eligible only when all of the following are true:

- shock state is `Confirmed` under the frozen detector version;
- the exposed-side order is known, resting, uniquely identified, and inside
  the frozen quote envelope;
- public lead and target streams are fresh, ordered, and inside one connection
  epoch;
- native queue, worker, transport, private stream, account, and reconciliation
  health gates pass;
- no unresolved mutation, ambiguous order outcome, or same-order cancel is
  active;
- inventory and risk state permit both KEEP and CANCEL;
- no same-side deduplication cooldown or opposite-direction conflict applies;
- sufficient same-session outcome horizon remains.

Ineligible events remain in the audit with exact reasons. They must not be
silently dropped or assigned after observing an outcome.

### 3.4 Time and identity

`receive_monotonic_ns` is authoritative for causal ordering and latency on one
host. Wall time is retained for operator correlation and cross-artifact
reporting. Exchange timestamps are evidence fields, not the local action clock.

Every normalized event must carry:

- source venue, stream, symbol, connection epoch, and source sequence where
  available;
- receive monotonic and wall timestamps;
- stable event ID and detector version;
- source health and book-sequence state;
- source raw-record identity or immutable provenance reference.

Every decision must carry:

- shock candidate and confirmation IDs;
- exposed order ID, client order ID, side, price, quantity, and generation;
- policy version, experiment version, assignment, and override reason;
- state hash before and after the decision;
- publish, dequeue, decision, mutation-call, first-write, response, private
  update, fill, and terminal timestamps where applicable.

## 4. Stage 0: Freeze `slice_contract_v1`

### Purpose

Remove policy ambiguity before connected measurements are inspected. This
stage defines what is being tested and prevents existing quote logic,
protective guards, or later threshold edits from contaminating attribution.

### Required contract

Publish a machine-readable `slice_contract_v1` and a readable specification
that freeze:

- instrument and direction mapping;
- Candidate, Confirmed, Rejected, timeout, deduplication, cooldown, overlap,
  and recovery semantics;
- KEEP and CANCEL_RISK_SIDE semantics;
- eligibility and safety-override rules;
- event identity, clocks, required audit fields, and canonical serialization;
- actionability endpoints and event EV formula;
- assignment method, sample boundaries, stopping rules, and analysis plan;
- all schema, source, binary, configuration, and contract hashes.

The current offline detector values are the v1 starting point:

- same-side consecutive trade burst within `10ms` of the first trade;
- Candidate when cumulative touch quantity reaches `30%` of the strict
  pre-shock impacted Binance queue;
- Confirmed within `100ms` when that queue is depleted or falls to at most
  `70%` of its pre-shock quantity;
- attribution and quality fields retained exactly, including all rejected and
  uncertain candidates.

The first policy version acts only on `Confirmed`. Acting on Candidate or on a
predicted probability of confirmation is a separate future contract and may
be considered only if confirmed-event actionability fails.

### Gate

Stage 0 passes only when one frozen contract hash can be consumed by both
repositories and every later artifact is required to reference it. Thresholds
may not be changed under the same version.

## 5. Stage 1: Move the Detector Into the Native Monotonic Path

### Purpose

Convert the offline label builder into a causal production state machine
without changing its accepted semantics.

### Work boundary

Implement and compare:

```text
normalized Binance event
-> Candidate
-> Confirmed or Rejected
-> exposed SKHX risk side
-> KEEP/CANCEL intent only
```

The first connected mode is public-only and no-order. It records decisions but
does not read credentials or call private, order, or cancel endpoints.

`hftbacktest` owns the research reference, source provenance, offline rebuild,
and detector comparison. `glft` owns native connected ingestion, monotonic
timing, single-writer state, and production-shaped decision records.

### Required evidence

- Python reference and native implementation produce identical candidate,
  confirmation, rejection, side, and deduplication projections for the frozen
  corpus;
- replay after every input event matches state hash and emitted decision
  records;
- Candidate never consumes the later confirmation event or any future state;
- reconnect, sequence gap, stale source, queue overflow, invalid quantity,
  and missing pre-state fail closed;
- connected public events can be archived and replayed into byte-identical
  deterministic projections;
- latency is measured from pre-parse receive through native decision emission.

### Gate

Stage 1 requires zero unexplained semantic mismatches, zero silent drops or
reorders, complete audit coverage, and passed connected no-order health and
latency gates. Controlled-loop evidence alone is insufficient.

## 6. Stage 2: Actionability Shadow

### Purpose

Test whether a correct shock arrives early enough for cancellation to matter.
Prediction quality alone cannot answer this question.

### Primary timeline

For each eligible confirmed shock record:

```text
t_confirm
t_native_decision
t_cancel_call
t_first_write
t_cancel_response
t_cancel_terminal_private
t_first_adverse_target_event
t_fill, if any
```

The conservative actionability margin is:

```text
margin_ns =
  t_first_adverse_target_event - t_cancel_terminal_private
```

Before active private evidence exists, shadow may estimate this value with a
frozen conservative cancel-latency distribution. Such rows must be labelled
`estimated_actionability`, never `observed_cancel_effective`.

The target adverse event must be frozen before the run. It should include the
earliest observable SKHX event that can make the exposed quote harmful:

- target aggressive trade reaches or crosses the quote price;
- impacted target best price is depleted or moves through the quote;
- directional midpoint move reaches the frozen adverse threshold.

Report the components separately. Do not retain only the most favorable one.

### Required analysis

Report by side, session, source quality, spread, volatility, shock attribution,
and confirmation delay:

- detection and decision latency;
- estimated and later observed actionability margin;
- fraction with positive margin and its confidence interval;
- events already adverse before decision;
- events that become adverse between write, response, and private terminal;
- coverage loss from health, inventory, conflict, and missing-order gates.

### Gate

The initial unlock rule is:

- at least three disjoint sessions;
- both directions represented;
- median conservative margin greater than zero;
- lower bound of a pre-registered 95% confidence interval for the positive
  margin rate greater than `0.50`;
- no session or side whose evidence shows the policy is systematically too
  late;
- zero future leakage and zero unresolved timing identity.

If this gate fails, the live lifecycle experiment stops. The permitted next
step is a separately frozen Candidate-time prediction contract, not a hidden
threshold relaxation.

## 7. Stage 3: Exact Own-Order Lifecycle

### Purpose

Connect `signal -> action -> own outcome`. Public BBO response and theoretical
fill proxies cannot establish whether this account's resting order was
canceled, filled, partially filled, charged a fee, or left unresolved.

### Authorization boundary

Stage 3 requires a new formal task, independent QA, explicit live
authorization, a bounded SKHX order envelope, and verified final account
cleanup. This plan grants none of those permissions.

### Required lifecycle

For every order used as evidence, reconcile:

```text
intent
-> submit call / first write
-> accepted, resting, or rejected response
-> oid/cloid correlation
-> active private order update
-> cancel request, when applicable
-> partial/full fill or cancel terminal
-> fee and liquidity role
-> inventory and balance transition
-> independent final open-orders and position proof
```

Classify fills relative to cancellation:

- fill before cancel request;
- fill after request but before first write;
- fill after first write but before response;
- fill after response but before private cancel terminal;
- fill after observed cancel terminal;
- ambiguous because of missing or disordered evidence.

A fill after observed cancel terminal is a critical contract failure until
independently explained.

### Gate

Stage 3 requires:

- exact active SKHX private update ordering and oid/cloid correlation;
- no unresolved or ambiguous terminal state in accepted samples;
- fees, maker/taker role, fills, and inventory reconcile to account evidence;
- independent final `open_orders=[]`;
- final position equals the authorized envelope's required state;
- cancel-effective timestamps have sufficient coverage for Stage 2
  actionability to be re-evaluated with observed rather than estimated data.

Failure to obtain a fill is not a safety failure, but it is insufficient for
fee, adverse-fill, or EV claims.

## 8. Stage 4: Randomized KEEP/CANCEL Event Experiment

### Purpose

Estimate the causal effect of cancellation. A before/after comparison or
deterministic guard comparison is confounded by market state, quote state,
inventory, and policy selection.

### Assignment

Randomize only after eligibility is frozen and before outcomes are observed.
Use deterministic hash assignment over:

```text
experiment_salt
+ contract_hash
+ shock_event_id
+ exposed_order_generation
```

Use balanced blocks by direction and bounded time block. The assignment is
`1:1 KEEP : CANCEL_RISK_SIDE` unless a separately reviewed risk decision
requires a more conservative treatment share.

Primary analysis is intention-to-treat:

- all assigned events remain in their assigned arm;
- safety overrides, protocol deviations, cancel failures, late cancels, and
  missing outcomes remain visible;
- per-protocol results are secondary diagnostics only.

### Outcome

Freeze one primary event horizon before the experiment. The default v1 horizon
is `2000ms`, with `250ms`, `500ms`, and `1000ms` diagnostics.

Primary event value is account-consistent marked value relative to assignment:

```text
event_value_h =
  cash_change_h
  + inventory_change_h * target_mid_h
  - fees_h
```

The primary treatment effect is:

```text
EV(CANCEL_RISK_SIDE) - EV(KEEP)
```

This naturally includes:

- avoided adverse fills;
- lost favorable fills and spread capture;
- maker fees or rebates;
- partial fills;
- inventory markout;
- cancel failure and cancel-through-fill risk.

No replacement is allowed before the primary horizon, so re-entry policy
cannot contaminate the first experiment.

Secondary outcomes include fill rate, adverse-fill rate, fee-adjusted markout,
maximum inventory excursion, cancel latency, quote downtime, and outcome
coverage.

### Sample plan

Before the first randomized event:

- estimate variance from accepted shadow/lifecycle evidence;
- freeze the required sample size, minimum detectable effect, maximum duration,
  and session/day blocks;
- require both directions and at least three disjoint sessions;
- freeze confidence intervals and any multiple-comparison treatment;
- freeze safety monitoring separately from economic significance testing.

Do not repeatedly inspect profit and stop when the result looks favorable.
Only safety and evidence-integrity stops may terminate the run early.

### Hard stops

Stop assignment and fail closed on:

- stale, gapped, reordered, or untrusted public/private state;
- native queue drop, worker failure, or affinity/latency breach;
- unresolved mutation or account reconciliation failure;
- inventory, notional, drawdown, order-count, or duration limit;
- unexpected taker fill, crossing order, wrong symbol, wrong side, or wrong
  quantity;
- fill after observed cancel terminal;
- missing independent final account proof.

### Gate

Stage 4 supports promotion only when:

- contract, binary, configuration, account envelope, and experiment hashes
  are exact and complete;
- randomized arms are balanced on the frozen strata;
- lifecycle and outcome coverage pass;
- the primary intention-to-treat effect is positive with the pre-registered
  confidence requirement;
- inventory-tail and failure metrics do not materially worsen;
- the effect is not dependent on one session, one side, or an unplanned
  post-hoc filter.

Otherwise the accepted conclusion is `not_supported`, `inconclusive`, or
`unsafe`. No wording may convert an inconclusive result into live promotion.

## 9. Stage 5: Decision and Versioning

Every published artifact must bind:

- raw input inventory and source provenance hashes;
- normalized event schema and detector version;
- `slice_contract_v1` hash;
- native binary/source hashes;
- strategy, risk, transport, and account-envelope hashes;
- assignment salt hash and experiment analysis version;
- host identity, clock source, run ID, and exact time interval.

Decision outcomes:

```text
supported
  Randomized EV and safety gates pass. A separate promotion plan is required.

not_supported
  The causal effect is non-positive or fails cross-session robustness.

inconclusive
  Evidence quantity, variance, lifecycle, or actionability is insufficient.

unsafe
  Safety, reconciliation, ordering, or terminal-state evidence fails.
```

Any change to detector threshold, action timing, quote behavior, target
horizon, outcome formula, assignment, risk envelope, or transport semantics
requires a new contract version. Evidence from different versions may be
compared but not pooled as one experiment.

## 10. Ordered Execution Queue

Execution, if later authorized, is strictly sequential:

1. Freeze and QA `slice_contract_v1`.
2. Port the detector state machine and pass deterministic reference parity.
3. Pass connected public-only native no-order evidence.
4. Run multi-session actionability shadow.
5. Obtain explicit authorization for bounded own-order lifecycle evidence.
6. Re-evaluate actionability with observed cancel-effective timestamps.
7. Freeze randomized sample size and analysis.
8. Obtain explicit authorization for the bounded KEEP/CANCEL experiment.
9. Run randomized evidence and independent QA.
10. Write a separate promotion, redesign, or stop plan from the accepted
    result.

No later step is automatically authorized by an earlier pass.

## 11. Non-Goals

This plan does not authorize or require:

- switching the instrument to BTC or comparing SKHX liquidity with BTC;
- Hyperliquid node deployment or raw node book-diff ingestion before the
  current public-feed path proves insufficient for this slice;
- broad feature search, motif mining, or new regime discovery;
- Candidate-time cancellation in v1;
- repricing, widening, resizing, replacement, multi-level, modify, batch, or
  taker behavior;
- GLFT parameter promotion or inventory-model redesign;
- stable daily PnL, scalable capacity, or production readiness claims;
- unattended or continuous live operation.

## 12. Completion Definition

The plan is complete only when one accepted evidence chain can answer all five
questions:

1. Did the live and replay systems identify the same causal shock event?
2. Was the exact exposed own order known before assignment?
3. Could cancellation become effective before the adverse target event?
4. What actually happened to that order, its fees, and inventory?
5. Did randomized CANCEL_RISK_SIDE improve event EV relative to KEEP?

Until all five are closed, the project may claim research, engineering,
actionability, or lifecycle progress separately, but must not claim that the
shock-triggered cancel policy improves live maker EV.
