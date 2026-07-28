# Principal Alignment P0-P3 Controller Contract

Original plan date: 2026-07-17

Recovery date: 2026-07-28

Recovery status: reconstructed from the accepted task/report chain.

## Recovery Notice

The original file referenced by Principal Alignment tasks was not present in
the canonical `cross-exchange` checkout, on
`amdserver:/home/molly/project/hftbacktest`, in its registered worktrees, in
reachable Git refs, or in unreachable commits inspected during `0728T069`.
The additional clone at
`z370:/home/liushuai/workspace/hftbacktest` was also checked through the
configured `z370-tunnel`; it was an older `cross-exchange/b538621` checkout
dated 2026-06-30 and did not contain the file in its tree or Git history.

This document restores the missing controller contract from the task files,
business reports, QA reports, `task_plan.md`, `progress.md`, and `findings.md`.
It is not represented as byte-identical original text.

This recovered document is non-authorizing:

- it does not grant or revive live, private, account, order, cancel, remote, or
  service permission;
- historical authorization facts remain historical evidence only;
- every future live action requires a new formal task, an exact envelope, and
  fresh authorization when required by that task;
- current QA and task files override this recovered summary if they conflict.

## Objective

Build a trustworthy Binance-lead / Hyperliquid-lag maker path in this order:

1. make price, risk, lifecycle, evidence, and replay contracts deterministic;
2. connect one shared decision path to a reconciled post-only order manager;
3. add adaptive pricing components behind explicit default-off gates;
4. accept live mechanisms only with exact source, account, lifecycle, terminal,
   and checksum evidence;
5. keep fill economics, profitability, promotion, and final MVP claims blocked
   until role-known fills and replay/live economics are available.

## Evidence Layers

Every conclusion must identify its evidence layer:

- `action-path coverage`: code, fixtures, fake endpoints, and deterministic
  state-machine behavior;
- `replay-model regression`: replay/shadow equivalence and anti-optimism checks;
- `live-derived source-path proof`: exact runtime source, account, exchange
  lifecycle, fill role, fees, inventory, and terminal state.

Passing an earlier layer never implies that a later layer passed.

## Principal Task Map

### Task 0 - Live Evidence Integrity Foundation

Recovered source:

- reconstructed historical `0717T006` task/business/QA records, corroborated
  by the accepted `0717T007`-`0717T011` chain;
- `0717T007` window/attempt identity;
- `0717T008` idempotent attempt-bounded fill attribution;
- `0717T009` watcher termination and timeout;
- `0717T010` terminal artifact sealing;
- `0717T011` integrated offline acceptance.

Contract:

- stable run/window/attempt identity;
- exact source and artifact provenance;
- idempotent, attempt-bounded fill attribution;
- bounded child termination and reap;
- heartbeat shutdown before terminal sealing;
- run-root-relative checksum manifest verified after final evidence is written;
- fail-closed integrated acceptance.

### Task 1 - Price Normalization

Formal task: `0718T012`, QA `已通过`.

- one authoritative Hyperliquid perp price-normalization helper;
- buy floor / sell ceil post-only behavior;
- invalid, nonfinite, or crossing prices fail closed.

### Task 2 - Persistent Kill Switch

Formal task: `0718T013`, QA `已通过`.

- durable explicit control state;
- persist halt before exchange action;
- idempotent cancel/flatten sequencing;
- quote paths fail closed on missing, corrupt, or active halt state.

### Task 3 - Aggregate Runtime Exposure

Formal task: `0718T014`, QA `已通过`.

- calculate worst long and worst short exposure from position, working,
  cancel-pending, inflight, and proposed leaves;
- validate aggregate position, notional, and submission caps at submit time;
- preserve reduce-side participation without ignoring sell-through/buy-through
  exposure.

### Task 4 - Typed Price Taxonomy And Quote Eligibility

Formal task: `0718T015`, QA `已通过`.

- versioned `PricingConfigV1`;
- stable config and normalization hashes;
- explicit mid, guarded microprice, forecast, reservation, bid, and ask fields;
- signal confidence is not by itself a quote-eligibility gate;
- invalid or incoherent market state remains fail closed.

### Task 5 - Reservation Price And Inventory Skew

Formal task: `0718T016`, QA `已通过`.

- typed reservation and two-sided post-only quote construction;
- bounded inventory penalty;
- near-cap add-side suppression and reduce-side preservation;
- skew remains default-off until separately accepted live lifecycle evidence.

### Task 6 - Exchange-Reconciled Single-Level Manager

Formal task: `0718T017`, QA `已通过`.

- strategy-owned stable logical quote keys and generations;
- startup/reconnect reconciliation;
- strict owned/foreign order separation;
- ambiguous submit/cancel state remains counted until exchange reconciliation;
- anti-churn and partial-fill/cancel-pending exposure.

### Task 7 - Watcher Wiring And Single-Level Live Boundary

Formal task: `0718T018`, QA `已通过`.

- shared decision path wired to the reconciled single-level manager;
- atomic throttled live status;
- public shadow before bounded live;
- exact risk envelope, terminal open-orders proof, checksum, and same-window
  conservative replay;
- mechanism acceptance does not imply a fill or economics result.

### Task 8 - Event-Time Estimators And Dynamic Spread

Formal task: `0718T019`, QA `已通过`.

- deterministic event-time buckets;
- side-aware volatility, liquidity, toxicity, and arrival-intensity evidence;
- bounded dynamic half-spread candidate;
- fixed fallback and explicit activation state;
- observe-only evidence precedes any quote behavior change.

### Task 9 - Exposure-Weighted Fill Feedback

Formal task: `0718T020`, QA `已通过`.

- attempt/fill identity and censoring contract;
- exposure-weighted fill statistics;
- bounded feedback with hysteresis, rate limit, anti-windup, and checksummed
  restart state;
- no activation claim without eligible lifecycle evidence.

### Task 10 - Multi-Level Ladder

Initial contract: `0718T021`, QA `已通过`.

Final offline code/action-path readiness: `0722T055`, QA `已通过`.

- level 0 preserves authoritative single-level pricing;
- deeper levels are deterministic, post-only, and aggregate-cap checked;
- prerequisite, activation, mutation ordering, and status identity fail closed;
- multi-level live remains outside the accepted scope when the exact live
  envelope or durable restart provenance is insufficient.

### Task 11 - Real-Time Status Contract

Formal task: `0718T022`, QA `已通过`.

- versioned atomic status covering identity, market, pricing, quotes, signal,
  exposure, orders, fills, toxicity, risk, kill switch, activity, and process;
- writer failure is observable and fail closed/degraded by explicit policy;
- dashboard availability is not required for correctness.

### Task 12 - Same-Window Mechanism And Evidence Acceptance

Recovered accepted boundary:

- fixed-quote single-level mechanism/evidence baseline closed by
  `0721T044` plus `0721T046`;
- later dynamic, fill-feedback, and multi-level tasks preserve strict
  provenance, lifecycle, anti-optimism, and fail-closed acceptance;
- Task 12 acceptance is a mechanism/evidence result, not an economics result.

Required checks:

- exact runtime source and canonical command;
- stable run/window/attempt and account identity;
- exact decision and lifecycle joins;
- raw-response-derived submit/resting/reject/cancel/fill reconciliation;
- per-reference terminal proof;
- terminal open-orders/position proof;
- independently rebuilt checksum and acceptance outputs;
- producer blockers cannot be silently downgraded.

## Standing Safety Boundary

Historical work used conservative ceilings such as:

- Hyperliquid BTC only;
- post-only `Alo`;
- at most `0.005 BTC` per order;
- at most `0.01 BTC` aggregate position delta;
- at most `1 USDC` loss;
- at most `2` submissions;
- bounded single-window duration.

These values describe historical task envelopes. They are not reusable
authorization. A future task must restate and authorize its exact limits.

## Current State At Recovery

Latest formal QA at recovery is `0726T068`, status `阻塞`.

Accepted:

- exact seeded-dynamic source and seed identity;
- strict pre-submit dynamic gate and actual quote change;
- BTC-only post-only reject/resting/cancel lifecycle;
- same-account/same-client evidence;
- terminal zero open orders and zero BTC position;
- deterministic artifact rebuild and credential boundary.

Blocked:

- total fills are zero;
- no `confirmed_maker` or `confirmed_taker` row exists;
- fee/rebate, fill rate, markout, realized PnL, maker viability, promotion, and
  final MVP remain unsupported.

## Workflow Rule

Continue one formal task at a time:

```text
业务线程 -> QA验收线程 -> 总控
```

The latest QA report is the acceptance source of truth. Planning documents
must be reconciled when task status changes; historical summaries must not
override a newer QA result.
