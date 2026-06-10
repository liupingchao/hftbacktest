# Basis-Positive Replay Lifecycle Semantics Source-Line Contract

Task: `0610T007`

This document defines a design-only contract for `replay_lifecycle_semantics_source_line`. It does not implement replay/live semantics, source readers, source collectors, runners, private/order/account/live endpoints, user streams, signing, nonce handling, real execution metrics, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Prerequisites

`0610T006` QA is accepted with final recommendation `private_order_response_contract_ready_for_qa`. That contract may serve only as future cross-check or future event-source dependency context for this source line. It is not a current proof source for queue priority or cancel-fill race.

`0610T005` QA is accepted with final recommendation `private_order_source_design_ready_next`. Its mapping assigns exactly these two primary gaps to `replay_lifecycle_semantics_source_line`:

- `queue_priority`
- `cancel_fill_race`

## Covered Gaps

This contract covers exactly the two primary gaps assigned by `0610T005`:

- `queue_priority`
- `cancel_fill_race`

Both remain future design labels only. Current artifacts do not prove queue priority, exact queue position, cancel-fill race, PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion.

## Event Schema Boundary

The future replay/live lifecycle event artifact is a local design contract. It is not a source reader, collector, endpoint, runner, strategy signal, or execution instruction.

Required schema groups:

- Event identity: lifecycle event id, opaque order reference, event type, lifecycle state design label, event sequence index, and ordering scope.
- Source provenance: source line id, upstream task id, artifact source class, replay/live domain, source policy, validation status, and proof-limit class.
- Time domains: decision time, replay time, exchange event time when available, local receive time, artifact generation time, and validation or reconciliation time as separate fields.
- Causal ordering: same-order causal sequence, cross-order ordering scope, predecessor and successor event references, ambiguity flag, conflict flag, and out-of-order flag.
- Proof limits: queue observation class, queue proof status, cancel-fill race observation class, race proof status, fail-closed reason, allowed future use, and forbidden current interpretation.

Forbidden schema fields include endpoint URLs, credentials, secrets, signing payloads, nonce fields, user stream details, order side, quote price, quote size, executable actions, strategy signals, live gates, deployment flags, and promotion flags.

## Queue Semantics Boundary

Queue priority cannot be proven from current public, replay, local audit, private-response, or proxy artifacts. Exact queue position is rejected. Queue-ahead, touch-age, join-age, replay fill order, public top-of-book, and local audit observations are allowed only as future diagnostic labels after a separate accepted semantics and validation task.

Allowed future labels are diagnostic only:

- `queue_observation_public_book_context_design_label`
- `queue_observation_replay_model_context_design_label`
- `queue_observation_local_audit_context_design_label`
- `queue_ahead_proxy_diagnostic_design_label`
- `queue_priority_unknown_fail_closed`
- `queue_priority_conflicting_fail_closed`

No future diagnostic label may become a trigger, case-library condition, shadow decision, live decision, parameter-search objective, deployment criterion, or promotion criterion inside this task.

## Cancel/Fill Race Ordering Policy

Cancel-fill race interpretation requires a future accepted event-ordering policy. This task defines only the policy boundary:

- Same-order ordering may compare submit, ack, cancel request, cancel acknowledgement, fill, partial fill, reject, expire, and terminal design events only when identity and required time domains are valid.
- Cross-order ordering is context only unless a later accepted policy defines a specific causal scope.
- Exchange event time is preferred when available and validated; local receive time may be a fallback ordering domain only when the policy records the fallback and its proof limit.
- Replay time can support regression and model comparison, but it is not execution proof.
- Missing, unknown, ambiguous, conflicting, incomplete, out-of-order, or unsupported event sequences fail closed.

This contract does not compute cancel-fill race metrics or prove adverse-selection cancel races.

## Timestamp Policy

The contract separates these time domains:

- `decision_time`: decision-time visible context timestamp.
- `replay_time`: replay engine or simulation timeline timestamp.
- `exchange_event_time`: exchange-reported event time when available and validated.
- `local_receive_time`: local observation time for a future lifecycle event artifact.
- `artifact_generated_time`: local artifact generation time.
- `validation_or_reconciliation_time`: time when the future artifact is validated or reconciled.

No future design may treat artifact generation time as exchange event time, replay time as live event time, or local receive time as exchange event time without an explicit fail-closed proof-limit rule.

## Replay/Live Proof Limits

Replay remains supporting regression, not execution proof. Replay can compare model behavior, lifecycle state transitions, and deterministic artifact handling. Replay cannot prove real queue priority, exact queue position, real cancel-fill race rate, real PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion.

Live-derived lifecycle events would require a later separately scoped source contract, validation oracle, permission boundary, and QA acceptance. This task does not authorize collecting, reading, or using such data.

## Private Order Response Dependency Boundary

`private_order_response_source_line` may later provide event-source dependency or cross-check context for lifecycle labels. It cannot by itself prove queue priority or cancel-fill race because those require queue semantics, causal event ordering, and replay/live proof-limit validation. Any contradiction between future private response context and replay lifecycle semantics must fail closed until a separately accepted reconciliation policy exists.

## Validation Gates

Validation gates are fail-closed. The required gate classes are:

- prerequisite QA and source mapping gate
- design-only boundary gate
- schema forbidden-field gate
- two-gap coverage gate
- timestamp separation gate
- queue exact-position rejection gate
- queue diagnostic-only gate
- cancel-fill event-ordering policy gate
- unknown/missing/ambiguous/conflicting event gate
- replay-as-regression-only gate
- private-response future-context-only gate
- overclaim rejection gate

## Overclaim Rules

This contract rejects current claims of:

- queue priority proof
- exact queue position proof
- cancel-fill race metric proof
- PnL or realized economics proof
- maker execution viability proof
- live/default-on/tiny-live readiness
- deployment readiness
- promotion readiness

## Allowed Future Use

If accepted by QA, this contract may become an input to a later separately dispatched design task. It may define the format a future replay/live lifecycle semantics artifact should satisfy. It does not authorize replay/live semantic implementation, source collection, runner extension, execution metric computation, or trading behavior.

Final recommendation: `replay_lifecycle_contract_ready_for_qa`.
