# Basis-Positive Private Order Response Source-Line Contract

Task: `0610T006`

This document defines a design-only contract for `private_order_response_source_line`. It does not implement source readers, source collectors, private/order/account/live endpoints, user streams, signing, nonce handling, runner behavior, real execution metrics, strategy behavior, case libraries, shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## Prerequisite

`0610T005` QA is accepted with final recommendation `private_order_source_design_ready_next`. That recommendation means only that a separately dispatched design-only contract may be defined for the private order response source line. It does not authorize endpoint implementation or metric proof.

## Covered Gaps

This contract covers exactly the three primary gaps assigned by `0610T005`:

- `fill_probability`
- `post_only_reject_behavior`
- `real_order_lifecycle`

All three remain design labels only. Current artifacts do not prove fill probability, post-only reject behavior, real order lifecycle, PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion.

## Artifact Schema Boundary

The future artifact is a local, accepted response-artifact contract. It is not an endpoint specification. The schema must not contain endpoint URLs, API keys, secrets, signing payloads, nonce fields, user stream subscription details, executable order side, quote price, quote size, strategy signal, live gate, deployment flag, or promotion flag.

Required schema groups:

- Source provenance: task/source identifiers, venue, instrument, artifact source class, source policy, and validation status.
- Event identity: local artifact event id, client order id hash or opaque order reference, exchange order id hash or opaque exchange reference when available, and event sequence index.
- Timing: exchange event time, local receive time, artifact generation time, and validation or reconciliation time as separate fields.
- Response labels: response category, lifecycle state label, post-only classification, terminal-state marker, and unknown/missing/conflicting flags.
- Validation: fail-closed reason, validation gate id, overclaim rejection id, allowed future use, and forbidden current interpretation.

## Label Taxonomy

Response labels are design labels, not execution metrics.

- `accepted_design_label`: the response artifact reports that an order was accepted or acknowledged, but this alone is not fill probability or lifecycle proof.
- `rejected_design_label`: the response artifact reports rejection, with reject taxonomy required before post-only behavior can be interpreted.
- `filled_design_label` and `partially_filled_design_label`: fill-state labels that may later be inputs to a separately accepted metric design, not current fill-rate proof.
- `canceled_design_label`, `expired_design_label`, and `terminal_design_label`: terminal labels that require consistency checks before any lifecycle interpretation.
- `unknown_design_label`, `missing_design_label`, and `conflicting_design_label`: mandatory fail-closed labels.

## Post-Only Reject Policy

Post-only behavior must not be inferred from hypothetical quote position, public book state, or a generic reject label. A future artifact must carry a reject-code or reject-reason taxonomy, and unknown codes must fail closed.

Allowed design classifications:

- `post_only_reject_explicit_design_label`
- `non_post_only_reject_explicit_design_label`
- `reject_reason_unknown_fail_closed`
- `reject_reason_missing_fail_closed`
- `reject_reason_conflicting_fail_closed`
- `unsupported_venue_code_fail_closed`

## Lifecycle And Terminal Consistency

Real order lifecycle proof cannot be inferred from a single response label when required evidence is missing or contradictory. A terminal state must pass consistency checks across event order, terminal marker, response category, timestamp availability, and conflict flags.

Fail-closed cases include:

- Missing event identity.
- Missing exchange event time and local receive time.
- Terminal marker without a compatible lifecycle label.
- Multiple incompatible terminal labels for the same future opaque order reference.
- Fill/cancel/reject conflict without an accepted resolution policy.
- Response label present but source policy or validation status is not accepted.

## Timestamp Policy

The contract separates these times:

- `exchange_event_time`: source-reported exchange event time if present.
- `local_receive_time`: local receive time for the response artifact.
- `artifact_generated_time`: local artifact generation time.
- `validation_or_reconciliation_time`: time when the future artifact is validated or reconciled.

No future design may treat artifact generation time as exchange event time. If exchange event time or local receive time is missing where required, the relevant label must fail closed.

## Validation Gates

Validation gates are fail-closed. The required gate classes are:

- prerequisite source-line gate
- schema forbidden-field gate
- timing separation gate
- identity consistency gate
- response label taxonomy gate
- post-only reject taxonomy gate
- lifecycle terminal consistency gate
- unknown/missing/conflicting event gate
- overclaim rejection gate
- boundary preservation gate

## Overclaim Rules

This contract rejects current claims of:

- fill probability proof
- post-only reject behavior proof
- real order lifecycle proof
- PnL or realized economics proof
- maker execution viability proof
- live/default-on/tiny-live readiness
- deployment readiness
- promotion readiness

## Allowed Future Use

If accepted by QA, this contract may become an input to a later separately dispatched design task. It may define the format a future artifact should satisfy. It does not authorize any endpoint use, source collection, runner extension, execution metric computation, or trading behavior.

Final recommendation: `private_order_response_contract_ready_for_qa`.
