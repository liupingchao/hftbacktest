# Basis-Positive Economics Fee/Rebate Source-Line Contract

Task: `0610T009`

This document defines a design-only contract for `economics_fee_rebate_source_line`. It does not implement economics, fee, rebate, settlement, account, private/order, or live endpoints; source readers; source collectors; runners; user streams; signing; nonce handling; real economics metrics; real execution metrics; PnL proof; strategy behavior; case libraries; shadow decisions; parameter search; deployment; promotion; or execution-layer maker viability proof.

## Prerequisites

`0610T008` QA is accepted with final recommendation `account_inventory_contract_ready_for_qa`. Its account inventory contract may serve only as future reconciliation context. It is not current economics, fee, rebate, spread-capture, or PnL proof.

`0610T007` QA is accepted with final recommendation `replay_lifecycle_contract_ready_for_qa`. Its replay lifecycle contract may serve only as future timestamp or order-consistency context. Replay remains supporting regression, not economics proof.

`0610T006` QA is accepted with final recommendation `private_order_response_contract_ready_for_qa`. Its private-order response contract may serve only as future fill dependency context. Fill notional or order fills alone cannot prove realized fees, rebates, spread capture, or PnL.

`0610T005` QA is accepted with final recommendation `private_order_source_design_ready_next`, and its source-line mapping assigns `fees_rebates_spread_capture` to `economics_fee_rebate_source_line`.

## Covered Gap

This contract covers exactly one primary gap assigned by `0610T005`:

- `fees_rebates_spread_capture`

The gap remains a design label only. Current artifacts do not prove realized fees, realized rebates, spread capture, realized economics, PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion.

## Core Boundary

Realized economics require accepted economics settlement records, fee/rebate arithmetic, maker/taker classification, currency and unit validation, conversion or tick-value policy, settlement timing, and fail-closed reconciliation.

The following evidence is insufficient by itself:

- hypothetical spread
- fill notional
- order fills alone
- public markout alone
- account inventory alone
- replay lifecycle alone

Each may later become context under a separately accepted design and validation policy, but none is current proof of fees, rebates, spread capture, realized economics, or PnL.

## Artifact Schema Boundary

The future economics artifact is a local accepted artifact contract. It is not an endpoint specification. The schema must not contain endpoint URLs, API keys, secrets, signing payloads, nonce fields, user stream subscription details, executable order side, quote price, quote size, strategy signal, live gate, deployment flag, or promotion flag.

Required schema groups:

- Settlement identity: settlement record id, settlement version, venue, instrument, account scope or opaque account reference, settlement scope, and validation status.
- Source provenance: source artifact id, source class, source policy, source task id, prior source-line context, validation gate id, and fail-closed reason.
- Fill dependency reference: opaque future fill reference, fill dependency status, fill dependency source line, fill validation status, and forbidden fill-only interpretation.
- Maker/taker classification: maker/taker label, classification evidence type, venue-rule dependency status, unknown/conflict flags, and classification validation status.
- Fee and rebate amounts: fee amount, rebate amount, net fee amount, amount sign convention, fee currency, rebate currency, settlement currency, precision, rounding policy, and arithmetic validation status.
- Currency and tick conversion: base asset, quote asset, fee asset, rebate asset, conversion source provenance, conversion timestamp, conversion rate, tick size, tick value, conversion tolerance, and conversion validation status.
- Spread-capture fields: quoted spread context, filled spread context, realized spread design label, markout context, hypothetical spread flag, spread-capture proof status, and spread validation status.
- Time fields: fill time, exchange settlement time, local receive time, account/economics reconciliation time, artifact generated time, and validation time as separate fields.
- Proof limits: allowed future use, forbidden current interpretation, overclaim rejection id, current proof status, and design-only marker.

## Fee/Rebate Settlement Taxonomy

Settlement labels are design labels, not current economics metrics:

- `maker_fee_settlement_design_label`
- `maker_rebate_settlement_design_label`
- `taker_fee_settlement_design_label`
- `zero_fee_settlement_design_label`
- `funding_commission_adjustment_design_label`
- `missing_settlement_fail_closed`
- `delayed_settlement_fail_closed`
- `partial_settlement_fail_closed`
- `unknown_settlement_fail_closed`
- `ambiguous_settlement_fail_closed`
- `conflicting_settlement_fail_closed`
- `unsupported_settlement_fail_closed`

Missing, delayed, partial, unknown, ambiguous, conflicting, and unsupported settlement evidence cannot support current fee, rebate, spread-capture, or PnL proof.

## Spread-Capture Taxonomy

Spread-capture labels must separate context from realized proof:

- `quoted_spread_context_design_label`
- `filled_spread_context_design_label`
- `realized_spread_design_label`
- `markout_context_design_label`
- `hypothetical_spread_context_design_label`
- `missing_spread_capture_fail_closed`
- `ambiguous_spread_capture_fail_closed`
- `conflicting_spread_capture_fail_closed`
- `unsupported_spread_capture_fail_closed`

Quoted spread, filled spread, markout context, and hypothetical spread are not current PnL proof. Realized spread remains a design label until settlement, conversion, maker/taker, and arithmetic gates are accepted by a separately scoped task.

## Maker/Taker Classification Policy

Maker/taker classification must be explicit and validated before any future economics design may use fee/rebate labels.

Accepted future evidence classes may include an accepted economics settlement artifact, accepted exchange settlement label, or accepted venue-rule classification record. A private fill label, public book context, or replay lifecycle label alone is insufficient.

Fail-closed classifications include:

- `maker_taker_missing_fail_closed`
- `maker_taker_unknown_fail_closed`
- `maker_taker_ambiguous_fail_closed`
- `maker_taker_conflicting_fail_closed`
- `venue_rule_dependency_unaccepted_fail_closed`
- `unsupported_classification_fail_closed`

If classification is missing, conflicting, unknown, venue-rule-dependent without accepted rule provenance, or unsupported, the economics record must fail closed.

## Currency Conversion And Tick-Value Policy

Economics records must preserve units and conversion provenance:

- Base asset, quote asset, fee currency, rebate currency, and settlement currency must be separate fields.
- Conversion timestamp must be separate from fill time and settlement time.
- Conversion source provenance must be explicit and validated.
- Tick size and tick value must be recorded with units and precision.
- Rounding and tolerance policy must be explicit.
- Missing conversion, conflicting conversion, unit-inconsistent conversion, precision-invalid conversion, and unsupported conversion must fail closed.

No future design may treat nominal fill notional, base quantity, quote quantity, or unconverted fee currency as a common PnL unit without accepted conversion and tick-value validation.

## Settlement Timestamp Policy

The contract separates these time domains:

- `fill_time`: future fill observation time when available.
- `exchange_settlement_time`: source-reported exchange settlement time when available and validated.
- `local_receive_time`: local observation time for the economics artifact.
- `account_economics_reconciliation_time`: time when account/inventory and economics settlement are reconciled.
- `artifact_generated_time`: local artifact generation time.
- `validation_time`: time when the artifact is validated.

No future design may treat artifact generation time as exchange settlement time, local receive time as fill time without proof-limit disclosure, or replay time as settlement time. Missing, delayed, out-of-order, contradictory, or unsupported timing evidence must fail closed.

## Reconciliation Boundary

The contract separates these evidence authorities:

- Private order/fill observations: future fill dependency context only.
- Economics settlement records: future authority for fees, rebates, settlement amounts, and economics arithmetic after validation.
- Account/inventory records: future reconciliation context for balances, positions, and settlement effects; not economics proof alone.
- Replay lifecycle observations: supporting regression or event-order context only; not economics proof.
- Public market markout context: future markout context only; not realized spread capture or PnL proof.
- Future endpoint or collector responsibilities: explicitly out of scope for this task.

Contradictions between fill observations, economics settlement records, account inventory records, replay lifecycle observations, or public markout context must fail closed until a separately accepted reconciliation policy exists.

## Validation Gates

Validation gates are fail-closed. The required gate classes are:

- prerequisite QA and source mapping gate
- design-only boundary gate
- schema forbidden-field gate
- one-gap coverage gate
- settlement identity gate
- source provenance gate
- fill dependency reference gate
- maker/taker classification gate
- fee/rebate arithmetic gate
- currency unit and asset identity gate
- conversion and tick-value gate
- settlement timestamp separation gate
- spread-capture taxonomy gate
- reconciliation boundary gate
- duplicate settlement gate
- out-of-order settlement gate
- missing delayed partial unknown ambiguous conflicting evidence gate
- non-arithmetic evidence gate
- unit precision and rounding gate
- unsupported economics evidence gate
- overclaim rejection gate

## Overclaim Rules

This contract rejects current claims of:

- fee proof from settlement labels that are missing, delayed, partial, unknown, ambiguous, conflicting, unsupported, non-arithmetic, unit-inconsistent, precision-invalid, conversion-missing, or conversion-conflicting
- rebate proof under the same fail-closed conditions
- spread-capture proof from hypothetical spread, quoted spread, filled spread, fill notional, public markout alone, account inventory alone, replay lifecycle alone, or order fills alone
- realized economics proof
- PnL proof
- maker execution viability proof
- live/default-on/tiny-live readiness
- deployment readiness
- promotion readiness
- future labels as strategy, shadow, case-library, parameter-search, live-decision, or deployment criteria

## Allowed Future Use

If accepted by QA, this contract may become an input to a later separately dispatched design task. It may define the format a future economics fee/rebate artifact should satisfy. It does not authorize economics endpoint use, source collection, runner extension, economics metric computation, PnL proof, or trading behavior.

Final recommendation: `economics_fee_rebate_contract_ready_for_qa`.
