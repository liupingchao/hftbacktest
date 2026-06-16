# Hyperliquid Private/Order Readiness Boundary

Task: `0616T002`

Status: design-only readiness boundary.

This document defines what must exist before Binance-led Hyperliquid maker
research may move toward private/order execution evidence. It does not implement
or call private endpoints, use credentials, sign payloads, manage nonce values,
subscribe to user streams, query accounts, place orders, cancel orders, run live,
change strategy behavior, prove PnL, deploy, or promote.

## Accepted Context

- `0616T001` corrected the `cross-exchange` branch objective to Binance lead /
  Hyperliquid lag maker research.
- `0601T004`, `0601T005`, and `0609T002` provide public-only read-only pricing
  context. They do not authorize execution behavior.
- `0615T001-T007` may be used as methodology templates only. They are not
  Hyperliquid private/order readiness artifacts.
- `0615T008` is Binance `BTCUSDT` live-risk reference only and is not a
  Hyperliquid live predecessor.

## Boundary Decision

Hyperliquid private/order evidence must be introduced through separate,
QA-accepted source lines before any live order task:

1. private/order response and lifecycle evidence;
2. account/inventory evidence;
3. economics, fees, rebates, funding, and settlement evidence;
4. cancel-all / shutdown proof evidence;
5. source-chain runner-consumption gate;
6. proof-limited execution evidence runner;
7. Hyperliquid-specific live-risk protocol with explicit human approval.

## Public Lead Context

Binance lead features remain read-only context:

- `binance_top5_imbalance`
- `binance_microprice_minus_mid_ticks`
- `binance_mid_move_ticks_from_prev`
- `binance_top5_bid_qty`

These may condition future research rows, but they must not become order side,
quote price, quote size, leverage, stop rule, take-profit rule, or live decision
fields inside this boundary.

## Future Private/Order Evidence Authority

Future Hyperliquid private/order artifacts must distinguish:

- order intent design label;
- client order reference and venue order reference after redaction;
- local request time, exchange acknowledgement time when available, local
  response time, private stream receive time, and validation time;
- post-only / maker-only instruction label;
- accepted, rejected, canceled, filled, partially filled, expired, unknown, and
  conflicting lifecycle labels;
- terminal state consistency;
- cancel request and cancel acknowledgement evidence;
- source authority: REST response, private stream event, info query, local
  artifact, or fail-closed placeholder.

## Post-Only Semantics

Post-only behavior is not proven by an intent flag alone. A future artifact must
separate:

- `post_only_intent_declared`
- `post_only_acceptance_observed`
- `post_only_reject_observed`
- `maker_fill_observed`
- `taker_fill_or_cross_detected_fail_closed`
- `post_only_semantics_unavailable_fail_closed`

Any future runner must fail closed if post-only intent exists but response,
fill, or reject evidence is missing or contradictory.

## Required Fail-Closed Gates

Future private/order tasks must fail closed for:

- missing source authority;
- missing or merged timestamp domains;
- duplicate or conflicting client/order references;
- unknown lifecycle state;
- terminal state contradiction;
- out-of-order lifecycle evidence;
- intent-only post-only proof;
- fill-only lifecycle proof;
- missing cancel-all or shutdown evidence;
- account/inventory/economics overclaim;
- PnL, maker viability, live readiness, deployment, or promotion overclaim.

## Next Task

The next auto-loop task may implement only a no-trading local fixture and
validator for this boundary. It must not call Hyperliquid endpoints, read
credentials, sign requests, manage nonce values, subscribe to user streams, query
accounts, place orders, cancel orders, or run live.

Final recommendation: `hyperliquid_private_order_readiness_boundary_ready_for_qa`.
