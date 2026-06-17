# Hyperliquid Tiny-Live Real-Order Executor

Task: `0618T001`

This document describes the minimal live-capable Hyperliquid tiny-live executor
introduced for the repaired final gate. It is not a live execution report.

## Boundary

`0618T001` implements the real-order code path and proves it through local
self-test / mock-client artifacts. It does not place real orders, cancel real
orders, query real account state, read credentials, disclose secrets, start a
live bot, deploy, promote, or prove PnL.

The executor remains fail-closed unless all hard caps and live gates are present:

- symbol: `BTC`
- duration: at most `600` seconds
- max order size: `0.01 BTC`
- max order notional: `700 USDC`
- max position: `0.04 BTC`
- max position notional: `2800 USDC`
- max notional: `3000 USDC`
- max loss: `30 USDC`
- order type: limit only
- time-in-force: Hyperliquid `Alo` only

Live mode also requires the exact operator acknowledgement
`I_UNDERSTAND_THIS_CAN_PLACE_REAL_HYPERLIQUID_ORDERS`.

## Official Interface Checks

The implementation relies on the official Hyperliquid Python SDK for signed
exchange actions. It does not implement signing, nonce handling, or raw
`/exchange` payload signing itself.

The rechecked SDK surface used by the wrapper is:

- `Exchange.order(name, is_buy, sz, limit_px, {"limit": {"tif": "Alo"}}, ...)`
- `Exchange.cancel(name, oid)`
- `Exchange.cancel_by_cloid(name, cloid)`
- `Exchange.schedule_cancel(time)`
- `Info.open_orders(address)`
- `Info.user_state(address)`
- `Info.query_order_by_oid(address, oid)`

The task artifacts also record the official documentation URLs for API overview,
exchange endpoint, info endpoint, tick/lot size, signing, nonces/API wallets,
rate limits/user limits, and the official SDK.

## What Can Be Tested Without Orders

Without placing a real order, the task can test:

- cap validation and fail-closed behavior
- post-only `Alo` order-intent construction
- max-loss pre-order stop behavior
- cancel-all control flow over tracked oid/cloid references
- artifact creation and redaction
- SDK dependency detection
- final gate consumption of executor evidence

It cannot fully prove the real submit-order API path succeeds. That proof needs
a later separately approved tiny-live or canary task that actually submits a
real post-only order and then cancels/shuts down under the approved caps.

## Artifacts

Primary local artifacts:

- `local_live_analysis/hyperliquid_tiny_live_real_order_executor_0618T001/executor_manifest.json`
- `local_live_analysis/hyperliquid_tiny_live_real_order_executor_0618T001/order_intent_audit.csv`
- `local_live_analysis/hyperliquid_tiny_live_real_order_executor_0618T001/cancel_shutdown_proof.json`
- `local_live_analysis/hyperliquid_tiny_live_real_order_executor_0618T001/final_safety_summary.json`

The repaired final gate consumes:

- `local_live_analysis/hyperliquid_tiny_live_real_order_executor_0618T001/executor_manifest.json`
- current `awsserver1` remote state facts
- accepted `0617T003` caps and `0617T006` `canonical_7` evidence

`allow_create_0617T008=true` only means total control may create a later live
task. It does not execute live orders inside `0618T001`.
