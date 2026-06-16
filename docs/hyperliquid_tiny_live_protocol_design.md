# Hyperliquid Tiny-Live Protocol Design

Task: `0616T005`

Status: design-only / human approval gate.

This document defines the structure of a future Hyperliquid tiny-live data
collection task. It does not authorize private endpoint calls, credentials,
signing, nonce handling, user streams, account queries, real order placement,
real cancellation, live startup, deployment, promotion, or PnL proof.

## Accepted Readiness Inputs

- `0616T002`: Hyperliquid private/order readiness boundary.
- `0616T003`: no-trading local private order artifact fixture / validator.
- `0616T004`: local fake cancel-all / shutdown dry-run proof gate.

These inputs are necessary but not sufficient for live execution. They define
artifact shape and dry-run proof levels only.

## Human Approval Fields

The following must be approved by the controller before any future live task:

- symbol
- max notional
- max order size
- max position
- max loss
- duration
- host / machine
- account scope
- whether real orders are allowed

All fields are currently `pending_controller_approval`.

## Required Future Live Evidence

A future live task, if separately approved, must preserve:

- deployment manifest;
- exact git commit and dirty-state policy;
- start and stop markers;
- market-data raw artifacts;
- private order response artifacts;
- account/inventory artifacts;
- economics/fee/funding/settlement artifacts where authorized;
- cancel-all / shutdown evidence;
- open-order reconciliation evidence;
- logs, archive, and checksums;
- immediate post-run safety summary.

## Stop Conditions

A future protocol must stop immediately on:

- loss cap breach;
- position cap breach;
- reject count cap breach;
- post-only contradiction;
- missing private/order artifact emission;
- missing account/inventory reconciliation;
- missing shutdown evidence;
- latency or stale-data breach;
- operator stop.

## Current Boundary

This task stops at protocol design. It must not be used as permission to run a
live task. The next action is controller review of the approval fields.

Final recommendation: `hyperliquid_tiny_live_protocol_design_ready_for_qa`.
