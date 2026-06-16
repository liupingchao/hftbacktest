# Hyperliquid Tiny-Live Live-Capable Preflight Operator Packet

Task: `0616T006`

Status: live-capable preparation only.

This document defines the operator packet for a future Hyperliquid tiny-live
window on `awsserver1` with artifacts pulled back to the local machine for
validation. It does not authorize real order placement, cancellation,
amendment, live bot startup, account query, credential disclosure, signing,
nonce handling, user stream implementation, deployment, promotion, PnL proof,
or maker viability proof.

## Accepted Inputs

- `0616T002`: Hyperliquid private/order readiness boundary.
- `0616T003`: no-trading local private order artifact validator.
- `0616T004`: local fake shutdown dry-run proof gate.
- `0616T005`: tiny-live protocol design and human approval gate.

`0616T005` remains the current live approval gate. The operator packet may make
the future run operationally concrete, but it must not convert pending approval
fields into approved values.

## Host Boundary

The intended future execution host is `awsserver1`.

Required host preflight facts are recorded in
`awsserver1_host_preflight.csv`:

- SSH alias resolves to the intended host.
- Repository path is present.
- Branch is `cross-exchange`.
- Python / conda environment is recorded.
- Host clock, timezone, disk space, log path, and process owner are recorded.
- Binance and Hyperliquid public market-data reachability is checked.
- Private endpoints are not called during preflight.
- Artifact archive and checksum policy is present.

These checks are operator-verification inputs for a later task. They are not a
live run.

## Approval Fields

The live-capable configuration keeps the required fields explicit:

- symbol
- max notional
- max order size
- max position
- max loss
- duration
- host machine
- account scope
- whether real orders are allowed

`approval_fields.csv` records `host_machine=awsserver1` only as the proposed
future host. Its approval status remains `pending_controller_approval`.
`real_orders_allowed` remains `pending_controller_approval` with proposed value
`false_until_separate_controller_approval`.

## Artifact Pullback

A future approved run must preserve enough information to pull data back from
`awsserver1` and validate it locally:

- `operator_packet_manifest.json`
- `approval_fields.csv`
- `awsserver1_host_preflight.csv`
- `future_run_intent_marker.json`
- `public_market_data_manifest.json`
- `shutdown_evidence_placeholder.json`
- `artifact_pullback_manifest.json`
- `sha256sums.txt`

The local validator fails closed if the manifest is missing, the host is not
`awsserver1`, live authorization is not false for this task, real orders are not
pending approval, required approval fields are missing, or boundary checks fail.

## Inert Commands

`inert_operator_commands.csv` contains only non-trading command templates:

- host metadata check;
- remote artifact directory creation;
- artifact pullback with `rsync`;
- local artifact validation.

The packet intentionally does not generate private endpoint, signing, account,
order placement, order cancellation, amendment, or live bot startup commands.

## Local Validator

The local validator is:

`examples/hyperliquid/hyperliquid_tiny_live_operator_packet.py`

Supported commands:

- `generate-artifacts`
- `validate-artifacts`

Both commands are local/offline. They do not call exchange endpoints or read
credentials.

## Current Result

Official artifacts are under:

`local_live_analysis/hyperliquid_tiny_live_live_capable_preflight_operator_packet_0616T006/`

Final recommendation:

`hyperliquid_tiny_live_live_capable_preflight_operator_packet_ready_for_qa`

The next step, if QA accepts this task, is controller review of the pending live
approval fields before any separately dispatched tiny-live execution task.
