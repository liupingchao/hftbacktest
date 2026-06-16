# Basis-Positive Private Order Response Read-Only Collector

Task: `0615T003`

This document describes the no-trading local read-only collector implementation for private order response artifacts. It implements the accepted `0615T002` boundary as a local transform into the accepted `0611T002` `private_order_response_source.py` schema.

## Scope

The implementation is limited to:

- task-local fixture input rows;
- local transform into redacted private order response artifact rows;
- opaque hash generation for client and exchange order references;
- timestamp-domain preservation;
- fail-closed checks for forbidden action, endpoint, credential, signing, nonce, user-stream, strategy, live, deployment, and promotion fields;
- validation through `private_order_response_source.py`;
- local CSV/JSON artifacts and tests.

It does not implement or call endpoints, endpoint clients, signed requests, nonce handling, user streams, real private/order/account/live/economics data reads, remote execution, venue collection, order placement, order cancellation, order amendment, runner consumption, strategy behavior, live/default-on/tiny-live behavior, parameter search, deployment, promotion, execution metrics, economics metrics, PnL proof, or maker viability proof.

## Implementation

The implementation is:

- `examples/binance_tick_mm/private_order_response_read_only_collector.py`

The CLI supports:

- `python examples/binance_tick_mm/private_order_response_read_only_collector.py --help`
- `python examples/binance_tick_mm/private_order_response_read_only_collector.py generate-artifacts --output-dir local_live_analysis/basis_positive_private_order_response_read_only_collector_0615T003`

## Collector Behavior

The collector reads task-local fixture rows and emits `collector_output_artifact.csv` using the accepted `private_order_response_source.py` fields. Client and exchange order references are hashed before artifact output. The official output is then validated through `private_order_response_source.validate_rows`.

The collector fails closed before artifact output when input rows contain forbidden fields such as endpoint URLs, API keys, secrets, signatures, nonce payloads, user-stream endpoints, order placement/cancellation/amendment fields, quote fields, strategy signals, live gates, deployment flags, or promotion flags.

## Artifacts

Official artifacts:

- `collector_fixture_inputs.csv`
- `collector_output_artifact.csv`
- `collector_validation_summary.csv`
- `redaction_audit.csv`
- `no_trading_safety_audit.csv`
- `private_order_response_read_only_collector_manifest.json`
- `boundary_validation.csv`

Final recommendation: `private_order_response_read_only_collector_ready_for_qa`.

This means only that the no-trading local read-only collector implementation is ready for QA/controller review. It does not authorize real venue collection, endpoint usage, credentials/signing/nonce/user-stream work, runner consumption, metrics, strategy/live behavior, deployment, promotion, PnL proof, or maker viability proof.
