# Basis-Positive Economics Fee/Rebate Read-Only Source

Task: `0615T005`

This document describes the no-trading local economics fee/rebate read-only source implementation. It implements the accepted `0612T001` validator boundary as a local transform into the accepted `economics_fee_rebate_source.py` schema.

## Scope

The implementation is limited to:

- task-local fixture input rows;
- local transform into redacted economics fee/rebate artifact rows;
- opaque hash generation for account scope and future fill references;
- settlement, conversion, receive, reconciliation, artifact, and validation timestamp-domain preservation;
- fee/rebate/net-fee, conversion, tick-value, and fee-adjusted spread arithmetic preservation;
- fail-closed checks for forbidden endpoint, credential, signing, nonce, user-stream, strategy, live, deployment, and promotion fields;
- validation through `economics_fee_rebate_source.py`;
- local CSV/JSON artifacts and tests.

It does not implement or call endpoints, endpoint clients, signed requests, nonce handling, user streams, real private/order/account/live/economics data reads, remote execution, collection, order placement, order cancellation, order amendment, runner consumption, strategy behavior, live/default-on/tiny-live behavior, parameter search, deployment, promotion, real economics metrics, execution metrics, PnL proof, or maker viability proof.

## Implementation

The implementation is:

- `examples/binance_tick_mm/economics_fee_rebate_read_only_source.py`

The CLI supports:

- `python examples/binance_tick_mm/economics_fee_rebate_read_only_source.py --help`
- `python examples/binance_tick_mm/economics_fee_rebate_read_only_source.py generate-artifacts --output-dir local_live_analysis/basis_positive_economics_fee_rebate_read_only_source_0615T005`

## Source Behavior

The source reads task-local fixture rows and emits `economics_output_artifact.csv` using the accepted `economics_fee_rebate_source.py` fields. Account scope and future fill references are hashed or made opaque before artifact output. The official output is then validated through `economics_fee_rebate_source.validate_rows`.

The source fails closed before artifact output when input rows contain forbidden fields such as endpoint URLs, API keys, secrets, signatures, nonce payloads, user-stream subscriptions, account IDs, order/action fields, quote fields, strategy signals, live gates, deployment flags, or promotion flags.

## Artifacts

Official artifacts:

- `economics_fixture_inputs.csv`
- `economics_output_artifact.csv`
- `economics_validation_summary.csv`
- `redaction_audit.csv`
- `no_trading_safety_audit.csv`
- `economics_fee_rebate_read_only_source_manifest.json`
- `boundary_validation.csv`

Final recommendation: `economics_fee_rebate_read_only_source_ready_for_qa`.

This means only that the no-trading local economics fee/rebate read-only source implementation is ready for QA/controller review. It does not authorize real venue economics collection, endpoint usage, credentials/signing/nonce/user-stream work, runner consumption, fees/rebates/spread-capture proof, real economics metrics, strategy/live behavior, deployment, promotion, PnL proof, or maker viability proof.
