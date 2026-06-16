# Hyperliquid Private Order Artifact Validator

Task: `0616T003`

This module implements a no-trading local fixture / validator for the
`0616T002` Hyperliquid private/order readiness boundary.

It validates task-local fixture rows only. It does not call Hyperliquid
endpoints, read credentials, sign requests, manage nonce values, subscribe to
user streams, query accounts, place orders, cancel orders, run live, change
strategy behavior, prove PnL, deploy, or promote.

## Implementation

- `examples/hyperliquid/hyperliquid_private_order_artifact_validator.py`
- `examples/hyperliquid/test_hyperliquid_private_order_artifact_validator.py`

The validator checks:

- required field presence;
- Hyperliquid venue identity;
- lifecycle and post-only enum values;
- local request / exchange ack / local response / private stream / validation
  timestamp ordering;
- duplicate event identity;
- terminal state consistency;
- forbidden endpoint, credential, signing, nonce, user-stream, action, strategy,
  live, deployment, and promotion fields.

## Artifacts

Generated under:

- `local_live_analysis/hyperliquid_private_order_artifact_validator_0616T003/`

Key outputs:

- `hyperliquid_private_order_validator_manifest.json`
- `fixture_private_order_events.csv`
- `accepted_artifact_rows.csv`
- `fail_closed_artifact_rows.csv`
- `validator_result_summary.csv`
- `validator_issue_details.csv`
- `boundary_validation.csv`

Final recommendation: `hyperliquid_private_order_artifact_validator_ready_for_qa`.

This means only that the local no-trading fixture / validator is ready for QA.
It does not authorize endpoint use, private data collection, real order
placement, real cancellation, account query, user stream, strategy behavior,
live/default-on/tiny-live, deployment, promotion, or PnL proof.
