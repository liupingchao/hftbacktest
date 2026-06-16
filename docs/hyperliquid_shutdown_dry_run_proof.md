# Hyperliquid Shutdown Dry-Run Proof

Task: `0616T004`

This module implements a local fake cancel-all / shutdown proof gate. It proves
the artifact shape and fail-closed classifications needed before a future live
protocol can discuss real exchange-side shutdown evidence.

It does not call Hyperliquid endpoints, query open orders, read credentials,
sign requests, manage nonce values, subscribe to user streams, place orders,
cancel orders, run live, deploy, or promote.

## Proof Levels

- `local_fake_proof_only`: local fake orders reached local fake terminal states.
- `insufficient_proof`: terminal state evidence is missing or contradictory.
- Future exchange-side no-open-order proof remains unproven by this task.

## Artifacts

Generated under:

- `local_live_analysis/hyperliquid_shutdown_dry_run_proof_0616T004/`

Key outputs:

- `shutdown_dry_run_manifest.json`
- `startup_manifest.json`
- `fake_open_orders.csv`
- `cancel_intent_log.csv`
- `local_terminal_proof.csv`
- `fail_closed_scenarios.csv`
- `proof_level_summary.csv`
- `boundary_validation.csv`

Final recommendation: `hyperliquid_shutdown_dry_run_proof_ready_for_qa`.
