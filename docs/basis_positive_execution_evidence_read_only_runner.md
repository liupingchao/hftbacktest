# Basis-Positive Proof-Limited Read-Only Execution Evidence Runner

Task: `0615T007`

This module implements the first proof-limited local runner over the `0615T006` source-chain gate. It consumes only accepted local artifacts and emits proof-limited rows. It does not call exchange endpoints, read credentials, run live, place or cancel orders, change strategy behavior, compute PnL, recommend deployment, or claim maker viability.

## Inputs

- `0615T006` proof-limit contract artifacts.
- `0615T003` private order response local artifact.
- `0611T003` replay lifecycle local fixture artifact.
- `0615T004` account inventory local artifact.
- `0615T005` economics fee/rebate local artifact.

## Outputs

- `execution_evidence_rows.csv`: proof-limited rows only.
- `runner_validation_summary.csv`: valid, missing-source, and overclaim cases.
- `no_live_safety_audit.csv`: explicit no-live/no-endpoint/no-order-action checks.
- `boundary_validation.csv`: task boundary assertions.
- `execution_evidence_read_only_runner_manifest.json`: final recommendation and next-task pointer.

## Interpretation

`proof_limited_read_only_runner_ready_for_qa` means the runner mechanics are ready for QA/controller review. It does not authorize real execution metrics, PnL, live readiness, strategy decisions, deployment, promotion, or maker viability proof.

The next task is `0615T008`, which may design and dry-run a small-cap live-test protocol. The first live-capable task remains `0615T009` after `0615T008` QA and explicit total-control approval.
