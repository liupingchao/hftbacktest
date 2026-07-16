# Future Controlled Evidence Task Template

This template is produced by `0716T005`. It is not an authorization to run live.

## Required Controller Inputs

- New task id: `<TBD>`
- Exact UTC schedule: `<required if live>`
- Host/account scope: `<required if live>`
- Live envelope: `<symbol, size, max submissions, duration, post-only, max loss>`
- Source branch/commit: `<required>`
- Artifact output root: `<required>`

## Required Boundaries

- No threshold change.
- No quote-envelope change.
- No order-size or max-submission expansion.
- No quote-policy implementation.
- No fee/PnL calibration inside the evidence acquisition task.
- No maker viability, T012, promotion, or final MVP claim.

## Required Artifacts

- `fill_liquidity_role_evidence.csv`
- `user_fills_pullback_audit.json`
- `live_fill_ledger.csv`
- `order_intent_audit.csv`
- `private_order_response_audit.json`
- `resting_interval_lifecycle_matrix.csv`
- `public_stream_coverage.csv`
- `boundary_manifest.json`

## Required Acceptance Checks

1. All generated JSON files parse.
2. All generated CSV files parse and include required headers.
3. Every attributed fill has a stable `attempt_key`.
4. Every attributed fill has `role_status` in:
   - `confirmed_maker`
   - `confirmed_taker`
   - `unknown_liquidity_role`
5. Any `unknown_liquidity_role` row blocks fee/PnL calibration.
6. Missing `user_fills_pullback_audit.json` blocks fill source-path acceptance.
7. Missing exact/source-tagged `fill_time_ms` blocks exact markout.
8. Missing interval coverage blocks public-flow adverse-selection interpretation.
9. Boundary manifest confirms no live/parameter/quote-policy/economics scope expansion beyond the task authorization.

## Required Final Route Values

- `route_to_public_shadow_with_role_source_evidence`
- `route_to_source_path_repair`
- `route_to_controlled_evidence_rerun`
- `blocked_boundary_violation`

