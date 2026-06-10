# 0610T004 Fail-Closed Runner Report

task_id: `0610T004`

schema_version: `basis_positive_execution_evidence_fail_closed_runner_v1`

final_recommendation: `fail_closed_runner_skeleton_ready_for_qa`

This runner is a local fail-closed/read-only skeleton. It validates accepted
T003/T002 gate and contract artifacts, then emits proof-limited unavailable
status rows. It does not authorize real execution metrics, private/order/account
or live data use, strategy decisions, case libraries, shadow decisions, parameter
search, deployment, promotion, or execution-layer maker viability proof.

## Generated Artifacts

- `fail_closed_runner_manifest.json`
- `execution_gap_status_rows.csv`
- `source_policy_validation.csv`
- `output_schema_validation.csv`
- `overclaim_reject_validation.csv`
- `boundary_validation.csv`
- `fail_closed_runner_report.md`

## Seven Gap Status

- `fill_probability`: `unavailable_proof_limited` / `unavailable`
- `queue_priority`: `unavailable_proof_limited` / `unavailable`
- `post_only_reject_behavior`: `unavailable_proof_limited` / `unavailable`
- `cancel_fill_race`: `unavailable_proof_limited` / `unavailable`
- `fees_rebates_spread_capture`: `unavailable_proof_limited` / `unavailable`
- `inventory_lifecycle`: `unavailable_proof_limited` / `unavailable`
- `real_order_lifecycle`: `unavailable_proof_limited` / `unavailable`

## Output Directory

`/home/molly/project/hftbacktest/local_live_analysis/basis_positive_execution_evidence_fail_closed_runner_0610T004`
