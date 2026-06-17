# Findings

## 0616T007 awsserver1 Preflight Blocker

- `0616T007` correctly stopped the auto loop before live execution.
- SSH to `awsserver1` succeeded, but the remote repo at `/home/admin/hft_live/hftbacktest` is on branch `master`, not `cross-exchange`.
- Controller clarification: that `master` checkout is the Binance maker route, so it should not be modified or repurposed for the Binance-lead / Hyperliquid-lag route.
- The remote repo has dirty changes (`29` status rows), so it is not a clean execution checkout for the approved cross-exchange tiny-live path.
- Remote `conda` is not available and remote `rsync` is not available. The task used `scp` to pull back dry-run evidence, but this does not satisfy the preferred operator packet path without either installing `rsync` or updating the packet to accept `scp`.
- Remote system Python is `/usr/bin/python3` at `Python 3.13.5`; this is acceptable for a future remote preflight if selected and recorded explicitly, while local validation continues to use `.conda-envs/hft-py38`.
- No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred.
- `0616T008` remains blocked despite the earlier limited live approval, because that approval was conditional on `0616T007` QA passing first.

## 0616T008 Live Approval Boundary

- The controller approved a single limited `0616T008` Hyperliquid tiny-live small-notional execution window on `2026-06-17`, conditional on `0616T006` QA and `0616T007` awsserver1 preflight dry-run QA passing first.
- Approved caps: `symbol=BTC`, `max_order_size=0.01 BTC`, `max_order_notional=700 USDC`, `max_position=0.04 BTC`, `max_position_notional=2800 USDC`, `max_notional=3000 USDC`, `max_loss=30 USDC`, `duration=10 minutes`, `host_machine=awsserver1`, `account_scope=Hyperliquid account configured on awsserver1`, `maker_only/post_only=true`, `real_orders_allowed=true`.
- BTC/USD reference at approval time was `65794.035`; the notional caps intentionally round above the `0.01 BTC` and `0.04 BTC` spot equivalents.
- This is not a general live authorization. Any cap mismatch, missing QA prerequisite, private credential disclosure, default-on behavior, deployment/promotion claim, scaling request, or later live window must stop for controller approval.

## 0616T006 Task Boundary

- `0616T006` has been created as the next formal task after `0616T005` QA passed.
- Scope is Hyperliquid tiny-live live-capable preflight / operator packet for future `awsserver1` execution and local artifact validation, not real live execution.
- The task may prepare operator commands, schemas, local validators, dry-run artifacts, artifact pullback/checksum policy, and host preflight requirements.
- The task must leave unapproved live fields as `pending_controller_approval`.
- It must not authorize or perform real order placement, cancellation, amendment, live bot startup, account query, credential disclosure, signing/nonce/user-stream implementation, deployment, promotion, PnL proof, or maker viability proof.
- Business execution produced a local/offline packet generator and validator, official artifacts, documentation, and a `待验收` business report.
- The generated packet names `awsserver1` as the intended host but keeps `host_machine` approval status as `pending_controller_approval`; it also keeps `real_orders_allowed=pending_controller_approval`.

## 0616 Cross-Exchange Auto Loop Stop Point

- The auto loop defined in `docs/cross_exchange_auto_loop_protocol.md` completed through `0616T005` and must now stop.
- `0616T002-T005` improved Hyperliquid readiness only through design, fixture validation, local fake shutdown proof, and protocol design. They do not authorize private endpoint calls, credentials, signing, nonce, user streams, account queries, real order placement, real cancellation, live startup, deployment, promotion, or PnL proof.
- The latest effective QA result is `0616T005` with status `已通过`.
- The next possible live-capable task requires explicit controller approval of all live fields recorded in `local_live_analysis/hyperliquid_tiny_live_protocol_design_0616T005/human_approval_fields.csv`.

## 0616T001 Cross-Exchange Correction Finding

- The active branch is `cross-exchange`; the user clarified the intended direction is Binance lead / Hyperliquid lag maker strategy.
- The previous `0615T009` interpretation as a Binance `BTCUSDT` single-venue small-cap live test is stopped for this branch.
- `0615T008` is not a valid cross-exchange live predecessor because it defines a Binance-specific small-cap protocol, not Hyperliquid maker private/order readiness.
- `0615T001-T007` are not wasted, but they must be treated as reusable source-chain/evidence methodology or migration templates only. They do not authorize Hyperliquid private/order endpoints, credentials/signing/nonce/user stream, order placement, strategy live, parameter search, promotion, PnL proof, or maker viability proof.
- The next correct task should define a Hyperliquid maker private/order execution-readiness boundary using the accepted cross-exchange evidence chain (`0601T004`, `0601T005`, `0609T002`) and must stay no-live/no-order until separate readiness gates pass QA.
- Official `0616T001` artifacts are under `local_live_analysis/cross_exchange_branch_correction_0616T001/`, with final recommendation `cross_exchange_branch_correction_ready_for_qa`.

## Post-0615T005 Forward-Path Finding

- Superseded for `cross-exchange` by `0616T001`; the bullets below describe the prior Binance small-cap path and must not be used as current next-task authority on this branch.
- The user wants the next design to end in a small-cap live test and then use the real-environment data for analysis and decision-making.
- That intent is compatible with the repository only if it is staged behind a runner-consumption gate, a proof-limited read-only runner, and a dedicated live-test risk protocol.
- The first live-capable task must be `0615T009`, not the immediate next task, because the current accepted artifacts still stop at no-trading source implementations.
- The live task must have hard caps, maker-only/post-only behavior, kill-switch criteria, cancel-all/shutdown evidence, and a post-run analysis task before any decision to repeat or scale.
- No current artifact authorizes direct live promotion, default-on behavior, or PnL-based scaling.

## 0615T006 Task Boundary

- `0615T006` has been created, business execution is complete, and QA is `已通过`.
- Scope is source-chain runner-consumption gate / synthesis design, not runner implementation or live execution.
- It defines dependency, timestamp, identity/redaction, runner input, proof-limit, fail-closed, and QA gates for a later `0615T007` proof-limited runner.
- Its final recommendation is `source_chain_runner_consumption_gate_ready_for_qa`; this can only mean the gate design is ready for QA/controller review.
- QA report is `.workflow/reports/0615T006-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T006` as the latest effective QA result.
- It must not be interpreted as authorization for endpoint/source collector implementation, credentials/signing/nonce/user stream, real private/order/account/live/economics data reads, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, real metrics, PnL proof, deployment, promotion, or maker viability proof.
- `0615T009` remains the first live-capable task, and only after `0615T007` / `0615T008` pass QA plus explicit total-control approval.

## 0615T007 Task Boundary

- `0615T007` has been created, business execution is complete, and QA is `已通过`.
- Scope is a proof-limited local read-only runner over already accepted artifacts, not endpoint/source collector work or live execution.
- The runner may emit proof-limited, unavailable, or fail-closed rows only.
- Missing-source and PnL/promotion overclaim requests fail closed.
- Its final recommendation is `proof_limited_read_only_runner_ready_for_qa`; this can only mean runner mechanics are ready for QA/controller review.
- QA report is `.workflow/reports/0615T007-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T007` as the latest effective QA result.
- It must not be interpreted as authorization for real execution/economics metrics, PnL, strategy decisions, live readiness, deployment, promotion, or maker viability proof.

## 0615T008 Task Boundary

- `0615T008` has been created, business execution is complete, and QA is `已通过`.
- Scope is small-cap live-test protocol / risk gate design and dry-run acceptance, not live execution.
- The protocol requires `BTCUSDT`, `10` minute duration cap, `25 USDT` gross notional cap, `5 USDT` single-order cap, `10 USDT` position cap, `2 USDT` max loss, maker-only/post-only, default-on forbidden, cancel-all/shutdown proof, and explicit total-control approval before `0615T009`.
- Its final recommendation is `small_cap_live_test_protocol_ready_for_qa`; this can only mean the protocol is ready for QA/controller review.
- QA report is `.workflow/reports/0615T008-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T008` as the latest effective QA result.
- It must not be interpreted as authorization to open live, use credentials, connect endpoints, place/cancel orders, change strategy defaults, deploy, promote, prove PnL, or prove maker viability.

## 0615T005 Task Boundary

- `0615T005` has been created and dispatched after `0615T004` QA passed.
- `0615T005` business execution is complete and QA is `已通过`.
- Scope is no-trading economics fee/rebate read-only source implementation over task-local fixture inputs only, not endpoint implementation, real venue economics/account/order collection, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, fees/rebates/spread-capture proof, PnL proof, or metric proof.
- It may implement local transform/redaction/artifact writing/arithmetic checks/fail-closed checks and validation handoff into `economics_fee_rebate_source.py`.
- It must preserve the accepted rule that fill notional, order fills alone, public markout alone, account inventory alone, or replay lifecycle alone cannot prove fees/rebates/spread capture or PnL.
- Its `ready` recommendation can only mean the local no-trading economics fee/rebate read-only source implementation is ready for QA/controller review; it cannot authorize real venue use, runner consumption, economics proof, PnL proof, strategy use, live readiness, deployment, promotion, or maker viability proof.
- Final recommendation is `economics_fee_rebate_read_only_source_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_economics_fee_rebate_read_only_source_0615T005/`.
- QA report is `.workflow/reports/0615T005-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T005` as the latest effective QA result.

## 0615T004 Task Boundary

- `0615T004` business execution is complete and QA is `已通过`.
- Scope is no-trading account inventory read-only source implementation over task-local fixture inputs only, not endpoint implementation, real venue account collection, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, inventory lifecycle proof, or metric proof.
- It may implement local transform/redaction/artifact writing/conservation checks/fail-closed checks and validation handoff into `account_inventory_source.py`.
- It must preserve the accepted rule that order fills alone cannot prove inventory lifecycle.
- Its `ready` recommendation can only mean the local no-trading account inventory read-only source implementation is ready for QA/controller review; it cannot authorize real venue use, runner consumption, inventory lifecycle proof, strategy use, live readiness, deployment, promotion, or maker viability proof.
- Final recommendation is `account_inventory_read_only_source_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_account_inventory_read_only_source_0615T004/`.
- QA report is `.workflow/reports/0615T004-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T004` as the latest effective QA result.

## 0615T004 Task Boundary

- `0615T004` has been created and dispatched after `0615T003` QA passed.
- Scope is no-trading account inventory read-only source implementation over task-local fixture inputs only, not endpoint implementation, real venue account collection, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, inventory lifecycle proof, or metric proof.
- It may implement local transform/redaction/artifact writing/conservation checks/fail-closed checks and validation handoff into `account_inventory_source.py`.
- It must preserve the accepted rule that order fills alone cannot prove inventory lifecycle.
- Its `ready` recommendation can only mean the local no-trading account inventory read-only source implementation is ready for QA/controller review; it cannot authorize real venue use, runner consumption, inventory lifecycle proof, strategy use, live readiness, deployment, promotion, or maker viability proof.

## 0615T003 Task Boundary

- `0615T003` has been created and dispatched after `0615T002` QA passed.
- `0615T003` business execution is complete and awaiting QA.
- Scope is no-trading private order response read-only collector implementation over task-local fixture inputs only, not endpoint implementation, real venue collection, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, or metric proof.
- It may implement local transform/redaction/artifact writing/fail-closed checks and validation handoff into `private_order_response_source.py`.
- It must not place, cancel, or amend orders, change strategy behavior, run live/default-on/tiny-live, collect real data, call endpoints, compute fill probability/post-only/real-order-lifecycle metrics, claim execution proof, claim PnL, recommend deployment, or recommend promotion.
- Its `ready` recommendation can only mean the local no-trading read-only collector implementation is ready for QA/controller review; it cannot authorize real venue use, runner consumption, strategy use, live readiness, deployment, promotion, or maker viability proof.
- Final recommendation is `private_order_response_read_only_collector_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_private_order_response_read_only_collector_0615T003/`.

## 0615T002 Task Boundary

- `0615T002` has been created and dispatched after `0615T001` QA passed.
- `0615T002` business execution is complete and awaiting QA.
- Scope is private order response read-only collector boundary / implementation design, not endpoint implementation, collector implementation, signing/nonce/user-stream implementation, real private/order/account/live/economics data read, runner consumption, strategy behavior, live behavior, or metric proof.
- It must preserve the `0615T001` convergence policy and must not consume the one allowed local-only exception; it is non-local in direction because it defines a future real source-line / read-only collector boundary.
- It may design endpoint/permission contracts, field handoff into `private_order_response_source.py`, redaction/local-storage policy, no-trading safety gates, future implementation QA gates, next-task sequence, local artifacts, and a business report.
- It must not place or cancel orders, change strategy behavior, run live/default-on/tiny-live, collect real data, call endpoints, compute fill probability/post-only/real-order-lifecycle metrics, claim execution proof, claim PnL, recommend deployment, or recommend promotion.
- Its `ready` recommendation can only mean the private order response read-only collector boundary design is ready for QA/controller review; it cannot authorize implementation, collection, runner consumption, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `private_order_response_read_only_collector_boundary_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_private_order_response_read_only_collector_boundary_0615T002/`.
- The next executable direction, if QA accepts, is a separate no-trading read-only collector implementation that emits local redacted artifacts validated by `private_order_response_source.py`; runner consumption, metrics, strategy/live, deployment, promotion, and maker viability proof remain forbidden until separately scoped and accepted.

## 0615T001 Convergence Point

- `0615T001` has been created as the next formal task after `0612T001` QA passed.
- `0615T001` business execution is complete and QA is `已通过`.
- The purpose is to stop the recent local artifact skeleton / validator chain from expanding indefinitely and to define the next move toward real source-line implementation or a read-only collector.
- Controller convergence policy: after `0615T001` QA, at most `1` additional local-only task may be dispatched.
- That single extra local-only task is allowed only if `0615T001` names a concrete blocker that must be repaired before any real source-line / read-only collector task can be safely scoped.
- If no such named blocker exists, or after that one extra local-only task completes, the next formal execution-proof task must move to real source-line implementation or read-only collector work.
- `0615T001` itself remains design / boundary work only. It must not implement endpoints, credentials, signing, nonce handling, user streams, source collectors, private/order/account/live/economics data reads, remote execution, collection, runner consumption, real metrics, PnL proof, strategy/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.
- The expected output is an actionable field-authority / permission-boundary / runner-consumption gate that lets total control dispatch the next non-local task without inventing missing source-line boundaries.
- Final recommendation is `real_source_line_readiness_boundary_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_execution_source_real_readiness_collector_boundary_0615T001/`.
- QA report is `.workflow/reports/0615T001-qa.md`, and `docs/qa-acceptance-report.md` now records `0615T001` as the latest effective QA result.

## 0612T001 Task Boundary

- `0612T001` has been created and dispatched after `0611T004` QA passed.
- `0612T001` business execution is complete and QA is `已通过`.
- Scope is local-only economics fee/rebate settlement artifact skeleton / validator, not economics/private/order/account/live endpoint work, source collector work, runner consumption, or metric proof.
- It may use the accepted `0610T009` economics fee/rebate source-line contract and `0611T001` synthesis gate as design inputs; `0611T002` private-order, `0611T003` replay lifecycle, and `0611T004` account inventory local artifacts may be used only as future cross-check context, not current fees/rebates/spread-capture or PnL proof.
- It may implement local economics/fee/rebate/spread-capture schema constants, fixture loader/parser, fail-closed validation, maker/taker classification checks, settlement checks, currency conversion / tick-value arithmetic checks, spread-capture consistency checks, timestamp policy artifacts, CLI/help, tests, design note, local artifacts, and a business report.
- It must fail closed for missing required fields, unknown enums, unsupported source policy, forbidden endpoint/action fields, missing or conflicting settlement authority, unsupported maker/taker classification, non-conserving fee/rebate arithmetic, currency conversion/tick-value mismatch, settlement timestamp merge/order defects, spread-capture overclaims, account-inventory-alone overclaim, order-fill-alone overclaim, public-markout-alone overclaim, PnL overclaim, and live/deployment/promotion overclaim.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, economics/account/private/order/live data, remote execution, collection, runner consumption, real economics metrics, real execution metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Its `ready` recommendation can only mean the local economics fee/rebate artifact skeleton / validator is ready for QA; it cannot authorize metrics, runner consumption, endpoint/source collector work, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `economics_fee_rebate_artifact_skeleton_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_economics_fee_rebate_source_artifact_skeleton_0612T001/`.

## 0611T004 Task Boundary

- `0611T004` business execution is complete and QA is `已通过`.
- Scope is local-only account inventory artifact skeleton / validator, not account/private/order/live endpoint work, source collector work, runner consumption, or metric proof.
- It may use the accepted `0610T008` account inventory source-line contract and `0611T001` synthesis gate as design inputs; `0611T002` private-order and `0611T003` replay lifecycle local artifacts may be used only as future cross-check context, not current inventory lifecycle proof.
- It implements local account/inventory schema constants, fixture loader/parser, fail-closed validation, snapshot / transition / conservation checks, reconciliation boundary artifacts, CLI/help, tests, design note, local artifacts, and a business report.
- It must fail closed for missing required fields, unknown enums, unsupported source policy, forbidden endpoint/action fields, missing/stale/partial/conflicting/unsupported snapshots, unknown/ambiguous/conflicting/unsupported transitions, non-conserving quantities, unit/precision/sign inconsistency, duplicate transitions, out-of-order transitions, order-fills-alone inventory proof overclaim, current inventory lifecycle proof overclaim, PnL/economics overclaim, and live/deployment/promotion overclaim.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, account/private/order/live data, remote execution, collection, runner consumption, real inventory metrics, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Its `ready` recommendation can only mean the local account inventory artifact skeleton / validator is ready for QA; it cannot authorize metrics, runner consumption, endpoint/source collector work, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `account_inventory_artifact_skeleton_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004/`.

## 0611T003 Task Boundary

- `0611T003` business execution is complete and QA is `已通过`.
- Scope is local-only replay lifecycle validation / reconciliation gate, not replay/live semantic implementation, exchange endpoint/source collector work, runner consumption, or metric proof.
- It may use the accepted `0610T007` replay lifecycle source-line contract and `0611T001` synthesis gate as design inputs; `0611T002` private-order local skeleton may be used only as future cross-check context, not current queue or race proof.
- It may implement local lifecycle schema constants, fixture loader/parser, fail-closed validation, ordering/reconciliation policy artifacts, CLI/help, tests, design note, local artifacts, and a business report.
- It must fail closed for missing required fields, unknown enums, unsupported source policy, merged or missing timestamp domains, non-monotonic same-order sequence, invalid terminal lifecycle ordering, ambiguous/conflicting/out-of-order events, duplicate lifecycle event identity, duplicate terminal state for the same opaque order reference, cross-order causal overclaim, replay-as-execution-proof overclaim, queue-priority proof overclaim, and cancel-fill-race metric overclaim.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, replay/live semantic implementation, private/order/account/live data, remote execution, collection, runner consumption, queue priority metrics, exact queue position proof, cancel-fill race metrics, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Its `ready` recommendation can only mean the local replay lifecycle validation / reconciliation gate is ready for QA; it cannot authorize metrics, runner consumption, endpoint/source collector work, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `replay_lifecycle_validation_gate_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_replay_lifecycle_validation_gate_0611T003/`.

## 0611T002 Task Boundary

- `0611T002` business execution is complete and QA is `已通过`.
- Scope is local-only `private_order_response` artifact skeleton / validator, not exchange endpoint/source collector work.
- It may use the accepted `0610T006` private-order source-line contract and `0611T001` synthesis gate as design inputs.
- It may implement local schema constants, fixture loader/parser, fail-closed validation, CLI/help, tests, design note, local artifacts, and a business report.
- It must fail closed for missing required fields, unknown enums, conflicting terminal states, duplicate event identity, incomplete lifecycle evidence, unsupported evidence source, and missing/out-of-order timestamps.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, private/order/account/live data, remote execution, collection, runner consumption, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Its `ready` recommendation can only mean the local artifact skeleton / validator is ready for QA; it cannot authorize metrics, runner consumption, endpoint/source collector work, strategy use, live readiness, deployment, or promotion.
- Final recommendation is `private_order_response_artifact_skeleton_ready_for_qa`; official artifacts are under `local_live_analysis/basis_positive_private_order_response_source_artifact_skeleton_0611T002/`.

## 0611T001 Task Boundary

- `0611T001` has been created as a design-only source-line synthesis / implementation-readiness gate task after `0610T009` QA passed.
- `0611T001` business execution is complete and QA is `已通过`.
- Official artifacts are under `local_live_analysis/basis_positive_execution_source_line_synthesis_gate_0611T001/`.
- Synthesis gate design is `docs/basis_positive_execution_source_line_synthesis_gate.md`.
- Final recommendation is `source_line_synthesis_gate_ready_for_qa`, meaning only that the synthesis/gate design is ready for QA/controller review.
- QA report is `.workflow/reports/0611T001-qa.md`, and `docs/qa-acceptance-report.md` now records `0611T001` as the latest effective QA result.
- It may consume only QA-passed `0610T009` / `0610T008` / `0610T007` / `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` local design artifacts, manifests, and QA/business reports as prior fact sources.
- It must cover exactly four source lines and exactly seven execution gaps from `0610T005`.
- It may define a source-line contract registry, implementation-readiness gate, source dependency reconciliation, forbidden overclaim matrix, and next-task sequence.
- It may recommend future separately scoped implementation tasks, but any such recommendation must explicitly require separate task dispatch and QA before implementation.
- It must not implement or use endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, real execution metrics, real economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0611T001` can only mean the synthesis/gate design is ready for QA/controller review; it cannot authorize implementation or metric proof.

## 0610T009 Task Boundary

- `0610T009` has been created as a design-only `economics_fee_rebate_source_line` contract task after `0610T007` / `0610T008` QA passed.
- `0610T009` business execution is complete and QA is `已通过`.
- Official artifacts are under `local_live_analysis/basis_positive_economics_fee_rebate_source_line_contract_0610T009/`.
- Source-line contract is `docs/basis_positive_economics_fee_rebate_source_line_contract.md`.
- Final recommendation is `economics_fee_rebate_contract_ready_for_qa`, meaning only that the design contract is ready for QA/controller review.
- It may consume only QA-passed `0610T008` / `0610T007` / `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` local design artifacts, manifests, and QA/business reports as prior fact sources.
- It may cover only the primary gap assigned to `economics_fee_rebate_source_line`: `fees_rebates_spread_capture`.
- Required contract coverage includes economics artifact schema, fee/rebate settlement taxonomy, spread-capture taxonomy, maker/taker classification policy, currency conversion / tick-value policy, settlement timestamp policy, reconciliation boundary, validation gates, overclaim reject rules, manifest, boundary validation, and business report.
- `0610T006` private-order response artifacts may be used only as future fill dependency context, `0610T007` replay lifecycle artifacts only as future timestamp/order consistency context, and `0610T008` account inventory artifacts only as future reconciliation context.
- The task must explicitly reject hypothetical spread, fill notional, order fills alone, public markout alone, account inventory alone, or replay lifecycle alone as proof of fees/rebates/spread capture or PnL.
- It must not implement or use economics/private/order/account/live endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, real economics metrics, real execution metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0610T009` can only mean the design contract is ready for QA/controller review; it cannot authorize endpoint implementation, source collection, runner implementation, economics proof, PnL proof, or metric proof.

## 0610T007 / 0610T008 Parallel Execution Finding

- `0610T007` and `0610T008` were executed in parallel after total control confirmed the parallel condition.
- Both tasks remained design-only contracts and QA is `已通过`.
- `0610T007` final recommendation is `replay_lifecycle_contract_ready_for_qa`; this means only that the replay lifecycle semantics source-line design is ready for QA/controller review.
- `0610T008` final recommendation is `account_inventory_contract_ready_for_qa`; this means only that the account inventory source-line design is ready for QA/controller review.
- Parallel write-scope check passed at the commit level: `139b76a` / `692f445` touched only T007 task/report/doc/artifact paths, and `66127b7` / `156f6da` touched only T008 task/report/doc/artifact paths.
- Shared tracking files were not modified by the business-thread commits; total control is responsible for this tracking update.
- QA reports were written to `.workflow/reports/0610T007-qa.md` and `.workflow/reports/0610T008-qa.md`; `docs/qa-acceptance-report.md` now contains the latest effective QA result for `0610T008`.
- Neither task authorizes source reader/collector implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T008 Task Boundary

- `0610T008` has been created as a prepared design-only `account_inventory_source_line` contract task.
- It is parallel-eligible with `0610T007` only because both tasks are design-only contracts, their primary gaps and output paths are disjoint, and neither business thread may modify shared tracking files (`task_plan.md`, `progress.md`, `findings.md`, `docs/qa-acceptance-report.md`).
- It may consume only QA-passed `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` local design artifacts, manifests, and QA/business reports as prior fact sources.
- It may cover only the primary gap assigned to `account_inventory_source_line`: `inventory_lifecycle`.
- Required contract coverage includes account/inventory artifact schema, inventory snapshot taxonomy, inventory transition taxonomy, conservation checks, reconciliation boundary, timestamp policy, fail-closed validation gates, overclaim rejection rules, manifest, boundary validation, and business report.
- `0610T006` private order response artifacts may be used only as future transition input / future cross-check context, not current inventory lifecycle proof.
- The task must explicitly state that order fills alone cannot prove inventory lifecycle.
- It must not implement or use account/private/order/live endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, real inventory metrics, real execution metrics, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0610T008` can only mean the design contract is ready for QA/controller review; it cannot authorize endpoint implementation, source collection, runner implementation, inventory lifecycle proof, or metric proof.

## 0610T007 Task Boundary

- `0610T007` has been created and dispatched as a design-only `replay_lifecycle_semantics_source_line` contract task after `0610T006` QA.
- It is parallel-eligible with `0610T008` only because both tasks are design-only contracts, their primary gaps and output paths are disjoint, and neither business thread may modify shared tracking files (`task_plan.md`, `progress.md`, `findings.md`, `docs/qa-acceptance-report.md`).
- It may consume only QA-passed `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` local design artifacts, manifests, and QA/business reports as prior fact sources.
- It may cover only the two primary gaps assigned to `replay_lifecycle_semantics_source_line`: `queue_priority` and `cancel_fill_race`.
- `0610T006` private order response artifacts may be used only as future cross-check / future event-source dependency context, not current proof source for queue priority or cancel-fill race.
- It must define replay/live lifecycle event schema, queue semantics boundary, cancel/fill race event-ordering policy, timestamp policy, replay/live proof-limit rules, validation gates, overclaim rejection rules, manifest, boundary validation, and business report.
- It must preserve replay as supporting regression, not execution proof.
- It must not implement replay/live semantics, private/order/account/live endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, exact queue position proof, cancel-fill race metric proof, real execution metrics, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0610T007` can only mean the design contract is ready for QA/controller review; it cannot authorize replay/live semantic implementation or metric proof.

## 0610T006 Task Boundary

- `0610T006` has been created and dispatched as a design-only `private_order_response_source_line` contract task after `0610T005` QA.
- `0610T006` business execution is complete and QA is `已通过`.
- Official artifacts are under `local_live_analysis/basis_positive_private_order_response_source_line_contract_0610T006/`.
- Final recommendation is `private_order_response_contract_ready_for_qa`, meaning only that the design contract is ready for QA/controller review.
- It may consume only QA-passed `0610T005` source decomposition artifacts plus necessary `0610T004` / `0610T003` / `0610T002` QA/business reports and manifests as prior fact sources.
- It may cover only the three primary gaps assigned to `private_order_response_source_line`: `fill_probability`, `post_only_reject_behavior`, and `real_order_lifecycle`.
- It must define response artifact schema, response label taxonomy, post-only reject taxonomy, lifecycle state taxonomy, timestamp policy, terminal-state consistency, validation gates, overclaim rejection rules, manifest, boundary validation, and business report.
- It must not implement or use private/order/account/live endpoints, source readers, source collectors, user streams, signing, nonce handling, runners, real execution metrics, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Any `ready` recommendation in `0610T006` can only mean the design contract is ready for QA/controller review; it cannot authorize endpoint implementation or metric proof.

## 0610T005 Task Boundary

- `0610T005` has been created as a prepared design-only source decomposition / source-line routing contract task and is unblocked by `0610T004` QA.
- `0610T005` business execution is complete and QA is `已通过`.
- Official artifacts are under `local_live_analysis/basis_positive_execution_source_design_decomposition_0610T005/`.
- Final recommendation is `private_order_source_design_ready_next`, meaning only that a later separately dispatched design-only task may define the `private_order_response_source_line` contract.
- It may consume only QA-passed `0610T004`/`0610T003`/`0610T002` contract, gate, skeleton artifacts plus necessary QA/business reports.
- It must split the seven execution gaps into source-design lines using truth authority, label unit, causal time semantics, permission boundary, validation oracle, and overclaim failure mode.
- It must keep `private_order_response_source_line`, `replay_lifecycle_semantics_source_line`, `account_inventory_source_line`, and `economics_fee_rebate_source_line` distinct unless all six split gates match.
- It must not implement source readers, source collectors, runners, private/order/account/live endpoints, user streams, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T004 Task Boundary

- `0610T004` has been created as a prepared fail-closed/read-only runner skeleton implementation task and is unblocked by `0610T003` QA.
- `0610T004` QA is `已通过`.
- `0610T004` business execution implemented the local fail-closed/read-only runner skeleton.
- Official artifacts are under `local_live_analysis/basis_positive_execution_evidence_fail_closed_runner_0610T004/`.
- Final recommendation is `fail_closed_runner_skeleton_ready_for_qa`, meaning only that the skeleton is ready for QA review.
- It may consume only QA-passed `0610T003`/`0610T002` contract and gate artifacts plus necessary QA/business reports.
- It may implement prerequisite/source-policy/schema/overclaim validation and proof-limited unavailable status rows for the seven execution gaps.
- It must not compute real execution metrics or claim fill probability, queue/priority, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, maker execution viability, live readiness, default-on readiness, tiny-live readiness, deployment readiness, or promotion proof.
- It must not read private/order/account/live data or authorize strategy/private/order/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T003 Task Boundary

- `0610T003` completed business execution as a design/gate-only source availability and runner implementation gate task after `0610T002` QA and passed QA.
- It may consume QA-passed `0610T002` artifacts plus necessary `0610T001` / `0609T011` / `0609T010` QA/business reports and manifests only as prior fact sources.
- It classified each of the seven execution gaps as fail-closed placeholder only under current sources and identified blocker/source-design paths before any execution-proof metric can be considered.
- Final recommendation is `runner_skeleton_ready_with_fail_closed_sources`, meaning only that a later separately scoped task may implement a fail-closed/read-only skeleton that validates prerequisites, source policies, output restrictions, gap coverage, and overclaim rejection.
- It must preserve private/order response artifacts as `forbidden_current_task / future_requires_separate_design`.
- It must preserve replay/simulation artifacts as `supporting_regression_not_execution_proof`.
- It must preserve public proxy artifacts as design context only, not execution proof.
- This task must not implement runner behavior or authorize runner implementation, case-library implementation, source-row case catalog generation, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T002 Task Boundary

- `0610T002` completed business execution as a design-only read-only execution-evidence runner contract task and passed QA.
- It consumed QA-passed `0610T001` artifacts plus necessary T010/T011/T001 QA/business reports only.
- Official artifacts are under `local_live_analysis/basis_positive_execution_evidence_runner_contract_0610T002/`.
- Runner contract is `docs/basis_positive_execution_evidence_runner_contract.md`.
- Final recommendation is `read_only_execution_evidence_runner_design_ready`, meaning only that the current runner contract/design artifacts are ready for QA/controller review.
- The recommendation does not indicate implementation readiness.
- Private/order response artifacts remain `forbidden_current_task / future_requires_separate_design`.
- Replay/simulation artifacts remain `supporting_regression_not_execution_proof`.
- Public proxy artifacts remain design context only, not execution proof.
- This task does not authorize runner implementation, case-library implementation, source-row case catalog generation, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T001 Task Boundary

- `0610T001` completed business execution as a design-only execution-evidence requirements contract task and passed QA.
- It consumed QA-passed `0609T011` artifacts plus necessary T010/T011 QA/business reports only.
- Official artifacts are under `local_live_analysis/basis_positive_execution_evidence_requirements_0610T001/`.
- Design contract is `docs/basis_positive_execution_evidence_requirements.md`.
- Final recommendation is `execution_evidence_runner_contract_ready`, meaning only that a later separately scoped read-only runner contract/design task can be considered after QA.
- Private/order response artifacts are classified as `forbidden_current_task / future_requires_separate_design`.
- Replay/simulation artifacts are classified as `supporting_regression_not_execution_proof`.
- This task does not authorize runner implementation, case-library implementation, source-row case catalog generation, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0609T011 Task Boundary

- `0609T011` completed business execution as a read-only proxy evidence synthesis task and is now `待验收`.
- It consumed only QA-passed T010 local proxy artifacts and T010 QA/business reports.
- Official artifacts are under `local_live_analysis/basis_positive_proxy_evidence_synthesis_0609T011/`.
- Final recommendation is `continue_to_execution_evidence_design`, meaning only that a later separately scoped design task can define execution-layer evidence requirements.
- Metric/sample/proof-class matrices summarize proxy evidence at aggregate level only; no source-row case catalog, case-library behavior, shadow decision, executable trigger, trading instruction, order side, quote price, or quote size is emitted.
- Execution evidence gaps remain unproven: fill probability, queue/priority, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, live readiness, default-on readiness, tiny-live readiness, deployment readiness, promotion, and maker execution viability.
- This task must not be interpreted as strategy/private/order/live/default-on/tiny-live authorization, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.

## 0609T010 Task Boundary

- `0609T010` completed business execution as the read-only maker-viability proxy runner implementation task after `0609T009` QA and has passed QA.
- It implemented only local proxy metrics over T008/T009 allowlisted public/canonical observation-layer artifacts.
- Official artifacts are under `local_live_analysis/basis_positive_maker_viability_proxy_0609T010/`.
- Final recommendation is `read_only_proxy_evidence_ready_for_qa`.
- It keeps all T008 execution-layer gaps unproven and reports proxy results with caveats from the T009 contract.
- It must not implement case-library behavior, source-row case catalogs, shadow decisions, executable triggers, order side, quote price/size, private/account/order endpoints, order lifecycle logic, strategy/live/default-on/tiny-live behavior, parameter search, deployment recommendation, promotion, or maker execution viability proof.

## 0609T009 Task Boundary

- `0609T009` completed business execution as a design-only execution-evidence gap planning task after `0609T008` QA and is now `待验收`.
- It may read `0609T008` row-level read-only artifacts and inherited T006/T007 boundary references only as local public/canonical observation-layer inputs.
- It must classify execution-layer questions into proxy-available, proxy-with-caveat, not-provable-without-separate-evidence, or forbidden-for-current-stage categories.
- It may design a later read-only maker-viability proxy runner contract, including input/output schema, validation requirements, and overclaim reject conditions.
- Final recommendation is `read_only_proxy_runner_ready_for_implementation`, meaning only that a later separately dispatched read-only proxy runner implementation task can be considered after QA.
- It must not implement the proxy runner, generate case-library entries, create source-row case catalogs, produce shadow decisions, output executable triggers or trading instructions, set order side/quote price/size, use private/account/order endpoints, touch order lifecycle logic, run live/default-on/tiny-live, run parameter search, recommend deployment, claim promotion, or claim maker execution viability is proven.

## 0609T001 Business Findings

- `0609T001` completed business execution as a read-only basis-positive wrong-way decomposition and targeted sample design task.
- Runner: `examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py`.
- Focused tests: `examples/hyperliquid/test_canonical_basis_positive_wrong_way_decomposition.py`.
- Official artifacts: `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/`.
- T006 prerequisite was validated from `basis_positive_robustness_manifest.json`: `final_recommendation=needs_more_samples`, `scope_policy=not_limited_to_regime_011`, and `t005_final_contract_decision=upgrade_to_context_only_supported`.
- Baseline comparison at `1000ms`: `basis > 0` has `2425` rows, `3` samples, hit rate `0.94600939`, mean future move `45.67216495` ticks, wrong-way count `69`, p95 wrong-way loss `138` ticks, and max wrong-way loss `180` ticks; `basis <= 0` has `7564` rows, hit rate `0.33853760`, and mean future move `-16.94407721` ticks.
- Positive-basis magnitude strengthens monotonically by mean future move from small `18.47337278` to medium `29.41516710` to large `90.09975062` ticks, while wrong-way tail is heavier in the small positive-basis bucket.
- Controlled checks classify basis-positive as retaining nontrivial effect within both Binance momentum buckets and Hyperliquid book-state buckets.
- Candidate visible tail hypotheses are `basis_positive_small`, negative Hyperliquid top5 imbalance, and negative Hyperliquid microprice-minus-mid; current sample/time concentration still requires targeted validation rather than direct promotion.
- Final recommendation is `targeted_collection_ready`, meaning only that a future separately dispatched and QA-accepted collection design is now supportable. It does not authorize new collection inside T001, strategy implementation, private/order endpoints, order lifecycle, case-library, shadow decisions, live/default-on/tiny-live, parameter search, or promotion.

## 0608T003-0608T005 Regime 011 / Basis Context QA Findings

- `0608T003` passed QA as the read-only directional momentum viability assessment for `regime_011_1000_spread_10_20_ticks`.
- `0608T003` final recommendation is `reject_directional_edge_unstable`: base regime row count `242`, sample count `3`, direction hit rate `0.51239669`, per-sample signed edge `10.51282051 / -9.04040404 / 2.5` ticks, conservative net edge `-8.07024793` ticks, and tail-risk proxy rejected.
- `0608T004` passed QA as the read-only feature-conditioned validity diagnosis for the same regime.
- `0608T004` final recommendation is `watch_needs_contract_visibility_clarification`, with `valid_supported_pattern_count=0`, `valid_watch_pattern_count=0`, and `invalid_pattern_count=21`.
- In `0608T004`, the strongest non-tail/non-redundancy-looking local pattern was `context_basis_mid_ticks > 0`: `57` rows, `3` samples, hit rate `0.94736842`, net edge proxy `88.54385965` ticks, and `tail_risk_acceptable_proxy`; it was still classified `invalid_not_decision_visible` only because the prior data contract treated basis as diagnostic/caveated context.
- `0608T005` passed QA as the read-only basis-context visibility / lineage diagnosis for `context_basis_mid_ticks > 0`.
- `0608T005` final contract decision is `upgrade_to_context_only_supported`: basis lineage is confirmed as `(binance_mid_px - hyperliquid_mid_px) / 0.1`, `57/57` rows pass formula reconstruction with max error `0.0`, `57/57` rows are timestamp/as-of clean, future input joins and missing input joins are both `0`, all three canonical samples contribute, and max sample row share is `0.38596491`.
- `0608T005` persistence check did not reverse direction at `1000/5000/10000ms`; `100/250ms` remains watch/alias context only.
- Controller interpretation: Regime 011 should not progress to maker case-library, directional case-library, shadow decisions, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion. `context_basis_mid_ticks > 0` may be used only as decision-time context in later read-only research, retaining execution-PnL caveat.
- Contract amendment after `0608T005`: `basis_mid_dislocation` / `context_basis_mid_ticks` is now `allow / context_only_supported` in the `0601T004` data contract; `basis_microprice_dislocation` remains `diagnostic_only / diagnostic_context`.
- `0608T006` business execution is complete and awaiting QA as a read-only basis-positive robustness diagnosis outside the Regime 011 shell.
- `0608T006` final recommendation is `needs_more_samples`: `context_basis_mid_ticks > 0` has `2425` primary rows, `3` samples, hit rate `0.94600939`, mean future move `45.67216495` ticks, and positive persistence at `1000/5000/10000ms`, but max sample row share is `0.67917526`, join-age and volatility coverage are narrow, and the cost/tail proxy classifies `cost_tail_reject` with p95 wrong-way loss `138` ticks.
- `0608T006` collinearity check did not classify basis-positive as solely explained by Hyperliquid top5 imbalance or microprice-minus-mid; both rows are `not_explained_solely_by_hl_book_state`.
- `0608T006` does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decisions, executable trading instructions, or promotion.

## 0604T015 Task Boundary

- `0604T015` has been created to diagnose the post-`0604T013` live shutdown gap where shutdown now calls `wait_order_response()` but still cannot prove cancel acknowledgement.
- The task must reproduce and attribute all reported issues: `0` as `Ok`/timeout and batch-folded received response, `3` as order response but not canceled-state proof, raw `wait_result` logging without classification, misleading `ack_waits` counter semantics, fake tests returning `0` as success, and missing final local/REST/audit tail proof.
- Scope is diagnosis, stable reproduction, range check, root cause, and test coverage gap reporting only.
- It must not repair `live_tick_mm.py`, py bindings, Rust live bot/backtest, connector, production config, audit schema, normal loop cancel semantics, or start live/default-on/tiny-live/promotion.

## 0604T016 Task Boundary

- `0604T016` has been created as the follow-up repair task for live shutdown cancel acknowledgement proof semantics.
- The repair must keep `order_response_received` and `terminal_confirmed` as independent dimensions.
- Hard acceptance line: no code path may set `terminal_confirmed=True` solely from `wait_result == 3` or `order_response_received=True`; terminal confirmation must require independent final order-state proof.
- The task now fixes allowed enum values: `wait_outcome` may only be `order_response_received`, `ok_unknown_or_timeout`, `wait_error`, or `not_requested`; `terminal_confirmation_source` may only be `local_orders`, `rest_open_orders`, or `none`, with `rest_open_orders` unavailable unless a safe local proof path exists.
- Scope is limited to `live_tick_mm.py` shutdown helper result semantics, shutdown summary logging, focused tests, and the task business report.
- It must not modify py bindings, Rust live bot/backtest, connector, production config, audit schema, normal loop cancel semantics, or start live/default-on/tiny-live/promotion.

## Binance Maker MM Test Environment Finding

- For `examples/binance_tick_mm`, prefer `/home/molly/anaconda3/envs/hftbacktest/bin/python` for pytest verification.
- Verified command on 2026-06-04: `/home/molly/anaconda3/envs/hftbacktest/bin/python -m pytest examples/binance_tick_mm` -> `282 passed in 2.92s`.
- The generic/base `python -m pytest examples/binance_tick_mm` can import `/home/molly/anaconda3/lib/python3.13/site-packages/hftbacktest/data/utils/tardis.py` during `run_env_test.py` collection and fail before test execution with numba cache locator error: `RuntimeError: cannot cache function '_convert_depth': no locator available`.
- Treat that base-env failure as an environment/collection issue, not as a Binance maker MM regression, when the same suite passes in the project `hftbacktest` conda env.

## 0604T006 Business Findings

- `0604T006` completed business execution; its initial QA found only a report bucket-consistency defect, and `0604T008` has repaired that defect with QA `已通过`.
- New runner: `examples/hyperliquid/canonical_signal_quality_ranking.py`.
- Focused tests: `examples/hyperliquid/test_canonical_signal_quality_ranking.py`.
- Task artifacts: `local_live_analysis/canonical_signal_quality_ranking_0604T006/`.
- The runner consumes `0604T003` canonical event-mode artifacts through the `0604T004` loader path and refuses non-canonical / diagnostic-only synthetic inputs.
- Ranking is limited to the four `0601T004` primary allowlist features and writes `signal_quality_ranking.csv`, `signal_quality_reject_watch_list.csv`, `signal_quality_ranking_manifest.json`, and `signal_quality_ranking_report.md`.
- Ranking result: `binance_mid_move_ticks_from_prev` rank 1 / `keep_for_read_only_research`; `binance_top5_imbalance` rank 2 / `watch_regime_dependent`; `binance_top5_bid_qty` rank 3 / `watch_regime_dependent`; `binance_microprice_minus_mid_ticks` rank 4 / `watch_regime_dependent`; rejects `0`.
- The result matches the current controller interpretation at the ordering level: mid move is the most stable global candidate, top5 imbalance remains the strongest book-pressure candidate but is watch-labeled by the strict concentration/short-horizon caveats, bid qty remains liquidity/context watch, and microprice-minus-mid remains regime-dependent watch.
- Downstream work should use the T008-refreshed T006 artifacts under `local_live_analysis/canonical_signal_quality_ranking_0604T006/`, where the report now mechanically matches each feature's `final_bucket`.
- This is read-only signal quality ranking only. It does not authorize regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, or promotion.

## 0604T007 Business Findings

- `0604T007` completed business execution and passed QA.
- New runner: `examples/hyperliquid/canonical_horizon_regime_diagnostics.py`.
- Focused tests: `examples/hyperliquid/test_canonical_horizon_regime_diagnostics.py`.
- Task artifacts are under `local_live_analysis/canonical_horizon_regime_diagnostics_0604T007/`: `horizon_independence_diagnostics.csv`, `regime_conditioning_diagnostics.csv`, `regime_watch_list.csv`, `horizon_regime_diagnostics_manifest.json`, and `horizon_regime_diagnostics_report.md`.
- The runner consumes the `0604T003` canonical event-mode aggregate through the `0604T004` loader path and refuses diagnostic-only synthetic inputs.
- Horizon findings: `100/250ms` are `watch_needs_more_samples`; `500/1000/5000/10000ms` are `diagnostic_supported`, with `1000ms+` explicitly preferred for interpretation.
- Regime findings remain watch-only diagnostics: support counts are `10 diagnostic_supported`, `6 watch_needs_more_samples`, and `2 reject_aliased_or_concentrated`; no bucket is promoted into a final regime or maker action.
- This task does not authorize new collection, final regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T008 QA Findings

- `0604T008` passed QA as the narrow T006 signal-ranking report bucket-consistency repair.
- It fixed `examples/hyperliquid/canonical_signal_quality_ranking.py` so the `Controller Interpretation Check` is generated from actual `ranking_rows` / `final_bucket` values instead of static text.
- Focused tests now include a regression check that every feature's report interpretation line matches its generated bucket and that `binance_top5_imbalance` cannot be written as kept when it is `watch_regime_dependent`.
- Refreshed T006 artifacts keep the same ranking semantics: `binance_mid_move_ticks_from_prev=keep_for_read_only_research`; `binance_top5_imbalance`, `binance_top5_bid_qty`, and `binance_microprice_minus_mid_ticks=watch_regime_dependent`.
- QA verified temporary and official report/CSV consistency with `checked_features 4` and `missing []`.
- This repair does not change ranking scoring, allowlist, source-lock guard, canonical loader, strategy, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.

## 0604T009 Business Findings

- `0604T009` completed business execution and is awaiting QA.
- New runner: `examples/hyperliquid/canonical_regime_synthesis.py`.
- Focused tests: `examples/hyperliquid/test_canonical_regime_synthesis.py`.
- Task artifacts are under `local_live_analysis/canonical_regime_synthesis_0604T009/`: `candidate_regime_definitions.csv`, `candidate_regime_evidence_summary.csv`, `candidate_regime_watch_reject_list.csv`, `canonical_regime_synthesis_manifest.json`, and `canonical_regime_synthesis_report.md`.
- The runner consumes `0604T003` canonical event-mode evidence through the accepted `0604T004/0604T005` loader/source-lock guard path and refuses diagnostic-only synthetic inputs.
- Classification counts: `candidate_for_milestone3_executability=1`, `watch_needs_more_samples=6`, `reject_unstable_direction=9`, `reject_concentrated_or_aliased=2`.
- The only read-only candidate for later Milestone 3 executability assessment is `regime_011_1000_spread_10_20_ticks`: primary anchor `binance_mid_move_ticks_from_prev`, horizon `1000ms`, context `primary_usable / fresh_0_50ms / spread_10_20_ticks`, row count `242`, sample count `3`, effective future-row-delta support `3`.
- `binance_top5_imbalance`, `binance_top5_bid_qty`, and `binance_microprice_minus_mid_ticks` remain secondary context only and are not allowed as promoted primary anchors.
- This task does not authorize maker side, quote behavior, order behavior, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, or promotion.

## 0604T005 QA Findings

- `0604T005` passed QA as read-only canonical evidence source lock / guard hardening.
- The reusable guard API was added to `examples/hyperliquid/canonical_event_mode_evidence.py`: `guard_canonical_event_mode_evidence`, `validate_canonical_source_lock_manifest`, and `build_canonical_source_lock_artifacts`.
- Task artifacts are under `local_live_analysis/canonical_evidence_source_lock_0604T005/`: `canonical_source_lock_manifest.json`, `canonical_guard_check_report.md`, and `negative_guard_validation_report.csv`.
- The canonical guard accepts `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/` as formal event-mode evidence with `canonical_sample_count=3` and names the `0604T004` loader/foundation as the required foundation artifact source.
- The guard rejects `synthetic_diagnostic_comparison` as formal evidence unless explicit negative validation is requested; negative validation reports `canonical_sample_count=0` and `diagnostic_rejection_count=3`.
- Focused pytest now covers accepted canonical source, rejected diagnostic source, missing source-lock metadata, zero-canonical formal-evidence failure, and downstream-worker-style guard consumption.
- QA verification passed: `--help`, `py_compile`, focused pytest (`10 passed`), true canonical source-lock rerun, synthetic diagnostic negative validation, manifest JSON parse, and `git diff --check`.
- This remains read-only canonical evidence source-lock / guard hardening only. It does not authorize signal ranking, regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T005-0604T007 Dispatch Boundary

- `0604T005`, `0604T006`, and `0604T007` have been dispatched as parallel read-only workers after `0604T004` QA.
- `0604T005` owns Milestone 0 canonical evidence source lock / guard hardening.
- `0604T006` owns Milestone 1 canonical signal quality ranking over the four accepted Binance lead allowlist features.
- `0604T007` owns Milestone 1 canonical horizon / regime diagnostics.
- All three workers must consume `0604T003` canonical event-mode evidence through the `0604T004` loader/foundation and must exclude ordinary synthetic fixed-grid diagnostics from formal evidence.
- The dispatch does not authorize new collection, final high-confidence regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T004 QA Findings

- `0604T004` passed QA as a read-only canonical event-mode evidence loader / validator foundation.
- New module: `examples/hyperliquid/canonical_event_mode_evidence.py`.
- Focused tests: `examples/hyperliquid/test_canonical_event_mode_evidence.py`.
- Task artifacts: `local_live_analysis/canonical_event_mode_evidence_0604T004/`.
- The loader validates required `0604T003` aggregate files and columns, requires canonical samples to have `decision_mode=event` plus `canonical_status=canonical_event_mode`, and excludes `diagnostic_only_synthetic_decision_grid` samples from canonical outputs.
- Accepted canonical aggregate result: `canonical_sample_count=3`, `diagnostic_rejection_count=0`.
- Synthetic diagnostic validation result: `canonical_sample_count=0`, `diagnostic_rejection_count=3`.
- Diagnostic-only synthetic comparison has an empty venue-state CSV because there are no canonical venue-conditioning rows; the loader accepts that only for all-diagnostic/no-canonical inputs while keeping canonical inputs strict.
- Required outputs exist: `canonical_sample_manifest.json`, `canonical_sample_quality_summary.csv`, `diagnostic_rejection_report.csv`, and `canonical_evidence_validation_report.md`; a parallel negative-validation output exists under `synthetic_diagnostic_validation/`.
- QA verification passed: `--help`, `py_compile`, focused pytest (`5 passed`), true canonical input rerun, synthetic diagnostic input rerun, and `git diff --check`.
- This is only a read-only loader/validator foundation for later parallel analysis. It does not authorize signal ranking, regime selection, case-library construction, shadow decision generation, strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T004 Task Boundary

- `0604T004` has been created as the narrow serial foundation before any parallel Milestone 0 / Milestone 1 development.
- It must implement only a read-only canonical event-mode evidence loader / validator over accepted `0604T003` artifacts.
- The loader must admit `decision_mode=event` / `canonical_status=canonical_event_mode` samples and exclude `diagnostic_only_synthetic_decision_grid` samples from the canonical evidence set.
- Required outputs are `canonical_sample_manifest.json`, `canonical_sample_quality_summary.csv`, `diagnostic_rejection_report.csv`, and `canonical_evidence_validation_report.md`.
- This task intentionally does not do signal ranking, regime selection, case-library construction, shadow decision generation, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, schema/API changes, or promotion.

## 0604T003 QA Findings

- `0604T003` passed QA and is the formal Binance-led Hyperliquid pricing-signal robustness evidence source.
- Canonical event-mode aggregate under `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/` produced `sample_count=3`, `canonical_sample_count=3`, `diagnostic_synthetic_sample_count=0`, and recommendation `continue_read_only_runner_refinement`.
- Synthetic diagnostic comparison under `local_live_analysis/event_mode_canonical_pricing_signal_0604T003/synthetic_diagnostic_comparison/` produced `sample_count=3`, `canonical_sample_count=0`, `diagnostic_synthetic_sample_count=3`, and recommendation `needs_more_public_samples`.
- Future robustness decisions must use event-driven Hyperliquid decision rows plus de-aliased future-row-delta diagnostics as canonical evidence.
- Ordinary synthetic fixed-grid artifacts remain parseable only for backward-compatible diagnostics and must not be interpreted as independent short-horizon stability evidence.
- This does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, connector/core API changes, standard npz schema changes, canonical Binance maker audit schema changes, or promotion.

## 0604T003 Task Boundary

- `0604T003` has been created to repair the remaining ordinary synthetic pricing-signal / robustness artifact risk after `0604T001` and `0604T002`.
- The task should make event-driven Hyperliquid decision rows plus de-aliased future-row-delta diagnostics the canonical decision path for Binance-led Hyperliquid pricing-signal and multi-sample robustness research.
- Synthetic fixed-grid artifacts may remain parseable for backward-compatible diagnostics, but must be marked or treated as diagnostic-only when `100/250/500ms` nominal horizons alias to the same future row.
- The task must ensure recommendation logic uses independent effective horizon / future-row-delta evidence rather than nominal horizon count alone.
- Required task-scoped evidence should use existing local event-mode artifacts under `local_live_analysis/event_horizon_comparison_0604T002/**`; no new collection is authorized.
- The task does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, connector/core API changes, standard npz schema changes, canonical Binance maker audit schema changes, or promotion.
- Business execution and QA are complete. Canonical event-mode a/b/c aggregate produced `canonical_sample_count=3` and `continue_read_only_runner_refinement`; ordinary synthetic a/b/c diagnostic comparison produced `canonical_sample_count=0`, `diagnostic_synthetic_sample_count=3`, and `needs_more_public_samples`.

## 0601T006 QA Findings

- `0601T006` passed QA as public-only collection / initial synthetic-grid multi-sample aggregate evidence.
- Accepted collection/process evidence includes `xemm_0603_quiet_b` and `xemm_0603_quiet_c` collected on `awsserver1`, copied back locally, and processed through local alignment, as-of join, lead-lag analysis, pricing-signal runner, and aggregate robustness runner.
- The task produced the required initial aggregate artifacts under `local_live_analysis/binance_led_hyperliquid_multisample_robustness_0601T006/` with `sample_count=4` and recommendation `continue_read_only_runner_refinement`.
- Because `0604T001-0604T003` later proved and repaired fixed-grid horizon aliasing, the ordinary synthetic-grid `0601T006` aggregate is accepted only as precursor collection / diagnostic evidence.
- The formal robustness interpretation is superseded by `0604T003` canonical event-mode artifacts; synthetic fixed-grid outputs must not be used as canonical short-horizon independent evidence.
- This does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## 0601T006 Task Boundary

- `0601T006` has been created as the Binance-led Hyperliquid public multi-sample robustness validation task.
- It is unblocked by `0601T005` QA, which accepted the four-feature read-only pricing-signal runner with a `single_public_sample_caveat=true`.
- The task must validate whether the four primary allowlist features are stable across multiple synchronized public samples: `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty`.
- Scope is public-only Binance lead / Hyperliquid lag synchronized samples. Target coverage is 2-3 new samples across active/high-vol, quiet/low-vol, and normal-liquidity regimes when feasible.
- Each sample should reuse the accepted chain: synchronized public collection, `0601T002` style as-of join, `0601T003` style lead-lag analysis, and `0601T005` pricing-signal runner.
- Required aggregate outputs are `multi_sample_manifest.json`, `sample_quality_matrix.csv`, `feature_horizon_stability_across_samples.csv`, `effective_horizon_aliasing_by_sample.csv`, `venue_state_conditioning_across_samples.csv`, and `pricing_signal_robustness_recommendation.md`.
- Final recommendation must stay within the task taxonomy: `continue_read_only_runner_refinement`, `needs_more_public_samples`, `narrow_to_specific_venue_state_regime`, or `reject_for_runner_design`.
- The task does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## 0601T005 QA Findings

- `0601T005` passed QA as the Binance-led Hyperliquid read-only pricing-signal runner implementation.
- QA reran help, py_compile, focused pytest, `/tmp` reproduction, official manifest JSON parse, row-count/allowlist/recommendation checks, and `git diff --check`; all passed.
- Official and `/tmp` reproduction row counts matched: input `3599`, primary `3596`, excluded `3`, pricing signal rows `21541`, feature quality rows `4`, horizon label rows `30`, feature/regime rows `540`, and venue-state conditioning rows `54`.
- Primary allowlist enforcement passed: only `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty` appear as primary features.
- Future labels are separated from decision-time input fields; QA found no `input_*future*` fields.
- Recommendation is `keep_for_read_only_research`, with `single_public_sample_caveat=true`.
- Boundary grep found only prohibitions/scope text/manifest flags, not execution paths for private/order/live/strategy/parameter/default-on/tiny-live/promotion.
- This result does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## 0601T005 Business Findings

- `0601T005` completed business execution and then passed QA as the Binance-led Hyperliquid read-only pricing-signal runner implementation.
- Runner: `examples/hyperliquid/binance_led_pricing_signal_runner.py`.
- Focused tests: `examples/hyperliquid/test_binance_led_pricing_signal_runner.py`.
- Output directory: `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/`.
- Required artifacts exist: `run_manifest.json`, `pricing_signal_rows.csv`, `pricing_signal_feature_quality.csv`, `horizon_label_summary.csv`, `feature_stability_by_regime.csv`, `venue_state_conditioning_summary.csv`, and `pricing_signal_recommendation.md`.
- The runner consumed only accepted local `0601T002/0601T003/0601T004` artifacts and did not perform network collection.
- Primary evidence remains `3596` rows with `3` excluded rows; generated pricing signal rows are `21541`.
- The runner enforces the four `0601T004` primary Binance allowlist features: `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty`.
- Future labels are separated from decision-time input columns and include nominal horizon plus `effective_future_age_ms`.
- Recommendation is `keep_for_read_only_research` with `single_public_sample_caveat=true`.
- Binance and Hyperliquid trade pressure remain disabled as `disabled_unverified_side_semantics`.
- This does not authorize strategy implementation, private/order endpoints, order lifecycle, live/default-on/tiny-live, parameter search, schema/connector/core API changes, or promotion.

## 0601T004 QA Findings

- `0601T004` passed QA as the Binance-led Hyperliquid maker data input / next-runner contract.
- Output contract: `docs/binance_led_hyperliquid_maker_data_input_contract.md`.
- Task artifacts: `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/`.
- The contract separates Binance lead pricing inputs from Hyperliquid lag venue-state/context inputs.
- Primary Binance lead allowlist is `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty`.
- Diagnostic-only inputs include absolute Binance top5 microprice, rolling volatility/liquidity context, Hyperliquid venue-state conditioning fields, and contract-caveated basis/dislocation fields.
- Binance and Hyperliquid trade pressure remain disabled until side semantics are separately proven by a QA-accepted task.
- The only authorized follow-up is a later read-only pricing-signal runner. No private/order endpoint, order lifecycle, strategy implementation, live/default-on/tiny-live, parameter search, schema/connector/core API change, or promotion is authorized.

## 0601T003 QA Findings

- `0601T003` passed QA as the read-only Binance-to-Hyperliquid lead-lag stability analyzer.
- Accepted output directory: `local_live_analysis/cross_exchange_lead_lag_analysis_0601T003/`.
- Primary rows: `3596`; excluded rows: `3`; horizon observations: `129246`.
- Verdict counts: `18 stable_enough_for_pricing_research`, `6 watch_only`, `30 unstable`, `0 insufficient_samples`.
- QA accepted the effective future-age audit fields in the horizon/regime/basis/venue-state summaries.
- Because the current Hyperliquid decision grid is roughly 500ms, nominal `100/250/500ms` horizons may map to the same future row; later tasks must report both nominal horizon and effective future age.
- The result supports only later read-only pricing-signal/data-input contract design and does not authorize strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.

## 0602T001 QA Findings

- `0602T001` passed QA as a synchronized public-only Binance lead / Hyperliquid lag collection task.
- The accepted output directory is `local_live_analysis/cross_exchange_public_sample_0602T001/`.
- Synchronization overlap is `1800.105259472s`, passing both the `600s` minimum and `1800s` target gates.
- Binance public data is present with `depthUpdate=67210`, `trade=135126`, `bookTicker=816079`, public depth snapshot status `ok`, `top5_row_count=67211`, `first_valid_update_aligned=true`, `depth_pu_mismatch_count=0`, and final data row mapping coverage `1.0`.
- Hyperliquid public data is present with `l2Book=3332`, `trades=3311`, `trade_event_count=12840`, `topn_coverage=1.0`, `decision_join_coverage=1.0`, `future_join_count=0`, `missing_join_count=0`, and classification `passes_pricing_research_market_view`.
- The sample is accepted only as synchronized public-data input for a later `0601T002` read-only join. It does not establish a Binance-lead / Hyperliquid-lag statistical effect and does not authorize private/order endpoints, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.

## 0601T001 QA Findings

- `0601T001` passed QA as a Hyperliquid lag-venue public-only BTC sample collection and alignment task.
- The accepted output directory is `local_live_analysis/hyperliquid_public_sample_0601T001/`.
- Raw sha256 is `62ebed4f4cc7a5fc9846f9491f9bd3ae0f06ab5b5f1a766d15aa844c50c1bd4c`; collection duration was `1800.093269476s`.
- Collection metrics are `l2Book=3328`, `trades=4939`, `subscriptionResponse=2`, reconnect count `0`, and recovery snapshot count `1`.
- Alignment classification is `passes_pricing_research_market_view`, with `topn_coverage=1.0`, `decision_join_coverage=1.0`, `future_join_count=0`, `missing_join_count=0`, and event order validation `passed`.
- This remains lag-venue state / execution-context evidence only and does not authorize private/order endpoints, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.

## 0531T001 QA Findings

- `0531T001` passed QA as the Hyperliquid public market-data research consumer read-only implementation.
- The consumer reads accepted `0529T004` local public artifacts and writes deterministic outputs under `local_live_analysis/hyperliquid_market_data_research_0531T001/`.
- Final classification is `passes_pricing_research_market_view`; trade pressure remains explicitly disabled with `unverified_side_semantics`.
- QA accepted the implementation as read-only/local/public-only and found no private/order/live/parameter/default-on/tiny-live/promotion boundary crossing.

## 0531T002 Task Boundary

- `0531T002` has been created as the Binance Stage 9N clean-fill evidence viability refinement task.
- It is unblocked by `0530T002` QA, which passed after total controller ratified / accepted the already collected Stage 9M artifact. The task remains read-only and still must not collect new data.
- The task must read only accepted local artifacts from `0529T002`, `0529T005`, and `0530T002`; it must not collect new data, deploy remotely, modify strategy behavior, enable candidates, relax guards, run parameter search, default-on behavior, tiny-live, promotion, replay semantic changes, connector/core API changes, schema changes, or Hyperliquid work.
- Required outputs live under `local_live_analysis/stage9n_clean_fill_refinement_0531T002/` and must include run manifest, fill-flow decomposition, axis fill-rate summary, top-gap viability, candidate-regime triage, collection-time estimate, and recommendation markdown.
- The core decision is whether Stage 9M evidence supports stopping top-gap collection, doing only a short threshold-crossing collection, or pivoting to alternative decision-visible regime refinement. Any policy design or strategy implementation remains a separate later task after QA.

## 0530T002 Findings

- `0530T002` passed QA after total controller explicitly ratified / accepted the already collected Stage 9M artifact `5-31-stage9m-cleanfill-control-120min-a`.
- Existing-sample scan came first. The only usable not-yet-included sample, `5-13-day-control-30min`, had `0` top-gap rows and `0` top-gap fills, so it could not add relevant Stage 9L clean-fill evidence.
- One new current-format no-rule/default-off control sample was collected as `5-31-stage9m-cleanfill-control-120min-a`, from `2026-05-30T16:16:06Z` to `2026-05-30T18:16:06Z`, with deployed commit `4760d481da3a06021ce25f9de4f2f0914662c5e0`, `git.dirty=false`, stop exit code `0`, and archive sha256 `f25ff59f0dc67bfc5a1ac99d43612ac1acdfdcb451ff7d26a20feb0eba3234f7`.
- New sample validation passed: maker acceptance and market-view passed; T009 decision join coverage is `1.0`; future/gap/missing joins are `0/0/0`; `top5_join_age_ms_p99=27.6887635`; top5 tick/qty match is `0.9618792312/0.9463946567`.
- New sample derived labels are usable: Stage 5 has `5742` submits, `106` fills, and `42` fill-after-cancel orders; Stage 5C has `0` post-only risk rows after recheck; Stage 6 is `methodology_valid_single_sample` with `5741` matched submits and live/replay fills `106/108`.
- Stage 9K aggregate after adding the sample has clean-only rows/fills `41008/1102`, up from `35266/994`, but `ready_for_policy_design` remains `0`; Shape A and Shape B candidate rows remain `0`.
- The top Stage 9L clean-fill gap only improved from `2605` rows / `34` fills / `7` samples / `6` fill samples to `2901` rows / `36` fills / `8` samples / `7` fill samples. It still needs `4` more fills for the Stage 9L minimum and did not meet the interpretive `+20` top-gap target.
- Stage 9L final classification remains `needs_targeted_clean_fills`; coarsened ready bucket count remains `0`; coarsened needs-more-clean-fills buckets are `335`; coarsened reject-quality-negative buckets are `677`; shape candidate count remains `0`.
- QA reproduced the Stage 9M chain and did not find metric/artifact failures. Original fixed logs/reports did not record the required pre-start approval for the remote/live `120min` collection, but total controller later ratified / accepted the already collected artifact; QA acceptance is based on that current controller decision.
- Policy design remains blocked. This task does not authorize strategy behavior changes, candidate enablement, guard relaxation, parameter search, default-on behavior, tiny-live, promotion, replay semantic changes, exact queue claims, hidden queue assumptions, connector/core API changes, or Hyperliquid work.

## 0531T001 Task Boundary

- `0531T001` has been created as a read-only Hyperliquid public market-data research consumer implementation task.
- It is unblocked by `0530T001` QA, which passed on 2026-05-31.
- The implementation must read only accepted local `0529T004` public artifacts and write deterministic research artifacts under `local_live_analysis/hyperliquid_market_data_research_0531T001/`.
- Required outputs are `run_manifest.json`, `market_view_timeseries.csv`, `pricing_features.csv`, `feature_quality_summary.json`, `sample_session_quality_summary.json`, and `research_recommendation.md`.
- The task must not collect a new sample, connect to Hyperliquid public or private endpoints, implement private connector or order lifecycle, run strategy live, run parameter search, default-on behavior, tiny-live, promotion, connector/core API changes, standard npz schema changes, or canonical audit schema changes.

## 0531T001 Implementation Notes

- The consumer has been implemented at `examples/hyperliquid/hyperliquid_market_data_research.py` with focused tests in `examples/hyperliquid/test_hyperliquid_market_data_research.py`.
- It reads only the accepted `0529T004` public sample artifacts, validates raw sha256 consistency across the raw file and manifests, and writes deterministic research outputs under `local_live_analysis/hyperliquid_market_data_research_0531T001/`.
- Generated outputs include `run_manifest.json`, `market_view_timeseries.csv`, `pricing_features.csv`, `feature_quality_summary.json`, `sample_session_quality_summary.json`, and `research_recommendation.md`.
- Final classification on the accepted sample is `passes_pricing_research_market_view`.
- Trade pressure is intentionally left disabled with explicit `unverified_side_semantics` status so ambiguous public trade side semantics do not become candidate-ready decision features.
- No fresh collection, private connector, order lifecycle, strategy live, parameter search, default-on behavior, tiny-live, promotion, or canonical schema change was introduced.

## 0531T002 Findings

- `0531T002` finalized as `short_collection_to_cross_minimum_only`.
- Top Stage 9L gap advanced from `34 -> 36` fills and remains `4` fills short of the clean `40` minimum.
- The new `120min` control sample added `+108` aggregate Stage 9K clean-only fills, but only `+2` fills reached the top gap.
- Top-gap viability is still better than the main alternatives on markout and spread capture; the alternatives are higher-fill-rate but materially worse quality.
- Estimated follow-up window is `4h-6h` only to cross the minimum; the `+20` interpretive target is not worth pursuing on this line.
- No new data collection, strategy change, candidate enablement, guard relaxation, parameter search, default-on, tiny-live, promotion, replay semantic change, connector/core API/schema change, or Hyperliquid work was introduced.

## 0530T002 Task Boundary

- `0530T002` has been created as the Binance Stage 9M targeted clean-fill evidence collection / read-only rerun task following `0529T005`.
- The task targets the top Stage 9L gap: `churn_warning_coarsened / large_skew_or_low_score / add_side / step_back_gt1 / edge_non_adverse / market_view_usable / post_only_clean / warning_churn_context`, which had `2605` rows, `34` fills, `7` samples, and needed `6` more clean fills for the Stage 9L minimum threshold.
- Target evidence is current-format no-rule/default-off control only. The task should first scan existing accepted current-format samples; if none can add relevant clean-fill evidence, it may prepare one `120min` control collection, but remote/live startup requires separate explicit approval.
- The target is preferably at least `+20` clean fills in the top-gap regime, or at least `+60` clean fills across Stage 9L gap regimes, followed by maker acceptance, T009 sidecar/join, Stage 5, Stage 5C, Stage 6, Stage 9K, and Stage 9L rerun. This target is interpretive, not a hard QA pass/fail gate.
- The task must report whether any coarsened bucket reaches `ready_for_policy_design_after_coarsening`; absent that, policy design remains blocked.
- It does not authorize strategy behavior changes, candidate enablement, guard relaxation, parameter search, default-on behavior, tiny-live, promotion, replay semantic changes, exact queue claims, hidden queue assumptions, connector/core API changes, or Hyperliquid work.

## 0530T001 Findings

- `0530T001` passed QA as a design-only/read-only Hyperliquid public market-data research consumer contract task.
- Direct input is the accepted `0529T004` fresh public-only BTC sample and its QA-approved artifacts under `local_live_analysis/hyperliquid_public_sample_0529T004/`.
- The design document is `docs/hyperliquid_public_market_data_research_consumer_design.md`.
- Official Hyperliquid public docs were reachable and rechecked successfully during execution; QA also rechecked the public docs URLs and received HTTP 200.
- The accepted consumer contract defines required inputs, output artifacts, allowed public decision-time-visible pricing / market-view features, diagnostic-only labels, quality gates, classification taxonomy, and a later read-only implementation boundary.
- The immediate next Hyperliquid task should be read-only consumer implementation over accepted local `0529T004` artifacts only.
- The task is independent from Binance `0529T005` and must not modify Binance Stage 9L work.
- It does not authorize consumer implementation, private connector, account endpoints, order submit/cancel/fill lifecycle, strategy live logic, parameter search, default-on behavior, tiny-live, promotion, connector/core API changes, standard npz schema changes, or canonical audit schema changes.

## 0529T005 Findings

- `0529T005` implements the read-only Stage 9L fill-quality rejection decomposition runner at `examples/binance_tick_mm/fill_quality_rejection_decomposition.py` with focused tests at `examples/binance_tick_mm/test_fill_quality_rejection_decomposition.py`.
- The runner consumes `local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002/run_manifest.json`, reconstructs row-level observed submit rows from existing Stage 5 labels, Stage 5 fill markouts, Stage 5C safety diagnostics, and live audit fields, and recomputes coarsened `sample_count` / `fill_sample_count` from row-level `sample_id`.
- Stage 9L artifacts are under `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/`: `run_manifest.json`, `rejection_reason_decomposition.csv`, `churn_gate_sensitivity.csv`, `coarsened_trigger_bucket_metrics.csv`, `coarsened_shape_candidates.csv`, `sample_gap_by_regime.csv`, and `stage9l_recommendation.md`.
- Final classification is `needs_targeted_clean_fills`; Shape A / Shape B candidate rows remain `0`, coarsened ready buckets remain `0`, coarsened needs-more-clean-fills buckets are `310`, and coarsened reject-quality-negative buckets are `662`.
- Churn hard-gate sensitivity shows demoting churn diagnostics to warning does not by itself create ready candidates:
  - original hard gate: `0` ready, `68` needs-more, `246` reject
  - recent reject/throttle as warning: `0` ready, `197` needs-more, `117` reject
  - fast-cancel / cancel-readd as warning: `0` ready, `154` needs-more, `160` reject
  - all non-true-reject churn diagnostics as warning: `0` ready, `283` needs-more, `31` reject
- Top targeted clean-fill gap is `churn_warning_coarsened / large_skew_or_low_score / add_side / step_back_gt1 / edge_non_adverse / market_view_usable / post_only_clean / warning_churn_context`, with `2605` rows, `34` fills, `7` samples, and `6` more clean fills needed to meet the Stage 9L minimum fill threshold.
- `0529T005` QA passed on 2026-05-30. QA reran help/py_compile/focused tests, manifest JSON parse, Stage 9L reproduction to `/tmp/qa_0529T005_stage9l`, and key artifact count checks; all passed.
- This remains read-only/default-off evidence. It does not authorize strategy behavior changes, live/default-on, parameter search, guard relaxation, tiny-live, promotion, exact queue proof, hidden queue assumptions, or replay semantic changes.

## 0529T004 Findings

- `0529T004` implements a narrow public-only Hyperliquid collector at `examples/hyperliquid/hyperliquid_public_sample.py` and focused tests at `examples/hyperliquid/test_hyperliquid_public_sample.py`.
- The collector writes line-oriented `raw.gz`, `raw.sha256`, `collection_manifest.json`, and `recovery_snapshots.jsonl` under `local_live_analysis/hyperliquid_public_sample_0529T004/`.
- The local `websockets` package was unavailable, so no dependency installation was performed; the collector used the already installed `websocket-client` fallback and records that fact in the manifest.
- Fresh 120s public-only BTC collection succeeded on mainnet after the first sandboxed DNS failure required network escalation: `l2Book=222`, `trades=111`, subscription responses `2`, connection attempts `1`, reconnect count `0`, startup Info `l2Book` recovery snapshot count `1`, raw sha256 `137018ef937b3692a5de0c12ee009c4a93a0e6d62ff15321061c377fc514389c`.
- The T003 alignment runner now consumes collection manifest and recovery snapshot evidence. Fresh T004 alignment produced `data.npz` with `4279` rows, raw parse errors `0`, trade events `418`, top-N coverage `1.0`, synthetic join coverage `1.0`, future joins `0`, missing joins `0`, and event-order validation `passed`.
- Final classification is `passes_pricing_research_market_view`, upgrading beyond T003's old local-sample `limited_pricing_research` because subscription/session/recovery evidence is now present.
- `0529T004` QA passed on 2026-05-30. QA reran help/py_compile/focused tests, raw sha256 check, metrics assertions, and alignment regeneration to `/tmp/qa_0529T004_alignment`; all passed.
- This remains market-data-only evidence. It does not authorize a Hyperliquid private connector, account endpoints, order submit/cancel, fill lifecycle, strategy live logic, parameter search, default-on behavior, guard relaxation, tiny-live, promotion, or Binance strategy changes.

## 0529T002 Findings

- `0529T002` implements the read-only Stage 9K fill-quality bucket synthesis runner and produces artifacts under `local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002/`.
- The runner uses existing artifacts only: Stage 5 execution labels, Stage 5 fill markouts, Stage 5C quote-anchor safety diagnostics, live audit quote-update fields, and Stage 6 manifest presence. It does not run live, replay, parameter search, or strategy code.
- Clean-only evidence across nine current-format samples has `35,266` observed submit rows and `994` fills.
- Decision-visible trigger bucket results: `314` clean-only buckets, `0` `ready_for_policy_design`, `68` `needs_more_clean_fills`, and `246` `reject_quality_negative`.
- Shape A passive quality gate with inventory sizing has `0` candidate rows; Shape B reduce-side participation with spread-capture floor also has `0` candidate rows.
- Current recommendation is `needs_more_clean_fills` / `collect_or_refine_read_only_evidence_before_policy_design`, not policy implementation. This does not authorize strategy behavior changes, live/default-on, parameter search, guard relaxation, tiny-live, promotion, exact queue proof, or replay semantic changes.
- `0529T002` QA passed. The accepted next read-only task is `0529T005`, which should test rejection decomposition and decision-visible bucket coarsening before any policy design or implementation.

## 0529T001 Findings

- `0529T001` shifts the Binance maker next-policy direction to fill-quality-first design. The immediate next task should be a read-only fill-quality bucket synthesis runner, not strategy implementation.
- `0528T001` rejected the fixed inventory-aware quote placement skeleton because request buckets increased fills but worsened quality: request 5s markout `-85.21` ticks versus no-change `-70.20` ticks, and request spread capture `6.85` ticks versus no-change `16.78` ticks.
- `0526T004` remains a negative constraint: the current `min_move_quote_age_churn_guard` projected-suppression grid produced `0` promising parameter sets, so it should not be the near-term main route.
- Stage 9I decomposition shows a tradeoff rather than a ready policy: `request_side_priority / allow_touch` has high fill rate (`0.0768`) but weak spread capture (`1.02` ticks), while `request_quote_adjustment / prefer_one_tick_tight` has better spread capture (`30.77` ticks) but lower fill rate (`0.0124`) and still adverse markout.
- Next policy candidates are design-only: passive quality gate with inventory sizing, and reduce-side participation gate with spread-capture floor. Both require bucket-level quality validation before any strategy implementation.
- `0529T001` does not authorize strategy behavior changes, live/default-on, parameter search, guard relaxation, tiny-live, promotion, exact queue proof, or more compact-audit work unless a regression appears.

## 0528T002 Findings

- `0528T002` implements the narrow `0527T001` recommendation: formal compact replay lifecycle audit export for Stage 6 input.
- Contract path: audit replay should write `out/backtest_audit_replay/audit_bt_audit_replay.compact_lifecycle.csv`; Stage 6 now prefers this compact artifact and falls back to the legacy CSV only when compact is absent.
- Compact semantics are intentionally limited: preserve decision rows and non-terminal lifecycle rows, while de-duplicating terminal lifecycle rows by first meaningful `event_type + order_id` fact.
- Bounded verification on the preserved `0526T008` full replay audit prefix scanned `250,000` rows, observed `227,080` `cancel_ack` rows, wrote `23,345` compact rows, and skipped `226,655` duplicate terminal rows.
- Stage 6 was validated in a task-scoped run dir against `audit_bt_audit_replay.compact_lifecycle.csv` and completed with `decision_state=methodology_valid_single_sample`; row counts matched the prior accepted lifecycle-min run shape (`live_submit_orders=11089`, `replay_submit_orders=11084`, `matched_submit_orders=11084`).
- This is an output/input scaling fix only. It does not improve or change live/replay fill/cancel semantics, queue/touch behavior, strategy behavior, live behavior, parameters, guards, default-on state, or promotion readiness.

## Open Findings

- `0510T001` completed the first workflow-run test against `5-10-day-control-1h-06` and is awaiting QA.
- `0510T002` completed cross-sample Stage 6J replay and is awaiting QA.
- `0511T001` should only design adverse-selection timing as a default-off rule using decision-time-visible inputs.
- `0511T002` should perform implementation only after the T001 design contract is accepted.
- `0511T003` should improve dashboard visibility of business results, QA conclusions, key metrics, and next-step decisions.
- `0511T004` completed the first diagnosis of why pure adverse timing candidates matched baseline and is awaiting QA.
- `0512T001` passed QA and established that Stage 6J replay cannot alone prove live adverse-selection source-path improvement.
- `0512T002` passed QA and authorized only the post-design implementation/offline replay task, not live.
- `0512T003` passed QA; `5-11-night-active` is accepted as the main current-format development/diagnostic sample, but does not authorize live.
- `0512T004` passed QA and authorizes `0512T002` only as a design-contract task.
- `0512T005` passed QA. It completed implementation/offline replay and does not authorize live.
- `0512T006` passed QA. It remains planning-only and authorizes only a later read-only T007 attribution implementation, not stricter candidate design, new replay, or live.
- `0512T007` passed QA. It is read-only attribution and does not authorize stricter candidate design, new replay, or live.
- `0512T008` passed QA. It is a read-only market-data / strategy-view quality gate and does not authorize strategy changes, new replay, or live.
- `0513T001` passed QA. It authorizes only bounded `0513T002` implementation for strategy-layer MarketView provenance / top5 audit transparency, not live, replay candidates, core API changes, converter/npz changes, or microprice/OFI/queue work.
- `0513T002` passed QA. It stays within the T001 file boundary and does not authorize live, replay candidates, core API changes, converter/npz changes, or strategy-rule changes.
- `0513T003` passed QA. It validated T002 provenance fields on `5-13-day-control-15min`, but does not prove PnL or full L2 alignment.
- `0513T004` passed QA. It adds a local deployment reproducibility and startup compatibility gate; no live, AWS, or strategy-rule change is authorized by it.
- `0513T005` passed QA as the Step 2 planning task for latency and market-data integrity baseline. It is planning-only and does not authorize code implementation, replay, live, or core/data schema changes.
- `0513T006` QA passed. It generated Step 2 read-only artifacts over existing local samples and does not authorize live, strategy changes, replay sweeps, or core/data schema changes.
- `0513T007` QA failed. It targets Binance raw provenance / top5 sidecar and read-only decision join, but full-run validation exposed a snapshot/bootstrap bug in the reconstructed top5 sidecar.
- `0513T008` QA passed. It collected one fresh no-rule control sample `5-13-day-control-30min` using T004 preflight and T007 full-run sidecar/join checks; it does not authorize new strategy rules or live promotion.
- `0513T009` QA passed. It fixes the T007 snapshot bootstrap / buffered depth replay bug using the existing `5-13-day-control-30min` sample only.
- `0514T001` QA passed. It implements Stage 3 market-view acceptance directly using `5-13-day-control-30min`; a separate planning-only task was not needed because Step 2 already supplied the required facts and artifacts.
- `0514T002` QA passed. It is planning-only for Stage 4 read-only pricing-model research and authorizes `0514T003` as a read-only implementation task.
- `0514T003` QA passed. It generated the Stage 4 read-only pricing-model research artifacts on `5-13-day-control-30min`.
- `0514T004` passed QA as a requirements-only follow-up for maker execution outcome research. It includes the seven added label gaps and per-label statistical method requirements, but does not itself authorize implementation or experiments.
- `0514T005` passed QA. It implemented the T004 requirements as a read-only execution outcome label runner with tests and dataset validation on `5-13-day-control-30min`.
- `0514T006` passed QA. It refines Stage 6 into replay/live fill-cancel lifecycle proxy calibration, defines matched-submit comparison as the primary unit, and does not authorize strategy changes, live, or exact queue proof.
- `0514T007` passed QA. It implemented the read-only Stage 6B replay/live lifecycle calibration runner on `5-13-day-control-30min` and concluded `diagnostic_only_gap_too_large`.
- `0514T008` passed QA. It concludes that the next useful task should diagnose replay fill/cancel lifecycle mismatch before sample-first expansion or quote-adjustment promotion discussion.
- `0515T001` passed QA. It built read-only diagnosis tables for replay-only fills, cancel timeline mismatches, terminal-state mismatches, and strata hot spots before any replay repair or sample expansion task.
- `0515T002` passed QA. It converts the mismatch evidence into a replay repair design contract with hypotheses, minimal scope, and validation gates.
- `0515T003` passed QA. It completed the narrow replay lifecycle repair and reduced the core same-sample mismatch materially, but residual queue/touch cases still need conservative handling.
- `0515T004` passed QA. It is now the accepted fact source for residual-case follow-up.
- `0516T001` passed QA. It classifies `4948` as `queue_ahead_depth_can_absorb_observed_trades`, but not as enough evidence for repair implementation.
- `0516T002` passed QA. It shows queue-ahead proxy no-fill pattern repeats, while replay-fill false-positive repeatability remains single-case (`4948`).
- `0518T001` passed QA as repair-design-only. It designs a future conservative queue proxy gate but explicitly does not authorize implementation.
- `0518T002` passed QA. It recommends fast BBO/bookTicker as the primary hard quote anchor, depth BBO as guarded fallback / consistency check, and top5 as pricing/risk/diagnostic context rather than the final hard post-only anchor. `0518T003` passed QA as a read-only diagnostic.
- `0518T004` has been created as a narrow Step 5C default-off / diagnostic-first quote-anchor safety task. It is not a source-level drift repair, not top5 hard-anchor promotion, not generic quote-control redesign, and not live promotion.
- `0518T004` passed QA. It keeps default behavior disabled, adds a reusable safety helper, and generates Stage 5C diagnostic counters with post-clamp risk `0` on `5-13-day-control-30min`.
- `0519T001` and `0519T002` have been created to close Step 6. `0519T001` is the read-only final lifecycle calibration rerun after accepted repairs; `0519T002` is the planning-only closure decision after `0519T001` QA.
- `0519T001` passed QA. Aggregate replay/live lifecycle is no longer `diagnostic_only_gap_too_large`; the rerun decision state is `requires_more_current_format_samples`, with one remaining `4948` residual and no repair authorization.
- `0519T002` passed QA. It closes Step 6 for roadmap progression, but not for promotion, live readiness, exact queue proof, or generalized queue/touch repair.
- `0519T003` passed QA. It closes Step 7 as a design-only inventory / execution model contract and does not implement strategy behavior, run experiments, start live, or authorize promotion.
- `0519T004` passed QA. It constrains quote-update mechanics, API/churn hygiene, stale/bad-price handling and post-only protection before any Step 7 controls are implemented.
- `0519T005` passed QA. Conclusion: `default_off_helper_candidate`; direct Step 9 remains blocked until a helper / instrumentation boundary is accepted.
- `0519T006` passed QA as Step 8C default-off quote-update helper / instrumentation implementation. It preserves default behavior.
- `0519T007` passed QA as Step 9A default-off quote-adjustment replay experiment design-only. It did not implement runner, run replay, start live, default-enable behavior, or make promotion claims.
- `0519T008` passed QA as Step 9B default-off quote-adjustment offline replay runner implementation. It validates runner / artifact mechanics on `5-13-day-control-30min` and correctly classifies that old sample as `needs_more_instrumentation`; no live, default-on, sample expansion, production behavior change, or promotion is authorized.
- `0519T009` passed QA. It collected `5-19-day-control-30min` as a current-format no-rule / default-off 30min control sample, verified the 15 T006 quote-update audit fields, and reran T008. It does not authorize candidate promotion or live readiness claims.
- `0519T010` passed QA as Step 9C planning-only. It defines multi-sample scenario coverage, replay validation method, cross-regime stability criteria, and promotion/live preconditions before any sample expansion or replay sweep.
- `0519T011` completed business-thread execution and passed QA. It collected 3 separated 30min current-format no-rule/default-off samples whose run ids begin `5-19-night-active`; it does not perform final multi-sample validation or authorize promotion.
- `0519T011` repaired and verified the `5-19-night-active-30min-b` second raw gzip after collection overran because the control session died. The original incomplete gzip is preserved as remote `.gz.corrupt`; local accepted artifacts use a regenerated 30min raw slice, and local/remote/archive gzip checks passed.
- `0519T011` reached Step 9C numeric research-comparison mass when combined with `5-19-day-control-30min`: about `123m01s`, `10005` submits, and `250` fills. The caveat below remains relevant for future read-only multi-sample validation.
- `0519T011` sample-quality caveat: `5-19-night-active-30min-a` has `first_valid_update_aligned=false`, `gap_crossed_join_count=28062`, and Step 5C missing anchor rows `28062`. If strict market-view quality is required on every sample, collect a replacement sample before read-only multi-sample validation.
- `0520T001` has been created for Step 9C read-only multi-sample validation. Total controller accepted `5-19-night-active-30min-a` only as a caveated research-comparison input, so T001 must also report clean-only sensitivity excluding that sample and must not claim strict market-view quality from it.
- `0520T001` passed QA. Accepted-set reaches Step 9C research-comparison mass (`123.02` min, `10005` submits, `253` fills), while clean-only sensitivity is under threshold (`92.92` min, `8721` submits, `232` fills).
- `0520T001` found no `ready_for_tiny_live_design` candidate. `fair_reservation_shift_edge_25` and `stale_latency_no_fresh_add` are rejected; `spread_widening_stale_latency` needs runner/artifact work; `inventory_reservation_shift_band`, `size_reduction_or_add_side_suppression_pressure`, `min_move_quote_age_churn_guard`, and `post_only_safety_interaction` remain `keep_for_research`.
- `0520T002` has been created as the next narrow runner/artifact hardening task. It should make the current 8 Step 9 families decisionable across existing current-format samples before any Step 10 tiny-live design discussion.
- `0520T002` QA passed. The current 8 Step 9 families now have explicit bucket verdicts, including a guard-suppressed verdict for `spread_widening_stale_latency`.
- `0521T001` collected `5-21-day-control-60min` successfully, then completed normal replay, audit replay, and archive. The sample kept no-rule / default-off behavior, no live promotion, no default-on, no strategy change, and no sample expansion.
- `0521T002` completed the read-only Step 9C candidate x scenario bucket multi-sample determination over five current-format samples and found no `ready_for_tiny_live_design` candidate.
- `0521T002` QA passed. The accepted next direction is to refine `keep_for_research` candidates by finer scenario buckets before any parameter-search or Step 10 tiny-live-design task.
- `0525T001` passed QA for Step 9D fine-bucket refinement. It determines whether `min_move_quote_age_churn_guard`, `inventory_reservation_shift_band`, and `size_reduction_or_add_side_suppression_pressure` contain stable promising buckets, reject buckets, or sample/fill gaps before any parameter-search task.
- `0525T001` completed business execution. `min_move_quote_age_churn_guard` is the strongest near-term parameter-sweep seed (`20` stable promising buckets, `8` parameter-sweep seed buckets, `0` reject buckets). `inventory_reservation_shift_band` has no parameter-sweep seed and `17` reject buckets, so it should not be a near-term search主线. `size_reduction_or_add_side_suppression_pressure` has only two narrow seed buckets (`one_tick_tight`, `volatility_high`) and `12` reject buckets, so it should not be global. Global tiny-live fill gap remains `160` fills (`340 / 500`).
- `0526T001` has been created as a targeted active current-format no-rule/default-off collection task. It should prioritize natural fills and `min_move_quote_age_churn_guard` seed regimes: young quote churn, API/churn normal plus churn pressure, stale latency medium/high, one-tick tight spread, medium volatility, and large inventory skew / low inventory score. It explicitly must not relax guards, target `inventory_reservation_shift_band`, enable candidates, or make promotion claims.
- `0526T001` QA passed after completing only the first sample because the user paused the second and third collection windows. `5-26-active-minmove-control-30min-a` is clean current-format no-rule/default-off: maker acceptance passed, sidecar/join has future/gap/missing `0/0/0`, top5 tick/qty match `0.9847/0.9711`, Stage 5 has `2526` submits and `46` fills, and Stage 6 live/replay fill/cancel counts match. The 5+1 aggregate keeps `min_move_quote_age_churn_guard` as the main seed line, but fill gap only shrinks from `160` to about `114`, so it still does not support `ready_for_tiny_live_design`.
- `0526T002` has been created for one 1H targeted active current-format no-rule/default-off control sample. It should sync the task commit to `awsserver1`, run from a task-scoped clean worktree, change only run id/output paths and stopper duration to `3600s`, and leave quote/risk/guard strategy parameters unchanged. The purpose is to reduce the remaining `~114` natural-fill gap while preserving the current safety boundary.
- `0526T002` completed business execution and is waiting for QA. `5-26-active-minmove-control-60min-a` added `5095` submits and `164` natural fills, with audit replay action/reject/throttle alignment `1.0`, Stage 6 live/replay fills `164/163`, and post-only risk after Step 5C recheck `0`. The sample is structurally clean for sidecar joins (`future/gap/missing=0/0/0`, top5 tick/qty `0.8930/0.8486`), but market-view strict acceptance is caveated because `top5_join_age_ms_p99=69.9977ms > 50ms`. Counting this caveated sample gives about `550` total fills, but strict clean-only fill mass remains about `386/500`; therefore it supports parameter-sweep design evidence but not `ready_for_tiny_live_design`.
- The 5+1+1 Step 9D aggregate after `0526T002` keeps `min_move_quote_age_churn_guard` as the only credible near-term parameter-sweep main line, but narrows decision-time-visible seed buckets to `inventory_state=large_skew_or_low_score` and `latency_stale_age=stale_latency_medium`. `inventory_reservation_shift_band` and `size_reduction_or_add_side_suppression_pressure` still have no parameter-sweep seed and should not be near-term main lines.
- `0526T002` QA passed. QA accepts `5-26-active-minmove-control-60min-a` only as caveated evidence, not as strict-clean market-view evidence or tiny-live readiness proof. The next reasonable task is a narrow `min_move_quote_age_churn_guard` parameter-sweep design around `large_skew_or_low_score` and `stale_latency_medium`, unless total controller chooses to prioritize another strict-clean active sample to satisfy the clean-only `500` fill gate first.
- `0526T003` passed QA as a design-only task for the narrow `min_move_quote_age_churn_guard` parameter-sweep contract. It is limited to the seed buckets `inventory_state=large_skew_or_low_score` and `latency_stale_age=stale_latency_medium`, designs the later sweep to use the current local `amdserver` resources (`32` CPU cores, about `60G` memory) through deterministic shard/reducer parallelism, and does not authorize runner implementation, sweep execution, live, tiny-live, default-on, guard relaxation, or promotion.
- `0526T004` passed QA. It implemented the read-only narrow `min_move_quote_age_churn_guard` sweep over seven current-format samples and found `0` `sweep_seed_promising`, `80` `reject`, and `676` `not_decisionable` parameter evaluations. This exact projected-suppression grid does not support live, tiny-live, default-on, guard relaxation, production behavior changes, or `ready_for_tiny_live_design`.
- Controller interpretation after `0526T004`: do not prioritize more filled-order collection specifically for the current `min_move_quote_age_churn_guard` projected-suppression line. The result did not produce `stable_but_low_fill` or other strong "collect more fills" evidence; it produced `0` promising, `80` reject, and `676` not-decisionable evaluations. The `inventory_only` seed produced the hard negative signal, while `stale_latency_only` and intersection were mostly under-filled. Therefore this guard should be downgraded to execution hygiene / instrumentation unless a new guard shape is proposed. Future sample collection should serve broader maker-edge evaluation, not continue this exact min-move/churn grid by default. The next higher-value research direction is fair-price / reservation / inventory / quote-distance / size-side logic that can directly improve spread capture, adverse-selection control, and inventory recovery.
- `0526T005` passed QA. The low-cost read-only triage found all five maker-edge families have enough clean evidence for follow-up, ranked by score as inventory, quote-distance, size-side, fair-price, then reservation. Inventory / quote-distance / size-side have the strongest immediate separation; fair-price and reservation remain promising but are highly similar under the current Stage 5 label view. This supports moving away from the min-move/churn guard grid and toward a focused maker-edge design task that combines inventory state, quote distance, side/size, and fair/reservation signal design. It still does not authorize live, default-on, or production behavior changes.
- `0526T006` has been formally dispatched as the current focused maker-edge design task. It should produce one inventory-aware quote-placement design, not five separate tracks: fair/reservation are pricing context, quote-distance and size/side are execution-control dimensions, and inventory state is the main organizing axis. After `0526T004` QA, the task contract was tightened to require a concrete candidate policy skeleton and a later read-only runner input/output contract, while still forbidding implementation, replay sweep, live, default-on, and promotion.
- `0526T006` passed QA. The accepted focused design is `inventory_aware_quote_placement_request`: inventory bucket controls side preference and add/reduce-side size pressure, fair/reservation edge controls whether a side is favorable enough to keep/place, quote-distance sets the touch/one-tick/step-back participation frontier, and post-only/stale/latency/anchor fields remain safety context. It supports a later read-only/default-off runner implementation task only; it does not authorize strategy implementation, parameter search, live/default-on, guard relaxation, or promotion.
- `0528T001` has been created to implement that accepted design as a read-only/default-off offline runner. The required output directory is `local_live_analysis/stage9i_inventory_aware_quote_placement_0528T001/`; required artifacts include candidate decision rows, bucket metrics, clean-only stability, caveated sensitivity, participation/fill-loss, inventory recovery quality, quote mechanics safety, and recommendation markdown. This task still does not authorize strategy behavior changes, parameter search, live/default-on, guard relaxation, or promotion.
- `0528T001` passed QA. The runner output is `local_live_analysis/stage9i_inventory_aware_quote_placement_0528T001/`; all nine current-format samples were usable with audit and Stage 5C join coverage `1.0`. Clean-only evidence has `35,266` rows, `994` fills, `14,363` candidate request rows, and `528` request fills, so fill mass is sufficient; however request buckets are worse than no-change on 5s markout (`-85.21` vs `-70.20` ticks) and spread capture (`6.85` vs `16.78` ticks). Clean-only and caveated sensitivity both recommend `reject`. Theory summary: the skeleton found more fills, but they were worse fills because it turned inventory state into quote-placement requests before proving those request buckets had positive fill quality. The result rejects this fixed policy skeleton as-is and does not authorize strategy behavior changes, parameter search, live/default-on, guard relaxation, or promotion.
- `0526T007` has been created and directly dispatched per controller request as a 180min current-format no-rule/default-off live control collection task. It is intended to add longer-window natural fills and maker-edge diagnostic regimes, not to relax guards, enable candidates, run parameter sweep, or authorize live/default-on/promotion.
- `0526T007` started remote collection on `awsserver1` using commit `43fb586`, worktree `/home/admin/hft_live/worktrees/0526T007-makeredge-180min`, and run id `5-26-active-makeredge-control-180min-a`. Remote preflight passed with `git.dirty=false`, `compatibility.passed=true`, and `audit_field_count=159`; early audit header check found no missing T006 fields. The run has now stopped, the data has been pulled locally, audit replay and base maker acceptance passed with `action/planned/reject/throttle = 1.0`, and the archive `local_live_analysis/archive/5-26-active-makeredge-control-180min-a.tar.gz` was written with sha256 `f4a2091de3f48e798736e2d675b40475cbbc0eb9db8778a378d31929d3152935`.
- `0526T007` post-processing is now complete on `5-26-active-makeredge-control-180min-a`: `t009_fixed_sidecar` is present with `data.npz`, `raw_provenance.csv`, `raw_to_npz_mapping.csv`, `top5_sidecar.csv`, `joined_decisions.csv`, and join metrics. Stage 5 labels were generated with `7016` submit orders and `148` fills, Stage 5C safety diagnostics reported `17004` bid-clamped rows, `6434` ask-clamped rows, and `0` post-only risk after recheck, Stage 6 calibration is `methodology_valid_single_sample` with matched submit coverage `7016/7016` and live/replay filled orders `148/156`, Step 9B classification is `promising_but_single_sample`, and Step 9D fine-bucket refinement found `0` stable promising buckets and `0` parameter-sweep seed buckets on this single sample. This sample is structurally usable for maker-edge diagnostics, but it still does not authorize live/default-on/promotion or a parameter-search claim.
- `0526T007` QA passed. Accepted scope is data collection plus derived diagnostic artifacts only; it remains no-rule / default-off control evidence and does not change the current conclusion that a multi-sample parameter-search-ready maker strategy direction has not yet been established.
- `0526T008` has been created and directly dispatched per user override as an immediate 30min current-format no-rule/default-off live control collection task based on `0526T002`, with run id `5-26-active-minmove-control-30min-b`. It intentionally does not wait for unfinished `0526T006` / `0526T007`, but it does not authorize candidate enablement, guard relaxation, strategy changes, parameter sweep, tiny live, default-on, or promotion.
- `0526T008` QA passed. `5-26-active-minmove-control-30min-b` completed collection, raw recovery, archive, maker acceptance, T009 sidecar/join, Stage 5, Step 5C, Stage 6, Step 9B, and Step 9D; maker and market-view gates passed, T009 future/gap/missing is `0/0/0`, Stage 5 has `11089` submit orders and `471` fills, Stage 6 is `methodology_valid_single_sample`, Step 9B is `promising_but_single_sample`, and Step 9D has `0` stable promising buckets. It remains no-rule/default-off control data only. The full replay audit repeated `cancel_ack` lifecycle rows and inflated to about 21GB, so Stage 6 used a documented lifecycle-minimized input workaround; future repair is tracked by `0527T001`.
- `0527T001` has been refined into a planning/diagnosis-only task for the `0526T008` replay audit bloat. It should not repair code yet; it should determine whether repeated `cancel_ack` rows originate from audit export repetition, replay lifecycle state repetition, or Stage 6 input assumptions, identify the duplicate key, decide whether repeated rows are redundant for Stage 6 labels, and recommend the smallest later implementation boundary.
- `0527T001` passed QA. The `0526T008` full replay audit has `32,893,719` `cancel_ack` rows but only `10,380` unique cancel-ack `order_id`s; top repeated orders each appear over `325k` times. The source is primarily audit replay lifecycle export / order-state tracking repetition: `OrderLifecycleTracker.observe()` retains current snapshots from `hbt.orders(0)`, and audit replay overlay writes both regular cancel_ack rows and forced live terminal constraint rows as decisions advance. Repeated rows are semantically redundant for Stage 6 once the first terminal fact per order is kept. QA recommends two next steps: create a narrow compact lifecycle replay audit export / writer-side terminal-order de-dup implementation task, and keep full forensic audit optional/off the default Stage 6 path while avoiding strategy/live/promotion changes.
- `0528T002` has been created and narrowed to implement the first `0527T001` QA recommendation through one route: produce a formal compact replay lifecycle audit artifact for Stage 6 input, with terminal lifecycle rows de-duplicated inside the compact export by Stage 6 semantics. Full forensic audit remains optional evidence only; strategy/live/fill-cancel semantic/promotion changes remain out of scope.
- Step 9 narrow `min_move_quote_age_churn_guard` parameter-sweep design can use these facts without re-discovery:
  - Primary decision sources are `local_live_analysis/stage9d_candidate_bucket_refinement_0526T002_aggregate/fine_bucket_stability_summary.csv`, `fine_bucket_metrics.csv`, and `candidate_bucket_recommendations.md`.
  - Current current-format sample set is `5-19-day-control-30min`, `5-19-night-active-30min-a`, `5-19-night-active-30min-b`, `5-19-night-active-30min-c`, `5-21-day-control-60min`, `5-26-active-minmove-control-30min-a`, `5-26-active-minmove-control-60min-a`, and `5-26-active-makeredge-control-180min-a`.
  - Caveated samples are `5-19-night-active-30min-a` and `5-26-active-minmove-control-60min-a`; use them for sensitivity / broader evidence only, not strict-clean promotion proof.
  - `5-26-active-makeredge-control-180min-a` is a newly added current-format no-rule/default-off control sample from `0526T007`: effective live audit duration is about `65.67` minutes, audit replay and base maker acceptance passed with `action/planned/reject/throttle = 1.0`, archive sha256 is `f4a2091de3f48e798736e2d675b40475cbbc0eb9db8778a378d31929d3152935`, and Stage 5 / Step 5C / Stage 6 / Step 9 derived artifacts still need to be generated before using it in aggregate candidate decisions.
  - Narrow sweep scope should be `min_move_quote_age_churn_guard` only, seeded only by decision-time-visible buckets `inventory_state=large_skew_or_low_score` and `latency_stale_age=stale_latency_medium`.
  - Step 9D aggregate is the primary decision evidence. Per-sample Step 9B outputs, Stage 5 execution labels, Stage 6 calibration, and Step 5C anchor/post-only safety outputs are supporting mechanism / risk diagnostics.
  - Current evidence does not support Step 10 tiny live, default-on, guard relaxation, or `ready_for_tiny_live_design`. Strict clean-only fill mass remains about `386/500`.
- `0521T002` accepted-set mass is `183.03` min, `15919` submits, `340` fills; clean-only mass is `152.93` min, `14635` submits, `319` fills.
- `0521T002` verdicts are: `baseline_control` keep_for_research, `fair_reservation_shift_edge_25` reject, `inventory_reservation_shift_band` keep_for_research, `min_move_quote_age_churn_guard` keep_for_research, `post_only_safety_interaction` reject, `size_reduction_or_add_side_suppression_pressure` keep_for_research, `spread_widening_stale_latency` keep_for_research but guard-suppressed, and `stale_latency_no_fresh_add` reject.
- `0521T002` became decisionable after restoring the missing derived chain for `5-21-day-control-60min`: `t009_fixed_sidecar/joined_decisions.csv`, Stage 5 labels, Stage 5C safety diagnostics, Stage 6 calibration, and the Stage 8B planning-decision placeholder.

## 0519T006 Task Boundary

- T006 exists to implement the next boundary recommended by T005, not to start Step 9.
- Scope is default-off helper / instrumentation only:
  - quote-update intent/action/reason
  - min-move, quote-age, join/anchor-age, latency-bucket fields
  - throttle/token/cancel-readd state fields
  - reject/throttle/drop cause
  - post-only pre/post-check fields
  - inventory request id placeholder
- Required invariant: existing action path, throttle/API/latency suppression, quote placement, cancel/submit behavior, Step 5C default-off status, and Step 7 design-only status remain unchanged by default.
- T006 does not authorize replay sweep, live, default-on behavior, Step 5C promotion, inventory-control implementation, or Step 9 promotion.

## 0519T006 Findings

- Step 8C now has a shared quote-update audit helper in `strategy_core.py`, wired from both `backtest_tick_mm.py` and `live_tick_mm.py`.
- `0519T006` QA passed on 2026-05-19 15:37 CST.
- New audit fields are present in `AUDIT_FIELDS`:
  - `quote_update_intent`
  - `quote_update_action`
  - `quote_update_reason`
  - `min_move_passed`
  - `quote_age_ms`
  - `join_age_ms`
  - `anchor_age_ms`
  - `latency_bucket`
  - `throttle_state`
  - `token_bucket_state`
  - `cancel_readd_bucket`
  - `reject_throttle_drop_cause`
  - `post_only_pre_check`
  - `post_only_post_check`
  - `inventory_request_id`
- Default behavior remains unchanged: the helper records snapshots and audit fields after existing latency/throttle/API/post-only decisions are formed; it does not choose, suppress, submit, cancel, reprice, or promote quotes.
- Placeholder / diagnostic-only boundary:
  - `inventory_request_id` is a passive placeholder.
  - `queue`, cancel-readd, quote/join/anchor age, latency bucket, token/throttle state, and post-only pre/post checks are diagnostic/proxy fields only.
  - No Step 9 replay sweep, live run, default-on behavior, Step 5C promotion, or Step 7 inventory-control implementation was done.

## 0519T007 Task Boundary

- T007 is Step 9A design-only.
- It should define candidate matrix, decision-time-visible inputs, metrics, artifacts, Step 9B implementation boundary, and non-goals for default-off quote-adjustment offline replay.
- It should explicitly carry forward:
  - Step 5C post-only safety remains default-off / diagnostic-first unless separately enabled
  - Step 6 lifecycle closure is enough for roadmap progression but not exact queue/live promotion proof
  - Step 7 inventory controls are design-only and can only express future requests through shared quote-update fields
  - Step 8 / T006 helper fields are instrumentation and explanation fields, not strategy control flow
- T007 does not authorize runner implementation, replay sweep, live, default-on behavior, sample expansion, Step 5C promotion, inventory-control implementation, or promotion claims.

## 0519T007 Findings

- Step 9A is ready for QA as a design contract, not as an implementation.
- `0519T007` QA passed on 2026-05-19 15:57 CST.
- Candidate matrix is grouped into eight families:
  - baseline/control no-change replay validation
  - fair / reservation shift
  - inventory reservation shift / recovery-side preference request
  - spread widening
  - size reduction / add-side suppression
  - stale or latency no-fresh-add regime
  - min-move / quote-age / API-churn guard
  - Step 5C post-only safety interaction
- Each candidate family is constrained to decision-time-visible inputs. Future markout, fill outcome, audit replay overlays, exact queue claims, and `4948`-specific logic remain disallowed as decision inputs.
- T006 fields become the Step 9 explanation layer:
  - `quote_update_intent`, `quote_update_action`, `quote_update_reason`
  - `min_move_passed`, `quote_age_ms`, `join_age_ms`, `anchor_age_ms`, `latency_bucket`
  - `throttle_state`, `token_bucket_state`, `cancel_readd_bucket`, `reject_throttle_drop_cause`
  - `post_only_pre_check`, `post_only_post_check`, `inventory_request_id`
- Step 9B should be a minimal default-off offline runner task if QA accepts T007:
  - validate runner mechanics on `5-13-day-control-30min`
  - emit candidate matrix, per-candidate metrics, action-path/audit coverage, fill-quality, inventory-cycle, API/churn and post-only safety artifacts
  - classify results as `no_effect`, `worse_due_to_churn_or_fill_quality`, `promising_but_single_sample`, or `blocked_by_replay_or_market_view`
  - keep all candidates default-off and offline-only
- Sample expansion should come after the runner and candidate methodology are accepted, unless QA finds a design blocker that requires data first.

## 0519T008 Task Boundary

- T008 is Step 9B default-off offline runner implementation.
- It should implement a local runner and focused tests, then validate runner / metrics / artifacts on `5-13-day-control-30min`.
- Required output directory:
  - `local_live_analysis/5-13-day-control-30min/stage9b_quote_adjustment_replay_0519T008/`
- Required candidate families:
  - `baseline_control`
  - `fair_reservation_shift`
  - `inventory_reservation_shift`
  - `spread_widening`
  - `size_reduction_or_add_side_suppression`
  - `stale_latency_no_fresh_add`
  - `min_move_quote_age_churn_guard`
  - `post_only_safety_interaction`
- Required classification:
  - `no_effect`
  - `worse_due_to_churn_or_fill_quality`
  - `promising_but_single_sample`
  - `blocked_by_replay_or_market_view`
  - `needs_more_instrumentation`
- T008 does not authorize live, default-on behavior, production strategy behavior changes, sample expansion, Step 5C promotion, inventory-control implementation, queue/touch repair, or promotion claims.

## 0519T008 Findings

- Step 9B runner exists at `examples/binance_tick_mm/quote_adjustment_replay.py` with focused tests in `examples/binance_tick_mm/test_quote_adjustment_replay.py`.
- Output directory:
  - `local_live_analysis/5-13-day-control-30min/stage9b_quote_adjustment_replay_0519T008/`
- All required artifacts were generated:
  - `run_manifest.json`
  - `candidate_matrix.csv`
  - `candidate_matrix.json`
  - `candidate_summary.json`
  - `candidate_metrics.csv`
  - `fill_quality_by_candidate.csv`
  - `inventory_cycle_metrics.csv`
  - `api_churn_metrics.csv`
  - `post_only_safety_metrics.csv`
  - `action_path_coverage.csv`
  - `audit_field_coverage.csv`
  - `candidate_decision_samples.csv`
  - `acceptance_decision.md`
- Classification is `needs_more_instrumentation`.
- Reason:
  - the existing `5-13-day-control-30min` audit was collected before T006 and is missing all 15 T006 quote-update audit fields
  - the runner therefore used proxy fields to validate mechanics and metrics rather than treating results as candidate performance proof
- Key diagnostic counts:
  - decision rows `47499`
  - submit orders `2516`
  - candidate families `8`
  - missing T006 fields `15`
  - baseline fill rate about `0.021065`
  - baseline fill-after-cancel rate about `0.006359`
- Interpretation:
  - T008 validates the runner / metrics / artifact path.
  - It does not authorize live, default-on, sample expansion, production behavior change, or promotion.
  - Before promotion-style claims or true candidate evaluation, use a sample/replay that contains T006 quote-update fields, or explicitly accept a proxy-only diagnostic boundary in a later task.

## 0519T009 Task Boundary

- T009 is a data-collection / audit-rerun task, not a strategy task.
- Dataset name is fixed as `5-19-day-control-30min`.
- The collection must remain no-rule / default-off control.
- The early audit header check must confirm all 15 T006 fields:
  - `quote_update_intent`
  - `quote_update_action`
  - `quote_update_reason`
  - `min_move_passed`
  - `quote_age_ms`
  - `join_age_ms`
  - `anchor_age_ms`
  - `latency_bucket`
  - `throttle_state`
  - `token_bucket_state`
  - `cancel_readd_bucket`
  - `reject_throttle_drop_cause`
  - `post_only_pre_check`
  - `post_only_post_check`
  - `inventory_request_id`
- If any of these fields are missing, stop the run and mark the task blocked rather than producing another proxy-only 30min sample.
- After collection, rerun `align_live_run.py`, `maker_acceptance.py`, and `quote_adjustment_replay.py` on the new dataset. A single 30min sample can remove the instrumentation blocker, but it still cannot prove promotion or live readiness.

## 0519T009 Findings

- Dataset: `5-19-day-control-30min`.
- Deployed commit: `2d0cae2`.
- Preflight passed:
  - `git.dirty=false`
  - `compatibility.passed=true`
  - audit field count `159`
- Run markers:
  - start marker UTC `2026-05-19T09:13:58Z`
  - stop marker UTC `2026-05-19T09:46:46Z`
  - stop marker exit code `0`
- Live audit:
  - rows `122124`
  - fields `159`
  - all 15 T006 quote-update fields are present
- Raw collection note:
  - collector gzip lacked a footer after tmux session shutdown
  - original remote file was preserved as `btcusdt_20260519.gz.corrupt`
  - complete raw lines were recovered and recompressed to a valid `btcusdt_20260519.gz` for local replay
- Maker acceptance:
  - passed `true`
  - hard failures `[]`
  - common rows `96340`
  - all 21 checks passed
- T009 sidecar / join:
  - first valid update aligned `true`
  - depth `pu` mismatch `0`
  - decision join coverage `1.0`
  - future join `0`
  - gap-crossed join `0`
- Stage 5 labels on the new dataset:
  - submit orders `4098`
  - filled orders `110`
  - fill-after-cancel orders `53`
  - fast-cancel churn rate about `0.86164`
- Step 5C diagnostics on the new dataset:
  - decision rows `96341`
  - bookTicker anchor rows `79449`
  - guarded depth fallback rows `16892`
  - post-only risk after re-check rows `0`
- T008 rerun on the new dataset:
  - `missing_t006_field_count=0`
  - classification `promising_but_single_sample`
  - reason: active candidates have nonzero coverage, but only one current-format sample is available
- Interpretation:
  - T009 removes the old `needs_more_instrumentation` blocker for one current-format sample.
  - The result is still single-sample default-off offline diagnostic evidence, not live readiness, not promotion, and not generalized profitability proof.

## 0519T010 Findings

- Step 9C exists to answer the question: can default-off quote-adjustment candidates improve maker execution quality across regimes, not just look good in one sample.
- Accepted input roles:
  - `0519T008` proves `quote_adjustment_replay.py` runner / metrics / artifact mechanics.
  - `0519T009` proves one current-format T006 control sample is usable and removes the one-sample instrumentation blocker.
- Primary blocker is now data scenario coverage:
  - single current-format sample evidence cannot establish stable maker strategy behavior
  - sample/event mass must cover multiple volatility, spread, trade intensity, latency/stale, API/churn, inventory, post-only safety, cancel-fill, and market-view quality regimes
- Minimum research comparison target:
  - at least `4` current-format samples including `5-19-day-control-30min`
  - at least `120` minutes aggregate duration
  - at least `10000` submit orders aggregate
  - at least `250` filled orders aggregate
- Minimum before a later `ready-for-tiny-live-design` classification:
  - at least `5` current-format samples
  - at least `180` minutes aggregate duration
  - at least `15000` submit orders aggregate
  - at least `500` filled orders aggregate
  - at least `2` distinct non-calm regimes
- Hard gates for every accepted validation sample:
  - T006 missing fields `0`
  - maker acceptance passed
  - action/planned/reject/throttle gates passed
  - working semantic/blocking mismatch `0`
  - strict replay lag passed
  - sidecar/join quality acceptable, with future/gap-crossed join `0`
  - post-only crossed-risk after re-check `0`
  - archive/raw integrity documented
- Candidate interpretation rules:
  - `reject` if a candidate fails hard gates, lacks coverage, worsens fill quality/adverse markout/cancel-fill/API-churn in multiple samples, or uses forbidden inputs
  - `keep_for_research` if it has coverage and favorable regimes but insufficient sample/event mass or proxy-only evidence
  - `ready_for_tiny_live_design` only after multi-sample hard gates, stable execution-quality behavior, no catastrophic worst-sample result, and QA; it still authorizes only a separate live-design planning task
- Recommended next step after T010 QA:
  - collect more current-format no-rule/default-off samples first, rather than modifying the runner
  - then run a read-only multi-sample validation using the existing runner
  - modify the runner only if the accepted plan cannot be executed with current artifacts
- T010 remains planning-only:
  - no code change
  - no replay sweep
  - no live
  - no default-on
  - no promotion
  - no single-sample PnL acceptance

## 0519T005 Findings

- Step 8B conclusion is `default_off_helper_candidate`, not `full_default_off_replay_candidate`.
- Current sample shows material quote-update pressure and suppression:
  - decision rows `47499`
  - planned submit decision rows `8971`
  - actual submit decision rows `2256`
  - planned/action mismatch rows `7186`
  - latency guard rows `16528`
  - quote throttle rows `5996`
  - api interval guard rows `1190`
- Submit-level churn is already high:
  - submit orders with labels `2516`
  - fast-cancel churn rows `1955`
  - fast-cancel churn rate about `0.777027`
- Step 5C safety diagnostics are useful but still default-off / diagnostic-first:
  - bid clamped rows `1394`
  - ask clamped rows `2281`
  - stale anchor rows `65`
  - post-only risk after re-check rows `0`
- Observable now:
  - `action` / `planned_action` as actual vs suppressed quote-activity proxies
  - `reject_reason` / `throttle_reason` for latency, quote-throttle, and API interval gates
  - Stage 5C fast-anchor / guarded-fallback / anchor-age / clamp / suppress / recheck diagnostics
  - Stage 5 submit-level placement, latency, inventory, recent reject/throttle, fast-cancel churn, fill horizon, and fill-after-cancel labels
- Missing or proxy-only before implementation:
  - missing `quote_update_intent`
  - missing unified `quote_update_reason`
  - missing `token_bucket_state`
  - missing `inventory_request_id`
  - proxy-only `min_move_passed`, `quote_age_ms`, `cancel_readd_bucket`, `latency_bucket`
  - diagnostic-only `anchor_age_ms` and post-clamp `post_only_post_check`
- Recommended next boundary:
  - create a separate default-off helper / instrumentation task before Step 9
  - centralize quote-update intent/action/reason
  - record throttle/token/cancel-readd/post-only/inventory-request fields
  - preserve existing throttle/API/latency suppression semantics
  - keep default behavior unchanged
- T005 does not authorize strategy implementation, replay sweep, live, default-on, Step 5C promotion, or Step 9 promotion.

## 0519T004 Findings

- Step 8 should be completed as a design contract before implementation.
- Quote-update mechanics should be driven by observable safety and usefulness triggers:
  - bad-price ticks
  - minimum quote move
  - quote age
  - stale or missing anchor
  - join-age / latency regime
  - post-clamp post-only risk
  - inventory regime request from Step 7
- Preferred future action ordering:
  - hold quote when price is still useful and API/churn budget should be preserved
  - modify/replace in place if supported and safer than cancel+new
  - cancel+new only when quote is materially unsafe, stale, crossed-risky, inventory-worsening, or past bounded age
- GTX/post-only reject is an exchange backstop and diagnostic bucket, not normal control flow.
- Step 5C quote-anchor safety remains default-off / diagnostic-first unless a later implementation task explicitly changes that boundary.
- Step 7 inventory controls must express quote-change requests through shared update-intent fields and cannot bypass anti-churn, throttling, stale-anchor suppression, or post-only re-check.
- Anti-churn controls should include:
  - per-side min tick move
  - min quote age
  - max cancel/re-add rate
  - in-flight order guard
  - cancel-pending guard
  - recent reject/throttle cooldown
  - emergency stale/bad-price override
- API hygiene must explicitly model token bucket, request spacing, per-action budgets, cancellation-limit risk, reject/throttle/drop buckets, and degraded modes.
- Required future audit fields include quote update intent/action/reason, min-move pass flag, quote/anchor/join age, latency bucket, throttle/token state, cancel-readd bucket, reject/throttle/drop cause, post-only pre/post checks, and Step 7 inventory request id.
- Recommended next task after QA is not Step 9 yet. Open a narrow Step 8B read-only diagnostic / implementation-planning task over existing artifacts to quantify current churn/API/stale/bad-price regimes and decide whether implementation should be no-change, default-off helper, or full default-off replay candidate.

## 0519T003 Findings

- Step 7 should be completed first as a design contract, not as code implementation.
- Inventory objective:
  - keep normal exposure close to flat / one-order-quantity bands
  - make larger inventory a distinct recovery regime
  - reduce time spent in directional exposure instead of relying on symmetric quote churn
- Initial candidate controls are allowed only as future default-off designs:
  - reservation / fair shift by inventory band
  - spread widening on inventory-worsening side
  - add-side size reduction or add-side suppression when beyond one order quantity
  - recovery-side size preference when inventory is above target
  - volatility / fill-intensity driven AS-style spread and order amount
  - TTL / triple-barrier style exit handling only after pricing and lifecycle evidence exists
- Zero-crossing should be treated as an inventory-cycle boundary for diagnostics:
  - report cycle duration, max inventory excursion, recovery fills, markout while reducing inventory, and whether inventory crossed through zero cleanly
- Step 7 must use decision-time-visible inputs only:
  - current position / notional
  - target and working quote ticks
  - fair/reservation signals available at decision time
  - volatility / spread / top-of-book or top5 size-age proxies
  - latency / stale / join-age flags
  - live-safe lifecycle state such as in-flight, cancel-requested, and recent fill/cancel events
- Step 7 must not use future markout, audit replay overlays, exact queue claims, or `4948`-specific repair logic as live decision inputs.
- Required future audit fields before implementation:
  - inventory band
  - inventory cycle id
  - skew regime
  - quote-side suppression reason
  - size multiplier
  - spread multiplier
  - reservation shift
  - TTL / barrier state
  - recovery-mode marker
- Evidence gates before any default-off implementation or experiment:
  - replay acceptance and market-view gate remain clean
  - Step 6 lifecycle diagnostics remain within the accepted event-classification boundary
  - inventory-cycle metrics improve without hiding fill-quality or markout degradation
  - API/churn and post-only safety stay inside Step 5C/Step 8 boundaries
  - single-sample PnL is not enough for promotion or live
- Recommended next step after QA:
  - proceed to Step 8 design-only quote-update / API-limit hygiene before implementing Step 7 controls
  - use Step 8 to constrain whether Step 7 candidates can be expressed safely without blind cancel/re-add churn

## 0519T002 Findings

- Step 6 final state is `closed_for_roadmap_progression_requires_more_samples_for_promotion`.
- The old Stage 6 blocker is resolved enough to move forward:
  - `0519T001` QA confirmed the state improved from `diagnostic_only_gap_too_large` to `requires_more_current_format_samples`
  - matched submit coverage is complete at `2516/2516`
  - price tick and qty equality are `2516/2516`
  - live/replay filled orders are close at `53/54`
  - live/replay fill-after-cancel orders are close at `16/15`
- Step 6 remains bounded:
  - it is an event-classification and lifecycle-proxy closure, not exact queue proof
  - timing magnitude gaps remain in time-to-fill and cancel-to-fill delay
  - one residual remains: `28940|sell` / order `4948`
  - queue/priority, opportunity cost, and realized PnL decomposition remain observed-only proxies
- `4948` / queue-ahead proxy residual stays parked:
  - no generalized queue/touch repair is authorized
  - future repair would require more current-format samples or repeated replay false-positive evidence
- More current-format samples are still needed before promotion-style conclusions, live micro tests, or generalized queue/touch repair.
- More samples do not need to block Step 7 / Step 8 design work.
- Recommended sequence after QA:
  - start Step 7 as a design-only inventory / execution model task
  - then start Step 8 as a design-only quote-update / API-limit hygiene task
  - only after those design boundaries are accepted, open Step 9 as a default-off offline replay experiment
  - do not treat Step 9 as live promotion, and do not rely on single-sample PnL

## 0519T001 Findings

- `0519T001` reran the existing read-only Stage 6 calibration on `5-13-day-control-30min` after the accepted replay lifecycle repairs.
- Decision state improved from the original Stage 6B `diagnostic_only_gap_too_large` to `requires_more_current_format_samples`.
- Matched submit coverage remains complete:
  - live submit orders `2516`
  - replay submit orders `2516`
  - matched submit orders `2516`
  - matched price tick equality `2516/2516`
  - matched qty equality `2516/2516`
- Aggregate lifecycle is now close:
  - live filled orders `53`
  - replay filled orders `54`
  - live fill-after-cancel orders `16`
  - replay fill-after-cancel orders `15`
  - final-state filled gap `0.000397`
  - final-state canceled gap `0.000397`
  - fill-after-cancel-request rate gap `0.000397`
  - fast-cancel-churn gap `0.0`
- Fill horizon gaps are aligned enough on the matched universe:
  - `100ms` gap `0.001192`
  - `500ms` gap `0.000397`
  - `1000ms` gap `0.000397`
  - `5000ms` gap `0.000397`
- Remaining non-perfect timing differences are concentrated in timing magnitude, not event classification:
  - matched-any-filled time-to-fill mean gap about `400.07ms`
  - matched-both-filled time-to-fill mean gap about `364.54ms`
  - cancel-to-fill delay mean gap about `96.88ms` on both-observed rows
- Markout observability is close but still not identical at all horizons:
  - fill markout coverage gap at `500ms` is about `0.00159`
  - fill markout coverage gap at `5000ms` is about `0.00119`
  - mean markout ticks are identical at `500ms/1000ms/5000ms`, but `100ms` has about `3.55` ticks mean gap on a small observed subset.
- Residual diagnosis still reports one case:
  - `28940|sell` / order `4948`
  - class `residual_replay_fill_trigger_uncertain`
  - no nearby supportive trade evidence before live or replay anchor
  - this remains a queue-exposure / replay trigger uncertainty, not a repair authorization.
- Queue/priority, opportunity cost, and realized PnL decomposition remain observed-only proxies, not exact queue proof.
- T001 did not modify replay behavior, strategy behavior, live scripts, schema, or default-on behavior.
- `0519T001` has passed QA and unblocks `0519T002` as a planning-only Step 6 closure decision. It does not authorize Step 9 promotion or live readiness by itself.

## 0515T001 Findings

- `0515T001` implements a read-only replay lifecycle mismatch diagnosis runner over the same matched submit opportunity universe used by Stage 6B.
- On `5-13-day-control-30min`, replay-only fills are highly concentrated in live-canceled / replay-filled cases: `120` replay-only fill rows and `120` live-cancel / replay-filled rows.
- Replay-only fills are not mainly ultra-short-horizon events: among replay-only fill cases, `100ms=0`, `500ms=4`, `1000ms=6`, `5000ms=31`. This supports a long-horizon persistence bias hypothesis.
- Cancel timeline evidence points toward replay-side terminal / cancel-ack persistence issues rather than submit matching issues: many rows show live cancel-ack already reached while replay keeps the same submit key fill-eligible and later marks it filled.
- Terminal-state mismatch is replay-side dominant: the main transition pattern is `canceled -> filled`, with additional `canceled -> open_or_missing`, rather than symmetric noise.
- Placement hot spots are concentrated in deeper step-back orders: `step_back_gt1` shows replay-only-fill rate about `0.0569`, terminal-state-diff rate about `0.1113`, and large time-to-fill gap.
- Latency buckets `q4` / `q5` and some `q2` buckets also show especially large fill-after-cancel and time-to-fill gaps, indicating the mismatch is not uniform across the sample.
- Markout observability mismatch is likely lifecycle-induced: replay creates more fills first, which then creates more observable markout rows. It does not currently read as an independent future-price sampling bug.
- The evidence is strong enough to justify a separate replay repair task next. Sample expansion should remain later validation work, not the immediate next step.
- For `0512T004` and `0512T002`, `5-11-night-active` is the main development/diagnostic sample; `5-10-day-control-1h-06`, `5-9-noon`, and `5-9-small` are cross-sample sanity checks.

## 0515T003 Findings

- `0515T003` implemented a narrow replay lifecycle repair in `audit_replay`, without touching strategy pricing, fair/reservation, quote placement, live collection, or sample policy.
- The repair uses live terminal constraints as an upper bound on replay lifecycle:
  - if live has already terminalized an order as `canceled` / `expired` / `rejected` by the current replay decision time, replay no longer keeps that order fill-eligible
  - replay `fill` / `partial_fill` / `order_update` events that violate that live terminal boundary are rewritten into terminal lifecycle events
  - if replay emits no further lifecycle event but live has already terminalized and the order is no longer visible in live working state, a synthetic terminal event is injected so the mismatch does not merely move from `filled` to `open_or_missing`
- Same-sample regression on `5-13-day-control-30min` shows the core mismatch has been materially reduced:
  - replay filled orders: `172 -> 53`
  - replay fill-after-cancel orders: `133 -> 14`
  - replay-only fill rows: `120 -> 1`
  - live-cancel / replay-filled rows: `120 -> 1`
  - terminal-state diff rows: `230 -> 2`
- Aggregate final-state gaps are now fully aligned on the matched submit universe:
  - `canceled`: `0.09062 -> 0.0`
  - `filled`: `0.04730 -> 0.0`
  - `open_or_missing`: `0.04332 -> 0.0`
- The original hot spots are no longer structural:
  - `step_back_gt1` replay-only-fill rate: `0.05693 -> 0.0`
  - `step_back_gt1` terminal-state-diff rate: `0.11134 -> 0.0`
  - latency `q5` replay-only-fill rate: `0.05347 -> 0.0`
  - latency `q2` terminal-state-diff rate: `0.10736 -> 0.00199`
- Residual mismatch remains in only two matched submits:
  - one `live_filled_replay_canceled`
  - one `live_canceled_replay_filled`
- Stage 6B decision state moves from `diagnostic_only_gap_too_large` to `requires_more_current_format_samples`. This means the dominant replay lifecycle defect is no longer the blocker on this sample; the next decision point should treat remaining issues as residual-case diagnosis or broader-sample validation, not the original large-scale lifecycle mismatch.
- Initial residual interpretation is now split into two mechanisms, but not yet enough for another repair:
  - `live_filled_replay_canceled` looks like a short cancel-race fill miss: live filled about `9.43ms` after cancel request, replay canceled instead.
  - `live_canceled_replay_filled` looks like an optimistic touch fill false positive: replay filled about `262ms` before live cancel request, while live never filled.
- The next useful step is a small read-only residual diagnosis task that classifies whether the remaining explanation is short cancel-race window miss, touch fill optimism, or submit-after-queue exposure approximation bias. Do not open another repair task until that residual trigger evidence is written down.

## 0515T004 Findings

- `0515T004` stayed read-only and only analyzed the 2 residual matched-submit replay/live mismatches left after `0515T003`.
- The two residual cases do not support the same explanation:
  1. `3879|buy` / order `572`
     - live `cancel_request -> fill` delay is about `9.43ms`
     - live fill is preceded by dense supportive trades; `10ms` before the live fill there are `41` supportive raw trades
     - replay instead terminalizes to `cancel_ack`
     - this is strong evidence for `cancel_race_window_too_short`
  2. `28940|sell` / order `4948`
     - replay fill happens about `262.28ms` before live cancel request
     - therefore it is not a cancel-after-fill race case
     - but the current raw-trade check finds `0` supportive trades in the `10/25/50ms` windows before the replay fill
     - this means the current evidence is not strong enough to safely call it `touch_fill_assumption_too_optimistic`
- The second residual is therefore best kept as `residual_replay_fill_trigger_uncertain`, not over-claimed as a known queue/touch bug.
- The practical implication is asymmetric:
  - a very narrow follow-up repair can be justified for the short cancel-race miss class
  - a general residual replay-fill repair is not yet justified for the remaining replay-only fill case without stronger trigger evidence
- A follow-up implementation task should therefore be scoped as a narrow cancel-race residual repair only. It should not include `4948` / `residual_replay_fill_trigger_uncertain`, and it should not be written as a generalized touch/queue repair.
- `0515T005` is not a plan-only task. It is already the next narrow implementation task and should proceed only after keeping that scope restriction intact.

## 0515T005 Findings

- `0515T005` stayed within the narrow repair boundary and only addressed the `cancel_race_window_too_short` residual class.
- The implementation did not introduce a generalized touch-fill or queue-proxy repair. It only maps a replay `cancel_ack` into a replay `fill` when live has already confirmed a short-window fill-after-cancel-request event.
- Same-sample regression outcome:
  - `live_filled_replay_canceled` residual count: `1 -> 0`
  - Stage 6E residual case count: `2 -> 1`
  - remaining residual case is only `28940|sell` / `4948`
- Aggregate metrics moved only slightly and remained aligned:
  - replay filled orders: `53 -> 54`
  - replay fill-after-cancel orders: `14 -> 15`
  - final-state gaps stayed near-zero (`~0.000397` on filled/canceled rates)
- The important controller fact is that `572` has been removed without broadening the replay fill model, while `4948` remains intentionally untouched and still uncertain.
- The next useful follow-up should stay read-only and single-case: diagnose `4948` more deeply before authorizing any touch/queue repair.

## 0515T006 Findings

- `0515T006` stayed read-only and focused only on the single residual case `28940|sell` / `4948`.
- The earlier `4948` uncertainty was partly a diagnosis-limit issue: float price comparison hid same-price supportive trades at `81132.7`.
- After tick-normalized single-case analysis:
  - replay fill occurs about `262.28ms` before live cancel request
  - supportive trades before replay fill are present and dense:
    - `10ms`: `13`
    - `25ms`: `13`
    - `50ms`: `13`
    - `100ms`: `14`
  - nearest supportive trade is only about `0.503ms` before replay fill
  - replay fill happens while the order is still at touch (`ask_top1 = 81132.7`)
- This means `4948` is no longer best described as “unknown trigger”. It is better described as `queue_exposure_proxy_bias_possible`:
  - there is visible market activity that could fill a touch order
  - but live did not fill and later canceled
  - the likely gap is replay-side queue / priority approximation rather than hidden trigger absence
- Even after that narrowing, the evidence is still not enough for an immediate repair task:
  - no exact queue-position proof
  - still only one case
  - not enough basis to safely change generalized touch/queue fill behavior
- `0515T006` has now passed QA. The accepted conclusion is:
  - `4948` is best treated as `queue_exposure_proxy_bias_possible`
  - but no repair task should be opened yet unless stronger queue / repeatability evidence is added
- The next follow-up should therefore be another single-case read-only task, focused specifically on queue / priority evidence for `4948`, not on repair implementation.

## 0516T001 Findings

- `0516T001` stayed read-only and only analyzed queue / priority / exposure evidence for `28940|sell` / `4948`.
- The order remained at touch from submit to the replay fill window:
  - `order_at_touch_share_submit_to_replay_fill = 1.0`
  - order price tick `811327`, side `sell`
- Same-price supportive aggressive trades existed, but their cumulative quantity was below visible touch depth:
  - submit -> replay fill same-price trade count `31`
  - submit -> replay fill same-price trade qty `8.884`
  - submit visible ask qty at the order price `21.143`
  - replay-fill visible ask qty at the order price `15.633`
  - same-price qty / submit visible qty `0.4202`
  - same-price qty / replay-fill visible qty `0.5683`
- The practical diagnosis is now stronger than `queue_exposure_proxy_bias_possible`: `4948` is best classified as `queue_ahead_depth_can_absorb_observed_trades`.
- Interpretation:
  - replay was not filling from a hidden trigger; market trades did hit the order price
  - live could still plausibly remain unfilled because visible queue ahead was large enough to absorb the observed same-price trade quantity
  - replay likely lacks queue-ahead / priority / order-exposure state and treats touch-level supportive trades too optimistically for this case
- This is still not enough for direct repair implementation:
  - no exact queue position
  - no order-id-level queue depletion proof
  - still only a single residual case
  - a future task, if created, should be repair-design first and should not directly change generalized queue/touch fill behavior
- `0516T002` has been created as the next read-only step to test repeatability within the existing `5-13-day-control-30min` sample:
  - find touch no-fill / replay-fill / fill-candidate cases similar to `4948`
  - compute same-price trade qty / visible qty, top1 visible qty decay, unexplained depth shrink, touch duration, quote age, join age, and latency
  - decide whether `4948` is an isolated residual or part of a repeatable queue-ahead proxy mismatch pattern

## 0516T002 Findings

- `0516T002` stayed read-only and scanned the existing `5-13-day-control-30min` matched submit universe for queue-ahead proxy repeatability.
- It found `366` live no-fill / later-canceled touch candidates with same-price aggressive trades:
  - `365` are proxy-only candidates
  - `1` is a replay-fill candidate
  - all `366` satisfy queue-ahead mismatch under the current visible-queue proxy
  - `325` are strong queue-ahead mismatch candidates with high touch share
- The only replay-fill queue-ahead mismatch remains `4948`:
  - `replay_fill_queue_ahead_mismatch_cases = 1`
  - `target_4948_cases = 1`
- Interpretation:
  - queue-ahead no-fill behavior is repeatable in the sample
  - replay usually does not falsely fill those cases after the `0515T003` / `0515T005` repairs
  - `4948` remains the only replay false-positive version of that proxy pattern
- This does not justify generalized queue/touch repair yet:
  - repeatability exists for the proxy-only no-fill phenomenon
  - repeatability does not yet exist for replay-fill false positives
  - the accepted next step should be QA and then a decision on whether to create a repair-design-only task or stop at documented replay limitation
- `0518T001` has been created as a repair-design-only follow-up:
  - describe `4948` and the queue-ahead proxy evidence clearly
  - design a conservative queue proxy gate for future use
  - explicitly defer implementation until more current-format samples or more replay false-positive cases exist
  - do not modify replay, strategy, live collection, or sample policy

## 0518T001 Findings

- `0518T001` stayed repair-design-only and did not modify replay, strategy, live collection, sample policy, or generated sample artifacts.
- The accepted `4948` evidence package for future design is:
  - submit_key `28940|sell`
  - order_id `4948`
  - live canceled / replay filled
  - same-price trade qty before replay fill `8.884`
  - submit visible qty `21.143`
  - replay-fill visible qty `15.633`
  - same-price qty / visible qty ratios `0.4202` and `0.5683`
  - order-at-touch share `1.0`
- The case is best interpreted as a queue-ahead proxy problem, not an unknown trigger:
  - market trades did hit the touch price
  - visible queue proxy was still large enough to absorb observed same-price trade qty
  - replay likely overstates fillability because it lacks exact queue-ahead / priority / exposure state
- The proposed future gate should be conservative and diagnostic-first:
  - apply only to replay-fill candidates on accepted market-view rows
  - use same-price trade qty / visible qty, touch share, visible qty decay, unexplained depth shrink, quote age, join age, and stale/gap guards
  - treat `same_price_trade_qty / visible_qty < 1.0` as suspicious, with `< 0.75` as stronger evidence, but not as exact no-fill proof
  - block decisions on stale/future/missing/gap-crossed joins
- Current implementation is intentionally deferred:
  - replay-fill false-positive repeatability remains single-case (`4948`)
  - exact queue position and order-id-level depletion are still missing
  - a future implementation must first be diagnostic-only / default-off and validated on more current-format samples or more replay false-positive cases

## 0518T002 Findings

- `0518T002` stayed design-only and did not modify quote placement, fair/reservation, strategy behavior, risk guards, live scripts, replay generation, or generated sample artifacts.
- The recommended Step 5A quote-anchor design is layered:
  - primary hard anchor: fast BBO/bookTicker-equivalent source
  - secondary check/fallback: depth BBO under strict freshness and accepted market-view quality
  - research/context source: top5 reconstructed BBO, top5 imbalance, top5 microprice, and top5 liquidity proxies
- Top5 role is explicitly limited:
  - pricing input for fair-price / reservation research
  - risk/context input for liquidity, imbalance, age, and queue-ahead proxy strata
  - not the current final hard post-only quote anchor because Stage 3 top5 tick/qty evidence is research-grade, not exact L2/queue proof
- Hard protection contract:
  - bid candidates should round down/floor to tick and then clamp to `<= anchor_best_bid_tick`
  - ask candidates should round up/ceil to tick and then clamp to `>= anchor_best_ask_tick`
  - post-clamp validity must be rechecked against the anchor BBO
  - stale/missing/gap-crossed anchors should suppress fresh add-side submits or re-add churn instead of relying on exchange rejects
- GTX/post-only remains the exchange backstop, but post-only reject, API reject, throttle, drop, and fast churn should be treated as evidence buckets for stale anchor, latency, rounding, source drift, or lifecycle uncertainty.
- Step 5B must quantify BBO source drift, quote-distance buckets, crossed/post-only-risk candidates, reject/throttle/churn, stale/join-age/latency regimes, fill/markout/spread-capture tradeoffs, current enforcement gaps, and a read-only rounding/clamp counterfactual before any default-off implementation task.
- Follow-up clarification before dispatching `0518T003`: the five Step 5A constraints are not all currently enforced by code. T003 must explicitly report which constraints are already backed by code/parameters/audit fields and which remain design gaps.

## 0518T003 Findings

- `0518T003` implemented a read-only quote-anchor / post-only diagnostic runner and generated all required artifacts under `local_live_analysis/5-13-day-control-30min/stage5b_quote_anchor_diagnostic_0518T003/`.
- The run stayed read-only: no quote placement, fair/reservation, risk guard, live script, replay lifecycle, or standard schema behavior changed.
- Dataset shape:
  - decision rows `47499`
  - submit label rows `2516`
  - bookTicker anchor available rows `47499`
  - top5 anchor available rows `47499`
  - join stale decision rows `432`
  - join missing / gap-crossed rows `0 / 0`
- BBO source drift:
  - audit_depth vs bookTicker mismatch rate is large: bid `0.3524495252531632`, ask `0.3531653297964168`
  - audit_depth vs bookTicker p99 abs drift is bid `142` ticks and ask `150` ticks
  - bookTicker vs top5_depth mismatch is much smaller: bid `0.002652687424998421`, ask `0.006589612412892903`
  - Interpretation: sidecar bookTicker/top5 depth BBO are close to each other, but live audit depth view and sidecar/as-of anchor view are not row-exact enough to claim existing fast-bookTicker hard-anchor implementation.
- Rounding/clamp counterfactual:
  - current path vs audit_depth post-round risk rows `0 / 47499`
  - current path vs bookTicker post-round risk rows `3675 / 47499`, rate `0.07737004989578727`
  - current path vs top5_depth post-round risk rows `3615 / 47499`, rate `0.07610686540769279`
  - T002 design path reduces all three anchor-source post-round risk counts to `0` in the read-only counterfactual
  - Interpretation: current path is clean against its current audit_depth anchor, but switching hard anchor to bookTicker/top5 requires explicit side-conservative rounding, anchor clamp, and post-clamp re-check.
- Enforcement gap matrix:
  - currently satisfied: `top5_not_final_hard_anchor`
  - design gaps: `fast_bbo_bookticker_hard_anchor`, `depth_bbo_guarded_fallback_only`, `side_conservative_rounding_and_post_round_recheck`
  - partial coverage: `stale_latency_join_age_submit_suppression`, `reject_throttle_drop_cooldown_path`
- Reject / throttle / churn:
  - decision reject reasons are `latency_guard=16528`, `quote_throttle=5996`, `api_interval_guard=1190`, `none=23785`
  - stage5 submit post_only_risk is `0 / 2516`
  - fast_cancel_churn remains high at `1955 / 2516`
- Current decision: `0518T003` is diagnostic-only and not ready for direct implementation. After QA, any follow-up implementation should be narrow, default-off or diagnostic-first, and limited to anchor arbitration plus side-conservative rounding/clamp/re-check.
- T003 also makes the future boundary explicit: top5 should stay pricing/risk/diagnostic context, not the final hard post-only anchor, unless a later task proves the anchor arbitration layer can be implemented safely behind a narrow default-off gate.
- `0518T003` has passed QA. The accepted conclusion remains diagnostic-only: do not repair source-level row-exact drift and do not implement production/default-on quote control from this task.
- The retained Step 5C path is a protective execution-safety layer, not a source-alignment repair:
  - keep: anchor arbitration, side-conservative rounding, clamp, post-clamp re-check, guarded fallback, stale/join-age suppression, diagnostic counters
  - exclude: audit_depth/bookTicker/top5 row-exact drift repair, top5 hard-anchor promotion, fair/reservation changes, quote-placement redesign, replay lifecycle changes, live collection, live promotion, and default-on behavior
- `0518T004` is the formal task file for this retained Step 5C path and should be executed only within that boundary.

## 0518T004 Findings

- `0518T004` implemented a default-off / diagnostic-first quote-anchor safety helper in `quote_anchor_safety.py`.
- Default behavior is unchanged unless `quote_anchor_safety.enabled=true`; existing backtest tests passed after integration.
- The helper enforces the narrow Step 5C contract:
  - bookTicker-equivalent fast anchor is preferred when fresh
  - guarded depth fallback is used only when fast anchor is missing or stale
  - top5 is not used as the final hard post-only anchor
  - bid side uses floor-style conservative ticks, ask side uses ceil-style conservative ticks
  - target ticks are clamped to the selected anchor and re-checked for post-only/crossed risk
  - missing/stale anchors suppress fresh add-side submits through the default-off safety result
- Stage 5C diagnostic on `5-13-day-control-30min` produced:
  - decision rows `47499`
  - bookTicker anchor rows `39261`
  - guarded depth fallback rows `8173`
  - stale anchor rows `65`
  - missing anchor rows `0`
  - bid clamped rows `1394`
  - ask clamped rows `2281`
  - suppress buy/sell rows `65 / 65`
  - post-only risk after re-check rows `0`
- This task did not repair audit_depth/bookTicker/top5 row-exact drift, did not promote top5 to hard anchor, did not change fair/reservation, did not change replay lifecycle, did not start live, and did not enable any default-on behavior.

## Known Repository Notes

- The repository is a Rust workspace with multiple crates.
- The active user work is under `examples/binance_tick_mm/`.
- Existing planning docs under `docs/` are part of the current project state and should not be ignored.
- `docs/maker_optimization_acceptance.md` defines the current hard gates before maker optimization.
- `docs/5-8-future-plan.md` records the current Stage 6G/5-9-small baseline and next-stage context.

## Strategy Design Principles

- Maker strategy optimization is a system engineering problem. Do not chase a single extreme component, such as mandatory full L2 provenance or exact queue position, while leaving fair price, execution, risk, latency, or replay/live alignment below the acceptance line.
- Current task design should raise every layer above a usable and verifiable baseline: data view, fair price, pricing signal generation, strategy logic, risk guards, execution mechanics, and replay/live alignment.
- A weak layer can dominate the whole strategy PnL. Future tasks should identify and raise the weakest accepted layer before adding complexity to an already adequate layer.
- Top5 provenance is the current practical data boundary for near-term pricing / OFI proxy / microprice proxy work. Full L2 provenance and exact queue position are later enhancements, not current Step 2 blockers, unless top5 evidence proves insufficient.
- Queue-related work under the current boundary should be named as top-of-book/top5 size and age proxy work, not exact queue-position modeling.
- Strategy changes should remain incremental and evidence-layered: first prove data/action-path coverage, then replay-model behavior, then live-derived source-path proof before any live promotion.

## 0513T008 Findings

- `5-13-day-control-30min` was collected with T004 preflight on remote clean worktree commit `f228950`; manifest reports `dirty=false`, schema compatibility passed, start marker exists, stop marker exists, and stop marker exit code is `0`.
- `align_live_run.py` and `maker_acceptance.py` passed. Audit replay common rows: `47496`; action/planned/reject/throttle match rates are all `1.0`; working-order semantic/blocking mismatch `0/0`; strict replay lag breach/drop/fail `0/0/0`.
- T007 full-run sidecar metrics: raw messages `461402`, npz rows `2418372`, final data row mapping coverage `1.0`, depth `pu` mismatch `0`, snapshot alignment status `present`, bookTicker/depth BBO match/mismatch `67306/11`.
- T007 decision join metrics: decision rows `47499`, join coverage `1.0`, future join count `0`, missing join count `0`, stale join count `432`, top5 join age p50/p90/p99 `13.602/24.206/28.134ms`.
- Negative data-quality result: first valid depth update does not satisfy Binance snapshot/update alignment (`first_valid_update_aligned=false`), and every joined decision is marked `gap_crossed` (`47499/47499`).
- Classification: `pricing_research_candidate`, but only for compressed action-path acceptance and BBO/bookTicker/compressed-mid sanity. It is not usable yet for top5 microprice / top5 OFI proxy or queue/fill proxy research.
- Root cause for the T007 sidecar negative result is a framework bug, not run duration. Snapshot `raw_seq=6` has `lastUpdateId=10537138804218`; buffered depth `raw_seq=5` has `U=10537138802913, u=10537138805036` and covers `lastUpdateId+1=10537138804219`; depth `raw_seq=7` has `pu=10537138805036` and should chain after `raw_seq=5`. Current T007 ignores the buffered pre-snapshot update and starts bootstrap at `raw_seq=7`, which makes `first_valid_update_aligned=false` and propagates `sync_gap=true` to every joined decision.

## 0513T009 Findings

- T009 fixes the T007 sidecar bootstrap bug by buffering pre-snapshot depthUpdate messages and replaying the first buffered update satisfying `U <= lastUpdateId + 1 <= u` after the snapshot arrives.
- On `5-13-day-control-30min`, snapshot `raw_seq=6` has `lastUpdateId=10537138804218`; buffered `raw_seq=5` has `U=10537138802913`, `u=10537138805036`, and covers `10537138804219`; future `raw_seq=7` has `pu=10537138805036` and chains from buffered `raw_seq=5`.
- Fixed full-run metrics: raw messages `461402`, npz rows `2418372`, final data row mapping coverage `1.0`, depth `pu` mismatch count `0`, first valid update aligned `true`, top5 rows `67322`, and bookTicker/depth BBO match/mismatch `67307/11`.
- Fixed decision join metrics: decision rows `47499`, join coverage `1.0`, future join count `0`, missing join count `0`, stale join count `432`, gap-crossed join count `0`, and top5 join age p50/p90/p99 `13.602/24.206/28.134ms`.
- Fields now aligned enough for Step 2: `first_valid_update_aligned=true`, `depth_pu_mismatch_count=0`, `final_data_row_mapping_coverage=1.0`, `decision_join_coverage=1.0`, `future_join_count=0`, `join_missing_count=0`, and `gap_crossed_join_count=0`.
- Fields not exactly aligned: live audit top5 vs T009 sidecar reconstructed top5 per-row exact matches remain partial: best bid tick `30787/47499`, best ask tick `30780/47499`, bid full top5 ticks `8898/47499`, ask full top5 ticks `12393/47499`, bid top5 qtys `8600/47499`, ask top5 qtys `11579/47499`, bid+ask full top5 ticks `3539/47499`, and all bid/ask ticks+qtys `3324/47499`.
- `join_stale` is not zero: stale join count `432`, max top5 join age `251.033249ms`, and max bookTicker join age `603.705003ms`.
- The sample is upgraded for later top5 microprice / top5 OFI proxy / top5 imbalance pricing research candidates, subject to explicit handling of remaining bookTicker BBO mismatches and stale bookTicker-age rows.
- T009 does not start live, recollect data, modify strategy behavior, modify core/connector APIs, or change the standard hftbacktest npz main event schema. It does not prove full L2 equivalence, exact queue position, queue/fill model correctness, strategy PnL, or live promotion readiness.
- Do not keep extending Step 2 to force live audit top5 and sidecar reconstructed top5 into exact equality. The next useful work is Step 3: formalize market-view acceptance thresholds for top5 tick/qty match, BBO drift, source fields, startup exclusion, stale age, and sample classification.

## 0514T001 Findings

- `5-13-day-control-30min` is sufficient for Stage 3 verification because it has T004 manifest/action-path acceptance, T009 fixed sidecar metrics, T009 joined-decision metrics, sidecar provenance CSVs, and known partial top5 tick/qty alignment diagnostics.
- Stage 3 is implemented as an optional market-view gate in `maker_acceptance.py`; existing action/planned/reject/throttle/working-order/replay-lag hard gates remain unchanged when no sidecar metrics are provided.
- Stage 3 required gates: `first_valid_update_aligned=true`, `depth_pu_mismatch_count=0`, `final_data_row_mapping_coverage>=1.0`, `decision_join_coverage>=1.0`, `future_join_count=0`, `join_missing_count=0`, and `gap_crossed_join_count=0`.
- Stage 3 quality gates: bookTicker/depth BBO mismatch rate `<=0.001`, stale join rate `<=0.02`, top5 join age p99 `<=50ms`, best bid/ask tick match rate `>=0.80`, top5 tick match rate `>=0.80`, and top5 qty match rate `>=0.75`.
- Full-run Stage 3 result on `5-13-day-control-30min`: `passed=true`, classification `passes_pricing_research_market_view`, hard failures `[]`.
- Full-run market-view metrics: BBO mismatch rate `0.00016340354734246414`, stale join rate `0.0090949283142803`, top5 join age p99 `28.13446387999999ms`, bid tick match `0.8236483072258717`, ask tick match `0.8236904160350346`, top5 tick match `0.8232061647296615`, and top5 qty match `0.8014359103924541`.
- Stage 3 still does not prove full L2 equivalence, exact queue position, queue/fill model correctness, strategy PnL, or live promotion readiness.
- Stage 4 preconditions are now satisfied for read-only pricing-model research: the accepted primary sample is `5-13-day-control-30min`, and Stage 3 confirms it is a `passes_pricing_research_market_view` candidate. This does not authorize strategy implementation, live collection, exact queue/fill work, or production promotion.

## 0514T002 Findings

- Stage 4 should start as read-only research, not strategy implementation.
- Primary sample is `5-13-day-control-30min`; optional `5-13-day-control-15min` may be used only for compressed BBO/mid sanity, not primary top5 signal conclusions.
- Candidate signal groups: BBO/bookTicker mid, current mid, weighted mid/top1 microprice, top5 microprice, top1/top5 imbalance, top5 OFI proxy, spread/volatility buckets, lead-lag/fresh-price proxies, and stale-age/freshness buckets.
- Markout horizons should be `100ms`, `500ms`, `1s`, and `5s`; metrics should include raw future mid change, side-adjusted markout, rank/correlation, monotonic quantile buckets, top-vs-bottom quantile spread, and time-split stability.
- Required row filters: exclude or explicitly bucket future joins, missing joins, gap-crossed joins, stale joins, and startup rows; primary conclusion should be based on accepted market-view rows.
- Recommended next task: `0514T003` read-only pricing research runner implementation, outputting summary markdown, candidate metrics CSV/JSON, bucket tables, markout-by-horizon CSV, rejected-signal list, and run manifest.
- A positive Stage 4 research result should only authorize a later design/implementation task for fair/reservation adjustment; it should not directly authorize live or strategy deployment.

## 0514T003 Findings

- `0514T003` implements `examples/binance_tick_mm/pricing_research.py`, a deterministic read-only runner over existing local artifacts only.
- Primary output directory is `local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003/`.
- Full-run row counts: audit decision rows `47499`, accepted-with-stale rows `47499`, primary non-stale rows `47067`, stale rows excluded from primary `432`, future/missing/gap-crossed/startup rows `0`.
- Output artifacts: summary markdown, candidate metrics CSV/JSON, per-signal bucket tables, markout-by-horizon CSV, rejected signals CSV, and run manifest.
- Strongest primary non-stale candidate families are top5/top1 imbalance and microprice-family signals at `500ms`; top5 OFI proxy is weaker but still above the default follow-up threshold.
- Downgraded signals include duplicate reservation/audit-mid fields, weak or unstable spread/volatility/freshness/join-age fields, and weak bookTicker-mid edge under the default threshold.
- QA grouped T003 signals by importance: strongest pricing candidates are top5/top1 imbalance and microprice family; depth-size/liquidity candidates are top5 depth imbalance and top5 side quantities; existing-model/recent-move diagnostics are fair edge, recent mid move, and audit BBO mid edge; weaker follow-up signals are OFI proxies, liquidity concentration, and audit feed latency.
- QA documented duplicates: `reservation_edge_ticks` duplicates `fair_edge_ticks`, `audit_mid_edge_ticks` duplicates `audit_bbo_mid_edge_ticks`, and `book_view_stale_ms` / `latency_signal_ms` duplicate `audit_feed_latency_ms` in this sample.
- This result is research-only. It does not modify strategy behavior, configs, live scripts, core/connector APIs, or the standard npz schema, and it does not prove PnL, full L2 equivalence, exact queue/fill correctness, or live readiness.

## 0514T004 Requirements Findings

- The next research requirement is to connect T003-style pricing signals to maker execution outcomes, not just raw future-mid markout.
- Required outcome categories: fill probability, time-to-fill, adverse selection after fill, spread capture, queue/priority proxies, cancel-to-fill race, post-only/reject/throttle/churn, and inventory impact.
- Required label categories for a later implementation plan: submit-to-fill within `100ms / 500ms / 1s / 5s`, time-to-fill, fill-after side-adjusted markout, spread capture vs future mid, fill-after-cancel-request, reject/throttle/drop/churn bucket, and inventory transition.
- Additional required label categories: quote placement / distance, missed-fill / opportunity cost, realized PnL decomposition, tail risk, partial-fill / order lifecycle, inventory cycle, and sample validity / censoring.
- Later analysis must use statistics appropriate to each label type: Spearman/Pearson plus bucket monotonicity for continuous labels; event-rate/lift/odds-ratio buckets for binary labels; Kaplan-Meier/discrete-hazard or Cox-style treatment for censored time-to-event labels; exposure-normalized rate ratios for count labels; contingency/conditional-probability/mutual-information summaries for lifecycle labels; and tail quantile/CVaR-like summaries for tail labels.
- Any later implementation must stay observed-only unless a separate queue/fill calibration task is authorized. It must not claim counterfactual queue/fill proof or strategy/live readiness.

## 0514T005 Planned Findings

- `0514T005` should implement the T004 contract in a separate task, not inside T004.
- The implementation should be a deterministic read-only runner over existing `5-13-day-control-30min` artifacts.
- Required output directory: `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/`.
- Acceptance should focus on label construction coverage and dataset validation: every T004 label class must be either implemented with rows/statistics or explicitly marked `unavailable`, `low_sample`, or `observed_only_proxy` with a reason.
- The task must not modify strategy behavior, live scripts, core/connector, canonical audit schema, standard npz schema, or historical Stage 4/T009 artifacts.

## 0514T005 Findings

- `0514T005` implemented a deterministic read-only execution-outcome label runner plus focused tests, with outputs under `local_live_analysis/5-13-day-control-30min/stage5_execution_outcome_labels_0514T005/`. It did not modify strategy behavior, live scripts, core/connector, canonical audit schema, or standard npz schema.
- Coverage result on `5-13-day-control-30min`: `fill_probability`, `time_to_fill`, `adverse_selection_after_fill`, `spread_capture`, `cancel_to_fill_race`, `inventory_impact`, `quote_placement_distance`, `partial_fill_lifecycle`, `inventory_cycle`, and `sample_validity_censoring` are `available`; `queue_priority_proxy`, `post_only_reject_throttle_churn`, `missed_fill_opportunity_cost`, and `realized_pnl_decomposition` are `observed_only_proxy`; `tail_risk` is `low_sample`; no T004 label class is `unavailable`.
- Current sample shape is dominated by high cancel / low fill behavior: submit orders `2516`, filled orders `53`, canceled orders `2452`, expired `9`, open-or-missing `2`, fill-after-cancel orders `16`, fast-cancel-churn `1955`, and partial-fill orders `0`.
- Fill mass is not concentrated only in ultra-short horizons: fill-by-`100/500/1000/5000ms` is `8/22/28/40`. For this sample, optimizing only around `100ms` behavior would miss a large fraction of observed fills.
- The observed lifecycle risk is not mainly tail-only: cancel-to-fill race is material in the realized sample, while tail-risk remains low-sample because only `53` filled orders are available.
- Placement and inventory state are first-order calibration strata for later work: Stage 5 shows meaningful differences across `placement_bucket`, `distance_to_bbo_ticks`, `edge_vs_fair_ticks`, and `inventory_score`, so later replay/live calibration should compare these strata explicitly instead of only reporting aggregate gaps.
- T005 remains observed-only. Queue/priority, missed-opportunity, and realized-PnL decomposition labels are useful for ordering later work, but they do not prove exact queue position, counterfactual fill outcomes, or strategy PnL.
- This result supports refining Stage 6 into read-only replay/live fill-cancel lifecycle proxy calibration. `0514T006` should define the contract first; `0514T007` should implement the calibration runner afterward.

## 0514T006 Findings

- Stage 6 should be framed as replay/live fill-cancel lifecycle proxy calibration, not broad exact-queue calibration. The current evidence base is rich enough for lifecycle comparison but still observed-only for queue priority, missed opportunity, and realized-PnL decomposition.
- The correct Stage 6 comparison unit is not raw cross-domain `order_id`. Live/replay alignment should be built around matched normalized submit opportunities, with submit-key coverage reported explicitly before interpreting lifecycle gaps.
- The required common Stage 6 label schema should reuse the Stage 5 core execution outcomes: fill-by-horizon, time-to-fill, final order state, fill-after-cancel-request, cancel-to-fill delay, fast-cancel-churn, fill markout / spread-retention, and coverage/censoring flags.
- Stage 6 results must be reported both in aggregate and across key strata: `placement_bucket`, `distance_to_bbo_ticks`, `edge_vs_fair_ticks`, `inventory_score`, top-of-book/top5 size-age proxy, and latency regime. Aggregate-only reporting would hide the main replay/live risk concentrations.
- `5-13-day-control-30min` is enough for Stage 6B runner implementation and single-sample methodology validation because it already has Stage 3 acceptance, T009 sidecar alignment, and T005 labels.
- `5-13-day-control-30min` is not enough alone to authorize quote-adjustment promotion: only `53` fills are observed, `partial_fill=0`, and tail-risk remains low-sample. Later promotion-style decisions need additional current-format samples with the same artifact chain.
- `0514T007` should stay strictly read-only. It may classify the result as `methodology_valid_single_sample`, `diagnostic_only_gap_too_large`, or `requires_more_current_format_samples`, but it must not claim exact queue proof, counterfactual fill proof, or live readiness.

## 0514T007 Findings

- `0514T007` implements a read-only replay/live execution outcome calibration runner and uses matched normalized submit opportunities as the primary comparison unit.
- On `5-13-day-control-30min`, submit-key coverage aligns perfectly: live submit orders `2516`, replay submit orders `2516`, matched submit orders `2516`, and matched price tick / qty equality are both `2516/2516`.
- Fill-horizon rates are relatively close on the matched submit universe: absolute gaps are about `0.0012` at `100ms`, `0.0012` at `500ms`, `0.0020` at `1000ms`, and `0.0119` at `5000ms`.
- Fast-cancel-churn is aligned (`0.7770` vs `0.7770`), so the large replay/live difference is not a generic quote-churn mismatch.
- The main replay/live gaps are in lifecycle outcomes rather than submit coverage:
  - replay filled orders `172` vs live `53`
  - replay fill-after-cancel orders `133` vs live `16`
  - final state gaps: `canceled` about `0.0906`, `filled` about `0.0473`, `open_or_missing` about `0.0433`
  - fill-after-cancel-request rate gap about `0.0465`
  - cancel-to-fill delay gap is also large
- Markout observability coverage is materially different between replay and live, even when submit matching is perfect. This means Stage 6 should keep coverage-gap reporting separate from lifecycle-gap reporting.
- Strata output confirms that important gaps concentrate in placement / inventory / latency buckets, especially deeper step-back placements, higher inventory-score buckets, larger same-side size buckets, and some higher join-age / latency buckets.
- Pre-repair decision state was `diagnostic_only_gap_too_large`: the Stage 6B methodology worked on a single accepted sample, but replay lifecycle still deviated too much from live to treat replay fill-side behavior as close enough for promotion-style quote-adjustment experiments. This historical finding was later superseded by the accepted `0515T003` / `0515T005` repairs and the `0519T001` final rerun.

## 0510T001 Findings

- `5-10-day-control-1h-06` is present under `local_live_analysis/` and has an archive tarball plus sha256.
- `maker_acceptance.py` passed on audit replay with `action/planned/reject/throttle = 1.0`, working semantic/blocking mismatch `0/0`, API/throttle mismatch `0`, and strict replay lag breach/drop/fail `0/0/0`.
- Cancel-requested fills are non-trivial: `21 / 46` fills, notional rate about `0.4564`.
- Risk source is more consistent with adverse-selection / inventory-reducing cancel race than same-side readd: adverse-selection candidate count `14`, add-side candidate count `7`, same-side readd then cancel-fill count `1`.
- Single-sample Stage 6J replay completed with 6 candidates and no hard failures, but decision is `diagnostic_only_no_promotion`.
- Next useful task should add cross-sample validation before any live micro test.

## 0510T002 Findings

- Added `.workflow/runners/run_task.py` and `.workflow/runners/run_0510T002.py`.
- Runner automatically generated missing `maker_acceptance.json` for `5-8-stage3-15m-livetest-v4`.
- Cross-sample Stage 6J replay ran on 4 samples:
  - `5-10-day-control-1h-06`
  - `5-9-small`
  - `5-9-noon`
  - `5-8-stage3-15m-livetest-v4`
- Result: `diagnostic_only_no_promotion`.
- Hard failures: `0`.
- Baseline across 4 runs: pnl sum `-0.8641`, cancel-fill count `3`, same-side worsening `3`.
- Add-side guard candidates: pnl sum `-0.2329`, cancel-fill count `0`, same-side worsening `0`.
- Broad add-side cooldown 200ms control: pnl sum `-0.0535`, cancel-fill count `1`, same-side worsening `1`.
- Do not auto-promote to live. QA and total controller should review whether this warrants a separate adverse-selection timing rule design task.

## 0511T001 / 0511T002 Split Findings

- Current live/backtest structure already shares `strategy_core.py` for action decisions, cancel-race guard helpers, lifecycle tracking, and audit row construction.
- Signal-to-target generation and guard-context assembly are still duplicated in live/backtest loops; this is acceptable for a narrow guard task but should not expand further without a follow-up refactor.
- The existing `cancel_race_guard` blocks add-side exposure while allowing reduce-side actions; adverse-selection timing should preserve this same safety shape.
- Future markout is an evaluation metric only. It must not become a live rule input.
- First adverse-selection timing candidate should target decision-time-visible toxic timing, such as cancel-requested fill timing or target deterioration, and should be validated through Stage 6J source-path metrics rather than PnL alone.
- `0511T001` is design-only and should not modify strategy code.
- `0511T002` is the first task allowed to modify strategy code for this rule, after `0511T001` is accepted.
- `0511T001` completed the design contract and recommends entering `0511T002` only for default-off implementation, unit tests, and offline replay. It does not authorize live micro test or default rule enablement.
- First implementation should use decision-time-visible inputs only: working/target ticks, position, in-flight/cancel-requested state, last cancel-fill timestamps, latency signals, and rolling historical state. Future mid/markout and audit overlays remain forbidden as live decision inputs.
- `0511T002` implemented a default-off adverse timing guard with shared live/backtest helper and audit fields, but cross-sample replay still decided `diagnostic_only_no_promotion`.
- In `0511T002`, pure `adverse_timing_target_deterioration_50/100/200ms` matched baseline across 4 samples: cancel-fill `3`, same-side worsening `3`, pnl sum `-0.8641`.
- In `0511T002`, `add_side_guard_only` and `add_side_guard_plus_adverse_timing_100ms` both reduced cross-sample cancel-fill to `0` and same-side worsening to `0`, with pnl sum `-0.2329`; adverse timing added no incremental replay benefit over add-side guard.
- Do not live-promote the adverse timing candidate from `0511T002`; future work should redesign the toxicity trigger or improve diagnostic replay sensitivity before any live micro test.
- Before redesigning the timing rule, the next diagnostic must distinguish whether `target_deterioration` did not trigger, triggered away from add-side decisions, or exposed a Stage 6J replay/source-path sensitivity gap.

## 0511T003 Dashboard Findings

- The current dashboard is useful as a task status index but weak as a decision board.
- For this project, the dashboard should expose task results such as Stage 6J decision, sample count, candidate count, hard failures, cancel-fill risk, QA conclusion, and live/no-live decision.
- The first dashboard upgrade should remain static and markdown-driven; interactive editing and task execution should stay out of scope.
- `0511T003` upgraded the dashboard into a static decision board with task status, decision summary, key metric chips, business results, QA conclusions, and next steps.
- Metrics are currently extracted from free-text business/QA reports with heuristics. Future reports should add explicit `metrics` or `decision_summary` fields to reduce ambiguity.

## 0511T004 Findings

- `target_deterioration` was not missing. In `0511T002` replay audits, each pure adverse timing candidate produced `buy_active=118429` and `sell_active=160062` across 4 samples.
- Pure adverse timing active rows did not overlap actual add-side submissions: `submit_buy` overlap `0`, `submit_sell` overlap `0`.
- Baseline vs `adverse_timing_target_deterioration_50/100/200ms` had zero row-level differences across action, planned_action, position, target ticks, and working ticks for all 4 samples.
- The first `target_deterioration` trigger is state-misaligned for the intended risk: it detects stale existing working quotes, while many target add-side submit/re-add moments happen when that side has no working order.
- Stage 6J replay currently under-observes the live adverse-selection source-path: live/current-format risk diagnostics show positive adverse-selection candidate counts, but Stage 6J replay summary reports `guard_candidate_adverse_selection_count=0` for baseline and adverse timing candidates.
- Do not implement or live-test another adverse timing rule until the next contract defines add-side submit/re-add coverage and replay/source-path observability.

## 0512 Planned Findings

- The next step is not to tune `target_deterioration`; it is to reconcile replay/source-path observability first.
- `0512T001` must determine whether the replay adverse-selection gap comes from missing lifecycle fields, fill model differences, field semantics, or source-path classification not fitting replay audits.
- `0512T004` must turn the `0512T001` diagnosis into an explicit observability gate before `0512T002` starts.
- `0512T002` must design a new add-side submit/re-add toxic timing rule that proves coverage of actual `submit_buy` / `submit_sell` rows, while treating Stage 6J replay as regression evidence rather than live source-path proof.
- The 4H `5-11-night-active` sample should be used for main diagnosis because it has much more event mass, but success on that sample alone must not authorize live promotion.

## 0512T001 Findings

- The Stage 6J adverse-selection source-path gap is mainly a replay lifecycle / fill-model gap, not a different source-path classifier. Stage 6J calls `run_backtest(...)` to generate a new simulated audit and then calls the same `analyze_audit_csv(...)` classifier.
- Live/current-format risk diagnostics show positive inventory-reducing cancel race counts, but Stage 6J baseline replay generates only `0/1/1/1` cancel-after-request fills across the four samples and zero adverse-selection candidate count.
- Sampled live adverse-selection events do not have matching order ids or same side/price fills in Stage 6J baseline replay audit.
- Stage 6J can compare candidates under its replay fill model, but cannot by itself prove live adverse-selection source-path improvement.
- `0512T002` may proceed as a design-only task, but its contract must use action-path coverage and explicit observability limits; no implementation or live test is authorized by `0512T001`.

## 0512T004 Findings

- `0512T004` defines the required evidence split for later rule work: action-path coverage, replay-model regression, and live-derived source-path proof.
- Action-path coverage must prove that a new guard actually reaches add-side submit/re-add rows: eligible add-side submit count, blocked add-side submit count, blocked reason, row-level action/planned_action diff, reduce-side allowed rows, and blocked reduce-side count.
- Stage 6J replay is a replay-model regression gate only: PnL, position, drop/churn, action churn, replay fill/cancel-fill, and replay source-path metrics can compare candidates within the same simulated lifecycle, but cannot alone prove live inventory-reducing cancel race improvement.
- Live-derived source-path proof must come from current-format live audit analysis: fill-after-cancel-request events, inventory-reducing cancel race, adverse-selection candidate count, add-side candidate count, cancel-to-fill latency, and side-adjusted markout.
- `5-11-night-active` is the main development/diagnostic sample; `5-10-day-control-1h-06`, `5-9-noon`, and `5-9-small` are cross-sample sanity checks. A single-sample win on `5-11-night-active` cannot authorize live promotion.
- `0512T002` may start after `0512T004` QA only as a design-contract task. It must not claim live source-path improvement, authorize implementation, or authorize live micro test.

## 0512T002 Findings

- `0512T002` defines a new default-off `add_side_toxic_timing_guard` design contract for add-side submit/re-add path coverage.
- The old `target_deterioration` trigger is no longer the main trigger. It may only be an auxiliary signal when evaluated from submit/re-add eligibility; threshold tuning of the old working-quote-only trigger is explicitly rejected.
- The new rule must first compute add-side submit eligibility, then suppress only the submit leg when a same-side toxic timing signal is active. Reduce-side submit must remain allowed and must be proven by audit with blocked reduce-side count `0`.
- Required audit evidence includes eligible add-side submit count, blocked add-side submit count, blocked reason, target move since last quote/cancel, last cancel request/fill ages, guard-until timestamps, and reduce-side allowed fields.
- Stage 6J replay remains replay-model regression only. It must report action-path coverage and regression metrics across `5-11-night-active` plus sanity checks `5-10-day-control-1h-06`, `5-9-noon`, and `5-9-small`.
- Live-derived source-path proof remains outside `0512T002`: historical live counterfactual overlay can show would-block overlap, but true source-path improvement requires a later QA-approved post-rule live micro test and current-format live audit risk analysis.
- `0512T002` does not authorize strategy implementation, default enablement, or live micro test. After QA, it may authorize creation of a separate implementation + offline replay task.

## 0512T005 Planned Findings

- `0512T005` is the implementation/offline replay task created from `0512T002` core conclusions.
- It must implement default-off `add_side_toxic_timing_guard`, shared live/backtest helper, config fields, audit fields, unit tests, Stage 6J replay candidates, and action-path coverage reporting.
- It must prove add-side submit/re-add coverage with eligible rows, blocked rows, submit overlap, action/planned_action diff, and blocked reduce-side rows equal to `0`.
- It must use `5-11-night-active` as the main development/diagnostic sample and `5-10-day-control-1h-06`, `5-9-noon`, `5-9-small` as cross-sample sanity checks.
- It does not authorize default enablement, live micro test, or claims of live-derived source-path improvement.

## 0512T005 Findings

- Commit `34c954e` implements default-off `add_side_toxic_timing_guard` with shared strategy-core helper, live/backtest config/state integration, audit fields, Stage 6J candidate matrix, and focused tests.
- The rule blocks only add-side submit/re-add legs. Stage 6J cross-sample coverage total: pure toxic timing candidates blocked `594` add-side submits, blocked reduce-side total `0`, baseline action/planned diff `199`, blocked submit overlap `56`, submit removed `56`.
- Per-sample `add_side_toxic_timing_100ms` coverage: `5-11-night-active` blocked `299` rows but no baseline submit removal under replay; `5-10-day-control-1h-06` blocked `6`; `5-9-noon` blocked `46` with `2` submit removals; `5-9-small` blocked `243` with `54` submit removals.
- Stage 6J replay completed 4 samples x 7 candidates with hard failures `0`, overlays `off/off/off`, decision `diagnostic_only_no_promotion`.
- Replay-model result: pure toxic timing 50/100/200ms has action-path coverage but does not improve replay cancel-fill or same-side worsening versus baseline (`cancel-fill 4`, `same-side worsening 3` for both). `add_side_guard_only` and combined guard reduce replay same-side worsening to `0`, but combined toxic timing adds no replay benefit over add-side guard.
- Live-derived source-path proof remains missing. Stage 6J remains a replay-model regression gate only and cannot prove live inventory-reducing cancel race improvement.
- No default enablement and no live micro test are allowed from `0512T005`.

## Signal Design Lessons

- Design trading risk signals around the action path they are meant to change, not only around a plausible market-state condition. The 0511 `target_deterioration` signal was plausible and fired often, but it observed existing working quotes and had zero `submit_buy` / `submit_sell` overlap; it could not change the intended add-side submit/re-add behavior.
- For submit/re-add risk, compute action eligibility first, then evaluate toxicity and suppress only the intended submit leg. T005 gained coverage because `buy_submit_eligible` / `sell_submit_eligible` became explicit inputs to the guard, so the signal was evaluated exactly where add-side submit decisions were made.
- Separate signal domain from signal strength. In T005, recent same-side cancel request / cancel-fill / cancel-requested inflight exposure defines the timing domain, while target move / latency / recent fill defines toxicity. A strong toxicity proxy is still ineffective if it is attached to the wrong decision domain.
- Preserve the intended safety shape in the action layer: allow stale quote cancel and reduce-side submit, but suppress toxic add-side re-add submit. This made blocked reduce-side rows stay at `0` while still producing blocked add-side submit rows.
- Future signal tasks should require action-path coverage evidence before discussing replay risk improvement: eligible rows, blocked rows, blocked reason, baseline action/planned diff, blocked submit overlap, submit removed, and blocked reduce-side rows. Only after those pass should replay-model regression and live-derived source-path proof be interpreted.

## 0512T006 Planned Findings

- T006 is a planning-only follow-up to explain why T005 pure toxic timing candidates had action-path coverage but no Stage 6J replay risk improvement.
- T006 must plan three analyses only: blocked-row stratification, actual submit-removal row analysis, and blocked reason attribution.
- T006 must not design stricter candidates. Candidate redesign belongs to a later task only after the attribution root cause is known.
- T006 must not modify code, run new Stage 6J replay, start live, or claim live-derived source-path proof.
- T007 is reserved for the actual read-only attribution / experiment implementation after T006 defines the inputs, outputs, and decision criteria.

## 0512T006 Findings

- T006 produced a planning-only attribution contract for T007 and did not modify strategy code, run new Stage 6J replay, create stricter candidates, or start live.
- The planned T007 blocked-row stratification must separate true submit removal from eligibility-only coverage, especially `5-11-night-active` where pure toxic timing blocked `299` rows but removed `0` baseline submits.
- The planned T007 submit-removal analysis must focus on the `56` true submit removals and link them to later replay cancel-fill / same-side worsening / source-path events before interpreting replay risk improvement.
- The planned T007 reason attribution must split pending cancel, recent cancel request, recent cancel fill, target move, and latency contributions, and explain why 50/100/200ms pure toxic timing candidates were identical.
- T006 passed QA and now authorizes creation of T007 as a read-only attribution implementation only. It does not authorize rule redesign or live micro test.

## 0512T007 Findings

- T007 explains why T005 pure toxic timing had action-path coverage but no replay risk improvement: most blocked rows did not remove a baseline submit, and the submits that were removed did not overlap the replay cancel-fill risk orders.
- For representative `add_side_toxic_timing_100ms`, blocked rows were `594`, but only `56` row-level submits were actually removed (`9.43%`), representing `42` unique submit-order keys after de-duplication.
- `5-11-night-active` is the clearest case: blocked rows `299`, true submit removed `0`, so coverage was eligibility-only on the main sample.
- True submit removals had `0 / 56` overlap with baseline Stage 6J cancel-fill risk event order ids. Baseline and candidate 100ms risk order ids stayed the same: `5-11-night-active` `2|1`, `5-9-noon` `3`, `5-9-small` `13`.
- 50/100/200ms pure toxic candidates were equivalent because blocked rows were dominated by `pending_cancel+target_move`; the window parameter did not change the blocked set in these replay samples.
- T007 remains replay-model attribution only. It does not provide live-derived source-path proof and does not authorize live micro test.

### 0512T007 Excluded Directions

- Excluded: `pure toxic timing has no action-path coverage`. It does have coverage: representative 100ms blocked rows `594`.
- Excluded: `reduce-side was harmed or the result came from reduce-side suppression`. Blocked reduce-side count remained `0`.
- Excluded: `50/100/200ms window tuning is likely to change this result`. The three windows had `100%` unique blocked-key equivalence across all four replay samples.
- Cause of window equivalence: blocked reason was dominated by `pending_cancel+target_move`; in these replay samples, the window-ms parameter did not become the active differentiator.
- Excluded: `the main sample lacked trigger mass`. `5-11-night-active` had `299` blocked rows, but `0` true submit removals.
- Excluded: `pure toxic timing filtered the Stage 6J replay adverse-selection / cancel-fill risk orders`. True submit removals had `0 / 56` overlap with baseline cancel-fill risk event order ids, and baseline/candidate 100ms risk order ids were unchanged.
- Excluded: `more blocked rows alone should improve replay risk`. The problem was not blocked-row count; `538 / 594` 100ms blocked rows did not actually remove a baseline submit.
- Excluded: `T007 can be used as live source-path improvement proof`. T007 is replay-model attribution only and does not supply live-derived source-path proof.
- Excluded design direction: continuing the same `pending_cancel+target_move` pure toxic timing rule with only window-ms tuning. Future work must redefine what makes a submit a risk-source submit before proposing a new rule.

## 0512T008 Findings

- `strategy_core.decide_actions()` does not receive full depth directly. The live/backtest outer loops read `hbt.depth(0)`, compress it into best bid/ask, mid, bid/ask size, top5 strings, fair/reservation/half_spread, and target ticks, then pass only target/action state into the shared action core.
- Shared `hbt.depth(0)` API does not imply identical live/replay view. Live reads connector-maintained local book at decision time; Stage 6J no-overlay reads replay-reconstructed depth.
- `market_state_overlay=audit` forces live audit compressed market/fair/target fields into audit replay, but it does not overlay top5 tick/qty strings. Top5 strings remain replay-depth derived and can still mismatch.
- On `5-11-night-active`, audit replay overlay had compressed market mismatch `0` and target tick mismatch `0`, but top5 mismatch `11.0159%`.
- On `5-11-night-active`, Stage 6J no-overlay with lag<=250ms had best bid mismatch `9.5306%`, best ask mismatch `9.5246%`, target bid tick mismatch `10.2907%`, and target ask tick mismatch `10.0460%` versus live audit at matched decision timestamps.
- Sanity samples also show Stage 6J no-overlay live/replay view differences: best bid/ask mismatch about `2.5%-4.2%`, and target tick mismatch about `3.1%-5.1%`.
- Raw gzip contains Binance depth sequence data and bookTicker events. Across four samples, depthUpdate `pu` mismatch count was `0`; `5-9-noon` raw gzip has an EOF trailer issue that was recorded as data-quality metadata.
- Converted npz schema is `ev|exch_ts|local_ts|px|qty|order_id|ival|fval` and does not retain Binance `U/u/pu` or `lastUpdateId`.
- Current audit/action alignment is sufficient for the existing simple compressed strategy view, but it is not sufficient to prove full L2 / queue / OFI / microprice equivalence.
- Before microprice / OFI / queue research or strategy changes, extend data-layer/audit schema with top-N book, update ids, exchange/local timestamps, bookTicker-vs-depth consistency, and per-decision book provenance.
- After the date change, this follow-up is `0513T001` for planning only and a later `0513T002` for implementation if QA accepts the plan.

## 0513T001 Findings

- Do not modify `hbt.depth(0)` core API first. The immediate problem is transparency at the Binance maker strategy layer, not proof that the core depth API itself is wrong.
- The next implementation should introduce an explicit strategy-layer `MarketView` / `BookViewSnapshot` wrapper and a shared `build_market_view_from_depth(...)` helper used by both live and backtest loops.
- `strategy_core.decide_actions()` should continue receiving compressed action inputs rather than full depth. The transparency layer should live before fair/target/action construction.
- Audit rows should carry provenance fields such as `market_view_source`, `top5_source`, `market_overlay_source`, `top5_overlay_source`, `book_view_ts_local`, `book_view_ts_exch`, `book_view_feed_latency_ns`, and `book_view_stale_ms`.
- top5 alignment is necessary for current decision-view transparency because current fair uses top5-derived `bid_size - ask_size`, but top5 alignment is not full L2 / queue / OFI proof.
- Overlay top5, if implemented, must be labeled as `audit_overlay` and treated only as audit replay decision-view alignment. It must not be used to claim replay reconstructed book alignment.
- If the Python strategy layer cannot expose Binance update ids, `lastUpdateId`, bookTicker provenance, or decision-row book provenance, create a later core/data task instead of fabricating fields.
- `0513T001` does not implement code. If QA passes, create a separate `0513T002` implementation task for MarketView provenance / top5 audit transparency minimal implementation.
- `0513T002` must keep a hard file boundary: only `strategy_core.py`, `live_tick_mm.py`, `backtest_tick_mm.py`, and `test_backtest_tick_mm.py` are allowed implementation files. Core API, converter/npz schema, connector book management, strategy rules, fair price formula, risk guards, configs, live scripts, and microprice/OFI/queue studies require separate tasks.

## 0513T002 Findings

- `0513T002` implemented a strategy-layer `MarketView` wrapper and `build_market_view_from_depth(...)` helper inside `strategy_core.py`.
- live and backtest loops now build the decision market view through the shared helper, then assign the same best bid/ask, mid, spread, top5, and top5-size values as before.
- Audit rows now include provenance fields: `market_view_source`, `top5_source`, `market_overlay_source`, `top5_overlay_source`, `book_view_ts_local`, `book_view_ts_exch`, `book_view_feed_latency_ns`, `book_view_stale_ms`, `top5_depth_best_bid_tick`, and `top5_depth_best_ask_tick`.
- In audit replay overlay mode, compressed market state is labeled `market_view_source=audit_overlay` and `market_overlay_source=audit`, while top5 remains labeled from replay depth. This preserves T001 option B and avoids pretending replay reconstructed depth has been fixed.
- `0513T002` did not implement Binance `U/u/pu`, `lastUpdateId`, or bookTicker provenance because these are not reliably exposed at the current Python strategy layer. Those require a separate core/data task if needed.
- Focused tests passed for helper output, audit schema fields, build-audit-row provenance, and market-state overlay related tests.
- No live run, no new Stage 6J replay/sweep, no core API change, no converter/npz change, no config change, and no strategy-rule change were performed.

## 0513T004 Findings

- `0513T004` implements a live startup preflight gate in `examples/binance_tick_mm/deploy/preflight_live_run.py` and wires it into `deploy/run_live.sh` before tmux/collector/connector/live bot startup.
- The preflight manifest records git commit, branch, dirty status, config hash, connector config hash, `audit_schema.py` hash, `strategy_core.py` hash, `live_tick_mm.py` hash, `run_live.sh` hash, preflight script hash, symbol, data dir, run dir, configured output paths, Python environment, and start/stop marker paths.
- The compatibility check imports the current `strategy_core`, builds representative decision and lifecycle audit rows, and requires their keys to exactly match the final `AUDIT_FIELDS`. Duplicate fields, missing fields, or extra row keys fail preflight before live starts.
- This directly addresses the `0513T003` failure mode where remote `audit_schema.py` lagged strategy code and live failed later at CSV writer time.
- `run_live.sh` now writes `deployment_manifest.json` and `start_marker.json` before startup, and configures the live bot pane to write `stop_marker.json` on process exit.
- Focused tests passed for current schema compatibility, missing/extra field detection, duplicate field detection, manifest/start marker output, and stop marker output.
- T004 is a deployment reproducibility gate only. It does not start live, modify AWS, change strategy behavior, prove PnL, or prove full L2 / market-view alignment.

## 0513T005 Planned Findings

- Step 2 must first classify sample usability before any pricing, queue, OFI, or microprice research.
- Existing `align_live_run.py` / `compare_audit.py` artifacts already expose useful latency, replay lag, action-path, and top5 mismatch diagnostics, but they do not preserve full Binance raw provenance in the converted npz.
- Current converted npz schema does not retain `U/u/pu`, `lastUpdateId`, or bookTicker provenance. Raw update-id continuity and bookTicker/depth consistency must be computed from raw gzip or moved into a later core/data task.
- Existing samples are useful for Step 2 diagnostics, but most are pre-T004 legacy samples. They cannot prove the new standard run-live deployment flow unless a later fresh T004-standard no-rule sample is collected.
- The Step 2 implementation should output sample-level classification:
  - `compressed_action_path_only`
  - `pricing_research_candidate`
  - `queue_fill_research_candidate`
  - `unusable`
- The likely near-term outcome is that existing samples support compressed action-path diagnostics and some pricing-input sanity checks, but queue/fill research requires stronger raw/top-N/update-id provenance.

## 0513T006 Findings

- T006 generated all required Step 2 read-only artifacts under `local_live_analysis/step2_market_data_baseline_0513T006/`.
- Classification result:
  - `5-13-day-control-15min`: `pricing_research_candidate`, but only for limited live-audit compressed BBO/mid sanity checks.
  - `5-11-night-active`: `compressed_action_path_only`.
  - `5-10-day-control-1h-06`: `compressed_action_path_only`.
  - `5-9-noon`: `compressed_action_path_only`.
  - `5-9-small`: `compressed_action_path_only`.
- No current sample qualifies as `queue_fill_research_candidate`.
- All five samples are legacy/pre-T004 from a deployment reproducibility perspective because they lack `deployment_manifest.json`.
- Only `5-13-day-control-15min` has T002 MarketView provenance fields in live audit; older samples lack strategy-layer source fields.
- Existing action-path acceptance remains useful for the current compressed strategy view, but it is not full L2 / queue / OFI / microprice proof.
- Top5 mismatch remains material across all samples: tick match about `0.9044` to `0.9747`, qty match about `0.8898` to `0.9558`.
- All samples show degraded order-entry tail latency in current artifacts: entry p99 about `4055ms` to `15112ms`.
- Raw gzip depth `pu` continuity was diagnosable in this bounded pass and showed `0` mismatches for the five samples.
- bookTicker/depth consistency in T006 is bounded best-effort only, with `20000` bookTicker checks per sample; it is not production-grade local book proof.
- Converted npz still does not retain Binance `U/u/pu`, snapshot `lastUpdateId`, bookTicker provenance, or per-decision top5 book provenance. The current near-term research boundary is top5-only; full L2 provenance and exact queue-position work are not required for T007.
- T006 did not start live, collect data, run replay/sweep, change strategy/deploy/core/converter code, or authorize live promotion.

## 0513T007 Planned Findings

- The Binance provenance fix should not change the standard hftbacktest `data` npz main event schema. That array should remain the replay-compatible market event stream.
- Raw Binance message provenance belongs in sidecars because `U/u/pu`, snapshot `lastUpdateId`, and bookTicker `u` are message/local-book metadata, not natural price-level event fields.
- Avoid stuffing Binance provenance into reserved `ival/fval` fields. That would be opaque, duplicate message-level data across split price-level rows, and risk corrupting event semantics.
- The sidecar design must make synchronization explicit, not implicit:
  - `raw_seq -> npz row range`
  - `raw_seq -> reconstructed book/top5 row`
  - `reconstructed book/top5 row -> decision row`
- Raw-to-npz mapping must reference final standard `data` array rows after `correct_local_timestamp()` and `correct_event_order()`, because converter ordering can split one input event into exchange/local rows.
- Decision joins must be as-of joins using historical top5 rows only. They must report join key, joined `raw_seq`, joined depth `u`, joined bookTicker `u`, split join age ms, stale status, gap-crossed status, and future join count `0`.
- bookTicker and depth are separate Binance streams; mismatch can come from stream timing and must be measured with age/mismatch buckets before being interpreted.
- Top5 sidecars should include schema/version/manifest metadata: schema version, raw file identity, converter opt, top5 levels, tick_size source, generation time, and input sample id.
- T007 should not modify canonical `audit_schema.py` or live audit CSV schema. Joined decision output should be an independent diagnostic artifact until a later task promotes fields into the formal audit schema.
- T007 is top5-only. It can support later top5 pricing / top5 OFI proxy / top5 microprice proxy research if join-age and mismatch metrics pass, but it cannot prove full L2 equivalence or exact queue position.
- Queue work under this boundary means top-of-book/top5 size and age proxies only.
- Long term, if the strategy must consume update ids/top5 provenance in real time, that should be a separate connector/core API task. T007 is a data-quality and read-only audit/provenance task.

## 0513T007 Findings

- T007 implemented `examples/binance_tick_mm/binance_top5_provenance.py` as a standalone top5 provenance / decision join tool.
- The tool keeps the standard hftbacktest `data` npz main event schema unchanged and writes Binance-specific provenance to sidecars instead.
- Sidecar outputs:
  - `raw_provenance.csv`
  - `raw_to_npz_mapping.csv`
  - `top5_sidecar.csv`
  - `joined_decisions.csv`
  - `sidecar_manifest.json`
  - `metrics.json`
- `raw_to_npz_mapping.csv` maps `raw_seq` to final standard `data` row indices after local timestamp correction and event-order correction.
- Decision joins are as-of joins only; T007 reports `future_join_count=0`.
- Smoke over the first `5000` messages of `5-13-day-control-15min` produced final data row mapping coverage `1.0`, depth `pu` mismatch `0`, and bookTicker/depth BBO match/mismatch `151/1`.
- The same smoke intentionally exposed poor sample-slice usability: `first_valid_update_aligned=false`, stale joins `243/244`, and gap-crossed joins `244/244`. This is expected for a bounded slice and shows the tool reports unusable sync/join states instead of fabricating clean top5 proof.
- T007 does not modify `align_live_run.py`, canonical `audit_schema.py`, live audit CSV schema, strategy behavior, core event schema, py `event_dtype`, connector local book management, or live deployment scripts.
- T007 does not start live, run strategy replay/sweep, prove full L2 equivalence, prove exact queue position, prove strategy PnL, or authorize live promotion.

## 0512T003 Findings

- `5-11-night-active` is a current-format 4H night-active sample with live rows `1,018,503`, decision rows `802,999`, and local archive `local_live_analysis/archive/5-11-night-active.tar.gz`.
- Analysis used the local repository package via conda env `hftbacktest` and `PYTHONPATH=py-hftbacktest:examples/binance_tick_mm`, not the generic site-packages path.
- Maker acceptance passed on audit replay: action/planned/reject/throttle all `1.0`, working semantic/blocking mismatch `0/0`, API/throttle mismatch `0`, strict lag gate breach/drop/fail `0/0/0`, and post-startup outside dual gate rows `0`.
- Audit replay consumed/scheduled `802,996 / 802,999`; the remaining `3` unconsumed rows are tail rows and did not fail the strict lag gate.
- Audit replay PnL for this sample was `-7.6108`, max abs position notional `246.29655`, below the current `250` cap.
- Cancel-requested fill risk repeated strongly in the longer night-active sample: `391 / 915` fills after cancel request, notional rate `0.427300`, same-side readd while cancel-requested `1436`, same-side readd then cancel-fill `13`, worsening cancel-fill `191`.
- Source-path split on `5-11-night-active`: add-side candidates `201`, adverse-selection / inventory-reducing candidates `190`.
- Cross-sample current-format risk summary over `5-9-noon`, `5-10-day-control-1h-06`, and `5-11-night-active` decided `proceed_to_stage6j_narrow_rule`.
- This sample supports narrow cancel-requested fill-risk rule design, but does not authorize direct maker parameter optimization or live micro test. The Stage 6J source-path observability limitation from `0512T001` still applies.
