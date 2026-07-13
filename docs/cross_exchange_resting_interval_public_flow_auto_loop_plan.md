# Cross-Exchange Resting-Interval Public-Flow Auto-Loop Plan

## Purpose

This is a controller-level auto-loop plan. It does not itself create formal `.workflow/tasks/` files, does not dispatch execution, and does not authorize live-submit.

The goal is to finish the narrow route opened by `0710T001` and refined by `0712T001`:

```text
0712T001 QA
  -> resting-interval capture instrumentation
  -> controlled same-envelope live evidence with the new artifacts
  -> quote/fill probability evidence rerun
```

The purpose is not to improve fill rate yet. The purpose is to make the evidence chain able to answer:

```text
During the actual order resting interval, what public flow, visible depth, and depletion/trade-through evidence existed?
```

No threshold change, quote-envelope change, order-size increase, max-submission increase, fee/PnL calibration, maker-viability claim, T012, promotion, or final MVP pass is allowed inside this plan.

## Current Input State

Latest accepted QA:

- `0713T001` QA: `已通过`
- accepted route: `stop_at_step_3_live_authorization_gate`

Current pending task:

- `0713T002` business execution is complete and pending QA.

Accepted result summary:

- `0712T001` QA accepted that current artifacts can bind proxy lifecycle/depth fields, but cannot reconstruct actual resting-interval public trades or actual depletion/trade-through.
- accepted resting/no-fill attempts: `3`
- actual interval public trades: `not_reconstructable_from_current_artifact` for all `3`
- actual interval depletion/trade-through: `not_reconstructable_from_current_artifact` for all `3`
- review-fix commit `65f2461` tightened future route semantics so partial interval trades cannot be treated as `offline_repair_sufficient` unless exact lifecycle/depth evidence is also present.
- `0713T001` QA accepted offline/mock resting-interval capture instrumentation in the existing watcher artifact path.
- Step 3 controlled same-envelope live evidence was later authorized by the user on 2026-07-13 with the default same conservative envelope and executed as `0713T002`.
- `0713T002` business result:
  - remote collection host: `awsserver1`
  - remote repo: `/home/admin/hftbacktest-cross-exchange`
  - remote artifact root: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/`
  - local pulled-back root: `local_live_analysis/cross_exchange_resting_interval_live_evidence_0713T002_20260713T064917Z/`
  - pullback method: `scp`
  - Window 1 classification: `submitted_resting_no_fill`
  - resting-interval artifact rows: lifecycle `1`, interval public trades `0`, resting-start L2/depth `1`, depletion `1`
  - pre-QA source-attribution repair commit `762e335` fixed future artifact task-id propagation and added `source_attribution_overlay.json` for the current pulled-back package.
  - interval public trades `0` means no matching attempt-keyed rows were captured in the proxy interval; it is not proof that no exchange public trades occurred.
  - local validation: sha256 reconciliation `70/70`, JSON parse errors `0/32`, CSV parse errors `0/35`, boundary `pass`
- Step 4 must not start until `0713T002` QA returns `已通过`.

Historical cross-exchange live evidence topology:

- The accepted cross-exchange live evidence route uses `awsserver1` for live collection.
- Remote repo: `awsserver1:/home/admin/hftbacktest-cross-exchange`
- Remote artifact parent: `awsserver1:/home/admin/hftbacktest-cross-exchange-artifacts/`
- Local pullback parent: `local_live_analysis/`
- Recent accepted examples:
  - `0709T001` remote root: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z/`
  - `0709T001` local root: `local_live_analysis/cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z/`
  - `0708T001` remote root: `/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_t010_fast_l2book_controlled_live_evidence_0708T001_20260707T160830Z/`
  - `0708T001` local root: `local_live_analysis/cross_exchange_t010_fast_l2book_controlled_live_evidence_0708T001_20260707T160830Z/`
- The older Binance maker route under `awsserver1:/home/admin/hft_live/` is a separate path and must not be repurposed for this cross-exchange plan unless a later formal task explicitly changes that boundary and passes QA.

## Auto-Loop Rules

- Only one formal task should be active at a time.
- Each implementation/evidence task must have a formal task file under `.workflow/tasks/`.
- Each implementation/evidence task must write a business report under `.workflow/reports/`.
- Each task must go through QA before the next task is created.
- `docs/qa-acceptance-report.md` remains the latest QA fact source.
- The loop stops immediately on `未通过` or `阻塞`.
- The loop stops before any live-submit step unless the exact live envelope is explicitly authorized in the formal task.
- If a step discovers a safety, endpoint, artifact, or replay optimism issue, the next task must be a narrow repair task, not the next planned step.

## Hard Boundaries For All Steps

These are forbidden unless a later formal task explicitly changes scope and receives QA acceptance:

- threshold change
- quote-envelope change
- order-size increase
- max-submission increase
- strategy behavior change
- live expansion
- fill-seeking placement
- fee/rebate calibration
- realized PnL claim
- queue-priority claim
- maker-viability claim
- T012 claim
- promotion
- final MVP pass

## Step 1: QA `0712T001`

Logical step name:

- `QA-0712T001-PUBLIC-FLOW-INTERVAL-ARTIFACT-REPAIR-DESIGN`

Action:

- Run normal QA over the completed `0712T001` business result.
- Verify the runner, tests, official output package, boundary manifest, task/report flow, and route semantics.

Primary files:

- `.workflow/tasks/0712T001.md`
- `.workflow/reports/0712T001-business.md`
- `examples/hyperliquid/cross_exchange_public_flow_interval_artifact_repair.py`
- `examples/hyperliquid/test_cross_exchange_public_flow_interval_artifact_repair.py`
- `local_live_analysis/cross_exchange_public_flow_interval_artifact_repair_0712T001/`
- `task_plan.md`
- `progress.md`
- `findings.md`

Required QA checks:

- Focused pytest passes.
- `py_compile` passes.
- CLI `--help` passes.
- JSON/CSV artifacts parse.
- Deterministic rerun matches official output.
- `git diff --check` passes.
- Boundary manifest confirms no live/remote/credential/private/account/order/cancel/market-data call and no parameter/strategy change.
- `offline_repair_sufficient` requires exact interval public trades, exact exchange resting timestamp, exact cancel/shutdown acknowledgement timestamp, exact resting-start L2 depth, and interval-derived depletion evidence.
- Unkeyed future `resting_interval_public_trades.csv` rows are not assigned to an order attempt.

Acceptance route:

- If QA is `已通过`, continue to Step 2.
- If QA is `未通过` because of a code/artifact issue, create a narrow `0712T001` repair task.
- If QA is `阻塞` because source artifacts are missing or ambiguous, stop for controller decision.

## Step 2: Resting-Interval Capture Instrumentation

Suggested logical task name:

- `T011-RESTING-INTERVAL-PUBLIC-FLOW-CAPTURE-INSTRUMENTATION`

Suggested formal ID if created on 2026-07-13:

- `0713T001`

Goal:

- Modify the controlled live watcher/artifact path so a future same-envelope live run can emit the fields required by `cross_exchange_resting_interval_public_flow_contract_v1`.
- This step is implementation and offline verification only. It must not run live and must not call endpoints.

Required artifact capability:

- `order_resting_exchange_time_ms` when available from the exchange response or nearest defensible source.
- `order_resting_local_receive_ts_ns`.
- `cancel_request_time_ms`.
- `cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms`.
- `resting_interval_public_trades.csv` keyed by exact `attempt`.
- `resting_start_l2_book_snapshot_at_or_after_order_resting`.
- `resting_interval_depth_depletion_matrix.csv` inputs sufficient to estimate trade-through/depletion during the actual resting interval.
- A boundary manifest confirming no threshold, quote-envelope, size, max-submission, or strategy behavior change.

Implementation constraints:

- Prefer extending existing Hyperliquid watcher artifact writers under `examples/hyperliquid/`.
- Do not alter quote placement, edge gates, anti-drift gates, size, TIF, or submission count.
- Do not introduce a fill-probability model.
- Do not infer exact fields from proxies without marking them as proxies.
- If an exact exchange timestamp is unavailable, write an explicit source/status field and preserve conservative routing.

Required verification:

- Focused pytest for new/changed artifact writer logic.
- `py_compile` for changed runner(s).
- CLI `--help` for changed runner(s).
- Synthetic/mock artifact test proving keyed public trades are captured only for the matching attempt.
- Synthetic/mock artifact test proving proxy lifecycle/depth does not route to `offline_repair_sufficient`.
- JSON/CSV schema validation for generated mock artifacts.
- `git diff --check`.

Acceptance route:

- If QA is `已通过`, continue to Step 3.
- If QA fails due to schema ambiguity, create one narrow schema repair task.
- If QA finds any endpoint/live/parameter boundary crossing, stop and repair before any live evidence task.

## Step 3: Controlled Same-Envelope Live Evidence With Resting-Interval Artifacts

Suggested logical task name:

- `T011-CONTROLLED-SAME-ENVELOPE-LIVE-EVIDENCE-WITH-RESTING-INTERVAL-PUBLIC-FLOW`

Suggested formal ID if created after Step 2 QA:

- `0713T002` or the next available `MMDDTxxx`

Goal:

- Run a controlled live evidence task using the Step 2 instrumented path, solely to capture actual resting-interval public-flow artifacts.
- The goal is not to improve fill rate.

Live authorization gate:

- This step must not start until a formal task records the exact live envelope and explicit controller authorization.
- If authorization is absent or ambiguous, stop the auto-loop and ask for authorization.

Execution topology and artifact flow:

- Live-related collection for this step must run on `awsserver1`.
- The formal task must record the remote repo as `/home/admin/hftbacktest-cross-exchange` and the remote artifact parent as `/home/admin/hftbacktest-cross-exchange-artifacts/` unless it explicitly narrows or changes that path and receives QA acceptance.
- Remote artifact roots should follow the historical pattern:
  - `/home/admin/hftbacktest-cross-exchange-artifacts/<TASK_ID_or_run_id>/`
- After collection finishes, the complete artifact package must be pulled back to the local workspace before offline processing, analysis, or QA:
  - `local_live_analysis/<same_TASK_ID_or_run_id>/`
- The business report must record the remote source root, local destination root, pullback method, file counts, checksum or sha256 reconciliation, and the accepted local artifact package path.
- `scp` is an acceptable pullback method when recorded explicitly; do not require `rsync` unless the formal task has first verified that it is available on `awsserver1`.
- Local processing and QA must use the pulled-back local artifact package. They must not analyze remote files in place over SSH as the accepted artifact source.
- The local machine must not perform live-submit, credential reads, private/account/order/cancel endpoint calls, or new live market-data collection for this step.
- `awsserver1` must not run Step 4 quote/fill probability analysis, alignment, or downstream processing. It is the live collection host for this step; accepted downstream processing happens locally after pullback.
- If pullback or checksum reconciliation is incomplete, route to artifact repair or `阻塞`; do not proceed to Step 4.

Default live envelope:

- Same conservative envelope as accepted T011 live evidence unless the formal task narrows it further:
  - Hyperliquid `BTC`
  - post-only `Alo`
  - fast Hyperliquid `l2Book`
  - max order size `0.005 BTC`
  - max submissions per window `2`
  - no threshold change
  - no quote-envelope change
  - no size increase
  - no max-submission increase
  - no fill-seeking placement
  - shutdown/cancel proof required
  - independent final open-orders proof required

Required evidence:

- Per-window lifecycle classification.
- Exact or explicitly-statused resting timestamp fields.
- Exact or explicitly-statused cancel/shutdown timestamp fields.
- Public trades during the actual resting interval, keyed by attempt.
- Resting-start L2/depth snapshot or explicit not-available status.
- Depletion/trade-through matrix over the actual resting interval.
- Feed health and reconnect/timeout summaries.
- Final open-orders `0` proof.
- Boundary manifest.

Acceptance route:

- If QA is `已通过` and at least one submitted/resting lifecycle has the new resting-interval artifacts, continue to Step 4.
- If QA passes but no submitted/resting lifecycle occurs, route to controller decision; do not automatically change thresholds or quote envelope.
- If QA fails because artifacts are malformed, create a narrow artifact repair task.
- If QA fails because of live safety, stop the auto-loop.

## Step 4: Quote/Fill Probability Evidence Rerun

Suggested logical task name:

- `T011-QUOTE-FILL-PROBABILITY-EVIDENCE-RERUN-WITH-RESTING-INTERVAL-PUBLIC-FLOW`

Suggested formal ID if created after Step 3 QA:

- `0713T003` or the next available `MMDDTxxx`

Goal:

- Rerun quote/fill probability evidence using the newly accepted resting-interval public-flow artifacts.
- Determine whether no-fill is explained by no trade-through, visible queue/depth, short horizon/censoring, post-only reject behavior, or still-insufficient artifacts.

Allowed scope:

- Offline analysis only.
- Existing accepted live/replay/artifact packages only.
- The Step 3 input, if used, must be the verified local pullback package under `local_live_analysis/<same_TASK_ID_or_run_id>/`, with the remote `awsserver1` source root recorded for provenance only.
- New or updated analysis runner/tests if required.

Not allowed:

- live-submit
- remote/AWS execution
- reading or processing remote `awsserver1` artifact paths in place instead of a verified local pullback package
- credential reads
- private/account/order/cancel endpoint calls
- new market-data collection
- threshold changes
- quote-envelope changes
- order-size or max-submission changes
- strategy behavior changes
- fee/rebate or realized PnL claims without fill-supported evidence
- maker-viability, T012, promotion, or final MVP claims

Required output:

- Attempt-level quote/fill evidence matrix.
- Resting-interval public-trades/depletion summary.
- Censoring/horizon matrix.
- Boundary manifest.
- Validation report.
- Machine-readable final route enum.

Allowed final route enums:

- `route_to_more_conservative_evidence`
- `route_to_quote_policy_design`
- `route_to_public_flow_artifact_repair`
- `route_to_controlled_same_envelope_live_evidence`
- `route_to_fee_inventory_pnl_calibration`
- `stop_for_human_strategy_decision`

Route restrictions:

- `route_to_fee_inventory_pnl_calibration` is allowed only if accepted fill-supported evidence exists.
- `route_to_quote_policy_design` is allowed only if actual resting-interval public-flow/depth evidence supports a quote-placement interpretation.
- `route_to_public_flow_artifact_repair` remains required if interval artifacts are still missing or not reconstructable.

Acceptance route:

- If QA is `已通过`, the controller may decide whether the next work is quote policy design, more evidence, or stop.
- If QA is `未通过` due to optimism or unsupported fill probability, repair the analysis before any strategy work.
- If QA is `阻塞` due to missing live artifacts, return to Step 2 or Step 3 depending on the blocker.

## Auto-Loop Stop Conditions

Stop immediately if any of these happen:

- Latest QA is `未通过` or `阻塞`.
- Any live task has missing cancel/shutdown proof.
- Any live task has nonzero final open orders.
- Any live task collects outside `awsserver1` or writes outside `/home/admin/hftbacktest-cross-exchange-artifacts/` without explicit formal authorization and QA acceptance.
- Any downstream analysis runs on `awsserver1` or uses remote artifacts in place before verified local pullback.
- Any task changes thresholds, quote envelope, size, max submissions, or strategy behavior without explicit formal authorization.
- Any analysis claims fill probability, queue priority, fee/rebate, PnL, maker viability, T012 readiness, promotion, or final MVP pass without accepted supporting evidence.
- Any task uses unkeyed or proxy-only public-flow data as exact resting-interval proof.

## Current Next Action

Create no new implementation or live task yet.

The next action is:

```text
Stop at the Step 3 live authorization gate.
```

Only after an explicit controller authorization records the exact live envelope, `awsserver1` execution topology, remote artifact root, local pullback path, and safety requirements should the controller create a separate Step 3 formal live evidence task.
