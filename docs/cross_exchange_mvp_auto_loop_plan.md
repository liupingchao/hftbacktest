# Cross-Exchange Maker MVP Auto Loop Plan

## 1. Purpose

This document defines the controller auto-loop for the Binance-lead / Hyperliquid-lag maker MVP after `0625T003` business execution.

The goal is to let the workflow proceed through all tasks that do not require a new major human decision:

```text
create task -> execute business thread -> run QA -> update facts -> create next task
```

This plan does not weaken the staged MVP gates in `docs/cross_exchange_maker_mvp_plan.md`. It only defines when the controller may continue automatically and when it must stop for human decision.

## 2. Current Starting Point

Current canonical branch:

- `cross-exchange`

Current workflow fact:

- Latest QA source of truth is still `0627T001 已通过`.
- `0625T003` business execution is complete and `待验收`.
- `0625T003` business recommendation is `signal_contract_accepted_for_shadow`.
- `0625T003` is not QA-accepted yet.

Current T003 business package:

- `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/`

T003 accepted business candidate:

- candidate: `binance_lead_composite`
- features:
  - `input_binance_top5_imbalance`
  - `input_binance_microprice_minus_mid_ticks`
  - `input_binance_mid_move_ticks_from_prev`
- threshold: `abs(z) >= 1.0`
- side mapping: `positive_signal_buy_negative_signal_sell`
- caveat: one source-age bucket has negative adjusted proxy; this must stay visible in QA and later shadow tasks.

## 3. Auto-Loop Controller Rules

The controller may automatically continue only when all conditions hold:

1. The previous task has a QA report with status `已通过`.
2. The latest valid QA result has been copied to `docs/qa-acceptance-report.md`.
3. The next task is exactly the next sequential task in this plan.
4. The next task does not expand risk, live order authority, symbol scope, venue scope, quote placement envelope, position cap, or production behavior beyond the accepted task boundary.
5. The next task has a task file under `.workflow/tasks/`.
6. The business/test result has a report under `.workflow/reports/`.
7. The task result is committed before QA, unless the task explicitly says no commit is required.

The controller must stop and request human decision if any condition occurs:

- QA result is `未通过` because the acceptance conclusion is negative, not merely because of a fixable implementation/report defect.
- QA result is `阻塞`.
- A task recommendation is one of:
  - `reject_current_signal_shape`
  - `needs_more_samples`
  - `mvp_needs_targeted_repair`
  - `mvp_rejected`
- A task proposes changing:
  - symbol or venue
  - prediction horizon
  - signal feature schema outside the accepted contract
  - side mapping
  - quote distance / one-tick-back / inside-spread behavior
  - post-only rule
  - order size cap
  - live order count cap
  - default-on behavior
  - production config
  - private/order endpoint usage beyond the explicitly scoped tiny-live calibration window
- A task would place real live orders and no current standing live authorization record exists.
- A task would claim final MVP pass.

## 4. Automatic Repair Policy

If QA fails because of a narrow implementation, artifact, documentation, or verification defect inside the existing task scope, the controller may create one automatic repair task.

Automatic repair task rules:

- Use the same milestone and the next available task suffix only if a separate task is clearer.
- Keep scope limited to the QA defect.
- Do not reinterpret the acceptance conclusion.
- Do not relax any gate.
- Send the repaired result back to QA.

Automatic repair is not allowed when the failure means the strategy/evidence did not pass the milestone gate. In that case, stop for human decision.

## 5. Auto-Loop Task Queue

### Step 0: `0625T003-QA` Signal Acceptance QA

Action:

- QA validates `.workflow/tasks/0625T003.md`, `.workflow/reports/0625T003-business.md`, runner/tests, and package `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/`.
- QA checks input gate, row filters, leave-one-window-out split, same-window backfill absence, candidate allowlist, side mapping stability, adjusted edge proxy, warning bucket treatment, boundary manifest, deterministic reproduction, and git hygiene.

Auto-continue condition:

- QA status `已通过`.
- QA conclusion accepts `signal_contract_accepted_for_shadow`.

Stop condition:

- T003 QA is `未通过` because signal acceptance is invalid.
- T003 QA is `阻塞`.
- QA requires changing the signal schema, threshold, side mapping, or horizon.

Next automatic task if passed:

- `0625T004`.

### Step 1: `0625T004` Shared Signal and Quote-Intent Kernel

Goal:

- Create or extract a shared pure decision kernel used by public shadow and replay.

Inputs:

- QA-accepted `0625T003` signal contract.
- Existing watcher/public-shadow/fair-mid gate code.

Required behavior:

- Implement a deterministic pure kernel for:
  - dual-market public view
  - accepted signal normalization / threshold
  - fair-mid calculation
  - side mapping
  - quote intent
  - block reason
- No private endpoints.
- No live orders.
- No production config change.
- No new signal tuning.

Expected artifacts:

- shared-kernel manifest
- fixture inputs
- fixture outputs
- boundary manifest
- business report

Minimum verification:

- focused unit tests for deterministic fixed fixtures
- `py_compile`
- CLI/help check if a runner is introduced
- JSON/CSV artifact validation
- `git diff --check`

Auto-continue condition:

- QA status `已通过`.
- Fixed fixtures produce deterministic identical outputs.
- Boundary confirms no live/private/order behavior.

Next automatic task if passed:

- `0625T005`.

### Step 2: `0625T005` Multi-Window Production Shadow Acceptance

Goal:

- Run production-equivalent public shadow over multiple fresh public windows using the shared kernel and frozen signal contract.

Inputs:

- QA-accepted `0625T003` signal contract.
- QA-accepted `0625T004` shared kernel.

Required behavior:

- Public-only.
- No submit.
- No private/account/order/cancel endpoints.
- Preserve full funnel:
  - public market view
  - signal
  - fair-mid
  - side
  - quote intent
  - block reason
  - would-submit
  - future markout / counterfactual edge
- Keep the T003 warning bucket visible in diagnostics.

Expected artifacts:

- per-window shadow manifests
- funnel matrix
- would-submit rows
- counterfactual markout rows
- edge proxy summary
- regime/source-age/basis stability report
- boundary manifest
- business report

Minimum verification:

- focused tests for no-submit shadow path
- public-only boundary validation
- JSON/CSV schema and row-count checks
- deterministic local replay of produced artifacts where applicable
- `git diff --check`

Auto-continue condition:

- QA status `已通过`.
- would-submit sample size is sufficient for bucket analysis.
- fee/adverse-buffer-adjusted counterfactual edge is not systematically negative.
- no single window dominates the result.

Stop condition:

- would-submit count remains zero or too thin.
- adjusted edge is systematically negative.
- result depends on one window or one fragile bucket.
- task recommends return to T003/T004.

Next automatic task if passed:

- `0625T006`.

### Step 3: `0625T006` Hyperliquid MVP Audit and Replay Contract

Goal:

- Define and implement the minimal audit/replay contract needed for the MVP.

Inputs:

- QA-accepted T003-T005 artifacts.
- Existing Hyperliquid raw conversion, public watcher, lifecycle, and ledger artifacts.

Required behavior:

- Define schema for:
  - run id
  - event id
  - decision id
  - order id
  - Binance/Hyperliquid timestamps and source ages
  - signal/fair-mid/side/quote intent
  - submit/resting/reject/cancel/fill lifecycle
  - fee/inventory/PnL fields
- Validate synthetic lifecycle fixtures.
- Validate existing tiny-live artifacts where possible.
- No new live execution.

Expected artifacts:

- audit schema manifest
- replay input contract
- schema hash
- validator
- synthetic lifecycle fixtures
- validation report
- boundary manifest
- business report

Minimum verification:

- focused validator tests
- synthetic fixture acceptance
- existing artifact compatibility check
- `py_compile`
- JSON/CSV validation
- `git diff --check`

Auto-continue condition:

- QA status `已通过`.
- Synthetic lifecycle fixture passes.
- Existing tiny-live/audit artifacts are either accepted or explicitly classified as unsupported with conservative reasons.

Next automatic task if passed:

- `0625T007`.

### Step 4: `0625T007` Hyperliquid Public Market-View Replay Alignment

Goal:

- Prove public market view and shared decision path can be replayed against the same public window.

Inputs:

- QA-accepted shared kernel.
- QA-accepted audit/replay contract.
- Public raw windows from accepted collection or newly scoped public-only collection.

Required behavior:

- Rebuild Binance/HL top5 from public raw.
- Replay decision cadence.
- Compare:
  - market view
  - source age
  - signal
  - fair-mid
  - side
  - quote intent
  - block reason
- No private endpoints.
- No live orders.

Expected artifacts:

- replay market-view manifest
- action-path comparison
- mismatch attribution
- cadence/source-age report
- future-join report
- boundary manifest
- business report

Minimum verification:

- focused replay tests
- fixture determinism
- future join count check
- action-path mismatch schema checks
- `git diff --check`

Auto-continue condition:

- QA status `已通过`.
- future join count is zero.
- market-view/source-age/cadence gates pass.
- action-path differences are within tolerance or fully attributed.

Stop condition:

- unexplained market-view or decision-path mismatch.
- future join is nonzero.
- replay would be systematically optimistic.

Next automatic task if passed:

- `0625T008-PREFLIGHT`.

## 6. Live Boundary: Preflight Is Automatic, Live Submit Is Not

Real live orders are a major decision unless the repository contains a current standing live authorization record.

The auto-loop may automatically create and execute a preflight task:

### Step 5: `0625T008-PREFLIGHT` Edge-Qualified Tiny-Live Calibration Packet

Goal:

- Prepare the exact tiny-live calibration packet without submitting orders.

Required behavior:

- Validate all T005-T007 QA gates.
- Freeze risk envelope:
  - Hyperliquid `Alo`
  - tiny size cap
  - max order count
  - tracked cancel
  - independent final `open_orders=[]` proof
  - fail-closed PnL ledger
- Generate operator packet and commands.
- Validate credentials location without reading secrets.
- Confirm no live client initialization and no order endpoint call during preflight.

Auto-continue condition:

- If a standing live authorization record exists and explicitly covers this exact envelope, auto-loop may create `0625T008`.

Stop condition:

- No standing live authorization record exists.
- Risk envelope differs from accepted plan.
- Any credential/private/order action would occur during preflight.

Human decision required:

- Authorize or reject the first live-submit calibration task.

## 7. Conditional Post-Authorization Queue

The following tasks may be auto-looped only after a standing live authorization record exists for the exact tiny-live envelope.

### Step 6: `0625T008` Edge-Qualified Tiny-Live Calibration

Goal:

- Obtain real resting/reject/cancel/fill lifecycle and cost evidence under the accepted tiny-live envelope.

Auto-continue condition:

- QA status `已通过`.
- Complete lifecycle, fee/inventory/PnL, shutdown, and open-orders evidence exists.
- No boundary violation.

Stop condition:

- No lifecycle evidence.
- No fill where fill evidence is required.
- PnL ledger cannot close fail-safe.
- Any order/risk boundary violation.

Next automatic task if passed:

- `0625T009`.

### Step 7: `0625T009` Execution Outcome Calibration

Goal:

- Convert T008 real events into conservative replay execution parameters.

Required calibration:

- submit/ack latency
- post-only reject
- resting duration
- cancel race
- fill horizon
- adverse markout
- fee/rebate
- inventory transition

Auto-continue condition:

- QA status `已通过`.
- Parameters are conservative and do not use future decision inputs.
- Unsupported parameters are explicitly marked unsupported, not invented.

Next automatic task if passed:

- `0625T010`.

### Step 8: `0625T010` End-to-End Same-Window Replay Acceptance

Goal:

- Replay the T008 tiny-live window and verify same-window live/replay decision path.

Auto-continue condition:

- QA status `已通过`.
- action-path hard gates pass.
- replay is not systematically optimistic.
- lifecycle/PnL attribution is explainable.

Stop condition:

- replay/live mismatch is unexplained.
- replay optimism is detected.

Next automatic task if passed:

- `0625T011`.

### Step 9: `0625T011` Multi-Sample MVP Robustness

Goal:

- Check accepted live/replay behavior across at least three windows.

Auto-continue condition:

- QA status `已通过`.
- median and worst-window results are acceptable.
- no single profitable window is used to claim MVP readiness.
- regime/inventory/cost tails are explainable.

Stop condition:

- only one window works.
- worst window exposes unexplained execution or risk failure.

Next automatic task if passed:

- `0625T012-PREFLIGHT`.

### Step 10: `0625T012-PREFLIGHT` Final Controlled Validation Packet

Goal:

- Prepare final MVP validation bundle without claiming MVP pass.

Required behavior:

- Freeze code/config/schema/risk parameters.
- Generate final validation task file and operator packet.
- Validate required evidence paths.
- Confirm no default-on, expansion, or production promotion.

Auto-continue condition:

- A human has already authorized final controlled validation under the exact frozen packet.

Stop condition:

- No final validation authorization exists.
- Any change to code/config/schema/risk is needed.

Human decision required:

- Authorize final controlled validation run.

### Step 11: `0625T012` MVP Final Controlled Validation

Goal:

- Run final controlled validation and produce one complete artifact bundle.

Final recommendation enum:

- `mvp_passed_for_extended_shadow`
- `mvp_needs_targeted_repair`
- `mvp_rejected`

Auto-loop stop:

- Always stop after T012 QA.
- Final MVP disposition is a human/controller decision even if QA accepts the artifact quality.

## 8. Auto-Loop State Updates Per Task

After each business task:

- Write `.workflow/reports/<TASK_ID>-business.md`.
- Update `progress.md`.
- Update `findings.md` if new risk/fact is discovered.
- Commit code, docs, reports, and required artifacts.

After each QA task:

- Write `.workflow/reports/<TASK_ID>-qa.md`.
- Copy latest valid result to `docs/qa-acceptance-report.md`.
- Update `progress.md`.
- Update `findings.md` if QA changes the accepted fact.
- Commit QA report and status updates.

The controller may create the next task only after reading the latest QA result.

## 9. Task ID Policy

Formal task IDs stay aligned with the MVP roadmap:

- `0625T004` through `0625T012` remain reserved for the staged MVP tasks.
- If a preflight task is needed, use the same base with a suffix in the title but assign a valid workflow ID in `.workflow/tasks/`, for example the next available `0705Txxx`.
- If a repair task is needed, assign the next available valid workflow ID and reference the original task explicitly.

The title must preserve the roadmap task name, for example:

- `0705T001 / 0625T008-PREFLIGHT Edge-Qualified Tiny-Live Calibration Packet`

## 10. Stop Gates Summary

The auto-loop stops for human decision at these points:

1. T003 QA does not accept `signal_contract_accepted_for_shadow`.
2. T005 shows no usable would-submit or systematic negative counterfactual edge.
3. T007 public replay alignment fails or is unexplained.
4. Any task proposes changing the accepted signal, horizon, side mapping, or risk envelope.
5. Before first live-submit T008 unless standing authorization exists.
6. T008 lacks complete lifecycle/fill/cost evidence.
7. T010 replay is systematically optimistic.
8. T011 robustness depends on a single window.
9. Before final controlled validation unless standing authorization exists.
10. After T012 QA, regardless of result.

## 11. Immediate Next Action

The next automatic action is:

```text
Create and run QA for 0625T003.
```

If QA accepts `signal_contract_accepted_for_shadow`, the controller may automatically create `0625T004 Shared Signal and Quote-Intent Kernel`.

If QA does not accept it, stop for human decision.
