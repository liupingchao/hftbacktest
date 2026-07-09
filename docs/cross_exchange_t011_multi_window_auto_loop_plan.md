# Cross-Exchange T011 Multi-Window Evidence Auto-Loop Plan

## Purpose

This plan defines the next sequential auto-loop after the accepted single-window `0625T010` same-window replay result.

Accepted current facts:

- `0708T001` produced one controlled live post-only `Alo` lifecycle with fast Hyperliquid `l2Book` enabled.
- The live order lifecycle was bounded: one `buy 0.002 BTC @ 63889.0`, status `resting`, no fill, cancel endpoint called, final open orders `0`, independent final open-orders proof `0`.
- `0708T002` replayed the same live window and passed same-window replay acceptance for market view, decision path, submit/resting/cancel/no-fill lifecycle, inventory/no-fill attribution, and non-optimistic replay behavior.

The next goal is not to prove maker profitability. The next goal is to move from one accepted same-window lifecycle to a small multi-window evidence set that can show whether the execution/replay stack remains faithful across windows.

## Auto-Loop Overview

```text
T001 controlled multi-window live evidence acquisition
  -> QA pass required
T002 batch same-window replay acceptance
  -> QA pass required
T003 multi-window robustness synthesis and next-route decision
  -> QA pass required
```

Only one formal task should be active at a time. `T002` must not be created or executed until `T001` has a QA result of `已通过`. `T003` must not be created or executed until `T002` has a QA result of `已通过`.

Formal task files must use the repository workflow ID format, for example `0709T001`, `0709T002`, `0709T003`. This document uses `T001/T002/T003` as logical auto-loop step names.

## Hard Boundaries

These boundaries apply to every task in this plan:

- No default-on or continuous bot operation.
- No promotion, production claim, stable PnL claim, maker viability claim, `T012` claim, or final MVP pass.
- No threshold change inside this plan.
- No quote-envelope change inside this plan.
- No order size increase above `0.005 BTC`.
- No max submission increase above `2` per live window.
- No taker, crossing, inside-spread, or one-tick-back behavior.
- Live orders, if any, must be post-only `Alo`.
- Fast Hyperliquid `l2Book` must be enabled for live evidence.
- Every live window must end with tracked cancel/shutdown evidence and independent final open-orders proof.
- If any window ends with nonzero open orders, missing cancel proof, malformed lifecycle evidence, or unexpected endpoint usage, stop the auto-loop and route to a safety repair task.

## Authorization Scope

The user has authorized low-risk live testing for this evidence route. This authorization is recorded only for `T001`, only inside the envelope below, and only after a formal task file is created.

The authorization does not permit:

- larger size,
- more submissions,
- different quote placement,
- changed thresholds,
- long-running unattended operation,
- strategy promotion,
- claims about profitability.

## T001: Controlled Multi-Window Live Evidence Acquisition

Logical task name:

- `T011-CONTROLLED-MULTI-WINDOW-LIVE-EVIDENCE`

Goal:

- Collect a small set of independent controlled live windows using the same conservative code path and risk envelope accepted in `0708T001`.
- Produce enough real lifecycle evidence for batch replay acceptance.

Default live envelope:

- Host: `awsserver1`
- Remote repo: `/home/admin/hftbacktest-cross-exchange`
- Interpreter: `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`
- Env file: `/home/admin/XEMM_rust_latest/.env`
- Runner: `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- Required mode: `--event-driven-edge-gate-live`
- Required feed option: `--hyperliquid-l2book-fast`
- Symbol: Hyperliquid `BTC`
- Order type: limit
- TIF: post-only `Alo`
- Side policy: `fresh_touch`
- Quote offset: `0` tick touch-only
- Window count target: `3`
- Max duration per window: `1800s`
- Max submissions per window: `2`
- Max order size: `0.005 BTC`
- Quote hold: `3s`
- Wait seconds: `10s`

Execution rules:

- Run windows sequentially, not in parallel.
- Stop each window after the runner naturally completes its trigger/lifecycle path or after the bounded duration.
- Use a distinct artifact root per window.
- Preserve raw logs, manifests, summaries, order lifecycle rows, cancel evidence, final open-orders proof, and feed health summaries.
- Do not continue to a later window if a prior window violates the hard boundaries.

Required output:

- Local artifact root, for example:
  - `local_live_analysis/cross_exchange_t011_multi_window_live_evidence_<FORMAL_TASK_ID>_<timestamp>/`
- Per-window classification:
  - `submitted_resting_no_fill`
  - `submitted_filled`
  - `submitted_rejected`
  - `no_submit_fail_closed`
  - `blocked_or_malformed`
- Per-window evidence:
  - public feed health: `l2Book`, trades, reconnects, timeouts, feed mode
  - candidate and trigger counts
  - trigger candidate audit fields
  - current reprice decision fields
  - post-open-orders public-state handoff latency
  - submit, resting, reject, cancel, fill, no-fill lifecycle
  - final open-orders proof
  - endpoint boundary manifest

Acceptance criteria:

- At least one formal business report is written under `.workflow/reports/`.
- QA report is written under `.workflow/reports/` and copied to `docs/qa-acceptance-report.md`.
- All live windows either satisfy the conservative envelope or are clearly marked fail-closed.
- No unexpected endpoint is called.
- Every submitted order has complete lifecycle evidence.
- Every window has final open-orders `0` by independent proof.
- The evidence is sufficient for `T002` to locate and replay each accepted window.

Route after QA:

- If QA passes, create `T002`.
- If QA fails or blocks because evidence is malformed, route to a narrow artifact/schema repair task.
- If QA fails or blocks because of live safety, stop and route to a safety repair task.
- If all windows are valid but no orders submit, `T002` may still proceed only if QA accepts that decision-path replay is meaningful for fail-closed windows.

## T002: Batch Same-Window Replay Acceptance

Logical task name:

- `T011-BATCH-SAME-WINDOW-REPLAY-ACCEPTANCE`

Goal:

- Replay `0708T001` plus all QA-accepted `T001` windows and verify that replay remains faithful and non-optimistic across the multi-window evidence set.

Allowed scope:

- Offline replay and analysis only.
- Existing replay runner or a small batch wrapper around it.
- Focused tests for any new batch aggregation code.
- Artifact schema normalization when needed to read accepted windows.

Not allowed:

- Live-submit.
- Credentials.
- Private/account/order/cancel endpoints.
- Threshold changes.
- Quote-envelope changes.
- Strategy behavior changes.
- Viability or profitability claims.

Required checks:

- Market view:
  - replay uses the same accepted same-window public state;
  - replay does not see future market data relative to decision/lifecycle events.
- Decision path:
  - trigger candidate, fair-mid, side, quote intent, reprice decision, guard reasons, and fail-closed reasons match the live artifact.
- Lifecycle:
  - submit/resting/reject/cancel/fill/no-fill state matches the live artifact;
  - replay does not synthesize fills;
  - replay does not erase rejects, cancels, or fail-closed paths.
- Economics:
  - no-fill windows attribute zero realized PnL and no invented rebate;
  - filled windows, if any, require fee/rebate/inventory/PnL attribution from supported facts only.
- Optimism boundary:
  - replay must not improve fill probability, latency, order priority, fees, PnL, or final inventory relative to the live evidence.

Required output:

- Per-window replay acceptance matrix.
- Aggregate replay acceptance summary.
- Machine-readable artifact containing:
  - window ID
  - live classification
  - replay classification
  - market-view pass/fail
  - decision-path pass/fail
  - lifecycle pass/fail
  - economics pass/fail
  - optimism-boundary pass/fail
  - blocker, if any

Acceptance criteria:

- `0708T001` remains passing under the batch path.
- Every QA-accepted `T001` window has an explicit pass/fail/block result.
- No accepted replay result depends on future data, invented fill, invented cancel, invented fee/rebate, or optimistic PnL.
- The task report clearly separates:
  - replay system acceptance,
  - execution safety,
  - opportunity/fill scarcity,
  - strategy viability.

Route after QA:

- If QA passes for all accepted windows, create `T003`.
- If only a subset passes, create a narrow replay/schema repair task before `T003`.
- If replay is optimistic for any submitted lifecycle, stop and repair replay before any further live task.

## T003: Multi-Window Robustness Synthesis

Logical task name:

- `T011-MULTI-WINDOW-ROBUSTNESS-SYNTHESIS`

Goal:

- Convert the accepted multi-window live/replay evidence into a route decision for the next MVP step.
- Decide whether the next task should be more evidence acquisition, a narrow model/execution repair, fee/PnL attribution, or a stop for human strategy decision.

Allowed scope:

- Offline analysis over accepted `0708T001`, `T001`, and `T002` artifacts.
- Documentation updates to task plan, progress, findings, and QA summary.
- No live activity.
- No strategy behavior changes.

Required synthesis metrics:

- Number of accepted windows.
- Feed health by window:
  - `l2Book` cadence,
  - trade cadence,
  - reconnects,
  - timeouts.
- Candidate funnel by window:
  - candidates,
  - fresh-touch allowed,
  - anti-drift pass/block,
  - edge pass/block,
  - would-submit or submitted.
- Execution lifecycle by window:
  - submitted count,
  - resting count,
  - rejects,
  - cancels,
  - fills,
  - final open orders.
- Handoff timing:
  - post-open-orders L2 latency,
  - candidate age at guard,
  - stale/fail-closed reason distribution.
- Replay acceptance:
  - market-view pass rate,
  - decision-path pass rate,
  - lifecycle pass rate,
  - economics pass rate,
  - optimism-boundary pass rate.
- Economics:
  - no-fill attribution,
  - fill attribution if any,
  - fee/rebate/inventory/PnL support status.

Route decision rules:

- If any safety invariant fails, route to safety repair and stop live expansion.
- If replay is optimistic in any accepted lifecycle, route to replay repair and stop live expansion.
- If all accepted windows are no-submit/fail-closed, route to signal/opportunity distribution diagnosis. Do not claim execution acceptance beyond fail-closed replay.
- If multiple submitted/no-fill windows replay faithfully, route to quote/fill probability evidence. Do not claim profitability.
- If at least one fill occurs and replay/economics attribution is supported, route to fee/rebate/inventory/PnL calibration before any `T012` promotion.
- If the evidence set is too small or too homogeneous, route to another controlled multi-window evidence task with the same envelope rather than changing thresholds.

Acceptance criteria:

- The synthesis has a single explicit recommendation:
  - `route_to_more_controlled_evidence`
  - `route_to_signal_distribution_diagnosis`
  - `route_to_quote_fill_probability_evidence`
  - `route_to_fee_inventory_pnl_calibration`
  - `route_to_replay_repair`
  - `route_to_execution_safety_repair`
  - `stop_for_human_strategy_decision`
- `task_plan.md`, `progress.md`, and `findings.md` are updated with the accepted route.
- The latest QA result is copied to `docs/qa-acceptance-report.md`.
- The report does not claim final MVP pass, stable PnL, or maker viability.

## Auto-Loop Stop Conditions

Stop the loop immediately if any of these occur:

- QA result is `未通过` or `阻塞`.
- A live artifact shows nonzero final open orders.
- A live artifact is missing cancel or shutdown evidence for a submitted order.
- Any unexpected endpoint is called.
- Replay invents a fill, cancel, fee, rebate, PnL, or market state.
- A task requires threshold, quote-envelope, size, or max-submission changes.
- A task requires a major strategy decision.

## Controller Checklist

For each step:

1. Create exactly one formal task file under `.workflow/tasks/`.
2. Dispatch execution with the scope and boundaries in this document.
3. Require a business or test report under `.workflow/reports/`.
4. Dispatch QA acceptance.
5. Copy latest QA to `docs/qa-acceptance-report.md`.
6. Update `task_plan.md`, `progress.md`, and `findings.md`.
7. Continue only if QA status is `已通过` and the route condition allows the next step.

## Current Dispatch Recommendation

Create and execute the first formal task:

- Logical: `T001 / T011-CONTROLLED-MULTI-WINDOW-LIVE-EVIDENCE`
- Suggested formal ID if created on 2026-07-09: `0709T001`

Do not create or execute `T002` or `T003` until `T001` has QA status `已通过`.
