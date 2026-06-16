# Cross-Exchange Auto Loop Protocol

Status: controller runbook

This document defines the default auto-loop sequence for the `cross-exchange`
branch. It is a controller-side operating note, not a task implementation.

## Objective

Minimize task interruption while keeping the branch boundary strict:

- Binance is read-only lead-side pricing context.
- Hyperliquid is the maker execution venue under separate readiness gates.
- No live order task may start until the prior readiness gates pass QA and the
  controller explicitly approves a live window.

## Auto-Continue Conditions

The loop may continue automatically when all of the following are true:

1. Current task report is `待验收`.
2. Latest QA report is `已通过`.
3. The task recommendation matches the pre-approved next step.
4. No private endpoint, credentials, signing, nonce, or user-stream work was introduced.
5. No real order placement, cancellation, amendment, or live bot work was introduced.
6. `git diff --check` and task-scoped verification passed.
7. The task did not expand scope beyond the dispatched boundary.

## Stop Conditions

Stop and ask the controller when any of the following happens:

- QA is `未通过` or `阻塞`.
- A task asks for real Hyperliquid private/order access.
- A task asks for credentials, signing, nonce, or user-stream implementation.
- A task asks for real order placement, cancellation, or account-query work.
- A task asks to switch from Hyperliquid maker readiness to Binance live trading.
- A task needs a symbol, notional cap, loss cap, account, machine, or live window decision.
- A task crosses from design / fixture / dry-run work into real execution.

## Fixed Five-Step Queue

1. `0616T001` QA acceptance.
2. Hyperliquid private/order readiness boundary.
3. Hyperliquid no-trading private artifact fixture / validator.
4. Hyperliquid cancel-all / shutdown dry-run proof gate.
5. Hyperliquid tiny-live protocol design.

## Human Approval Gate

Auto loop must stop before any real live order task.

The controller must approve the following before live can proceed:

- symbol
- max notional
- max order size
- max position
- max loss
- duration
- machine / host
- whether real orders are allowed

## Required Facts Per Step

Each task report should preserve:

- task id
- status
- files changed
- action taken
- verification command(s)
- done / blockers
- commit id and commit message

## Intended Use

Use this file as the single controller reference for the current `cross-exchange`
auto loop. Do not treat it as authorization for live execution.
