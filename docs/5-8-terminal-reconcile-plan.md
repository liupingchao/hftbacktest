# 5-8 Terminal Reconciliation Plan

## Goal

Fix the live/connector state reconciliation path so a late `cancel` after a `fill` or
`cancel` does not leave ghost working orders in the bot and trigger a premature safety stop.

## Failure Signature

The archived `5-8-night` sample is a failing baseline, not an acceptance sample.

Observed end state:

- `rest_position = 0.0`
- `local_position = 0.001`
- `rest_open_orders = 0`
- `local_open_orders = 2`
- final safety stop: `open_order_mismatch`

Connector-side evidence:

- Binance WS reported `TRADE status=Filled` for the sell leg.
- A later cancel on the same order hit the "client_order_id not found" path.

This means the trade/cancel race is real, but our code did not converge the local state back to terminal.

## Hypotheses

1. `OrderManager` drops the `order_id_map` too early for late cancel/fill reconciliation.
2. The connector does not emit a final terminal reconcile event when the mapping is already gone.
3. The live bot keeps the order in working state until the connector/bot event pipeline fully resolves the terminal transition.
4. `live_safety` turns the resulting ghost order into a hard stop instead of a targeted reconcile.

## Fix Plan

1. Make terminal order handling explicit in the connector.
2. Preserve enough mapping/state to deliver a final terminal order update even when a late cancel arrives.
3. Add a deterministic fallback for cancel-after-fill / cancel-after-cancel so the bot can clear the working order.
4. In the live bot, clear terminal working orders immediately on terminal order updates and add a narrow reconcile pass when REST says there are no open orders but local still shows only cancel-pending ghosts.
5. Keep the hard safety stop for real divergence, but do not stop on a recoverable terminal-order ghost.

## Code Areas

- `connector/src/binancefutures/ordermanager.rs`
- `connector/src/binancefutures/mod.rs`
- `connector/src/binancefutures/user_data_stream.rs`
- `examples/binance_tick_mm/live_tick_mm.py`

## Tests

Add coverage for:

- fill -> late cancel
- cancel -> late fill
- duplicate terminal updates
- REST zero open orders with local cancel-pending ghosts

The tests should assert that:

- terminal events clear the working order,
- local/rest open-order counts converge,
- no stale mapping warning becomes a state leak,
- the bot does not end with a ghost working order.

## Verification

The current `5-8-night` archive cannot prove the fix. It already contains the bug.

Use it only as a regression baseline:

- if the fix is correct, a fresh live run after the patch should not end with
  `open_order_mismatch`,
- final `rest_open_orders` and `local_open_orders` should match,
- final `rest_position` and `local_position` should match,
- the bot should not stop solely because a terminal order raced with a late cancel.

## Acceptance

The fix is accepted when:

- the new live sample completes without a ghost-order safety stop,
- terminal-order cleanup is visible in the audit/logs,
- the connector tests for late fill/cancel races pass,
- the archived `5-8-night` sample remains a known failing baseline, not a success case.
