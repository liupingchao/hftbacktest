# Planned-Action Audit and Quote Throttle Alignment Design

## Goal

Improve live/backtest alignment by making audit rows distinguish planned actions from executed actions, then add a shared quote throttle to reduce live API interval rejects without breaking existing audit consumers.

## Background

Round 2 live alignment found that the live bot produced high API interval rejects:

```text
api_reject_rate = 40.56%
decisions/sec avg = 69.18
non_keep/sec avg = 4.16
```

The quote churn report also showed misleading audit semantics:

```text
action = keep
reject_reason = api_interval_guard
```

This happens when the strategy planned an action, but API interval guard rejected it before execution. The current `action` field records executed action only, so rejected planned actions are indistinguishable from true keep decisions unless `reject_reason` is inspected.

## Scope

This design covers:

- Audit schema changes for planned vs executed action semantics.
- Shared quote throttle logic used by both live and backtest.
- Config additions for enabling throttle.
- Tests for the shared helper behavior and audit row semantics.

This design does not cover:

- Same-UTC Tardis replay.
- New strategy alpha signals.
- Changing tuned risk parameters.
- Deployment of new live config to the server.

## Audit Semantics

Keep existing fields:

```text
action
order_id
```

Their meaning remains:

```text
action   = actual executed action, or keep when nothing was sent
order_id = actual executed order IDs
```

Add fields:

```text
planned_action
planned_order_id
throttle_reason
```

Meanings:

```text
planned_action   = what the strategy wanted to do before local guards
planned_order_id = order IDs associated with planned actions
throttle_reason  = specific local throttle/guard reason when a planned action was suppressed
```

Examples:

```text
No action:
planned_action=keep
action=keep
reject_reason=
throttle_reason=

API interval reject:
planned_action=cancel_buy|submit_buy
action=keep
reject_reason=api_interval_guard
throttle_reason=api_interval

Partial token bucket reject:
planned_action=cancel_buy|submit_buy|cancel_sell|submit_sell
action=cancel_buy|submit_buy
reject_reason=token_bucket
throttle_reason=token_bucket

Quote throttle reject:
planned_action=cancel_buy|submit_buy
action=keep
reject_reason=quote_throttle
throttle_reason=min_quote_update_interval
```

Existing scripts can continue using `action` as executed action. New alignment reports should use `planned_action` when analyzing strategy intent.

## Quote Throttle Behavior

Add config:

```toml
[strategy]
quote_throttle_enabled = false
min_quote_update_interval_ms = 100
min_quote_move_ticks = 2
```

Default is disabled to preserve old benchmarks.

When enabled, quote throttle suppresses planned quote updates only when both are true:

```text
last successful API send was less than min_quote_update_interval_ms ago
and target bid/ask moved less than min_quote_move_ticks from last sent target
```

Throttle should not block risk-reducing or cleanup actions:

- Do not block `cancel_extra`.
- Do not block actions when `pos_limit` is true.
- Do not block actions that remove an undesired side due to position limit.

Initial live recommendation:

```toml
[strategy]
quote_throttle_enabled = true
min_quote_update_interval_ms = 100
min_quote_move_ticks = 2
```

Expected effect:

```text
lower api_reject_rate
shorter api reject bursts
lower non_keep/sec
clearer audit rows for rejected planned actions
```

## Architecture

Add shared helpers to `examples/binance_tick_mm/strategy_core.py`:

```text
QuoteThrottleConfig
QuoteThrottleState
format_actions(...)
should_throttle_quote_update(...)
```

Responsibilities:

- `format_actions(actions)` converts `Action` lists to `order_id` and action strings.
- `QuoteThrottleConfig` parses `[strategy]` config values with safe defaults.
- `QuoteThrottleState` tracks the last successfully sent API timestamp and target ticks.
- `should_throttle_quote_update(...)` decides whether planned actions should be suppressed.

Both `backtest_tick_mm.py` and `live_tick_mm.py` follow this flow:

```text
1. compute planned_actions
2. format planned_order_id and planned_action
3. apply quote throttle if enabled
4. apply API interval guard
5. apply token bucket guard
6. execute allowed actions
7. format executed order_id and action
8. write audit row with planned and executed fields
```

Update `audit_schema.py`:

```text
planned_order_id
planned_action
throttle_reason
```

Update `build_audit_row()` in `strategy_core.py` to accept and write those fields.

## Error Handling

Missing `[strategy]` section should not fail. Defaults are:

```text
quote_throttle_enabled = false
min_quote_update_interval_ms = 100
min_quote_move_ticks = 2
```

Invalid negative throttle values should be clamped to zero or treated as disabled behavior:

```text
min_quote_update_interval_ms < 0 -> 0
min_quote_move_ticks < 0 -> 0
```

Audit fields should always be populated:

```text
planned_action defaults to keep
planned_order_id defaults to empty string
action defaults to keep
order_id defaults to empty string
throttle_reason defaults to empty string
```

## Testing Plan

### Shared Helper Tests

1. `format_actions([])` returns:

```text
order_id=""
action="keep"
```

2. `format_actions([cancel buy, submit buy])` returns:

```text
order_id="1|2"
action="cancel_buy|submit_buy"
```

3. Throttle disabled never throttles planned actions.

4. Throttle enabled blocks small updates inside `min_quote_update_interval_ms`.

5. Throttle enabled allows updates outside `min_quote_update_interval_ms`.

6. Throttle enabled allows large target tick moves inside interval.

7. Throttle does not block `cancel_extra`.

8. Throttle does not block when `pos_limit` is true.

### Audit Row Tests

9. API interval reject records:

```text
planned_action != keep
action = keep
reject_reason = api_interval_guard
throttle_reason = api_interval
```

10. Quote throttle reject records:

```text
planned_action != keep
action = keep
reject_reason = quote_throttle
throttle_reason = min_quote_update_interval
```

11. Executed action records:

```text
planned_action == action
planned_order_id == order_id
throttle_reason = ""
```

### Integration Check

12. Run a short backtest with throttle enabled and `audit.mode = actions_only`.

Confirm:

- CSV has new columns.
- `planned_action` is populated.
- API interval guard rows show planned action and executed keep.
- Quote throttle rows show planned action and executed keep.
- Existing `action` remains executed action.

## Success Criteria

- Live/backtest audit rows clearly separate intended strategy actions from actions actually sent.
- Existing consumers that use `action` as executed action remain compatible.
- Quote throttle logic is shared by live and backtest.
- Quote throttle can be enabled by config without changing default benchmark behavior.
- Short live/backtest audit after enabling throttle should show substantially lower API interval guard rate than the observed 40.56% baseline.
