# Cancel Extra API Interval Bypass Design

## Goal

Allow pure `cancel_extra` cleanup actions to bypass the API min-interval guard while still preserving token-bucket protection and existing quote-update throttling.

## Background

The 15-minute live validation with quote throttle enabled showed that many cleanup actions were suppressed by the API interval guard:

```text
cancel_extra -> keep | api_interval_guard | api_interval = 3543
```

`cancel_extra` is a cleanup/risk-control action, not a normal quote refresh. Repeatedly suppressing it can leave stale or undesired orders in the book and creates retry churn.

## Scope

This change covers only pure `cancel_extra` planned actions:

```text
planned_action = cancel_extra
```

It does not bypass interval guard for mixed actions such as:

```text
cancel_extra|cancel_buy|submit_buy
cancel_buy|submit_buy
cancel_sell|submit_sell
```

## Behavior

Add shared helper:

```text
is_pure_cancel_extra(actions)
```

It returns true only when the action list is non-empty and every action is:

```text
action.kind == "cancel"
action.side == "extra"
```

Backtest and live keep the same guard order:

```text
1. compute planned_actions
2. apply quote throttle
3. apply API interval guard
4. apply token bucket
5. execute allowed actions
6. write planned/executed audit fields
```

The API interval guard changes to:

```text
if inside min_interval and not is_pure_cancel_extra(planned_actions):
    reject with api_interval_guard
else:
    continue to token bucket/execution
```

Pure `cancel_extra` still goes through token bucket. If token bucket rejects it, audit remains:

```text
planned_action = cancel_extra
action = keep
reject_reason = token_bucket
throttle_reason = token_bucket
```

If it executes:

```text
planned_action = cancel_extra
action = cancel_extra
reject_reason =
throttle_reason =
```

## Testing

Add tests for:

1. `is_pure_cancel_extra([])` is false.
2. one or more `cancel_extra` actions are true.
3. mixed `cancel_extra` and quote actions are false.
4. the API interval bypass condition allows pure `cancel_extra` but not quote updates.

Run focused tests:

```bash
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
python -m py_compile examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/strategy_core.py
```
