# Cancel Extra Diagnostics and Two-Phase Replace Design

## Goal

Reduce live `cancel_extra` churn by confirming its source with explicit audit diagnostics and adding an optional two-phase quote replacement policy that avoids same-cycle cancel+submit replacement.

## Background

The one-hour live validation after the `cancel_extra` API interval bypass showed:

```text
planned cancel_extra      = 38,043
executed cancel_extra     = 10,861
cancel_extra|api_interval = 0
cancel_extra|token_bucket = 27,182
```

Grouping by `planned_order_id` showed repeated cleanup attempts for the same order ID:

```text
unique cancel_extra order_ids = 2,921
p99 attempts per order_id     = 128
max attempts for one order_id = 542
```

Most `cancel_extra` order IDs appeared immediately after quote replacement actions:

```text
cancel_buy|submit_buy
cancel_sell|submit_sell
cancel_buy|submit_buy|cancel_sell|submit_sell
```

The current root-cause hypothesis is:

```text
same-cycle cancel+submit
+ live async order-state convergence
+ one-order-per-side invariant
= temporary duplicate NEW orders
= repeated cancel_extra retries
```

## Design Overview

Implement two linked changes:

1. Add audit diagnostics that expose current working order IDs and extras.
2. Add a configurable two-phase replace policy that cancels stale quotes first and submits replacements only after the side becomes empty.

These changes should be shared by backtest and live where practical, so local behavior remains aligned with live behavior.

## Audit Diagnostics

Add fields to audit rows:

```text
working_buy_order_id
working_sell_order_id
extra_order_ids
extra_order_sides
extra_order_price_ticks
```

Meanings:

- `working_buy_order_id`: primary NEW buy order selected by `collect_working_orders()`, or empty string.
- `working_sell_order_id`: primary NEW sell order selected by `collect_working_orders()`, or empty string.
- `extra_order_ids`: pipe-separated extra NEW order IDs, or empty string.
- `extra_order_sides`: pipe-separated sides for those extra orders, or empty string.
- `extra_order_price_ticks`: pipe-separated price ticks for those extra orders, or empty string.

To support this, extend `WorkingOrders` from:

```text
buy
sell
extra_ids
```

to also preserve structured extras:

```text
extras: list[ExtraOrder]
```

`extra_ids` may remain as a convenience property/list for existing logic.

## Two-Phase Replace Policy

Add config under `[strategy]`:

```toml
two_phase_replace_enabled = false
```

Default is false to preserve benchmark compatibility.

When disabled, current behavior remains unchanged:

```text
if working buy exists and price differs by >1 tick:
    cancel old buy
    submit new buy in the same decision
```

When enabled:

```text
if working buy exists and price differs by >1 tick:
    cancel old buy only
    do not submit replacement in the same decision

if desired buy and working buy is absent:
    submit buy
```

Sell side follows the same rule.

`cancel_extra` keeps priority over replacement logic:

```text
if extras exist:
    cancel one extra
    return immediately
```

Position-limit behavior keeps priority as today:

```text
if side is undesired due to pos_limit:
    cancel undesired side
    do not add replacement for that undesired side
```

## Expected Effect

Two-phase replace should reduce temporary duplicate NEW orders caused by live async order-state convergence.

Expected next live validation:

```text
planned cancel_extra/sec decreases substantially
cancel_extra|token_bucket decreases substantially
api_reject/sec decreases
Binance visible open orders become more stable
```

The audit diagnostic fields will show whether remaining extras are old stale quotes, newly submitted replacements, or mixed state from partial execution.

## Error Handling and Compatibility

- Missing `[strategy]` section defaults to `two_phase_replace_enabled = false`.
- Existing `action`, `order_id`, `planned_action`, and `planned_order_id` semantics do not change.
- Existing audit readers may need schema updates because this project treats audit schema as strict.
- Default-off policy preserves historical benchmark behavior.

## Testing Plan

### Unit Tests

Add tests for working-order diagnostics:

1. No orders produces empty working IDs and extras.
2. One buy and one sell populates primary order IDs.
3. Multiple buys/sells preserve extras with order ID, side, and price tick.

Add tests for action decisions:

1. `two_phase_replace_enabled = false` and buy diff > 1 produces:

```text
cancel_buy|submit_buy
```

2. `two_phase_replace_enabled = true` and buy diff > 1 produces:

```text
cancel_buy
```

3. `two_phase_replace_enabled = true` and no working buy produces:

```text
submit_buy
```

4. Extras still have priority:

```text
cancel_extra
```

5. Sell side mirrors buy side behavior.

### Integration Checks

Run:

```bash
python -m py_compile examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/strategy_core.py
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

### Live Validation

After implementation, run a short live validation with:

```toml
[strategy]
quote_throttle_enabled = true
min_quote_update_interval_ms = 100
min_quote_move_ticks = 2
two_phase_replace_enabled = true
```

Compare against the previous one-hour run:

```text
planned cancel_extra/sec
cancel_extra|token_bucket
api_reject/sec
quote_throttle count
executed_non_keep/sec
extra_order_ids source pattern
```
