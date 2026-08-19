# Live REST Safety Reconciliation Design

## Goal

Prevent live trading on stale connector state by reconciling the Python bot's local `hbt.position(0)` and open orders against Binance Futures REST ground truth on startup and periodically during live trading.

## Incident Summary

During the isolated one-hour live alignment run, the bot reported:

```text
hbt.position(0) = -0.029 BTC
```

Binance REST reported:

```text
positionAmt = +0.045 BTC
openOrders = []
```

Read-only user-trade query for the run window showed:

```text
74 BUY fills = +0.074 BTC
-0.029 + 0.074 = +0.045
```

So fills happened and the exchange position changed, but the bot's local position did not update. The likely root cause is that connector position updates rely on `ACCOUNT_UPDATE`; `ORDER_TRADE_UPDATE` updates order state but not position, so missed/delayed/unpropagated account updates can leave `hbt.position(0)` stale.

## Approach

Use Approach C:

1. Add fail-closed REST safety reconciliation in Python live bot.
2. Add audit fields for REST/local position and safety status.
3. Add connector instrumentation in Rust to make `ACCOUNT_UPDATE` and `ORDER_TRADE_UPDATE` observable.

The Python safety layer is the immediate protection. Connector instrumentation helps root-cause and later connector fixes, but live trading safety must not depend on connector correctness.

## Live Safety Config

Add:

```toml
[live_safety]
enabled = true
rest_check_interval_sec = 5
position_tolerance = 0.003
open_order_check = true
fail_on_mismatch = true
connector_config = "~/hft_live/config/binancefutures.toml"
```

`position_tolerance = 0.003` is intentionally larger than one lot (`0.001 BTC`) to avoid failing on a single in-flight fill/update race. A mismatch of three lots means local state is materially stale for this strategy.

## Startup Behavior

After creating the live bot and before sending any strategy orders:

```text
1. Query Binance REST positionRisk for BTCUSDT.
2. Query Binance REST openOrders for BTCUSDT.
3. Let the live bot process its connector startup batch.
4. Compare REST position with hbt.position(0).
5. Compare REST open-order count with local open-order count.
6. If mismatch exceeds tolerance or order counts differ, abort before trading.
```

No API keys are logged.

## Periodic Behavior

Inside the live loop, every `rest_check_interval_sec`:

```text
1. Query REST positionRisk/openOrders.
2. Compare REST position to hbt.position(0).
3. Compare REST open-order count to local NEW working orders.
4. If mismatch and fail_on_mismatch=true:
   - log critical error
   - stop event loop
   - attempt existing graceful shutdown path
```

The bot does not try to auto-correct positions. It fails closed.

## Audit Fields

Add fields:

```text
rest_position
position_mismatch
rest_open_order_count
local_open_order_count
safety_status
```

`safety_status` values:

```text
ok
startup_mismatch
position_mismatch
open_order_mismatch
rest_error
safety_disabled
```

Rows use the latest safety check values until the next REST check.

## REST Client Scope

The Python REST client is read-only for normal safety checks:

```text
GET /fapi/v2/positionRisk
GET /fapi/v1/openOrders
```

It reads `api_url`, `api_key`, and `secret` from the connector TOML config. It must never print credentials.

## Connector Instrumentation

Add Rust logs for:

```text
startup get_position_information result
ACCOUNT_UPDATE position amount per symbol
ORDER_TRADE_UPDATE side/status/fill quantity/client order id
```

This is observability only. It should not change connector behavior in this phase.

## Testing

Add unit tests for Python safety helpers:

1. Position mismatch within tolerance returns `ok`.
2. Position mismatch greater than tolerance returns `position_mismatch`.
3. Open-order count mismatch returns `open_order_mismatch`.
4. REST errors return `rest_error`.
5. Audit row includes safety fields.

Run:

```bash
python -m py_compile examples/binance_tick_mm/live_tick_mm.py examples/binance_tick_mm/strategy_core.py
python -m pytest examples/binance_tick_mm/test_strategy_core.py examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

For Rust instrumentation:

```bash
cargo check -p connector
```

## Live Validation

Before resuming trading:

1. Query REST position manually.
2. If existing exchange position is not intended, user must manually flatten or explicitly authorize the bot to reduce it.
3. Start bot with safety enabled.
4. Confirm startup safety row logs:

```text
rest_position ~= hbt.position within 0.003
safety_status = ok
```

5. During a short live run, verify:

```text
position_mismatch count = 0
open_order_mismatch count = 0
rest_position tracks hbt.position
```
