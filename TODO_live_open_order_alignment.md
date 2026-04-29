# Live open-order alignment TODO

Context: 2026-04-28 2H live test run `live_btcusdt_1777342116` stopped after ~95 minutes due to open-order safety, not position mismatch.

## Confirmed

- Binance private `ACCOUNT_UPDATE` stream fix worked for position alignment.
- `position_mismatch` stayed safe:
  - `bad_gt_0.003 = 0`
  - `critical_position_mismatch = 0`
  - `max_position_mismatch = 0.001`
- Stop reason was open-order state drift:
  - local position matched REST position
  - local open orders and REST open orders diverged

## Reminder

Before the next long/production live test, fix open-order state alignment.

Likely root cause:

- Connector now subscribes only to Binance private `ACCOUNT_UPDATE`.
- Binance new private stream seems to require one `events=` type per stream.
- Without `ORDER_TRADE_UPDATE`, local order state depends mostly on REST submit/cancel responses and can drift during fills/cancel races.

## Plan C proposed fix

1. Add a typed Binance futures private user stream event enum:
   - `ACCOUNT_UPDATE`
   - `ORDER_TRADE_UPDATE`
2. Refactor the private user stream URL builder to accept the event enum.
3. Start two private user stream connections:
   - one `ACCOUNT_UPDATE` stream for position updates, startup sync, and symbol registration cleanup
   - one `ORDER_TRADE_UPDATE` stream for order lifecycle updates only
4. Keep `OrderManager` semantics unchanged. It already consumes `ORDER_TRADE_UPDATE` through `update_from_ws`.
5. Add observability log line when `ORDER_TRADE_UPDATE` is received.
6. Verify with a non-filling submit/cancel order that connector receives `ORDER_TRADE_UPDATE`.
7. Run short live validation.
8. Then rerun 2H aligned live test.

Do not disable or loosen open-order safety unless intentionally doing a diagnostic-only run.

## Plan C validation result

Status: **complete**.

- 2H validation run id: `plan_c_btcusdt_1777420030`
- Short ORDER_TRADE_UPDATE validation run id: `plan_c_short_btcusdt_1777432745`
- Short validation: `passed`
- 2H validation: `passed operationally; manually stopped after >2H`
- `ORDER_TRADE_UPDATE` observed: `yes` in short validation after promoting receipt log to info level
- Final 2H safety status: manual stop, no CRITICAL open-order drift stop observed
- Open-order mismatch recurrence: `transient only in audit status; no persistent safety stop`
- Position mismatch status: `safe`, max observed mismatch `0.001`

Conclusion: Plan C fixed the practical live open-order lifecycle alignment issue: the 2H run no longer stopped on open-order drift, and the short validation confirmed visible `ORDER_TRADE_UPDATE` lifecycle events for connector-owned orders.
