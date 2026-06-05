# Shutdown Proof Dry-Run Report

- task_id: `0605T004`
- classification: `local_fake_no_order_no_network_dry_run`
- scenario_count: `6`
- all_scenarios_passed: `true`
- final_proof_level_counts: `{"exchange_absent_only": 1, "exchange_reconciled": 2, "exchange_still_open": 1, "local_only": 2}`

This dry-run uses fake HBT and fake REST objects only. It does not connect to an exchange, start live trading, place orders, cancel real orders, or call real private/order endpoints.

## Scenarios

- `local_absent_exchange_absent` -> `exchange_reconciled` (exchange_status=`exchange_open_order_absent`, local_status=`local_absent_from_local_orders`, passed=`1`)
- `local_absent_exchange_still_open` -> `exchange_still_open` (exchange_status=`exchange_order_still_open`, local_status=`local_absent_from_local_orders`, passed=`1`)
- `exchange_check_failed` -> `local_only` (exchange_status=`open_orders_error:RuntimeError:dry-run open_orders failure`, local_status=`local_absent_from_local_orders`, passed=`1`)
- `local_active_exchange_absent` -> `exchange_absent_only` (exchange_status=`exchange_open_order_absent`, local_status=`local_active_order:new`, passed=`1`)
- `local_terminal_exchange_absent` -> `exchange_reconciled` (exchange_status=`exchange_open_order_absent`, local_status=`local_terminal_order:canceled`, passed=`1`)
- `no_rest_client` -> `local_only` (exchange_status=`not_checked:no_rest_client`, local_status=`local_absent_from_local_orders`, passed=`1`)
