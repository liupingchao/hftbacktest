# 0625T006 Audit Replay Contract Validation Report

- final_recommendation: `audit_replay_contract_ready_for_qa`
- schema_hash: `0a899c61d63cf5326e16fa8b2d95ae7dc965b04ada72f3ba99811abfca0b9ab5`
- field_count: `63`
- required_field_count: `44`
- T005 caveat preserved: median adjusted counterfactual edge is `-1.5` ticks.
- T003 warning bucket preserved for replay diagnostics.

## Synthetic Fixtures
- accepted_lifecycle: expected `pass`, actual `pass`, rows `9`, issues `0`
- fail_closed_lifecycle: expected `fail_closed`, actual `fail_closed`, rows `2`, issues `5`

## Existing Artifact Compatibility
- 0625T005_production_shadow: `accepted_partial_contract`; scope `public_market_signal_fair_mid_quote_intent_counterfactual_markout`; conservative_reason ``
- 0618T007_m1_repeated_canary: `accepted_lifecycle_reference_no_fill_pnl`; scope `submit_resting_cancel_shutdown_reference`; conservative_reason `no_fill_or_settlement_evidence`
- 0618T008_m2_pnl_ledger: `accepted_fail_closed_ledger_reference`; scope `fee_inventory_pnl_ledger_schema_and_fail_closed_status`; conservative_reason `fail_closed_no_realized_live_pnl`

## Boundary
- Offline/local artifacts only.
- No network, AWS, remote, credential, private/account/order/cancel, user stream, live client, live order, canary, promotion, watcher change, or production config change.
