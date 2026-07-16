# 0716T001 Fill Attribution Repair Report

Result: `implemented_and_locally_validated`

Corrected 0715T001 attribution:

- window_01: `0.005 BTC @ 65335`, sourced from external trade-history exact price/size/time reconciliation.
- window_02: `0.005 BTC @ 65366`, sourced from external trade-history exact price/size/time reconciliation.
- window_03: no external fill match for submitted intents.

Repair behavior:

- Future live artifacts persist `user_fills_pullback_audit.json`.
- `live_fill_ledger.csv` now includes attribution fields.
- Fill attribution prefers tracked oid and falls back to symbol/side/price/size within the intent size budget.
- Ambiguous Hyperliquid cancel responses require reconciliation before no-fill classification.
- Missing liquidity role is recorded as `unknown`, not maker.

Unsupported:

- maker/taker role for 0715T001 from the current external export.
- fee/PnL calibration, maker viability, T012, promotion, final MVP pass.
