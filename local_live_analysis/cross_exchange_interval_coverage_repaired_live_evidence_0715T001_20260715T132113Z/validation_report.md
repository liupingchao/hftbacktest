# 0715T001 Local Validation Report

Validation status: `artifact_internal_pass_external_fill_reconciliation_failed`

External trade-history reconciliation:

- Source: user-provided `trade_logs/trade_history.csv`, not committed raw.
- Derived files:
  - `external_trade_history_reconciliation.csv`
  - `external_trade_history_reconciliation_summary.json`
- Result: exchange trade history matches two submitted intents exactly while artifact `live_fill_ledger.csv` remains empty.
- Matched fills: window_01 `0.005 BTC @ 65335`, window_02 `0.005 BTC @ 65366`.
- Current maker/taker role is unsupported by the external export and must be repaired through live fill attribution.

## Windows

- window_01: original artifact `submitted_resting_no_fill`, submissions `1`, artifact fills `0`, external matched size `0.005 BTC`, final open orders `0`, independent empty `True`, coverage rows `1`, interval trade rows `6`
- window_02: original artifact `submitted_resting_no_fill`, submissions `2`, artifact fills `0`, external matched size `0.005 BTC`, final open orders `0`, independent empty `True`, coverage rows `1`, interval trade rows `18`
- window_03: `submitted_no_resting_reject_or_error_no_fill`, submissions `2`, fills `0`, final open orders `0`, independent empty `True`, coverage rows `0`, interval trade rows `0`

## Checks

- JSON/CSV parse errors: `0`
- SHA256 reconciliation: `pass`
- `remote_sha256_manifest.txt` is excluded from strict matching because it is self-referential and was created by shell redirection before `sha256sum` populated it.
- Final open orders all empty: `True`
- No threshold/quote-envelope/size/max-submission expansion was performed by this validation.
- This report does not claim no-fill, fill probability, maker fill count, fee/PnL, maker viability, T012, promotion, or final MVP pass.
