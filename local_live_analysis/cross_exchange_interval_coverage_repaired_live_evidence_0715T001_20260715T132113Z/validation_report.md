# 0715T001 Local Validation Report

Validation status: `pass`

## Windows

- window_01: `submitted_resting_no_fill`, submissions `1`, fills `0`, final open orders `0`, independent empty `True`, coverage rows `1`, interval trade rows `6`
- window_02: `submitted_resting_no_fill`, submissions `2`, fills `0`, final open orders `0`, independent empty `True`, coverage rows `1`, interval trade rows `18`
- window_03: `submitted_no_resting_reject_or_error_no_fill`, submissions `2`, fills `0`, final open orders `0`, independent empty `True`, coverage rows `0`, interval trade rows `0`

## Checks

- JSON/CSV parse errors: `0`
- SHA256 reconciliation: `pass`
- `remote_sha256_manifest.txt` is excluded from strict matching because it is self-referential and was created by shell redirection before `sha256sum` populated it.
- Final open orders all empty: `True`
- No threshold/quote-envelope/size/max-submission expansion was performed by this validation.
- This report does not claim fill probability, fee/PnL, maker viability, T012, promotion, or final MVP pass.
