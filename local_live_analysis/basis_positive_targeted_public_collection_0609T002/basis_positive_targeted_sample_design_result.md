# Basis-Positive Targeted Sample Design Result

Task: `0609T002`

## Result

- Final recommendation: `tail_filter_hypothesis_validated_for_read_only_research`
- Reason: candidate visible tail filters repeated across at least three samples and concentration was reduced below the sample-share gate
- Remote public samples collected on `awsserver1`: `4`
- Aggregate canonical sample count: `7`
- Basis-positive rows after targeted collection: `5726`
- Basis-positive wrong-way rows after targeted collection: `101`
- Controlled support: `{'binance_momentum_bucket': True, 'hl_book_state_bucket': True}`

## Validated Read-Only Tail Hypotheses

- `basis_magnitude_bucket=basis_positive_small`: wrong-way `51`, samples `7`, classification `promising_visible_filter`.
- `hl_top5_imbalance_bucket=hl_top5_imbalance_negative_small`: wrong-way `35`, samples `7`, classification `promising_visible_filter`.
- `hl_microprice_minus_mid_bucket=hl_microprice_minus_mid_negative_small`: wrong-way `33`, samples `7`, classification `promising_visible_filter`.

## Boundary

- All accepted raw collection happened on `awsserver1` and all accepted analysis/testing happened on the local machine from pulled public raw artifacts.
- This is read-only public market-data research evidence only.
- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, case-library trigger, shadow decision, private/order endpoint, order lifecycle, strategy implementation, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion is authorized.
