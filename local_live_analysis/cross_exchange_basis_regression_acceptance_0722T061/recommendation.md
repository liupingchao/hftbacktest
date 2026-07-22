# Basis Regression Alpha Recommendation

`accept_basis_regression_for_shadow`

- Valid regression rows: `10704`
- Windows: `3`
- Models: `binance_lead_regression`, `basis_regression`, `binance_lead_plus_basis_regression`
- Split: `leave_one_window_out`
- Fit scope: train-fold normalization and coefficients only
- Frozen shadow contract: `True`

Warnings and blockers:
- `warning`: `combined_underperforms_baseline_in_heldout_window`
- `warning`: `combined_mae_worse_than_baseline`
- `warning`: `cross_window_raw_intercept_drift`
- `warning`: `cross_window_prediction_mean_drift`
- `warning`: `limited_to_three_accepted_public_windows`
- `warning`: `basis_contract_caveat_binance_usdm_BTCUSDT_vs_hyperliquid_BTC`

This artifact does not authorize live orders, execution-PnL claims, promotion, or default-on behavior.
