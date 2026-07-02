# T003 Recommendation

`signal_contract_accepted_for_shadow`

- Input package: `/Users/liu/Documents/hftbacktest/local_live_analysis/cross_exchange_mvp_hl_fast_sample_expansion_0627T001`
- Valid near-target 1000ms rows: `10704`
- Per-window valid rows: `xemm_0627_t001_hlfast_utc16_a=3587`, `xemm_0627_t001_hlfast_utc17_b=3567`, `xemm_0627_t001_hlfast_utc17_c=3550`
- Split method: `leave_one_window_out`
- Fee/adverse buffer used for acceptance proxy: `1.5` ticks
- Final recommendation: `signal_contract_accepted_for_shadow`

Accepted signal contract:

- Candidate: `binance_lead_composite`
- Fields: `input_binance_top5_imbalance,input_binance_microprice_minus_mid_ticks,input_binance_mid_move_ticks_from_prev`
- Threshold abs z: `1.0`
- Side mapping: `positive_signal_buy_negative_signal_sell`
- Horizon: `1000ms` with near-target effective horizon gate

Caveats:

- Some source-age or basis buckets have negative adjusted proxy; see `regime_stability.csv` before promoting beyond public shadow.

No watcher/live strategy behavior changed; no private/order endpoints, live orders, shadow execution, canary, or promotion are authorized by this package.
