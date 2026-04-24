# Stability Hyperparameter Sweep Plan

## Goal

Run a two-stage parameter sweep to find a more stable market-making parameter set.

## Steps

1. Create Stage 1 manifest for 2026-04-01 only.
2. Create Stage 1 config using audit off, summary enabled, first_6h.
3. Create Stage 1 grid with 192 configs over base_spread, k_inv, k_pos, w_imb.
4. Run `sweep_backtest.py` with parallel workers.
5. Rank Stage 1 by stability score and hard filters.
6. Create Stage 2 grid from the top 10 Stage 1 configs.
7. Run Stage 2 on full 10-day manifest.
8. Rank Stage 2 and report best candidate.

## Score

```text
score = pnl_mtm / (1 + max_drawdown_mtm)
        - 0.1 * max_abs_position_notional
        - 100 * drop_api_rate
        - 50 * drop_latency_rate
```

## Hard Filters

```text
max_abs_position_notional <= 1000
avg_abs_position_notional <= 500
drop_api_rate <= 0.30
drop_latency_rate <= 0.15
```
