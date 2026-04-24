# Stability-Oriented Hyperparameter Sweep Design

## Goal

Find a more stable market-making parameter set than the current baseline, targeting positive or less-negative PnL with controlled drawdown, inventory, API churn, and latency drops.

## Current Baseline

10-day audit-off backtest, current params:

- PnL MTM: -1250.52
- max abs position notional: 1261.58
- avg abs position notional: 588.34
- drop latency rate: 7.84%
- drop API rate: 22.48%
- runtime: 1303.3s for 10 full days

## Objective

Primary objective: Sharpe-like stability, not raw PnL.

Score:

```text
score = pnl_mtm / (1 + max_drawdown_mtm)
        - 0.1 * max_abs_position_notional
        - 100 * drop_api_rate
        - 50 * drop_latency_rate
```

Hard filters:

```text
max_abs_position_notional <= 1000
avg_abs_position_notional <= 500
drop_api_rate <= 0.30
drop_latency_rate <= 0.15
```

## Stage 1: Coarse Sweep

Use a short sample to find broad direction:

- Date: 2026-04-01
- Window: first_6h
- Audit mode: off
- Summary enabled

Grid:

```toml
[risk]
base_spread = [0.5, 1.0, 1.5, 2.0]
k_inv = [0.0005, 0.001, 0.002, 0.004]
k_pos = [0.00005, 0.0001, 0.0002]

[fair]
w_imb = [0.0, 0.1, 0.25, 0.5]
```

Total: 192 configs.

## Stage 2: Validation

Run top configs from Stage 1 on full 10-day period:

- Date range: 2026-04-01 to 2026-04-10
- Window: full_day
- Audit mode: off
- Summary enabled

Validate the top 10 by stability score.

## Output

Write:

- Stage 1 sweep outputs under `/tmp/bt10days/tune_stage1`
- Stage 1 ranked CSV under `/tmp/bt10days/tune_stage1/ranked.csv`
- Stage 2 run outputs under `/tmp/bt10days/tune_stage2`
- Stage 2 ranked CSV under `/tmp/bt10days/tune_stage2/ranked.csv`
- Final selected candidate config summary in the final response
