# Stage 5 Maker Parameter Optimization Dry Run Plan

Date: 2026-05-09

## Goal

Validate the maker-optimization pipeline without treating one live sample as production truth.

Stage 5 should prove that we can:

- gate the input sample with the Stage 4 maker acceptance contract,
- run a small parameter sweep in the aligned replay environment,
- reject unsafe or misleading candidates,
- rank surviving candidates with risk-aware metrics,
- produce artifacts that can feed Stage 6 out-of-sample validation.

Stage 5 does not select final live parameters.

## Input Baseline

Primary dry-run sample:

- `local_live_analysis/5-8-stage3-15m-livetest-v4`

Required input artifacts:

- `alignment_report_audit_replay.json`
- `backtest_audit_replay_result.json`
- `config_backtest_audit_replay.toml`
- `out/live_raw/btcusdt/manifest_2026-05-08_to_2026-05-08.json`

Preflight gate:

```bash
python examples/binance_tick_mm/maker_acceptance.py \
  --alignment-report local_live_analysis/5-8-stage3-15m-livetest-v4/alignment_report_audit_replay.json \
  --backtest-result local_live_analysis/5-8-stage3-15m-livetest-v4/backtest_audit_replay_result.json \
  --out local_live_analysis/5-8-stage3-15m-livetest-v4/maker_acceptance_stage5_preflight.json
```

The dry run may proceed only if the gate exits with code `0`.

## Implementation Work

Use the existing sweep runner:

- `examples/binance_tick_mm/sweep_backtest.py`

The sweep must run with optimization replay semantics:

- keep audit replay cadence and strict lag gate,
- disable `backtest_cadence.market_state_overlay`,
- disable `backtest_cadence.strategy_position_overlay`,
- keep `alignment_init` only as the initial position seed.

This is required because alignment replay may overlay live fair/reservation/target ticks and live position, which is correct for parity verification but invalid for parameter optimization.

Use the post-processing script:

- `examples/binance_tick_mm/rank_sweep.py`

The ranking script should read `sweep_summary.csv` and write:

- `ranked.csv`
- `rejected.csv`
- `stage5_dry_run_summary.json`

## Tier A Grid

Start with a small grid that checks the pipeline and avoids overfitting.

Parameters:

```toml
[risk]
base_spread = [0.4, 0.5, 0.7]
k_vol = [1.5, 2.0]
k_inv = [0.0003, 0.0005]

[strategy]
min_quote_update_interval_ms = [100, 200]
```

Total runs: `24`.

Fixed parameters:

```toml
[risk]
order_notional = 100.0

[strategy]
min_quote_move_ticks = 2
two_phase_replace_enabled = true
```

## Tier B Extension

Only run Tier B after Tier A produces safe candidates.

Take the best 4-6 Tier A candidates and test:

```toml
[risk]
order_notional = [50.0, 100.0]

[strategy]
min_quote_move_ticks = [1, 2]
```

Do not mix `two_phase_replace_enabled = false` into the main grid. If needed, test it as a separate control group because it changes lifecycle/API behavior.

## Rejection Rules

Reject a parameter set if any condition is true:

- backtest status is not `ok`
- required metrics are missing
- `max_abs_position_notional` exceeds the live safety cap
- `avg_abs_position_notional` is materially worse than control without enough PnL benefit
- `drop_api_rate` is materially worse than control
- `drop_latency_rate` is materially worse than control
- submit/cancel churn is materially worse than control
- max drawdown dominates PnL improvement
- PnL improvement is explained mainly by inventory exposure rather than plausible maker spread capture

Initial dry-run thresholds:

- `max_abs_position_notional <= 250.0`
- `drop_api_rate <= control_drop_api_rate + 0.02`
- `drop_latency_rate <= control_drop_latency_rate + 0.02`
- `max_drawdown_mtm <= max(abs(pnl_mtm) * 2.0, 5.0)`

These thresholds are conservative pipeline checks, not production risk limits.

## Ranking Rule

Rank surviving candidates by a risk-adjusted score:

```text
score =
  pnl_mtm
  - 1.0 * max_drawdown_mtm
  - 0.002 * avg_abs_position_notional
  - 50.0 * max(0, drop_api_rate - control_drop_api_rate)
  - 25.0 * max(0, drop_latency_rate - control_drop_latency_rate)
  - churn_penalty
```

`churn_penalty` should use action counts if present. If action counts are unavailable in `sweep_summary.csv`, keep the field in the output as diagnostic-missing and do not silently treat churn as zero.

## Execution Commands

Create the Tier A grid file:

```bash
cat > local_live_analysis/5-8-stage3-15m-livetest-v4/stage5_tier_a_grid.toml <<'TOML'
[risk]
base_spread = [0.4, 0.5, 0.7]
k_vol = [1.5, 2.0]
k_inv = [0.0003, 0.0005]

[strategy]
min_quote_update_interval_ms = [100, 200]
TOML
```

Run the sweep:

```bash
python examples/binance_tick_mm/sweep_backtest.py \
  --base-config local_live_analysis/5-8-stage3-15m-livetest-v4/config_backtest_audit_replay.toml \
  --manifest local_live_analysis/5-8-stage3-15m-livetest-v4/out/live_raw/btcusdt/manifest_2026-05-08_to_2026-05-08.json \
  --grid local_live_analysis/5-8-stage3-15m-livetest-v4/stage5_tier_a_grid.toml \
  --optimization-replay \
  --workers 4 \
  --window full_day \
  --out local_live_analysis/5-8-stage3-15m-livetest-v4/stage5_tier_a_sweep
```

Rank the sweep:

```bash
python examples/binance_tick_mm/rank_sweep.py \
  --sweep-summary local_live_analysis/5-8-stage3-15m-livetest-v4/stage5_tier_a_sweep/sweep_summary.csv \
  --control-summary local_live_analysis/5-8-stage3-15m-livetest-v4/out/backtest_audit_replay/summary_audit_replay.json \
  --out local_live_analysis/5-8-stage3-15m-livetest-v4/stage5_tier_a_sweep/ranking
```

## Acceptance

Stage 5 is complete when:

- Stage 4 maker acceptance preflight passes.
- Sweep outputs show `audit_replay_market_state_overlay_mode = off`.
- Sweep outputs show `audit_replay_strategy_position_overlay_mode = off`.
- Tier A sweep finishes with no failed runs.
- `sweep_summary.csv`, `sweep_summary.json`, and `sweep_meta.json` exist.
- `ranked.csv`, `rejected.csv`, and `stage5_dry_run_summary.json` exist.
- At least 3 candidates pass rejection filters, or the rejection summary clearly explains why not.
- The report compares control vs candidates on:
  - `pnl_mtm`
  - `max_drawdown_mtm`
  - `avg_abs_position_notional`
  - `max_abs_position_notional`
  - `drop_api_rate`
  - `drop_latency_rate`
  - action/churn metrics if present
- No result is described as production-ready before Stage 6 out-of-sample validation.

## Expected Output

Write a short stage summary after execution:

- `local_live_analysis/5-8-stage3-15m-livetest-v4/stage5_tier_a_sweep/STAGE5_SUMMARY.md`

The summary should include:

- preflight gate result,
- number of sweep runs,
- number of rejected runs,
- top 5 ranked candidates,
- control metrics,
- key reasons for rejection,
- recommendation for Stage 6 candidate set.
