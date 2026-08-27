# SKHYNIX Binance Phase Alignment Track A0-A4 Execution

Task: `0827T004`

Generated: `2026-08-27T17:37:01.839742+00:00`

## Primary Result

`continuous_state_no_discrete_phase_support`

The run is historical and outcome-blind. It contains no truly prospective
session, does not perform N/S/P/R semantic mapping, and does not authorize
Track B.

## Stage Results

| Stage | Result | Evidence |
| --- | --- | --- |
| A0 data admissibility | `passed` | 29 captures, 35.917 hours, zero declared depth gaps |
| A1 causal representation | `passed` | 100ms reconstruction grid; state observation at 200ms |
| A2 neutral state stability | `failed` | selected `student_t_negative_binomial_state_scale_k6`, K=6, beats all baselines=False |
| A3 grammar recurrence | `not_eligible_upstream_state_gate_failed` | runs=418056, diagnostic grammars=21, null p=0.024390 |
| A4 online recognition | `failed` | replay recall=0.364976, late=0.528353, OOD=0.000240 |
| Minute-scale low-parameter robustness | `continuous_state_still_preferred_at_minute_scale` | K=3, parameters=66, 5min-minus-VAR=-2.406063 |

The minute-scale branch uses a fixed 13-dimensional L1-L5 projection, a 1s
observation grid, shared emission scales and two negative-binomial duration
parameters per state. Its 66-parameter K=3 HSMM also trails a 39-parameter
diagonal AR(1) by -2.375018 log-density units per row. Minute-duration tails
are jointly identified across all states: `false`.

## Interpretation Boundary

This package tests repeated market-structure alignment only. It does not read
future returns, future midpoint/BBO, future volatility, markout, fills or PnL.
The August 26 and August 27 sessions are chronological no-refit historical
replays, not prospective evidence, because all were collected before this
protocol was frozen.

Runtime: `491.516` seconds.
