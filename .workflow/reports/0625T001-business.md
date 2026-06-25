```md
执行线程：
- 业务线程-research

任务ID：
- 0625T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0625T001.md`
- `.workflow/reports/0625T001-business.md`
- `docs/cross_exchange_maker_mvp_plan.md`
- `examples/hyperliquid/alpha_edge_decomposition.py`
- `examples/hyperliquid/test_alpha_edge_decomposition.py`
- `local_live_analysis/cross_exchange_mvp_alpha_edge_decomposition_0625T001/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Implemented a deterministic offline `alpha_edge_decomposition.py` runner.
- Kept four evidence layers separate:
  - historical single-sample decision-time signal rows
  - accepted three-sample canonical event-mode aggregate evidence
  - canonical maker public proxy
  - `0624T003` production public-shadow funnel
- Did not row-join evidence layers without a shared clock.
- Generated:
  - `alpha_edge_decomposition_manifest.json`
  - `source_coverage_matrix.csv`
  - `signal_response_by_horizon.csv`
  - `lead_move_calibration.csv`
  - `anti_drift_markout_interaction.csv`
  - `edge_buffer_sensitivity.csv`
  - `root_cause_summary.csv`
  - `recommendation.md`
  - `boundary_manifest.json`
- Recorded that the task input path `canonical_event_mode_evidence_0604T003` does not exist locally; the accepted canonical validation artifact is `canonical_event_mode_evidence_0604T004`.
- Preserved historical as-of policy `binance_local_ts <= hyperliquid_decision_ts`; future Hyperliquid rows are labels only.

evidence：
- Historical join:
  - input rows `3599`
  - primary rows `3596`
  - excluded/watch rows `3`
  - future join `0`
  - missing Binance join `0`
  - stale Binance source `0`
  - Binance source-age p50/p90/p99 `13.348889/25.051643/32.91798816ms`
- Canonical event-mode evidence:
  - accepted samples `3`
  - all four allowlist features have positive, direction-consistent, `stable_across_samples` Hyperliquid future-mid response at `1000ms`
  - canonical 1000ms high-minus-low effects:
    - `binance_top5_imbalance`: `43.54494073` ticks
    - `binance_microprice_minus_mid_ticks`: `23.70366985` ticks
    - `binance_mid_move_ticks_from_prev`: `82.83682671` ticks
    - `binance_top5_bid_qty`: `72.11457773` ticks
- Production public-shadow funnel:
  - candidate rows `599`
  - fresh-touch allowed / anti-drift rows `68`
  - anti-drift block `64`, pass `4`
  - fair-mid pass `3`, block/stale `1`
  - edge rows `4`, edge pass `0`
  - fresh numeric edges `-24.5`, `-24.5`, `0.5` ticks
  - at diagnostic threshold `0` ticks, `1/3` fresh rows passes; at `1/2/3/5/7` ticks, `0/3` pass
- Production side/lead relationship:
  - all edge-evaluated candidates were `buy`
  - two fresh rows had `lead_move_ticks=-25`, opposed to buy side
  - one fresh row had `lead_move_ticks=0`
  - therefore the current `edge_gate_pass_count=0` is not primarily evidence that the `7` tick buffer is too high
- Anti-drift future markout:
  - unavailable for the production rows
  - output is explicitly `unsupported_missing_same_window_future_labels`
  - the canonical maker adverse-selection proxy remains separate and is not attributed to production anti-drift rows

recommendation：
- `needs_more_public_samples`
- Historical directional alpha exists, but the production signal contract is not ready to freeze.
- Root-cause order:
  1. only four production rows reached edge
  2. candidate side was opposed/zero relative to observed lead move in all three fresh rows
  3. anti-drift blocked `64/68`, but same-window markout is unavailable
  4. one fair-mid source row was stale
  5. historical alpha exists but the live top5-to-`lead_move_ticks` composite is not frozen
- Proposed `0625T002` evidence minimum:
  - at least `3` separated windows
  - at least `30` minutes each
  - at least `100` edge-evaluable rows aggregate and `20` per window
  - at least `2` volatility/liquidity regimes
  - dual top5, local/exchange timestamps, source seq/age, signal components/composite, lead move, side/quote, anti-drift, fair-mid, edge, and future HL mid/microprice labels on every decision

verify：
- `python -m pytest examples/hyperliquid/test_alpha_edge_decomposition.py -q` -> `3 passed`
- `python -m py_compile examples/hyperliquid/alpha_edge_decomposition.py examples/hyperliquid/test_alpha_edge_decomposition.py` -> passed
- `python examples/hyperliquid/alpha_edge_decomposition.py --help` -> passed
- Real accepted-artifact decomposition run -> passed
- JSON parsing and required artifact validation -> passed
- Required output files `9`, empty files `0`
- Boundary flags all true
- Deterministic rerun SHA comparison -> no differences
- `git diff --check` -> passed

done：
- `0625T001` completed the requested public alpha / edge decomposition and is ready for QA.
- The result does not accept a frozen production signal contract. It requests additional same-schema public samples with same-window future labels.
- No live strategy behavior was modified.
- No network collection, live orders, credential reads, private/account/order/cancel endpoint, live client, remote refresh, remote final gate, quote-distance/cap/post-only relaxation, taker/crossing, canary, M3, stable-PnL, default-on, or promotion action occurred.

blockers：
- No task execution blocker.
- Forward blocker: production edge evidence has only four rows and production anti-drift rows lack same-window future markout.

commit：
- dd771a9

提交信息：
- 0625 decompose cross-exchange alpha edge
```

