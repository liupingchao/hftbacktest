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
- Repaired the QA-rejected T001 scope without changing the production watcher or any live strategy behavior.
- Upgraded the offline runner schema to `cross_exchange_mvp_alpha_edge_decomposition_v2`.
- Added explicit effective-horizon statistics to `signal_response_by_horizon.csv`:
  - count/min/mean/max effective age
  - mean offset from nominal horizon
  - `aligned` / `materially_delayed` / `insufficient_coverage` status
- Added deterministic historical conditioning analysis in `venue_state_conditioning.csv` for:
  - basis mid ticks
  - Hyperliquid spread
  - Hyperliquid top5 imbalance
  - Hyperliquid microprice-minus-mid
  - Hyperliquid join-age bucket
- Numeric conditions use deterministic global tertiles. Categorical join age preserves source buckets.
- Buckets with fewer than `30` rows fail closed as `insufficient_coverage`.
- Conditioning output is diagnostic association only; manifest sets `causal_claim_allowed=false`.
- Added timing, basis-conditioning, and Hyperliquid venue-state conditioning assessments to `root_cause_summary.csv`.
- Removed the extra EOF blank line in `docs/cross_exchange_maker_mvp_plan.md`.

evidence：
- Effective horizon:
  - nominal `100ms`: `3595` labels, effective mean `500.41724618ms`, offset `400.41724618ms`, `materially_delayed`
  - nominal `250ms`: `3595` labels, effective mean `500.41724618ms`, offset `250.41724618ms`, `materially_delayed`
  - nominal `500ms`: `3595` labels, effective mean `500.41724618ms`, `aligned`
  - nominal `1000ms`: `3594` labels, effective mean `1000.55648303ms`, `aligned`
- Conditioning at `1000ms`, maximum sufficiently-covered bucket mean range:
  - basis: `47.0199146` ticks
  - Hyperliquid top5 imbalance: `34.33546961` ticks
  - Hyperliquid microprice-minus-mid: `34.23344999` ticks
  - Hyperliquid spread: `18.65645906` ticks
  - Hyperliquid join-age bucket: `4.30117147` ticks
- These results identify sample association, not a production causal signal or authorization.
- Production funnel remains unchanged:
  - fresh-touch `68`
  - anti-drift `64` block / `4` pass
  - fair-mid `3` pass / `1` block
  - edge `4` block / `0` pass
- Recommendation remains exactly `needs_more_public_samples`.

verify：
- `python -m pytest examples/hyperliquid/test_alpha_edge_decomposition.py examples/hyperliquid/test_cross_exchange_lead_lag_analysis.py examples/hyperliquid/test_binance_led_pricing_signal_runner.py -q` -> `13 passed in 2.15s`
- `python -m py_compile examples/hyperliquid/alpha_edge_decomposition.py examples/hyperliquid/test_alpha_edge_decomposition.py` -> passed
- `python examples/hyperliquid/alpha_edge_decomposition.py --help` -> passed
- Formal accepted-artifact decomposition run -> passed
- Required artifacts `10`, empty files `0`
- JSON/CSV schema, timing status, conditioning fields, root-cause assessments, join quality and boundary flags -> passed
- Two-output-dir rerun with output paths normalized -> identical across `10` files
- `git diff --check b21afff..27c08dd` -> passed
- `git diff --check` -> passed

done：
- Both QA defects are repaired using existing accepted local fields.
- T001 now reports effective horizon/timing and basis/Hyperliquid venue-state conditioning.
- The original whitespace verification mismatch is repaired and the full task commit range passes.
- No live strategy behavior, edge threshold, quote distance, size cap, post-only policy, watcher, network collection, order path, credential path, private endpoint, canary, M3, stable-PnL, default-on, or promotion behavior changed.
- T001 is ready for repeat QA. `0625T002` remains undispatched.

blockers：
- No repair execution blocker.
- Forward evidence blocker remains insufficient production edge rows and missing same-window future markout for production anti-drift rows.

commit：
- 27c08dd

提交信息：
- 0625 repair alpha edge timing conditioning
```
