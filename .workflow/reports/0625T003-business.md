# 线程回报

执行线程：
- 业务线程-research

任务ID：
- 0625T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0625T003.md`
- `.workflow/reports/0625T003-business.md`
- `examples/hyperliquid/binance_led_pricing_signal_runner.py`
- `examples/hyperliquid/test_binance_led_pricing_signal_runner.py`
- `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Added an explicit `--run-t003-signal-acceptance` mode to the existing pricing-signal runner while keeping the original `0601T005` default mode unchanged.
- Consumed the accepted `0625T002` sample package at `local_live_analysis/cross_exchange_mvp_sample_expansion_0625T002/`.
- Used a fixed chronological train/evaluation split from the T002 manifest:
  - train: `xemm_0625_t002_utc15_a`
  - evaluation: `xemm_0625_t002_utc15_b`, `xemm_0625_t002_utc16_c`
- Fitted normalization and feature weights on train rows only, then evaluated held-out rows without same-window threshold backfill.
- Generated deterministic T003 artifacts under `local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/`.

verify：
- `python -m pytest examples/hyperliquid/test_binance_led_pricing_signal_runner.py -q` -> `4 passed`
- `python examples/hyperliquid/binance_led_pricing_signal_runner.py --help` -> passed
- `python examples/hyperliquid/binance_led_pricing_signal_runner.py --run-t003-signal-acceptance` -> passed
- `python -m json.tool local_live_analysis/cross_exchange_mvp_signal_acceptance_0625T003/signal_acceptance_manifest.json` -> passed
- `python -m json.tool local_live_analysis/cross_exchange_mvp_sample_expansion_0625T002/sample_expansion_manifest.json` -> passed
- `python -m json.tool local_live_analysis/cross_exchange_mvp_sample_expansion_0625T002/boundary_manifest.json` -> passed
- CSV schema / non-empty checks for T003 artifacts -> passed
- `python -m py_compile examples/hyperliquid/binance_led_pricing_signal_runner.py` -> passed
- `git diff --check` -> passed

done：
- Accepted sample ids: `xemm_0625_t002_utc15_a`, `xemm_0625_t002_utc15_b`, `xemm_0625_t002_utc16_c`.
- Train/evaluation boundary: train `xemm_0625_t002_utc15_a`; evaluation `xemm_0625_t002_utc15_b` and `xemm_0625_t002_utc16_c`; no same-window threshold backfill.
- Horizon: nominal `1000ms`; effective held-out median `5000ms`, p90 `5500ms`.
- Row counts: `1999` complete contexts, `668` train rows, `1331` evaluation rows.
- Held-out direction/markout summary: direction hit rate `0.60096154`, mean signed future mid move `48.24943651` ticks, mean touch markout `54.8271976` ticks.
- Regime stability: normal activity/liquidity direction hit `0.59621451`, low activity/liquidity direction hit `0.60586319`.
- Source-age summary: held-out `binance_source_age_ms_p99=28.0983442`, `hyperliquid_join_age_ms_p99=993.4497915`.
- Basis/context conditioning artifacts were generated in `context_conditioning_summary.csv`.
- Final recommendation: `signal_contract_needs_repair`.
- `0625T004` may not be created from this result.
- T003 did not modify live behavior, did not place orders, did not read credentials, did not call private/account/order/cancel endpoints, and did not authorize shadow/live promotion.

blockers：
- Nominal `1000ms` labels are effectively around `5000ms`; horizon / edge formula / side mapping cannot be frozen for shadow until this is repaired or explicitly re-scoped.

commit：
- 待回填

提交信息：
- 待回填
